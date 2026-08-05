//! `tsne_host` — the parallel host engine behind `TSNE` (TSNE-PARAMS): the
//! Barnes-Hut quadtree, both gradient objectives, and the two-phase gradient
//! descent that drives them.
//!
//! ## Why the t-SNE descent is a HOST algorithm on every backend
//! The [`gmm_host`](super::gmm_host) argument applies here almost verbatim, and
//! one structural fact makes it stronger:
//!
//! 1. **The loop is long and each step is small.** A default fit is
//!    `max_iter = 1000` objective evaluations. A cubecl launch costs ~50 µs of
//!    pure dispatch ([[mlrs-rf-fit-optimization]]) and on `cubecl-cpu` one
//!    launch is one OS thread per unit ([[mlrs-cubecl-cpu-execution-model]]),
//!    so the dispatch alone is ~0.05 s × (launches per iteration) × 1000 before
//!    any arithmetic happens.
//! 2. **Barnes-Hut is a pointer chase.** The negative force walks a quadtree
//!    whose traversal depth, branch pattern, and work per query point all vary
//!    per point and per iteration. That is the exact shape a SIMT device is
//!    worst at — every lane in a warp diverges — and the shape an out-of-order
//!    host core with a branch predictor is best at. It is not a kernel that was
//!    measured to be slow; it is a kernel that cannot be written well.
//! 3. **The per-iteration host tail is unavoidable anyway.** The gains /
//!    momentum update rule reads the gradient and writes the embedding, so a
//!    device gradient would pay an upload + readback per iteration — the
//!    round-trip [[mlrs-gpu-perf-root-cause]] identifies as the pathology
//!    behind every iterative prim that lost.
//!
//! The pre-existing `method='exact'` DEVICE prim ([`super::tsne`]) is retained
//! and still serves GPU backends; this module owns the Barnes-Hut method on
//! every backend, and the exact method wherever the host arm is faster.
//!
//! ## What is reproduced, and what is deliberately better
//! The quadtree is a line-for-line port of `sklearn/neighbors/_quad_tree.pyx`
//! — the same `M · (1 + 1e-3·sign(M))` bounding-box inflation, the same
//! `1e-6` duplicate epsilon, the same `squared_max_width / dist² < θ²`
//! summary test, and the same child-visit order — because every one of those
//! changes which cells summarize, and therefore the embedding.
//!
//! Three differences are deliberate:
//!
//! - **Determinism.** sklearn accumulates `sum_Q` and the KL error into
//!   OpenMP reduction variables, so its output depends on `OMP_NUM_THREADS`.
//!   Here every reduction is written per-point and summed in POINT order, so a
//!   fit is bit-identical at any thread count.
//! - **`f64` throughout.** sklearn casts the embedding, `P`, and the tree to
//!   `float32` for Barnes-Hut regardless of the input dtype.
//! - **The tree is rebuilt in place.** sklearn constructs a fresh `_QuadTree`
//!   (and its allocations) on each of the 1000 iterations; [`QuadTree::rebuild`]
//!   reuses one cell arena for the whole fit.
//!
//! Tests live in `crates/mlrs-backend/tests/tsne_host_test.rs` (AGENTS.md §2).

use crate::prims::host_pool::{Shared, WorkerPool};

/// sklearn `MACHINE_EPSILON` — `np.finfo(np.double).eps`.
pub const MACHINE_EPSILON: f64 = 2.220_446_049_250_313e-16;

/// sklearn `_EXPLORATION_MAX_ITER` — the early-exaggeration phase length.
pub const EXPLORATION_MAX_ITER: usize = 250;
/// sklearn `_N_ITER_CHECK` — error/convergence check cadence.
pub const N_ITER_CHECK: usize = 50;
/// sklearn `min_gain`.
const MIN_GAIN: f64 = 0.01;

/// Barnes-Hut relies on a quad-/oct-tree, so sklearn caps `n_components <= 3`.
pub const BH_MAX_COMPONENTS: usize = 3;
/// Embedding dimensionalities up to this take the register-accumulator path in
/// the exact gradient. Covers every `n_components` t-SNE is used at (2 and 3);
/// above it the slice path keeps the method total.
const REG_D: usize = 4;
/// `_quad_tree.pyx`'s `EPSILON`: two points within this on EVERY axis are the
/// same point, and a leaf holding one absorbs the other instead of splitting
/// forever.
const QT_EPSILON: f64 = 1e-6;

/// The joint-probability matrix, in whichever layout the method uses.
pub enum TsneP<'a> {
    /// `method='exact'`: the dense row-major `n × n` matrix, diagonal 0.
    Dense(&'a [f64]),
    /// `method='barnes_hut'`: CSR over the symmetrized k-NN graph.
    Sparse {
        /// Row offsets, length `n + 1`.
        indptr: &'a [usize],
        /// Column indices, ascending within a row.
        indices: &'a [u32],
        /// Values.
        data: &'a [f64],
    },
}

impl TsneP<'_> {
    fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse { .. })
    }
}

/// Everything the descent needs that is not the data.
#[derive(Debug, Clone)]
pub struct TsneDescentConfig {
    /// Samples.
    pub n: usize,
    /// Embedding dimensionality (`n_components`).
    pub d: usize,
    /// `degrees_of_freedom = max(n_components − 1, 1)`.
    pub dof: f64,
    /// Total iteration budget (sklearn `max_iter`).
    pub max_iter: usize,
    /// sklearn `early_exaggeration`.
    pub early_exaggeration: f64,
    /// The RESOLVED step size (sklearn's `learning_rate_`, after `'auto'`).
    pub learning_rate: f64,
    /// sklearn `min_grad_norm`.
    pub min_grad_norm: f64,
    /// sklearn `n_iter_without_progress` (main phase only).
    pub n_iter_without_progress: usize,
    /// Barnes-Hut `angle` (θ). Ignored by the exact objective.
    pub angle: f64,
    /// Worker count for the parallel passes.
    pub threads: usize,
    /// sklearn `verbose`.
    pub verbose: usize,
}

/// What the descent produced.
pub struct TsneDescentOutcome {
    /// The KL divergence at the final embedding, against the UN-exaggerated
    /// `P` (sklearn's `kl_divergence_` contract).
    pub kl_divergence: f64,
    /// Iterations actually run (sklearn `n_iter_`).
    pub n_iter: usize,
}

// ===========================================================================
// Public entry
// ===========================================================================

/// Run sklearn `_tsne`'s two-phase schedule over `y` (row-major `n × d`,
/// updated in place).
///
/// Phase 1 runs `EXPLORATION_MAX_ITER` iterations against `P · early_exaggeration`
/// at momentum 0.5 with its own length as the no-progress window; phase 2 runs
/// the remaining budget against the un-exaggerated `P` at momentum 0.8. The
/// returned KL is ALWAYS re-evaluated against the un-exaggerated `P`, so a fit
/// short enough to end inside phase 1 does not report an exaggeration-inflated
/// value.
pub fn tsne_descent(y: &mut [f64], p: TsneP<'_>, cfg: &TsneDescentConfig) -> TsneDescentOutcome {
    let n = cfg.n;
    let d = cfg.d;
    let units = cfg.threads.max(1);
    let pool = WorkerPool::new(units);
    let mut ws = Workspace::new(n, d, p.is_sparse());

    // The exaggerated copy is materialized once (sklearn scales `P` in place
    // and scales it back; a copy keeps the caller's `P` untouched, which is
    // what lets the final KL be evaluated against it without a second scaling).
    let exaggerated: Vec<f64> = match &p {
        TsneP::Dense(v) => v.iter().map(|&x| x * cfg.early_exaggeration).collect(),
        TsneP::Sparse { data, .. } => data.iter().map(|&x| x * cfg.early_exaggeration).collect(),
    };
    let p_early = match &p {
        TsneP::Dense(_) => TsneP::Dense(&exaggerated),
        TsneP::Sparse {
            indptr, indices, ..
        } => TsneP::Sparse {
            indptr,
            indices,
            data: &exaggerated,
        },
    };

    let explore_iters = EXPLORATION_MAX_ITER.min(cfg.max_iter);
    let (_kl_early, it_early) = gradient_descent(
        &pool,
        &mut ws,
        y,
        &p_early,
        cfg,
        0,
        explore_iters,
        0.5,
        EXPLORATION_MAX_ITER,
    );

    let mut it_final = it_early;
    let remaining = cfg.max_iter.saturating_sub(EXPLORATION_MAX_ITER);
    if it_early + 1 < explore_iters || remaining > 0 {
        let (_kl2, it2) = gradient_descent(
            &pool,
            &mut ws,
            y,
            &p,
            cfg,
            it_early + 1,
            cfg.max_iter,
            0.8,
            cfg.n_iter_without_progress,
        );
        it_final = it2;
    }

    let kl = evaluate(&pool, &mut ws, y, &p, cfg, true);
    TsneDescentOutcome {
        kl_divergence: kl,
        n_iter: it_final,
    }
}

// ===========================================================================
// Gradient descent (sklearn `_gradient_descent`)
// ===========================================================================

/// Verbatim port of sklearn `_gradient_descent`: gains `±0.2` / `×0.8` clipped
/// at `min_gain`, momentum update, and a `grad_norm` measured AFTER the gains
/// scaling. Returns `(error at the last check, last iteration index run)`.
#[allow(clippy::too_many_arguments)]
fn gradient_descent(
    pool: &WorkerPool,
    ws: &mut Workspace,
    y: &mut [f64],
    p: &TsneP<'_>,
    cfg: &TsneDescentConfig,
    it_start: usize,
    max_iter: usize,
    momentum: f64,
    n_iter_without_progress: usize,
) -> (f64, usize) {
    let nd = cfg.n * cfg.d;
    ws.update[..nd].iter_mut().for_each(|v| *v = 0.0);
    ws.gains[..nd].iter_mut().for_each(|v| *v = 1.0);

    let mut error = f64::MAX;
    let mut best_error = f64::MAX;
    let mut best_iter = it_start;
    let mut i = it_start;

    if it_start >= max_iter {
        return (error, it_start.saturating_sub(1));
    }

    for iter in it_start..max_iter {
        i = iter;
        let check_convergence = (iter + 1) % N_ITER_CHECK == 0 || iter == max_iter - 1;

        let kl = evaluate(pool, ws, y, p, cfg, check_convergence);
        if check_convergence {
            error = kl;
        }

        // The update rule is `O(n·d)` and touches three vectors plus the
        // embedding; splitting it costs more in barrier crossings than it saves
        // below a few hundred thousand elements, so it stays on the driver.
        let mut grad_norm_sq = 0.0f64;
        for k in 0..nd {
            let g = ws.grad[k];
            if ws.update[k] * g < 0.0 {
                ws.gains[k] += 0.2;
            } else {
                ws.gains[k] *= 0.8;
            }
            if ws.gains[k] < MIN_GAIN {
                ws.gains[k] = MIN_GAIN;
            }
            let gg = g * ws.gains[k];
            grad_norm_sq += gg * gg;
            ws.update[k] = momentum * ws.update[k] - cfg.learning_rate * gg;
            y[k] += ws.update[k];
        }
        let grad_norm = grad_norm_sq.sqrt();

        if check_convergence {
            if cfg.verbose >= 2 {
                println!(
                    "[t-SNE] Iteration {}: error = {:.7}, gradient norm = {:.7}",
                    iter + 1,
                    error,
                    grad_norm
                );
            }
            if error < best_error {
                best_error = error;
                best_iter = iter;
            } else if iter - best_iter > n_iter_without_progress {
                if cfg.verbose >= 2 {
                    println!(
                        "[t-SNE] Iteration {}: did not make any progress during the \
                         last {} episodes. Finished.",
                        iter + 1,
                        n_iter_without_progress
                    );
                }
                break;
            }
            if grad_norm <= cfg.min_grad_norm {
                if cfg.verbose >= 2 {
                    println!(
                        "[t-SNE] Iteration {}: gradient norm {:.7}. Finished.",
                        iter + 1,
                        grad_norm
                    );
                }
                break;
            }
        }
    }
    (error, i)
}

// ===========================================================================
// Workspace
// ===========================================================================

/// Every buffer the descent reuses across its (up to) 1000 iterations. Nothing
/// in the hot loop allocates.
struct Workspace {
    /// The gradient, row-major `n × d`.
    grad: Vec<f64>,
    /// sklearn's `update` (the momentum term).
    update: Vec<f64>,
    /// sklearn's `gains`.
    gains: Vec<f64>,
    /// Barnes-Hut positive forces, `n × d`.
    pos_f: Vec<f64>,
    /// Barnes-Hut negative forces, `n × d`.
    neg_f: Vec<f64>,
    /// Per-point `Σ size·qijZ`, reduced in POINT order so the result does not
    /// depend on the thread count.
    sum_q_row: Vec<f64>,
    /// Per-row KL contribution, reduced in row order for the same reason.
    err_row: Vec<f64>,
    /// The tree, rebuilt in place each iteration (Barnes-Hut only).
    tree: QuadTree,
    /// Dense unnormalized affinities `n × n` (exact only).
    qnum: Vec<f64>,
}

impl Workspace {
    fn new(n: usize, d: usize, sparse: bool) -> Self {
        let nd = n * d;
        Self {
            grad: vec![0.0; nd],
            update: vec![0.0; nd],
            gains: vec![1.0; nd],
            pos_f: if sparse { vec![0.0; nd] } else { Vec::new() },
            neg_f: if sparse { vec![0.0; nd] } else { Vec::new() },
            sum_q_row: vec![0.0; n],
            err_row: vec![0.0; n],
            tree: if sparse {
                QuadTree::new(d, n)
            } else {
                QuadTree::new(d.min(BH_MAX_COMPONENTS).max(1), 0)
            },
            qnum: if sparse { Vec::new() } else { vec![0.0; n * n] },
        }
    }
}

/// Evaluate the objective at `y`, leaving the gradient in `ws.grad`. Returns
/// the KL divergence when `compute_error`, and `NaN` otherwise (sklearn's
/// contract — the descent only reads it on check iterations).
fn evaluate(
    pool: &WorkerPool,
    ws: &mut Workspace,
    y: &[f64],
    p: &TsneP<'_>,
    cfg: &TsneDescentConfig,
    compute_error: bool,
) -> f64 {
    match p {
        TsneP::Sparse {
            indptr,
            indices,
            data,
        } => evaluate_bh(pool, ws, y, indptr, indices, data, cfg, compute_error),
        TsneP::Dense(pm) => evaluate_exact(pool, ws, y, pm, cfg, compute_error),
    }
}

// ===========================================================================
// Barnes-Hut objective
// ===========================================================================

/// sklearn `_kl_divergence_bh` → `_barnes_hut_tsne.compute_gradient`.
#[allow(clippy::too_many_arguments)]
fn evaluate_bh(
    pool: &WorkerPool,
    ws: &mut Workspace,
    y: &[f64],
    indptr: &[usize],
    indices: &[u32],
    data: &[f64],
    cfg: &TsneDescentConfig,
    compute_error: bool,
) -> f64 {
    let (n, d) = (cfg.n, cfg.d);
    ws.tree.rebuild(y, n, d);

    let exponent = (cfg.dof + 1.0) / 2.0;
    let float_dof = cfg.dof;
    let sq_theta = cfg.angle * cfg.angle;
    let units = pool.units();

    // --- ONE pass: both forces per point, in one visit to `y[i]`. ---
    //
    // sklearn runs these as two `prange` loops because its positive loop needs
    // `sum_Q` for the error term. That dependency is removable: with the
    // clamps inactive (see `err_row` below) the KL is
    //   `Σ p·log p − Σ p·log q_unnorm + log(sum_Q)·Σ p`,
    // whose first two terms need no normalizer at all. Fusing halves the
    // barrier crossings and, more importantly, lets point `i`'s embedding row
    // serve BOTH forces from one cache line.
    {
        let neg = Shared::new(&mut ws.neg_f);
        let pos = Shared::new(&mut ws.pos_f);
        let sumq = Shared::new(&mut ws.sum_q_row);
        let errs = Shared::new(&mut ws.err_row);
        let tree = &ws.tree;
        pool.run(&|unit: usize| {
            // SAFETY: row `i` is owned by exactly one unit (see `span`), and
            // the pass is bracketed by the pool's barriers.
            let neg = unsafe { neg.get_mut() };
            let pos = unsafe { pos.get_mut() };
            let sumq = unsafe { sumq.get_mut() };
            let errs = unsafe { errs.get_mut() };
            let (lo, hi) = span(n, unit, units);
            for i in lo..hi {
                let yi = &y[i * d..i * d + d];

                // Negative: walk the tree, accumulating the force AS the
                // summary cells are found rather than into a scratch array
                // first. The visit order is unchanged, so this is the same
                // summation and the same value — it just never spills
                // `n · (d + 2)` doubles to memory and read them back.
                let mut nf = [0.0f64; BH_MAX_COMPONENTS];
                let local_q = tree.negative_force(yi, sq_theta, float_dof, exponent, &mut nf);
                sumq[i] = local_q;
                neg[i * d..i * d + d].copy_from_slice(&nf[..d]);

                // Positive: the sparse edges out of `i`.
                let mut pf = [0.0f64; BH_MAX_COMPONENTS];
                let mut c = 0.0f64;
                for k in indptr[i]..indptr[i + 1] {
                    let j = indices[k] as usize;
                    let pij = data[k];
                    let yj = &y[j * d..j * d + d];
                    let mut buff = [0.0f64; BH_MAX_COMPONENTS];
                    let mut dij = 0.0f64;
                    for ax in 0..d {
                        let t = yi[ax] - yj[ax];
                        buff[ax] = t;
                        dij += t * t;
                    }
                    let mut qij = float_dof / (float_dof + dij);
                    if float_dof != 1.0 {
                        qij = qij.powf(exponent);
                    }
                    let scale = pij * qij;
                    if compute_error {
                        // The `sum_Q`-free half of `p·log(p/q)`; the
                        // `log(sum_Q)·Σp` term is added once, below. Both
                        // logarithms take a positive-clamped argument, standing
                        // in for sklearn's `max(·, FLOAT32_TINY)` on `p` and on
                        // the NORMALIZED `q`: an embedding diverged far enough
                        // to underflow `q` to zero would otherwise turn the
                        // reported error into `+inf` instead of a large finite
                        // number, and the two behave differently at the
                        // no-progress check.
                        c += pij
                            * (pij.max(f64::MIN_POSITIVE).ln()
                                - qij.max(f64::MIN_POSITIVE).ln());
                    }
                    for (ax, pfa) in pf.iter_mut().enumerate().take(d) {
                        *pfa += scale * buff[ax];
                    }
                }
                errs[i] = c;
                pos[i * d..i * d + d].copy_from_slice(&pf[..d]);
            }
        });
    }
    // Reduced in POINT order: the value no longer depends on `units`.
    let sum_q: f64 = ws.sum_q_row.iter().sum::<f64>().max(MACHINE_EPSILON);

    // --- grad = c · (pos − neg/sum_Q). ---
    let c_f = 2.0 * (cfg.dof + 1.0) / cfg.dof;
    let inv_q = 1.0 / sum_q;
    for k in 0..n * d {
        ws.grad[k] = c_f * (ws.pos_f[k] - ws.neg_f[k] * inv_q);
    }

    if compute_error {
        // `P` is normalized to sum 1 over the stored entries, so the deferred
        // `log(sum_Q)·Σp` term is just `ln(sum_Q)`. Summing `Σp` explicitly
        // instead of assuming 1 keeps the identity exact under the
        // exaggerated `P` of phase 1, whose entries sum to
        // `early_exaggeration`.
        let psum: f64 = data.iter().sum();
        ws.err_row.iter().sum::<f64>() + sum_q.ln() * psum
    } else {
        f64::NAN
    }
}

/// Contiguous `[lo, hi)` share of `total` for `wid` of `workers`.
#[inline(always)]
fn span(total: usize, wid: usize, workers: usize) -> (usize, usize) {
    let lo = total * wid / workers;
    let hi = total * (wid + 1) / workers;
    (lo, hi)
}

// ===========================================================================
// Exact objective
// ===========================================================================

/// sklearn `_kl_divergence`: the dense `O(n²)` Student-t affinities and the
/// full-matrix gradient.
///
/// Two passes, both row-parallel. The first fills the symmetric `qnum` block
/// (upper triangle evaluated, lower mirrored — every unit owns whole rows, so
/// the mirror write is still exclusive) and the per-row affinity sums; the
/// second turns them into the gradient and the KL.
fn evaluate_exact(
    pool: &WorkerPool,
    ws: &mut Workspace,
    y: &[f64],
    p: &[f64],
    cfg: &TsneDescentConfig,
    compute_error: bool,
) -> f64 {
    let (n, d) = (cfg.n, cfg.d);
    let exponent = -(cfg.dof + 1.0) / 2.0;
    let inv_dof = 1.0 / cfg.dof;
    let units = pool.units();

    {
        let qnum = Shared::new(&mut ws.qnum);
        let sumq = Shared::new(&mut ws.sum_q_row);
        pool.run(&|unit: usize| {
            // SAFETY: row `i` (and the mirrored column `i`) belongs to the one
            // unit that owns `i` under the round-robin deal below.
            let qnum = unsafe { qnum.get_mut() };
            let sumq = unsafe { sumq.get_mut() };
            let mut i = unit;
            while i < n {
                let yi = &y[i * d..i * d + d];
                let mut row_sum = 0.0f64;
                qnum[i * n + i] = 0.0;
                for j in (i + 1)..n {
                    let yj = &y[j * d..j * d + d];
                    let mut dsq = 0.0f64;
                    for ax in 0..d {
                        let t = yi[ax] - yj[ax];
                        dsq += t * t;
                    }
                    let num = (1.0 + dsq * inv_dof).powf(exponent);
                    qnum[i * n + j] = num;
                    qnum[j * n + i] = num;
                    // Counted twice — once for (i, j), once for (j, i) — which
                    // is what makes the ordered reduction below the FULL sum.
                    row_sum += 2.0 * num;
                }
                sumq[i] = row_sum;
                i += units;
            }
        });
    }
    let sum_q: f64 = ws.sum_q_row.iter().sum::<f64>().max(MACHINE_EPSILON);

    {
        let grad = Shared::new(&mut ws.grad);
        let errs = Shared::new(&mut ws.err_row);
        let qnum = &ws.qnum;
        let inv_q = 1.0 / sum_q;
        pool.run(&|unit: usize| {
            // SAFETY: each unit writes only its own contiguous row block.
            let grad = unsafe { grad.get_mut() };
            let errs = unsafe { errs.get_mut() };
            let (lo, hi) = span(n, unit, units);
            let c_f = 2.0 * (cfg.dof + 1.0) / cfg.dof;
            for i in lo..hi {
                let yi = &y[i * d..i * d + d];
                let mut c = 0.0f64;
                let qrow = &qnum[i * n..i * n + n];
                let prow = &p[i * n..i * n + n];
                // The `d <= REG_D` path keeps the accumulator in registers.
                // Accumulating straight into `grad` instead would make the
                // compiler reload it every `j` (it cannot prove `grad` and `y`
                // do not alias), which costs more than the branch in the
                // `O(n²)` loop this sits in. `n_components > REG_D` is legal
                // for the exact method, so the slice path exists for it.
                let mut g = [0.0f64; REG_D];
                if d <= REG_D {
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let num = qrow[j];
                        let pij = prow[j];
                        let q = (num * inv_q).max(MACHINE_EPSILON);
                        let f = (pij - q) * num;
                        let yj = &y[j * d..j * d + d];
                        for (ax, ga) in g.iter_mut().enumerate().take(d) {
                            *ga += f * (yi[ax] - yj[ax]);
                        }
                        if compute_error {
                            c += pij * (pij.max(MACHINE_EPSILON) / q).ln();
                        }
                    }
                    for ax in 0..d {
                        grad[i * d + ax] = c_f * g[ax];
                    }
                } else {
                    grad[i * d..i * d + d].iter_mut().for_each(|v| *v = 0.0);
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let num = qrow[j];
                        let pij = prow[j];
                        let q = (num * inv_q).max(MACHINE_EPSILON);
                        let f = (pij - q) * num;
                        let yj = &y[j * d..j * d + d];
                        for ax in 0..d {
                            grad[i * d + ax] += f * (yi[ax] - yj[ax]);
                        }
                        if compute_error {
                            c += pij * (pij.max(MACHINE_EPSILON) / q).ln();
                        }
                    }
                    for ax in 0..d {
                        grad[i * d + ax] *= c_f;
                    }
                }
                errs[i] = c;
            }
        });
    }

    if compute_error {
        ws.err_row.iter().sum()
    } else {
        f64::NAN
    }
}


// ===========================================================================
// QuadTree (port of sklearn/neighbors/_quad_tree.pyx)
// ===========================================================================
//
// ## Why the cell is SPLIT in two
// The negative force walks this tree once per point per iteration — 5 million
// traversals in a default `n = 5000` fit — and an `angle` sweep showed that
// traversal, not arithmetic, is what decides whether this engine beats
// sklearn's: at `angle = 0.8` (few cells visited) mlrs won 1.17x, at
// `angle = 0.2` (many cells visited) it lost. So the traversal's working set is
// the thing to shrink.
//
// Of the 11 fields sklearn's single `Cell` struct carries, the traversal reads
// FOUR: `barycenter`, `squared_max_width`, `cumulative_size`, `is_leaf` (plus
// the child links). `center` / `min_bounds` / `max_bounds` / `point_index`
// exist only to place a point during INSERTION and are never read again.
// Keeping them in the same struct means every cache line the walk pulls in is
// ~60% payload it will not touch.
//
// Splitting them puts the traversal's fields in a 40-byte [`HotCell`] and
// banishes the rest to [`ColdCell`], which the walk never touches at all. The
// child links move to their own array with stride `2^d`, so a 2-D embedding —
// the overwhelmingly common case — stores 4 links per cell rather than the 8 a
// fixed `[i32; 8]` would.
//
// ## Why the walk RECURSES
// The first version used an explicit `Vec<u32>` stack. Every visited cell then
// paid a heap indirection, a capacity check on push, and an `Option` unwrap on
// pop — real work next to the handful of flops a summarized cell actually
// costs. sklearn's `summarize` recurses, and so does this: the frame is small,
// the depth is logarithmic in `n`, and the natural `0..2^d` child order is the
// same order sklearn visits in, which keeps the force summation bit-identical
// to the stack version it replaces.

/// The fields the negative-force traversal reads. 40 bytes.
#[derive(Clone)]
struct HotCell {
    /// Center of mass of the points under this cell.
    barycenter: [f64; BH_MAX_COMPONENTS],
    /// The squared side length used by the `width² / dist² < θ²` summary test.
    squared_max_width: f64,
    /// Points under this cell (duplicates counted).
    cumulative_size: u32,
    /// A leaf always summarizes; an inner cell only if it is far enough.
    is_leaf: bool,
}

/// The fields only INSERTION reads. Never touched by the traversal.
#[derive(Clone)]
struct ColdCell {
    /// Split point of the cell's box, per axis.
    center: [f64; BH_MAX_COMPONENTS],
    /// Lower corner of the cell's box.
    min_bounds: [f64; BH_MAX_COMPONENTS],
    /// Upper corner of the cell's box.
    max_bounds: [f64; BH_MAX_COMPONENTS],
    /// The single point a leaf holds; `-1` on an inner cell.
    point_index: i32,
}

/// The array-backed quad-/oct-tree the negative force is summarized over.
struct QuadTree {
    hot: Vec<HotCell>,
    cold: Vec<ColdCell>,
    /// Child links, `n_cells_per_cell` per cell. `< 0` means "empty octant".
    children: Vec<i32>,
    cell_count: usize,
    n_dimensions: usize,
    n_cells_per_cell: usize,
}

impl QuadTree {
    fn new(n_dimensions: usize, capacity_hint: usize) -> Self {
        let d = n_dimensions.clamp(1, BH_MAX_COMPONENTS);
        // A balanced tree over `n` points needs roughly `2n` cells; the arenas
        // grow on demand, so this only avoids the early doublings.
        let cap = capacity_hint.saturating_mul(2).max(16);
        Self {
            hot: Vec::with_capacity(cap),
            cold: Vec::with_capacity(cap),
            children: Vec::with_capacity(cap * (1usize << d)),
            cell_count: 0,
            n_dimensions: d,
            n_cells_per_cell: 1usize << d,
        }
    }

    /// Append a blank cell to all three arenas and return its id.
    #[inline]
    fn push_blank(&mut self) -> usize {
        let id = self.cell_count;
        self.hot.push(HotCell {
            barycenter: [0.0; BH_MAX_COMPONENTS],
            squared_max_width: 0.0,
            cumulative_size: 0,
            is_leaf: true,
        });
        self.cold.push(ColdCell {
            center: [0.0; BH_MAX_COMPONENTS],
            min_bounds: [0.0; BH_MAX_COMPONENTS],
            max_bounds: [0.0; BH_MAX_COMPONENTS],
            point_index: -1,
        });
        self.children
            .extend(std::iter::repeat_n(-1i32, self.n_cells_per_cell));
        self.cell_count += 1;
        id
    }

    /// Discard the previous iteration's tree and build one over `y`, REUSING
    /// the arenas (sklearn allocates a fresh tree per iteration).
    fn rebuild(&mut self, y: &[f64], n: usize, d: usize) {
        self.n_dimensions = d.clamp(1, BH_MAX_COMPONENTS);
        self.n_cells_per_cell = 1usize << self.n_dimensions;
        self.hot.clear();
        self.cold.clear();
        self.children.clear();
        self.cell_count = 0;

        let dd = self.n_dimensions;
        let mut min_bounds = [f64::INFINITY; BH_MAX_COMPONENTS];
        let mut max_bounds = [f64::NEG_INFINITY; BH_MAX_COMPONENTS];
        for i in 0..n {
            for ax in 0..dd {
                let v = y[i * d + ax];
                if v < min_bounds[ax] {
                    min_bounds[ax] = v;
                }
                if v > max_bounds[ax] {
                    max_bounds[ax] = v;
                }
            }
        }
        // sklearn: `M = np.maximum(M * (1 + 1e-3 * np.sign(M)), M + 1e-3)` —
        // inflate the upper bound so every point is STRICTLY inside the box.
        for ax in 0..dd {
            let m = max_bounds[ax];
            let signed = m
                * (1.0
                    + 1e-3
                        * if m > 0.0 {
                            1.0
                        } else if m < 0.0 {
                            -1.0
                        } else {
                            0.0
                        });
            max_bounds[ax] = signed.max(m + 1e-3);
        }

        // Root.
        self.push_blank();
        {
            let hot = &mut self.hot[0];
            let cold = &mut self.cold[0];
            for ax in 0..dd {
                cold.min_bounds[ax] = min_bounds[ax];
                cold.max_bounds[ax] = max_bounds[ax];
                cold.center[ax] = (max_bounds[ax] + min_bounds[ax]) / 2.0;
                let width = max_bounds[ax] - min_bounds[ax];
                hot.squared_max_width = hot.squared_max_width.max(width * width);
            }
        }

        let mut pt = [0.0f64; BH_MAX_COMPONENTS];
        for i in 0..n {
            pt[..dd].copy_from_slice(&y[i * d..i * d + dd]);
            self.insert_point(&pt, i as i32);
        }
    }

    /// `_quad_tree.pyx::insert_point`, with the tail recursion turned into a
    /// loop. The leaf-split case re-enters at the SAME cell (now an inner
    /// node), which is exactly what sklearn's `return self.insert_point(point,
    /// point_index, cell_id)` does.
    fn insert_point(&mut self, point: &[f64; BH_MAX_COMPONENTS], point_index: i32) {
        let dd = self.n_dimensions;
        let mut cell_id = 0usize;
        loop {
            // Empty leaf: the point lands here.
            if self.hot[cell_id].cumulative_size == 0 {
                let hot = &mut self.hot[cell_id];
                hot.cumulative_size = 1;
                hot.barycenter[..dd].copy_from_slice(&point[..dd]);
                self.cold[cell_id].point_index = point_index;
                return;
            }

            if !self.hot[cell_id].is_leaf {
                let n_point = self.hot[cell_id].cumulative_size as f64;
                {
                    let hot = &mut self.hot[cell_id];
                    for ax in 0..dd {
                        hot.barycenter[ax] =
                            (n_point * hot.barycenter[ax] + point[ax]) / (n_point + 1.0);
                    }
                    hot.cumulative_size += 1;
                }
                let selected = self.select_child(point, cell_id);
                if selected < 0 {
                    self.insert_point_in_new_child(point, cell_id, point_index, 1);
                    return;
                }
                cell_id = selected as usize;
                continue;
            }

            // A leaf that already holds a point.
            let is_dup = {
                let bary = &self.hot[cell_id].barycenter;
                (0..dd).all(|ax| (point[ax] - bary[ax]).abs() <= QT_EPSILON)
            };
            if is_dup {
                self.hot[cell_id].cumulative_size += 1;
                return;
            }
            // Push the resident point down into a new child, then retry HERE:
            // the cell is now an inner node and the branch above applies.
            let saved_pt = self.hot[cell_id].barycenter;
            let saved_idx = self.cold[cell_id].point_index;
            let saved_size = self.hot[cell_id].cumulative_size;
            self.insert_point_in_new_child(&saved_pt, cell_id, saved_idx, saved_size);
        }
    }

    /// Which child octant of `cell_id` contains `point`; `-1` when that octant
    /// has not been created yet.
    #[inline]
    fn select_child(&self, point: &[f64; BH_MAX_COMPONENTS], cell_id: usize) -> i32 {
        let center = &self.cold[cell_id].center;
        let mut selected = 0usize;
        for ax in 0..self.n_dimensions {
            selected *= 2;
            if point[ax] >= center[ax] {
                selected += 1;
            }
        }
        self.children[cell_id * self.n_cells_per_cell + selected]
    }

    /// `_insert_point_in_new_child`: carve the octant of `cell_id` that holds
    /// `point` into a fresh leaf carrying `size` points.
    fn insert_point_in_new_child(
        &mut self,
        point: &[f64; BH_MAX_COMPONENTS],
        cell_id: usize,
        point_index: i32,
        size: u32,
    ) -> usize {
        let dd = self.n_dimensions;
        let child_id = self.push_blank();

        self.hot[cell_id].is_leaf = false;
        self.cold[cell_id].point_index = -1;
        let parent_center = self.cold[cell_id].center;
        let parent_min = self.cold[cell_id].min_bounds;
        let parent_max = self.cold[cell_id].max_bounds;

        let mut slot = 0usize;
        {
            let hot = &mut self.hot[child_id];
            let cold = &mut self.cold[child_id];
            for ax in 0..dd {
                slot *= 2;
                if point[ax] >= parent_center[ax] {
                    slot += 1;
                    cold.min_bounds[ax] = parent_center[ax];
                    cold.max_bounds[ax] = parent_max[ax];
                } else {
                    cold.min_bounds[ax] = parent_min[ax];
                    cold.max_bounds[ax] = parent_center[ax];
                }
                cold.center[ax] = (cold.min_bounds[ax] + cold.max_bounds[ax]) / 2.0;
                let width = cold.max_bounds[ax] - cold.min_bounds[ax];
                hot.barycenter[ax] = point[ax];
                hot.squared_max_width = hot.squared_max_width.max(width * width);
            }
            cold.point_index = point_index;
            hot.cumulative_size = size;
        }
        self.children[cell_id * self.n_cells_per_cell + slot] = child_id as i32;
        child_id
    }

    /// `_quad_tree.pyx::summarize`, with the t-SNE negative force accumulated
    /// INLINE instead of through a `(Δ, dist², size)` scratch array.
    ///
    /// sklearn materializes the summary and then loops over it a second time.
    /// Both loops visit the summarizing cells in the same order, so folding the
    /// force into the traversal is the same summation of the same terms — but
    /// it never writes `n · (d + 2)` doubles per point per iteration to memory
    /// and reads them back, which at `n = 2000` is ~5 MB of round-tripped
    /// traffic per iteration that buys nothing.
    ///
    /// Adds into `nf` and returns this point's `Σ size · qijZ` contribution to
    /// `sum_Q`.
    #[inline]
    fn negative_force(
        &self,
        point: &[f64],
        squared_theta: f64,
        float_dof: f64,
        exponent: f64,
        nf: &mut [f64; BH_MAX_COMPONENTS],
    ) -> f64 {
        let mut sum_q = 0.0f64;
        self.walk(0, point, squared_theta, float_dof, exponent, nf, &mut sum_q);
        sum_q
    }

    /// One cell of the traversal. Children are visited in `0..2^d` order — the
    /// order sklearn's recursion uses, and therefore the summation order of the
    /// negative force rather than an implementation detail.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        cell_id: usize,
        point: &[f64],
        squared_theta: f64,
        float_dof: f64,
        exponent: f64,
        nf: &mut [f64; BH_MAX_COMPONENTS],
        sum_q: &mut f64,
    ) {
        let dd = self.n_dimensions;
        let cell = &self.hot[cell_id];
        let mut delta = [0.0f64; BH_MAX_COMPONENTS];
        let mut dist2 = 0.0f64;
        let mut duplicate = true;
        for ax in 0..dd {
            let v = point[ax] - cell.barycenter[ax];
            delta[ax] = v;
            dist2 += v * v;
            duplicate &= v.abs() <= QT_EPSILON;
        }

        // A leaf sitting on the query point is the point itself: no self
        // interaction.
        if duplicate && cell.is_leaf {
            return;
        }
        if cell.is_leaf || (cell.squared_max_width / dist2) < squared_theta {
            let size = cell.cumulative_size as f64;
            let mut qijz = float_dof / (float_dof + dist2);
            if float_dof != 1.0 {
                qijz = qijz.powf(exponent);
            }
            *sum_q += size * qijz;
            let mult = size * qijz * qijz;
            for ax in 0..dd {
                nf[ax] += mult * delta[ax];
            }
            return;
        }

        let base = cell_id * self.n_cells_per_cell;
        for c in 0..self.n_cells_per_cell {
            let ch = self.children[base + c];
            if ch >= 0 {
                self.walk(
                    ch as usize,
                    point,
                    squared_theta,
                    float_dof,
                    exponent,
                    nf,
                    sum_q,
                );
            }
        }
    }
}
