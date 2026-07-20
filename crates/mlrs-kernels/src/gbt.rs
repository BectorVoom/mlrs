//! `gbt` — HistGradientBoosting device kernels (GBT-01): sequential gradient
//! boosting over batched level-wise histogram trees, generic over `<F: Float +
//! CubeElement>` with NO backend feature (D-13).
//!
//! ## Design (sklearn `HistGradientBoosting*`, on the `tree.rs` chassis)
//! One boosting iteration = compute per-sample gradients/hessians from the
//! current raw predictions, then grow `K` trees SIMULTANEOUSLY (`K = 1` for
//! regression / binary log-loss, `K = n_classes` for multiclass — sklearn's
//! `n_trees_per_iteration_`), then fold the shrunk leaf values back into the
//! raw predictions — all launch-only (zero device→host readbacks, the
//! `sgd_solve` / `random_forest` perf lesson).
//!
//! Split semantics are the sklearn/XGBoost histogram form (splitting.pyx):
//! - histogram slots per bin: `count`, `Σ gradient`, `Σ hessian` (`ncs = 3`);
//! - node value `v = −G/(H + λ + 1e-15)`, node loss `v·G`;
//! - split gain `= loss(node) − loss(left) − loss(right)`; the ROOT of every
//!   tree has `value = 0` ⇒ `loss = 0` (sklearn `TreeNode(value=0)`);
//! - a split is VALID iff both children have `count ≥ min_samples_leaf` and
//!   `Σ hessian ≥ min_hessian_to_split`; a node is a LEAF iff the best valid
//!   gain is `≤ 0` (sklearn `gain <= 0 → _finalize_leaf`) or the depth bound
//!   forces it;
//! - leaf value stored SHRUNK: `−learning_rate · G/(H + λ + 1e-15)`.
//!
//! Unlike the Random Forest there is NO feature subsampling, NO bootstrap and
//! NO RNG anywhere — fits are bit-deterministic across runs and backends.
//!
//! ## Root-level parallelism (the RF memory's optimization lever)
//! With only `K` trees per launch, per-(node, feature) histogram units would
//! leave the device idle at shallow levels (level 0 = `K·d` units scanning all
//! `n` rows). The gather is therefore ROW-BLOCKED: [`gbt_hist`] writes
//! `n_blocks` partial histograms per (tree, node, feature) — each unit owns
//! one block slice exclusively — and [`gbt_hist_reduce`] sums the block axis.
//!
//! ## Sibling histogram SUBTRACTION (perf: ~15-20% faster fit)
//! [`gbt_hist`]'s `node_stride` parameter lets the host gather only the LEFT
//! child of every split directly (`node_stride = 2`, every EVEN real node
//! id); [`gbt_hist_subtract`] then derives the RIGHT sibling by subtracting
//! from the retained PARENT-level histogram (`right = parent − left`, valid
//! on CUMULATIVE histograms since cumsum is linear and a non-leaf parent's
//! rows partition exactly into its two children — see that kernel's doc for
//! the leaf-phantom-node correctness argument). This roughly halves the
//! total row-scan work per level without shrinking `gbt_hist`'s launch grid
//! (an earlier attempt that shrank the grid to cut redundant reads was
//! REVERTED — this machine's histogram gather is occupancy-bound, not
//! bandwidth-bound; fewer/fatter threads regressed fit by ~30% despite
//! moving fewer bytes). The host orchestration
//! (`hist_gradient_boosting.rs::hgb_fit_impl`) only uses subtraction while a
//! level's histogram fits in ONE unchunked buffer (a `subtract_cap` derived
//! from the existing `HGB_HIST_BUDGET_BYTES` chunking discipline); deep/wide
//! trees that need chunking fall back to the original direct-gather path.
//!
//! ## cpu-MLIR safety (the primary correctness gate)
//! Every kernel stays inside the proven op-set (spike findings 001/002/003,
//! see `tree.rs`): bare-`ABSOLUTE_POS` 1D launches with `if tid < total`
//! guards, ascending `while` scans, ≤ 2 coupled loop-carried `F` accumulators,
//! cross-loop values staged through GLOBAL arrays (`row_max` / `row_sumexp`),
//! statement-form `if`, no `SharedMemory`/atomics/`F::INFINITY`. Each unit
//! writes only memory it exclusively owns (its own histogram-block / score /
//! node / raw slice) — race-free without atomics (the `lbfgs.rs` precedent).
//! ONE exception: [`gbt_hist_atomic`] uses shared-memory FLOAT atomics and is
//! host-gated to the CUDA/ROCm backends (never launched on cpu-MLIR or wgpu —
//! see its doc); cpu and wgpu keep the deterministic gather path end to end.
//!
//! LANDMINE (wgpu): small constants MUST be `F::new(1e-15_f32)`, never
//! `F::cast_from(1e-15)` — the f64-literal cast makes the WGSL shader fail to
//! compile SILENTLY on wgpu (the kernel launches as a no-op and every score
//! reads back 0). `F::new` takes an f32 literal and lowers cleanly.
//!
//! ## Buffer contract (row-major, indices `u32`, values `F`)
//! - `binned` — `n × d` `u32` bin index per (row, feature) (`tree.rs`
//!   [`rf_bin_features`](crate::tree::rf_bin_features) output).
//! - `raw` — `n × K` current raw predictions (baseline + Σ tree values).
//! - `g` / `h` — `n × K` per-sample gradients / hessians.
//! - `order` / `ranges` — the `tree.rs` per-tree row partition, `K` trees wide
//!   (per-ITERATION scratch, reset by [`gbt_init_partition`]).
//! - `hist_part` — `K × nodes_chunk × d × n_blocks × nb × 3` partial counts;
//!   `hist` — the block-reduced `K × nodes_chunk × d × nb × 3`, made
//!   CUMULATIVE over the bin axis by [`rf_hist_cum`](crate::tree::rf_hist_cum)
//!   (`ncs = 3`).
//! - model arrays — `(n_iters · K) × total_nodes` complete-tree layout
//!   (`tree.rs` convention); kernels take a `tree_base` (= `iter · K`) so one
//!   allocation holds every stage.
//!
//! Tests live in `crates/mlrs-backend/tests/hist_gradient_boosting_test.rs`
//! (this crate is feature-free and cannot launch; AGENTS.md §2).

use cubecl::prelude::*;

/// Squared-error gradients: `g = raw − y`, `h = 1` (sklearn
/// `HalfSquaredError`; the constant hessian makes `Σh` the sample count).
/// One unit per sample — a pure per-element map.
#[cube(launch)]
pub fn gbt_grad_reg<F: Float + CubeElement>(
    raw: &Array<F>,
    y: &Array<F>,
    g: &mut Array<F>,
    h: &mut Array<F>,
    n: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < n as usize {
        g[tid] = raw[tid] - y[tid];
        h[tid] = F::new(1.0_f32);
    }
}

/// Binary log-loss gradients: `p = σ(raw)`, `g = p − y`, `h = p·(1 − p)`
/// (sklearn `HalfBinomialLoss`; `y ∈ {0, 1}` as `F`). One unit per sample.
#[cube(launch)]
pub fn gbt_grad_binary<F: Float + CubeElement>(
    raw: &Array<F>,
    y: &Array<F>,
    g: &mut Array<F>,
    h: &mut Array<F>,
    n: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < n as usize {
        let p = F::new(1.0_f32) / (F::new(1.0_f32) + F::exp(-raw[tid]));
        g[tid] = p - y[tid];
        h[tid] = p * (F::new(1.0_f32) - p);
    }
}

/// Per-row max of the `n × k` raw scores → `row_max` (softmax staging: the
/// cross-loop value goes through GLOBAL memory — FINDING 002-B). Running-max
/// statement-form `if`, one unit per row.
#[cube(launch)]
pub fn gbt_row_max<F: Float + CubeElement>(
    raw: &Array<F>,
    row_max: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < n as usize {
        let base = (tid as u32) * k;
        let mut mx = raw[base as usize];
        let mut c = 1u32;
        while c < k {
            let v = raw[(base + c) as usize];
            if v > mx {
                mx = v;
            }
            c += 1u32;
        }
        row_max[tid] = mx;
    }
}

/// Per-row `Σ_c exp(raw − row_max)` → `row_sumexp` (softmax staging pass 2;
/// reads the GLOBAL `row_max`, single `F` accumulator). One unit per row.
#[cube(launch)]
pub fn gbt_row_sumexp<F: Float + CubeElement>(
    raw: &Array<F>,
    row_max: &Array<F>,
    row_sumexp: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < n as usize {
        let base = (tid as u32) * k;
        let mx = row_max[tid];
        let mut acc = F::new(0.0_f32);
        let mut c = 0u32;
        while c < k {
            acc += F::exp(raw[(base + c) as usize] - mx);
            c += 1u32;
        }
        row_sumexp[tid] = acc;
    }
}

/// Multiclass log-loss gradients: `p = softmax(raw)` (from the staged
/// `row_max` / `row_sumexp`), `g = p − 1{y = c}`, `h = p·(1 − p)` (sklearn
/// `HalfMultinomialLoss`). One unit per `(row, class)` element — no loops.
#[cube(launch)]
pub fn gbt_grad_multi<F: Float + CubeElement>(
    raw: &Array<F>,
    row_max: &Array<F>,
    row_sumexp: &Array<F>,
    y_idx: &Array<u32>,
    g: &mut Array<F>,
    h: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        let c = (tid as u32) % k;
        let p = F::exp(raw[tid] - row_max[i as usize]) / row_sumexp[i as usize];
        let mut ind = F::new(0.0_f32);
        if y_idx[i as usize] == c {
            ind = F::new(1.0_f32);
        }
        g[tid] = p - ind;
        h[tid] = p * (F::new(1.0_f32) - p);
    }
}

/// Initialize the `n × k` raw predictions to the per-class baseline (sklearn
/// `loss.fit_intercept_only`, host-computed). One unit per element.
#[cube(launch)]
pub fn gbt_init_raw<F: Float + CubeElement>(
    baseline: &Array<F>,
    raw: &mut Array<F>,
    n: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let c = (tid as u32) % k;
        raw[tid] = baseline[c as usize];
    }
}

/// Reset the per-iteration row partition: identity `order` per tree and the
/// level-0 root range `[0, n)` per tree (zero uploads — the launch-only
/// discipline). One unit per `(tree, row)`; the first `k` units also write
/// their tree's root range (exclusively owned — no race).
#[cube(launch)]
pub fn gbt_init_partition(order: &mut Array<u32>, ranges: &mut Array<u32>, n: u32, k: u32) {
    let tid = ABSOLUTE_POS;
    let total = k * n;
    if tid < total as usize {
        order[tid] = (tid as u32) % n;
        if (tid as u32) < k {
            ranges[(2u32 * (tid as u32)) as usize] = 0u32;
            ranges[(2u32 * (tid as u32) + 1u32) as usize] = n;
        }
    }
}

/// Row-BLOCKED histogram gather: one unit per `(tree, node_in_chunk, feature,
/// block)` zeroes then accumulates its OWN `nb × 3` partial slice
/// (`count`, `Σg`, `Σh`) over its stripe of the node's row range. Exclusive
/// slice ownership keeps the three `+=` read-modify-writes race-free without
/// atomics (the `lbfgs.rs` / `rf_hist_*` precedent).
///
/// `nodes_total` is the FULL level width (ranges indexing); `node_base` +
/// `nodes_chunk` select the node chunk this launch covers (deep-level memory
/// chunking, the `RF_HIST_BUDGET_BYTES` discipline). `node_stride` scales the
/// per-thread `node` offset ONLY for the `ranges` lookup (`1` for a normal
/// dense gather; `2` selects every EVEN real node id — the sibling
/// histogram-SUBTRACTION "gather the left child only" pass, see
/// `gbt_hist_subtract`); the OUTPUT stays densely packed by `node` either
/// way, so [`gbt_hist_reduce`] is unaffected.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_hist<F: Float + CubeElement>(
    binned: &Array<u32>,
    g: &Array<F>,
    h: &Array<F>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    hist_part: &mut Array<F>,
    n: u32,
    d: u32,
    nb: u32,
    k: u32,
    nodes_total: u32,
    node_base: u32,
    nodes_chunk: u32,
    n_blocks: u32,
    node_stride: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes_chunk * d * n_blocks;
    if tid < total as usize {
        let tt = (tid as u32) / (nodes_chunk * d * n_blocks);
        let rem = (tid as u32) % (nodes_chunk * d * n_blocks);
        let node = rem / (d * n_blocks);
        let rem2 = rem % (d * n_blocks);
        let f = rem2 / n_blocks;
        let blk = rem2 % n_blocks;

        let base = (tid as u32) * (nb * 3u32);
        // Zero the exclusively-owned slice (pool buffers arrive uninitialized).
        let mut z = 0u32;
        while z < nb * 3u32 {
            hist_part[(base + z) as usize] = F::new(0.0_f32);
            z += 1u32;
        }

        let rbase = (tt * nodes_total + node_base + node * node_stride) * 2u32;
        let s = ranges[rbase as usize];
        let e = ranges[(rbase + 1u32) as usize];
        let len = e - s;
        let lo = s + (blk * len) / n_blocks;
        let hi = s + ((blk + 1u32) * len) / n_blocks;
        let mut r = lo;
        while r < hi {
            let i = order[(tt * n + r) as usize];
            let b = binned[(i * d + f) as usize];
            let slot = base + b * 3u32;
            hist_part[slot as usize] += F::new(1.0_f32);
            hist_part[(slot + 1u32) as usize] += g[(i * k + tt) as usize];
            hist_part[(slot + 2u32) as usize] += h[(i * k + tt) as usize];
            r += 1u32;
        }
    }
}

/// Reduce the block axis of [`gbt_hist`]'s partial histograms: one unit per
/// `(tree, node_in_chunk, feature, bin, slot)` sums its `n_blocks` strided
/// partials with a single `F` accumulator.
#[cube(launch)]
pub fn gbt_hist_reduce<F: Float + CubeElement>(
    hist_part: &Array<F>,
    hist: &mut Array<F>,
    nb: u32,
    cols: u32,
    n_blocks: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = cols * nb * 3u32;
    if tid < total as usize {
        let col = (tid as u32) / (nb * 3u32); // (tt, node, f) flat index
        let within = (tid as u32) % (nb * 3u32);
        let mut acc = F::new(0.0_f32);
        let mut b = 0u32;
        while b < n_blocks {
            acc += hist_part[((col * n_blocks + b) * (nb * 3u32) + within) as usize];
            b += 1u32;
        }
        hist[tid] = acc;
    }
}

/// Derive the RIGHT sibling's histogram via SUBTRACTION from a directly
/// GATHERED left-child histogram (`left`, produced by [`gbt_hist`] with
/// `node_stride = 2`) and a RETAINED parent-level histogram (`parent_hist`)
/// — valid on CUMULATIVE histograms too, since cumsum is linear:
/// `parent_cum − left_cum = right_cum` exactly, because a non-leaf parent's
/// row range partitions EXACTLY into its two children (`gbt_count_left` /
/// `gbt_partition` never drop or duplicate a row). `left` is copied straight
/// into the EVEN (left-child) slot of `hist_full`; the derived value lands in
/// the ODD (right-child) slot — EXCEPT when the parent is a LEAF, where both
/// children are phantom complete-tree slots with an EMPTY row range: naive
/// subtraction would hand the phantom right slot the whole (nonzero) parent
/// histogram back, so it is forced to zero instead (the phantom left slot is
/// already correctly zero — `gbt_hist` scans its empty range and stays at its
/// zeroed initial value). One unit per `(tree, parent, d·nb·3 element)` — no
/// loops, straight-line reads (the `gbt_split_scores` shape).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_hist_subtract<F: Float + CubeElement>(
    left: &Array<F>,
    parent_hist: &Array<F>,
    is_leaf: &Array<u32>,
    hist_full: &mut Array<F>,
    elem: u32,
    k: u32,
    left_children: u32,
    tree_base: u32,
    parent_level_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * left_children * elem;
    if tid < total as usize {
        let col = (tid as u32) / elem;
        let e = (tid as u32) % elem;
        let tt = col / left_children;
        let p = col % left_children;

        let lv = left[tid];
        let pv = parent_hist[tid];

        let nodes_cur = left_children * 2u32;
        let even_col = tt * nodes_cur + p * 2u32;
        hist_full[(even_col * elem + e) as usize] = lv;

        let midx = (tree_base + tt) * total_nodes + parent_level_base + p;
        let mut rv = pv - lv;
        if is_leaf[midx as usize] == 1u32 {
            rv = F::new(0.0_f32);
        }
        hist_full[((even_col + 1u32) * elem + e) as usize] = rv;
    }
}

/// Zero an `F` histogram buffer (the atomic gather flushes with `fetch_add`,
/// so the pool buffer must start at 0 — pool buffers arrive uninitialized).
/// One unit per element.
#[cube(launch)]
pub fn gbt_hist_zero<F: Float + CubeElement>(hist: &mut Array<F>, len: u32) {
    let tid = ABSOLUTE_POS;
    if tid < len as usize {
        hist[tid] = F::new(0.0_f32);
    }
}

/// Shared-memory FLOAT-atomic histogram gather (the `rf_hist_class_atomic`
/// shape adapted to the 3-slot `count/Σg/Σh` lattice): one 256-thread cube
/// per `(tree, node_in_chunk, feature chunk, row block)` accumulates its row
/// stripe into a fixed 4096-slot `SharedMemory<Atomic<F>>` with `fetch_add`,
/// then atomically flushes the non-zero slots to the global histogram.
///
/// CUDA/ROCm ONLY (host-gated): float `atomicAdd` is native there; WGSL has
/// no `atomic<f32>` and the cpu-MLIR path miscompiles shared/atomic kernels
/// (spike findings 001–003) — both keep the deterministic [`gbt_hist`]
/// gather. Unlike the forest's integer path, float atomics make the
/// accumulation ORDER non-deterministic, so sums differ from the gather path
/// by summation-order float noise (last-ULP scale — well inside the 1e-5
/// oracle gate; fits on these backends are tolerance-exact, not bitwise).
///
/// The FEATURE CHUNK axis (`f_chunk` features per cube, `n_fchunks =
/// ceil(d / f_chunk)`) keeps the shared lattice `f_chunk · nb · 3 ≤ 4096`
/// slots for any `nb` (at the default `nb = 64` one chunk covers `d ≤ 21`;
/// at `nb = 256` a chunk is 5 features). `order`/`g`/`h` row loads repeat
/// per chunk, but each cube still owns MANY rows per thread — the grid-width
/// occupancy lesson from the reverted gather restructure does not apply to
/// the 256-wide cube shape. `node_stride` mirrors [`gbt_hist`]: it scales the
/// `ranges` lookup only (`2` = gather EVEN/left real nodes for the sibling
/// subtraction), the output stays densely packed by `node`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_hist_atomic<F: Float + CubeElement>(
    binned_t: &Array<u32>,
    g: &Array<F>,
    h: &Array<F>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    hist: &mut Array<Atomic<F>>,
    n: u32,
    d: u32,
    nb: u32,
    k: u32,
    nodes_total: u32,
    node_base: u32,
    nodes_chunk: u32,
    f_chunk: u32,
    n_fchunks: u32,
    bcount: u32,
    node_stride: u32,
) {
    let shared = SharedMemory::<Atomic<F>>::new(4096usize);
    let lid = UNIT_POS as u32;
    let dim = CUBE_DIM_X;
    // Explicit 2D cube linearization (the grid folds into Y past 65535).
    let cube = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as u32;

    let blk = cube % bcount;
    let rest = cube / bcount;
    let fci = rest % n_fchunks;
    let col = rest / n_fchunks;
    let node = col % nodes_chunk;
    let tt = col / nodes_chunk;

    // Slack-cube guard: UNIFORM per cube (derived from CUBE_POS only), so
    // every unit of a cube takes the same branch and the barriers are safe.
    if tt < k {
        let f_base = fci * f_chunk;
        let mut fc_now = f_chunk;
        if f_base + fc_now > d {
            fc_now = d - f_base;
        }
        let slots = fc_now * nb * 3u32;

        let mut z = lid;
        while z < slots {
            shared[z as usize].store(F::new(0.0_f32));
            z += dim;
        }
        sync_cube();

        let rbase = (tt * nodes_total + node_base + node * node_stride) * 2u32;
        let s = ranges[rbase as usize];
        let e = ranges[(rbase + 1u32) as usize];
        let mut r = s + blk * dim + lid;
        while r < e {
            let i = order[(tt * n + r) as usize];
            let gv = g[(i * k + tt) as usize];
            let hv = h[(i * k + tt) as usize];
            let mut f = 0u32;
            while f < fc_now {
                let b = binned_t[((f_base + f) * n + i) as usize];
                let sbase = (f * nb + b) * 3u32;
                shared[sbase as usize].fetch_add(F::new(1.0_f32));
                shared[(sbase + 1u32) as usize].fetch_add(gv);
                shared[(sbase + 2u32) as usize].fetch_add(hv);
                f += 1u32;
            }
            r += bcount * dim;
        }
        sync_cube();

        let hbase = (col * d + f_base) * (nb * 3u32);
        let mut z2 = lid;
        while z2 < slots {
            let v2 = shared[z2 as usize].load();
            if v2 != F::new(0.0_f32) {
                hist[(hbase + z2) as usize].fetch_add(v2);
            }
            z2 += dim;
        }
    }
}

/// Split gains over the CUMULATIVE 3-slot histogram: one unit per `(tree,
/// node_in_chunk, feature, split bin s)` writes the sklearn gain
/// `G_l²/(H_l+λ+ε) + G_r²/(H_r+λ+ε) − loss_node` (`ε = 1e-15`,
/// `loss_node = G²/(H+λ+ε)`, or `0` on the ROOT level — sklearn
/// `TreeNode(value=0)`), or `−1` when a child violates `count ≥ min_leaf` or
/// `Σh ≥ min_hessian`. Straight-line global reads — no loops at all (the
/// `rf_split_scores_reg` shape).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_split_scores<F: Float + CubeElement>(
    hist: &Array<F>,
    scores: &mut Array<F>,
    min_leaf: F,
    min_hessian: F,
    l2: F,
    d: u32,
    nb: u32,
    nodes_chunk: u32,
    k: u32,
    root_level: u32,
) {
    let tid = ABSOLUTE_POS;
    let nsplit = nb - 1u32;
    let total = k * nodes_chunk * d * nsplit;
    if tid < total as usize {
        let col = (tid as u32) / nsplit; // (tt, node, f) flat
        let s = (tid as u32) % nsplit;
        let tn = col / d; // (tt, node) flat
        let fbase = col * (nb * 3u32);
        let tbase = (tn * d) * (nb * 3u32) + (nb - 1u32) * 3u32; // feature 0 totals

        let eps = F::new(1e-15_f32);
        let nl = hist[(fbase + s * 3u32) as usize];
        let gl = hist[(fbase + s * 3u32 + 1u32) as usize];
        let hl = hist[(fbase + s * 3u32 + 2u32) as usize];
        let nt = hist[tbase as usize];
        let gt = hist[(tbase + 1u32) as usize];
        let ht = hist[(tbase + 2u32) as usize];
        let nr = nt - nl;
        let gr = gt - gl;
        let hr = ht - hl;

        let mut loss_node = F::new(0.0_f32);
        if root_level == 0u32 {
            loss_node = gt * gt / (ht + l2 + eps);
        }

        let mut sc = F::new(-1.0_f32);
        if nl >= min_leaf {
            if nr >= min_leaf {
                if hl >= min_hessian {
                    if hr >= min_hessian {
                        sc = gl * gl / (hl + l2 + eps) + gr * gr / (hr + l2 + eps) - loss_node;
                    }
                }
            }
        }
        scores[tid] = sc;
    }
}

/// Per-node split finalize: one unit per `(tree, node_in_chunk)` arg-maxes its
/// `d × (nb−1)` gain slice (strict `>` → lowest-(feature, bin) tie-break, the
/// sklearn scan order), decides leaf-ness, and writes the model arrays at the
/// node's GLOBAL id:
///
/// leaf ⇔ `force_leaf` (bottom level) ∨ best gain `≤ 0` (sklearn
/// `gain <= 0 → _finalize_leaf`; all-invalid slices read `−1`).
///
/// `leaf_value` is ALWAYS written — the SHRUNK sklearn node value
/// `−lr · G/(H + λ + 1e-15)` (interior values are never read; an empty node
/// yields `0`). `level_base` must already fold in the node-chunk offset
/// (`2^L − 1 + node_base`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_best_split<F: Float + CubeElement>(
    hist: &Array<F>,
    scores: &Array<F>,
    edges: &Array<F>,
    split_feature: &mut Array<u32>,
    split_bin: &mut Array<u32>,
    threshold: &mut Array<F>,
    is_leaf: &mut Array<u32>,
    leaf_value: &mut Array<F>,
    lr: F,
    l2: F,
    d: u32,
    nb: u32,
    nodes_chunk: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
    force_leaf: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes_chunk;
    if tid < total as usize {
        let tt = (tid as u32) / nodes_chunk;
        let node = (tid as u32) % nodes_chunk;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;

        // Running best over the flat (f, s) gain slice — strict `>` keeps the
        // lowest flat index on ties (the sklearn feature-then-bin scan order).
        let nsplit = nb - 1u32;
        let sbase = (tid as u32) * d * nsplit;
        let mut best = F::new(-1.0_f32);
        let mut bk = 0u32;
        let mut i = 0u32;
        while i < d * nsplit {
            let sc = scores[(sbase + i) as usize];
            if sc > best {
                best = sc;
                bk = i;
            }
            i += 1u32;
        }

        // Node totals from the cumulative histogram (feature 0, last bin).
        let tbase = ((tid as u32) * d) * (nb * 3u32) + (nb - 1u32) * 3u32;
        let gt = hist[(tbase + 1u32) as usize];
        let ht = hist[(tbase + 2u32) as usize];
        let eps = F::new(1e-15_f32);
        leaf_value[midx as usize] = -lr * gt / (ht + l2 + eps);

        let mut leaf = 0u32;
        if force_leaf == 1u32 {
            leaf = 1u32;
        }
        if best <= F::new(0.0_f32) {
            leaf = 1u32;
        }
        is_leaf[midx as usize] = leaf;

        if leaf == 0u32 {
            let bf = bk / nsplit;
            let bs = bk % nsplit;
            split_feature[midx as usize] = bf;
            split_bin[midx as usize] = bs;
            threshold[midx as usize] = edges[(bf * nsplit + bs) as usize];
        } else {
            split_feature[midx as usize] = 0xFFFF_FFFFu32;
            split_bin[midx as usize] = 0u32;
            threshold[midx as usize] = F::new(0.0_f32);
        }
    }
}

/// Children `[start, end)` ranges for the NEXT level (the `rf_count_left`
/// shape with a `tree_base` model offset): one unit per `(tree, node)` counts
/// its left rows (`bin ≤ split_bin`, single `u32` accumulator) and writes the
/// two child ranges; leaves emit two EMPTY child ranges.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_count_left(
    binned: &Array<u32>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    split_feature: &Array<u32>,
    split_bin: &Array<u32>,
    is_leaf: &Array<u32>,
    ranges_next: &mut Array<u32>,
    n: u32,
    d: u32,
    nodes: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes;
    if tid < total as usize {
        let tt = (tid as u32) / nodes;
        let node = (tid as u32) % nodes;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;
        let next_nodes = nodes * 2u32;
        let lbase = (tt * next_nodes + node * 2u32) * 2u32;

        if is_leaf[midx as usize] == 1u32 {
            ranges_next[lbase as usize] = 0u32;
            ranges_next[(lbase + 1u32) as usize] = 0u32;
            ranges_next[(lbase + 2u32) as usize] = 0u32;
            ranges_next[(lbase + 3u32) as usize] = 0u32;
        } else {
            let fr = split_feature[midx as usize];
            let bs = split_bin[midx as usize];
            let s = ranges[((tt * nodes + node) * 2u32) as usize];
            let e = ranges[((tt * nodes + node) * 2u32 + 1u32) as usize];
            let mut cnt = 0u32;
            let mut r = s;
            while r < e {
                let i = order[(tt * n + r) as usize];
                if binned[(i * d + fr) as usize] <= bs {
                    cnt += 1u32;
                }
                r += 1u32;
            }
            ranges_next[lbase as usize] = s;
            ranges_next[(lbase + 1u32) as usize] = s + cnt;
            ranges_next[(lbase + 2u32) as usize] = s + cnt;
            ranges_next[(lbase + 3u32) as usize] = e;
        }
    }
}

/// Stable two-way partition into `order_next` (the `rf_partition` shape with a
/// `tree_base` model offset): one unit per `(tree, node)` re-scans its range
/// and scatters rows to the child ranges from [`gbt_count_left`] (read from
/// GLOBAL `ranges_next` — FINDING 002-B). Each parent owns its output range
/// exclusively — race-free without atomics.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_partition(
    binned: &Array<u32>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    ranges_next: &Array<u32>,
    split_feature: &Array<u32>,
    split_bin: &Array<u32>,
    is_leaf: &Array<u32>,
    order_next: &mut Array<u32>,
    n: u32,
    d: u32,
    nodes: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes;
    if tid < total as usize {
        let tt = (tid as u32) / nodes;
        let node = (tid as u32) % nodes;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;

        if is_leaf[midx as usize] == 0u32 {
            let fr = split_feature[midx as usize];
            let bs = split_bin[midx as usize];
            let s = ranges[((tt * nodes + node) * 2u32) as usize];
            let e = ranges[((tt * nodes + node) * 2u32 + 1u32) as usize];
            let next_nodes = nodes * 2u32;
            let lbase = (tt * next_nodes + node * 2u32) * 2u32;
            let mut li = ranges_next[lbase as usize];
            let mut ri = ranges_next[(lbase + 2u32) as usize];
            let mut r = s;
            while r < e {
                let i = order[(tt * n + r) as usize];
                if binned[(i * d + fr) as usize] <= bs {
                    order_next[(tt * n + li) as usize] = i;
                    li += 1u32;
                } else {
                    order_next[(tt * n + ri) as usize] = i;
                    ri += 1u32;
                }
                r += 1u32;
            }
        }
    }
}

/// Blocked left-count (the `rf_count_left_blocks` shape with a `tree_base`
/// model offset and feature-major `binned_t` reads): one unit per `(tree,
/// node, row block)` counts left-going rows in its contiguous chunk. The
/// serial [`gbt_count_left`] left level 0 to `K` single threads scanning all
/// `n` rows each — the dominant serial bottleneck once the histogram is
/// fast; blocking restores the row-scan parallelism on every level.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_count_left_blocks(
    binned_t: &Array<u32>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    split_feature: &Array<u32>,
    split_bin: &Array<u32>,
    is_leaf: &Array<u32>,
    blk_cnt: &mut Array<u32>,
    n: u32,
    nodes: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
    bcount: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes * bcount;
    if tid < total as usize {
        let blk = (tid as u32) % bcount;
        let tn = (tid as u32) / bcount;
        let node = tn % nodes;
        let tt = tn / nodes;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;

        let mut cnt = 0u32;
        if is_leaf[midx as usize] == 0u32 {
            let fr = split_feature[midx as usize];
            let bs = split_bin[midx as usize];
            let s = ranges[((tt * nodes + node) * 2u32) as usize];
            let e = ranges[((tt * nodes + node) * 2u32 + 1u32) as usize];
            let len = e - s;
            let per = (len + bcount - 1u32) / bcount;
            let lo = s + blk * per;
            let mut hi = lo + per;
            if hi > e {
                hi = e;
            }
            let mut r = lo;
            while r < hi {
                let i = order[(tt * n + r) as usize];
                if binned_t[(fr * n + i) as usize] <= bs {
                    cnt += 1u32;
                }
                r += 1u32;
            }
        }
        blk_cnt[tid] = cnt;
    }
}

/// Per-node block-count PREFIX + child ranges (the `rf_child_ranges` shape
/// with a `tree_base` model offset): one unit per `(tree, node)` converts its
/// own `bcount` slice of `blk_cnt` to an EXCLUSIVE prefix sum in place and
/// writes the two child `[start, end)` ranges for the next level; leaves emit
/// two EMPTY child ranges (exactly the serial [`gbt_count_left`] behavior).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_child_ranges(
    ranges: &Array<u32>,
    is_leaf: &Array<u32>,
    blk_cnt: &mut Array<u32>,
    ranges_next: &mut Array<u32>,
    nodes: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
    bcount: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes;
    if tid < total as usize {
        let tt = (tid as u32) / nodes;
        let node = (tid as u32) % nodes;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;
        let next_nodes = nodes * 2u32;
        let lbase = (tt * next_nodes + node * 2u32) * 2u32;

        if is_leaf[midx as usize] == 1u32 {
            ranges_next[lbase as usize] = 0u32;
            ranges_next[(lbase + 1u32) as usize] = 0u32;
            ranges_next[(lbase + 2u32) as usize] = 0u32;
            ranges_next[(lbase + 3u32) as usize] = 0u32;
        } else {
            let cbase = (tid as u32) * bcount;
            let mut run = 0u32;
            let mut b = 0u32;
            while b < bcount {
                let c = blk_cnt[(cbase + b) as usize];
                blk_cnt[(cbase + b) as usize] = run;
                run += c;
                b += 1u32;
            }
            let s = ranges[((tt * nodes + node) * 2u32) as usize];
            let e = ranges[((tt * nodes + node) * 2u32 + 1u32) as usize];
            ranges_next[lbase as usize] = s;
            ranges_next[(lbase + 1u32) as usize] = s + run;
            ranges_next[(lbase + 2u32) as usize] = s + run;
            ranges_next[(lbase + 3u32) as usize] = e;
        }
    }
}

/// Blocked STABLE two-way partition (the `rf_partition_blocks` shape with a
/// `tree_base` model offset): one unit per `(tree, node, row block)` re-scans
/// its chunk and scatters to the child ranges; its left cursor starts at
/// `s + prefix[blk]`, its right cursor at `mid + (blk·per − prefix[blk])`
/// (`prefix` = the in-place exclusive prefix from [`gbt_child_ranges`], `mid`
/// read from GLOBAL `ranges_next` — FINDING 002-B). Blocks write disjoint
/// output sub-ranges in original row order — STABLE, bitwise-identical
/// `order_next` to the serial [`gbt_partition`], race-free without atomics.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_partition_blocks(
    binned_t: &Array<u32>,
    order: &Array<u32>,
    ranges: &Array<u32>,
    ranges_next: &Array<u32>,
    split_feature: &Array<u32>,
    split_bin: &Array<u32>,
    is_leaf: &Array<u32>,
    blk_cnt: &Array<u32>,
    order_next: &mut Array<u32>,
    n: u32,
    nodes: u32,
    k: u32,
    tree_base: u32,
    level_base: u32,
    total_nodes: u32,
    bcount: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * nodes * bcount;
    if tid < total as usize {
        let blk = (tid as u32) % bcount;
        let tn = (tid as u32) / bcount;
        let node = tn % nodes;
        let tt = tn / nodes;
        let gnode = level_base + node;
        let midx = (tree_base + tt) * total_nodes + gnode;

        if is_leaf[midx as usize] == 0u32 {
            let fr = split_feature[midx as usize];
            let bs = split_bin[midx as usize];
            let s = ranges[((tt * nodes + node) * 2u32) as usize];
            let e = ranges[((tt * nodes + node) * 2u32 + 1u32) as usize];
            let len = e - s;
            let per = (len + bcount - 1u32) / bcount;
            let lo = s + blk * per;
            let mut hi = lo + per;
            if hi > e {
                hi = e;
            }
            let next_nodes = nodes * 2u32;
            let lbase = (tt * next_nodes + node * 2u32) * 2u32;
            let pfx = blk_cnt[(tn * bcount + blk) as usize];
            let mid = ranges_next[(lbase + 1u32) as usize];
            // `blk·per ≥ pfx` always (the prefix counts a subset of the rows
            // before the block), so the subtraction cannot wrap.
            let rows_before = blk * per;
            let mut li = s + pfx;
            let mut ri = mid + (rows_before - pfx);
            let mut r = lo;
            while r < hi {
                let i = order[(tt * n + r) as usize];
                if binned_t[(fr * n + i) as usize] <= bs {
                    order_next[(tt * n + li) as usize] = i;
                    li += 1u32;
                } else {
                    order_next[(tt * n + ri) as usize] = i;
                    ri += 1u32;
                }
                r += 1u32;
            }
        }
    }
}

/// Fold this iteration's trees into the TRAIN raw predictions: one unit per
/// `(row, class)` walks tree `tree_base + c` on the BINNED features (bounded
/// `max_depth` descent, `bin ≤ split_bin → left` — exactly the training
/// partition rule) and adds the reached leaf's shrunk value to `raw[i, c]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gbt_update_raw<F: Float + CubeElement>(
    binned: &Array<u32>,
    split_feature: &Array<u32>,
    split_bin: &Array<u32>,
    is_leaf: &Array<u32>,
    leaf_value: &Array<F>,
    raw: &mut Array<F>,
    n: u32,
    d: u32,
    k: u32,
    max_depth: u32,
    tree_base: u32,
    total_nodes: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = n * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        let c = (tid as u32) % k;
        let tbase = (tree_base + c) * total_nodes;
        let mut cur = 0u32;
        let mut l = 0u32;
        while l < max_depth {
            if is_leaf[(tbase + cur) as usize] == 0u32 {
                let fr = split_feature[(tbase + cur) as usize];
                let bs = split_bin[(tbase + cur) as usize];
                let mut nxt = 2u32 * cur + 2u32;
                if binned[(i * d + fr) as usize] <= bs {
                    nxt = 2u32 * cur + 1u32;
                }
                cur = nxt;
            }
            l += 1u32;
        }
        raw[tid] += leaf_value[(tbase + cur) as usize];
    }
}

/// Predict-time FUSED traverse-and-sum: one unit per `(iter block, query
/// row, class)` walks its `iters_per_block` complete-layout trees and
/// accumulates the reached leaf values directly into
/// `out[blk·(q·k) + row·k + c]` — no per-`(tree, row)` leaf-index
/// intermediate (the old `rf_predict_leaf` → `gbt_sum_raw` pair wrote and
/// re-read an `n_trees × q` u32 buffer, which dominated predict bandwidth at
/// large `n_trees × q`). Block 0 seeds `baseline[c]`; the host folds the
/// (deterministically ordered) block axis with [`gbt_sum_partials`], or skips
/// it entirely when one block suffices — the block split exists only to keep
/// the launch grid wide when `q·k` alone would under-occupy the device.
///
/// Unit index order is `row`-fastest so a warp walks 32 consecutive rows of
/// the SAME tree: the tree's node arrays stay hot in L1/L2 while `x` rows are
/// read once per level. A reached leaf exits the level loop early
/// (`l = max_depth` — `cur` would be a fixpoint anyway, the remaining
/// `is_leaf` re-reads are pure waste).
#[cube(launch)]
pub fn gbt_predict_fused<F: Float + CubeElement>(
    x: &Array<F>,
    split_feature: &Array<u32>,
    threshold: &Array<F>,
    is_leaf: &Array<u32>,
    leaf_value: &Array<F>,
    baseline: &Array<F>,
    out: &mut Array<F>,
    q: u32,
    d: u32,
    k: u32,
    max_depth: u32,
    n_iters: u32,
    total_nodes: u32,
    iters_per_block: u32,
    n_blocks: u32,
) {
    let tid = ABSOLUTE_POS;
    let qk = q * k;
    let total = qk * n_blocks;
    if tid < total as usize {
        let blk = (tid as u32) / qk;
        let rem = (tid as u32) % qk;
        let c = rem / q;
        let row = rem % q;
        let it0 = blk * iters_per_block;
        let mut it_end = it0 + iters_per_block;
        if it_end > n_iters {
            it_end = n_iters;
        }
        let mut acc = F::new(0.0_f32);
        if blk == 0u32 {
            acc = baseline[c as usize];
        }
        let mut it = it0;
        while it < it_end {
            let tbase = (it * k + c) * total_nodes;
            let mut cur = 0u32;
            let mut l = 0u32;
            while l < max_depth {
                if is_leaf[(tbase + cur) as usize] == 0u32 {
                    let fr = split_feature[(tbase + cur) as usize];
                    let thr = threshold[(tbase + cur) as usize];
                    let mut nxt = 2u32 * cur + 2u32;
                    if x[(row * d + fr) as usize] < thr {
                        nxt = 2u32 * cur + 1u32;
                    }
                    cur = nxt;
                    l += 1u32;
                } else {
                    l = max_depth;
                }
            }
            acc += leaf_value[(tbase + cur) as usize];
            it += 1u32;
        }
        out[(blk * qk + row * k + c) as usize] = acc;
    }
}

/// Fold [`gbt_predict_fused`]'s block axis: one unit per `(row, class)` slot
/// sums the `n_blocks` partials in ascending block order (deterministic —
/// same run, same grouping, same bits). Only launched when `n_blocks > 1`.
#[cube(launch)]
pub fn gbt_sum_partials<F: Float + CubeElement>(
    partials: &Array<F>,
    raw_out: &mut Array<F>,
    qk: u32,
    n_blocks: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < qk as usize {
        let mut acc = F::new(0.0_f32);
        let mut b = 0u32;
        while b < n_blocks {
            acc += partials[(b * qk + (tid as u32)) as usize];
            b += 1u32;
        }
        raw_out[tid] = acc;
    }
}

/// Binary predict_proba: `p = σ(raw)` per query row → `[1 − p, p]` (the
/// sklearn two-column form). One unit per row.
#[cube(launch)]
pub fn gbt_proba_binary<F: Float + CubeElement>(raw: &Array<F>, proba: &mut Array<F>, q: u32) {
    let tid = ABSOLUTE_POS;
    if tid < q as usize {
        let p = F::new(1.0_f32) / (F::new(1.0_f32) + F::exp(-raw[tid]));
        proba[(2u32 * (tid as u32)) as usize] = F::new(1.0_f32) - p;
        proba[(2u32 * (tid as u32) + 1u32) as usize] = p;
    }
}

/// Multiclass predict_proba: `softmax(raw)` per `(row, class)` element from
/// the staged `row_max` / `row_sumexp` ([`gbt_row_max`] / [`gbt_row_sumexp`]
/// over the `q × k` raw scores). No loops.
#[cube(launch)]
pub fn gbt_proba_multi<F: Float + CubeElement>(
    raw: &Array<F>,
    row_max: &Array<F>,
    row_sumexp: &Array<F>,
    proba: &mut Array<F>,
    q: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = q * k;
    if tid < total as usize {
        let i = (tid as u32) / k;
        proba[tid] = F::exp(raw[tid] - row_max[i as usize]) / row_sumexp[i as usize];
    }
}
