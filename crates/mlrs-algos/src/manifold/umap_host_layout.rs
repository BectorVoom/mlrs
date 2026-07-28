//! Host-side UMAP SGD layout driver (UMAP-PERF-CPU) — the cpu-backend twin of
//! the [`umap_layout_step`](mlrs_kernels::umap_layout_step) device kernel.
//!
//! ## Why this exists (the measurement)
//! [`host_epoch_driver`](super::umap) runs ONE device launch per epoch, and each
//! epoch re-uploads the whole embedding plus four CSR buffers and reads the
//! embedding back. On a GPU that is a fine shape — the launch is `n` cubes wide
//! and the round-trip amortizes over real parallel work. On `cubecl-cpu` all
//! three costs are pure loss: the launch is `CubeDim {x: 1}`, i.e. a SINGLE OS
//! thread walking all `n` cubes serially at LLVM `-O0`, and the per-epoch
//! upload/read-back pair is a host sync either side of it. Measured on this
//! 16-core host at `n = 500, d = 8, k = 15`
//! (`umap_perf_test::umap_fit_stage_breakdown`):
//!
//! ```text
//!   layout SGD    0.1725 s / epoch    -> 34.5 s at the umap-learn default of 200
//! ```
//!
//! umap-learn fits the whole `n = 2000` problem — graph included — in ~2 s.
//!
//! ## What this is NOT
//! It is not a different algorithm and not an approximation. [`drive`] replays
//! the device kernel's arithmetic STATEMENT FOR STATEMENT, in the same owner
//! order (ascending, which is exactly how `cubecl-cpu` sequences the per-owner
//! cubes) and the same within-owner edge order, so on the cpu backend the
//! embedding it produces is bit-identical to the kernel's up to `powf`'s libm
//! rounding. The three things that make it fast are all bookkeeping, not math:
//!
//! - **No device round-trip.** The coordinates never leave the host, so the
//!   per-epoch upload + launch + read-back disappears entirely.
//! - **No per-epoch allocation.** The old driver built `2n` fresh `Vec`s per
//!   epoch to bucket edges by owner and then flattened them into CSR arrays. The
//!   edge→owner map is FIXED across epochs, so [`OwnerIndex`] computes the
//!   grouping ONCE (counting sort) and every epoch just walks it.
//! - **Native code.** Scalar host Rust at `-O2` instead of `-O0` MLIR.
//!
//! ## Precision
//! The device path computes each epoch in `F` (the estimator's float type),
//! round-tripping the host `f64` buffer through `F` at every epoch boundary. This
//! driver runs the WHOLE layout in one precision, selected from `F`: `f64` for
//! the `f64` estimator (bit-identical to the kernel), `f32` for the `f32` one
//! (which drops the redundant `f32 → f64 → f32` intermediates and matches
//! umap-learn, whose layout is `float32` throughout).
//!
//! Reproducibility (D-05) is preserved: the driver is single-threaded and every
//! negative-sample index is still drawn HOST-side by `SplitMix64` keyed as a pure
//! function of `(seed, epoch, edge)`, so two same-`random_state` fits stay
//! byte-identical.
//!
//! Tests live in `crates/mlrs-algos/tests/umap_test.rs` (AGENTS.md §2).

use mlrs_backend::capability;
use mlrs_backend::prims::rng::SplitMix64;

/// Should the host driver run the layout on this backend?
///
/// True on `cpu` only — on wgpu/cuda/rocm the device kernel is a real parallel
/// launch and this serial host loop would be a large regression. A perf path is
/// gated on the target it was MEASURED on, never extrapolated.
///
/// `MLRS_UMAP_HOST_LAYOUT=0` forces the device kernel back on for on-target A/B;
/// `=1` cannot force the host driver onto a non-cpu backend.
pub fn host_layout_applicable() -> bool {
    capability::active_backend_name() == "cpu"
        && mlrs_backend::abflag::var("MLRS_UMAP_HOST_LAYOUT")
            .map(|v| v != "0")
            .unwrap_or(true)
}

/// The fixed edge→owner grouping, computed once and reused by every epoch.
///
/// The device driver rebuilt this per epoch as `2n` `Vec`s plus a flatten pass.
/// The mapping never changes, so it is hoisted: `order` lists the edge ids
/// grouped by owner (stable — ascending edge id within an owner, which is the
/// order the old per-owner buckets were filled in, hence the order the kernel
/// consumed them), and `offsets[o]..offsets[o+1]` is owner `o`'s slice of it.
/// The fixed edge→owner grouping, with the per-edge schedule PERMUTED into that
/// order so an owner's edges are a CONTIGUOUS range of every per-edge array.
///
/// Contiguity is what makes the epoch loop splittable: owner `o`'s slice
/// `offsets[o]..offsets[o+1]` indexes its edges, its sample clocks, and (since
/// owner `o` occupies coordinate row `o`) its embedding row, so a block of
/// owners hands a worker disjoint mutable slices of all three with no locking.
pub struct OwnerIndex {
    offsets: Vec<u32>,
    /// Original edge id, kept because the negative-sampling substream is keyed
    /// on it (`(seed, epoch, edge)` — permuting must not change the RNG stream).
    edge_id: Vec<u32>,
    /// Positive-edge target vertex, permuted into owner order.
    tail: Vec<u32>,
    /// `epochs_per_sample`, permuted into owner order.
    eps: Vec<f64>,
}

impl OwnerIndex {
    /// Group the edges by owner via counting sort and permute the schedule.
    ///
    /// `head` gives each edge's owner and `tail` its target; `n_owners` bounds
    /// the owner axis, and edges whose owner is out of range are DROPPED exactly
    /// as the kernel's `row < n_owners` guard drops them. Within an owner the
    /// edges stay in ASCENDING EDGE ID — the order the device driver's per-owner
    /// buckets were filled in, hence the order the kernel consumed them.
    pub fn build(
        head: &[usize],
        tail: &[usize],
        epochs_per_sample: &[f64],
        n_owners: usize,
    ) -> Self {
        let mut counts = vec![0u32; n_owners + 1];
        for &h in head {
            if h < n_owners {
                counts[h + 1] += 1;
            }
        }
        for o in 0..n_owners {
            counts[o + 1] += counts[o];
        }
        let offsets = counts.clone();
        let mut cursor = counts;
        let total = offsets[n_owners] as usize;
        let mut edge_id = vec![0u32; total];
        let mut tail_ord = vec![0u32; total];
        let mut eps_ord = vec![0.0f64; total];
        for (e, &h) in head.iter().enumerate() {
            if h < n_owners {
                let slot = cursor[h] as usize;
                edge_id[slot] = e as u32;
                tail_ord[slot] = tail[e] as u32;
                eps_ord[slot] = epochs_per_sample[e];
                cursor[h] += 1;
            }
        }
        Self {
            offsets,
            edge_id,
            tail: tail_ord,
            eps: eps_ord,
        }
    }

    /// Number of scheduled edges (the length of every permuted per-edge array).
    fn len(&self) -> usize {
        self.edge_id.len()
    }
}

/// The layout problem, in whichever precision the estimator's `F` selects.
///
/// Only the operations the kernel actually performs are required, so this stays
/// a two-impl trait rather than pulling in a numeric-tower dependency.
pub trait LayoutFloat: Copy + PartialOrd + Send + Sync {
    const ZERO: Self;
    const FOUR: Self;
    const ONE: Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    fn powf(self, e: Self) -> Self;
}

impl LayoutFloat for f64 {
    const ZERO: Self = 0.0;
    const FOUR: Self = 4.0;
    const ONE: Self = 1.0;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        self / o
    }
    #[inline]
    fn powf(self, e: Self) -> Self {
        f64::powf(self, e)
    }
}

impl LayoutFloat for f32 {
    const ZERO: Self = 0.0;
    const FOUR: Self = 4.0;
    const ONE: Self = 1.0;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        self / o
    }
    #[inline]
    fn powf(self, e: Self) -> Self {
        f32::powf(self, e)
    }
}

/// Everything the epoch loop needs that does not change between epochs.
pub struct LayoutParams {
    /// Number of contiguous OWNER rows at the front of the coordinate buffer.
    pub n_owners: usize,
    /// Total vertex count — the bound the kernel's GATHER index check uses.
    pub n_vertices: usize,
    /// Embedding dimensionality (`n_components`).
    pub dim: usize,
    /// UMAP `a`/`b` curve parameters.
    pub a: f64,
    pub b: f64,
    /// Repulsion strength (`repulsion_strength`).
    pub gamma: f64,
    /// Initial learning rate, before the per-epoch `1 − n/n_epochs` decay.
    pub initial_alpha: f64,
    /// Negative samples drawn per positive sample.
    pub negative_sample_rate: usize,
    pub n_epochs: usize,
    /// RNG seed for the `(seed, epoch, edge)` negative-sampling substreams.
    pub seed: u64,
    /// The two FIXED substream separators from `umap.rs` (part of the D-05
    /// byte-identical contract — passed in rather than duplicated here so there
    /// stays exactly one definition of each).
    pub substream_seed_mult: u64,
    pub substream_epoch_mult: u64,
    /// Worker count override. `None` uses
    /// [`cpu_launch_units`](mlrs_backend::capability::cpu_launch_units).
    ///
    /// Exists so the thread-count independence the epoch snapshot buys is
    /// TESTABLE rather than merely asserted: `umap_test::
    /// host_layout_is_thread_count_independent` drives the same problem at
    /// several worker counts and requires bit-identical coordinates. Production
    /// callers pass `None`.
    pub units: Option<usize>,
}

/// Run the whole SGD layout on the host, updating `embedding` (row-major
/// `(n_vertices, dim)` host `f64`) in place.
///
/// `owners` is the precomputed edge grouping (which also carries the permuted
/// per-edge schedule); `p` carries the curve/decay constants. The generic `C` is
/// the arithmetic precision — see the module docs.
///
/// ## The epoch snapshot, and why it is what makes this parallel
/// Each owner writes ONLY its own coordinate row (`move_other = 0` on both the
/// fit and transform paths), and reads its own row plus its neighbours'. This
/// splits the epoch over workers the moment the neighbour reads are pinned:
/// every worker reads foreign rows from `snapshot`, a copy of the coordinates
/// taken at the START of the epoch, and writes only rows it exclusively owns.
///
/// The consequence worth stating plainly is DETERMINISM, not speed. Because no
/// worker can observe another's in-flight write, the result does not depend on
/// the worker count, the split, or the interleaving — 1 thread and 16 threads
/// produce bit-identical coordinates, so the D-05 same-seed reproducibility
/// contract survives the parallelism instead of being traded away for it.
/// (`umap_test::reproducible_f64` is the executable form of that claim.)
///
/// It does change the update from Gauss-Seidel to Jacobi WITHIN an epoch: a
/// neighbour already processed earlier in the same epoch is read at its
/// epoch-start position rather than its updated one. That is a deliberate,
/// gated choice, not an oversight — umap-learn's own parallel mode is Hogwild,
/// which pins nothing at all, and the structural property gates
/// (`umap_test::layout_property_*`, trustworthiness / kNN-overlap / downstream
/// ARI against the umap-learn oracle) are what decide whether the substitution
/// is acceptable. An owner still reads its OWN row live, so the sequential
/// accumulation of one owner's edges within an epoch is unchanged.
pub fn drive<C: LayoutFloat>(embedding: &mut [f64], owners: &OwnerIndex, p: &LayoutParams) {
    let dim = p.dim;
    let n_owners = p.n_owners;
    let n_vertices = p.n_vertices;

    // Per-edge sample clocks (umap's epoch_of_next_sample / _negative_sample),
    // in host f64 exactly as the device driver kept them — but indexed in OWNER
    // ORDER, so each owner's clocks are a contiguous slice a worker can own.
    let mut next_sample: Vec<f64> = owners.eps.clone();
    let epochs_per_negative: Vec<f64> = owners
        .eps
        .iter()
        .map(|&e| {
            if e > 0.0 {
                e / p.negative_sample_rate as f64
            } else {
                -1.0
            }
        })
        .collect();
    let mut next_negative: Vec<f64> = epochs_per_negative.clone();
    debug_assert_eq!(next_sample.len(), owners.len());

    // The coordinates, in the arithmetic precision (see the module docs). The
    // device path round-tripped this buffer through `F` at every epoch boundary;
    // here it is converted once in and once out.
    let mut emb: Vec<C> = embedding.iter().map(|&v| C::from_f64(v)).collect();
    let mut snapshot: Vec<C> = emb.clone();

    let consts = Consts::<C>::new(p);

    // Worker count: one block of owners each. Blocks must be large enough that
    // the spawn is not the dominant cost — this loop runs once per EPOCH.
    let units = p
        .units
        .unwrap_or_else(|| capability::cpu_launch_units() as usize)
        .max(1);
    let rows_per_block = n_owners.div_ceil(units).max(MIN_OWNERS_PER_WORKER);

    for epoch in 0..p.n_epochs {
        let alpha = C::from_f64(p.initial_alpha * (1.0 - epoch as f64 / p.n_epochs as f64));
        let epoch_f = epoch as f64;

        // Pin the neighbour reads for this epoch (see the fn docs).
        snapshot.copy_from_slice(&emb);

        if rows_per_block >= n_owners || units == 1 {
            epoch_block(
                &mut emb[..n_owners * dim],
                &snapshot,
                &mut next_sample,
                &mut next_negative,
                &epochs_per_negative,
                owners,
                p,
                &consts,
                0,
                alpha,
                epoch_f,
                epoch as u64,
                dim,
                n_vertices,
            );
        } else {
            // Split owners into contiguous blocks. Owner `o` holds coordinate row
            // `o` and clock slots `offsets[o]..offsets[o+1]`, so a block of owners
            // maps to ONE contiguous slice of each — handed out by `split_at_mut`,
            // which is what proves the disjointness to the compiler.
            let (owned_rows, _frozen_tail) = emb.split_at_mut(n_owners * dim);
            let mut rows_rest = owned_rows;
            let mut ns_rest: &mut [f64] = &mut next_sample;
            let mut nn_rest: &mut [f64] = &mut next_negative;
            let snapshot_ref: &[C] = &snapshot;
            std::thread::scope(|scope| {
                let mut owner0 = 0usize;
                while owner0 < n_owners {
                    let rows = rows_per_block.min(n_owners - owner0);
                    let clocks = (owners.offsets[owner0 + rows] - owners.offsets[owner0]) as usize;
                    let (rows_blk, rows_tail) = rows_rest.split_at_mut(rows * dim);
                    let (ns_blk, ns_tail) = ns_rest.split_at_mut(clocks);
                    let (nn_blk, nn_tail) = nn_rest.split_at_mut(clocks);
                    rows_rest = rows_tail;
                    ns_rest = ns_tail;
                    nn_rest = nn_tail;
                    let epn = &epochs_per_negative;
                    let consts = &consts;
                    let start = owner0;
                    scope.spawn(move || {
                        epoch_block(
                            rows_blk,
                            snapshot_ref,
                            ns_blk,
                            nn_blk,
                            epn,
                            owners,
                            p,
                            consts,
                            start,
                            alpha,
                            epoch_f,
                            epoch as u64,
                            dim,
                            n_vertices,
                        );
                    });
                    owner0 += rows;
                }
            });
        }
    }

    for (dst, src) in embedding.iter_mut().zip(emb.iter()) {
        *dst = src.to_f64();
    }
}

/// Smallest owner block worth its own worker. The epoch loop spawns once per
/// epoch, so a block too small pays the spawn more often than it saves work.
const MIN_OWNERS_PER_WORKER: usize = 64;

/// The loop-invariant curve constants, hoisted out of the epoch loop.
///
/// Hoisting is exact — these are pure functions of `a`, `b` and `gamma` — and
/// the ASSOCIATION matters: each is grouped exactly as the kernel writes it
/// (`(-2·a)·b`, `(2·gamma)·b`), so the products round identically.
struct Consts<C> {
    a: C,
    b: C,
    b_minus_1: C,
    neg_two_ab: C,
    two_gamma_b: C,
    milli: C,
}

impl<C: LayoutFloat> Consts<C> {
    fn new(p: &LayoutParams) -> Self {
        let a = C::from_f64(p.a);
        let b = C::from_f64(p.b);
        let gamma = C::from_f64(p.gamma);
        Self {
            a,
            b,
            b_minus_1: b.sub(C::ONE),
            neg_two_ab: C::from_f64(-2.0).mul(a).mul(b),
            two_gamma_b: C::from_f64(2.0).mul(gamma).mul(b),
            milli: C::from_f64(0.001),
        }
    }
}

/// One epoch's updates for the contiguous owner block starting at `owner0`.
///
/// `rows` is the block's slice of the LIVE coordinates (owner `owner0 + i` is
/// `rows[i*dim..]`); `snapshot` is the whole epoch-start buffer, read for every
/// FOREIGN vertex. `next_sample` / `next_negative` are the block's slice of the
/// clocks. Nothing here is shared mutably with another block.
#[allow(clippy::too_many_arguments)]
fn epoch_block<C: LayoutFloat>(
    rows: &mut [C],
    snapshot: &[C],
    next_sample: &mut [f64],
    next_negative: &mut [f64],
    epochs_per_negative: &[f64],
    owners: &OwnerIndex,
    p: &LayoutParams,
    k: &Consts<C>,
    owner0: usize,
    alpha: C,
    epoch_f: f64,
    epoch: u64,
    dim: usize,
    n_vertices: usize,
) {
    let n_block = rows.len() / dim;
    // Clock slots are indexed globally; this block's slices start here.
    let clock0 = owners.offsets[owner0] as usize;

    // Reused across owners: the edges this owner sampled this epoch and how many
    // negatives each drew. The kernel walks all of an owner's positive edges
    // FIRST and only then its negatives, so the negative pass needs the positive
    // pass's due-set — recorded here instead of re-derived (the clocks have
    // already advanced by then).
    let mut due: Vec<(u32, i64)> = Vec::with_capacity(32);

    for i in 0..n_block {
        let owner = owner0 + i;
        let cur = &mut rows[i * dim..(i + 1) * dim];
        let lo = owners.offsets[owner] as usize;
        let hi = owners.offsets[owner + 1] as usize;
        due.clear();

        // ============================================================
        // ATTRACTIVE pass — the owner's due positive edges, in edge order.
        // ============================================================
        for t in lo..hi {
            let eps_e = owners.eps[t];
            if eps_e <= 0.0 {
                continue; // never-sampled edge (zero weight)
            }
            let slot = t - clock0;
            if next_sample[slot] > epoch_f {
                continue; // not due this epoch
            }

            // How many negative samples this edge draws this epoch — settled
            // here, with the clocks, so the negative pass replays exactly the
            // sequence the device driver packed into `neg_idx`.
            let epn = epochs_per_negative[t];
            let n_neg = if epn > 0.0 {
                ((epoch_f - next_negative[slot]) / epn).floor() as i64
            } else {
                0
            };
            if n_neg > 0 {
                next_negative[slot] += n_neg as f64 * epn;
            }
            next_sample[slot] += eps_e;
            due.push((owners.edge_id[t], n_neg));

            let other = owners.tail[t] as usize;
            if other >= n_vertices {
                continue; // the kernel's GATHER bound check (T-14-10)
            }
            let other_base = other * dim;

            let mut dist_sq = C::ZERO;
            for d0 in 0..dim {
                let diff = cur[d0].sub(snapshot[other_base + d0]);
                dist_sq = dist_sq.add(diff.mul(diff));
            }

            let mut grad_coeff = C::ZERO;
            if dist_sq > C::ZERO {
                let pow_b = dist_sq.powf(k.b);
                let pow_bm1 = dist_sq.powf(k.b_minus_1);
                let num = k.neg_two_ab.mul(pow_bm1);
                let den = k.a.mul(pow_b).add(C::ONE);
                grad_coeff = num.div(den);
            }

            for d1 in 0..dim {
                let cur_d = cur[d1];
                let other_d = snapshot[other_base + d1];
                let grad_d = clip4(grad_coeff.mul(cur_d.sub(other_d)));
                cur[d1] = cur_d.add(grad_d.mul(alpha));
                // `move_other` is 0 on BOTH the fit and transform paths
                // (`umap::FIT_MOVE_OTHER`), so the neighbour is never written —
                // the branch the kernel keeps for a two-sided mode nothing
                // currently launches is simply absent here. (It is also what
                // makes the owner rows disjoint, hence splittable.)
            }
        }

        // ============================================================
        // REPULSIVE pass — the owner's host-drawn negative samples, in the same
        // (edge, draw) order the device driver packed them.
        // ============================================================
        for &(edge_id, n_neg) in &due {
            if n_neg <= 0 {
                continue;
            }
            let sub_seed = p
                .seed
                .wrapping_mul(p.substream_seed_mult)
                .wrapping_add(epoch.wrapping_mul(p.substream_epoch_mult))
                .wrapping_add(edge_id as u64);
            let mut rng = SplitMix64::new(sub_seed);
            for _ in 0..n_neg {
                let other = rng.next_below(n_vertices as u64) as usize;
                if other >= n_vertices || other == owner {
                    continue; // the kernel's bound check + self-sample skip
                }
                let other_base = other * dim;

                let mut dist_sq = C::ZERO;
                for d2 in 0..dim {
                    let diff = cur[d2].sub(snapshot[other_base + d2]);
                    dist_sq = dist_sq.add(diff.mul(diff));
                }

                if dist_sq > C::ZERO {
                    let pow_b = dist_sq.powf(k.b);
                    let den = k.milli.add(dist_sq).mul(k.a.mul(pow_b).add(C::ONE));
                    let grad_coeff = k.two_gamma_b.div(den);
                    for d3 in 0..dim {
                        let cur_d = cur[d3];
                        let other_d = snapshot[other_base + d3];
                        let grad_d = clip4(grad_coeff.mul(cur_d.sub(other_d)));
                        cur[d3] = cur_d.add(grad_d.mul(alpha));
                    }
                } else {
                    // Coincident points: umap's fixed per-dim push of 4.0.
                    for d3 in 0..dim {
                        let cur_d = cur[d3];
                        cur[d3] = cur_d.add(C::FOUR.mul(alpha));
                    }
                }
            }
        }
    }
}

/// `clip(v, −4, 4)` — the kernel's finite-literal statement-`if` form (never a
/// `min`/`max` intrinsic or an infinity sentinel), replicated so the clipping
/// boundary rounds identically.
#[inline]
fn clip4<C: LayoutFloat>(v: C) -> C {
    let mut g = v;
    if g > C::FOUR {
        g = C::FOUR;
    }
    if g < C::ZERO.sub(C::FOUR) {
        g = C::ZERO.sub(C::FOUR);
    }
    g
}
