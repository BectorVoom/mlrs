//! `kmeans` — Lloyd centroid-update + inertia `#[cube]` kernels (CLUSTER-01,
//! D-01).
//!
//! Two feature-free `#[cube]` kernels generic over `<F: Float + CubeElement>`
//! for the n-heavy parts of the Lloyd iteration, composed by
//! `mlrs_backend::prims::kmeans`:
//!
//! - [`centroid_sumcount`] — per (centroid `c`, feature `j`) GATHER: scan all
//!   `n` rows and accumulate `Σ X[i, j]` over rows whose `label[i] == c` plus
//!   the per-centroid count. One unit per `(c, j)` output element reads every
//!   row (a GATHER, never a scatter) so there is NO atomic / no race — the host
//!   then divides each sum by its count to get the mean. This is the n-heavy
//!   accumulation (the `center_columns` `c = tid % cols` per-element map
//!   generalised to a per-output gather).
//! - [`inertia_rows`] — per ROW `i` GATHER: `Σ_j (X[i, j] − centers[label_i,
//!   j])²`, the squared distance from each sample to its ASSIGNED center (the
//!   `reduce_sumsq` squared-norm accumulation specialised to the assigned
//!   center). The host then sums the `n` per-row partials to the scalar inertia
//!   `Σ_i ‖X_i − centers[labels_i]‖²`.
//!
//! ## cubecl-cpu MLIR safety (the primary correctness gate)
//! cpu(f64) is the primary gate and its MLIR `run pass` lowering REJECTS
//! `SharedMemory` combined with mutable `bool` flags, the `F::INFINITY` const,
//! and descending-shift loops (proven in plan 05-02). Both kernels here use ONLY
//! `F`/`u32` accumulators and ascending `while` scans with `if` guards — no
//! `SharedMemory`, no mutable `bool`, no infinity sentinel, no descending shift
//! — so they LAUNCH (not just compile) on cubecl-cpu. The per-output GATHER
//! layout also avoids cross-unit atomics, which the cpu backend does not lower
//! reliably.
//!
//! All kernels carry NO backend feature (D-13). Empty-cluster relocation and the
//! final mean/sum finalize are HOST concerns (small-k reductions, RESEARCH Open
//! Q1) handled in `prims::kmeans`. Tests live in
//! `crates/mlrs-backend/tests/{kmeanspp,lloyd}_test.rs` (AGENTS.md §2).

use cubecl::prelude::*;

pub use self::centroid_sumcount as kmeans_centroid_sumcount;
pub use self::inertia_rows as kmeans_inertia_rows;

/// Per-centroid feature SUM + per-centroid COUNT for the Lloyd centroid update
/// (CLUSTER-01).
///
/// One unit per `(c, j)` output element (`c` in `0..k`, `j` in `0..d`,
/// `ABSOLUTE_POS = c * d + j`): scan ALL `n` rows of the row-major `x`
/// (`n × d`) and accumulate `Σ X[i, j]` over the rows whose `labels[i] == c`.
/// Unit `(c, 0)` additionally writes the per-centroid count into `counts[c]`.
///
/// - `x` is the row-major `n × d` sample matrix.
/// - `labels` is the length-`n` assignment (`u32`, one cluster id per row).
/// - `sums` is the `k × d` output (per-centroid feature sums).
/// - `counts` is the length-`k` output (per-centroid assigned-row count).
/// - `n`, `d`, `k` are scalar `u32` args passed BY VALUE (cubecl 0.10 — no
///   `ScalarArg`, like `dist_combine_clamp`'s `rows`/`cols`).
///
/// This is a GATHER (each output reads many inputs), so there is no scatter
/// race and no atomic — cubecl-cpu lowers it directly. The host divides
/// `sums[c, j]` by `counts[c]` to form the centroid mean (and relocates empty
/// clusters first, so `counts[c] == 0` never reaches a divide).
#[cube(launch)]
pub fn centroid_sumcount<F: Float + CubeElement>(
    x: &Array<F>,
    labels: &Array<u32>,
    sums: &mut Array<F>,
    counts: &mut Array<u32>,
    n: u32,
    d: u32,
    k: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = k * d;
    if tid < total as usize {
        let c = (tid as u32) / d;
        let j = (tid as u32) % d;

        // GATHER: accumulate Σ X[i, j] over rows with label == c. Only F/u32
        // accumulators + an ascending scan + an `if` guard — no SharedMemory, no
        // mutable bool, no atomic (cubecl-cpu MLIR safe, plan 05-02).
        let mut acc = F::new(0.0);
        let mut cnt = 0u32;
        let mut i = 0u32;
        while i < n {
            if labels[i as usize] == c {
                acc += x[(i * d + j) as usize];
                cnt += 1u32;
            }
            i += 1u32;
        }
        sums[tid] = acc;
        // The first feature unit of each centroid records the count once.
        if j == 0u32 {
            counts[c as usize] = cnt;
        }
    }
}

/// Per-row squared distance to the ASSIGNED center, for the inertia accumulation
/// (CLUSTER-01).
///
/// One unit per ROW `i` (`ABSOLUTE_POS = i`): compute `Σ_j (X[i, j] −
/// centers[labels_i, j])²` — the squared Euclidean distance from sample `i` to
/// its assigned center. The host then sums the length-`n` `out` partials to the
/// scalar inertia `Σ_i ‖X_i − centers[labels_i]‖²` (no sqrt — squared, Pitfall
/// 8 / D-08).
///
/// - `x` is the row-major `n × d` sample matrix.
/// - `centers` is the row-major `k × d` centroid matrix.
/// - `labels` is the length-`n` assignment (`u32`).
/// - `out` is the length-`n` per-row squared distance.
/// - `n`, `d` are scalar `u32` args passed BY VALUE.
///
/// A GATHER over the assigned center's `d` features (the `reduce_sumsq`
/// squared-difference accumulation), only F/u32 accumulators + an ascending
/// scan — cubecl-cpu MLIR safe.
#[cube(launch)]
pub fn inertia_rows<F: Float + CubeElement>(
    x: &Array<F>,
    centers: &Array<F>,
    labels: &Array<u32>,
    out: &mut Array<F>,
    n: u32,
    d: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let c = labels[i];
        let xbase = (i as u32) * d;
        let cbase = c * d;
        let mut acc = F::new(0.0);
        let mut j = 0u32;
        while j < d {
            let diff = x[(xbase + j) as usize] - centers[(cbase + j) as usize];
            acc += diff * diff;
            j += 1u32;
        }
        out[i] = acc;
    }
}
