//! `umap_layout` — the ONE new Phase-14 device kernel: `umap_layout_step<F>`,
//! a vertex-owner GATHER SGD layout step for UMAP (UMAP-03), generic over the
//! float type (`f32`/`f64`) and the CubeCL runtime, with NO backend feature
//! (D-13). It is the named cpu-MLIR feasibility unknown of the phase (Spike flag
//! item 1).
//!
//! ## What it does (one epoch's worth of one owner's coordinate updates)
//! One OWNER vertex per cube (`row = CUBE_POS_X`, work under
//! `if row < n_owners { if UNIT_POS_X == 0u32 { … } }` — the proven `topk.rs` /
//! `self_drop_gather` launch shape, NEVER a bare-`ABSOLUTE_POS` 1D launch, which
//! mis-lowers under cpu-MLIR (FINDING 002-A)). For each owner the unit walks
//!   1. the owner's POSITIVE (attractive) edges, then
//!   2. the owner's host-drawn NEGATIVE samples (repulsive),
//! computing each per-dimension coordinate delta INSIDE the same loop iteration
//! that consumes it (no cross-sibling-loop accumulator — FINDING 002-B silent
//! miscompile) and applying it to `embedding[owner]` (and, when `move_other`,
//! to the positive neighbour too).
//!
//! ## Frozen-subset mode (D-03)
//! The kernel takes `n_owners` (the count of contiguous OWNER rows at the FRONT
//! of `embedding`) and writes only `embedding[owner]`; every non-owner vertex is
//! a read-only GATHER target. `fit` launches with `owners = all n`,
//! `move_other = 1` (two-sided). `transform` (Plan 05) launches the SAME kernel
//! with `owners = m new points placed contiguously` and `move_other = 0` so the
//! trained coordinates stay frozen.
//!
//! ## Negative sampling is HOST-drawn (D-05)
//! There is NO in-kernel / device RNG (backend-divergent, breaks byte-identical
//! reproducibility, and the device-RNG ban is a project landmine). The host
//! draws every negative-sample target index with `SplitMix64::next_below` and
//! packs them into the `neg_idx` device buffer this kernel merely GATHERs.
//!
//! ## cpu-MLIR safety (the primary correctness gate)
//! Uses ONLY `F`/`u32` accumulators + `if` guards + the STATIC `F::powf` form.
//! NO `SharedMemory`, NO `Atomic`, NO `F::INFINITY`, NO mutable-`bool` scan, NO
//! descending-shift loop, NO instance `x.powf()`, NO cross-sibling accumulator.
//! Clipping uses finite literals (±4) with statement-form `if` (the topk.rs
//! running-best idiom), never an infinity sentinel or a `max`/`min` intrinsic.
//!
//! Tests (the launch-at-cpu-MLIR smoke proof, Spike flag item 1) live in
//! `crates/mlrs-backend/tests/umap_layout_test.rs` — `mlrs-kernels` carries no
//! runtime feature, so the kernel can only be LAUNCHED from `mlrs-backend`
//! (AGENTS.md §2 — tests in a dedicated file, never an in-source `mod tests`).

use cubecl::prelude::*;

pub use self::umap_layout_step as umap_layout_kernel;

/// One vertex-owner GATHER SGD layout step (UMAP-03). Updates the coordinates of
/// every OWNER vertex from its positive (attractive) edges and its host-drawn
/// negative (repulsive) samples, in SQUARED-distance UMAP gradients.
///
/// Buffer contract (all device arrays, row-major / CSR):
/// - `embedding` — `(n_vertices, dim)` row-major coordinates, updated IN PLACE.
///   Owners are the contiguous rows `0..n_owners`; rows `n_owners..n_vertices`
///   are read-only frozen GATHER targets (D-03).
/// - `pos_offsets` — length `n_owners + 1` CSR offsets: owner `o`'s positive
///   edges are `pos_tail[pos_offsets[o] .. pos_offsets[o+1]]`.
/// - `pos_tail` — positive-edge target vertex indices (`u32`, `< n_vertices`).
/// - `neg_offsets` — length `n_owners + 1` CSR offsets into `neg_idx` for owner
///   `o`'s negative samples (host-drawn, D-05).
/// - `neg_idx` — host-drawn negative-sample target vertex indices (`u32`,
///   `< n_vertices`), GATHERed only (NO device RNG).
///
/// Scalars passed BY VALUE (cubecl 0.10 — no `ScalarArg`):
/// - `a`, `b` — the UMAP a/b curve parameters.
/// - `gamma` — repulsion strength (`repulsion_strength`).
/// - `alpha` — this epoch's learning rate (host applies the `1 − n/n_epochs`
///   decay).
/// - `dim` — embedding dimensionality (`n_components`).
/// - `n_owners` — count of contiguous OWNER rows at the front of `embedding`.
/// - `n_vertices` — total vertex count (bound for the GATHER index check).
/// - `move_other` — `1u32` = two-sided update (the owner also writes the
///   positive neighbour's coordinates); `0u32` = owner-only (each owner writes
///   only its own vertex). Both the `fit` and `transform` paths launch with
///   `0u32` (`FIT_MOVE_OTHER`) over the already-symmetric COO, so no cube writes
///   a foreign vertex's slots — the D-05 cross-cube write race cannot occur on
///   any parallel backend.
///
/// Launch: `CubeCount::Static(n_owners, 1, 1)`, `CubeDim {x:1, y:1, z:1}` (the
/// per-owner topk.rs shape).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn umap_layout_step<F: Float + CubeElement>(
    embedding: &mut Array<F>,
    pos_offsets: &Array<u32>,
    pos_tail: &Array<u32>,
    neg_offsets: &Array<u32>,
    neg_idx: &Array<u32>,
    a: F,
    b: F,
    gamma: F,
    alpha: F,
    dim: u32,
    n_owners: u32,
    n_vertices: u32,
    move_other: u32,
) {
    let row = CUBE_POS_X;
    // Per-owner, one selecting unit — the topk.rs / self_drop_gather launch shape
    // (NEVER a bare-`ABSOLUTE_POS` 1D launch — FINDING 002-A).
    if row < n_owners {
        if UNIT_POS_X == 0u32 {
            let cur_base = row * dim;

            // ============================================================
            // ATTRACTIVE: walk owner `row`'s positive edges. Each edge's
            // per-dim delta is computed AND applied WITHIN this iteration
            // (self-contained — no cross-sibling-loop accumulator, 002-B).
            // ============================================================
            let p_lo = pos_offsets[row as usize];
            let p_hi = pos_offsets[(row + 1u32) as usize];
            let mut e = p_lo;
            while e < p_hi {
                let other = pos_tail[e as usize];
                // Guard the GATHER target against the vertex bound (T-14-10).
                if other < n_vertices {
                    let other_base = other * dim;

                    // dist_squared = Σ_d (cur_d − other_d)² — a self-contained
                    // nested accumulate (read in the SAME iteration it is built).
                    let mut dist_sq = F::from_int(0i64);
                    let mut d0 = 0u32;
                    while d0 < dim {
                        let diff = embedding[(cur_base + d0) as usize]
                            - embedding[(other_base + d0) as usize];
                        dist_sq += diff * diff;
                        d0 += 1u32;
                    }

                    // Attractive scalar grad coefficient (SQUARED distance):
                    //   grad = (-2·a·b·pow(dist²,b−1)) / (a·pow(dist²,b)+1), dist²>0
                    //   grad = 0,                                            dist²≤0
                    let mut grad_coeff = F::from_int(0i64);
                    if dist_sq > F::from_int(0i64) {
                        let pow_b = F::powf(dist_sq, b);
                        let pow_bm1 = F::powf(dist_sq, b - F::new(1.0));
                        let num = F::new(-2.0) * a * b * pow_bm1;
                        let den = a * pow_b + F::new(1.0);
                        grad_coeff = num / den;
                    }

                    // Per-dim clipped update, applied IN PLACE this iteration.
                    let mut d1 = 0u32;
                    while d1 < dim {
                        let cur_d = embedding[(cur_base + d1) as usize];
                        let other_d = embedding[(other_base + d1) as usize];
                        // grad_d = clip(grad·(cur_d − other_d), −4, 4) — finite
                        // literals + statement-`if` (NO F::INFINITY / max / min).
                        let mut grad_d = grad_coeff * (cur_d - other_d);
                        if grad_d > F::new(4.0) {
                            grad_d = F::new(4.0);
                        }
                        if grad_d < F::new(-4.0) {
                            grad_d = F::new(-4.0);
                        }
                        embedding[(cur_base + d1) as usize] = cur_d + grad_d * alpha;
                        // Two-sided push for the `fit` path (frozen on transform).
                        if move_other == 1u32 {
                            embedding[(other_base + d1) as usize] = other_d - grad_d * alpha;
                        }
                        d1 += 1u32;
                    }
                }
                e += 1u32;
            }

            // ============================================================
            // REPULSIVE: walk owner `row`'s host-drawn negative samples.
            // Self-contained per-sample update (002-B safe). `other` here is
            // never moved (umap moves only the owner on the repulsive step).
            // ============================================================
            let q_lo = neg_offsets[row as usize];
            let q_hi = neg_offsets[(row + 1u32) as usize];
            let mut q = q_lo;
            while q < q_hi {
                let other = neg_idx[q as usize];
                // Skip a self-sample (k == j) and bounds-guard the GATHER.
                if other < n_vertices {
                    if other != row {
                        let other_base = other * dim;

                        let mut dist_sq = F::from_int(0i64);
                        let mut d2 = 0u32;
                        while d2 < dim {
                            let diff = embedding[(cur_base + d2) as usize]
                                - embedding[(other_base + d2) as usize];
                            dist_sq += diff * diff;
                            d2 += 1u32;
                        }

                        // Repulsive scalar grad coefficient (SQUARED distance):
                        //   grad = (2·gamma·b) / ((0.001+dist²)·(a·pow(dist²,b)+1)), dist²>0
                        // At dist²==0 umap uses the fixed per-dim grad_d = 4.0
                        // (the `dist_sq <= 0` branch below), avoiding 0/0.
                        let mut d3 = 0u32;
                        if dist_sq > F::from_int(0i64) {
                            let pow_b = F::powf(dist_sq, b);
                            let den = (F::new(0.001) + dist_sq) * (a * pow_b + F::new(1.0));
                            let grad_coeff = (F::new(2.0) * gamma * b) / den;
                            while d3 < dim {
                                let cur_d = embedding[(cur_base + d3) as usize];
                                let other_d = embedding[(other_base + d3) as usize];
                                let mut grad_d = grad_coeff * (cur_d - other_d);
                                if grad_d > F::new(4.0) {
                                    grad_d = F::new(4.0);
                                }
                                if grad_d < F::new(-4.0) {
                                    grad_d = F::new(-4.0);
                                }
                                embedding[(cur_base + d3) as usize] = cur_d + grad_d * alpha;
                                d3 += 1u32;
                            }
                        } else {
                            // dist²==0 coincident points: fixed push grad_d = 4.0.
                            while d3 < dim {
                                let cur_d = embedding[(cur_base + d3) as usize];
                                embedding[(cur_base + d3) as usize] =
                                    cur_d + F::new(4.0) * alpha;
                                d3 += 1u32;
                            }
                        }
                    }
                }
                q += 1u32;
            }
        }
    }
}
