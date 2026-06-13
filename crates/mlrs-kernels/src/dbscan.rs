//! `dbscan` — eps-threshold + per-row core-count mask kernel (PRIM, CLUSTER-02).
//!
//! A `#[cube]` kernel over the `n × n` pairwise SQUARED-distance matrix `D` that,
//! per point `i`, thresholds every `D[i,j] <= eps²` into a self-inclusive
//! eps-adjacency bit and accumulates the row's eps-neighbor COUNT. The host reads
//! the count + the `n × n` adjacency back (the `prims::cholesky` tiny-readback
//! idiom, scaled to n²; D-04 documented round-trip), derives
//! `is_core[i] = count[i] >= min_samples`, and walks the adjacency with an
//! index-ordered DFS — that host graph traversal is the ESTIMATOR's job (plan 07),
//! NOT this prim (D-04: inherently sequential expansion is host-side).
//!
//! ## Layout & parallelism — GATHER per ROW, no atomics, no SharedMemory
//! One UNIT per point `i` (`ABSOLUTE_POS_X` selects the row). Unit `i` scans all
//! `n` columns of its OWN row, writing each adjacency bit `adj[i*n+j]` and
//! accumulating `count[i]` in a private `u32`. This is a GATHER: every output
//! slot (`count[i]`, the `adj[i, *]` row) is owned by exactly one unit, so there
//! is NO scatter and NO cross-unit atomic — the `cubecl-cpu` MLIR lowering does
//! not lower cross-unit atomics, and the primary correctness gate is cpu(f64)
//! (plans 05-02/05-03). The body uses ONLY `F`/`u32` accumulators and `if`-guards
//! — no `SharedMemory`, no mutable `bool`, no `F::INFINITY`, no descending shift
//! (constructs the cpu backend rejects at launch). No hardcoded plane width.
//!
//! ## Self-inclusive `<= eps²` (Pitfall 7)
//! The neighborhood is self-inclusive: `D[i,i] = 0 <= eps²` always, so point `i`
//! counts itself. The threshold is on the SQUARED distance (`<= eps` on Euclidean
//! ⇔ `<= eps²` on squared), so `prims::dbscan` feeds the un-sqrt'd distance matrix
//! and passes `eps2 = eps*eps`. The count/threshold is INTEGER-exact (no
//! tolerance) — it is a comparison + count, not a float reduction.
//!
//! All kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature (D-13). Tests live in `crates/mlrs-backend/tests/dbscan_mask_test.rs`
//! (AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::eps_core_count as dbscan_eps_core_count;

/// eps-threshold + per-row eps-neighbor count over the `n × n` row-major SQUARED
/// distance matrix `d2` (D-04): for each point `i`, write the self-inclusive
/// eps-adjacency row `adj[i, j] = (d2[i,j] <= eps2)` and the row's eps-neighbor
/// `count[i] = Σ_j adj[i,j]`.
///
/// - `d2` is the `n × n` row-major SQUARED pairwise-distance matrix (`d2[i,i]=0`).
/// - `adj` is the `n × n` adjacency bitmask (`1u32` if `d2[i,j] <= eps2`, else
///   `0u32`) — the host DFS walks it (D-04 readback).
/// - `count` is the length-`n` per-row self-inclusive eps-neighbor count; the host
///   derives `is_core[i] = count[i] >= min_samples` after readback (the core
///   decision stays host-side — the device does only the n² threshold/count).
/// - `eps2 : F` is `eps*eps`; `n : u32` is the point count. Scalar args BY VALUE
///   (cubecl 0.10 — no `ScalarArg`, mirroring `dist_combine_clamp`'s `rows: u32`).
///
/// Launched ONE unit per point (`ABSOLUTE_POS_X = i`); the kernel bounds-checks
/// `i < n` so over-provisioned threads write nothing. GATHER (no atomics, no
/// SharedMemory) — every written slot is owned by exactly one unit.
#[cube(launch)]
pub fn eps_core_count<F: Float + CubeElement>(
    d2: &Array<F>,
    adj: &mut Array<u32>,
    count: &mut Array<u32>,
    eps2: F,
    n: u32,
) {
    let i = ABSOLUTE_POS_X;
    // One unit owns row i; the idle units of an over-provisioned cube write
    // nothing (no `continue` in #[cube] — the whole body is `if`-wrapped).
    if i < n {
        let base = i * n;

        // Self-inclusive eps-neighbor GATHER over row i: write each adjacency bit
        // and accumulate the private per-row count. `count_i` is an explicitly
        // typed u32 accumulator (the cube macro needs the annotation to infer the
        // cross-loop flag type — 05-02 patterns-established). No atomics: count[i]
        // and the adj[i, *] row are owned solely by this unit.
        let mut count_i: u32 = 0u32;
        let mut j = 0u32;
        while j < n {
            // bit = (d2[i,j] <= eps2). D[i,i]=0 <= eps2 always ⇒ self counted.
            let mut bit: u32 = 0u32;
            if d2[(base + j) as usize] <= eps2 {
                bit = 1u32;
                count_i += 1u32;
            }
            adj[(base + j) as usize] = bit;
            j += 1u32;
        }
        count[i as usize] = count_i;
    }
}
