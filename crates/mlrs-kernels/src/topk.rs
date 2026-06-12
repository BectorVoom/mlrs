//! `topk` — partial select-k kernel (PRIM, D-02).
//!
//! A `#[cube]` kernel that, per query row of an `rows × cols` distance matrix,
//! selects the `k` smallest `(value, index)` pairs with a LOWEST-INDEX tie-break
//! — generalizing `reduce.rs`'s `argmin_shared` (value+index carry) from `k = 1`
//! to `k`. It writes two outputs (`out_val: &mut Array<F>` of `rows × k`,
//! `out_idx: &mut Array<u32>` of `rows × k`); the host re-uploads the `u32`
//! indices as `i32` (D-06).
//!
//! ## Layout & parallelism
//! One CUBE per query ROW (`CUBE_POS_X` selects the row). Within the cube, unit 0
//! emits the row's `k` smallest by SELECTION-BY-RANK: order candidates by the
//! PAIR `(value, index)` (A precedes B iff `A.val < B.val`, or equal value with
//! `A.idx < B.idx` — the exact lowest-index tie rule of `argmin_shared`,
//! reduce.rs:373-381); slot 0 is the global minimum pair and slot r>0 is the
//! minimum pair STRICTLY GREATER than the slot-(r-1) winner. Each slot is a full
//! `cols` scan, so the kernel is k full passes per row.
//!
//! ## Why selection-by-rank on one unit (not SharedMemory insertion)
//! `k` is small for the brute-force KNN consumers (sklearn default ≤ ~30) and the
//! lowest-index tie-break must be applied deterministically to match
//! numpy/sklearn. The single-unit rank scan makes the tie semantics unambiguous
//! and identical to a `k`-fold `argmin_shared`, at the cost of leaving the other
//! units of the cube idle — acceptable for small-`k` selection (Pitfall 8).
//! Crucially it uses ONLY `F`/`u32` accumulators and `if` guards (no mutable
//! `bool`, no `SharedMemory`, no descending-shift loop) — constructs the
//! `cubecl-cpu` MLIR lowering rejects (the cpu backend is the primary gate). No
//! hardcoded plane width.
//!
//! All kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature (D-13). Tests live in `crates/mlrs-backend/tests/topk_test.rs`
//! (AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::select_k as topk_select_k;

/// Partial select-k over a `rows × cols` row-major distance matrix (D-02): for
/// each query ROW, emit the `k` smallest values (ascending) and their column
/// indices, applying the LOWEST-INDEX tie-break on equal values.
///
/// - `dist` is the row-major `rows × cols` distance matrix (one query per row,
///   one train point per column).
/// - `out_val` is the `rows × k` ascending k-smallest values per row.
/// - `out_idx` is the `rows × k` column indices of those values.
/// - `rows`, `cols`, `k` are scalar args passed BY VALUE (cubecl 0.10 — no
///   `ScalarArg` wrapper, mirroring `dist_combine_clamp`'s `rows: u32`).
///
/// Launched ONE cube per row (`CUBE_POS_X` = row); only unit 0 of each cube does
/// the selection (small-`k` selection-by-rank — see the module docs). Each output
/// slot is the minimum candidate pair strictly greater than the previous slot's
/// winner, seeded so slot 0 admits the global minimum.
#[cube(launch)]
pub fn select_k<F: Float + CubeElement>(
    dist: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows: u32,
    cols: u32,
    k: u32,
) {
    let row = CUBE_POS_X;
    // Only unit 0 selects; guard both the row bound and the unit so the idle
    // units of an over-provisioned cube write nothing (no `continue` in #[cube]
    // — everything is `if`-wrapped).
    if row < rows {
        if UNIT_POS_X == 0u32 {
            let base = row * cols;
            let out_base = row * k;

            // SELECTION-BY-RANK (no SharedMemory, no mutable bool, no descending
            // shift — the cubecl-cpu MLIR lowering rejects those; this body uses
            // only `F`/`u32` accumulators and `if` guards, matching the proven
            // `argmin_shared` shape generalized k-fold).
            //
            // Order candidates by the PAIR (value, index): pair A precedes pair B
            // iff `A.val < B.val` OR (`A.val == B.val` AND `A.idx < B.idx`) — the
            // exact lowest-index tie rule. Indices within a row are distinct, so
            // every pair is unique and the k smallest pairs are a strict ascending
            // chain. Slot 0 is the global minimum pair; slot r>0 is the minimum
            // pair STRICTLY GREATER than the pair emitted at slot r-1.
            //
            // `prev_*` carry the last emitted pair. There is NO float-infinity
            // sentinel (cubecl-cpu's MLIR lowering rejects `F::INFINITY` inside a
            // #[cube]) and NO mutable bool flag (the cube macro fails to infer a
            // cross-loop `0u32` flag): slot 0 admits EVERY candidate via the
            // `r == 0` branch, and each rank pass SEEDS its running best from the
            // FIRST admissible candidate it encounters by initialising best from
            // candidate `c = 0`'s slot and only updating from `c = 1` onward when a
            // candidate both is admissible AND precedes the running best.
            //
            // Concretely the running best is initialised to candidate 0 (value +
            // index 0). For slot 0 that is already a valid admissible candidate.
            // For slot r>0 candidate 0 may be INADMISSIBLE (≤ the previous pair);
            // the scan below repairs that by taking the first admissible candidate
            // as the running best (tracked by comparing on the PAIR order, which is
            // total over the distinct row indices).
            // Slot 0: the global minimum pair (lowest-index tie-break).
            let mut best0_val = dist[base as usize];
            let mut best0_idx = 0u32;
            let mut c0 = 1u32;
            while c0 < cols {
                let cv = dist[(base + c0) as usize];
                if cv < best0_val {
                    best0_val = cv;
                    best0_idx = c0;
                }
                // equal value can't lower the index here: c0 ascends, so the first
                // occurrence already holds the lowest index (strict `<` keeps it).
                c0 += 1u32;
            }
            out_val[out_base as usize] = best0_val;
            out_idx[out_base as usize] = best0_idx;
            let mut prev_val = best0_val;
            let mut prev_idx = best0_idx;

            // Slots 1..k: each is the minimum pair STRICTLY GREATER than the
            // previous emitted pair (value, or equal value with higher index).
            let mut r = 1u32;
            while r < k {
                // Seed the running best from prev so the first admissible candidate
                // (guaranteed to exist since k ≤ cols and pairs are distinct)
                // overwrites it; until then no real candidate equals (prev_val,
                // prev_idx) so the seed is never emitted.
                let mut best_val = prev_val;
                let mut best_idx = prev_idx;

                let mut c = 0u32;
                while c < cols {
                    let cv = dist[(base + c) as usize];
                    let ci = c;

                    // admit = (cv, ci) is strictly GREATER than the previous pair.
                    let mut admit: u32 = 0u32;
                    if cv > prev_val {
                        admit = 1u32;
                    } else if cv == prev_val {
                        if ci > prev_idx {
                            admit = 1u32;
                        }
                    }

                    if admit == 1u32 {
                        // better = (cv, ci) precedes the running best, where the
                        // best is still the (prev) seed (best == prev) OR a real
                        // earlier-admitted candidate. `best == prev` is detected by
                        // `best_idx == prev_idx && best_val == prev_val`; since a
                        // real admissible candidate is strictly greater than prev,
                        // ANY admissible candidate precedes the prev-seed.
                        let mut better: u32 = 0u32;
                        if best_idx == prev_idx {
                            // running best is still the prev seed → admit replaces it.
                            better = 1u32;
                        } else if cv < best_val {
                            better = 1u32;
                        } else if cv == best_val {
                            if ci < best_idx {
                                better = 1u32;
                            }
                        }
                        if better == 1u32 {
                            best_val = cv;
                            best_idx = ci;
                        }
                    }

                    c += 1u32;
                }

                out_val[(out_base + r) as usize] = best_val;
                out_idx[(out_base + r) as usize] = best_idx;

                prev_val = best_val;
                prev_idx = best_idx;

                r += 1u32;
            }
        }
    }
}
