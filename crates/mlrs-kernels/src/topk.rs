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
//! maintains a `k`-length running best — values in a `SharedMemory::<F>`, indices
//! in a parallel `SharedMemory::<u32>` — kept ASCENDING by value. It linearly
//! scans the row's `cols` candidates and INSERTS each into the running top-k,
//! shifting larger entries right (insertion-select). On EQUAL value the LOWER
//! column index wins, replicating `argmin_shared`'s exact tie rule
//! (`else if ov == cv { if oi < ci {..} }`, reduce.rs:373-381). The `k` ≤ `cols`
//! bound is validated host-side BEFORE launch (prims/topk.rs), so the comptime
//! shared cap (256, the reduce.rs idiom) is never exceeded at runtime.
//!
//! ## Why insertion-select on one unit (not a parallel tree)
//! `k` is small for the brute-force KNN consumers (sklearn default ≤ ~30) and the
//! lowest-index tie-break must be applied in a strict left-to-right candidate
//! order to match numpy/sklearn deterministically. A single-unit ascending
//! insertion makes the tie semantics unambiguous and identical to a `k`-fold
//! `argmin_shared`, at the cost of leaving the other units of the cube idle —
//! acceptable for the small-`k` selection (Pitfall 8). No hardcoded plane width;
//! the only comptime size is the 256 shared cap (matches the reduce kernels).
//!
//! All kernels are generic over `<F: Float + CubeElement>` and carry NO backend
//! feature (D-13). Tests live in `crates/mlrs-backend/tests/topk_test.rs`
//! (AGENTS.md §2 — never an in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::select_k as topk_select_k;

/// Comptime cap on the running top-k buffers (matches the reduce kernels'
/// `SharedMemory::new(256)` ceiling). The host validates `k <= cols` and the
/// launch never requests more than 256, so the runtime `k` always fits.
const MAX_K: usize = 256usize;

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
/// the selection (small-`k` insertion-select — see the module docs). The running
/// buffers are seeded with `+inf` / a sentinel-high index so an unfilled slot
/// never beats a real candidate and a tie never lets the sentinel index win.
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
            let mut best_val = SharedMemory::<F>::new(MAX_K);
            let mut best_idx = SharedMemory::<u32>::new(MAX_K);

            // Seed the running top-k: +inf value, sentinel-high index so an
            // unfilled slot is never selected and never wins a tie.
            let inf = F::new(f32::INFINITY);
            let mut s = 0u32;
            while s < k {
                best_val[s as usize] = inf;
                best_idx[s as usize] = cols;
                s += 1u32;
            }

            // Scan every candidate in the row and insert it into the ascending
            // running top-k (insertion-select), preserving the lowest-index tie.
            let base = row * cols;
            let mut c = 0u32;
            while c < cols {
                let cand_val = dist[(base + c) as usize];
                let cand_idx = c;

                // Find the insertion slot: the first position whose current
                // entry the candidate should precede. STRICTLY-smaller value
                // wins; on EQUAL value the LOWER index wins (argmin_shared rule).
                // `pos == k` means the candidate is not in the top-k.
                let mut pos = k;
                let mut found = false;
                let mut p = 0u32;
                while p < k {
                    if !found {
                        let cv = best_val[p as usize];
                        let ci = best_idx[p as usize];
                        let mut precedes = false;
                        if cand_val < cv {
                            precedes = true;
                        } else if cand_val == cv {
                            if cand_idx < ci {
                                precedes = true;
                            }
                        }
                        if precedes {
                            pos = p;
                            found = true;
                        }
                    }
                    p += 1u32;
                }

                // If the candidate belongs in the top-k, shift the entries from
                // `pos..k-1` one slot right (dropping the last) and drop it in.
                if found {
                    // Shift right from the tail down to `pos + 1` (descending
                    // index walk, so no entry is overwritten before it is moved).
                    let mut q = k - 1u32;
                    while q > pos {
                        best_val[q as usize] = best_val[(q - 1u32) as usize];
                        best_idx[q as usize] = best_idx[(q - 1u32) as usize];
                        q -= 1u32;
                    }
                    best_val[pos as usize] = cand_val;
                    best_idx[pos as usize] = cand_idx;
                }

                c += 1u32;
            }

            // Emit the row's k smallest (ascending) into the row-major outputs.
            let out_base = row * k;
            let mut w = 0u32;
            while w < k {
                out_val[(out_base + w) as usize] = best_val[w as usize];
                out_idx[(out_base + w) as usize] = best_idx[w as usize];
                w += 1u32;
            }
        }
    }
}
