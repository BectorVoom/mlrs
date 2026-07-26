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
pub use self::select_k_shared as topk_select_k_shared;

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

/// Block-parallel select-k (KNN-01 perf lever) — BITWISE-IDENTICAL output to
/// [`select_k`], but the per-rank `cols` scan is spread across every unit of the
/// cube instead of running serially on unit 0.
///
/// ## Why (the measured pathology)
/// [`select_k`] launches ONE cube per query row with a ONE-unit cube, so a
/// `n_query × n_train` selection is `n_query` serial threads each performing `k`
/// full `n_train` scans. On a wgpu/Vulkan probe at `n_train = 10_000`,
/// `n_query = 2_000`, `k = 5` that stage cost **3.82 s** against **0.09 s** for
/// the whole `distance` GEMM-expansion that feeds it — 41× the cost of the
/// operation it post-processes — and at `n_train = 50_000` it ran long enough to
/// trip the GPU's watchdog and lose the device context. The selection, not the
/// distance matrix, is the KNN bottleneck.
///
/// ## The transform
/// Same SELECTION-BY-RANK algorithm (slot 0 = the global minimum pair, slot
/// `r > 0` = the minimum pair STRICTLY GREATER than slot `r-1`, ordering pairs by
/// `(value, index)` with the lowest-index tie-break), but each rank pass is:
///
/// 1. a STRIDED local scan — unit `t` visits columns `t, t + CUBE_DIM_X, …`,
///    keeping its own best admissible pair;
/// 2. a `log₂` SharedMemory tree reduce over those per-unit bests under the same
///    pair order (the `reduce.rs::argmin_shared` idiom, generalized to carry a
///    validity flag);
/// 3. a broadcast of the winning pair out of shared slot 0 into every unit's
///    `prev_*`, so the next rank's admission test is uniform across the cube.
///
/// Per-rank serial work drops from `cols` to `cols / CUBE_DIM_X + log₂
/// CUBE_DIM_X`. Because the reduction is over a TOTAL order on distinct pairs
/// (indices within a row are unique), the tree's association order cannot change
/// the winner — the emitted `(value, index)` sequence is identical to the serial
/// kernel's, so the sklearn-matching tie-break is preserved exactly rather than
/// approximated.
///
/// ## Validity flag instead of a sentinel
/// A unit whose strided slice holds NO admissible candidate must not contribute.
/// There is no float-infinity sentinel available (`F::INFINITY` inside a `#[cube]`
/// is rejected by the cubecl-cpu MLIR lowering, see [`select_k`]), and a mutable
/// `bool` is likewise unusable, so validity travels as a `u32` 0/1 flag in a third
/// SharedMemory array — the same `admit: u32` idiom the serial kernel already
/// uses. Only `F`/`u32` accumulators and `if` guards appear in the body.
///
/// `sync_cube` calls sit OUTSIDE any non-uniform branch: `row` is `CUBE_POS_X`,
/// which is uniform across a cube, and `k` is a scalar, so every unit executes the
/// same number of barriers.
///
/// Launched ONE cube per row with a POWER-OF-TWO `CUBE_DIM_X <= 256` (the shared
/// arrays are sized 256, matching `reduce.rs`); the host picks the width.
#[cube(launch)]
pub fn select_k_shared<F: Float + CubeElement>(
    dist: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows: u32,
    cols: u32,
    k: u32,
) {
    let mut sval = SharedMemory::<F>::new(256usize);
    let mut sidx = SharedMemory::<u32>::new(256usize);
    let mut shas = SharedMemory::<u32>::new(256usize);

    let tid = UNIT_POS_X;
    let row = CUBE_POS_X;

    // `row < rows` is uniform over the cube (every unit shares CUBE_POS_X), so the
    // barriers below are reached by all units or none.
    if row < rows {
        let base = row * cols;
        let out_base = row * k;

        // The previously emitted pair, replicated in EVERY unit so the admission
        // test needs no per-iteration broadcast beyond the shared slot-0 read.
        // Seeded from column 0 (always in range: `cols >= k >= 1`); the `r == 0`
        // branch admits unconditionally, so the seed's value is never consulted on
        // the first rank.
        let mut prev_val = dist[base as usize];
        let mut prev_idx = 0u32;

        let mut r = 0u32;
        while r < k {
            // --- 1. Strided local scan: unit `tid` owns columns tid, tid+D, … ---
            let mut has: u32 = 0u32;
            let mut bval = dist[base as usize];
            let mut bidx = 0u32;

            let mut c = tid;
            while c < cols {
                let cv = dist[(base + c) as usize];

                // admit = (cv, c) is strictly GREATER than the previous emitted
                // pair. Rank 0 has no predecessor, so every column is admissible.
                let mut admit: u32 = 0u32;
                if r == 0u32 {
                    admit = 1u32;
                } else if cv > prev_val {
                    admit = 1u32;
                } else if cv == prev_val {
                    if c > prev_idx {
                        admit = 1u32;
                    }
                }

                if admit == 1u32 {
                    // better = (cv, c) precedes this unit's running best, where an
                    // as-yet-unset best (has == 0) is preceded by anything.
                    let mut better: u32 = 0u32;
                    if has == 0u32 {
                        better = 1u32;
                    } else if cv < bval {
                        better = 1u32;
                    } else if cv == bval {
                        if c < bidx {
                            better = 1u32;
                        }
                    }
                    if better == 1u32 {
                        bval = cv;
                        bidx = c;
                        has = 1u32;
                    }
                }

                c += CUBE_DIM_X;
            }

            sval[tid as usize] = bval;
            sidx[tid as usize] = bidx;
            shas[tid as usize] = has;
            sync_cube();

            // --- 2. log₂ tree reduce under the SAME pair order (argmin_shared
            //        shape, plus the validity flag). ---
            let mut s = CUBE_DIM_X / 2u32;
            while s > 0u32 {
                if tid < s {
                    let oh = shas[(tid + s) as usize];
                    // An invalid partner can never win, so it is skipped whole.
                    if oh == 1u32 {
                        let ov = sval[(tid + s) as usize];
                        let oi = sidx[(tid + s) as usize];
                        let ch = shas[tid as usize];
                        let cv = sval[tid as usize];
                        let ci = sidx[tid as usize];

                        let mut better: u32 = 0u32;
                        if ch == 0u32 {
                            better = 1u32;
                        } else if ov < cv {
                            better = 1u32;
                        } else if ov == cv {
                            if oi < ci {
                                better = 1u32;
                            }
                        }
                        if better == 1u32 {
                            sval[tid as usize] = ov;
                            sidx[tid as usize] = oi;
                            shas[tid as usize] = 1u32;
                        }
                    }
                }
                sync_cube();
                s /= 2u32;
            }

            // --- 3. Broadcast the rank winner to every unit (slot 0 is settled;
            //        the loop above ends on a barrier). `k <= cols` and pairs are
            //        distinct, so an admissible candidate always exists and
            //        `shas[0] == 1` here. ---
            let wval = sval[0usize];
            let widx = sidx[0usize];
            if tid == 0u32 {
                out_val[(out_base + r) as usize] = wval;
                out_idx[(out_base + r) as usize] = widx;
            }
            prev_val = wval;
            prev_idx = widx;

            // Every unit has now READ slot 0; barrier before the next rank's
            // scan overwrites the shared arrays.
            sync_cube();

            r += 1u32;
        }
    }
}

/// SINGLE-PASS select-k (KNN-01 fusion campaign) — BITWISE-IDENTICAL output to
/// [`select_k`] / [`select_k_shared`], but the distance matrix is read from
/// global memory exactly ONCE instead of `k` times.
///
/// ## Why (the traffic model)
/// [`select_k_shared`] parallelized each rank's scan but kept the
/// SELECTION-BY-RANK structure: `k` full passes over the row. Its global traffic
/// is therefore `k × rows × cols × 4` bytes — at `k = 10` that is 10× the matrix
/// the distance kernel wrote, and after the distance kernel was tiled and
/// register-blocked (KNN-01) this redundant re-reading became the dominant term
/// of KNN predict (the selection reads 40 B per matrix element the distance
/// stage produced for 8 B of combined read+write).
///
/// ## The transform
/// One cube per row, `CUBE_DIM_X` units, one strided pass:
///
/// 1. **Local scan** — unit `t` visits columns `t, t + CUBE_DIM_X, …` ONCE,
///    maintaining its own k-smallest pairs as an ASCENDING-sorted list in a
///    fixed-capacity local array (`ONEPASS_K_CAP` slots). Admission is a
///    register compare against the unit's current worst kept pair, so the local
///    arrays are touched only on the (rare) admissions; an admitted pair is
///    placed by an ascending carry-swap insertion, which pushes the old worst
///    off the end when the list is full.
/// 2. **k-round head merge** — the global k smallest are a subset of the union
///    of the per-unit lists (any globally kept pair is also kept by its owning
///    unit). Each round, every unit proposes its smallest UNCONSUMED pair
///    (`p` is its cursor); the proposals are folded through the SAME
///    SharedMemory pair-order tree as [`select_k_shared`] (validity flag
///    included), the slot-0 winner is emitted, and the unique owning unit
///    (pairs are distinct — column indices differ) advances its cursor.
///
/// Rank `r`'s winner is the exact r-th smallest pair under the total
/// `(value, index)` order, so the emitted sequence is identical to the serial
/// kernel's — the sklearn lowest-index tie-break is preserved exactly.
///
/// ## Cap
/// The local lists are comptime-sized at [`ONEPASS_K_CAP`] slots; the host
/// dispatch falls back to [`select_k_shared`] for `k > ONEPASS_K_CAP` (KNN
/// consumers use small k — sklearn default 5). `k <= cols` is validated
/// host-side, and the launch width never exceeds `cols`, so every unit owns at
/// least one column.
///
/// cpu-MLIR contract: `SharedMemory` + `sync_cube`, only `F`/`u32` accumulators
/// and STATEMENT-form `if` guards (no mutable `bool`, no float-infinity
/// sentinel — validity travels as a `u32` flag; empty-list heads propose with
/// `has = 0` and are skipped whole by the tree). Every `sync_cube` is outside
/// any non-uniform branch: `row < rows` is uniform over the cube and both the
/// merge loop (`k`) and the tree loop (`CUBE_DIM_X`) are scalar-driven.
#[cube(launch)]
pub fn select_k_onepass<F: Float + CubeElement>(
    dist: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows: u32,
    cols: u32,
    k: u32,
) {
    let mut sval = SharedMemory::<F>::new(256usize);
    let mut sidx = SharedMemory::<u32>::new(256usize);
    let mut shas = SharedMemory::<u32>::new(256usize);

    // Per-unit sorted k-list, comptime-capacity local arrays (ONEPASS_K_CAP).
    // Only the first `cnt <= k` slots are ever meaningful.
    let mut lval = Array::<F>::new(32usize);
    let mut lidx = Array::<u32>::new(32usize);

    let tid = UNIT_POS_X;
    let row = CUBE_POS_X;

    // Uniform over the cube (CUBE_POS_X), so the barriers below are reached by
    // all units or none.
    if row < rows {
        let base = row * cols;
        let out_base = row * k;

        // Register mirror of the list's current WORST kept pair, so the per
        // -element admission test never touches the local array. Seeded from
        // column 0 (always in range); meaningless until `cnt == k`, and the
        // `cnt < k` branch admits without consulting it.
        let mut worst_val = dist[base as usize];
        let mut worst_idx = 0u32;
        let mut cnt = 0u32;

        // --- 1. ONE strided pass: unit `tid` owns columns tid, tid+D, … ---
        let mut c = tid;
        while c < cols {
            let cv = dist[(base + c) as usize];

            // admit = list not yet full, or (cv, c) strictly precedes the
            // current worst kept pair under the (value, index) order.
            let mut admit: u32 = 0u32;
            if cnt < k {
                admit = 1u32;
            } else if cv < worst_val {
                admit = 1u32;
            } else if cv == worst_val {
                if c < worst_idx {
                    admit = 1u32;
                }
            }

            if admit == 1u32 {
                // Ascending carry-swap insertion: the carry settles at its
                // sorted position and everything after it shifts up one; with a
                // full list the old worst falls off the end.
                let mut cav = cv;
                let mut cai = c;
                let mut j = 0u32;
                while j < cnt {
                    let jv = lval[j as usize];
                    let ji = lidx[j as usize];
                    let mut swap: u32 = 0u32;
                    if cav < jv {
                        swap = 1u32;
                    } else if cav == jv {
                        if cai < ji {
                            swap = 1u32;
                        }
                    }
                    if swap == 1u32 {
                        lval[j as usize] = cav;
                        lidx[j as usize] = cai;
                        cav = jv;
                        cai = ji;
                    }
                    j += 1u32;
                }
                if cnt < k {
                    lval[cnt as usize] = cav;
                    lidx[cnt as usize] = cai;
                    cnt += 1u32;
                }
                worst_val = lval[(cnt - 1u32) as usize];
                worst_idx = lidx[(cnt - 1u32) as usize];
            }

            c += CUBE_DIM_X;
        }

        // --- 2. k-round head merge through the shared pair-order tree. ---
        let mut p = 0u32;
        let mut r = 0u32;
        while r < k {
            // Propose this unit's smallest unconsumed pair (validity-flagged;
            // the value behind `has == 0` is never consulted by the tree).
            let mut hv = worst_val;
            let mut hi = 0u32;
            let mut hh: u32 = 0u32;
            if p < cnt {
                hv = lval[p as usize];
                hi = lidx[p as usize];
                hh = 1u32;
            }
            sval[tid as usize] = hv;
            sidx[tid as usize] = hi;
            shas[tid as usize] = hh;
            sync_cube();

            // log₂ tree reduce under the pair order — the argmin_shared idiom
            // with the validity flag, identical to select_k_shared's step 2.
            let mut s = CUBE_DIM_X / 2u32;
            while s > 0u32 {
                if tid < s {
                    let oh = shas[(tid + s) as usize];
                    if oh == 1u32 {
                        let ov = sval[(tid + s) as usize];
                        let oi = sidx[(tid + s) as usize];
                        let ch = shas[tid as usize];
                        let cv = sval[tid as usize];
                        let ci = sidx[tid as usize];

                        let mut better: u32 = 0u32;
                        if ch == 0u32 {
                            better = 1u32;
                        } else if ov < cv {
                            better = 1u32;
                        } else if ov == cv {
                            if oi < ci {
                                better = 1u32;
                            }
                        }
                        if better == 1u32 {
                            sval[tid as usize] = ov;
                            sidx[tid as usize] = oi;
                            shas[tid as usize] = 1u32;
                        }
                    }
                }
                sync_cube();
                s /= 2u32;
            }

            // Slot 0 holds the round winner: the exact r-th smallest pair
            // (`k <= cols` guarantees an admissible proposal every round, so
            // `shas[0] == 1` here). Emit it, and let the unique owning unit
            // advance past it.
            let wval = sval[0usize];
            let widx = sidx[0usize];
            if tid == 0u32 {
                out_val[(out_base + r) as usize] = wval;
                out_idx[(out_base + r) as usize] = widx;
            }
            if p < cnt {
                if lval[p as usize] == wval {
                    if lidx[p as usize] == widx {
                        p += 1u32;
                    }
                }
            }

            // Every unit has READ slot 0; barrier before the next round's
            // proposals overwrite the shared arrays.
            sync_cube();

            r += 1u32;
        }
    }
}

/// [`select_k_onepass`] over a CANDIDATE matrix — the strip-merge companion of
/// the fused KNN kernel (KNN-02).
///
/// `cand_val`/`cand_idx` are `rows × cols` row-major candidate pairs: the
/// concatenated per-strip top-k lists the fused kernel emitted (strip `s` at
/// columns `[s*k, s*k + k)`). This kernel selects the k smallest by
/// `(value, POSITION)` pair order and emits `cand_idx[..]` — the GLOBAL train
/// index the candidate carries — instead of the position.
///
/// ## Why the position tie-break is exact
/// The final result must tie-break equal distances by lowest GLOBAL train
/// index (the sklearn rule). Candidate position order coincides with global
/// index order for equal values: within one strip the list is ascending in
/// `(value, index)` (so equal values sit in ascending index order), and
/// strips cover CONTIGUOUS ascending index ranges (every index in strip `s`
/// is smaller than every index in strip `s+1`). Selecting by position and
/// emitting the mapped index therefore reproduces the unstripped kernel's
/// output exactly, bit for bit.
///
/// Same structure, shared budget, and cpu-MLIR contract as
/// [`select_k_onepass`]; `cols` here is `strips * k` (a few dozen), so the
/// scan is trivially short.
#[cube(launch)]
pub fn select_k_onepass_indexed<F: Float + CubeElement>(
    cand_val: &Array<F>,
    cand_idx: &Array<u32>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows: u32,
    cols: u32,
    k: u32,
) {
    let mut sval = SharedMemory::<F>::new(256usize);
    let mut sidx = SharedMemory::<u32>::new(256usize);
    let mut shas = SharedMemory::<u32>::new(256usize);

    let mut lval = Array::<F>::new(32usize);
    let mut lpos = Array::<u32>::new(32usize);

    let tid = UNIT_POS_X;
    let row = CUBE_POS_X;

    if row < rows {
        let base = row * cols;
        let out_base = row * k;

        let mut worst_val = cand_val[base as usize];
        let mut worst_pos = 0u32;
        let mut cnt = 0u32;

        let mut c = tid;
        while c < cols {
            let cv = cand_val[(base + c) as usize];

            let mut admit: u32 = 0u32;
            if cnt < k {
                admit = 1u32;
            } else if cv < worst_val {
                admit = 1u32;
            } else if cv == worst_val {
                if c < worst_pos {
                    admit = 1u32;
                }
            }

            if admit == 1u32 {
                let mut cav = cv;
                let mut cap = c;
                let mut j = 0u32;
                while j < cnt {
                    let jv = lval[j as usize];
                    let jp = lpos[j as usize];
                    let mut swap: u32 = 0u32;
                    if cav < jv {
                        swap = 1u32;
                    } else if cav == jv {
                        if cap < jp {
                            swap = 1u32;
                        }
                    }
                    if swap == 1u32 {
                        lval[j as usize] = cav;
                        lpos[j as usize] = cap;
                        cav = jv;
                        cap = jp;
                    }
                    j += 1u32;
                }
                if cnt < k {
                    lval[cnt as usize] = cav;
                    lpos[cnt as usize] = cap;
                    cnt += 1u32;
                }
                worst_val = lval[(cnt - 1u32) as usize];
                worst_pos = lpos[(cnt - 1u32) as usize];
            }

            c += CUBE_DIM_X;
        }

        let mut p = 0u32;
        let mut r = 0u32;
        while r < k {
            let mut hv = worst_val;
            let mut hi = 0u32;
            let mut hh: u32 = 0u32;
            if p < cnt {
                hv = lval[p as usize];
                hi = lpos[p as usize];
                hh = 1u32;
            }
            sval[tid as usize] = hv;
            sidx[tid as usize] = hi;
            shas[tid as usize] = hh;
            sync_cube();

            let mut s = CUBE_DIM_X / 2u32;
            while s > 0u32 {
                if tid < s {
                    let oh = shas[(tid + s) as usize];
                    if oh == 1u32 {
                        let ov = sval[(tid + s) as usize];
                        let oi = sidx[(tid + s) as usize];
                        let ch = shas[tid as usize];
                        let cv = sval[tid as usize];
                        let ci = sidx[tid as usize];

                        let mut better: u32 = 0u32;
                        if ch == 0u32 {
                            better = 1u32;
                        } else if ov < cv {
                            better = 1u32;
                        } else if ov == cv {
                            if oi < ci {
                                better = 1u32;
                            }
                        }
                        if better == 1u32 {
                            sval[tid as usize] = ov;
                            sidx[tid as usize] = oi;
                            shas[tid as usize] = 1u32;
                        }
                    }
                }
                sync_cube();
                s /= 2u32;
            }

            // The winner's POSITION is mapped through `cand_idx` at emission —
            // the only difference from `select_k_onepass`.
            let wval = sval[0usize];
            let wpos = sidx[0usize];
            if tid == 0u32 {
                out_val[(out_base + r) as usize] = wval;
                out_idx[(out_base + r) as usize] = cand_idx[(base + wpos) as usize];
            }
            if p < cnt {
                if lval[p as usize] == wval {
                    if lpos[p as usize] == wpos {
                        p += 1u32;
                    }
                }
            }

            sync_cube();

            r += 1u32;
        }
    }
}
