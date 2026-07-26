//! `knn` — fused neighbor-target GATHER kernels for the brute-force KNN
//! estimators (KNN-01 perf lever).
//!
//! ## Why (the round-trip this replaces)
//! `KNeighborsRegressor::predict` used to finish the device pipeline
//! (`distance` → `top_k`) and then leave the device to form the answer: the
//! `n_query × k` `u32` neighbor indices were read back to the host, the fitted
//! targets were kept as a HOST `Vec<F>`, a host `n_query × k` double loop summed
//! them, and the resulting predictions were uploaded again with
//! `DeviceArray::from_host`. That is three boundary crossings (index read-back,
//! prediction upload, plus the fit-time target read-back) around an operation
//! that is a pure `k`-element gather — the same `gemm → to_host → host loop →
//! from_host` pathology already removed from the dense linear-model predict path
//! by `linear_predict.rs`.
//!
//! [`knn_regress_mean`] performs that gather ON the device, so `predict` never
//! reads the indices back and the fitted targets stay device-resident: the whole
//! KNN predict pipeline becomes device-resident end to end (D-05) with a single
//! terminal read-back of the `n_query` predictions at the API boundary.
//!
//! GATHER-only — no `SharedMemory`, no mutable `bool`, no plane ops — so the
//! `cubecl-cpu` MLIR lowering accepts it and NO backend feature gate is needed
//! (D-13); it is the `mutual_reachability.rs` / `linear_predict.rs` shape.
//!
//! ## Two kernel families live here (KNN-02/03 vs KNN-04)
//! - `euclidean_topk_fused{,_dot,_dot_vec,_dot_vec8}` are the GPU family:
//!   `SharedMemory` staging tiles, 256-unit cubes, `sync_cube` between phases.
//! - `euclidean_topk_rows_cpu{,_vec}` are the **cpu** family: no shared memory,
//!   no barriers, one unit per block of query rows. They exist because
//!   `cubecl-cpu` maps one OS THREAD per unit and implements `sync_cube` as a
//!   spin barrier, which makes the GPU family's shape pathological there rather
//!   than merely slower — see `euclidean_topk_rows_cpu`'s docs for the
//!   measurement and `euclidean_topk_rows_cpu_vec`'s for why the cpu arm is
//!   bound by IR-op count (the cpu JIT runs at LLVM `-O0`).
//!
//! Both families produce the same top-k under the same `(value, index)` order;
//! `crates/mlrs-backend/tests/knn_test.rs` gates that.
//!
//! Tests live in `crates/mlrs-backend/tests/knn_test.rs` (AGENTS.md §2 — never an
//! in-source `#[cfg(test)] mod tests`).

use cubecl::prelude::*;

pub use self::knn_regress_mean as knn_regress_mean_kernel;

/// Fused neighbor-target gather + uniform mean: one output per QUERY row.
///
/// `out[q] = (Σ_{j<k} y_train[idx[q*k + j]]) / k` — the `weights='uniform'`
/// neighbor mean that defines `KNeighborsRegressor.predict`.
///
/// - `y_train` is the fitted `n_train` regression targets (device-resident).
/// - `idx` is the `n_query × k` row-major neighbor index buffer produced by
///   `topk::select_k_shared` (column indices into the training set).
/// - `out` is the `n_query` predictions.
/// - `n_query`, `k`, `n_train` are scalar args passed BY VALUE (cubecl 0.10 — no
///   `ScalarArg` wrapper, mirroring `select_k`'s `rows: u32`).
///
/// ## Bounds guard (WR-02 parity)
/// The host loop this replaces returned a typed `ShapeMismatch` when a neighbor
/// index fell outside the target vector. A kernel cannot return an error, so the
/// same defence is expressed as an `if t < n_train` guard: a corrupted or
/// oversized index CONTRIBUTES NOTHING instead of performing an out-of-bounds
/// device read. The producing `select_k_shared` only ever emits column indices in
/// `[0, n_train)`, so the guard is unreachable in practice — it exists so that a
/// future regression in the selection kernel degrades to a wrong number rather
/// than to undefined behaviour.
///
/// ## Compensated (Neumaier) accumulation — why the sum is not naive
/// The host loop this kernel replaced accumulated the `k` neighbour targets in
/// `f64` and rounded once, so the f32 estimator's mean carried ~1 ulp of error
/// regardless of `k`. A device kernel cannot widen to `f64` — half the supported
/// backends have no f64 at all — so the accuracy is recovered instead with a
/// NEUMAIER compensated sum: the low-order bits lost by each `acc + y` are
/// tracked in `comp` and added back once at the end. That bounds the error at
/// ~1 ulp of the result independent of `k`, where a naive f32 running sum grows
/// as `k * eps` and additionally suffers cancellation on mixed-sign targets —
/// at large `n_neighbors` (30-100) with large-magnitude targets that can exceed
/// the project's 1e-5 sklearn-agreement bar on its own.
///
/// The branchless `if` form (`if |acc| >= |y| { .. } else { .. }`) is the
/// cubecl-cpu-MLIR-safe spelling of Neumaier's magnitude test — no `abs`
/// intrinsic on a mutable, no early exit.
///
/// Launched 1D over `n_query` (one unit per query row); the kernel bounds-checks
/// `q < n_query`, so an over-provisioned launch writes nothing.
#[cube(launch)]
pub fn knn_regress_mean<F: Float + CubeElement>(
    y_train: &Array<F>,
    idx: &Array<u32>,
    out: &mut Array<F>,
    n_query: u32,
    k: u32,
    n_train: u32,
) {
    let q = ABSOLUTE_POS_X;
    if q < n_query {
        let base = q * k;
        let zero = F::new(0.0_f32);
        let mut acc = zero;
        let mut comp = zero;
        let mut j = 0u32;
        while j < k {
            let t = idx[(base + j) as usize];
            if t < n_train {
                let v = y_train[t as usize];
                let sum = acc + v;
                // Neumaier: the lost low bits live in the LARGER operand when the
                // magnitudes differ, so pick the branch by magnitude.
                let mut a_mag = acc;
                if a_mag < zero {
                    a_mag = zero - a_mag;
                }
                let mut v_mag = v;
                if v_mag < zero {
                    v_mag = zero - v_mag;
                }
                if a_mag >= v_mag {
                    comp += (acc - sum) + v;
                } else {
                    comp += (v - sum) + acc;
                }
                acc = sum;
            }
            j += 1u32;
        }
        // `k >= 1` is validated host-side before launch, so the divide is safe.
        out[q as usize] = (acc + comp) / F::cast_from(k);
    }
}

/// FUSED squared-Euclidean distance + top-k selection (KNN-02) — the distance
/// matrix is NEVER materialized.
///
/// ## Why (the traffic model this removes)
/// After the KNN-01 campaign the two-kernel pipeline (`euclidean_sq_dist_rb4` →
/// `select_k_onepass`) is bounded by the `n_query × n_train` intermediate: the
/// distance kernel writes it once (4 B/element) and the selection kernel reads
/// it once (4 B/element), while the feature reads the tiled distance kernel
/// actually needs are only `d/8` B/element. At the benchmark shapes
/// (`d = 16..32`) the intermediate is therefore ~2/3 of all global traffic.
/// This kernel keeps each 64×64 distance block in the registers that computed
/// it, folds the block STRAIGHT into per-row top-k lists, and emits only the
/// `n_query × k` result — global traffic falls to the feature reads plus a
/// negligible output.
///
/// ## Structure
/// One cube per 64 QUERY rows (`CUBE_POS_Y`); the cube walks ALL `n_train`
/// columns in 64-wide blocks. Per block:
///
/// 1. **Compute** — byte-for-byte the [`euclidean_sq_dist_rb4`] tile loop
///    (same staging, same 4×4 register blocking, same ascending-`k`
///    accumulation), so every squared distance is BITWISE identical to the
///    two-kernel path's.
/// 2. **Fold** — each unit merges its 16 accumulators into 4 sorted top-k
///    lists in local arrays, one per query row it covers (`a*16 + uy`). List
///    stride is the comptime [`FUSED_K_CAP`] (16); admission is a register
///    compare against the list's current worst kept pair, so the fold touches
///    memory only on the (rare) admissions, and it uses NO shared memory and
///    NO barriers beyond the compute phase's own.
///
/// After the column walk, each query row's exact top-k is the k smallest of
/// the union of the 16 per-unit lists covering it (any globally kept pair is
/// kept by its owning unit). A k-round head merge emits them in ascending
/// `(value, index)` order: every unit posts its smallest unconsumed pair into
/// a per-row group of 16 shared slots, a log₂ tree reduces each row group
/// under the pair order (validity-flagged — the argmin_shared idiom), unit
/// `ux == 0` of the group emits the winner, and the unique owning unit
/// advances its cursor. The emitted sequence equals `select_k`'s exactly (the
/// pair order is total — column indices are distinct), so the sklearn
/// lowest-index tie-break is preserved bitwise.
///
/// ## Shared budget (the dispatch gate)
/// `xs + ys` (2 × 2048 `F`) and NOTHING else — the k-round merge reuses the
/// staging tiles (dead after the column walk) as its scratch, F-encoding the
/// candidate index (exact below 2^24 at f32; the host gate caps f32 `n_train`
/// accordingly) and the validity flag. 16 KiB at f32: exactly a default wgpu
/// adapter's workgroup limit, and 4 resident cubes per 64 KiB CUDA SM (a 19 KiB
/// budget allowed only 3, which cost a third of the T4's occupancy). The HOST
/// dispatch checks `capability::active_max_shared_memory()` and falls back to
/// the two-kernel path otherwise (a capability gate, not a perf extrapolation).
///
/// `k <= FUSED_K_CAP` and `k <= n_train` are validated host-side.
///
/// cpu-MLIR contract: `SharedMemory` + `sync_cube`, `F`/`u32` accumulators and
/// STATEMENT-form `if` guards only (no mutable `bool`, no float-infinity
/// sentinel — validity travels as a `u32` flag). Every `sync_cube` sits
/// outside any non-uniform branch: the block walk (`rows_y`), the feature loop
/// (`cols`), the merge rounds (`k`) and the tree strides are all scalar-driven.
#[cube(launch)]
pub fn euclidean_topk_fused<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    k: u32,
    strip_len: u32,
    strips: u32,
) {
    // [k][row] staging tiles, exactly euclidean_sq_dist_rb4's.
    let mut xs = SharedMemory::<F>::new(2048usize);
    let mut ys = SharedMemory::<F>::new(2048usize);
    // NO separate merge scratch: after the column walk the staging tiles are
    // dead, so the k-round merge reuses `xs` (values) and `ys` (index in
    // slots 0..256, validity flag in slots 256..512), both F-encoded. The
    // index cast is exact — `u32 -> F -> u32` is lossless for idx < 2^24 in
    // f32 (the host gate caps f32 `n_train` accordingly) and for all u32 in
    // f64. This keeps the cube's shared budget at exactly `4096 * size_of::<F>()`
    // (16 KiB at f32): 4 cubes per 64 KiB SM instead of 3, and within a
    // default wgpu adapter's 16 KiB workgroup limit.

    // Per-unit running top-k lists: list `a` (query row `a*16 + uy`) occupies
    // slots [a*16, a*16 + lcnt[a]) — FUSED_K_CAP = 16 comptime stride. The
    // worst kept pair of each list is mirrored in lworst_* so the per-element
    // admission test stays out of the arrays.
    let mut lval = Array::<F>::new(64usize);
    let mut lidx = Array::<u32>::new(64usize);
    let mut lcnt = Array::<u32>::new(4usize);
    let mut lworst_val = Array::<F>::new(4usize);
    let mut lworst_idx = Array::<u32>::new(4usize);
    // Per-block register dump so the fold can index the 16 accumulators.
    let mut blk = Array::<F>::new(16usize);

    let ux = UNIT_POS_X;
    let uy = UNIT_POS_Y;
    let base_i = CUBE_POS_Y * 64u32;

    let load_k = ux + 16u32 * (uy % 2u32);
    let load_r_base = uy / 2u32;

    let zero_f = F::from_int(0i64);
    let one_f = F::from_int(1i64);
    // Unrolled init: a `while` over these writes breaks the cube macro's type
    // inference for the loop counter (E0282), so the 4 slots are spelled out.
    lcnt[0usize] = 0u32;
    lcnt[1usize] = 0u32;
    lcnt[2usize] = 0u32;
    lcnt[3usize] = 0u32;
    lworst_val[0usize] = zero_f;
    lworst_val[1usize] = zero_f;
    lworst_val[2usize] = zero_f;
    lworst_val[3usize] = zero_f;
    lworst_idx[0usize] = 0u32;
    lworst_idx[1usize] = 0u32;
    lworst_idx[2usize] = 0u32;
    lworst_idx[3usize] = 0u32;

    // ---- walk THIS STRIP's train columns in 64-wide blocks ----
    // The train dimension is split over `CUBE_POS_X` strips (KNN-02 third
    // pass): a single strip per query block leaves the launch a few dozen
    // cubes wide at moderate `n_query`, which quantizes onto the SM count in
    // 1-2 ragged waves; strips multiply the cube count so the tail wave is
    // amortized. The host guarantees every strip holds >= 64 columns (so a
    // strip's candidate union always covers k <= 16), and rounds `strip_len`
    // to the 64-wide block size.
    let strip = CUBE_POS_X;
    let c_begin = strip * strip_len;
    let mut c_end = c_begin + strip_len;
    if c_end > rows_y {
        c_end = rows_y;
    }
    let mut b0 = c_begin;
    while b0 < c_end {
        let base_j = b0;

        // === 1. compute: the euclidean_sq_dist_rb4 tile loop, verbatim ===
        let mut acc00 = F::from_int(0i64);
        let mut acc01 = F::from_int(0i64);
        let mut acc02 = F::from_int(0i64);
        let mut acc03 = F::from_int(0i64);
        let mut acc10 = F::from_int(0i64);
        let mut acc11 = F::from_int(0i64);
        let mut acc12 = F::from_int(0i64);
        let mut acc13 = F::from_int(0i64);
        let mut acc20 = F::from_int(0i64);
        let mut acc21 = F::from_int(0i64);
        let mut acc22 = F::from_int(0i64);
        let mut acc23 = F::from_int(0i64);
        let mut acc30 = F::from_int(0i64);
        let mut acc31 = F::from_int(0i64);
        let mut acc32 = F::from_int(0i64);
        let mut acc33 = F::from_int(0i64);

        let mut k0 = 0u32;
        while k0 < cols {
            let kc = k0 + load_k;

            #[unroll]
            for l in 0..8u32 {
                let sr = l * 8u32 + load_r_base;

                let mut xv = F::from_int(0i64);
                if base_i + sr < rows_x {
                    if kc < cols {
                        xv = x[((base_i + sr) * cols + kc) as usize];
                    }
                }
                xs[(load_k * 64u32 + sr) as usize] = xv;

                let mut yv = F::from_int(0i64);
                if base_j + sr < rows_y {
                    if kc < cols {
                        yv = y[((base_j + sr) * cols + kc) as usize];
                    }
                }
                ys[(load_k * 64u32 + sr) as usize] = yv;
            }

            sync_cube();

            // Guard-free inner loops (KNN-02 second pass): `kk` iterates over
            // EXACTLY the valid feature range of this slice (`rem`), in the
            // same ascending order as the guarded rb4 loop, so results stay
            // bitwise identical while the per-iteration `k0 + kk < cols`
            // guard disappears. The hot body is comptime-unrolled in 16-wide
            // chunks (one shared body instance keeps the NVRTC/JIT cost far
            // below a flat 32-wide unroll); sub-16 tails take the runtime
            // -bounded loop.
            // `rem` is the valid feature count of THIS 32-wide slice: the
            // remaining columns, CAPPED at the staged tile width (the rb4
            // loop's `kk < 32` bound) — without the cap a `cols` above the
            // slice width would read past the staged tile.
            let mut rem = cols - k0;
            if rem > 32u32 {
                rem = 32u32;
            }
            let chunks = rem / 16u32;
            let mut h = 0u32;
            while h < chunks {
                let hbase = h * 16u32;
                #[unroll]
                for kki in 0..16u32 {
                    let kk = hbase + kki;
                    let xa0 = xs[(kk * 64u32 + 0u32 + uy) as usize];
                    let xa1 = xs[(kk * 64u32 + 16u32 + uy) as usize];
                    let xa2 = xs[(kk * 64u32 + 32u32 + uy) as usize];
                    let xa3 = xs[(kk * 64u32 + 48u32 + uy) as usize];
                    let yb0 = ys[(kk * 64u32 + 0u32 + ux) as usize];
                    let yb1 = ys[(kk * 64u32 + 16u32 + ux) as usize];
                    let yb2 = ys[(kk * 64u32 + 32u32 + ux) as usize];
                    let yb3 = ys[(kk * 64u32 + 48u32 + ux) as usize];
                    let d00 = xa0 - yb0;
                    acc00 += d00 * d00;
                    let d01 = xa0 - yb1;
                    acc01 += d01 * d01;
                    let d02 = xa0 - yb2;
                    acc02 += d02 * d02;
                    let d03 = xa0 - yb3;
                    acc03 += d03 * d03;
                    let d10 = xa1 - yb0;
                    acc10 += d10 * d10;
                    let d11 = xa1 - yb1;
                    acc11 += d11 * d11;
                    let d12 = xa1 - yb2;
                    acc12 += d12 * d12;
                    let d13 = xa1 - yb3;
                    acc13 += d13 * d13;
                    let d20 = xa2 - yb0;
                    acc20 += d20 * d20;
                    let d21 = xa2 - yb1;
                    acc21 += d21 * d21;
                    let d22 = xa2 - yb2;
                    acc22 += d22 * d22;
                    let d23 = xa2 - yb3;
                    acc23 += d23 * d23;
                    let d30 = xa3 - yb0;
                    acc30 += d30 * d30;
                    let d31 = xa3 - yb1;
                    acc31 += d31 * d31;
                    let d32 = xa3 - yb2;
                    acc32 += d32 * d32;
                    let d33 = xa3 - yb3;
                    acc33 += d33 * d33;
                }
                h += 1u32;
            }
            let mut kk = chunks * 16u32;
            while kk < rem {
                    let xa0 = xs[(kk * 64u32 + 0u32 + uy) as usize];
                    let xa1 = xs[(kk * 64u32 + 16u32 + uy) as usize];
                    let xa2 = xs[(kk * 64u32 + 32u32 + uy) as usize];
                    let xa3 = xs[(kk * 64u32 + 48u32 + uy) as usize];
                    let yb0 = ys[(kk * 64u32 + 0u32 + ux) as usize];
                    let yb1 = ys[(kk * 64u32 + 16u32 + ux) as usize];
                    let yb2 = ys[(kk * 64u32 + 32u32 + ux) as usize];
                    let yb3 = ys[(kk * 64u32 + 48u32 + ux) as usize];
                    let d00 = xa0 - yb0;
                    acc00 += d00 * d00;
                    let d01 = xa0 - yb1;
                    acc01 += d01 * d01;
                    let d02 = xa0 - yb2;
                    acc02 += d02 * d02;
                    let d03 = xa0 - yb3;
                    acc03 += d03 * d03;
                    let d10 = xa1 - yb0;
                    acc10 += d10 * d10;
                    let d11 = xa1 - yb1;
                    acc11 += d11 * d11;
                    let d12 = xa1 - yb2;
                    acc12 += d12 * d12;
                    let d13 = xa1 - yb3;
                    acc13 += d13 * d13;
                    let d20 = xa2 - yb0;
                    acc20 += d20 * d20;
                    let d21 = xa2 - yb1;
                    acc21 += d21 * d21;
                    let d22 = xa2 - yb2;
                    acc22 += d22 * d22;
                    let d23 = xa2 - yb3;
                    acc23 += d23 * d23;
                    let d30 = xa3 - yb0;
                    acc30 += d30 * d30;
                    let d31 = xa3 - yb1;
                    acc31 += d31 * d31;
                    let d32 = xa3 - yb2;
                    acc32 += d32 * d32;
                    let d33 = xa3 - yb3;
                    acc33 += d33 * d33;
                kk += 1u32;
            }

            sync_cube();
            k0 += 32u32;
        }

        // === 2. fold the 16 accumulators into the per-row lists ===
        // (registers + local arrays only — no shared reads, no barriers, so
        // the phase adds nothing to the cube's synchronization schedule.)
        blk[0usize] = acc00;
        blk[1usize] = acc01;
        blk[2usize] = acc02;
        blk[3usize] = acc03;
        blk[4usize] = acc10;
        blk[5usize] = acc11;
        blk[6usize] = acc12;
        blk[7usize] = acc13;
        blk[8usize] = acc20;
        blk[9usize] = acc21;
        blk[10usize] = acc22;
        blk[11usize] = acc23;
        blk[12usize] = acc30;
        blk[13usize] = acc31;
        blk[14usize] = acc32;
        blk[15usize] = acc33;

        // Comptime-unrolled (KNN-02 second pass): with `a`/`b` constant every
        // `lcnt[a]` / `lworst_*[a]` / `blk[..]` index is STATIC, so the
        // compiler can keep the list heads in registers instead of local
        // memory across the (hot) admission tests.
        #[unroll]
        for a in 0..4u32 {
            #[unroll]
            for b in 0..4u32 {
                let v = blk[(a * 4u32 + b) as usize];
                let j = base_j + b * 16u32 + ux;
                // Pad columns stage zeros through the compute phase; the bound
                // guard keeps their garbage accumulators out of the lists.
                if j < rows_y {
                    let cnt = lcnt[a as usize];
                    let mut admit: u32 = 0u32;
                    if cnt < k {
                        admit = 1u32;
                    } else if v < lworst_val[a as usize] {
                        admit = 1u32;
                    } else if v == lworst_val[a as usize] {
                        if j < lworst_idx[a as usize] {
                            admit = 1u32;
                        }
                    }

                    if admit == 1u32 {
                        // Ascending carry-swap insertion (select_k_onepass's):
                        // the carry settles at its sorted slot; with a full
                        // list the old worst falls off the end.
                        let mut cav = v;
                        let mut cai = j;
                        let mut fs = 0u32;
                        while fs < cnt {
                            let jv = lval[(a * 16u32 + fs) as usize];
                            let ji = lidx[(a * 16u32 + fs) as usize];
                            let mut swap: u32 = 0u32;
                            if cav < jv {
                                swap = 1u32;
                            } else if cav == jv {
                                if cai < ji {
                                    swap = 1u32;
                                }
                            }
                            if swap == 1u32 {
                                lval[(a * 16u32 + fs) as usize] = cav;
                                lidx[(a * 16u32 + fs) as usize] = cai;
                                cav = jv;
                                cai = ji;
                            }
                            fs += 1u32;
                        }
                        let mut ncnt = cnt;
                        if cnt < k {
                            lval[(a * 16u32 + cnt) as usize] = cav;
                            lidx[(a * 16u32 + cnt) as usize] = cai;
                            ncnt = cnt + 1u32;
                            lcnt[a as usize] = ncnt;
                        }
                        lworst_val[a as usize] = lval[(a * 16u32 + ncnt - 1u32) as usize];
                        lworst_idx[a as usize] = lidx[(a * 16u32 + ncnt - 1u32) as usize];
                    }
                }
            }
        }

        b0 += 64u32;
    }

    // ---- k-round head merge, one row group (16 units, same uy) at a time ----
    let flat = uy * 16u32 + ux;
    let mut am = 0u32;
    while am < 4u32 {
        let orow = base_i + am * 16u32 + uy;
        let mut p = 0u32;
        let mut r = 0u32;
        while r < k {
            // Post this unit's smallest unconsumed pair for row group `uy`
            // (validity-flagged; the value behind `has == 0` is never read).
            let mut hv = zero_f;
            let mut hi = 0u32;
            let mut hh = zero_f;
            if p < lcnt[am as usize] {
                hv = lval[(am * 16u32 + p) as usize];
                hi = lidx[(am * 16u32 + p) as usize];
                hh = one_f;
            }
            xs[flat as usize] = hv;
            ys[flat as usize] = F::cast_from(hi);
            ys[(256u32 + flat) as usize] = hh;
            sync_cube();

            // log₂ tree over the 16 slots of each row group, pair order +
            // validity flag (the argmin_shared idiom).
            let mut s = 8u32;
            while s > 0u32 {
                if ux < s {
                    let oh = ys[(256u32 + flat + s) as usize];
                    if oh == one_f {
                        let ov = xs[(flat + s) as usize];
                        let oi = ys[(flat + s) as usize];
                        let ch = ys[(256u32 + flat) as usize];
                        let cv = xs[flat as usize];
                        let ci = ys[flat as usize];

                        let mut better: u32 = 0u32;
                        if ch == zero_f {
                            better = 1u32;
                        } else if ov < cv {
                            better = 1u32;
                        } else if ov == cv {
                            if oi < ci {
                                better = 1u32;
                            }
                        }
                        if better == 1u32 {
                            xs[flat as usize] = ov;
                            ys[flat as usize] = oi;
                            ys[(256u32 + flat) as usize] = one_f;
                        }
                    }
                }
                sync_cube();
                s /= 2u32;
            }

            // Slot ux == 0 of the row group holds the round winner. `k <=
            // rows_y` (validated host-side) guarantees the union of a REAL
            // row's lists has at least k pairs, so its winner is always valid;
            // pad rows (orow >= rows_x) are simply never written.
            let wv = xs[(uy * 16u32) as usize];
            let wi = ys[(uy * 16u32) as usize];
            let wh = ys[(256u32 + uy * 16u32) as usize];
            if ux == 0u32 {
                if wh == one_f {
                    if orow < rows_x {
                        // Strip-blocked output: row-major `rows_x x (strips*k)`
                        // with strip s owning slots [s*k, s*k + k).
                        let slot = orow * (strips * k) + strip * k + r;
                        out_val[slot as usize] = wv;
                        out_idx[slot as usize] = u32::cast_from(wi);
                    }
                }
            }
            // The unique owning unit advances past the emitted pair.
            if p < lcnt[am as usize] {
                if lval[(am * 16u32 + p) as usize] == wv {
                    if lidx[(am * 16u32 + p) as usize] == u32::cast_from(wi) {
                        if wh == one_f {
                            p += 1u32;
                        }
                    }
                }
            }

            // Every unit has READ its row group's slot 0; barrier before the
            // next round's posts overwrite the scratch.
            sync_cube();

            r += 1u32;
        }
        am += 1u32;
    }
}

/// FUSED k-nearest search via the DOT-PRODUCT EXPANSION, specialized for
/// `cols <= 32` (KNN-02 fifth pass) — the arithmetic-density form of
/// [`euclidean_topk_fused`].
///
/// ## Why (the instruction budget)
/// The direct fused kernel's inner step spends 8 shared loads + 16 subtracts
/// + 16 FMAs per 16 pair-products; at sustained T4 clocks it reaches ~10.9
/// Gpair/s while cuML's expansion-based kernel reaches ~18.8. This kernel
/// switches the inner loop to the expansion `d² = ‖x‖² + ‖y‖² − 2·x·y`: the
/// hot step is `acc += xa * yb` — 16 FMAs per 8 loads, the subtracts gone —
/// and the norm terms are folded in ONCE per 64×64 block at admission time.
/// Additionally, with `cols <= 32` the whole query tile fits one feature
/// slice, so `xs` is staged ONCE per cube instead of once per train block:
/// per-block staging halves, and the redundant global re-reads of `x`
/// (`n_train/64` per query block) disappear.
///
/// ## Numerics (why this is allowed, and its limits)
/// sklearn's own brute-force KNN computes distances through this same
/// expansion, and the project bar is 1e-5 agreement with sklearn — which the
/// expansion (with the `max(d², 0)` cancellation clamp, the
/// `distance_with_ynorm` precedent) meets. On the tie-saturated QUANTIZED
/// fixtures the expansion is EXACT (small-integer products and sums below
/// 2^24 in f32), so selection remains bitwise-identical to the direct kernel
/// there; on real-valued data the last-ulp rounding may differ from the
/// direct form, so the equivalence gate for this kernel asserts equal
/// indices and tolerance-equal values rather than bitwise values. The
/// cancellation-free direct kernel stays available (`MLRS_KNN_DIRECT=1`, and
/// automatically for `cols > 32`).
///
/// - `xnorm` / `ynorm` are the per-row squared norms of `x` / `y`
///   (`row_reduce(SumSq)`), computed once by the host.
/// - Everything else — grid/strip layout, per-unit fold lists, the k-round
///   merge that reuses the staging tiles (16 KiB budget unchanged), the
///   host-side validation contract — matches [`euclidean_topk_fused`].
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn euclidean_topk_fused_dot<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xnorm: &Array<F>,
    ynorm: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    k: u32,
    strip_len: u32,
    strips: u32,
) {
    let mut xs = SharedMemory::<F>::new(2048usize);
    let mut ys = SharedMemory::<F>::new(2048usize);

    let mut lval = Array::<F>::new(64usize);
    let mut lidx = Array::<u32>::new(64usize);
    let mut lcnt = Array::<u32>::new(4usize);
    let mut lworst_val = Array::<F>::new(4usize);
    let mut lworst_idx = Array::<u32>::new(4usize);
    let mut blk = Array::<F>::new(16usize);
    // Register cache of this unit's 4 query-row norms (rows `a*16 + uy`).
    let mut xn = Array::<F>::new(4usize);

    let ux = UNIT_POS_X;
    let uy = UNIT_POS_Y;
    let base_i = CUBE_POS_Y * 64u32;

    let load_k = ux + 16u32 * (uy % 2u32);
    let load_r_base = uy / 2u32;

    let zero_f = F::from_int(0i64);
    let one_f = F::from_int(1i64);
    lcnt[0usize] = 0u32;
    lcnt[1usize] = 0u32;
    lcnt[2usize] = 0u32;
    lcnt[3usize] = 0u32;
    lworst_val[0usize] = zero_f;
    lworst_val[1usize] = zero_f;
    lworst_val[2usize] = zero_f;
    lworst_val[3usize] = zero_f;
    lworst_idx[0usize] = 0u32;
    lworst_idx[1usize] = 0u32;
    lworst_idx[2usize] = 0u32;
    lworst_idx[3usize] = 0u32;
    #[unroll]
    for a in 0..4u32 {
        let xr = base_i + a * 16u32 + uy;
        let mut v = zero_f;
        if xr < rows_x {
            v = xnorm[xr as usize];
        }
        xn[a as usize] = v;
    }

    // --- Stage the ENTIRE query tile once: cols <= 32 is one feature slice.
    //     Same staging map and zero-padding as the per-block staging of the
    //     direct kernel; `xs` then stays untouched until the final merge. ---
    #[unroll]
    for l in 0..8u32 {
        let sr = l * 8u32 + load_r_base;
        let mut xv = F::from_int(0i64);
        if base_i + sr < rows_x {
            if load_k < cols {
                xv = x[((base_i + sr) * cols + load_k) as usize];
            }
        }
        xs[(load_k * 64u32 + sr) as usize] = xv;
    }
    sync_cube();

    let strip = CUBE_POS_X;
    let c_begin = strip * strip_len;
    let mut c_end = c_begin + strip_len;
    if c_end > rows_y {
        c_end = rows_y;
    }
    let mut b0 = c_begin;
    while b0 < c_end {
        let base_j = b0;

        // === 1a. stage this block's train tile (y only — x is resident) ===
        #[unroll]
        for l in 0..8u32 {
            let sr = l * 8u32 + load_r_base;
            let mut yv = F::from_int(0i64);
            if base_j + sr < rows_y {
                if load_k < cols {
                    yv = y[((base_j + sr) * cols + load_k) as usize];
                }
            }
            ys[(load_k * 64u32 + sr) as usize] = yv;
        }
        sync_cube();

        // === 1b. dot-product accumulation: 8 loads feed 16 FMAs ===
        let mut acc00 = F::from_int(0i64);
        let mut acc01 = F::from_int(0i64);
        let mut acc02 = F::from_int(0i64);
        let mut acc03 = F::from_int(0i64);
        let mut acc10 = F::from_int(0i64);
        let mut acc11 = F::from_int(0i64);
        let mut acc12 = F::from_int(0i64);
        let mut acc13 = F::from_int(0i64);
        let mut acc20 = F::from_int(0i64);
        let mut acc21 = F::from_int(0i64);
        let mut acc22 = F::from_int(0i64);
        let mut acc23 = F::from_int(0i64);
        let mut acc30 = F::from_int(0i64);
        let mut acc31 = F::from_int(0i64);
        let mut acc32 = F::from_int(0i64);
        let mut acc33 = F::from_int(0i64);

        // Chunked comptime unroll over exactly the valid feature range (the
        // select-loop discipline of the direct kernel; cols <= 32 so the
        // chunk count is 0..2 plus a sub-16 tail).
        let chunks = cols / 16u32;
        let mut h = 0u32;
        while h < chunks {
            let hbase = h * 16u32;
            #[unroll]
            for kki in 0..16u32 {
                let kk = hbase + kki;
                let xa0 = xs[(kk * 64u32 + 0u32 + uy) as usize];
                let xa1 = xs[(kk * 64u32 + 16u32 + uy) as usize];
                let xa2 = xs[(kk * 64u32 + 32u32 + uy) as usize];
                let xa3 = xs[(kk * 64u32 + 48u32 + uy) as usize];
                let yb0 = ys[(kk * 64u32 + 0u32 + ux) as usize];
                let yb1 = ys[(kk * 64u32 + 16u32 + ux) as usize];
                let yb2 = ys[(kk * 64u32 + 32u32 + ux) as usize];
                let yb3 = ys[(kk * 64u32 + 48u32 + ux) as usize];
                acc00 += xa0 * yb0;
                acc01 += xa0 * yb1;
                acc02 += xa0 * yb2;
                acc03 += xa0 * yb3;
                acc10 += xa1 * yb0;
                acc11 += xa1 * yb1;
                acc12 += xa1 * yb2;
                acc13 += xa1 * yb3;
                acc20 += xa2 * yb0;
                acc21 += xa2 * yb1;
                acc22 += xa2 * yb2;
                acc23 += xa2 * yb3;
                acc30 += xa3 * yb0;
                acc31 += xa3 * yb1;
                acc32 += xa3 * yb2;
                acc33 += xa3 * yb3;
            }
            h += 1u32;
        }
        let mut kk = chunks * 16u32;
        while kk < cols {
            let xa0 = xs[(kk * 64u32 + 0u32 + uy) as usize];
            let xa1 = xs[(kk * 64u32 + 16u32 + uy) as usize];
            let xa2 = xs[(kk * 64u32 + 32u32 + uy) as usize];
            let xa3 = xs[(kk * 64u32 + 48u32 + uy) as usize];
            let yb0 = ys[(kk * 64u32 + 0u32 + ux) as usize];
            let yb1 = ys[(kk * 64u32 + 16u32 + ux) as usize];
            let yb2 = ys[(kk * 64u32 + 32u32 + ux) as usize];
            let yb3 = ys[(kk * 64u32 + 48u32 + ux) as usize];
            acc00 += xa0 * yb0;
            acc01 += xa0 * yb1;
            acc02 += xa0 * yb2;
            acc03 += xa0 * yb3;
            acc10 += xa1 * yb0;
            acc11 += xa1 * yb1;
            acc12 += xa1 * yb2;
            acc13 += xa1 * yb3;
            acc20 += xa2 * yb0;
            acc21 += xa2 * yb1;
            acc22 += xa2 * yb2;
            acc23 += xa2 * yb3;
            acc30 += xa3 * yb0;
            acc31 += xa3 * yb1;
            acc32 += xa3 * yb2;
            acc33 += xa3 * yb3;
            kk += 1u32;
        }

        // Barrier before the next block's staging overwrites `ys` — ALSO
        // required before the fold below reads nothing shared (the fold is
        // local-only), so a single end-of-compute barrier suffices.
        sync_cube();

        // === 2. fold: expand to the squared distance, clamp, admit ===
        blk[0usize] = acc00;
        blk[1usize] = acc01;
        blk[2usize] = acc02;
        blk[3usize] = acc03;
        blk[4usize] = acc10;
        blk[5usize] = acc11;
        blk[6usize] = acc12;
        blk[7usize] = acc13;
        blk[8usize] = acc20;
        blk[9usize] = acc21;
        blk[10usize] = acc22;
        blk[11usize] = acc23;
        blk[12usize] = acc30;
        blk[13usize] = acc31;
        blk[14usize] = acc32;
        blk[15usize] = acc33;

        #[unroll]
        for a in 0..4u32 {
            #[unroll]
            for b in 0..4u32 {
                let j = base_j + b * 16u32 + ux;
                if j < rows_y {
                    // d² = ‖x‖² + ‖y‖² − 2·x·y, clamped at 0 so cancellation
                    // can never admit a negative distance (Criterion 3 /
                    // distance_with_ynorm precedent).
                    let dot = blk[(a * 4u32 + b) as usize];
                    let mut v = xn[a as usize] + ynorm[j as usize] - (dot + dot);
                    if v < zero_f {
                        v = zero_f;
                    }
                    let cnt = lcnt[a as usize];
                    let mut admit: u32 = 0u32;
                    if cnt < k {
                        admit = 1u32;
                    } else if v < lworst_val[a as usize] {
                        admit = 1u32;
                    } else if v == lworst_val[a as usize] {
                        if j < lworst_idx[a as usize] {
                            admit = 1u32;
                        }
                    }

                    if admit == 1u32 {
                        let mut cav = v;
                        let mut cai = j;
                        let mut fs = 0u32;
                        while fs < cnt {
                            let jv = lval[(a * 16u32 + fs) as usize];
                            let ji = lidx[(a * 16u32 + fs) as usize];
                            let mut swap: u32 = 0u32;
                            if cav < jv {
                                swap = 1u32;
                            } else if cav == jv {
                                if cai < ji {
                                    swap = 1u32;
                                }
                            }
                            if swap == 1u32 {
                                lval[(a * 16u32 + fs) as usize] = cav;
                                lidx[(a * 16u32 + fs) as usize] = cai;
                                cav = jv;
                                cai = ji;
                            }
                            fs += 1u32;
                        }
                        let mut ncnt = cnt;
                        if cnt < k {
                            lval[(a * 16u32 + cnt) as usize] = cav;
                            lidx[(a * 16u32 + cnt) as usize] = cai;
                            ncnt = cnt + 1u32;
                            lcnt[a as usize] = ncnt;
                        }
                        lworst_val[a as usize] = lval[(a * 16u32 + ncnt - 1u32) as usize];
                        lworst_idx[a as usize] = lidx[(a * 16u32 + ncnt - 1u32) as usize];
                    }
                }
            }
        }

        b0 += 64u32;
    }

    // ---- k-round head merge: identical to euclidean_topk_fused (the staging
    //      tiles are dead — xs was only read until the last compute barrier —
    //      so they serve as merge scratch with the same F-encoding). ----
    let flat = uy * 16u32 + ux;
    let mut am = 0u32;
    while am < 4u32 {
        let orow = base_i + am * 16u32 + uy;
        let mut p = 0u32;
        let mut r = 0u32;
        while r < k {
            let mut hv = zero_f;
            let mut hi = 0u32;
            let mut hh = zero_f;
            if p < lcnt[am as usize] {
                hv = lval[(am * 16u32 + p) as usize];
                hi = lidx[(am * 16u32 + p) as usize];
                hh = one_f;
            }
            xs[flat as usize] = hv;
            ys[flat as usize] = F::cast_from(hi);
            ys[(256u32 + flat) as usize] = hh;
            sync_cube();

            let mut s = 8u32;
            while s > 0u32 {
                if ux < s {
                    let oh = ys[(256u32 + flat + s) as usize];
                    if oh == one_f {
                        let ov = xs[(flat + s) as usize];
                        let oi = ys[(flat + s) as usize];
                        let ch = ys[(256u32 + flat) as usize];
                        let cv = xs[flat as usize];
                        let ci = ys[flat as usize];

                        let mut better: u32 = 0u32;
                        if ch == zero_f {
                            better = 1u32;
                        } else if ov < cv {
                            better = 1u32;
                        } else if ov == cv {
                            if oi < ci {
                                better = 1u32;
                            }
                        }
                        if better == 1u32 {
                            xs[flat as usize] = ov;
                            ys[flat as usize] = oi;
                            ys[(256u32 + flat) as usize] = one_f;
                        }
                    }
                }
                sync_cube();
                s /= 2u32;
            }

            let wv = xs[(uy * 16u32) as usize];
            let wi = ys[(uy * 16u32) as usize];
            let wh = ys[(256u32 + uy * 16u32) as usize];
            if ux == 0u32 {
                if wh == one_f {
                    if orow < rows_x {
                        let slot = orow * (strips * k) + strip * k + r;
                        out_val[slot as usize] = wv;
                        out_idx[slot as usize] = u32::cast_from(wi);
                    }
                }
            }
            if p < lcnt[am as usize] {
                if lval[(am * 16u32 + p) as usize] == wv {
                    if lidx[(am * 16u32 + p) as usize] == u32::cast_from(wi) {
                        if wh == one_f {
                            p += 1u32;
                        }
                    }
                }
            }

            sync_cube();

            r += 1u32;
        }
        am += 1u32;
    }
}

/// VECTORIZED-shared-read variant of [`euclidean_topk_fused_dot`] (KNN-03) —
/// same expansion arithmetic, `f32x4`-shaped tile accesses.
///
/// ## Why (the shared-load instruction budget)
/// The scalar dot kernel's inner step issues 8 scalar shared loads per 16 FMAs
/// (24 instructions per 16 pair-products); at sustained T4 clocks it reaches
/// ~10.9 Gpair/s while cuML's fusedL2kNN reaches ~18.8 — the gap is
/// LDS-instruction issue, not data volume. This kernel re-shapes the tiles so
/// each unit's 4 query rows / 4 train columns are CONSECUTIVE (`[k][g]` tiles
/// of `Vector<F, 4>`: vector `g` of slice `kk` holds rows `4g..4g+4`), so the
/// inner step is 2 VECTOR loads + 16 FMAs. Byte size and budget are identical
/// (512 × 4-wide vectors = 2048 `F` per tile).
///
/// ## Bitwise contract
/// For every `(query row, train column)` pair the accumulation is the SAME
/// scalar sequence `acc += x[r,kk] * y[c,kk]` over ascending `kk` as the
/// scalar dot kernel — only WHICH UNIT computes a pair changes (unit `(ux,uy)`
/// owns rows `uy*4+a` × cols `ux*4+b` instead of `a*16+uy` × `b*16+ux`). The
/// per-unit fold partitions the candidate set differently, but the k-round
/// merge is over the total `(value, index)` pair order, so the emitted top-k
/// is invariant under the partition: results are BITWISE identical to
/// [`euclidean_topk_fused_dot`], and the equivalence gate asserts exactly
/// that.
///
/// ## Staging trade (measured reasoning, see the KNN-02 notes)
/// A staged vector gathers 4 rows at one feature, so the staging global reads
/// keep the warp coalesced along the feature axis (lanes span `sk`) while the
/// vector STORES concentrate on a 4-bank group (~2 conflicted store
/// instructions per tile per unit). That replaces the scalar kernels' 8
/// stride-64 stores (16-way conflicted) per tile per unit — staging cost goes
/// DOWN, and the inner loop it feeds runs 32 iterations per staging, so the
/// store conflicts are second-order.
///
/// The k-round merge reuses `xs` as scratch like the scalar kernels, but packs
/// the whole `(value, F-encoded index, validity)` triple into ONE vector slot
/// (components 0/1/2), so the tree copies one vector instead of three scalars.
///
/// Host contract (norms, strips, validation, output layout) is exactly
/// [`euclidean_topk_fused_dot`]'s.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn euclidean_topk_fused_dot_vec<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xnorm: &Array<F>,
    ynorm: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    k: u32,
    strip_len: u32,
    strips: u32,
) {
    // [k][group] VECTOR tiles: 32 feature slices × 16 groups of 4 consecutive
    // rows. 512 × Vector<F, 4> = 2048 F — byte-identical to the scalar tiles.
    let mut xs = SharedMemory::<Vector<F, Const<4>>>::new(512usize);
    let mut ys = SharedMemory::<Vector<F, Const<4>>>::new(512usize);

    let mut lval = Array::<F>::new(64usize);
    let mut lidx = Array::<u32>::new(64usize);
    let mut lcnt = Array::<u32>::new(4usize);
    let mut lworst_val = Array::<F>::new(4usize);
    let mut lworst_idx = Array::<u32>::new(4usize);
    let mut blk = Array::<F>::new(16usize);
    // Register cache of this unit's 4 query-row norms (rows `uy*4 + a`).
    let mut xn = Array::<F>::new(4usize);

    let ux = UNIT_POS_X;
    let uy = UNIT_POS_Y;
    let base_i = CUBE_POS_Y * 64u32;

    // Staging map: lanes of a warp span the feature slice `sk` (coalesced
    // global reads along a row); each unit stages the two row-group vectors
    // `sgb` and `sgb + 8` of its slice.
    let sk = ux + 16u32 * (uy % 2u32);
    let sgb = uy / 2u32;

    let zero_f = F::from_int(0i64);
    let one_f = F::from_int(1i64);
    lcnt[0usize] = 0u32;
    lcnt[1usize] = 0u32;
    lcnt[2usize] = 0u32;
    lcnt[3usize] = 0u32;
    lworst_val[0usize] = zero_f;
    lworst_val[1usize] = zero_f;
    lworst_val[2usize] = zero_f;
    lworst_val[3usize] = zero_f;
    lworst_idx[0usize] = 0u32;
    lworst_idx[1usize] = 0u32;
    lworst_idx[2usize] = 0u32;
    lworst_idx[3usize] = 0u32;
    #[unroll]
    for a in 0..4u32 {
        let xr = base_i + uy * 4u32 + a;
        let mut v = zero_f;
        if xr < rows_x {
            v = xnorm[xr as usize];
        }
        xn[a as usize] = v;
    }

    // --- Stage the ENTIRE query tile once (cols <= 32 is one feature slice).
    //     Zero-padded exactly like the scalar staging: out-of-range rows and
    //     features stage 0, so pad lanes are inert through the compute. ---
    #[unroll]
    for l in 0..2u32 {
        let g = l * 8u32 + sgb;
        let mut v = Vector::<F, Const<4>>::from_int(0i64);
        #[unroll]
        for i in 0..4u32 {
            let r = base_i + g * 4u32 + i;
            let mut xv = zero_f;
            if r < rows_x {
                if sk < cols {
                    xv = x[(r * cols + sk) as usize];
                }
            }
            v[i as usize] = xv;
        }
        xs[(sk * 16u32 + g) as usize] = v;
    }
    sync_cube();

    let strip = CUBE_POS_X;
    let c_begin = strip * strip_len;
    let mut c_end = c_begin + strip_len;
    if c_end > rows_y {
        c_end = rows_y;
    }
    let mut b0 = c_begin;
    while b0 < c_end {
        let base_j = b0;

        // === 1a. stage this block's train tile (y only — x is resident) ===
        #[unroll]
        for l in 0..2u32 {
            let g = l * 8u32 + sgb;
            let mut v = Vector::<F, Const<4>>::from_int(0i64);
            #[unroll]
            for i in 0..4u32 {
                let r = base_j + g * 4u32 + i;
                let mut yv = zero_f;
                if r < rows_y {
                    if sk < cols {
                        yv = y[(r * cols + sk) as usize];
                    }
                }
                v[i as usize] = yv;
            }
            ys[(sk * 16u32 + g) as usize] = v;
        }
        sync_cube();

        // === 1b. dot-product accumulation: 2 VECTOR loads feed 16 FMAs ===
        let mut acc00 = F::from_int(0i64);
        let mut acc01 = F::from_int(0i64);
        let mut acc02 = F::from_int(0i64);
        let mut acc03 = F::from_int(0i64);
        let mut acc10 = F::from_int(0i64);
        let mut acc11 = F::from_int(0i64);
        let mut acc12 = F::from_int(0i64);
        let mut acc13 = F::from_int(0i64);
        let mut acc20 = F::from_int(0i64);
        let mut acc21 = F::from_int(0i64);
        let mut acc22 = F::from_int(0i64);
        let mut acc23 = F::from_int(0i64);
        let mut acc30 = F::from_int(0i64);
        let mut acc31 = F::from_int(0i64);
        let mut acc32 = F::from_int(0i64);
        let mut acc33 = F::from_int(0i64);

        // Chunked comptime unroll over exactly the valid feature range (the
        // scalar dot kernel's discipline; cols <= 32 so the chunk count is
        // 0..2 plus a sub-16 tail).
        let chunks = cols / 16u32;
        let mut h = 0u32;
        while h < chunks {
            let hbase = h * 16u32;
            #[unroll]
            for kki in 0..16u32 {
                let kk = hbase + kki;
                let xv = xs[(kk * 16u32 + uy) as usize];
                let yv = ys[(kk * 16u32 + ux) as usize];
                let xa0 = xv[0usize];
                let xa1 = xv[1usize];
                let xa2 = xv[2usize];
                let xa3 = xv[3usize];
                let yb0 = yv[0usize];
                let yb1 = yv[1usize];
                let yb2 = yv[2usize];
                let yb3 = yv[3usize];
                acc00 += xa0 * yb0;
                acc01 += xa0 * yb1;
                acc02 += xa0 * yb2;
                acc03 += xa0 * yb3;
                acc10 += xa1 * yb0;
                acc11 += xa1 * yb1;
                acc12 += xa1 * yb2;
                acc13 += xa1 * yb3;
                acc20 += xa2 * yb0;
                acc21 += xa2 * yb1;
                acc22 += xa2 * yb2;
                acc23 += xa2 * yb3;
                acc30 += xa3 * yb0;
                acc31 += xa3 * yb1;
                acc32 += xa3 * yb2;
                acc33 += xa3 * yb3;
            }
            h += 1u32;
        }
        let mut kk = chunks * 16u32;
        while kk < cols {
            let xv = xs[(kk * 16u32 + uy) as usize];
            let yv = ys[(kk * 16u32 + ux) as usize];
            let xa0 = xv[0usize];
            let xa1 = xv[1usize];
            let xa2 = xv[2usize];
            let xa3 = xv[3usize];
            let yb0 = yv[0usize];
            let yb1 = yv[1usize];
            let yb2 = yv[2usize];
            let yb3 = yv[3usize];
            acc00 += xa0 * yb0;
            acc01 += xa0 * yb1;
            acc02 += xa0 * yb2;
            acc03 += xa0 * yb3;
            acc10 += xa1 * yb0;
            acc11 += xa1 * yb1;
            acc12 += xa1 * yb2;
            acc13 += xa1 * yb3;
            acc20 += xa2 * yb0;
            acc21 += xa2 * yb1;
            acc22 += xa2 * yb2;
            acc23 += xa2 * yb3;
            acc30 += xa3 * yb0;
            acc31 += xa3 * yb1;
            acc32 += xa3 * yb2;
            acc33 += xa3 * yb3;
            kk += 1u32;
        }

        // Barrier before the next block's staging overwrites `ys`.
        sync_cube();

        // === 2. fold: expand to the squared distance, clamp, admit ===
        blk[0usize] = acc00;
        blk[1usize] = acc01;
        blk[2usize] = acc02;
        blk[3usize] = acc03;
        blk[4usize] = acc10;
        blk[5usize] = acc11;
        blk[6usize] = acc12;
        blk[7usize] = acc13;
        blk[8usize] = acc20;
        blk[9usize] = acc21;
        blk[10usize] = acc22;
        blk[11usize] = acc23;
        blk[12usize] = acc30;
        blk[13usize] = acc31;
        blk[14usize] = acc32;
        blk[15usize] = acc33;

        #[unroll]
        for a in 0..4u32 {
            #[unroll]
            for b in 0..4u32 {
                let j = base_j + ux * 4u32 + b;
                if j < rows_y {
                    let dot = blk[(a * 4u32 + b) as usize];
                    let mut v = xn[a as usize] + ynorm[j as usize] - (dot + dot);
                    if v < zero_f {
                        v = zero_f;
                    }
                    let cnt = lcnt[a as usize];
                    let mut admit: u32 = 0u32;
                    if cnt < k {
                        admit = 1u32;
                    } else if v < lworst_val[a as usize] {
                        admit = 1u32;
                    } else if v == lworst_val[a as usize] {
                        if j < lworst_idx[a as usize] {
                            admit = 1u32;
                        }
                    }

                    if admit == 1u32 {
                        let mut cav = v;
                        let mut cai = j;
                        let mut fs = 0u32;
                        while fs < cnt {
                            let jv = lval[(a * 16u32 + fs) as usize];
                            let ji = lidx[(a * 16u32 + fs) as usize];
                            let mut swap: u32 = 0u32;
                            if cav < jv {
                                swap = 1u32;
                            } else if cav == jv {
                                if cai < ji {
                                    swap = 1u32;
                                }
                            }
                            if swap == 1u32 {
                                lval[(a * 16u32 + fs) as usize] = cav;
                                lidx[(a * 16u32 + fs) as usize] = cai;
                                cav = jv;
                                cai = ji;
                            }
                            fs += 1u32;
                        }
                        let mut ncnt = cnt;
                        if cnt < k {
                            lval[(a * 16u32 + cnt) as usize] = cav;
                            lidx[(a * 16u32 + cnt) as usize] = cai;
                            ncnt = cnt + 1u32;
                            lcnt[a as usize] = ncnt;
                        }
                        lworst_val[a as usize] = lval[(a * 16u32 + ncnt - 1u32) as usize];
                        lworst_idx[a as usize] = lidx[(a * 16u32 + ncnt - 1u32) as usize];
                    }
                }
            }
        }

        b0 += 64u32;
    }

    // ---- k-round head merge: the scalar kernels' tree, with the whole
    //      (value, F-encoded index, validity) triple packed into components
    //      0/1/2 of ONE scratch vector per unit (xs is dead after the walk).
    //      Row `uy*4 + am` is merged over the 16 units sharing `uy`. ----
    let flat = uy * 16u32 + ux;
    let mut am = 0u32;
    while am < 4u32 {
        let orow = base_i + uy * 4u32 + am;
        let mut p = 0u32;
        let mut r = 0u32;
        while r < k {
            let mut hv = zero_f;
            let mut hi = 0u32;
            let mut hh = zero_f;
            if p < lcnt[am as usize] {
                hv = lval[(am * 16u32 + p) as usize];
                hi = lidx[(am * 16u32 + p) as usize];
                hh = one_f;
            }
            let mut post = Vector::<F, Const<4>>::from_int(0i64);
            post[0usize] = hv;
            post[1usize] = F::cast_from(hi);
            post[2usize] = hh;
            xs[flat as usize] = post;
            sync_cube();

            let mut s = 8u32;
            while s > 0u32 {
                if ux < s {
                    let o = xs[(flat + s) as usize];
                    if o[2usize] == one_f {
                        let c = xs[flat as usize];

                        let mut better: u32 = 0u32;
                        if c[2usize] == zero_f {
                            better = 1u32;
                        } else if o[0usize] < c[0usize] {
                            better = 1u32;
                        } else if o[0usize] == c[0usize] {
                            if o[1usize] < c[1usize] {
                                better = 1u32;
                            }
                        }
                        if better == 1u32 {
                            xs[flat as usize] = o;
                        }
                    }
                }
                sync_cube();
                s /= 2u32;
            }

            let w = xs[(uy * 16u32) as usize];
            let wv = w[0usize];
            let wi = w[1usize];
            let wh = w[2usize];
            if ux == 0u32 {
                if wh == one_f {
                    if orow < rows_x {
                        let slot = orow * (strips * k) + strip * k + r;
                        out_val[slot as usize] = wv;
                        out_idx[slot as usize] = u32::cast_from(wi);
                    }
                }
            }
            if p < lcnt[am as usize] {
                if lval[(am * 16u32 + p) as usize] == wv {
                    if lidx[(am * 16u32 + p) as usize] == u32::cast_from(wi) {
                        if wh == one_f {
                            p += 1u32;
                        }
                    }
                }
            }

            sync_cube();

            r += 1u32;
        }
        am += 1u32;
    }
}

/// 4×8 REGISTER-TILE variant of [`euclidean_topk_fused_dot_vec`] (KNN-03
/// round 3) — 3 vector loads feed 32 FMAs.
///
/// ## Why (arithmetic per shared-load instruction)
/// The 4×4 vec kernel's inner step is 2 vector loads + 16 FMAs (18
/// instructions per 16 pair-products). Widening each unit's output tile to 4
/// query rows × 8 train columns makes the step 3 vector loads + 32 FMAs (35
/// per 32) — arithmetic density rises from 0.89 to 0.91 FMA/instr and, more
/// importantly, per-unit ILP doubles, halving the loop-overhead share. The
/// cube shrinks to 8×16 = 128 units covering the same 64×64 block (each unit
/// owns train-column groups `2ux` and `2ux+1`).
///
/// ## Bitwise contract
/// Same as the 4×4 vec kernel: per pair the scalar accumulation sequence is
/// unchanged, only the unit→pair partition differs; the k-round merge (now a
/// log₂ tree over the 8 units sharing a query row) is over the total pair
/// order, so results are bitwise-identical to [`euclidean_topk_fused_dot`].
///
/// Shared budget identical (2 × 512 vectors = 16 KiB at f32). Launched with
/// `CubeDim { x: 8, y: 16 }` — the HOST dispatch must match.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn euclidean_topk_fused_dot_vec8<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xnorm: &Array<F>,
    ynorm: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    k: u32,
    strip_len: u32,
    strips: u32,
) {
    let mut xs = SharedMemory::<Vector<F, Const<4>>>::new(512usize);
    let mut ys = SharedMemory::<Vector<F, Const<4>>>::new(512usize);

    let mut lval = Array::<F>::new(64usize);
    let mut lidx = Array::<u32>::new(64usize);
    let mut lcnt = Array::<u32>::new(4usize);
    let mut lworst_val = Array::<F>::new(4usize);
    let mut lworst_idx = Array::<u32>::new(4usize);
    let mut blk = Array::<F>::new(32usize);
    let mut xn = Array::<F>::new(4usize);
    // Per-block register cache of the unit's 8 train-column norms (r7).
    let mut yn = Array::<F>::new(8usize);

    let ux = UNIT_POS_X; // 0..8 — train-column octet
    let uy = UNIT_POS_Y; // 0..16 — query-row quartet
    let base_i = CUBE_POS_Y * 64u32;

    // Staging map for 128 units: lanes of a warp span the feature slice `sk`
    // (coalesced global reads along a row); each unit stages FOUR row-group
    // vectors of its slice.
    let sk = (uy % 4u32) * 8u32 + ux;
    let sgb = uy / 4u32;

    let zero_f = F::from_int(0i64);
    lcnt[0usize] = 0u32;
    lcnt[1usize] = 0u32;
    lcnt[2usize] = 0u32;
    lcnt[3usize] = 0u32;
    lworst_val[0usize] = zero_f;
    lworst_val[1usize] = zero_f;
    lworst_val[2usize] = zero_f;
    lworst_val[3usize] = zero_f;
    lworst_idx[0usize] = 0u32;
    lworst_idx[1usize] = 0u32;
    lworst_idx[2usize] = 0u32;
    lworst_idx[3usize] = 0u32;
    #[unroll]
    for a in 0..4u32 {
        let xr = base_i + uy * 4u32 + a;
        let mut v = zero_f;
        if xr < rows_x {
            v = xnorm[xr as usize];
        }
        xn[a as usize] = v;
    }

    // --- Stage the ENTIRE query tile once (cols <= 32 is one feature slice) ---
    #[unroll]
    for l in 0..4u32 {
        let g = l * 4u32 + sgb;
        let mut v = Vector::<F, Const<4>>::from_int(0i64);
        #[unroll]
        for i in 0..4u32 {
            let r = base_i + g * 4u32 + i;
            let mut xv = zero_f;
            if r < rows_x {
                if sk < cols {
                    xv = x[(r * cols + sk) as usize];
                }
            }
            v[i as usize] = xv;
        }
        xs[(sk * 16u32 + g) as usize] = v;
    }
    sync_cube();

    let strip = CUBE_POS_X;
    let c_begin = strip * strip_len;
    let mut c_end = c_begin + strip_len;
    if c_end > rows_y {
        c_end = rows_y;
    }
    let mut b0 = c_begin;
    while b0 < c_end {
        let base_j = b0;

        // === 1a. stage this block's train tile ===
        #[unroll]
        for l in 0..4u32 {
            let g = l * 4u32 + sgb;
            let mut v = Vector::<F, Const<4>>::from_int(0i64);
            #[unroll]
            for i in 0..4u32 {
                let r = base_j + g * 4u32 + i;
                let mut yv = zero_f;
                if r < rows_y {
                    if sk < cols {
                        yv = y[(r * cols + sk) as usize];
                    }
                }
                v[i as usize] = yv;
            }
            ys[(sk * 16u32 + g) as usize] = v;
        }
        sync_cube();

        // Prefetch this block's 8 train-column norms into registers NOW, so
        // the loads issue before the feature loop and their latency hides
        // under it — in the fold they sat on every candidate's dependent
        // chain (KNN-03 r7). Same values read in the same admission order, so
        // results are bit-identical.
        #[unroll]
        for b in 0..8u32 {
            let j = base_j + ux * 8u32 + b;
            let mut nv = zero_f;
            if j < rows_y {
                nv = ynorm[j as usize];
            }
            yn[b as usize] = nv;
        }

        // === 1b. dot-product accumulation: 3 VECTOR loads feed 32 FMAs ===
        let mut acc00 = F::from_int(0i64);
        let mut acc01 = F::from_int(0i64);
        let mut acc02 = F::from_int(0i64);
        let mut acc03 = F::from_int(0i64);
        let mut acc04 = F::from_int(0i64);
        let mut acc05 = F::from_int(0i64);
        let mut acc06 = F::from_int(0i64);
        let mut acc07 = F::from_int(0i64);
        let mut acc10 = F::from_int(0i64);
        let mut acc11 = F::from_int(0i64);
        let mut acc12 = F::from_int(0i64);
        let mut acc13 = F::from_int(0i64);
        let mut acc14 = F::from_int(0i64);
        let mut acc15 = F::from_int(0i64);
        let mut acc16 = F::from_int(0i64);
        let mut acc17 = F::from_int(0i64);
        let mut acc20 = F::from_int(0i64);
        let mut acc21 = F::from_int(0i64);
        let mut acc22 = F::from_int(0i64);
        let mut acc23 = F::from_int(0i64);
        let mut acc24 = F::from_int(0i64);
        let mut acc25 = F::from_int(0i64);
        let mut acc26 = F::from_int(0i64);
        let mut acc27 = F::from_int(0i64);
        let mut acc30 = F::from_int(0i64);
        let mut acc31 = F::from_int(0i64);
        let mut acc32 = F::from_int(0i64);
        let mut acc33 = F::from_int(0i64);
        let mut acc34 = F::from_int(0i64);
        let mut acc35 = F::from_int(0i64);
        let mut acc36 = F::from_int(0i64);
        let mut acc37 = F::from_int(0i64);

        let chunks = cols / 16u32;
        let mut h = 0u32;
        while h < chunks {
            let hbase = h * 16u32;
            #[unroll]
            for kki in 0..16u32 {
                let kk = hbase + kki;
                let xv = xs[(kk * 16u32 + uy) as usize];
                let yv0 = ys[(kk * 16u32 + 2u32 * ux) as usize];
                let yv1 = ys[(kk * 16u32 + 2u32 * ux + 1u32) as usize];
                let xa0 = xv[0usize];
                let xa1 = xv[1usize];
                let xa2 = xv[2usize];
                let xa3 = xv[3usize];
                let yb0 = yv0[0usize];
                let yb1 = yv0[1usize];
                let yb2 = yv0[2usize];
                let yb3 = yv0[3usize];
                let yb4 = yv1[0usize];
                let yb5 = yv1[1usize];
                let yb6 = yv1[2usize];
                let yb7 = yv1[3usize];
                acc00 += xa0 * yb0;
                acc01 += xa0 * yb1;
                acc02 += xa0 * yb2;
                acc03 += xa0 * yb3;
                acc04 += xa0 * yb4;
                acc05 += xa0 * yb5;
                acc06 += xa0 * yb6;
                acc07 += xa0 * yb7;
                acc10 += xa1 * yb0;
                acc11 += xa1 * yb1;
                acc12 += xa1 * yb2;
                acc13 += xa1 * yb3;
                acc14 += xa1 * yb4;
                acc15 += xa1 * yb5;
                acc16 += xa1 * yb6;
                acc17 += xa1 * yb7;
                acc20 += xa2 * yb0;
                acc21 += xa2 * yb1;
                acc22 += xa2 * yb2;
                acc23 += xa2 * yb3;
                acc24 += xa2 * yb4;
                acc25 += xa2 * yb5;
                acc26 += xa2 * yb6;
                acc27 += xa2 * yb7;
                acc30 += xa3 * yb0;
                acc31 += xa3 * yb1;
                acc32 += xa3 * yb2;
                acc33 += xa3 * yb3;
                acc34 += xa3 * yb4;
                acc35 += xa3 * yb5;
                acc36 += xa3 * yb6;
                acc37 += xa3 * yb7;
            }
            h += 1u32;
        }
        let mut kk = chunks * 16u32;
        while kk < cols {
            let xv = xs[(kk * 16u32 + uy) as usize];
            let yv0 = ys[(kk * 16u32 + 2u32 * ux) as usize];
            let yv1 = ys[(kk * 16u32 + 2u32 * ux + 1u32) as usize];
            let xa0 = xv[0usize];
            let xa1 = xv[1usize];
            let xa2 = xv[2usize];
            let xa3 = xv[3usize];
            let yb0 = yv0[0usize];
            let yb1 = yv0[1usize];
            let yb2 = yv0[2usize];
            let yb3 = yv0[3usize];
            let yb4 = yv1[0usize];
            let yb5 = yv1[1usize];
            let yb6 = yv1[2usize];
            let yb7 = yv1[3usize];
            acc00 += xa0 * yb0;
            acc01 += xa0 * yb1;
            acc02 += xa0 * yb2;
            acc03 += xa0 * yb3;
            acc04 += xa0 * yb4;
            acc05 += xa0 * yb5;
            acc06 += xa0 * yb6;
            acc07 += xa0 * yb7;
            acc10 += xa1 * yb0;
            acc11 += xa1 * yb1;
            acc12 += xa1 * yb2;
            acc13 += xa1 * yb3;
            acc14 += xa1 * yb4;
            acc15 += xa1 * yb5;
            acc16 += xa1 * yb6;
            acc17 += xa1 * yb7;
            acc20 += xa2 * yb0;
            acc21 += xa2 * yb1;
            acc22 += xa2 * yb2;
            acc23 += xa2 * yb3;
            acc24 += xa2 * yb4;
            acc25 += xa2 * yb5;
            acc26 += xa2 * yb6;
            acc27 += xa2 * yb7;
            acc30 += xa3 * yb0;
            acc31 += xa3 * yb1;
            acc32 += xa3 * yb2;
            acc33 += xa3 * yb3;
            acc34 += xa3 * yb4;
            acc35 += xa3 * yb5;
            acc36 += xa3 * yb6;
            acc37 += xa3 * yb7;
            kk += 1u32;
        }

        sync_cube();

        // === 2. fold: expand, clamp, admit (4 rows × 8 columns) ===
        blk[0usize] = acc00;
        blk[1usize] = acc01;
        blk[2usize] = acc02;
        blk[3usize] = acc03;
        blk[4usize] = acc04;
        blk[5usize] = acc05;
        blk[6usize] = acc06;
        blk[7usize] = acc07;
        blk[8usize] = acc10;
        blk[9usize] = acc11;
        blk[10usize] = acc12;
        blk[11usize] = acc13;
        blk[12usize] = acc14;
        blk[13usize] = acc15;
        blk[14usize] = acc16;
        blk[15usize] = acc17;
        blk[16usize] = acc20;
        blk[17usize] = acc21;
        blk[18usize] = acc22;
        blk[19usize] = acc23;
        blk[20usize] = acc24;
        blk[21usize] = acc25;
        blk[22usize] = acc26;
        blk[23usize] = acc27;
        blk[24usize] = acc30;
        blk[25usize] = acc31;
        blk[26usize] = acc32;
        blk[27usize] = acc33;
        blk[28usize] = acc34;
        blk[29usize] = acc35;
        blk[30usize] = acc36;
        blk[31usize] = acc37;

        #[unroll]
        for a in 0..4u32 {
            #[unroll]
            for b in 0..8u32 {
                let j = base_j + ux * 8u32 + b;
                if j < rows_y {
                    let dot = blk[(a * 8u32 + b) as usize];
                    let mut v = xn[a as usize] + yn[b as usize] - (dot + dot);
                    if v < zero_f {
                        v = zero_f;
                    }
                    let cnt = lcnt[a as usize];
                    let mut admit: u32 = 0u32;
                    if cnt < k {
                        admit = 1u32;
                    } else if v < lworst_val[a as usize] {
                        admit = 1u32;
                    } else if v == lworst_val[a as usize] {
                        if j < lworst_idx[a as usize] {
                            admit = 1u32;
                        }
                    }

                    if admit == 1u32 {
                        let mut cav = v;
                        let mut cai = j;
                        let mut fs = 0u32;
                        while fs < cnt {
                            let jv = lval[(a * 16u32 + fs) as usize];
                            let ji = lidx[(a * 16u32 + fs) as usize];
                            let mut swap: u32 = 0u32;
                            if cav < jv {
                                swap = 1u32;
                            } else if cav == jv {
                                if cai < ji {
                                    swap = 1u32;
                                }
                            }
                            if swap == 1u32 {
                                lval[(a * 16u32 + fs) as usize] = cav;
                                lidx[(a * 16u32 + fs) as usize] = cai;
                                cav = jv;
                                cai = ji;
                            }
                            fs += 1u32;
                        }
                        let mut ncnt = cnt;
                        if cnt < k {
                            lval[(a * 16u32 + cnt) as usize] = cav;
                            lidx[(a * 16u32 + cnt) as usize] = cai;
                            ncnt = cnt + 1u32;
                            lcnt[a as usize] = ncnt;
                        }
                        lworst_val[a as usize] = lval[(a * 16u32 + ncnt - 1u32) as usize];
                        lworst_idx[a as usize] = lidx[(a * 16u32 + ncnt - 1u32) as usize];
                    }
                }
            }
        }

        b0 += 64u32;
    }

    // ---- DUMP + 8-WAY SERIAL MERGE (KNN-03 r5) ----
    // The tree merge cost ~7 cube barriers per emitted pair (4·k rounds ×
    // post/tree/advance) — ~280 barriers at k=10, a per-launch fixed cost that
    // dominated small-`n_train` launches (measured: the n_train sweep's
    // constant ~6-8 ms, and a k-sweep at fixed shape scaling with k). Instead:
    // each unit dumps its sorted list `am` into the dead staging tiles (values
    // in `xs`, F-encoded indices in `ys` — 128 units × 4 vectors fills each
    // tile EXACTLY), then ONE unit per query row merges the row's 8 sorted
    // lists serially (k rounds × 8 head compares — trivial work). TWO barriers
    // per row-quartet, 8 total.
    //
    // Padding pairs use (max_value, max-encodable-index) so they lose every
    // comparison against a real pair; a padding pair can only be emitted if a
    // row's union holds fewer than k pairs, which the host's `k <= n_train`
    // validation rules out (every column of the row lives in exactly one
    // unit's candidate stream, and each unit keeps its k best). The emitted
    // sequence is the k smallest of the union in ascending total pair order —
    // identical to the tree merge's, hence bitwise-identical output.
    let flat = uy * 8u32 + ux;
    let big = F::max_value();
    let pad_idx = F::cast_from(16777215u32);
    let mut am = 0u32;
    while am < 4u32 {
        let orow = base_i + uy * 4u32 + am;

        // 1. dump list `am`, padded to 16 slots, as 4 value + 4 index vectors.
        #[unroll]
        for q in 0..4u32 {
            let mut vv = Vector::<F, Const<4>>::from_int(0i64);
            let mut vi = Vector::<F, Const<4>>::from_int(0i64);
            #[unroll]
            for c in 0..4u32 {
                let i = q * 4u32 + c;
                let mut pv = big;
                let mut pi = pad_idx;
                if i < lcnt[am as usize] {
                    pv = lval[(am * 16u32 + i) as usize];
                    pi = F::cast_from(lidx[(am * 16u32 + i) as usize]);
                }
                vv[c as usize] = pv;
                vi[c as usize] = pi;
            }
            xs[(flat * 4u32 + q) as usize] = vv;
            ys[(flat * 4u32 + q) as usize] = vi;
        }
        sync_cube();

        // 2. one unit per row: 8-way merge of the sorted k-lists.
        if ux == 0u32 {
            if orow < rows_x {
                // Per-list head cursors + the currently loaded 4-entry vector
                // (entries h_b/4*4 .. +4 of list b), unpacked to local arrays.
                let mut hh = Array::<u32>::new(8usize);
                let mut ql = Array::<u32>::new(8usize);
                let mut bv = Array::<F>::new(32usize);
                let mut bi = Array::<F>::new(32usize);
                #[unroll]
                for b in 0..8u32 {
                    hh[b as usize] = 0u32;
                    ql[b as usize] = 0u32;
                    let vv = xs[((uy * 8u32 + b) * 4u32) as usize];
                    let vi = ys[((uy * 8u32 + b) * 4u32) as usize];
                    #[unroll]
                    for c in 0..4u32 {
                        bv[(b * 4u32 + c) as usize] = vv[c as usize];
                        bi[(b * 4u32 + c) as usize] = vi[c as usize];
                    }
                }
                let mut r = 0u32;
                while r < k {
                    let mut bestb: u32 = 0u32;
                    let mut bestv = big;
                    let mut besti = pad_idx;
                    let mut first: u32 = 1u32;
                    #[unroll]
                    for b in 0..8u32 {
                        let h = hh[b as usize];
                        // Heads advance at most k-1 slots before the loop
                        // exits, and lists are padded to 16 slots, so `q` is
                        // always a valid vector of list `b`.
                        let q = h / 4u32;
                        if q != ql[b as usize] {
                            let vv = xs[((uy * 8u32 + b) * 4u32 + q) as usize];
                            let vi = ys[((uy * 8u32 + b) * 4u32 + q) as usize];
                            #[unroll]
                            for c in 0..4u32 {
                                bv[(b * 4u32 + c) as usize] = vv[c as usize];
                                bi[(b * 4u32 + c) as usize] = vi[c as usize];
                            }
                            ql[b as usize] = q;
                        }
                        let cv = bv[(b * 4u32 + (h % 4u32)) as usize];
                        let ci = bi[(b * 4u32 + (h % 4u32)) as usize];
                        let mut better: u32 = 0u32;
                        if first == 1u32 {
                            better = 1u32;
                        } else if cv < bestv {
                            better = 1u32;
                        } else if cv == bestv {
                            if ci < besti {
                                better = 1u32;
                            }
                        }
                        if better == 1u32 {
                            bestb = b;
                            bestv = cv;
                            besti = ci;
                            first = 0u32;
                        }
                    }
                    let slot = orow * (strips * k) + strip * k + r;
                    out_val[slot as usize] = bestv;
                    out_idx[slot as usize] = u32::cast_from(besti);
                    hh[bestb as usize] += 1u32;
                    r += 1u32;
                }
            }
        }
        // Barrier before the next quartet's dump overwrites the tiles.
        sync_cube();
        am += 1u32;
    }
}

/// Per-row squared norm (`out[r] = Σ_c x[r,c]²`) — the norm feeder for
/// [`euclidean_topk_fused_dot`].
///
/// One unit per row, serial bounded feature loop — the GATHER shape
/// (`knn_regress_mean` / `linear_predict` precedent): no `SharedMemory`, no
/// plane ops, portable to every backend including cpu-MLIR.
///
/// ## Why not `prims::reduce::row_reduce`
/// `row_reduce` materializes the WHOLE input on the host and then performs an
/// upload → launch → readback round-trip PER ROW: at the KNN scales
/// (`n_train` up to hundreds of thousands of rows) that is hundreds of
/// thousands of host round-trips — measured as a >100× pathology locally and
/// a wall-clock timeout on the T4 (it was the actual culprit behind numbers
/// initially attributed to the dot kernel). For the small `cols` this feeder
/// serves (`<= 32`), a serial per-row loop is one coalesced-enough kernel
/// launch and is exact-integer-preserving for the quantized test fixtures.
#[cube(launch)]
pub fn row_sumsq<F: Float + CubeElement>(
    x: &Array<F>,
    out: &mut Array<F>,
    rows: u32,
    cols: u32,
) {
    let r = ABSOLUTE_POS_X;
    if r < rows {
        let base = r * cols;
        let mut acc = F::from_int(0i64);
        let mut c = 0u32;
        while c < cols {
            let v = x[(base + c) as usize];
            acc += v * v;
            c += 1u32;
        }
        out[r as usize] = acc;
    }
}

/// Query rows ONE unit owns in the cpu row-scan kernels — the SIMD lane count
/// of [`euclidean_topk_rows_cpu_vec`] and the register-blocking factor of
/// [`euclidean_topk_rows_cpu`].
///
/// 16 is one AVX-512 `f32` register, and it is also the reuse factor of the
/// training stream: every training row a unit loads feeds this many query rows.
pub const CPU_QUERY_TILE: u32 = 16;

/// Largest `n_features` the cpu row-scan kernels cache in their local
/// transposed query tile (`CPU_QUERY_TILE * CPU_MAX_COLS` `F` of stack).
pub const CPU_MAX_COLS: u32 = 32;

/// Local top-k list stride of the cpu row-scan kernels — the same comptime cap
/// the shared-memory fused kernels use (`FUSED_K_CAP`).
pub const CPU_K_CAP: u32 = 16;

/// CPU-SHAPED fused k-nearest search (KNN-04): one unit per BLOCK of
/// `CPU_QUERY_TILE` query rows, a single straight-line scan of the training
/// set, and NO shared memory / NO barriers anywhere.
///
/// ## Why a separate kernel instead of the GPU fused family
/// `cubecl-cpu` executes a launch by spawning ONE OS THREAD PER UNIT
/// (`cube_dim.num_elems()` threads) and having each thread run the whole
/// `cube_count` grid loop; parallelism is the CUBE DIM, and the cube grid is
/// the SERIAL loop. Two consequences invert every GPU design choice:
///
/// 1. The GPU kernels launch `CubeDim { x: 16, y: 16 }` = 256 units, i.e. 256
///    OS threads on a machine with a dozen cores — a 16× oversubscription.
/// 2. Their `sync_cube()` barriers are implemented as a SPIN barrier over all
///    256 threads (`cubecl_cpu::compute::compute_task::sync_cube`). With more
///    threads than cores, every barrier makes the runnable threads burn their
///    whole scheduling quantum spinning for descheduled peers, and the fused
///    dot kernel executes several barriers per 64-column block — hundreds of
///    thousands of 256-way spin rendezvous per predict. MEASURED on a 16-core
///    Zen5: the two-kernel/fused paths did not finish the smallest ladder rung
///    (`n_train = 2_000`, `n_query = 512`) in 10 minutes; this kernel runs the
///    whole 200k rung in seconds.
///
/// This kernel therefore uses the shape a CPU actually wants: the host
/// launches exactly `available_parallelism` units, each unit owns a disjoint
/// block of query rows, and the only synchronization in the whole predict is
/// the launch's own join.
///
/// ## Structure
/// - The unit's `CPU_QUERY_TILE` query rows are staged ONCE into a local
///   TRANSPOSED cache `xt[c * CPU_QUERY_TILE + a]`, so the inner feature step
///   reads the tile's values for feature `c` from CONSECUTIVE stack slots
///   instead of `cols` apart.
/// - The train scan is a flat `t < rows_y` loop. Per train row, the feature
///   loop broadcasts `y[t, c]` against those staged values — `CPU_QUERY_TILE`
///   FMAs per global load, which is what keeps the scan off the memory bus.
/// - Admission into the per-row sorted lists is the BYTE-FOR-BYTE admission
///   rule of [`euclidean_topk_fused_dot`] (`d² = ‖x‖² + ‖y‖² − 2·x·y` with the
///   cancellation clamp, then a `(value, index)`-ordered insertion guarded by a
///   compare against the list's current worst kept pair).
///
/// ## Result contract
/// The emitted top-k is the `k` smallest `(value, index)` pairs of the FULL
/// candidate set in ascending pair order — the same total order the GPU fused
/// kernels' k-round merge emits, and the same lowest-index tie-break sklearn
/// uses. The accumulation order (`acc += x[r,c] * y[t,c]` over ascending `c`)
/// is identical too, so distances are BITWISE identical to
/// [`euclidean_topk_fused_dot`]'s and `knn_test.rs` gates exactly that.
///
/// - `xnorm` / `ynorm` are the per-row squared norms ([`row_sumsq`]).
/// - `out_val` / `out_idx` are the plain `rows_x × k` results (this kernel has
///   no strip split — a CPU launch has no occupancy tail to fill).
/// - `cols <= CPU_MAX_COLS` and `k <= CPU_K_CAP` are validated host-side.
///
/// cpu-MLIR contract: no `SharedMemory`, no `sync_cube`, no mutable `bool`, no
/// plane ops — the `knn_regress_mean` / `row_sumsq` GATHER shape.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn euclidean_topk_rows_cpu<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xnorm: &Array<F>,
    ynorm: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    k: u32,
) {
    let q0 = ABSOLUTE_POS_X * 16u32;
    if q0 < rows_x {
        let zero_f = F::from_int(0i64);

        // Transposed query-row cache: `xt[c * 16 + a]` is feature `c` of the
        // unit's query row `a`.
        let mut xt = Array::<F>::new(512usize);
        let mut acc = Array::<F>::new(16usize);
        // Sorted top-k lists, stride `CPU_K_CAP`.
        let mut lval = Array::<F>::new(256usize);
        let mut lidx = Array::<u32>::new(256usize);
        let mut lcnt = Array::<u32>::new(16usize);
        let mut lworst_val = Array::<F>::new(16usize);
        let mut lworst_idx = Array::<u32>::new(16usize);
        let mut xn = Array::<F>::new(16usize);

        // --- stage: norms, empty lists, transposed feature cache ---
        #[unroll]
        for a in 0..16u32 {
            let r = q0 + a;
            let mut nv = zero_f;
            if r < rows_x {
                nv = xnorm[r as usize];
            }
            xn[a as usize] = nv;
            lcnt[a as usize] = 0u32;
            lworst_val[a as usize] = zero_f;
            lworst_idx[a as usize] = 0u32;

            let mut c = 0u32;
            while c < cols {
                // An over-provisioned lane (`r >= rows_x`) stages zeros and
                // simply never reaches the emit guard below.
                let mut v = zero_f;
                if r < rows_x {
                    v = x[(r * cols + c) as usize];
                }
                xt[(c * 16u32 + a) as usize] = v;
                c += 1u32;
            }
        }

        // --- scan: every training row, once ---
        let mut t = 0u32;
        while t < rows_y {
            let ybase = t * cols;
            #[unroll]
            for a in 0..16u32 {
                acc[a as usize] = zero_f;
            }

            let mut c = 0u32;
            while c < cols {
                let yv = y[(ybase + c) as usize];
                let xb = c * 16u32;
                #[unroll]
                for a in 0..16u32 {
                    acc[a as usize] += xt[(xb + a) as usize] * yv;
                }
                c += 1u32;
            }

            let yn = ynorm[t as usize];
            #[unroll]
            for a in 0..16u32 {
                // d² = ‖x‖² + ‖y‖² − 2·x·y, clamped at 0 so cancellation can
                // never admit a negative distance (the fused-dot precedent).
                let dot = acc[a as usize];
                let mut v = xn[a as usize] + yn - (dot + dot);
                if v < zero_f {
                    v = zero_f;
                }
                let cnt = lcnt[a as usize];
                let mut admit: u32 = 0u32;
                if cnt < k {
                    admit = 1u32;
                } else if v < lworst_val[a as usize] {
                    admit = 1u32;
                } else if v == lworst_val[a as usize] {
                    if t < lworst_idx[a as usize] {
                        admit = 1u32;
                    }
                }

                if admit == 1u32 {
                    let mut cav = v;
                    let mut cai = t;
                    let mut fs = 0u32;
                    while fs < cnt {
                        let jv = lval[(a * 16u32 + fs) as usize];
                        let ji = lidx[(a * 16u32 + fs) as usize];
                        let mut swap: u32 = 0u32;
                        if cav < jv {
                            swap = 1u32;
                        } else if cav == jv {
                            if cai < ji {
                                swap = 1u32;
                            }
                        }
                        if swap == 1u32 {
                            lval[(a * 16u32 + fs) as usize] = cav;
                            lidx[(a * 16u32 + fs) as usize] = cai;
                            cav = jv;
                            cai = ji;
                        }
                        fs += 1u32;
                    }
                    let mut ncnt = cnt;
                    if cnt < k {
                        lval[(a * 16u32 + cnt) as usize] = cav;
                        lidx[(a * 16u32 + cnt) as usize] = cai;
                        ncnt = cnt + 1u32;
                        lcnt[a as usize] = ncnt;
                    }
                    lworst_val[a as usize] = lval[(a * 16u32 + ncnt - 1u32) as usize];
                    lworst_idx[a as usize] = lidx[(a * 16u32 + ncnt - 1u32) as usize];
                }
            }

            t += 1u32;
        }

        // --- emit: the lists are already in ascending pair order ---
        #[unroll]
        for a in 0..16u32 {
            let r = q0 + a;
            if r < rows_x {
                let mut j = 0u32;
                while j < k {
                    out_val[(r * k + j) as usize] = lval[(a * 16u32 + j) as usize];
                    out_idx[(r * k + j) as usize] = lidx[(a * 16u32 + j) as usize];
                    j += 1u32;
                }
            }
        }
    }
}

/// SIMD form of [`euclidean_topk_rows_cpu`] (KNN-04 second pass) — same shape,
/// same arithmetic, but the query tile IS the vector lane axis.
///
/// ## Why (the op-count budget, and why it is the budget)
/// `cubecl-cpu` hands its MLIR module to `ExecutionEngine::new(module, 0, ..)`
/// — LLVM optimization level **0**. There is no vectorizer, no unroller and no
/// meaningful register allocation downstream: every emitted IR op costs real
/// instructions, and every SSA value round-trips through a stack slot. So the
/// only lever on a cpu kernel's speed is the NUMBER OF IR OPS PER PAIR, and
/// measurement matches — the scalar kernel's inner step (`CPU_QUERY_TILE`
/// stack loads + 1 global load + `CPU_QUERY_TILE` scalar FMAs plus loop
/// bookkeeping) runs at ~22 cycles per feature iteration on a 16-core Zen5,
/// about 1% of the machine's FMA peak.
///
/// This kernel collapses that step: the tile is `Vector<F, CPU_QUERY_TILE>` —
/// ONE vector per FEATURE, holding that feature for every query row of the
/// tile — so the step becomes 1 vector load + 1 broadcast + 1 vector FMA
/// regardless of the tile width, and `cubecl-cpu` lowers those to genuine MLIR
/// `vector<16xf32>` ops (`compiler::visitor::item`). The feature loop is
/// additionally walked in comptime-unrolled chunks so the loop's own
/// bookkeeping is paid once per chunk instead of once per feature.
///
/// ## Bitwise contract
/// Lane `a` accumulates `acc[a] += xt[c][a] * y[t, c]` over ascending `c` —
/// the identical scalar sequence [`euclidean_topk_rows_cpu`] runs in its
/// `acc[a]` slot, just issued a tile at a time; the expansion is likewise
/// `(‖x‖² + ‖y‖²) − 2·x·y` with the same association. The clamp, admission
/// rule and emit are copied verbatim. Results are therefore BITWISE identical
/// to the scalar kernel's, which is what `knn_test.rs` asserts.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn euclidean_topk_rows_cpu_vec<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    xnorm: &Array<F>,
    ynorm: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    k: u32,
) {
    let q0 = ABSOLUTE_POS_X * 16u32;
    if q0 < rows_x {
        let zero_f = F::from_int(0i64);

        // One 16-lane vector PER FEATURE: `xt[c]` lane `a` is feature `c` of
        // the unit's query row `a`. `CPU_MAX_COLS` vectors = 2 KiB at f32.
        let mut xt = Array::<Vector<F, Const<16>>>::new(32usize);
        let mut lval = Array::<F>::new(256usize);
        let mut lidx = Array::<u32>::new(256usize);
        let mut lcnt = Array::<u32>::new(16usize);
        let mut lworst_val = Array::<F>::new(16usize);
        let mut lworst_idx = Array::<u32>::new(16usize);

        // The query norms live as a VECTOR so the expansion below is one
        // vector add/sub for the whole tile.
        let mut xnv = Vector::<F, Const<16>>::from_int(0i64);
        #[unroll]
        for a in 0..16u32 {
            let r = q0 + a;
            let mut nv = zero_f;
            if r < rows_x {
                nv = xnorm[r as usize];
            }
            xnv[a as usize] = nv;
            lcnt[a as usize] = 0u32;
            lworst_val[a as usize] = zero_f;
            lworst_idx[a as usize] = 0u32;
        }
        let mut c = 0u32;
        while c < cols {
            let mut v = Vector::<F, Const<16>>::from_int(0i64);
            #[unroll]
            for a in 0..16u32 {
                let r = q0 + a;
                // An over-provisioned lane (`r >= rows_x`) stages zeros and
                // never reaches the emit guard below.
                let mut xv = zero_f;
                if r < rows_x {
                    xv = x[(r * cols + c) as usize];
                }
                v[a as usize] = xv;
            }
            xt[c as usize] = v;
            c += 1u32;
        }

        let chunks = cols / 8u32;
        let tail0 = chunks * 8u32;
        let mut t = 0u32;
        while t < rows_y {
            // FOUR training rows per pass, each with its OWN accumulator. At
            // -O0 an accumulator round-trips through its stack slot on every
            // update, so a single chain stalls on store-to-load forwarding;
            // four independent chains overlap those stalls, and the tile load
            // `xt[cc]` is paid once per four rows instead of once per row.
            // Rows past the end are CLAMPED to the last row so the inner loop
            // stays branch-free — their results are discarded by the
            // `tt < rows_y` guard at admission.
            let mut yb = Array::<u32>::new(4usize);
            #[unroll]
            for u in 0..4u32 {
                let mut tt = t + u;
                if tt >= rows_y {
                    tt = rows_y - 1u32;
                }
                yb[u as usize] = tt * cols;
            }
            let mut accs = Array::<Vector<F, Const<16>>>::new(4usize);
            #[unroll]
            for u in 0..4u32 {
                accs[u as usize] = Vector::<F, Const<16>>::from_int(0i64);
            }

            // Comptime-unrolled 8-feature chunks, then the sub-8 tail — the
            // loop bookkeeping is what the unroll is buying, not the FMAs.
            let mut h = 0u32;
            while h < chunks {
                let cb = h * 8u32;
                #[unroll]
                for i in 0..8u32 {
                    let cc = cb + i;
                    let xv = xt[cc as usize];
                    #[unroll]
                    for u in 0..4u32 {
                        accs[u as usize] +=
                            xv * Vector::<F, Const<16>>::new(y[(yb[u as usize] + cc) as usize]);
                    }
                }
                h += 1u32;
            }
            let mut ct = tail0;
            while ct < cols {
                let xv = xt[ct as usize];
                #[unroll]
                for u in 0..4u32 {
                    accs[u as usize] +=
                        xv * Vector::<F, Const<16>>::new(y[(yb[u as usize] + ct) as usize]);
                }
                ct += 1u32;
            }

            #[unroll]
            for u in 0..4u32 {
                let tt = t + u;
                if tt < rows_y {
                    // d² = (‖x‖² + ‖y‖²) − 2·x·y for the whole tile at once;
                    // the same association the scalar kernel evaluates per
                    // lane.
                    let acc = accs[u as usize];
                    let dv = (xnv + Vector::<F, Const<16>>::new(ynorm[tt as usize])) - (acc + acc);
                    admit_tile::<F>(
                        &mut lval,
                        &mut lidx,
                        &mut lcnt,
                        &mut lworst_val,
                        &mut lworst_idx,
                        dv,
                        tt,
                        k,
                    );
                }
            }

            t += 4u32;
        }

        #[unroll]
        for a in 0..16u32 {
            let r = q0 + a;
            if r < rows_x {
                let mut j = 0u32;
                while j < k {
                    out_val[(r * k + j) as usize] = lval[(a * 16u32 + j) as usize];
                    out_idx[(r * k + j) as usize] = lidx[(a * 16u32 + j) as usize];
                    j += 1u32;
                }
            }
        }
    }
}

/// The per-train-row admission step of the cpu SIMD kernels, factored out
/// because their 4-row inner loops apply it four times.
///
/// `dv` is the tile's squared distances against training row `t` — formed by
/// the expansion in [`euclidean_topk_rows_cpu_vec`] and directly by
/// [`euclidean_topk_rows_cpu_direct`]. This function clamps at 0 against
/// cancellation and folds each lane into its sorted top-k list under the
/// `(value, index)` order — BYTE-FOR-BYTE the admission rule of
/// [`euclidean_topk_fused_dot`].
#[cube]
#[allow(clippy::too_many_arguments)]
fn admit_tile<F: Float + CubeElement>(
    lval: &mut Array<F>,
    lidx: &mut Array<u32>,
    lcnt: &mut Array<u32>,
    lworst_val: &mut Array<F>,
    lworst_idx: &mut Array<u32>,
    dv: Vector<F, Const<16>>,
    t: u32,
    k: u32,
) {
    let zero_f = F::from_int(0i64);
    #[unroll]
    for a in 0..16u32 {
        let mut v = dv[a as usize];
        if v < zero_f {
            v = zero_f;
        }
        let cnt = lcnt[a as usize];
        let mut admit: u32 = 0u32;
        if cnt < k {
            admit = 1u32;
        } else if v < lworst_val[a as usize] {
            admit = 1u32;
        } else if v == lworst_val[a as usize] {
            if t < lworst_idx[a as usize] {
                admit = 1u32;
            }
        }

        if admit == 1u32 {
            let mut cav = v;
            let mut cai = t;
            let mut fs = 0u32;
            while fs < cnt {
                let jv = lval[(a * 16u32 + fs) as usize];
                let ji = lidx[(a * 16u32 + fs) as usize];
                let mut swap: u32 = 0u32;
                if cav < jv {
                    swap = 1u32;
                } else if cav == jv {
                    if cai < ji {
                        swap = 1u32;
                    }
                }
                if swap == 1u32 {
                    lval[(a * 16u32 + fs) as usize] = cav;
                    lidx[(a * 16u32 + fs) as usize] = cai;
                    cav = jv;
                    cai = ji;
                }
                fs += 1u32;
            }
            let mut ncnt = cnt;
            if cnt < k {
                lval[(a * 16u32 + cnt) as usize] = cav;
                lidx[(a * 16u32 + cnt) as usize] = cai;
                ncnt = cnt + 1u32;
                lcnt[a as usize] = ncnt;
            }
            lworst_val[a as usize] = lval[(a * 16u32 + ncnt - 1u32) as usize];
            lworst_idx[a as usize] = lidx[(a * 16u32 + ncnt - 1u32) as usize];
        }
    }
}

/// CANCELLATION-FREE form of [`euclidean_topk_rows_cpu_vec`]: accumulates
/// `Σ_c (x[r,c] − y[t,c])²` directly instead of expanding
/// `‖x‖² + ‖y‖² − 2·x·y`.
///
/// ## Why it exists
/// The expansion is the fast form — one FMA per feature, and the norms are
/// hoisted out of the scan — but it is a DIFFERENCE OF LARGE NUMBERS: when two
/// training points sit within a few f32 ULPs of the same distance from a query,
/// the expansion can order them differently from the exact answer. Measured at
/// the `n_train = 100_000, d = 32, k = 10` rung: 1 query row in 10_000 picked a
/// 10th neighbour whose exact-f64 distance was 2e-6 (0.24 f32 ULP) further than
/// sklearn's pick, moving that row's prediction by 0.105. Reproducing the same
/// arithmetic in plain numpy gives mlrs's answer, so this is a property of the
/// expansion in f32 and not of any one implementation — cuML's fused kernel and
/// sklearn's own `ArgKmin` share the hazard.
///
/// This kernel spends one extra vector op per feature (subtract, then a squaring
/// FMA) to remove it. Select it with `MLRS_KNN_DIRECT=1`, the same escape hatch
/// the GPU dot path already honours.
///
/// Structure, blocking, admission, tie-break and output layout are exactly
/// [`euclidean_topk_rows_cpu_vec`]'s; only the inner arithmetic differs, and it
/// takes no norm operands.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn euclidean_topk_rows_cpu_direct<F: Float + CubeElement>(
    x: &Array<F>,
    y: &Array<F>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows_x: u32,
    rows_y: u32,
    cols: u32,
    k: u32,
) {
    let q0 = ABSOLUTE_POS_X * 16u32;
    if q0 < rows_x {
        let zero_f = F::from_int(0i64);

        let mut xt = Array::<Vector<F, Const<16>>>::new(32usize);
        let mut lval = Array::<F>::new(256usize);
        let mut lidx = Array::<u32>::new(256usize);
        let mut lcnt = Array::<u32>::new(16usize);
        let mut lworst_val = Array::<F>::new(16usize);
        let mut lworst_idx = Array::<u32>::new(16usize);

        #[unroll]
        for a in 0..16u32 {
            lcnt[a as usize] = 0u32;
            lworst_val[a as usize] = zero_f;
            lworst_idx[a as usize] = 0u32;
        }
        let mut c = 0u32;
        while c < cols {
            let mut v = Vector::<F, Const<16>>::from_int(0i64);
            #[unroll]
            for a in 0..16u32 {
                let r = q0 + a;
                // An over-provisioned lane (`r >= rows_x`) stages zeros and
                // never reaches the emit guard below.
                let mut xv = zero_f;
                if r < rows_x {
                    xv = x[(r * cols + c) as usize];
                }
                v[a as usize] = xv;
            }
            xt[c as usize] = v;
            c += 1u32;
        }

        let chunks = cols / 8u32;
        let tail0 = chunks * 8u32;
        let mut t = 0u32;
        while t < rows_y {
            let mut yb = Array::<u32>::new(4usize);
            #[unroll]
            for u in 0..4u32 {
                let mut tt = t + u;
                if tt >= rows_y {
                    tt = rows_y - 1u32;
                }
                yb[u as usize] = tt * cols;
            }
            let mut accs = Array::<Vector<F, Const<16>>>::new(4usize);
            #[unroll]
            for u in 0..4u32 {
                accs[u as usize] = Vector::<F, Const<16>>::from_int(0i64);
            }

            let mut h = 0u32;
            while h < chunks {
                let cb = h * 8u32;
                #[unroll]
                for i in 0..8u32 {
                    let cc = cb + i;
                    let xv = xt[cc as usize];
                    #[unroll]
                    for u in 0..4u32 {
                        let dif =
                            xv - Vector::<F, Const<16>>::new(y[(yb[u as usize] + cc) as usize]);
                        accs[u as usize] += dif * dif;
                    }
                }
                h += 1u32;
            }
            let mut ct = tail0;
            while ct < cols {
                let xv = xt[ct as usize];
                #[unroll]
                for u in 0..4u32 {
                    let dif = xv - Vector::<F, Const<16>>::new(y[(yb[u as usize] + ct) as usize]);
                    accs[u as usize] += dif * dif;
                }
                ct += 1u32;
            }

            #[unroll]
            for u in 0..4u32 {
                let tt = t + u;
                if tt < rows_y {
                    admit_tile::<F>(
                        &mut lval,
                        &mut lidx,
                        &mut lcnt,
                        &mut lworst_val,
                        &mut lworst_idx,
                        accs[u as usize],
                        tt,
                        k,
                    );
                }
            }

            t += 4u32;
        }

        #[unroll]
        for a in 0..16u32 {
            let r = q0 + a;
            if r < rows_x {
                let mut j = 0u32;
                while j < k {
                    out_val[(r * k + j) as usize] = lval[(a * 16u32 + j) as usize];
                    out_idx[(r * k + j) as usize] = lidx[(a * 16u32 + j) as usize];
                    j += 1u32;
                }
            }
        }
    }
}
