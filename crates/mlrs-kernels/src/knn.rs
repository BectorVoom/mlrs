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
/// defence is split in two: an `if t < n_train` guard means a corrupted or
/// oversized index CONTRIBUTES NOTHING instead of performing an out-of-bounds
/// device read, AND the row records that it happened in `oob[q]` so the host can
/// still raise the typed error.
///
/// The guard alone is not enough. Skipping a neighbor while still dividing by
/// the full `k` yields a numerically PLAUSIBLE but wrong prediction (the mean of
/// `k-1` targets over `k`), which is a worse failure mode than the hard error it
/// replaced: a corruption in the selection kernels would reach the user as a
/// slightly-off regression value with nothing to flag it. `oob` restores
/// "loud", at the cost of one `u32` per query row.
///
/// The producing selection kernels only ever emit column indices in
/// `[0, n_train)`, so this stays unreachable in practice — it exists so a future
/// regression there fails loudly instead of silently.
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
#[allow(clippy::too_many_arguments)]
pub fn knn_regress_mean<F: Float + CubeElement>(
    y_train: &Array<F>,
    idx: &Array<u32>,
    out: &mut Array<F>,
    oob: &mut Array<u32>,
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
        // Set to 1 by the `else` arm below when this row saw an index outside
        // `[0, n_train)` — see the `oob[q]` write at the end of the row.
        let mut bad = 0u32;
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
            } else {
                bad = 1u32;
            }
            j += 1u32;
        }
        // `k >= 1` is validated host-side before launch, so the divide is safe.
        out[q as usize] = (acc + comp) / F::cast_from(k);
        // Row-private slot (unit `q` is the only writer), so reporting the
        // corruption needs no atomic and no barrier. The host reads this back
        // and raises the typed error the pre-KNN-01 host gather loop raised —
        // see `prims::knn::knn_regress_mean_gather`.
        oob[q as usize] = bad;
    }
}

/// GENERAL neighbor-target gather (KNN-REG-PARAMS): `weights='distance'` and/or
/// a MULTI-OUTPUT target, in one kernel.
///
/// `out[q, o] = (Σ_j w_qj · y_train[idx[q,k+j], o]) / (Σ_j w_qj)` where `w` is
/// `1` for `weighted == 0` (the uniform mean) and `1/dist[q, j]` for
/// `weighted == 1`.
///
/// - `y_train` is the fitted `n_train × n_outputs` row-major target matrix; a
///   single-output target is the `n_outputs == 1` case.
/// - `idx` / `dist` are the `n_query × k` row-major neighbor indices and their
///   TRUE (already sqrt'd / already-metric) distances from the selection stage.
///   `dist` is read only when `weighted == 1`, but it is a required binding —
///   an unbound `Array` argument is not expressible, so the uniform path passes
///   the same buffer and simply never reads it.
/// - `out` is `n_query × n_outputs`; `oob` is the per-row corruption flag with
///   exactly the semantics of [`knn_regress_mean`]'s.
///
/// ## Zero distances (sklearn parity, `_get_weights`)
/// `1/0` is `inf`, and sklearn does NOT let that propagate: when a query row has
/// ANY neighbor at distance 0, sklearn gives those neighbors weight `1` and
/// EVERY other neighbor in that row weight `0`, so the prediction is the plain
/// mean of the coincident training points. Reproducing that exactly needs a
/// pre-pass over the row (`any_zero`) before the weights can be formed, which is
/// why the accumulation below is a second loop rather than one fused pass.
/// Getting this wrong is silent: `inf/inf` is NaN, so a duplicated training
/// point would turn a correct prediction into NaN with nothing to flag it.
///
/// ## Compensated (Neumaier) accumulation
/// Both the numerator and the weight sum use the same Neumaier compensation as
/// [`knn_regress_mean`], for the same reason: a naive `f32` running sum grows as
/// `k · eps` and can exceed the project's 1e-5 sklearn-agreement bar on its own
/// at large `n_neighbors` with large-magnitude targets. `weights='distance'`
/// makes it strictly worse than the uniform case — `1/d` spans orders of
/// magnitude within a single row, which is precisely the mixed-magnitude regime
/// a naive sum loses the most bits in. The branchless magnitude test is the
/// cubecl-cpu-MLIR-safe spelling (no `abs` intrinsic on a mutable, no early
/// exit).
///
/// Launched 1D over `n_query` (one unit per query row); the kernel bounds-checks
/// `q < n_query`, so an over-provisioned launch writes nothing. GATHER-only — no
/// `SharedMemory`, no mutable `bool`, no plane ops — so the `cubecl-cpu` MLIR
/// lowering accepts it and no backend feature gate is needed (D-13).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn knn_regress_gather<F: Float + CubeElement>(
    y_train: &Array<F>,
    idx: &Array<u32>,
    dist: &Array<F>,
    out: &mut Array<F>,
    oob: &mut Array<u32>,
    n_query: u32,
    k: u32,
    n_train: u32,
    n_outputs: u32,
    weighted: u32,
) {
    let q = ABSOLUTE_POS_X;
    if q < n_query {
        let base = q * k;
        let zero = F::new(0.0_f32);
        let one = F::new(1.0_f32);
        let mut bad = 0u32;

        // --- Pre-pass: does this row contain a coincident neighbor? sklearn's
        //     `_get_weights` switches the WHOLE row to an indicator weighting
        //     when it does (see the doc note), so this must be known before any
        //     weight is formed. Cheap: k <= a few hundred, and it is skipped
        //     entirely on the uniform path. ---
        //
        // The flag is carried as the row's MINIMUM DISTANCE (`F`), not as a
        // `u32` counter: a mutable `u32` local compared against a literal
        // (`if flag > 0u32`) is an E0282 inference failure inside `#[cube]`,
        // while the mutable-`F`-vs-immutable-`F` compare below is the
        // `chebyshev_dist` running-maximum idiom the macro is known to lower.
        // `dist` is a bound argument on both paths so the read is always in
        // range; the uniform path simply never consults `dmin`.
        let mut dmin = dist[base as usize];
        let mut z = 1u32;
        while z < k {
            let dz = dist[(base + z) as usize];
            if dz < dmin {
                dmin = dz;
            }
            z += 1u32;
        }

        let mut o = 0u32;
        while o < n_outputs {
            let mut acc = zero;
            let mut acc_c = zero;
            let mut wsum = zero;
            let mut wsum_c = zero;
            let mut j = 0u32;
            while j < k {
                let t = idx[(base + j) as usize];
                // WR-02 parity with `knn_regress_mean`: an index outside
                // `[0, n_train)` contributes NOTHING (no out-of-bounds device
                // read) and flags the row instead of silently skewing the mean.
                if t < n_train {
                    let v = y_train[(t * n_outputs + o) as usize];
                    let mut w = one;
                    if weighted == 1u32 {
                        let d = dist[(base + j) as usize];
                        if dmin <= zero {
                            // Coincident-neighbor row: indicator weighting.
                            w = zero;
                            if d <= zero {
                                w = one;
                            }
                        } else {
                            w = one / d;
                        }
                    }
                    let term = w * v;

                    // Neumaier fold of `term` into `acc`.
                    let s = acc + term;
                    let mut a_mag = acc;
                    if a_mag < zero {
                        a_mag = zero - a_mag;
                    }
                    let mut t_mag = term;
                    if t_mag < zero {
                        t_mag = zero - t_mag;
                    }
                    if a_mag >= t_mag {
                        acc_c += (acc - s) + term;
                    } else {
                        acc_c += (term - s) + acc;
                    }
                    acc = s;

                    // Neumaier fold of `w` into `wsum`. Weights are
                    // non-negative, so the magnitude test always takes the
                    // first arm — it is kept in the same shape as the numerator
                    // so a future signed weighting cannot silently break it.
                    let ws = wsum + w;
                    let mut w_mag = wsum;
                    if w_mag < zero {
                        w_mag = zero - w_mag;
                    }
                    let mut wj_mag = w;
                    if wj_mag < zero {
                        wj_mag = zero - wj_mag;
                    }
                    if w_mag >= wj_mag {
                        wsum_c += (wsum - ws) + w;
                    } else {
                        wsum_c += (w - ws) + wsum;
                    }
                    wsum = ws;
                } else {
                    bad = 1u32;
                }
                j += 1u32;
            }
            // `k >= 1` is validated host-side and every weight is > 0 on both
            // paths (uniform: 1; distance: 1/d with d > 0, or the indicator
            // weighting where at least the coincident neighbor carries 1), so
            // the denominator is never zero for an uncorrupted row.
            out[(q * n_outputs + o) as usize] = (acc + acc_c) / (wsum + wsum_c);
            o += 1u32;
        }
        oob[q as usize] = bad;
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

/// Query rows one unit owns in [`euclidean_topk_rows_cpu_direct`] — the cpu
/// DEFAULT kernel, which runs a WIDER tile than the expansion arms above.
///
/// ## Why wider (KNN-REG-01, measured — not a guess)
/// The inner loop's per-feature cost is not the arithmetic, it is the ops
/// SURROUNDING it: one scalar `y[t, c]` load plus one splat feed each vector
/// FMA, and those two are paid per FMA regardless of how many query rows the
/// vector covers. Doubling the lane count therefore halves the per-query-row
/// overhead. A roofline probe of this exact loop shape on a 16-core Zen5
/// (`vector<Nxf32>`, 4 independent training rows, one splat per FMA) measured:
///
/// | lanes | GFMA/s | vs 16 |
/// |-------|--------|-------|
/// | 16    | 147    | 1.00× |
/// | 32    | 219    | 1.49× |
/// | 64    | 258    | 1.75× |
/// | 128   | 171    | 1.16× |
///
/// ## Why 32 and NOT the 64 peak
/// 64 is the per-row throughput peak, and a 64-lane kernel was built, measured
/// and then DELETED. Two reasons, both of which the roofline table above cannot
/// see:
///
/// 1. **JIT cost.** `cubecl-cpu` compiles a kernel on first launch, and the cost
///    scales with emitted IR — which a `#[unroll]`ed lane loop multiplies by the
///    lane count. Measured on a 16-core Zen5 (release): the 16-lane kernel this
///    work started from JITs in 0.83 s, the 32-lane form in 1.72 s, the 64-lane
///    form in 6.36 s. Shipping both meant a process that predicted at both
///    grains paid 8.08 s before doing any work, which made a ONE-SHOT predict
///    ~5× SLOWER end to end even though steady-state per-launch time improved
///    2.5× — a regression the perf probe's best-of-3 timing hid by design.
/// 2. **The width is also the SCHEDULING GRAIN.** A launch splits
///    `ceil(n_query / tile)` tiles over `cpu_launch_units()` threads, so a wide
///    tile leaves threads idle until `n_query >= units * tile` — 1024 rows at 64
///    lanes against 512 at 32, on 16 threads.
///
/// One width also removes two hazards the two-width dispatch carried: the choice
/// was dtype-blind (the table is `vector<Nxf32>`; at f64 a 64-lane vector is
/// eight zmm of pressure and nothing measured it), and the switch threshold was
/// itself wrong in the `units*32 .. units*64` band.
///
/// The `k`-list stride stays [`CPU_K_CAP`], independent of the lane count.
pub const CPU_QUERY_TILE_WIDE: u32 = 32;


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
        let mut lworst_val = Array::<F>::new(16usize);
        let mut lworst_idx = Array::<u32>::new(16usize);
        let mut xn = Array::<F>::new(16usize);

        // Sentinel-prefilled lists + disabled dummy lanes, IDENTICAL to the SIMD
        // twin's (KNN-REG-01): this kernel used to carry its own counted-admission
        // copy of the rule, which made the two disagree on any input producing a
        // NaN distance despite the bitwise-identity contract below.
        let mut active = rows_x - q0;
        if active > 16u32 {
            active = 16u32;
        }
        prefill_lists::<F>(
            &mut lval,
            &mut lidx,
            &mut lworst_val,
            &mut lworst_idx,
            16u32,
            active,
            k,
        );

        // --- stage: norms, transposed feature cache ---
        #[unroll]
        for a in 0..16u32 {
            let r = q0 + a;
            let mut nv = zero_f;
            if r < rows_x {
                nv = xnorm[r as usize];
            }
            xn[a as usize] = nv;

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
                insert_lane::<F>(
                    &mut lval,
                    &mut lidx,
                    &mut lworst_val,
                    &mut lworst_idx,
                    a,
                    v,
                    t,
                    k,
                );
            }

            t += 1u32;
        }

        // --- emit: ascending pair order already, sentinels clamped ---
        emit_lists::<F>(&lval, &lidx, out_val, out_idx, q0, rows_x, 16u32, k);
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
/// rule and emit are the same too — this kernel keeps the plain `mul` + `add`
/// accumulation and does NOT take the DIRECT kernel's `fma()`, precisely
/// because `fma` rounds once where the scalar twin rounds twice. Results are
/// therefore BITWISE identical to the scalar kernel's, which is what
/// `knn_test.rs` asserts.
///
/// KNN-REG-01 did give this kernel the two changes that cost no accuracy: its
/// accumulators and row bases are `let` bindings rather than an `Array` (stack
/// memory at `-O0`), and its top-k lists arrive sentinel-prefilled from
/// `prefill_lists` so admission's reject path drops the per-lane `lcnt` load and
/// `cnt < k` branch. It keeps the per-row `admit_tile` rather than the DIRECT
/// kernel's mask-screened form, because its clamp has to be applied per lane
/// before any comparison is valid.
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
        let mut lworst_val = Array::<F>::new(16usize);
        let mut lworst_idx = Array::<u32>::new(16usize);

        // Lanes mapping to a real query row; beyond it `prefill_lists` seeds a
        // bound nothing can beat, so a tail tile's dummy lanes never admit.
        let mut active = rows_x - q0;
        if active > 16u32 {
            active = 16u32;
        }
        prefill_lists::<F>(
            &mut lval,
            &mut lidx,
            &mut lworst_val,
            &mut lworst_idx,
            16u32,
            active,
            k,
        );

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
            let mut t1 = t + 1u32;
            if t1 >= rows_y {
                t1 = rows_y - 1u32;
            }
            let mut t2 = t + 2u32;
            if t2 >= rows_y {
                t2 = rows_y - 1u32;
            }
            let mut t3 = t + 3u32;
            if t3 >= rows_y {
                t3 = rows_y - 1u32;
            }
            let b0 = t * cols;
            let b1 = t1 * cols;
            let b2 = t2 * cols;
            let b3 = t3 * cols;
            let mut a0 = Vector::<F, Const<16>>::from_int(0i64);
            let mut a1 = Vector::<F, Const<16>>::from_int(0i64);
            let mut a2 = Vector::<F, Const<16>>::from_int(0i64);
            let mut a3 = Vector::<F, Const<16>>::from_int(0i64);

            // Comptime-unrolled 8-feature chunks, then the sub-8 tail — the
            // loop bookkeeping is what the unroll is buying, not the FMAs.
            let mut h = 0u32;
            while h < chunks {
                let cb = h * 8u32;
                #[unroll]
                for i in 0..8u32 {
                    let cc = cb + i;
                    let xv = xt[cc as usize];
                    a0 += xv * Vector::<F, Const<16>>::new(y[(b0 + cc) as usize]);
                    a1 += xv * Vector::<F, Const<16>>::new(y[(b1 + cc) as usize]);
                    a2 += xv * Vector::<F, Const<16>>::new(y[(b2 + cc) as usize]);
                    a3 += xv * Vector::<F, Const<16>>::new(y[(b3 + cc) as usize]);
                }
                h += 1u32;
            }
            let mut ct = tail0;
            while ct < cols {
                let xv = xt[ct as usize];
                a0 += xv * Vector::<F, Const<16>>::new(y[(b0 + ct) as usize]);
                a1 += xv * Vector::<F, Const<16>>::new(y[(b1 + ct) as usize]);
                a2 += xv * Vector::<F, Const<16>>::new(y[(b2 + ct) as usize]);
                a3 += xv * Vector::<F, Const<16>>::new(y[(b3 + ct) as usize]);
                ct += 1u32;
            }

            // d² = (‖x‖² + ‖y‖²) − 2·x·y for the whole tile at once; the same
            // association the scalar kernel evaluates per lane.
            let d0 = (xnv + Vector::<F, Const<16>>::new(ynorm[t as usize])) - (a0 + a0);
            admit_tile::<F>(
                &mut lval,
                &mut lidx,
                &mut lworst_val,
                &mut lworst_idx,
                d0,
                t,
                k,
            );
            if t + 1u32 < rows_y {
                let d1 = (xnv + Vector::<F, Const<16>>::new(ynorm[t1 as usize])) - (a1 + a1);
                admit_tile::<F>(
                    &mut lval,
                    &mut lidx,
                    &mut lworst_val,
                    &mut lworst_idx,
                    d1,
                    t1,
                    k,
                );
            }
            if t + 2u32 < rows_y {
                let d2 = (xnv + Vector::<F, Const<16>>::new(ynorm[t2 as usize])) - (a2 + a2);
                admit_tile::<F>(
                    &mut lval,
                    &mut lidx,
                    &mut lworst_val,
                    &mut lworst_idx,
                    d2,
                    t2,
                    k,
                );
            }
            if t + 3u32 < rows_y {
                let d3 = (xnv + Vector::<F, Const<16>>::new(ynorm[t3 as usize])) - (a3 + a3);
                admit_tile::<F>(
                    &mut lval,
                    &mut lidx,
                    &mut lworst_val,
                    &mut lworst_idx,
                    d3,
                    t3,
                    k,
                );
            }

            t += 4u32;
        }

        emit_lists::<F>(&lval, &lidx, out_val, out_idx, q0, rows_x, 16u32, k);
    }
}

/// THE admission rule, in one place: fold candidate `(v, t)` into lane `a`'s
/// sorted top-k list, or reject it.
///
/// EVERY cpu row-scan kernel funnels through here — the scalar
/// [`euclidean_topk_rows_cpu`], the SIMD [`euclidean_topk_rows_cpu_vec`] and the
/// cancellation-free [`euclidean_topk_rows_cpu_direct`] — so the rule that
/// decides RESULTS (the `(value, index)` order and its lowest-index tie-break)
/// exists exactly once, however the callers differ in lane width, clamping or
/// screening. That is also what makes the bitwise identity those kernels claim
/// STRUCTURAL rather than a test result: they cannot drift, because there is only
/// one rule. It was not always so — the scalar kernel carried its own counted
/// copy until KNN-REG-01, and the two forms silently disagreed on any input
/// producing a NaN distance.
///
/// `a` is a RUNTIME lane index (the list is an `Array`, i.e. memory, so a dynamic
/// index costs nothing extra); only the `Vector` LANE EXTRACT that produced `v`
/// needs a comptime index, and that happens in the caller.
///
/// ## Lists arrive SENTINEL-PREFILLED (KNN-REG-01)
/// The caller must fill slots `0 .. k` of every lane's list with
/// `(+∞, u32::MAX)` and set `lworst_val`/`lworst_idx` to that pair — see
/// [`prefill_lists`]. That is a PERF invariant, not a style choice: with a
/// partially-filled list the reject test has to load a per-lane `lcnt` and
/// branch on `cnt < k` before it can even look at the candidate, and this test
/// runs `n_train × lanes` times per unit while the insert body runs ~`k + k·ln`
/// times. Prefilling makes the reject path a single compare against
/// `lworst_val`, and `+∞`/`u32::MAX` sort strictly after every real candidate
/// under the same `(value, index)` order, so the k slots are all displaced by
/// real candidates before the scan ends (`k <= n_train` is validated host-side)
/// and the emitted list is IDENTICAL to the counted form's.
///
/// The ONE input that invariant does not cover is a non-finite distance: `v < w`
/// and `v == w` are both false for NaN, so a query row whose distances are all
/// NaN admits nothing and keeps its sentinels. [`emit_lists`] clamps the index on
/// the way out so that cannot reach a caller.
///
/// Rejecting is one compare against `lworst_val[a]` because the lists arrive
/// sentinel-prefilled — see [`prefill_lists`].
#[cube]
#[allow(clippy::too_many_arguments)]
fn insert_lane<F: Float + CubeElement>(
    lval: &mut Array<F>,
    lidx: &mut Array<u32>,
    lworst_val: &mut Array<F>,
    lworst_idx: &mut Array<u32>,
    a: u32,
    v: F,
    t: u32,
    k: u32,
) {
    let w = lworst_val[a as usize];
    let mut admit: u32 = 0u32;
    if v < w {
        admit = 1u32;
    } else if v == w {
        if t < lworst_idx[a as usize] {
            admit = 1u32;
        }
    }

    if admit == 1u32 {
        let mut cav = v;
        let mut cai = t;
        let mut fs = 0u32;
        while fs < k {
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
        // The pair that fell off the end is gone; slot `k - 1` is the new worst
        // kept pair.
        lworst_val[a as usize] = lval[(a * 16u32 + k - 1u32) as usize];
        lworst_idx[a as usize] = lidx[(a * 16u32 + k - 1u32) as usize];
    }
}

/// The per-train-row admission step of the EXPANSION arms, generated over the
/// lane count.
///
/// `dv` is the tile's squared distances against training row `t`; each lane is
/// clamped and handed to [`insert_lane`], which owns the admission rule itself.
///
/// ## `$clamp`: whether the 0-clamp is live
/// The expansion arms need `max(dv, 0)` because `‖x‖² + ‖y‖² − 2·x·y` is a
/// difference of large numbers and can land just below zero. The DIRECT kernel
/// accumulates `Σ (x−y)²` — a sum of squares, which no IEEE rounding can carry
/// below zero — so for it the clamp is provably dead, and dead code is not free
/// at LLVM `-O0`: it is a compare and a select per LANE per TRAINING ROW. That
/// kernel therefore uses [`impl_admit_row`]'s screened form instead.
///
/// ## Why a macro over the lane count
/// A `Vector`'s width is part of its TYPE, and a lane must be indexed at comptime
/// to stay a register extract, so one function cannot serve two widths. Only the
/// 16-lane form is instantiated today; the macro remains because the width is a
/// measured tuning parameter (see [`CPU_QUERY_TILE_WIDE`]) rather than a
/// constant of the algorithm, and re-instantiating at another width must not
/// mean re-typing the body.
macro_rules! impl_admit_tile {
    ($name:ident, $lanes:literal, $lanes_u32:literal, $clamp:literal) => {
        #[cube]
        #[allow(clippy::too_many_arguments)]
        fn $name<F: Float + CubeElement>(
            lval: &mut Array<F>,
            lidx: &mut Array<u32>,
            lworst_val: &mut Array<F>,
            lworst_idx: &mut Array<u32>,
            dv: Vector<F, Const<$lanes>>,
            t: u32,
            k: u32,
        ) {
            #[unroll]
            for a in 0..$lanes_u32 {
                let mut v = dv[a as usize];
                if comptime!($clamp) {
                    let zero_f = F::from_int(0i64);
                    if v < zero_f {
                        v = zero_f;
                    }
                }
                insert_lane::<F>(lval, lidx, lworst_val, lworst_idx, a, v, t, k);
            }
        }
    };
}

impl_admit_tile!(admit_tile, 16, 16u32, true);

/// MASK-SCREENED admission of ONE training row's tile of squared distances.
///
/// ## Why a screen at all
/// Admission is a REJECT in the overwhelming majority of cases: with
/// `n_train = 100_000` and `k = 10` a query row admits ~10 + O(k·ln n_train)
/// candidates out of 100_000, so ~99.9% of the work is proving a candidate is
/// not good enough. Doing that per lane costs a vector lane extract, a
/// `lworst_val` load and a compare — measured at ~40% of this kernel's
/// wall-clock once the dot loop was fused (`knn_cpu_shape_probe` separates the
/// two: the per-feature slope is the dot loop, the intercept is this).
///
/// ## The screen: one vector compare + an INTEGER horizontal reduction
/// `dv.less_equal(wv)` is one vector op giving a per-lane mask, and
/// `Vector::<u32, _>::cast_from(mask).vector_sum()` collapses it to a scalar
/// count. If the count is zero, NO lane can admit and the whole per-lane pass is
/// skipped — ~4 ops in place of `LANES` extract/load/compare triples.
///
/// The dtype of the reduction is the whole trick, and it is measured: on
/// `cubecl-cpu` a `vector_sum` over `Vector<u32, _>` lowers to
/// `vector.reduction<add>` on integers, which LLVM legalizes to a log-TREE of
/// `vpaddd` — 4.2× faster than scanning the lanes by hand over an all-reject
/// input. The same reduction over `Vector<f32, _>` is a SEQUENTIAL `fadd` chain
/// (no `reassoc` fast-math flag), which measured 1.00× — i.e. exactly as slow as
/// the scan it would replace. Reduce the MASK, never the distances.
///
/// ## Why PER ROW and not per 4-row block
/// An earlier form screened a whole 4-row training block at once with
/// `min(min(d0,d1),min(d2,d3))`, which is fewer screens. It was wrong: `min`
/// PROPAGATES NaN, so one NaN training row made the block's bound NaN, `NaN <=
/// lworst_val[a]` false, and all four rows were skipped for every lane —
/// silently dropping the NaN row's three FINITE block-mates from the candidate
/// set (reproduced: 4 query rows changed their top-k). Screening per row keeps a
/// non-finite row's blast radius to itself, and costs nothing: 4 screens at ~4
/// ops still beats one screen plus `LANES` lane visits.
///
/// ## Why the per-lane pass is a RUNTIME loop
/// The taken branch walks lanes with `while a < LANES` over a spilled `dbuf`
/// rather than `#[unroll]`, so [`insert_lane`] is inlined ONCE per row instead
/// of `LANES` times. That is a JIT-time decision, not a runtime one:
/// `cubecl-cpu` compiles on first launch and the cost scales with emitted IR, and
/// the unrolled form's `LANES × 4` inlined admission bodies were most of why the
/// 64-lane kernel took 6.36 s to compile. The loop's own bookkeeping is only
/// paid on the rare rows that actually admit.
macro_rules! impl_admit_row {
    ($name:ident, $lanes:literal, $lanes_u32:literal) => {
        #[cube]
        #[allow(clippy::too_many_arguments)]
        fn $name<F: Float + CubeElement>(
            lval: &mut Array<F>,
            lidx: &mut Array<u32>,
            lworst_val: &mut Array<F>,
            lworst_idx: &mut Array<u32>,
            dbuf: &mut Array<F>,
            dv: Vector<F, Const<$lanes>>,
            wv: Vector<F, Const<$lanes>>,
            t: u32,
            k: u32,
        ) -> Vector<F, Const<$lanes>> {
            let mut wv_out = wv;
            let mask = dv.less_equal(wv);
            let nhit = u32::cast_from(Vector::<u32, Const<$lanes>>::cast_from(mask).vector_sum());
            if nhit > 0u32 {
                #[unroll]
                for a in 0..$lanes_u32 {
                    dbuf[a as usize] = dv[a as usize];
                }
                let mut a = 0u32;
                while a < $lanes_u32 {
                    insert_lane::<F>(lval, lidx, lworst_val, lworst_idx, a, dbuf[a as usize], t, k);
                    a += 1u32;
                }
                // Refresh the vector mirror of `lworst_val`, which the inserts
                // above may have lowered. Only reached when something admitted.
                #[unroll]
                for a in 0..$lanes_u32 {
                    wv_out[a as usize] = lworst_val[a as usize];
                }
            }
            wv_out
        }
    };
}

impl_admit_row!(admit_row_w32, 32, 32u32);

/// Sentinel-prefill one unit's `lanes` top-k lists so [`insert_lane`] can skip
/// the fill-phase bookkeeping, and DISABLE the lanes that own no query row.
///
/// Slots `0 .. k` of each ACTIVE lane get `(+∞, u32::MAX)`, the pair that sorts
/// after every real candidate under the `(value, index)` order; slots
/// `k .. CPU_K_CAP` are never read (both the insert loop and the emit loop bound
/// by `k`). That is what lets admission reject with a single compare instead of
/// loading a per-lane count and branching on `cnt < k`, and it is
/// result-preserving: `k <= n_train` is validated host-side, so every one of the
/// `k` slots is displaced by a real candidate before the scan ends.
///
/// `active` is the number of lanes that map to a real query row (`min(lanes,
/// rows_x - q0)`). A lane beyond it gets `lworst_val = −∞`, which nothing can
/// beat — squared distances are non-negative — so it is screened out on every
/// training row and never runs an insert. Without that, an over-provisioned lane
/// would build and maintain a full top-k list of distances-to-the-origin across
/// the ENTIRE training scan, for a result the emit guard then discards: up to
/// 31/32 of a tail tile's admission cost.
///
/// `lanes` and `active` are runtime arguments because this loop is not unrolled —
/// it runs once per unit, not once per training row.
#[cube]
fn prefill_lists<F: Float + CubeElement>(
    lval: &mut Array<F>,
    lidx: &mut Array<u32>,
    lworst_val: &mut Array<F>,
    lworst_idx: &mut Array<u32>,
    lanes: u32,
    active: u32,
    k: u32,
) {
    let inf = F::new(f32::INFINITY);
    let neg_inf = F::new(f32::NEG_INFINITY);
    let sentinel_idx = u32::MAX;
    let mut a = 0u32;
    while a < lanes {
        let mut j = 0u32;
        while j < k {
            lval[(a * 16u32 + j) as usize] = inf;
            lidx[(a * 16u32 + j) as usize] = sentinel_idx;
            j += 1u32;
        }
        if a < active {
            lworst_val[a as usize] = inf;
        } else {
            lworst_val[a as usize] = neg_inf;
        }
        lworst_idx[a as usize] = sentinel_idx;
        a += 1u32;
    }
}

/// Write one unit's finished top-k lists out, CLAMPING any surviving sentinel
/// index so nothing out of range can reach a caller.
///
/// The lists are already in ascending `(value, index)` order, so this is a copy.
/// The clamp is the safety net for the one case the sentinel prefill cannot
/// satisfy: a query row whose distances are all NaN admits nothing (`v < w` and
/// `v == w` are both false for NaN), so its `k` slots still hold
/// `(+∞, u32::MAX)` when the scan ends. Emitting `u32::MAX` would push an
/// out-of-range training index into whatever consumes the neighbor list —
/// `knn_regress_mean` skips it and silently predicts `0.0`, while
/// `KNeighborsClassifier` rejects the whole batch with a shape error. Emitting
/// `0` keeps the index in range for every consumer; the paired `+∞` distance is
/// what actually tells a caller the row had no finite neighbor.
///
/// Non-finite input is REJECTED before it ever reaches here on the Python
/// surface (`_io.normalize_X`/`normalize_y` run sklearn
/// `check_array(ensure_all_finite=True)`, so mlrs raises the same
/// `ValueError: Input contains NaN` sklearn does). This clamp exists for the
/// Rust API, which takes an already-uploaded `DeviceArray` and so has no such
/// gate.
///
/// The lane loop is `while a < lanes` rather than `#[unroll]` so this is inlined
/// once per kernel instead of once per lane — the same JIT-cost reason
/// [`impl_admit_row`] gives.
#[cube]
#[allow(clippy::too_many_arguments)]
fn emit_lists<F: Float + CubeElement>(
    lval: &Array<F>,
    lidx: &Array<u32>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    q0: u32,
    rows_x: u32,
    lanes: u32,
    k: u32,
) {
    let mut a = 0u32;
    while a < lanes {
        let r = q0 + a;
        if r < rows_x {
            let mut j = 0u32;
            while j < k {
                out_val[(r * k + j) as usize] = lval[(a * 16u32 + j) as usize];
                let mut ix = lidx[(a * 16u32 + j) as usize];
                if ix == u32::MAX {
                    ix = 0u32;
                }
                out_idx[(r * k + j) as usize] = ix;
                j += 1u32;
            }
        }
        a += 1u32;
    }
}

/// Generates the cancellation-free cpu row-scan kernel over a query-tile width.
///
/// The generated kernel's own documentation lives on the `pub` instantiation
/// below (rustdoc does not render docs written on a private `macro_rules!`, so
/// putting the rationale here would hide it from exactly the people who need it).
macro_rules! impl_topk_rows_cpu_direct {
    (
        $(#[$doc:meta])*
        $name:ident, $admit:ident, $lanes:literal, $lanes_u32:literal, $lanes_sz:literal,
        $list:literal
    ) => {
        $(#[$doc])*
        #[cube(launch)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<F: Float + CubeElement>(
            x: &Array<F>,
            y: &Array<F>,
            out_val: &mut Array<F>,
            out_idx: &mut Array<u32>,
            rows_x: u32,
            rows_y: u32,
            cols: u32,
            k: u32,
        ) {
            let q0 = ABSOLUTE_POS_X * $lanes_u32;
            if q0 < rows_x {
                let zero_f = F::from_int(0i64);

                let mut xt = Array::<Vector<F, Const<$lanes>>>::new(32usize);
                let mut lval = Array::<F>::new($list);
                let mut lidx = Array::<u32>::new($list);
                let mut lworst_val = Array::<F>::new($lanes_sz);
                let mut lworst_idx = Array::<u32>::new($lanes_sz);

                let mut dbuf = Array::<F>::new($lanes_sz);
                // Lanes that map to a real query row. Beyond it `prefill_lists`
                // seeds a bound nothing can beat, so a tail tile's dummy lanes
                // are screened out instead of maintaining a discarded top-k.
                let mut active = rows_x - q0;
                if active > $lanes_u32 {
                    active = $lanes_u32;
                }
                prefill_lists::<F>(
                    &mut lval,
                    &mut lidx,
                    &mut lworst_val,
                    &mut lworst_idx,
                    $lanes_u32,
                    active,
                    k,
                );
                // Vector mirror of `lworst_val`, the screen's operand. Kept in
                // sync by `impl_admit_row`, which returns the refreshed value.
                let mut wv = Vector::<F, Const<$lanes>>::new(F::new(f32::INFINITY));
                #[unroll]
                for a in 0..$lanes_u32 {
                    wv[a as usize] = lworst_val[a as usize];
                }
                let mut c = 0u32;
                while c < cols {
                    let mut v = Vector::<F, Const<$lanes>>::from_int(0i64);
                    #[unroll]
                    for a in 0..$lanes_u32 {
                        let r = q0 + a;
                        // An over-provisioned lane (`r >= rows_x`) stages zeros
                        // and never reaches the emit guard below.
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
                    // The four training-row bases and their four accumulators are
                    // plain `let` bindings, NOT an `Array` — an `Array` is
                    // `memref.alloca`, i.e. real stack memory that costs a load
                    // and a store on every update, while a binding is an SSA
                    // value MLIR's `mem2reg` keeps in a register.
                    let mut t1 = t + 1u32;
                    if t1 >= rows_y {
                        t1 = rows_y - 1u32;
                    }
                    let mut t2 = t + 2u32;
                    if t2 >= rows_y {
                        t2 = rows_y - 1u32;
                    }
                    let mut t3 = t + 3u32;
                    if t3 >= rows_y {
                        t3 = rows_y - 1u32;
                    }
                    let b0 = t * cols;
                    let b1 = t1 * cols;
                    let b2 = t2 * cols;
                    let b3 = t3 * cols;
                    let mut a0 = Vector::<F, Const<$lanes>>::from_int(0i64);
                    let mut a1 = Vector::<F, Const<$lanes>>::from_int(0i64);
                    let mut a2 = Vector::<F, Const<$lanes>>::from_int(0i64);
                    let mut a3 = Vector::<F, Const<$lanes>>::from_int(0i64);

                    // Comptime-unrolled 8-feature chunks, then the sub-8 tail:
                    // at `-O0` the loop's own compare/branch/increment is real
                    // work, so paying it once per chunk instead of once per
                    // feature is the point (the expansion arm does the same).
                    let mut h = 0u32;
                    while h < chunks {
                        let cb = h * 8u32;
                        #[unroll]
                        for i in 0..8u32 {
                            let cc = cb + i;
                            let xv = xt[cc as usize];
                            let d0 = xv - Vector::<F, Const<$lanes>>::new(y[(b0 + cc) as usize]);
                            let d1 = xv - Vector::<F, Const<$lanes>>::new(y[(b1 + cc) as usize]);
                            let d2 = xv - Vector::<F, Const<$lanes>>::new(y[(b2 + cc) as usize]);
                            let d3 = xv - Vector::<F, Const<$lanes>>::new(y[(b3 + cc) as usize]);
                            a0 = fma(d0, d0, a0);
                            a1 = fma(d1, d1, a1);
                            a2 = fma(d2, d2, a2);
                            a3 = fma(d3, d3, a3);
                        }
                        h += 1u32;
                    }
                    let mut cc = tail0;
                    while cc < cols {
                        let xv = xt[cc as usize];
                        let d0 = xv - Vector::<F, Const<$lanes>>::new(y[(b0 + cc) as usize]);
                        let d1 = xv - Vector::<F, Const<$lanes>>::new(y[(b1 + cc) as usize]);
                        let d2 = xv - Vector::<F, Const<$lanes>>::new(y[(b2 + cc) as usize]);
                        let d3 = xv - Vector::<F, Const<$lanes>>::new(y[(b3 + cc) as usize]);
                        a0 = fma(d0, d0, a0);
                        a1 = fma(d1, d1, a1);
                        a2 = fma(d2, d2, a2);
                        a3 = fma(d3, d3, a3);
                        cc += 1u32;
                    }

                    // One mask-screened pass PER ROW — see `impl_admit_row` for
                    // why per row and not per block (a block-wide `min` screen
                    // propagates NaN and would drop a NaN row's FINITE
                    // block-mates). Rows the dot loop CLAMPED past the end are
                    // skipped entirely rather than screened, so a clamped
                    // duplicate can never be admitted twice.
                    wv = $admit::<F>(
                        &mut lval,
                        &mut lidx,
                        &mut lworst_val,
                        &mut lworst_idx,
                        &mut dbuf,
                        a0,
                        wv,
                        t,
                        k,
                    );
                    if t + 1u32 < rows_y {
                        wv = $admit::<F>(
                            &mut lval,
                            &mut lidx,
                            &mut lworst_val,
                            &mut lworst_idx,
                            &mut dbuf,
                            a1,
                            wv,
                            t1,
                            k,
                        );
                    }
                    if t + 2u32 < rows_y {
                        wv = $admit::<F>(
                            &mut lval,
                            &mut lidx,
                            &mut lworst_val,
                            &mut lworst_idx,
                            &mut dbuf,
                            a2,
                            wv,
                            t2,
                            k,
                        );
                    }
                    if t + 3u32 < rows_y {
                        wv = $admit::<F>(
                            &mut lval,
                            &mut lidx,
                            &mut lworst_val,
                            &mut lworst_idx,
                            &mut dbuf,
                            a3,
                            wv,
                            t3,
                            k,
                        );
                    }

                    t += 4u32;
                }

                emit_lists::<F>(&lval, &lidx, out_val, out_idx, q0, rows_x, $lanes_u32, k);
            }
        }
    };
}

impl_topk_rows_cpu_direct!(
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
    /// FMA) to remove it, and it is the cpu DEFAULT — `MLRS_KNN_DOT=1` forces the
    /// expansion arms back for A/B.
    ///
    /// ## Shape (KNN-REG-01 — where it differs from the expansion arms)
    /// Same admission rule, tie-break and output layout as
    /// [`euclidean_topk_rows_cpu_vec`] (all three cpu kernels share
    /// `insert_lane`, `prefill_lists` and `emit_lists`), but four things that set
    /// its speed are its own:
    ///
    /// 1. **Wider query tile** ([`CPU_QUERY_TILE_WIDE`] = 32, against 16 for the
    ///    expansion arms). The per-feature `y[t, c]` scalar load and splat are paid
    ///    per vector FMA whatever its width, so lane count is the lever on them. See
    ///    that constant for why 32 and not the measured 64-lane throughput peak — the
    ///    answer is JIT cost, and it is the reason a 64-lane sibling was built,
    ///    measured and deleted.
    /// 2. **`fma()` instead of `mul` + `add`** — one IR op instead of two, and one
    ///    rounding instead of two. At LLVM `-O0` (which is what `cubecl-cpu` JITs
    ///    at) op count IS the cost. Being a `Σ` of squares, this is also strictly
    ///    MORE accurate, so it moves toward sklearn, not away — but the host
    ///    reference in `knn_test.rs` uses `f32::mul_add` to match it exactly rather
    ///    than leaving the index assertion to near-tie luck. The expansion arms
    ///    deliberately do NOT take this, so they stay bitwise-identical to each other.
    /// 3. **Accumulators as `let` bindings, not an `Array`** — an `Array` is
    ///    `memref.alloca`, real stack memory with a load and a store per update.
    /// 4. **Mask-screened admission** (`impl_admit_row`) instead of a per-lane
    ///    scan: one vector compare plus an integer horizontal reduction decides
    ///    whether ANY lane can admit this training row.
    ///
    /// Together these measured ~2.4× on the `n_train = 100_000 / 200_000, d = 32,
    /// k = 10` rungs of `knn_regressor_perf_test`, at LOWER cold-start cost than the
    /// 16-lane kernel they replaced (0.22 s of JIT against 0.83 s).
    euclidean_topk_rows_cpu_direct,
    admit_row_w32,
    32,
    32u32,
    32usize,
    512usize
);
