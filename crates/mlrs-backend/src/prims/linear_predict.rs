//! Dense linear-model inference host API (LINEAR-01/02 predict perf lever) —
//! `y = X·coef + intercept`, device-resident, single kernel launch, no host
//! round-trip.
//!
//! ## Why this exists (see `mlrs_kernels::linear_predict` module docs)
//! Every dense linear regressor (`LinearRegression`, `Ridge`, `Lasso`,
//! `ElasticNet`) shared ONE predict body: `raw = gemm(X, coef)` (a skinny
//! `m×1` output) followed by a HOST intercept broadcast — `intercept.to_host()`
//! (blocking scalar readback) + `raw.to_host()` (`m`-length device→host) +
//! an element-wise host loop + `DeviceArray::from_host()` (`m`-length host→
//! device, only for the PyO3 boundary to read it back to host AGAIN). On a
//! discrete GPU those crossings — not the FLOPs — dominate `predict` (the same
//! host-sync pathology `center`/`gram` fixed for the fit path). [`linear_predict`]
//! replaces the whole dance with a single [`linear_predict_bias`] launch that
//! computes `y[r] = Σ_c X[r,c]·coef[c] + bias` fully on device; the caller's
//! own terminal readback is then the ONLY host↔device crossing.
//!
//! ## Two kernels: coalesced shared-tile (wgpu perf lever) + GATHER (default)
//! The GATHER kernel ([`linear_predict_bias`]) is thread-per-row over the
//! ROW-MAJOR `x`, so a warp's rows stride by `n` — an UNCOALESCED read. On
//! THIS ENV's wgpu adapter (AMD iGPU) that measurably starves the
//! bandwidth-bound matvec (worst at `n = 64`: ~10× the per-element cost of a
//! coalesced read). [`linear_predict_bias_shared`]
//! (`mlrs_kernels::linear_predict` docs) fixes that by staging each row block
//! into `SharedMemory` with a fully COALESCED load, then doing the per-row dot
//! from a bank-conflict-free padded tile — confirmed a 2.5–4× wgpu win in its
//! band (`PREDICT_SHARED_MIN_FEATURES ≤ n ≤ PREDICT_MAX_FEATURES`).
//!
//! **On a real Tesla T4 (CUDA) this does NOT hold**: a within-session A/B (5
//! reps each, `LR_PREDICT_GATHER` toggle, `n = 64` / `m = 100000`, both
//! LinearRegression and Ridge) measured GATHER ~15–28% FASTER than the shared
//! kernel, consistently across all 10 comparisons with zero overlap — the
//! T4's large L2 (4 MiB) apparently already absorbs the strided-read cost that
//! hurts on this env's iGPU, so the extra `SharedMemory` barriers and halved
//! block occupancy (64 vs 256 threads) are pure overhead there. [`linear_predict`]
//! reserves the shared kernel for [`use_shared_predict`]'s wgpu-only gate;
//! cuda/rocm/cpu always use GATHER (already validated to beat cuML/sklearn
//! substantially on the T4 — see `mlrs-linear-predict-coalesced` project
//! memory). The shared kernel stays compiled and tested (not dead code) in
//! case a future backend/GPU generation profile differs; `LR_PREDICT_GATHER`
//! remains available to force GATHER on wgpu too for A/B re-verification.
//!
//! ## The cpu backend does NOT take either kernel — [`linear_predict_host`]
//! Both kernels above assume `x` is already device-resident. On a discrete GPU
//! that is a given (the operand had to cross PCIe anyway). On the **cpu**
//! backend it is a self-inflicted wound: "device" memory IS host memory, so
//! `DeviceArray::from_host` is a pure `memcpy` of the whole `m × n` test matrix
//! — and the matvec then reads that copy exactly once. Measured on a 16-core
//! Zen5 at `m = 1_000_000`, `n = 16` (a 64 MiB f32 operand):
//!
//! | stage | wall-clock |
//! |---|---|
//! | one host copy of `x` (`numpy` `x.copy()`) | 13.5 ms |
//! | `sklearn` `LinearRegression.predict` END TO END | 4.4 ms |
//! | └ of which its BLAS `X @ coef` | 1.2 ms |
//!
//! The copy alone is 3× sklearn's ENTIRE predict, because sklearn reads the
//! caller's buffer in place and mlrs was reading 64 MiB, writing 64 MiB, then
//! reading 64 MiB again. No kernel tuning can pay that back: the ingress, not
//! the arithmetic, is the whole cost. [`linear_predict_host`] deletes it — it
//! computes straight out of the CALLER'S borrowed host slice (on the Python
//! path, the validated Arrow buffer numpy already owns), with no upload, no
//! pooled operand and no cubecl launch at all.
//!
//! Two further reasons the host path wins on cpu even discounting the copy:
//! `cubecl-cpu` JITs at LLVM **`-O0`** (no vectorizer — see the
//! `mlrs-cubecl-cpu-execution-model` notes), whereas this function is compiled
//! into the crate at the release profile's `-O3` and auto-vectorizes; and a
//! cubecl launch spawns one OS thread PER UNIT, while this splits the row axis
//! across exactly [`crate::capability::cpu_launch_units`] scoped threads.

use std::sync::OnceLock;

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
// The canonical per-dimension CubeCL grid cap (`65_535`) — the same const
// `prims::center` imports for its identical row-block grid fold; reused here
// (not redefined) so a future adjustment to the launch grid math lives in one
// place. The launch folds the row-block count across the X/Y axes so an
// arbitrarily large `m` (`> 65535·256 ≈ 16.7M` predict rows) never overflows a
// single grid dimension and silently drops tail rows.
use mlrs_kernels::colmean::MAX_GRID_DIM;
use mlrs_kernels::{
    linear_predict_bias, linear_predict_bias_multi, linear_predict_bias_shared,
    PREDICT_ROWS_PER_BLOCK,
};
// The shared-tile winning-band bounds are only consulted by `use_shared_predict`'s
// wgpu arm (the ONE backend where the kernel measurably wins — module docs
// above); importing them unconditionally would warn `unused` on cuda/rocm/cpu.
#[cfg(feature = "wgpu")]
use mlrs_kernels::{PREDICT_MAX_FEATURES, PREDICT_SHARED_ELEMS, PREDICT_SHARED_MIN_FEATURES};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Compute `y = X·coef + intercept` for the `m × n` row-major test matrix `x`,
/// the length-`n` fitted `coef`, and the length-1 device-resident `bias`
/// (the intercept; a real `0`-valued length-1 buffer for the no-intercept
/// case). Returns the length-`m` device-resident predictions — NO host
/// round-trip (D-05).
///
/// - Shapes are validated (`m * n == x.len()`, `coef.len() == n`,
///   `bias.len() >= 1`, both dims non-zero) BEFORE the launch; a mismatch
///   returns [`PrimError::ShapeMismatch`] / [`PrimError::DimMismatch`].
/// - A SINGLE [`linear_predict_bias`] launch (one unit per output row, grid
///   folded across X/Y for large `m`) does the fused dot-product-plus-bias on
///   device.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn linear_predict<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    coef: &DeviceArray<ActiveRuntime, F>,
    bias: &DeviceArray<ActiveRuntime, F>,
    (m, n): (usize, usize),
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), (m, n), coef.len(), bias.len())?;

    let elem = size_of::<F>();
    let out_handle = pool.acquire(m * elem);
    let client = pool.client().clone();

    // SAFETY (both paths): `x.len()`/`coef.len()`/`bias.len()`/`m`/`n` are the
    // carried/validated element counts; every kernel bounds-checks its row id
    // and reads only `x[r*n + c]` for `c < n`, `coef[c]`, and `bias[0]` — all in
    // range by the geometry validation above.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let coef_arg = unsafe { ArrayArg::from_raw_parts(coef.handle().clone(), coef.len()) };
    let bias_arg = unsafe { ArrayArg::from_raw_parts(bias.handle().clone(), bias.len()) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), m) };

    if use_shared_predict(n, elem) {
        // Coalesced shared-tile path (the wgpu perf lever): one
        // `PREDICT_ROWS_PER_BLOCK`-thread cube per row block, grid folded across
        // X/Y so `nblocks` never overflows a single dimension's cap.
        let b_rows = PREDICT_ROWS_PER_BLOCK as usize;
        let nblocks = m.div_ceil(b_rows);
        let (ccount, cdim) = launch_cubes_block(nblocks);
        linear_predict_bias_shared::launch::<F, ActiveRuntime>(
            &client, ccount, cdim, x_arg, coef_arg, bias_arg, out_arg, m as u32, n as u32,
            nblocks as u32,
        );
    } else {
        // GATHER thread-per-row default: every backend except wgpu-in-its-band
        // (cpu, cuda, rocm always; wgpu outside `[MIN_FEATURES, MAX_FEATURES]`
        // or under the `LR_PREDICT_GATHER` A/B hatch — see `use_shared_predict`).
        // The kernel bounds-checks `r < m`, masking the slack lanes of the
        // final block.
        let (ccount, cdim) = super::launch_dims_1d_folded(m, crate::capability::gather_launch_width());
        linear_predict_bias::launch::<F, ActiveRuntime>(
            &client, ccount, cdim, x_arg, coef_arg, bias_arg, out_arg, m as u32, n as u32,
        );
    }

    Ok(DeviceArray::from_raw(out_handle, m))
}

/// Multi-target twin of [`linear_predict`] (RIDGE-MULTI-TARGET): `out[r,t] =
/// Σ_c x[r,c]·coef[c,t] + bias[t]` for `k` targets, `coef` row-major `n × k`,
/// `bias` length `k`. Returns the `m × k` row-major device-resident predictions.
///
/// One [`linear_predict_bias_multi`] GATHER launch — no shared-tile arm (unlike
/// [`linear_predict`]'s wgpu path): the coalescing win that kernel chases is
/// orthogonal to the target axis this adds, and `k` is small in every fitted
/// multi-output model, so the extra per-target row re-read (see the kernel's
/// module docs) is not worth a second staged variant until measurement says
/// otherwise. `k == 1` callers should use [`linear_predict`] instead — this
/// prim does not special-case it away.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn linear_predict_multi<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    coef: &DeviceArray<ActiveRuntime, F>,
    bias: &DeviceArray<ActiveRuntime, F>,
    (m, n): (usize, usize),
    k: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry_multi(x.len(), (m, n), coef.len(), bias.len(), k)?;

    let elem = size_of::<F>();
    let out_handle = pool.acquire(m * k * elem);
    let client = pool.client().clone();

    // SAFETY: `x.len()`/`coef.len()`/`bias.len()`/`m`/`n`/`k` are the
    // validated element counts above; the kernel bounds-checks `r < m` and
    // reads only `x[r*n + c]` for `c < n`, `coef[c*k + t]` and `bias[t]` for
    // `t < k` — all in range by the geometry validation.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let coef_arg = unsafe { ArrayArg::from_raw_parts(coef.handle().clone(), coef.len()) };
    let bias_arg = unsafe { ArrayArg::from_raw_parts(bias.handle().clone(), bias.len()) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), m * k) };

    let (ccount, cdim) = super::launch_dims_1d_folded(m, crate::capability::gather_launch_width());
    linear_predict_bias_multi::launch::<F, ActiveRuntime>(
        &client, ccount, cdim, x_arg, coef_arg, bias_arg, out_arg, m as u32, n as u32, k as u32,
    );

    Ok(DeviceArray::from_raw(out_handle, m * k))
}

/// Multi-target twin of [`linear_predict_host`]: `out[r,t] = Σ_c x[r,c]·coef[c,t]
/// + bias[t]`, `coef` row-major `n × k`, `bias` length `k`. Returns the `m × k`
/// row-major predictions plus the operand-finiteness verdict — same no-upload,
/// no-launch, zero-copy contract as the single-target host path (module docs
/// §"The cpu backend does NOT take either kernel").
///
/// Row-major over `(row, target)`: thread `i` owns a contiguous run of
/// ROWS (never splits a row's `k` targets across threads), so `out`'s per-thread
/// slice stays one unbroken range exactly as [`matvec_bias_parallel`] does.
pub fn linear_predict_multi_host<F>(
    x: &[F],
    coef: &[F],
    bias: &[F],
    (m, n): (usize, usize),
    k: usize,
) -> Result<HostPrediction<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry_multi(x.len(), (m, n), coef.len(), bias.len(), k)?;

    let mut values: Vec<F> = vec![bytemuck::Zeroable::zeroed(); m * k];
    let operand_finite = match size_of::<F>() {
        4 => matvec_bias_multi_parallel::<f32>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(coef),
            bytemuck::cast_slice(bias),
            bytemuck::cast_slice_mut(&mut values),
            n,
            k,
        ),
        8 => matvec_bias_multi_parallel::<f64>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(coef),
            bytemuck::cast_slice(bias),
            bytemuck::cast_slice_mut(&mut values),
            n,
            k,
        ),
        other => {
            unreachable!("linear_predict_multi_host is f32/f64 only, got a {other}-byte element")
        }
    };
    Ok(HostPrediction {
        values,
        operand_finite,
    })
}

/// [`matvec_bias_parallel`]'s multi-target twin: `out` is `m × k` row-major,
/// split into contiguous ROW chunks across [`host_units`] scoped threads (so a
/// thread's slice of `out` is one unbroken range and its `x` slab is
/// contiguous, exactly as the single-target split).
fn matvec_bias_multi_parallel<T: HostFloat>(
    x: &[T],
    coef: &[T],
    bias: &[T],
    out: &mut [T],
    n: usize,
    k: usize,
) -> bool {
    let m = out.len() / k;
    let units = host_units(out.len() * n).max(1);
    if units <= 1 || m <= 1 {
        return matvec_bias_multi_rows(x, coef, bias, out, n, k);
    }

    let rows_per_unit = m.div_ceil(units);
    std::thread::scope(|scope| {
        let handles: Vec<_> = out
            .chunks_mut(rows_per_unit * k)
            .enumerate()
            .map(|(i, chunk)| {
                let rows_here = chunk.len() / k;
                let slab = &x[i * rows_per_unit * n..(i * rows_per_unit + rows_here) * n];
                scope.spawn(move || matvec_bias_multi_rows(slab, coef, bias, chunk, n, k))
            })
            .collect();
        handles
            .into_iter()
            .all(|h| h.join().expect("linear_predict_multi_host row worker panicked"))
    })
}

/// The serial multi-target row loop: for each row, `k` dot products against
/// `coef`'s columns plus their own `bias[t]`. Returns whether every element of
/// `x` was finite.
fn matvec_bias_multi_rows<T: HostFloat>(
    x: &[T],
    coef: &[T],
    bias: &[T],
    out: &mut [T],
    n: usize,
    k: usize,
) -> bool {
    let mut finite = true;
    let rows = out.len() / k;
    for r in 0..rows {
        let row = &x[r * n..(r + 1) * n];
        finite &= row_all_finite(row);
        for t in 0..k {
            let mut acc = T::ZERO;
            for c in 0..n {
                acc = acc + row[c] * coef[c * k + t];
            }
            out[r * k + t] = acc + bias[t];
        }
    }
    finite
}

/// Validate the multi-target inference operand geometry: `x` is `m × n`
/// row-major, `coef` is `n × k` row-major, `bias` is length `k` (one intercept
/// per target). All three dims non-zero.
fn validate_geometry_multi(
    x_len: usize,
    (m, n): (usize, usize),
    coef_len: usize,
    bias_len: usize,
    k: usize,
) -> Result<(), PrimError> {
    if k == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "coef",
            rows: n,
            cols: 0,
            len: coef_len,
        });
    }
    if m == 0 || n == 0 || m.checked_mul(n).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: m,
            cols: n,
            len: x_len,
        });
    }
    if coef_len != n * k {
        return Err(PrimError::DimMismatch {
            dim: "n_features*n_targets",
            lhs: coef_len,
            rhs: n * k,
        });
    }
    if bias_len != k {
        return Err(PrimError::ShapeMismatch {
            operand: "bias",
            rows: k,
            cols: 1,
            len: bias_len,
        });
    }
    Ok(())
}

/// Compute `y = X·coef + bias` **entirely on the host**, reading the caller's
/// borrowed `x` in place — the cpu-backend predict path (module docs §"The cpu
/// backend does NOT take either kernel").
///
/// `x` is the `m × n` row-major test matrix, `coef` the length-`n` fitted
/// coefficients, `bias` the intercept scalar (`0` for the no-intercept case).
/// Returns the length-`m` predictions plus the operand-finiteness verdict, as a
/// [`HostPrediction`].
///
/// The point is what does NOT happen: no `DeviceArray::from_host` upload, no
/// pooled operand, no kernel launch, no read-back. `x` stays exactly where the
/// caller has it (on the Python path, the validated Arrow buffer numpy owns),
/// so the whole predict touches `m·n` operand bytes ONCE — the same traffic
/// sklearn's BLAS `sgemv` pays, instead of the 3× a copy-then-read costs.
///
/// ## Why the finiteness verdict rides along ([`HostPrediction::operand_finite`])
/// The sklearn-compatible surface must hard-reject a NaN/±inf test matrix, and
/// the shim used to get that from `check_array(ensure_all_finite=True)` — a
/// SECOND full pass over `x`, single-threaded, measured at 2.4 ms for a 64 MiB
/// operand against 1.4 ms for this entire matvec. Since `x` no longer fits in
/// last-level cache at that size, that pass is a genuine second trip to DRAM:
/// validating separately costs as much as predicting. So the scan is fused
/// here, where every row is ALREADY in L1 from the dot product that just read
/// it — the check becomes ALU work on cached bytes and the operand is streamed
/// exactly once. The verdict is a plain "was anything non-finite"; classifying
/// it (NaN vs infinity, for the error message) is left to the caller's cold
/// path, since it only runs when the input is already being rejected.
///
/// ## Accumulation order (why this is still inside the 1e-5 oracle contract)
/// [`linear_predict_bias`] sums a row's products strictly in ascending `c`.
/// This function splits each row's dot product across
/// [`HOST_DOT_LANES`] independent accumulators and adds those together at the
/// end — a REASSOCIATION, so it is not bit-identical to the kernel. It is the
/// same reassociation every vectorized BLAS `sgemv` (including the one behind
/// `numpy`'s `X @ coef`, which is the sklearn reference) performs, and for the
/// capped feature axis these models fit (`GRAM_EIG_MAX_FEATURES = 64`) it is
/// strictly the more accurate of the two — a `⌈n/L⌉`-term chain per lane rather
/// than one `n`-term chain. `linear_predict_test.rs` gates both paths against
/// the same f64 host reference at the project tolerance.
///
/// ## Parallelism
/// The row axis is split into [`host_units`] CONTIGUOUS chunks over scoped
/// threads, so each thread owns a disjoint, cache-line-aligned run of `out` (no
/// false sharing on the output) and streams a contiguous slab of `x`. The thread
/// count scales with the OPERAND SIZE rather than the core count — see
/// [`HOST_ELEMS_PER_UNIT`] — so a small batch runs on the calling thread instead
/// of paying a fan-out that exceeds its whole cost.
///
/// Generic over `F` (`f32` / `f64`) via a size dispatch onto the two concrete
/// monomorphizations — `F` is opaque arithmetic-wise (its `Float` ops are
/// CubeCL *kernel* ops, not host ones), so the host math is done on the
/// bytemuck-cast primitive view.
pub fn linear_predict_host<F>(
    x: &[F],
    coef: &[F],
    bias: F,
    (m, n): (usize, usize),
) -> Result<HostPrediction<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    linear_predict_host_units(x, coef, bias, (m, n), None)
}

/// [`linear_predict_host`] with an explicit worker-thread count — the test seam
/// for the one property the thread split must have: NONE.
///
/// `units = None` is the production behaviour ([`host_units`] sizes the set from
/// the operand). `Some(u)` pins it, so a test can sweep the split and assert the
/// results are bit-identical.
///
/// This exists so that sweep does NOT go through `MLRS_CPU_UNITS`. That variable
/// is read per call by design ([`crate::capability::cpu_launch_units`]), and
/// libtest runs a binary's `#[test]`s on parallel threads — so `set_var`ing it
/// from a test body both races glibc's `environ` against every sibling test's
/// `getenv` (a real data race, which is why `set_var` is `unsafe`) and silently
/// changes the launch width those siblings run under. An argument has neither
/// problem. `MLRS_CPU_UNITS` remains the knob for whole-process A/B runs, where
/// it is set from the environment and never mutated.
#[doc(hidden)]
pub fn linear_predict_host_units<F>(
    x: &[F],
    coef: &[F],
    bias: F,
    (m, n): (usize, usize),
    units: Option<usize>,
) -> Result<HostPrediction<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), (m, n), coef.len(), 1)?;

    // `Pod: Zeroable`, and an all-zero bit pattern is `0.0` for both float
    // widths — every element is overwritten below, this just gets a typed,
    // initialized buffer without unsafe.
    let mut values: Vec<F> = vec![bytemuck::Zeroable::zeroed(); m];
    let operand_finite = match size_of::<F>() {
        4 => matvec_bias_parallel::<f32>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(coef),
            bytemuck::cast(bias),
            bytemuck::cast_slice_mut(&mut values),
            n,
            units,
        ),
        8 => matvec_bias_parallel::<f64>(
            bytemuck::cast_slice(x),
            bytemuck::cast_slice(coef),
            bytemuck::cast(bias),
            bytemuck::cast_slice_mut(&mut values),
            n,
            units,
        ),
        other => {
            unreachable!("linear_predict_host is f32/f64 only, got a {other}-byte element")
        }
    };
    Ok(HostPrediction {
        values,
        operand_finite,
    })
}

/// What [`linear_predict_host`] produces: the predictions, plus whether every
/// element of the operand it read was finite.
///
/// The two travel together because they come from ONE pass over `x` — see
/// [`linear_predict_host`]'s docs for why re-scanning separately would double
/// the cost of the whole operation.
#[derive(Debug, Clone)]
pub struct HostPrediction<F> {
    /// The length-`m` predictions, `y[r] = Σ_c x[r,c]·coef[c] + bias`.
    ///
    /// **Only meaningful when `operand_finite` is `true`** — check that first.
    /// A `false` verdict means the caller is about to reject the input, so the
    /// producer is free to skip the work: the cpu arm still returns a full
    /// vector (its verdict is fused into the arithmetic, so there was nothing to
    /// skip), while the device arms return an EMPTY vector rather than paying an
    /// `m × n` upload, a launch and an `m`-element read-back for a result that
    /// is immediately discarded.
    pub values: Vec<F>,
    /// `false` if ANY element of `x` was NaN or ±infinity.
    ///
    /// A sklearn-compatible caller rejects the input in that case. `true` means
    /// every element read was finite — note this is a statement about the
    /// OPERAND, not about `values`, which can still overflow to infinity for a
    /// finite-but-extreme `x` (and must not be rejected for that).
    pub operand_finite: bool,
}

/// Memoized host copy of a fitted `(coef, bias)` pair, for the cpu arm of
/// [`linear_predict_from_host`].
///
/// The estimator owns one of these next to its device-resident `coef_` /
/// `intercept_` (the IN-05 `OnceLock` host-mirror idiom the covariance
/// estimators already use for their fitted attributes). The fitted state is
/// immutable — the typestate has no mutating method on the `Fitted` arm — so a
/// value read once is valid for the estimator's whole life.
///
/// ## Why memoize a 64-byte read
/// `DeviceArray::to_host` is `client.read_one`, whose cost is a *synchronization*
/// and barely depends on length: **4.3 µs** for the 16-element `coef` and
/// **4.5 µs** for the 1-element `bias`, measured through the PyO3 boundary on
/// the cpu backend. Reading both on every call put ~8.6 µs of pure
/// synchronization in front of every prediction, which is invisible on a
/// million-row batch and is 30% of the whole call on a small one — and
/// small-batch inference is a real workload (`HOST_ELEMS_PER_UNIT` in this
/// module exists for the same reason). On a discrete GPU the same two reads are
/// two full device syncs, so this is not a cpu-only saving.
///
/// Filled LAZILY, and only by the cpu arm: the device arms never read `coef` /
/// `bias` to host at all, and pay nothing for holding one of these.
#[derive(Debug)]
pub struct HostMirror<F> {
    cell: OnceLock<(Vec<F>, F)>,
}

// Hand-written rather than derived: `#[derive(Default)]` would add a spurious
// `F: Default` bound, which the estimators holding one of these do not require
// of their float parameter.
impl<F> Default for HostMirror<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> HostMirror<F> {
    /// An empty mirror. Estimators construct one per `fit`, alongside the
    /// device buffers it mirrors.
    pub fn new() -> Self {
        Self {
            cell: OnceLock::new(),
        }
    }
}

impl<F: Pod> HostMirror<F> {
    /// The host `(coef, bias)`, reading them back on first call only.
    fn get(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        coef: &DeviceArray<ActiveRuntime, F>,
        bias: &DeviceArray<ActiveRuntime, F>,
    ) -> &(Vec<F>, F) {
        self.cell
            .get_or_init(|| (coef.to_host(pool), bias.to_host(pool)[0]))
    }
}

/// Multi-target twin of [`HostMirror`]: caches a fitted `(coef, bias)` pair
/// where `bias` is length `k` (one intercept per target) instead of a scalar.
/// Kept as its OWN type rather than widening [`HostMirror`] itself — every
/// other dense linear regressor ([`DensePredictHost`](crate) callers:
/// `LinearRegression`/`Lasso`/`ElasticNet`/`LinearSVR`/`BayesianRidge`) is
/// single-target-only and already depends on `HostMirror`'s `(Vec<F>, F)`
/// shape, so this stays Ridge-multi-target-only rather than touching a type
/// four other estimators share.
#[derive(Debug)]
pub struct HostMirrorMulti<F> {
    cell: OnceLock<(Vec<F>, Vec<F>)>,
}

impl<F> Default for HostMirrorMulti<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> HostMirrorMulti<F> {
    /// An empty mirror. The estimator constructs one per multi-target `fit`.
    pub fn new() -> Self {
        Self {
            cell: OnceLock::new(),
        }
    }
}

impl<F: Pod> HostMirrorMulti<F> {
    /// The host `(coef, bias)` — `coef` row-major `n × k`, `bias` length `k` —
    /// reading them back on first call only (see [`HostMirror::get`] for why
    /// this read is worth memoizing).
    fn get(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        coef: &DeviceArray<ActiveRuntime, F>,
        bias: &DeviceArray<ActiveRuntime, F>,
    ) -> &(Vec<F>, Vec<F>) {
        self.cell
            .get_or_init(|| (coef.to_host(pool), bias.to_host(pool)))
    }
}

/// Multi-target twin of [`linear_predict_from_host`]: routes a **host-resident**
/// `m × n` test matrix to the cpu no-upload matvec or the fused device kernel,
/// producing `m × k` row-major predictions plus the operand-finiteness verdict.
///
/// Same backend split as the single-target path: cpu never uploads `x` at all;
/// wgpu/cuda/rocm upload once and run [`linear_predict_multi`]'s fused kernel.
pub fn linear_predict_multi_from_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    coef: &DeviceArray<ActiveRuntime, F>,
    bias: &DeviceArray<ActiveRuntime, F>,
    mirror: &HostMirrorMulti<F>,
    (m, n): (usize, usize),
    k: usize,
) -> Result<HostPrediction<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if crate::capability::active_backend_name() == "cpu" {
        validate_geometry_multi(x.len(), (m, n), coef.len(), bias.len(), k)?;
        let (coef_host, bias_host) = mirror.get(pool, coef, bias);
        return linear_predict_multi_host::<F>(x, coef_host, bias_host, (m, n), k);
    }

    if !operand_all_finite(x) {
        validate_geometry_multi(x.len(), (m, n), coef.len(), bias.len(), k)?;
        return Ok(HostPrediction {
            values: Vec::new(),
            operand_finite: false,
        });
    }

    let x_dev = DeviceArray::from_host(pool, x);
    let out = linear_predict_multi::<F>(pool, &x_dev, coef, bias, (m, n), k)?;
    let values = out.to_host_metered(pool);
    x_dev.release_into(pool);
    out.release_into(pool);
    Ok(HostPrediction {
        values,
        operand_finite: true,
    })
}

/// [`linear_predict_multi_from_host`] with the HOST arm forced on EVERY backend
/// (RIDGE-PREDICT-CUDA-VS-CPU) — see [`linear_predict_from_host_forced_host`]'s
/// docs for the measured justification; this is its multi-target twin, used by
/// [`crate`]'s Ridge multi-target predict.
pub fn linear_predict_multi_from_host_forced_host<F>(
    pool: &BufferPool<ActiveRuntime>,
    x: &[F],
    coef: &DeviceArray<ActiveRuntime, F>,
    bias: &DeviceArray<ActiveRuntime, F>,
    mirror: &HostMirrorMulti<F>,
    (m, n): (usize, usize),
    k: usize,
) -> Result<HostPrediction<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry_multi(x.len(), (m, n), coef.len(), bias.len(), k)?;
    let (coef_host, bias_host) = mirror.get(pool, coef, bias);
    linear_predict_multi_host::<F>(x, coef_host, bias_host, (m, n), k)
}

/// Predict from a **host-resident** `x` — the backend-routing entry point the
/// estimator layer calls when the test matrix arrives from the host (which,
/// coming through the Arrow/PyO3 boundary, is always).
///
/// - **cpu**: [`linear_predict_host`] straight off the caller's slice. No
///   upload, no launch, no read-back (module docs §"The cpu backend does NOT
///   take either kernel" for the measurement that motivates it).
/// - **wgpu / cuda / rocm**: unchanged — upload `x`, run [`linear_predict`]'s
///   fused device kernel, read the length-`m` result back. There the operand
///   has to cross the bus no matter what, and the device does the arithmetic
///   far faster than any host loop; this path is already measured well ahead of
///   cuML/sklearn on a Tesla T4 and is deliberately untouched.
///
/// `coef` / `bias` stay device-resident in the estimator on every backend; the
/// cpu arm needs those two SMALL buffers (length `n ≤ 64` and `1`) on the host,
/// and takes them from the caller's [`HostMirror`] so the read-back happens once
/// per fitted estimator rather than once per call. The fitted-state contract
/// (D-03) is identical across backends.
///
/// [`HostPrediction::operand_finite`] is a real verdict on EVERY backend, so a
/// caller may rely on it uniformly. The cpu arm gets it fused into the matvec
/// for free; the device arms cannot (the arithmetic happens on the device, over
/// a copy) and pay an explicit host scan before the upload — cheap next to the
/// bus transfer that follows it, and the alternative (a verdict that silently
/// means "unchecked" on three of four backends) is exactly the kind of
/// backend-dependent validation gap this project's abstraction is supposed to
/// rule out.
pub fn linear_predict_from_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    coef: &DeviceArray<ActiveRuntime, F>,
    bias: &DeviceArray<ActiveRuntime, F>,
    mirror: &HostMirror<F>,
    (m, n): (usize, usize),
) -> Result<HostPrediction<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if crate::capability::active_backend_name() == "cpu" {
        validate_geometry(x.len(), (m, n), coef.len(), bias.len())?;
        let (coef_host, bias_host) = mirror.get(pool, coef, bias);
        return linear_predict_host::<F>(x, coef_host, *bias_host, (m, n));
    }

    // Scanned BEFORE the upload so a rejected operand never crosses the bus:
    // the caller discards `values` in that case, and for the 64 MiB operand the
    // module docs measure, the skipped upload + launch + read-back is tens of
    // milliseconds of pure waste on an error path.
    if !operand_all_finite(x) {
        validate_geometry(x.len(), (m, n), coef.len(), bias.len())?;
        return Ok(HostPrediction {
            values: Vec::new(),
            operand_finite: false,
        });
    }

    let x_dev = DeviceArray::from_host(pool, x);
    let out = linear_predict::<F>(pool, &x_dev, coef, bias, (m, n))?;
    let values = out.to_host_metered(pool);
    x_dev.release_into(pool);
    out.release_into(pool);
    Ok(HostPrediction {
        values,
        operand_finite: true,
    })
}

/// [`linear_predict_from_host`] with the HOST arm forced on EVERY backend,
/// including cuda/rocm/wgpu (RIDGE-PREDICT-CUDA-VS-CPU, 2026-08-03).
///
/// `linear_predict_from_host`'s "always device on non-cpu backends" default was
/// validated only against cuML/sklearn on a Tesla T4 (7-50x faster — see the
/// `mlrs-linear-predict-optimization`/`-coalesced` project memory) — never
/// against mlrs's OWN [`linear_predict_host`], which did not exist yet when that
/// default shipped. Measured on a Kaggle P100
/// (`ridge_predict_device_vs_host_perf_test.rs`, single-target): the device
/// kernel LOSES to this exact host arithmetic by **10-23x** across every shape
/// tried (`n` 10k-1M, `d` 16-64). The reason generalizes past this one adapter:
/// `predict` is `O(n·d)` compute over the SAME `O(n·d)` transfer — a strictly
/// WORSE compute-to-transfer ratio than `fit`'s `O(n·d²)`, so the GPU's
/// advantage never gets the chance to pay back the upload+launch+readback the
/// way it does on the fit side (`mlrs-ridge-default-cuda` memory's `d ≥ 128`
/// fit crossover has no `predict` analogue in this shape range).
///
/// Used by `Ridge::predict_from_host` UNCONDITIONALLY for single-target
/// predict. Other dense linear regressors (`LinearRegression`/`Lasso`/
/// `ElasticNet`/`LinearSVR`/`BayesianRidge`, which still call
/// [`linear_predict_from_host`] via their shared `DensePredictHost` plumbing in
/// `mlrs-py`) are UNCHANGED — this fix is scoped to Ridge, where it was
/// measured; the same class of fix likely applies to the others too (same
/// kernel, same shape regime) but that is not yet verified on hardware for them
/// and is left as a documented follow-up rather than shipped by inference.
pub fn linear_predict_from_host_forced_host<F>(
    pool: &BufferPool<ActiveRuntime>,
    x: &[F],
    coef: &DeviceArray<ActiveRuntime, F>,
    bias: &DeviceArray<ActiveRuntime, F>,
    mirror: &HostMirror<F>,
    (m, n): (usize, usize),
) -> Result<HostPrediction<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), (m, n), coef.len(), bias.len())?;
    let (coef_host, bias_host) = mirror.get(pool, coef, bias);
    linear_predict_host::<F>(x, coef_host, *bias_host, (m, n))
}

/// Whether every element of a host operand is finite, scanned in parallel —
/// the STANDALONE form of the check [`matvec_bias_rows`] fuses into its row
/// loop. Used by [`linear_predict_from_host`]'s device arms, where the
/// arithmetic runs on the device and there is no host pass to fuse into.
///
/// Same `F` → `f32`/`f64` size dispatch as [`linear_predict_host`].
fn operand_all_finite<F: Pod>(x: &[F]) -> bool {
    match size_of::<F>() {
        4 => all_finite_parallel::<f32>(bytemuck::cast_slice(x)),
        8 => all_finite_parallel::<f64>(bytemuck::cast_slice(x)),
        other => unreachable!("linear predict is f32/f64 only, got a {other}-byte element"),
    }
}

/// [`row_all_finite`] over a whole operand, split across [`host_units`] scoped
/// threads on contiguous chunks.
fn all_finite_parallel<T: HostFloat>(x: &[T]) -> bool {
    let units = host_units(x.len());
    if units <= 1 {
        return row_all_finite(x);
    }
    let chunk = x.len().div_ceil(units);
    std::thread::scope(|scope| {
        let handles: Vec<_> = x
            .chunks(chunk)
            .map(|c| scope.spawn(move || row_all_finite(c)))
            .collect();
        handles
            .into_iter()
            .all(|h| h.join().expect("linear predict finiteness worker panicked"))
    })
}

/// Operand elements one worker thread must be given before spawning it pays.
///
/// The host paths size their thread set by WORK, not by core count:
/// `units = clamp(elems / HOST_ELEMS_PER_UNIT, 1, cpu_launch_units())`. Spawning
/// and joining a `std::thread` costs tens of microseconds, which is the entire
/// budget of a small predict, and small-batch inference (a handful of rows at a
/// time) is a real workload rather than a benchmark tail — an unconditional
/// 16-thread fan-out makes it several times SLOWER than doing the work on the
/// calling thread.
///
/// `1 << 18` (262 144 elements, a 1 MiB `f32` operand) is the measured knee on a
/// 16-core Zen5. Effective bandwidth of `linear_predict_host` by operand size
/// and forced thread count (`MLRS_CPU_UNITS`, GB/s, best of 6):
///
/// | elements | 1 | 2 | 4 | 8 | 16 |
/// |---|---|---|---|---|---|
/// | 160 K   | **21.6** | 14.4 | 17.1 | 11.2 | 6.7 |
/// | 800 K   | 21.4 | 23.9 | **40.0** | 32.7 | 25.1 |
/// | 1.6 M   | 21.2 | 18.2 | **44.2** | 43.4 | 35.0 |
/// | 6.4 M   | 26.7 | 29.5 | 57.0 | **68.8** | 60.7 |
/// | 16 M    | 20.2 | 26.5 | **50.2** | 47.2 | 43.8 |
///
/// Below the knee one thread wins outright (3× at 160 K); above it the curve is
/// broad and flat — anything from 4 threads up is within noise of the peak,
/// because these passes saturate DRAM bandwidth long before they run out of
/// cores. So the constant only has to get the SMALL end right, and a ratio that
/// reaches the core count by a few million elements does that.
const HOST_ELEMS_PER_UNIT: usize = 1 << 18;

/// Worker threads to split `elems` operand elements across — see
/// [`HOST_ELEMS_PER_UNIT`]. Never more than the machine offers
/// ([`crate::capability::cpu_launch_units`], which `MLRS_CPU_UNITS` overrides
/// for A/B), never fewer than one.
fn host_units(elems: usize) -> usize {
    (elems / HOST_ELEMS_PER_UNIT)
        .clamp(1, crate::capability::cpu_launch_units().max(1) as usize)
}

/// Independent accumulators [`host_dot`] splits a row's dot product across.
///
/// Chosen as the natural SIMD group, not tuned per machine: at `-O3` LLVM keeps
/// the fixed-size `[T; 8]` in one AVX2 `f32` register (or two AVX `f64` ones)
/// and turns the body into a single multiply-add pair per chunk, while the 8
/// independent chains hide FP-add latency. It also divides both feature counts
/// the fitted dense linear models actually produce (`16` and the `64` cap), so
/// the scalar remainder loop is usually empty.
const HOST_DOT_LANES: usize = 8;

/// Host float arithmetic shared by the `f32` and `f64` monomorphizations of
/// [`linear_predict_host`]. Deliberately minimal: `+`, `*` and a zero — the
/// operations a dot product needs and nothing that could pull in a libm call.
///
/// In particular this does NOT use `mul_add`: without `target-feature=+fma`
/// (which the default `x86-64` baseline does not have) `f32::mul_add` lowers to
/// a `fmaf` LIBRARY CALL, which is an order of magnitude slower than the
/// `mul`+`add` pair LLVM vectorizes here. (On the `cubecl-cpu` kernels the
/// opposite is true — there `fma()` is one MLIR op instead of two and is a
/// measured win. Different compiler, opposite advice.)
trait HostFloat:
    Copy + Send + Sync + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self>
{
    /// The additive identity the accumulators start from.
    const ZERO: Self;

    /// Whether this value is neither NaN nor ±infinity.
    ///
    /// `f32`/`f64::is_finite` is an exponent-bits comparison, so a loop of them
    /// vectorizes into one packed compare per SIMD group alongside the dot
    /// product's own arithmetic (see [`row_all_finite`]).
    fn finite(self) -> bool;
}

impl HostFloat for f32 {
    const ZERO: f32 = 0.0;

    #[inline]
    fn finite(self) -> bool {
        self.is_finite()
    }
}

impl HostFloat for f64 {
    const ZERO: f64 = 0.0;

    #[inline]
    fn finite(self) -> bool {
        self.is_finite()
    }
}

/// `out[r] = Σ_c x[r·n + c]·coef[c] + bias` over `out.len()` rows, split across
/// scoped threads on contiguous row chunks (see [`linear_predict_host`]).
/// Returns whether every element of `x` it read was finite.
///
/// `units` overrides the work-proportional [`host_units`] count; it is the
/// [`linear_predict_host_units`] test seam and is `None` in production.
fn matvec_bias_parallel<T: HostFloat>(
    x: &[T],
    coef: &[T],
    bias: T,
    out: &mut [T],
    n: usize,
    units: Option<usize>,
) -> bool {
    let units = units.unwrap_or_else(|| host_units(out.len() * n)).max(1);
    if units <= 1 {
        return matvec_bias_rows(x, coef, bias, out, n);
    }

    // Contiguous chunks: thread `i` owns rows `[i·rows, i·rows + chunk.len())`,
    // so its `out` run and its `x` slab are both one unbroken range.
    let rows = out.len().div_ceil(units);
    std::thread::scope(|scope| {
        let handles: Vec<_> = out
            .chunks_mut(rows)
            .enumerate()
            .map(|(i, chunk)| {
                let slab = &x[i * rows * n..(i * rows + chunk.len()) * n];
                scope.spawn(move || matvec_bias_rows(slab, coef, bias, chunk, n))
            })
            .collect();
        // Every chunk is joined before the verdict is folded, so one thread
        // seeing a non-finite element rejects the whole operand.
        handles
            .into_iter()
            .all(|h| h.join().expect("linear_predict_host row worker panicked"))
    })
}

/// The serial row loop — one dot product plus the bias per output element.
/// Returns whether every element of `x` was finite.
fn matvec_bias_rows<T: HostFloat>(x: &[T], coef: &[T], bias: T, out: &mut [T], n: usize) -> bool {
    let mut finite = true;
    for (r, o) in out.iter_mut().enumerate() {
        let row = &x[r * n..(r + 1) * n];
        *o = host_dot(row, coef) + bias;
        // Scanned AFTER the dot product, so the row is already in L1 — this
        // costs ALU work, not a second trip to memory. Accumulated with `&`
        // rather than an early `return` so the loop stays branch-free and
        // vectorizable on the overwhelmingly common all-finite path.
        finite &= row_all_finite(row);
    }
    finite
}

/// Whether every element of `row` is finite.
#[inline]
fn row_all_finite<T: HostFloat>(row: &[T]) -> bool {
    let mut ok = true;
    for v in row {
        ok &= v.finite();
    }
    ok
}

/// `Σ_i row[i]·coef[i]` over [`HOST_DOT_LANES`] independent accumulators, with
/// a scalar remainder for the sub-lane tail. `row` and `coef` are the same
/// length (`n`, validated by the caller).
#[inline]
fn host_dot<T: HostFloat>(row: &[T], coef: &[T]) -> T {
    let mut acc = [T::ZERO; HOST_DOT_LANES];
    let mut rows = row.chunks_exact(HOST_DOT_LANES);
    let mut cols = coef.chunks_exact(HOST_DOT_LANES);
    for (xc, cc) in rows.by_ref().zip(cols.by_ref()) {
        for i in 0..HOST_DOT_LANES {
            acc[i] = acc[i] + xc[i] * cc[i];
        }
    }
    let mut sum = T::ZERO;
    for a in acc {
        sum = sum + a;
    }
    for (xv, cv) in rows.remainder().iter().zip(cols.remainder()) {
        sum = sum + *xv * *cv;
    }
    sum
}

/// Route `predict` to the coalesced shared-tile kernel
/// ([`linear_predict_bias_shared`]) only on the ONE backend where it is
/// confirmed to win: wgpu, in its measured band. Every other backend keeps the
/// GATHER kernel:
/// - **cpu**: its MLIR lowering rejects the `SharedMemory` kernel (the
///   `prims::gram::use_shared_gram` `#[cfg(feature = "cpu")]` precedent).
/// - **cuda / rocm**: a within-session Tesla T4 A/B (5 reps each, both
///   LinearRegression and Ridge, `n = 64`/`m = 100000`) measured GATHER
///   ~15–28% FASTER than the shared kernel, consistently, with zero overlap
///   across 10 comparisons — the opposite of the wgpu result (see the module
///   docs above). rocm is untested but treated the same conservative way
///   pending its own measurement.
/// - **wgpu, `n < PREDICT_SHARED_MIN_FEATURES`**: GATHER's short row stride
///   still coalesces there, and the shared cube's fixed cost would regress it
///   (the measured wgpu crossover).
/// - **`n > PREDICT_MAX_FEATURES`**: the padded shared tile is a comptime
///   `64·65` allocation; a fitted dense linear model never exceeds
///   `GRAM_EIG_MAX_FEATURES = 64`, so this is a defensive bound.
/// - **the tile would not FIT the adapter's shared-memory budget**: the tile is
///   sized against the CUDA 48 KiB budget, but a wgpu adapter can advertise as
///   little as 16 KiB (`f32` tile ≈ 16.5 KiB, `f64` ≈ 33 KiB). Launching a
///   `SharedMemory` kernel larger than `max_shared_memory_size` fails pipeline
///   creation, so we fall back to GATHER (which uses no shared memory) rather
///   than break `predict` where it previously worked. `elem = size_of::<F>()`.
/// - **`LR_PREDICT_GATHER` set**: A/B escape hatch (mirrors `LR_GRAM_GEMM` /
///   `KM_SUMS_GATHER`), also usable on wgpu to re-verify the win. Read once and
///   cached (an inference loop calls `predict` repeatedly; the toggle never
///   changes within a process).
fn use_shared_predict(n: usize, elem: usize) -> bool {
    #[cfg(feature = "wgpu")]
    {
        use std::sync::OnceLock;
        static FORCE_GATHER: OnceLock<bool> = OnceLock::new();
        if *FORCE_GATHER.get_or_init(|| std::env::var("LR_PREDICT_GATHER").is_ok()) {
            return false;
        }
        n >= PREDICT_SHARED_MIN_FEATURES as usize
            && n <= PREDICT_MAX_FEATURES as usize
            && PREDICT_SHARED_ELEMS * elem <= crate::capability::active_max_shared_memory()
    }
    #[cfg(not(feature = "wgpu"))]
    {
        let (_, _) = (n, elem);
        false
    }
}

/// `PREDICT_ROWS_PER_BLOCK`-thread workgroup grid for the shared-tile predict
/// kernel: one cube per row block, folded across the X/Y grid axes so the cube
/// count never exceeds `MAX_GRID_DIM` in any single dimension (the slack cubes
/// are guarded in-kernel by `b < nblocks`). Mirrors `prims::gram::launch_cubes_64`.
fn launch_cubes_block(nblocks: usize) -> (CubeCount, CubeDim) {
    let c = (nblocks as u32).max(1);
    let y = c.div_ceil(MAX_GRID_DIM);
    let x = c.div_ceil(y);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: PREDICT_ROWS_PER_BLOCK, y: 1, z: 1 },
    )
}

/// Validate the inference operand geometry. `x` is `m × n` row-major; `coef`
/// is length `n`; `bias` holds at least the length-1 intercept scalar. Both
/// dims non-zero (an empty test batch / feature axis has no prediction).
fn validate_geometry(
    x_len: usize,
    (m, n): (usize, usize),
    coef_len: usize,
    bias_len: usize,
) -> Result<(), PrimError> {
    if m == 0 || n == 0 || m.checked_mul(n).map(|v| v != x_len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: m,
            cols: n,
            len: x_len,
        });
    }
    if coef_len != n {
        return Err(PrimError::DimMismatch {
            dim: "n_features",
            lhs: coef_len,
            rhs: n,
        });
    }
    if bias_len == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "bias",
            rows: 1,
            cols: 1,
            len: bias_len,
        });
    }
    Ok(())
}
