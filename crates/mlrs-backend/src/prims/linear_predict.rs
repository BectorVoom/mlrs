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
use mlrs_kernels::{linear_predict_bias, linear_predict_bias_shared, PREDICT_ROWS_PER_BLOCK};
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

// ==================== BAYES-GPU — predict(X, return_std=True) ====================

/// `BayesianRidge::predict(X, return_std=True)`'s standard deviation, over a
/// DEVICE-resident test matrix.
///
/// ```text
/// x̃ᵢ     = xᵢ − offset
/// std[i] = √( ‖Mᵀ·x̃ᵢ‖² + noise )
/// ```
///
/// `mt` is `Mᵀ` row-major (`d × d`), the factor of the posterior covariance
/// `Σ = M·Mᵀ`; `offset` is `X_offset_` (length `d`, zeros when the model was fit
/// without an intercept); `noise` is `1/α`. See
/// [`mlrs_kernels::linear_predict::bayes_predict_std`] for why the quadratic
/// form is evaluated through the factor rather than through `Σ` directly.
///
/// One launch, one output buffer, no read-back — the result stays on the device
/// (D-05), so a caller that goes on to reduce it never pays a round trip.
pub fn bayes_predict_std<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    offset: &DeviceArray<ActiveRuntime, F>,
    mt: &DeviceArray<ActiveRuntime, F>,
    (n, d): (usize, usize),
    noise: F,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(x.len(), (n, d), offset.len(), 1)?;
    if mt.len() != d * d {
        return Err(PrimError::ShapeMismatch {
            operand: "mt",
            rows: d,
            cols: d,
            len: mt.len(),
        });
    }

    let out_handle = pool.acquire(n * size_of::<F>());
    let client = pool.client().clone();

    // Slice the row axis so no single dispatch exceeds `STD_MACS_PER_LAUNCH`
    // multiply-adds — the watchdog guard the kernel's `row0` parameter exists
    // for. `chunk` is at least one row, so a `d` large enough to blow the budget
    // on its own still makes progress (one row per launch) rather than looping
    // forever on a zero-length slice.
    // The kernel owns 4 rows per unit, so the launch is over ROW TILES.
    let ntiles = n.div_ceil(STD_ROWS_PER_UNIT);
    let chunk = (STD_MACS_PER_LAUNCH / d.saturating_mul(d).saturating_mul(STD_ROWS_PER_UNIT).max(1))
        .max(1);
    let mut row0 = 0usize;
    while row0 < ntiles {
        let rows = chunk.min(ntiles - row0);
        // SAFETY: every length is a carried/validated element count; the kernel
        // bounds-checks `tile0 + ABSOLUTE_POS < ntiles`, clamps its four row
        // ids to `n - 1`, and reads only `x[i·d + k]`,
        // `offset[k]` and `mt[j·d + k]` for `j, k < d`, all in range by the
        // checks above.
        let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
        let off_arg = unsafe { ArrayArg::from_raw_parts(offset.handle().clone(), offset.len()) };
        let mt_arg = unsafe { ArrayArg::from_raw_parts(mt.handle().clone(), mt.len()) };
        let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };
        let (ccount, cdim) =
            super::launch_dims_1d_folded(rows, crate::capability::gather_launch_width());
        mlrs_kernels::linear_predict::bayes_predict_std::launch::<F, ActiveRuntime>(
            &client,
            ccount,
            cdim,
            x_arg,
            off_arg,
            mt_arg,
            out_arg,
            row0 as u32,
            ntiles as u32,
            n as u32,
            d as u32,
            noise,
        );
        row0 += rows;
    }
    Ok(DeviceArray::from_raw(out_handle, n))
}

/// Multiply-adds one [`bayes_predict_std`] dispatch may cover before the row
/// axis is split (`2³⁰`, ~1.07 G).
///
/// A display-driving GPU cancels any submission that overruns its compositor
/// timeout, and this is the crate's first kernel whose one-launch cost grows as
/// `n·d²` — at `d = 256` over 100 000 rows that is 6.5 G multiply-adds in a
/// single dispatch, which lost the device outright on this environment's wgpu
/// adapter (see the kernel's docs for the failure). The budget is expressed in
/// WORK rather than rows so the bound holds at every `d`.
///
/// The value is set from the SLOW end, because that is the end where the
/// consequence is a crash rather than a few lost percent. This environment's
/// wgpu adapter was measured at **0.72 G multiply-adds/s** on this kernel at
/// `f64` (`d = 128`, 100 000 rows, 1.64 G MACs in 2.28 s) — consumer integrated
/// GPUs run `f64` at a small fraction of `f32`, and an unoptimized `f64` path
/// can be slower still. A first attempt at `2³⁰` was NOT enough there: one
/// dispatch would have been ~1.5 s, and `d = 256` lost the device again.
///
/// `2²⁸` is ~0.37 s per dispatch at that measured rate — inside a typical 2 s
/// compositor timeout with margin — and ~0.4 ms on the compute card this
/// estimator targets, so the ~50 µs dispatch cost (measured on a T4) stays
/// around a tenth of a launch. At the largest ladder rung it is 24 dispatches
/// over a 6.5 G-MAC problem.
const STD_MACS_PER_LAUNCH: usize = 1 << 28;

/// Test rows one unit of
/// [`bayes_predict_std`](mlrs_kernels::linear_predict::bayes_predict_std) owns
/// — the row extent of its `4 × 4` register tile. The launch is sized in TILES,
/// so this is the divisor between the two.
const STD_ROWS_PER_UNIT: usize = 4;

/// [`bayes_predict_std`] for a test matrix that is still on the HOST, routed the
/// same way [`linear_predict_from_host`] routes the mean.
///
/// - **cpu backend** → [`bayes_predict_std_host`], reading the caller's buffer
///   in place. No upload, no launch, and `f64` throughout.
/// - **every device backend** → upload, one launch, read back `n` values.
///
/// The upload is `n · d` elements against `n · d²` multiply-adds, so the ratio
/// of transfer to work is `1/d` — this is the predict path where the device arm
/// has the most room, and the reason it exists: the host arm is `O(n·d²)` of
/// `f64` FMA and nothing on the host makes that cheaper.
///
/// `offset` / `mt` are the host `f64` fitted state; they are narrowed to `F`
/// for the device arm, which is exact for `F = f64` and rounds once for
/// `F = f32` — the same narrowing `coef_` already takes at `fit`. The result is
/// always widened back to `f64`, because sklearn's `return_std` output is
/// `float64` regardless of the design's dtype.
pub fn bayes_predict_std_from_host<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &[F],
    offset: &[f64],
    mt: &[f64],
    noise: f64,
    (n, d): (usize, usize),
) -> Result<Vec<f64>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    if n == 0 || d == 0 || n.checked_mul(d).map(|v| v != x.len()).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand: "x",
            rows: n,
            cols: d,
            len: x.len(),
        });
    }
    if offset.len() != d {
        return Err(PrimError::DimMismatch {
            dim: "n_features",
            lhs: offset.len(),
            rhs: d,
        });
    }
    if mt.len() != d * d {
        return Err(PrimError::ShapeMismatch {
            operand: "mt",
            rows: d,
            cols: d,
            len: mt.len(),
        });
    }

    if use_host_std(d) {
        return Ok(bayes_predict_std_host(x, offset, mt, noise, (n, d)));
    }

    let off_f: Vec<F> = offset.iter().map(|v| mlrs_core::f64_to_host::<F>(*v)).collect();
    let mt_f: Vec<F> = mt.iter().map(|v| mlrs_core::f64_to_host::<F>(*v)).collect();
    let x_dev = DeviceArray::from_host(pool, x);
    let off_dev = DeviceArray::from_host(pool, &off_f);
    let mt_dev = DeviceArray::from_host(pool, &mt_f);
    let out = bayes_predict_std::<F>(
        pool,
        &x_dev,
        &off_dev,
        &mt_dev,
        (n, d),
        mlrs_core::f64_to_host::<F>(noise),
    )?;
    let host = out.to_host_metered(pool);
    x_dev.release_into(pool);
    off_dev.release_into(pool);
    mt_dev.release_into(pool);
    out.release_into(pool);
    Ok(host.into_iter().map(mlrs_core::host_to_f64::<F>).collect())
}

/// Which arm [`bayes_predict_std_from_host`] takes: the host sweep, or upload +
/// [`bayes_predict_std`].
///
/// Two backends take the host arm, for two DIFFERENT measured reasons.
///
/// **cpu**, for the reason this module's docs give for the mean: "device" memory
/// IS host memory there, so the upload is a pure `memcpy` of the whole test
/// matrix and the launch spawns one OS thread per unit against `-O0` JIT'd code.
///
/// **wgpu**, because the kernel is `f64` and consumer integrated GPUs are not.
/// Measured on this environment's adapter (`n_test = 100 000`, min-of-N):
///
/// | `d` | host sweep | device kernel | |
/// |---|---|---|---|
/// | 16 | 5.1 ms | 60.6 ms | 0.08× |
/// | 64 | 24.2 ms | 598 ms | 0.04× |
/// | 128 | 119 ms | 2 190 ms | 0.05× |
/// | 256 | — | **device lost** | — |
///
/// That is a 12–25× LOSS at every size, and at `d = 256` not a loss but a
/// crash: the adapter also drives the display, and 6.5 G `f64` multiply-adds at
/// its measured ~0.72 G MAC/s is ~9 s of GPU time, far past any compositor
/// timeout. The `STD_MACS_PER_LAUNCH` split bounds each DISPATCH but not the
/// SUBMISSION — cubecl queues dispatches and flushes them at the read-back, so
/// the driver still sees one long job — and no dispatch size can make a 9 s
/// problem fit under a 2 s timer anyway. Routing wgpu to the host sweep is
/// therefore not a workaround for the crash, it is the faster arm on its own
/// merits; the crash is just what made the gap impossible to ignore. This is the
/// same shape of measured, backend-specific routing as
/// [`use_shared_predict`]'s wgpu gate, with the sign reversed.
///
/// cuda and rocm keep the device arm: they are the backends with real `f64`
/// silicon behind them, and (on cuda) no display watchdog.
///
/// `MLRS_BAYES_STD_HOST` overrides the verdict in BOTH directions (`1` forces
/// the host sweep on a device backend, `0` forces the device arm anywhere), read
/// through [`crate::abflag`] so the equivalence test can scope it to its own
/// thread without an environment data race. The device-forcing direction is the
/// one the test needs: without it, "these two arms agree" would compare the host
/// arm against ITSELF on cpu and wgpu — passing while gating nothing.
fn use_host_std(d: usize) -> bool {
    if let Some(v) = crate::abflag::var("MLRS_BAYES_STD_HOST") {
        return v != "0";
    }
    if matches!(crate::capability::active_backend_name(), "cpu" | "wgpu") {
        return true;
    }
    d < STD_DEVICE_MIN_FEATURES
}

/// Fitted feature count below which the HOST-INGRESS `predict_std` keeps the
/// host sweep even on a device backend.
///
/// This path uploads, so it pays `O(n·d)` of transfer for `O(n·d²)` of
/// arithmetic — the same `d`-governed trade as the fit ingress, and on the same
/// ~0.28 GB/s Kaggle bus (`prims::normal_eq::device_fit_preferred`). Unlike the
/// fit, it does cross over, because the exponent gap is a full factor of `d`
/// rather than `d/2` and the register-tiled kernel closed most of the compute
/// side. Measured on a Tesla P100, `n_test = 100 000`, `f64`, min-of-5, upload
/// INSIDE the timer:
///
/// | `d` | host sweep | device kernel | |
/// |---|---|---|---|
/// | 16 | 8.2 ms | 21.6 ms | 0.38× |
/// | 64 | 130.3 ms | 131.6 ms | 0.99× |
/// | 128 | 526.6 ms | 412.4 ms | **1.28×** |
/// | 256 | 2 152.7 ms | 1 328.3 ms | **1.62×** |
///
/// `128` rather than `64`: the `d = 64` rung is a dead heat, and a threshold
/// placed on a tie flips arms on measurement noise while buying nothing.
///
/// Note this bound governs ONLY the uploading ingress.
/// [`BayesianRidge::predict_std`](mlrs_algos) over an already-resident design
/// does not consult it and always runs the kernel — correctly, since it beat
/// the host arm at EVERY `d` measured there (11.2× / 11.5× / 12.5× / 3.7×) with
/// no transfer to amortize.
///
/// The crossover is machine-specific in two directions at once: a host with
/// more cores pushes it up, a bus faster than 0.28 GB/s pulls it down. This
/// value is the one measured configuration; `MLRS_BAYES_STD_HOST` re-takes the
/// measurement elsewhere.
const STD_DEVICE_MIN_FEATURES: usize = 128;

/// [`bayes_predict_std`] entirely on the host, in `f64`, reading the caller's
/// borrowed `x` in place — the cpu-backend arm.
///
/// Rows are split into [`host_units`] CONTIGUOUS chunks over scoped threads,
/// the same work-proportional sizing (and the same reason for it) as
/// [`linear_predict_host`]: each worker owns a disjoint run of `out`, so there
/// is no false sharing, and a batch too small to amortize a fan-out runs on the
/// calling thread. The work here is `O(n·d²)` rather than the mean's `O(n·d)`,
/// so the split is sized off `n · d` — the OPERAND, matching `host_units`'
/// contract — but reaches the core count far sooner in wall-clock terms.
///
/// Accumulation is `f64` on every path, including for an `f32` design, because
/// the caller's `mt`/`offset` are `f64` and widening the design costs nothing
/// inside a loop that is already FMA-bound.
pub fn bayes_predict_std_host<F>(
    x: &[F],
    offset: &[f64],
    mt: &[f64],
    noise: f64,
    (n, d): (usize, usize),
) -> Vec<f64>
where
    F: Pod + Sync,
{
    let mut out = vec![0.0f64; n];
    let units = host_units(n * d).min(n.max(1));
    if units <= 1 {
        std_rows::<F>(x, offset, mt, noise, d, &mut out);
        return out;
    }
    let rows = n.div_ceil(units);
    std::thread::scope(|scope| {
        for (u, band) in out.chunks_mut(rows).enumerate() {
            let slab = &x[u * rows * d..(u * rows + band.len()) * d];
            scope.spawn(move || std_rows::<F>(slab, offset, mt, noise, d, band));
        }
    });
    out
}

/// One worker's row band for [`bayes_predict_std_host`]: `out[i] = √(‖Mᵀx̃ᵢ‖² +
/// noise)` for each row of `x`.
///
/// ## Blocked over ROWS, because the naive form is bandwidth-bound
/// The obvious loop — centre one row, then dot it against all `d` rows of `mt` —
/// is `O(d²)` of arithmetic per row but also `O(d²)` of TRAFFIC per row, because
/// it walks the whole `mt` matrix once per test row. At `d = 256` that is 512 KiB
/// per row, well past any core's private cache, so `mt` is re-streamed from L3 or
/// DRAM 100 000 times: **51 GB moved to do 13 GFLOP**. Measured on a 4-vCPU VM it
/// ran at 6.1 GFLOP/s, and scikit-learn's `(X @ sigma_ * X).sum(axis=1)` — a
/// blocked BLAS GEMM — beat it **8×** on the same machine and the same data.
///
/// Rows are therefore processed [`STD_ROW_BLOCK`] at a time: the block's centred
/// rows (`STD_ROW_BLOCK · d` doubles, 64 KiB at `d = 256`) are materialized once
/// and stay resident, and each row of `mt` is read ONCE per block instead of
/// once per row, cutting `mt`'s traffic by `STD_ROW_BLOCK`. The arithmetic is
/// unchanged — same products, same per-`j` accumulation order — so this is a
/// scheduling change, not a numerical one.
///
/// ## What that actually bought: essentially nothing, and the honest reason
/// Re-measured on the same VM it is **within noise of the unblocked form**
/// (2 269 ms vs 2 135 ms raw at `d = 256`, on a run whose `fit` column was
/// uniformly ~10% slower, so ~3% faster once normalized). The blocking trades
/// `mt`'s DRAM traffic for `xc`'s: with the `j` loop outside, the block's 64 KiB
/// of centred rows is re-read `d` times from L2, which is cheaper per byte but
/// not free. Cutting the total requires blocking BOTH axes with a register tile
/// — i.e. writing a GEMM, which is exactly what scikit-learn gets for free by
/// calling OpenBLAS, and why it still leads this arm ~8× at `d = 256`.
///
/// It is kept because the traffic argument is sound and the measurement is
/// neutral-to-slightly-positive, and recorded honestly because the obvious
/// next reader will otherwise re-derive the same idea and expect the 8×. The
/// device kernel took the register-tile route instead
/// ([`mlrs_kernels::linear_predict::bayes_predict_std`]'s `4 × 4` tile).
///
/// Centring is hoisted out of the `j` loop for the same reason it was already:
/// the `d` subtractions are paid once per row rather than once per `(row, j)`.
fn std_rows<F: Pod + Sync>(x: &[F], offset: &[f64], mt: &[f64], noise: f64, d: usize, out: &mut [f64]) {
    let wide = |v: F| -> f64 {
        match size_of::<F>() {
            4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
            8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
            other => unreachable!("bayes predict_std is f32/f64 only, got a {other}-byte element"),
        }
    };
    let mut xc = vec![0.0f64; STD_ROW_BLOCK * d];
    for (blk, qblk) in out.chunks_mut(STD_ROW_BLOCK).enumerate() {
        let r0 = blk * STD_ROW_BLOCK;
        let rows = qblk.len();

        // Centre this block's rows once. `xc` is reused across blocks, so the
        // allocation is per WORKER, not per block.
        for r in 0..rows {
            let src = &x[(r0 + r) * d..(r0 + r + 1) * d];
            let dst = &mut xc[r * d..(r + 1) * d];
            for (k, slot) in dst.iter_mut().enumerate() {
                *slot = wide(src[k]) - offset[k];
            }
        }

        // `noise` is seeded here rather than added at the end so the block's
        // accumulators need no second pass.
        for q in qblk.iter_mut() {
            *q = noise;
        }
        for j in 0..d {
            let mrow = &mt[j * d..(j + 1) * d];
            for (r, q) in qblk.iter_mut().enumerate() {
                let acc: f64 = mrow
                    .iter()
                    .zip(xc[r * d..(r + 1) * d].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                *q += acc * acc;
            }
        }
        for q in qblk.iter_mut() {
            *q = q.sqrt();
        }
    }
}

/// Test rows [`std_rows`] centres and holds resident while it streams `mt` past
/// them.
///
/// The block's working set is `STD_ROW_BLOCK · d` doubles — 64 KiB at
/// `d = 256`, 16 KiB at `d = 64` — which is sized to sit in L2 alongside the
/// `d`-element `mt` row being streamed, not in L1. That is the right target:
/// the value being bought is `mt`'s DRAM traffic (cut by this factor), and `mt`
/// is streamed exactly once per block either way, so a bigger block buys less
/// and less while pushing the centred rows out of cache.
const STD_ROW_BLOCK: usize = 32;
