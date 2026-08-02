//! Non-negative ridge host API — the `Ridge(positive=True)` / `solver='lbfgs'`
//! solve, run entirely on-device by [`mlrs_kernels::ridge_nnls_cd`].
//!
//! ## What this replaces
//! The `positive` arm's operands are the `d×d` Gram and the length-`d` `Xᵀy`
//! that `Ridge` has ALREADY formed on-device via [`crate::prims::gram::gram_xty`].
//! Before this prim, that pair was read back to the host, solved by
//! `mlrs_algos::linear::ridge_solvers::nonnegative_cd`, and the resulting `coef`
//! re-uploaded — two synchronising read-backs and an upload around a solve whose
//! every operand was already in device memory. [`ridge_nnls`] launches ONE cube
//! that runs the whole sweep loop, including the convergence test, and writes
//! `coef` straight into a pooled device buffer: no host round-trip at all.
//!
//! ## Dispatch ([`device_nnls_applicable`])
//! The kernel is a genuine parallel launch on a GPU and a pathology on cpu (one
//! OS thread per unit spin-waiting at `2·d` barriers per sweep, at LLVM `-O0` —
//! the `prims::eig` / KNN / HDBSCAN cpu finding). The host twin therefore stays
//! the cpu arm, and also carries any `d` above the kernel's cube-dim cap. The
//! caller (`mlrs_algos::linear::ridge`) asks this module which arm applies
//! rather than the reverse, because the host twin lives in the algos crate.
//!
//! Tests live in `crates/mlrs-backend/tests/nnls_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{f64_to_host, PrimError};
use mlrs_kernels::{ridge_intercept, ridge_nnls_cd, NNLS_MAX_DIM};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Sweep cap when the caller passes `max_iter = None` — the host twin's
/// `nonnegative_cd` default, kept identical so the two arms stop at the same
/// place.
pub const NNLS_DEFAULT_MAX_ITER: usize = 1000;

/// Solve `min_{w ≥ 0} ½·wᵀGw − (Xᵀy)ᵀw + ½α‖w‖²` on-device, returning the
/// device-resident length-`d` non-negative coefficient vector.
///
/// - `gram` is the row-major `d × d` raw Gram `XᵀX` (TRUSTED symmetric, D-06 —
///   `gram_xty`'s output is symmetric by construction); `xty` is `Xᵀy`.
/// - `alpha` is the L2 penalty (Gram diagonal only — the intercept is recovered
///   by the caller after the solve, D-05).
/// - `tol` / `max_iter` are the sweep stop; `max_iter = None` takes
///   [`NNLS_DEFAULT_MAX_ITER`].
/// - Geometry is validated BEFORE the unsafe launch (ASVS V5): `gram.len() ==
///   d*d`, `xty.len() == d`, `d` non-zero and within the kernel's cap.
///
/// Callers must gate on [`device_nnls_applicable`] first; an over-cap or
/// mis-shaped `d` returns [`PrimError::ShapeMismatch`] rather than launching.
///
/// Generic over the float element type `F` (`f32` / `f64`).
pub fn ridge_nnls<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    gram: &DeviceArray<ActiveRuntime, F>,
    xty: &DeviceArray<ActiveRuntime, F>,
    d: usize,
    alpha: f64,
    tol: f64,
    max_iter: Option<usize>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_geometry(gram.len(), xty.len(), d)?;

    let elem = size_of::<F>();
    let w_handle = pool.acquire(d * elem);
    let client = pool.client().clone();

    // ONE cube of `d` units — unit `i` owns feature `i` (the `jacobi_eig_sweep`
    // launch shape). `d <= NNLS_MAX_DIM` is validated above, and NNLS_MAX_DIM is
    // the portable max workgroup size X.
    let count = CubeCount::Static(1, 1, 1);
    let dim = CubeDim {
        x: d as u32,
        y: 1,
        z: 1,
    };

    // The sweep cap saturates to u32: the host twin's default is 1000 and
    // sklearn's `max_iter` is a positive integer, so this only guards an absurd
    // caller value rather than changing any reachable behaviour.
    let sweeps = max_iter.unwrap_or(NNLS_DEFAULT_MAX_ITER).min(u32::MAX as usize) as u32;

    // SAFETY: the lengths are the geometry validated immediately above (d*d, d,
    // d), never raw caller values; the kernel bounds every loop by the runtime
    // `d` and idles units with `i >= d`.
    let gram_arg = unsafe { ArrayArg::from_raw_parts(gram.handle().clone(), d * d) };
    let xty_arg = unsafe { ArrayArg::from_raw_parts(xty.handle().clone(), d) };
    let w_arg = unsafe { ArrayArg::from_raw_parts(w_handle.clone(), d) };

    ridge_nnls_cd::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        gram_arg,
        xty_arg,
        w_arg,
        d as u32,
        f64_to_host::<F>(alpha),
        f64_to_host::<F>(tol),
        sweeps,
    );

    Ok(DeviceArray::from_raw(w_handle, d))
}

/// Can the non-negative ridge solve run on-device for this backend, element
/// width, and feature count?
///
/// `false` sends the caller to the host twin
/// (`mlrs_algos::linear::ridge_solvers::nonnegative_cd`), which is always
/// correct — this predicate only ever chooses between two arms that agree.
///
/// - **cpu** — the single-cube kernel is `d` OS THREADS spin-waiting at `2·d`
///   barriers per sweep under an LLVM `-O0` JIT (`cubecl-cpu`'s execution
///   model). That is the same GPU-shaped-kernel pathology that made `eig`,
///   KNN, and HDBSCAN host arms; the native serial CD is far faster.
/// - **`d` outside `1..=NNLS_MAX_DIM`** — the kernel launches one unit per
///   feature, and `NNLS_MAX_DIM` is the portable maximum workgroup size X.
///   Wider problems stay on the host rather than losing support.
/// - **shared memory** — the kernel stages `w` and `g` at the comptime cap
///   (`2 · NNLS_MAX_DIM · size_of::<F>()`, plus the 2-slot broadcast) whatever
///   the runtime `d`, so an adapter below that budget must take the host arm.
///   This is the `prims::eig::eig_shared_memory_fits` guard; at 2 KiB (`f32`) /
///   4 KiB (`f64`) it should never bind, but a silent pipeline-creation failure
///   would surface as an all-zero `coef_`, so it is checked rather than assumed.
///
/// `MLRS_RIDGE_NNLS_HOST=1` forces the host arm anywhere, for on-target A/B
/// (the `MLRS_EIG_HOST` knob precedent — read through
/// [`crate::abflag`], never `std::env` directly).
pub fn device_nnls_applicable<F>(d: usize) -> bool {
    if crate::capability::active_backend_name() == "cpu" {
        return false;
    }
    if crate::abflag::is_on("MLRS_RIDGE_NNLS_HOST") {
        return false;
    }
    if d == 0 || d > NNLS_MAX_DIM as usize {
        return false;
    }
    let needed = (2 * NNLS_MAX_DIM as usize + 2) * std::mem::size_of::<F>();
    needed <= crate::capability::active_max_shared_memory()
}

/// Validate the non-negative-ridge operand geometry (ASVS V5). `gram` must be
/// the square `d × d` normal matrix and `xty` the length-`d` right-hand side,
/// with `d` non-zero and within the single-cube kernel's [`NNLS_MAX_DIM`] cap.
fn validate_geometry(gram_len: usize, xty_len: usize, d: usize) -> Result<(), PrimError> {
    if d == 0 || d > NNLS_MAX_DIM as usize || d.checked_mul(d).map(|v| v != gram_len).unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "nnls.gram",
            rows: d,
            cols: d,
            len: gram_len,
        });
    }
    if xty_len != d {
        return Err(PrimError::ShapeMismatch {
            operand: "nnls.xty",
            rows: d,
            cols: 1,
            len: xty_len,
        });
    }
    Ok(())
}

/// `intercept = ȳ − x̄·coef` on-device, returning the length-1 device buffer.
///
/// The last host round-trip in the `positive` fit. Before this, recovering the
/// intercept read `x̄`, `ȳ` and `coef` back (three BLOCKING read-backs) to do a
/// `d`-term dot on the CPU and upload the single scalar again — four
/// synchronisation points to produce one number from operands that were all
/// already resident.
///
/// Geometry is validated before the launch (ASVS V5): `xmean.len() == coef.len()
/// == d`, `ymean.len() == 1`, `d` non-zero and within the kernel's cap.
///
/// See [`mlrs_kernels::ridge_intercept`] for why this runs on a single unit,
/// and for the accumulator-width caveat that keeps the host twin alive.
pub fn ridge_intercept_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    xmean: &DeviceArray<ActiveRuntime, F>,
    ymean: &DeviceArray<ActiveRuntime, F>,
    coef: &DeviceArray<ActiveRuntime, F>,
    d: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // No `NNLS_MAX_DIM` here: that cap belongs to `ridge_nnls`, which launches
    // one unit PER FEATURE. This kernel is a single unit walking `c < d`, so any
    // `d` is in range — and the default (`positive = false`) arm reaches it at
    // `d` the non-negative solve never sees.
    if d == 0 || xmean.len() != d || coef.len() != d {
        return Err(PrimError::ShapeMismatch {
            operand: "ridge_intercept.xmean",
            rows: d,
            cols: 1,
            len: xmean.len(),
        });
    }
    if ymean.len() != 1 {
        return Err(PrimError::ShapeMismatch {
            operand: "ridge_intercept.ymean",
            rows: 1,
            cols: 1,
            len: ymean.len(),
        });
    }

    let elem = size_of::<F>();
    let out = pool.acquire(elem);
    let client = pool.client().clone();

    // SAFETY: lengths are the geometry validated immediately above; the kernel
    // walks `c < d` only and writes a single slot.
    let xm = unsafe { ArrayArg::from_raw_parts(xmean.handle().clone(), d) };
    let ym = unsafe { ArrayArg::from_raw_parts(ymean.handle().clone(), 1) };
    let cf = unsafe { ArrayArg::from_raw_parts(coef.handle().clone(), d) };
    let ov = unsafe { ArrayArg::from_raw_parts(out.clone(), 1) };

    ridge_intercept::launch::<F, ActiveRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim { x: 1, y: 1, z: 1 },
        xm,
        ym,
        cf,
        ov,
        d as u32,
    );

    Ok(DeviceArray::from_raw(out, 1))
}
