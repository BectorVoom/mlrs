//! Covariance / XᵀX (Gram) host API (PRIM-04) — column-mean centering, the
//! Gram matrix `AᵀA` via GEMM with the `transa` transpose flag (no materialized
//! transpose buffer, D-09), and `1/(n_samples − ddof)` normalisation for both
//! the population (`ddof = 0`) and sample (`ddof = 1`) conventions.
//!
//! ## Composition (no new kernel — D-09 / T-0204-SC)
//! Covariance is a pure composition of the already-validated Phase-2 primitives:
//!   1. **Column-mean centering** — `prims::reduce::column_reduce(.., Mean, ..)`
//!      (Plan 02) gives the per-feature mean; each column is centred by
//!      subtracting it (two-pass for numerical stability, RESEARCH Pitfall 4).
//!   2. **`AᵀA` via GEMM(transa = true)** — `prims::gemm::gemm` (Plan 01) with
//!      `transa = true` reads `A`'s transpose as a logical `(shape, strides)`
//!      swap, so the `n_features × n_features` Gram is produced with NO
//!      materialized transpose buffer (D-09 / D-06).
//!   3. **`1/(n − ddof)` scale** — the `mlrs_kernels::scale` per-element kernel
//!      (Plan 03 `elementwise.rs`) folds the normalisation in place.
//! Zero new external dependencies (T-0204-SC) and no new device kernels.
//!
//! ## GEMM-output-buffer reuse (D-10 gate 3 — load-bearing for Plan 05)
//! The internal GEMM is driven into a single pool-acquired output buffer (its
//! `out` handle), and that SAME handle is then passed as the `scale` kernel's
//! output target — the normalisation is applied **in place** over the Gram
//! buffer. The `DeviceArray` this function returns therefore wraps the *exact*
//! GEMM output handle, never a second parallel allocation. Plan 05's D-10
//! memory-gate assertion 3 ("Gram reuses the GEMM buffer") relies on this: after
//! a GEMM of the same `n_features × n_features` output shape, computing
//! covariance does not bump `PoolStats.allocations` for a fresh Gram buffer —
//! the Gram handle IS the GEMM handle (or, when the caller supplies `out` via
//! D-11, the caller's buffer threads straight through GEMM → scale).
//!
//! ## Device residency (D-05)
//! Inputs and the Gram result stay on the device as [`DeviceArray`]s; this API
//! performs NO host read-back (the device-residency grep gate over this file is
//! `0`). The column means are obtained through the Plan-02 reduction (whose own
//! internal host slicing is the reduction's behaviour, not a covariance
//! mid-pipeline round-trip) and the centred matrix is re-uploaded once before
//! the GEMM; the GEMM → scale chain is `DeviceArray` → `DeviceArray` on device.
//! Scratch + the output buffer are drawn from the [`BufferPool`] (D-11).
//!
//! Tests live in `crates/mlrs-backend/tests/covariance_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::{center_columns, scale};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gemm::gemm;
use crate::prims::reduce::{column_reduce, ReducePath, ScalarOp};
use crate::runtime::ActiveRuntime;

/// Compute the covariance / Gram matrix of `a` (`n_samples × n_features`,
/// row-major) as the column-centred `AᵀA` normalised by `1/(n_samples − ddof)`,
/// returning the device-resident `n_features × n_features` result.
///
/// - `a` is the row-major `n_samples × n_features` data matrix (observations in
///   rows, features in columns — the `rowvar = False` convention `np.cov`
///   matches when called as `np.cov(A, rowvar=False, ddof=ddof)`).
/// - Shapes are validated (`n_samples * n_features == a.len()`) BEFORE any
///   launch (D-04 / T-0204-02); a mismatch returns [`PrimError::ShapeMismatch`].
/// - `ddof = 0` is the population normalisation (`1/n`); `ddof = 1` is the
///   sample normalisation (`1/(n − 1)`) — both selectable, pinned by the
///   `np.cov` fixtures (D-12).
/// - The `n_features × n_features` Gram is acquired from `pool` when `out` is
///   `None`, else the supplied buffer is reused (D-11). The GEMM output buffer
///   is itself the scaled-in-place result — covariance reuses it rather than
///   allocating a parallel one (D-10 gate 3). The result stays device-resident
///   (D-05): NO host round-trip inside this API.
///
/// ## Internal reduction path (CR-01 / D-03)
/// The column-mean reduction is an INTERNAL implementation detail of covariance,
/// not a caller-visible kernel choice, so it always runs on the always-portable
/// [`ReducePath::Shared`] path. The plane (subgroup) path is capability-gated and
/// returns `None` on adapters without subgroup support (e.g. cpu); forwarding a
/// caller-chosen `Plane` into the mean term would unwrap that `None` and PANIC.
/// Covariance therefore exposes NO `path` parameter; the mean reduction is
/// unconditionally shared-path-backed and can never be plane-gated to `None`.
///
/// ## `n_samples - ddof` must be positive (WR-01)
/// The `1/(n_samples - ddof)` normalisation divides by `n_samples - ddof`; when
/// it is `<= 0` (e.g. a single-sample matrix with `ddof = 1`) the result would
/// be `inf`/`NaN`. This is rejected at the boundary with
/// [`PrimError::DimMismatch`] before any launch.
///
/// Generic over the float element type `F` (`f32` / `f64`); the f64 path is
/// capability-gated by the caller via `skip_f64_with_log`.
pub fn covariance<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    (n_samples, n_features): (usize, usize),
    ddof: u32,
    out: Option<DeviceArray<ActiveRuntime, F>>,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    // --- D-04 / T-0204-02 / WR-01: validate geometry (incl. n_samples-ddof > 0)
    //     BEFORE any unsafe launch. ---
    validate_geometry(a.len(), (n_samples, n_features), ddof, out.as_ref().map(DeviceArray::len))?;

    // --- 1. Column means (length n_features) via the Plan-02 column reduction.
    //        Two-pass centring for stability (RESEARCH Pitfall 4): mean first,
    //        then subtract per column. column_reduce is shared-path-backed (the
    //        plane path is gated separately), so it is always `Some` here. The
    //        means stay device-resident as `means_dev`. ---
    // CR-01: force the always-portable Shared path for the INTERNAL mean term —
    // never the caller's choice — so the reduction is never plane-gated to None
    // (which would panic the `.expect` below on a non-subgroup adapter, e.g. cpu).
    let means_dev = column_reduce::<F>(pool, a, n_samples, n_features, ScalarOp::Mean, ReducePath::Shared)?
        .expect("shared path is never plane-gated to None");

    // --- 2. Centre the columns on the DEVICE: centred[r, c] = A[r, c] − mean[c]
    //        via the `center_columns` per-element kernel (broadcasting the
    //        length-n_features means). This keeps covariance.rs device-resident
    //        (no host round-trip here, D-05); the Gram chain below is
    //        DeviceArray → DeviceArray. ---
    let a_len = n_samples * n_features;
    let elem = size_of::<F>();
    let centred_handle = pool.acquire(a_len * elem);
    let client = pool.client().clone();
    let (ccount, cdim) = launch_dims_1d(a_len);
    // SAFETY: `a_len`/`n_features` are the carried/validated element counts; the
    // kernel bounds-checks `tid < a.len()` and reads `mean[tid % cols]` for
    // `cols = n_features` (mitigates T-0204-01).
    let a_arg = unsafe { ArrayArg::from_raw_parts(a.handle().clone(), a_len) };
    let mean_arg = unsafe { ArrayArg::from_raw_parts(means_dev.handle().clone(), n_features) };
    let centred_arg = unsafe { ArrayArg::from_raw_parts(centred_handle.clone(), a_len) };
    center_columns::launch::<F, ActiveRuntime>(
        &client,
        ccount,
        cdim,
        a_arg,
        mean_arg,
        centred_arg,
        // scalar u32 by value (cubecl 0.10, like dist_combine_clamp).
        n_features as u32,
    );
    let centred_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_raw(centred_handle, a_len);

    // CR-02 / WR-07: the column means are TRANSIENT scratch — consumed by the
    // `center_columns` launch above and never read again (not returned to the
    // caller). Release them at their TRUE byte size (`n_features * size_of::<F>`)
    // so the buffer conserves `live_bytes` and lands on the free-list for reuse.
    // The consuming kernel is already launched and same-stream-ordered before any
    // later kernel that could reuse this buffer, so this is not a live-aliasing
    // release.
    means_dev.release_into(pool);

    // --- 3. Gram AᵀA via GEMM(transa = true): the stored matrix is
    //        (n_samples × n_features); transa reads it as its transpose
    //        (n_features × n_samples) with NO transpose buffer (D-06 / D-09),
    //        producing the n_features × n_features Gram. The GEMM output buffer
    //        is reused as covariance's output (D-10 gate 3): we drive GEMM into
    //        `out` (the caller's buffer when supplied, else a pool acquisition)
    //        and then scale THAT SAME buffer in place. ---
    //
    // logical lhs shape (m, k) = (n_features, n_samples); transa=true ⇒ stored
    // (n_samples, n_features), which is exactly `centred`'s layout.
    let gram = gemm::<F>(
        pool,
        &centred_dev,
        (n_features, n_samples),
        &centred_dev,
        // logical rhs shape (k, n) = (n_samples, n_features); transb=false ⇒
        // stored (n_samples, n_features), again `centred`'s layout.
        (n_samples, n_features),
        true,
        false,
        out,
    )?;

    // CR-02 / WR-07: the centred copy is TRANSIENT scratch — consumed by the GEMM
    // above (both lhs and rhs read it) and never read again. The GEMM output
    // (`gram`) is a SEPARATE handle (a fresh pool acquisition, or the caller's
    // `out`), so releasing `centred_dev` does NOT touch the returned buffer.
    // Release at its TRUE byte size so `live_bytes` is conserved and the buffer is
    // reusable; the GEMM's reads are same-stream-ordered before any reuse.
    centred_dev.release_into(pool);

    // --- 4. Normalise by 1/(n_samples − ddof) IN PLACE over the GEMM output
    //        buffer (the load-bearing reuse for D-10 gate 3). ddof = 0 ⇒ 1/n
    //        (population), ddof = 1 ⇒ 1/(n−1) (sample). The scale kernel writes
    //        the Gram handle back into itself (input == output handle), so the
    //        returned DeviceArray wraps the EXACT GEMM output handle. ---
    let denom = (n_samples as i64) - (ddof as i64);
    let factor = recip::<F>(denom);

    let gram_len = n_features * n_features;
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(gram_len);

    // SAFETY: `gram_len` is the carried Gram element count (n_features², itself
    // derived from the validated `a.len()`); the `scale` kernel bounds-checks
    // `tid < input.len()` (mitigates T-0204-01). input and output are the SAME
    // handle so the scale is applied in place over the reused GEMM buffer.
    let in_arg = unsafe { ArrayArg::from_raw_parts(gram.handle().clone(), gram_len) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(gram.handle().clone(), gram_len) };
    scale::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, factor);

    // The result IS the GEMM output buffer, scaled in place (D-10 gate 3), and
    // stays device-resident (D-05).
    Ok(gram)
}

/// Validate the covariance operand geometry (D-04 / T-0204-02 / WR-01). `a` is
/// `n_samples × n_features`; `n_samples * n_features == a.len()`. The output
/// (if supplied) must be the `n_features × n_features` Gram. The normalisation
/// divisor `n_samples - ddof` must be `> 0` (WR-01) — otherwise the `1/(n-ddof)`
/// scale divides by zero (or negative) and silently produces `inf`/`NaN`.
fn validate_geometry(
    a_len: usize,
    (n_samples, n_features): (usize, usize),
    ddof: u32,
    out_len: Option<usize>,
) -> Result<(), PrimError> {
    if n_samples
        .checked_mul(n_features)
        .map(|v| v != a_len)
        .unwrap_or(true)
    {
        return Err(PrimError::ShapeMismatch {
            operand: "a",
            rows: n_samples,
            cols: n_features,
            len: a_len,
        });
    }
    // CR-03: reject empty geometry at the boundary so a 0×cols / rows×0 matrix
    // never reaches the reduction/GEMM driver (where an empty-axis reduction's
    // identity is ambiguous). A valid covariance needs at least one sample and
    // one feature.
    if n_samples == 0 || n_features == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "a",
            rows: n_samples,
            cols: n_features,
            len: a_len,
        });
    }
    // WR-01: the 1/(n_samples - ddof) divisor must be strictly positive.
    if (n_samples as i64) - (ddof as i64) <= 0 {
        return Err(PrimError::DimMismatch {
            dim: "n_samples-ddof",
            lhs: n_samples,
            rhs: ddof as usize,
        });
    }
    if let Some(o) = out_len {
        let expect = n_features * n_features;
        if o != expect {
            return Err(PrimError::ShapeMismatch {
                operand: "out",
                rows: n_features,
                cols: n_features,
                len: o,
            });
        }
    }
    Ok(())
}

/// Standard ceiling-division 1D launch config for the in-place scale pass
/// (matches the `elementwise` per-element launch idiom used by distance).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256u32;
    let cubes = ((n as u32) + block - 1) / block;
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}

/// Host reciprocal `1/denom` as an `F` (f32 / f64). `denom` is the
/// `n_samples − ddof` normalisation divisor (`> 0` for any valid covariance);
/// the scalar `factor` is passed by value to the `scale` kernel.
fn recip<F: Pod>(denom: i64) -> F {
    let v = 1.0_f64 / (denom as f64);
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("covariance is f32/f64 only"),
    }
}
