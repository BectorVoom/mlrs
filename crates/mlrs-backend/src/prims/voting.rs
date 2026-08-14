//! Prediction voting, device arm (VOTE-01) — the CubeCL half of
//! `mlrs.VotingRegressor`'s `predict` and `transform`.
//!
//! A fitted `VotingRegressor` holds `k` members. Each answers `predict(X)` with
//! an `n`-long column; the estimator then either stacks those columns into an
//! `n × k` matrix (`transform`) or reduces them to a weighted mean
//! (`predict`). This module runs both on the device:
//! [`vote_transform_device`] is `k` launches of
//! [`mlrs_kernels::voting::vote_write_col`], and [`vote_average_device`] is `k`
//! accumulate launches followed by one divide.
//!
//! ## Three arms, one knob
//! [`vote_engine`] resolves `MLRS_VOTING_ENGINE` into [`VoteEngine`], mirroring
//! `MLRS_STACK_META_ENGINE` / `MLRS_HUBER_ENGINE` / `MLRS_RANSAC_ENGINE`:
//!
//! | value | arm |
//! |---|---|
//! | unset (default) | [`VoteEngine::Numpy`] — `np.average` in the shim |
//! | `numpy` | the same, forced |
//! | `host` | `mlrs_algos::ensemble::voting::{stack_columns, weighted_average}` |
//! | `device` | this module |
//!
//! ## How this differs from the stacking meta-matrix arm
//! `stacking_meta` moves `n · width` in and `n · width` out and computes
//! nothing, which is why `np.hstack` beats it on every backend measured. The
//! `predict` half here is a REDUCTION: `n · k` up, `n` back, with `k` multiplies
//! and `k − 1` adds per row in between. The download therefore shrinks by a
//! factor of `k`, and there is real arithmetic to amortise the crossing against.
//! Whether that is enough to beat `np.average` — which is also just a BLAS-free
//! strided pass over host-resident data — is a question for
//! `scripts/bench_voting.py`, not for a comment; the shipping default is
//! whatever that ladder says, and today it says `numpy`.
//!
//! The `transform` half has no such advantage — it is `stacking_meta`'s
//! situation exactly, `n · k` each way with no arithmetic — and is here so the
//! two methods can run on the same arm rather than silently splitting.
//!
//! ## Agreement with the host arm: exact for `transform`, 1 ULP for `predict`
//! [`vote_transform_device`] performs no arithmetic and is byte-identical to
//! the host transpose. [`vote_average_device`] is NOT bit-identical on a real
//! GPU: `acc + pred·w` contracts into a fused multiply-add, which rounds once
//! where numpy rounds twice (measured on rocm gfx1151, f32: every value within
//! one ULP, most of them equal). The cpu backend does not contract, so the same
//! source is bit-exact there. See `mlrs_kernels::voting` for why this is not
//! suppressible and why the more-accurate answer is the acceptable one.
//!
//! Tests live in `crates/mlrs-backend/tests/voting_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::voting::{vote_add_weighted, vote_divide, vote_init_weighted, vote_write_col};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Which arm performs a voting aggregation. Resolved by [`vote_engine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteEngine {
    /// `np.average` / `np.asarray(...).T` in the Python shim — the default.
    Numpy,
    /// `mlrs_algos::ensemble::voting`'s host loops, through the Arrow capsule.
    Host,
    /// [`vote_average_device`] / [`vote_transform_device`].
    Device,
}

impl VoteEngine {
    /// The knob spelling, so a report of the arm that ran round-trips.
    pub fn as_str(self) -> &'static str {
        match self {
            VoteEngine::Numpy => "numpy",
            VoteEngine::Host => "host",
            VoteEngine::Device => "device",
        }
    }
}

/// The A/B knob naming the voting-aggregation arm.
pub const ENGINE_KNOB: &str = "MLRS_VOTING_ENGINE";

/// Resolve [`ENGINE_KNOB`] into the arm to use.
///
/// An unrecognized value falls back to [`VoteEngine::Numpy`] rather than
/// raising, for the reason `stacking_meta::meta_engine` gives: this is a
/// benchmarking affordance, and a typo in a sweep script must not become a
/// user-visible exception out of `predict`. The resolved name is reported back
/// (`_mlrs.voting_engine()`), so a typo is still visible as "the arm I asked for
/// is not the arm that ran".
pub fn vote_engine() -> VoteEngine {
    match crate::abflag::var(ENGINE_KNOB).as_deref() {
        Some("host") => VoteEngine::Host,
        Some("device") => VoteEngine::Device,
        _ => VoteEngine::Numpy,
    }
}

/// The weighted mean of `k` prediction columns, computed on the device.
///
/// `preds[j]` is member `j`'s `n_rows`-long prediction; `weights[j]` is its
/// weight and `denom` is their sum, computed by the caller in the SAME order and
/// dtype the host arm uses so the two arms divide by identical bits.
///
/// Returns the `n_rows`-long weighted mean. The accumulator is the only device
/// allocation beyond the uploads, and it is written by the first launch rather
/// than zero-filled (see the kernel module for why that matters at `-0.0`).
pub fn vote_average_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    preds: &[&[F]],
    weights: &[F],
    denom: F,
    n_rows: usize,
) -> Result<Vec<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_columns(preds, n_rows)?;
    if weights.len() != preds.len() {
        return Err(PrimError::ShapeMismatch {
            operand: "vote_weights",
            rows: preds.len(),
            cols: 1,
            len: weights.len(),
        });
    }
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let elem = size_of::<F>();
    let acc_handle = pool.acquire(n_rows * elem);
    let client = pool.client().clone();
    let (count, dim) =
        super::launch_dims_1d_folded(n_rows, crate::capability::gather_launch_width());

    // Each upload stays alive until every launch that reads it has been
    // submitted; dropping a `DeviceArray` returns its handle to the pool, and a
    // recycled handle underneath an in-flight kernel is a use-after-free.
    let mut uploads: Vec<DeviceArray<ActiveRuntime, F>> = Vec::with_capacity(preds.len());

    for (j, &pred) in preds.iter().enumerate() {
        let col = DeviceArray::<ActiveRuntime, F>::from_host(pool, pred);
        // SAFETY: `pred.len() == n_rows` (checked by `validate_columns`) and the
        // accumulator handle is `n_rows` elements long (allocated just above),
        // so every index either kernel forms is in range; both additionally
        // bounds-check `tid < pred.len()`.
        let pred_arg = unsafe { ArrayArg::from_raw_parts(col.handle().clone(), n_rows) };
        let acc_arg = unsafe { ArrayArg::from_raw_parts(acc_handle.clone(), n_rows) };
        if j == 0 {
            vote_init_weighted::launch::<F, ActiveRuntime>(
                &client,
                count.clone(),
                dim,
                pred_arg,
                acc_arg,
                weights[j],
            );
        } else {
            // Read-after-write against the previous member's launch, ordered by
            // the single client stream (see the kernel module docs).
            vote_add_weighted::launch::<F, ActiveRuntime>(
                &client,
                count.clone(),
                dim,
                pred_arg,
                acc_arg,
                weights[j],
            );
        }
        uploads.push(col);
    }

    // SAFETY: same handle and length as every accumulate launch above.
    let acc_arg = unsafe { ArrayArg::from_raw_parts(acc_handle.clone(), n_rows) };
    vote_divide::launch::<F, ActiveRuntime>(&client, count, dim, acc_arg, denom);

    let acc = DeviceArray::<ActiveRuntime, F>::from_raw(acc_handle, n_rows);
    let host = acc.to_host_metered(pool);
    acc.release_into(pool);
    for upload in uploads {
        upload.release_into(pool);
    }
    Ok(host)
}

/// The `n_rows × k` transform matrix — member `j`'s column at column `j`.
///
/// The transpose sklearn writes as `np.asarray([est.predict(X) …]).T`, performed
/// as `k` independent scatters into one output handle. The launches touch
/// disjoint columns of every row, so they need no barrier between them.
pub fn vote_transform_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    preds: &[&[F]],
    n_rows: usize,
) -> Result<Vec<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_columns(preds, n_rows)?;
    let k = preds.len();
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let out_len = n_rows * k;
    let elem = size_of::<F>();
    let out_handle = pool.acquire(out_len * elem);
    let client = pool.client().clone();
    let (count, dim) =
        super::launch_dims_1d_folded(n_rows, crate::capability::gather_launch_width());

    let mut uploads: Vec<DeviceArray<ActiveRuntime, F>> = Vec::with_capacity(k);
    for (j, &pred) in preds.iter().enumerate() {
        let col = DeviceArray::<ActiveRuntime, F>::from_host(pool, pred);
        // SAFETY: `pred.len() == n_rows` and `out` is `n_rows * k` long, so the
        // largest index formed is `(n_rows - 1) * k + (k - 1) == out_len - 1`.
        let pred_arg = unsafe { ArrayArg::from_raw_parts(col.handle().clone(), n_rows) };
        let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
        vote_write_col::launch::<F, ActiveRuntime>(
            &client,
            count.clone(),
            dim,
            pred_arg,
            out_arg,
            k as u32,
            j as u32,
        );
        uploads.push(col);
    }

    let out = DeviceArray::<ActiveRuntime, F>::from_raw(out_handle, out_len);
    let host = out.to_host_metered(pool);
    out.release_into(pool);
    for upload in uploads {
        upload.release_into(pool);
    }
    Ok(host)
}

/// The whole shape contract both entry points hold callers to: at least one
/// member, every column exactly `n_rows` long.
///
/// Validated BEFORE the first launch — a kernel cannot return an error, and a
/// short column would read past its upload rather than fail.
fn validate_columns<F>(preds: &[&[F]], n_rows: usize) -> Result<(), PrimError> {
    if preds.is_empty() {
        return Err(PrimError::ShapeMismatch {
            operand: "vote_columns",
            rows: n_rows,
            cols: 0,
            len: 0,
        });
    }
    for pred in preds {
        if pred.len() != n_rows {
            return Err(PrimError::ShapeMismatch {
                operand: "vote_column",
                rows: n_rows,
                cols: 1,
                len: pred.len(),
            });
        }
    }
    Ok(())
}
