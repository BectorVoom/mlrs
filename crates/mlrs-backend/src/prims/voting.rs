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

use cubecl::server::Handle;
use mlrs_core::PrimError;
use mlrs_kernels::voting::{
    vote_add_weighted, vote_argmax_bounded, vote_argmax_rows, vote_bincount_add, vote_counts_zero,
    vote_divide, vote_hi_zero, vote_init_weighted, vote_write_block, vote_write_col,
};

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
    validate_weights(weights, preds.len())?;
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let acc_handle = accumulate_weighted(pool, preds, weights, denom, n_rows);
    let acc = DeviceArray::<ActiveRuntime, F>::from_raw(acc_handle, n_rows);
    let host = acc.to_host_metered(pool);
    acc.release_into(pool);
    Ok(host)
}

/// The `Σⱼ predⱼ·wⱼ / Σⱼ wⱼ` accumulation, left DEVICE-RESIDENT.
///
/// Factored out of [`vote_average_device`] so [`vote_soft_predict_device`] can
/// consume the accumulator with [`vote_argmax_rows`] instead of downloading it —
/// which is the entire reason soft voting's device arm is worth its crossing.
/// Both entry points therefore run byte-identical arithmetic; a second copy of
/// this loop would be a second place for the summation order to drift from
/// `np.average`'s.
///
/// Returns the accumulator handle, `len` elements long and already divided.
/// Every uploaded column is released before returning; the caller owns what
/// comes back.
///
/// Callers must have run [`validate_columns`] and [`validate_weights`] first —
/// a kernel cannot return an error, and a short column would read past its
/// upload rather than fail.
fn accumulate_weighted<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    cols: &[&[F]],
    weights: &[F],
    denom: F,
    len: usize,
) -> Handle
where
    F: Float + CubeElement + Pod,
{
    let elem = size_of::<F>();
    let acc_handle = pool.acquire(len * elem);
    let client = pool.client().clone();
    let (count, dim) = super::launch_dims_1d_folded(len, crate::capability::gather_launch_width());

    // Each upload stays alive until every launch that reads it has been
    // submitted; dropping a `DeviceArray` returns its handle to the pool, and a
    // recycled handle underneath an in-flight kernel is a use-after-free.
    let mut uploads: Vec<DeviceArray<ActiveRuntime, F>> = Vec::with_capacity(cols.len());

    for (j, &col_host) in cols.iter().enumerate() {
        let col = DeviceArray::<ActiveRuntime, F>::from_host(pool, col_host);
        // SAFETY: `col_host.len() == len` (checked by the caller) and the
        // accumulator handle is `len` elements long (allocated just above), so
        // every index either kernel forms is in range; both additionally
        // bounds-check `tid < pred.len()`.
        let pred_arg = unsafe { ArrayArg::from_raw_parts(col.handle().clone(), len) };
        let acc_arg = unsafe { ArrayArg::from_raw_parts(acc_handle.clone(), len) };
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
    let acc_arg = unsafe { ArrayArg::from_raw_parts(acc_handle.clone(), len) };
    vote_divide::launch::<F, ActiveRuntime>(&client, count, dim, acc_arg, denom);

    for upload in uploads {
        upload.release_into(pool);
    }
    acc_handle
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

// ------------------------------------------------------------------------- //
// VotingClassifier (VOTE-CLF-01)
// ------------------------------------------------------------------------- //

/// `voting='hard'` — the weighted majority label per row, computed on the device.
///
/// `labels[j]` is member `j`'s `n_rows`-long **encoded** prediction (`0 ..
/// n_bins`), `weights[j]` its weight. `n_bins` must be strictly greater than
/// every label; the caller derives it from the same scan that proved the labels
/// non-negative, because `np.bincount` rejects a negative element rather than
/// wrapping it.
///
/// Three phases, `k + 3` launches: clear the `n_rows × n_bins` tally and the
/// per-row ceiling, accumulate one member per launch
/// ([`vote_bincount_add`]), then reduce each row ([`vote_argmax_bounded`]).
/// The tally is the only sizeable allocation and it never leaves the device —
/// the download is `n_rows` labels, `n_bins` times smaller than the tally and
/// `k` times smaller than the uploads.
///
/// Returns the argmax indices, which are positions in the caller's class order.
pub fn vote_hard_predict_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    labels: &[&[u32]],
    weights: &[F],
    n_rows: usize,
    n_bins: u32,
) -> Result<Vec<u32>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_columns(labels, n_rows)?;
    validate_weights(weights, labels.len())?;
    if n_bins == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "vote_n_bins",
            rows: n_rows,
            cols: 0,
            len: 0,
        });
    }
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let elem = size_of::<F>();
    let n_bins_usize = n_bins as usize;
    let counts_len = n_rows * n_bins_usize;
    let counts_handle = pool.acquire(counts_len * elem);
    let hi_handle = pool.acquire(n_rows * size_of::<u32>());
    let out_handle = pool.acquire(n_rows * size_of::<u32>());
    let client = pool.client().clone();
    let width = crate::capability::gather_launch_width();
    let (row_count, row_dim) = super::launch_dims_1d_folded(n_rows, width);
    let (tally_count, tally_dim) = super::launch_dims_1d_folded(counts_len, width);

    // SAFETY for every `from_raw_parts` below: `counts_handle` is
    // `n_rows * n_bins` elements, `hi_handle` and `out_handle` are `n_rows`, and
    // each kernel bounds-checks its own unit index against the array it walks.
    // `vote_bincount_add`'s `r * n_bins + label` is in range because the caller
    // guarantees `label < n_bins` (see the doc comment).
    let counts_arg = unsafe { ArrayArg::from_raw_parts(counts_handle.clone(), counts_len) };
    vote_counts_zero::launch::<F, ActiveRuntime>(&client, tally_count, tally_dim, counts_arg);
    let hi_arg = unsafe { ArrayArg::from_raw_parts(hi_handle.clone(), n_rows) };
    vote_hi_zero::launch::<ActiveRuntime>(&client, row_count.clone(), row_dim, hi_arg);

    let mut uploads: Vec<DeviceArray<ActiveRuntime, u32>> = Vec::with_capacity(labels.len());
    for (j, &col_host) in labels.iter().enumerate() {
        let col = DeviceArray::<ActiveRuntime, u32>::from_host(pool, col_host);
        let labels_arg = unsafe { ArrayArg::from_raw_parts(col.handle().clone(), n_rows) };
        let counts_arg = unsafe { ArrayArg::from_raw_parts(counts_handle.clone(), counts_len) };
        let hi_arg = unsafe { ArrayArg::from_raw_parts(hi_handle.clone(), n_rows) };
        // Read-after-write against the previous member on BOTH `counts` and
        // `hi`, ordered by the single client stream — the same guarantee
        // `vote_add_weighted`'s chain relies on.
        vote_bincount_add::launch::<F, ActiveRuntime>(
            &client,
            row_count.clone(),
            row_dim,
            labels_arg,
            counts_arg,
            hi_arg,
            n_bins,
            weights[j],
        );
        uploads.push(col);
    }

    let counts_arg = unsafe { ArrayArg::from_raw_parts(counts_handle.clone(), counts_len) };
    let hi_arg = unsafe { ArrayArg::from_raw_parts(hi_handle.clone(), n_rows) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n_rows) };
    vote_argmax_bounded::launch::<F, ActiveRuntime>(
        &client, row_count, row_dim, counts_arg, hi_arg, out_arg, n_bins,
    );

    let out = DeviceArray::<ActiveRuntime, u32>::from_raw(out_handle, n_rows);
    let host = out.to_host_metered(pool);
    out.release_into(pool);
    DeviceArray::<ActiveRuntime, F>::from_raw(counts_handle, counts_len).release_into(pool);
    DeviceArray::<ActiveRuntime, u32>::from_raw(hi_handle, n_rows).release_into(pool);
    for upload in uploads {
        upload.release_into(pool);
    }
    Ok(host)
}

/// `voting='soft'` — the weighted probability average of `k` `n_rows × n_cols`
/// blocks, computed on the device.
///
/// This is `np.average(probas, axis=0, weights=w)`, and it is
/// [`vote_average_device`]'s arithmetic verbatim: the reduced axis is still the
/// member axis, each member still contributes one contiguous run, so the
/// accumulation runs over `n_rows · n_cols` elements instead of `n_rows`. No new
/// kernel is involved — see [`accumulate_weighted`].
pub fn vote_soft_proba_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    blocks: &[&[F]],
    weights: &[F],
    denom: F,
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let len = n_rows * n_cols;
    validate_columns(blocks, len)?;
    validate_weights(weights, blocks.len())?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let acc_handle = accumulate_weighted(pool, blocks, weights, denom, len);
    let acc = DeviceArray::<ActiveRuntime, F>::from_raw(acc_handle, len);
    let host = acc.to_host_metered(pool);
    acc.release_into(pool);
    Ok(host)
}

/// `voting='soft'` — the argmax of that average, WITHOUT downloading it.
///
/// `argmax(np.average(probas, axis=0, weights=w), axis=1)`, fused: the
/// accumulator stays on the device and [`vote_argmax_rows`] turns it into
/// `n_rows` labels there. The download is therefore `n_rows` `u32`s rather than
/// `n_rows · n_cols` floats — the one shape in this module where the device arm
/// has a structural advantage over `numpy` rather than merely a chance at one,
/// since numpy has to materialise the full average before it can reduce it.
///
/// The labels are bit-for-bit the same decision `predict_proba(...).argmax(1)`
/// makes on the same arm: the average is the same kernel chain and
/// [`vote_argmax_rows`] takes the FIRST maximum, as `np.argmax` does.
pub fn vote_soft_predict_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    blocks: &[&[F]],
    weights: &[F],
    denom: F,
    n_rows: usize,
    n_cols: usize,
) -> Result<Vec<u32>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let len = n_rows * n_cols;
    validate_columns(blocks, len)?;
    validate_weights(weights, blocks.len())?;
    if n_cols == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "vote_n_classes",
            rows: n_rows,
            cols: 0,
            len: 0,
        });
    }
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let acc_handle = accumulate_weighted(pool, blocks, weights, denom, len);
    let out_handle = pool.acquire(n_rows * size_of::<u32>());
    let client = pool.client().clone();
    let (count, dim) =
        super::launch_dims_1d_folded(n_rows, crate::capability::gather_launch_width());

    // SAFETY: `acc_handle` is `n_rows * n_cols` elements and `out_handle` is
    // `n_rows`; the kernel bounds-checks `tid < out.len()` and walks
    // `[tid * n_cols, tid * n_cols + n_cols)`, the last of which is `len - 1`.
    let acc_arg = unsafe { ArrayArg::from_raw_parts(acc_handle.clone(), len) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n_rows) };
    vote_argmax_rows::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        acc_arg,
        out_arg,
        n_cols as u32,
    );

    let out = DeviceArray::<ActiveRuntime, u32>::from_raw(out_handle, n_rows);
    let host = out.to_host_metered(pool);
    out.release_into(pool);
    DeviceArray::<ActiveRuntime, F>::from_raw(acc_handle, len).release_into(pool);
    Ok(host)
}

/// `voting='soft', flatten_transform=True` — `np.hstack(probas)`.
///
/// `k` blocks of `n_rows × width` laid side by side into one
/// `n_rows × (k · width)` matrix, as `k` independent scatters
/// ([`vote_write_block`]) into disjoint column ranges of the same output — so,
/// like [`vote_transform_device`], the launches need no barrier between them.
///
/// This is the copy-shaped half of the classifier's device arm and carries the
/// same warning `stacking_meta` does: `n · k · width` in, the same out, no
/// arithmetic. It is here so a caller who has selected the `device` arm gets it
/// for every method rather than for some of them, and `docs/voting.md` carries
/// the measurement that says whether it should ever be the default.
pub fn vote_hstack_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    blocks: &[&[F]],
    n_rows: usize,
    width: usize,
) -> Result<Vec<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_columns(blocks, n_rows * width)?;
    if width == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "vote_block_width",
            rows: n_rows,
            cols: 0,
            len: 0,
        });
    }
    let k = blocks.len();
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let block_len = n_rows * width;
    let out_stride = k * width;
    let out_len = n_rows * out_stride;
    let out_handle = pool.acquire(out_len * size_of::<F>());
    let client = pool.client().clone();
    let (count, dim) =
        super::launch_dims_1d_folded(block_len, crate::capability::gather_launch_width());

    let mut uploads: Vec<DeviceArray<ActiveRuntime, F>> = Vec::with_capacity(k);
    for (j, &block_host) in blocks.iter().enumerate() {
        let block = DeviceArray::<ActiveRuntime, F>::from_host(pool, block_host);
        // SAFETY: the largest index the kernel forms is
        // `(n_rows - 1) * out_stride + (k - 1) * width + (width - 1)`, which is
        // `out_len - 1`; the kernel bounds-checks `tid < block.len()`.
        let block_arg = unsafe { ArrayArg::from_raw_parts(block.handle().clone(), block_len) };
        let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
        vote_write_block::launch::<F, ActiveRuntime>(
            &client,
            count.clone(),
            dim,
            block_arg,
            out_arg,
            width as u32,
            out_stride as u32,
            (j * width) as u32,
        );
        uploads.push(block);
    }

    let out = DeviceArray::<ActiveRuntime, F>::from_raw(out_handle, out_len);
    let host = out.to_host_metered(pool);
    out.release_into(pool);
    for upload in uploads {
        upload.release_into(pool);
    }
    Ok(host)
}

/// One weight per member, checked before any launch.
///
/// Shared by every entry point that weights, so a mis-sized weight vector
/// reports the same `PrimError` whichever aggregation the caller asked for.
fn validate_weights<F>(weights: &[F], k: usize) -> Result<(), PrimError> {
    if weights.len() != k {
        return Err(PrimError::ShapeMismatch {
            operand: "vote_weights",
            rows: k,
            cols: 1,
            len: weights.len(),
        });
    }
    Ok(())
}

/// The whole shape contract every entry point holds callers to: at least one
/// member, every column exactly `n_rows` long.
///
/// Validated BEFORE the first launch — a kernel cannot return an error, and a
/// short column would read past its upload rather than fail.
///
/// Generic over the element type so the classifier's `u32` label columns are
/// held to the same contract as the regressor's float ones; `n_rows` is the
/// column's full LENGTH, which for a 2-D block is `n_rows · width`.
fn validate_columns<T>(preds: &[&[T]], n_rows: usize) -> Result<(), PrimError> {
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
