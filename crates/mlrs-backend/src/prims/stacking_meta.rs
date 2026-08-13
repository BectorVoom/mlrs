//! Meta-matrix assembly, device arm (STACK-META-01) — the CubeCL half of
//! `mlrs.StackingRegressor`'s `_concatenate_predictions`.
//!
//! `StackingRegressor` builds an `n × width` meta matrix by writing `k`
//! prediction blocks side by side, optionally followed by the design `X`
//! (`passthrough=True`). The column layout is decided by
//! `mlrs_algos::ensemble::stacking::meta_layout`; the host arm that performs the
//! copy is `…::concatenate_predictions`. This module adds the second arm:
//! upload the blocks, scatter them with `k` (or `k + 1`) launches of
//! [`mlrs_kernels::stacking::stack_meta_block`], read the matrix back.
//!
//! ## Three arms, one knob
//! [`meta_engine`] resolves `MLRS_STACK_META_ENGINE` into [`MetaEngine`],
//! mirroring `MLRS_HUBER_ENGINE` / `MLRS_RANSAC_ENGINE`:
//!
//! | value | arm |
//! |---|---|
//! | unset (default) | [`MetaEngine::Numpy`] — `np.hstack` in the shim |
//! | `numpy` | the same, forced |
//! | `host` | `concatenate_predictions`, reached through the Arrow capsule |
//! | `device` | this module |
//!
//! ## Why the default is `numpy`, stated as a measurement
//! This operation contains no arithmetic. It is one `n × width` strided copy,
//! and `np.hstack` already performs exactly that copy at C speed on data that is
//! ALREADY in host memory — the blocks arrive as numpy arrays returned by the
//! composed estimators' `predict`, whatever device those estimators fitted on.
//! The two Rust arms therefore start in debt: the host arm by one Arrow capsule
//! round-trip each way, the device arm by that plus an upload and a download of
//! the whole matrix over the bus.
//!
//! `scripts/bench_stacking_meta.py` measured it rather than assuming it, and the
//! measurement agrees: on rocm (gfx1151, f64, min of 5 fresh processes) the host
//! arm runs at **0.08–0.74x** of `np.hstack` and this device arm at
//! **0.01–0.33x**, with cpu and wgpu landing in the same band. The host arm's
//! floor is a FIXED cost — the capsule crossing and the egress copy, which do
//! not shrink with `n` — so its ratio improves toward ~0.7x as the copy grows
//! but never crosses 1. The device arm is bus-bound by construction: it uploads
//! `n × width` and downloads `n × width` with no arithmetic in between. The full
//! ladder is in `docs/stacking.md`.
//!
//! The arm is kept because it is the piece a device-RESIDENT stacking path would
//! need — a stack whose members are all mlrs estimators could hand over their
//! prediction handles instead of numpy arrays, which turns the upload above into
//! a no-op — and because "numpy is faster here" is now a number that can be
//! re-checked on new hardware with one command.
//!
//! Tests live in `crates/mlrs-backend/tests/stacking_meta_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::stacking::stack_meta_block;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Which arm assembles the meta matrix. Resolved by [`meta_engine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaEngine {
    /// `np.hstack` in the Python shim — the default (see the module docs).
    Numpy,
    /// `mlrs_algos::ensemble::stacking::concatenate_predictions` on the host.
    Host,
    /// [`concat_meta_device`] — upload, scatter, read back.
    Device,
}

impl MetaEngine {
    /// The knob spelling, so a report of the arm that ran round-trips.
    pub fn as_str(self) -> &'static str {
        match self {
            MetaEngine::Numpy => "numpy",
            MetaEngine::Host => "host",
            MetaEngine::Device => "device",
        }
    }
}

/// The A/B knob naming the meta-assembly arm.
pub const ENGINE_KNOB: &str = "MLRS_STACK_META_ENGINE";

/// Resolve [`ENGINE_KNOB`] into the arm to use.
///
/// An unrecognized value falls back to [`MetaEngine::Numpy`] rather than
/// raising: this knob is a benchmarking affordance, and a typo in a sweep
/// script must not turn into a user-visible exception from `fit`. The resolved
/// name is reported to the caller (`_mlrs.stacking_meta_engine()`), so a typo is
/// still visible as "the arm I asked for is not the arm that ran".
pub fn meta_engine() -> MetaEngine {
    match crate::abflag::var(ENGINE_KNOB).as_deref() {
        Some("host") => MetaEngine::Host,
        Some("device") => MetaEngine::Device,
        _ => MetaEngine::Numpy,
    }
}

/// Assemble the `n_rows × width` meta matrix on the device and return it.
///
/// `blocks[i]` is `(data, cols)` in kept-estimator order, row-major and
/// `n_rows`-tall; `offsets[i]` is its first column in the output. `x` is
/// `(data, n_features)` and is required exactly when `n_meta < width` (the
/// `passthrough` case), where it occupies the columns from `n_meta` on.
///
/// Every shape is validated BEFORE the first launch — a kernel cannot return an
/// error, and an out-of-range write would silently corrupt a neighbouring
/// block's columns rather than fail.
pub fn concat_meta_device<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    blocks: &[(&[F], usize)],
    offsets: &[usize],
    n_rows: usize,
    n_meta: usize,
    width: usize,
    x: Option<(&[F], usize)>,
) -> Result<Vec<F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate(blocks, offsets, n_rows, n_meta, width, x)?;

    let out_len = n_rows * width;
    let elem = size_of::<F>();
    let out_handle = pool.acquire(out_len * elem);

    // Every block writes a disjoint column range, so the launches need no
    // barrier between them; they queue on the one stream in submission order.
    // `uploads` holds each block's device buffer alive until the last launch
    // has been submitted — dropping a `DeviceArray` returns its handle, and a
    // recycled handle underneath an in-flight kernel is a use-after-free.
    let mut uploads: Vec<DeviceArray<ActiveRuntime, F>> = Vec::with_capacity(blocks.len() + 1);

    for (i, &(data, cols)) in blocks.iter().enumerate() {
        scatter_one(
            pool,
            data,
            cols,
            offsets[i],
            width,
            (&out_handle, out_len),
            &mut uploads,
        );
    }
    if let Some((data, cols)) = x {
        if width > n_meta {
            scatter_one(
                pool,
                data,
                cols,
                n_meta,
                width,
                (&out_handle, out_len),
                &mut uploads,
            );
        }
    }

    let out = DeviceArray::<ActiveRuntime, F>::from_raw(out_handle, out_len);
    let host = out.to_host_metered(pool);
    out.release_into(pool);
    for upload in uploads {
        upload.release_into(pool);
    }
    Ok(host)
}

/// Upload one block and launch the scatter that writes it into `out_handle`.
fn scatter_one<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    data: &[F],
    cols: usize,
    offset: usize,
    width: usize,
    (out_handle, out_len): (&cubecl::server::Handle, usize),
    uploads: &mut Vec<DeviceArray<ActiveRuntime, F>>,
) where
    F: Float + CubeElement + Pod,
{
    if data.is_empty() {
        return;
    }
    let block = DeviceArray::<ActiveRuntime, F>::from_host(pool, data);
    let client = pool.client().clone();
    let (count, dim) =
        super::launch_dims_1d_folded(data.len(), crate::capability::gather_launch_width());
    // SAFETY: `data.len()` is the uploaded element count and `out_handle` is
    // `n_rows * width` long (allocated by the caller). `validate` has already
    // established `offset + cols <= width` and `data.len() == n_rows * cols`,
    // so every index the kernel forms is in range; the kernel additionally
    // bounds-checks `tid < block.len()` for the over-provisioned units.
    let block_arg = unsafe { ArrayArg::from_raw_parts(block.handle().clone(), data.len()) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
    stack_meta_block::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        block_arg,
        out_arg,
        cols as u32,
        width as u32,
        offset as u32,
    );
    uploads.push(block);
}

/// Host-side shape validation — the whole contract the kernel assumes.
fn validate<F>(
    blocks: &[(&[F], usize)],
    offsets: &[usize],
    n_rows: usize,
    n_meta: usize,
    width: usize,
    x: Option<(&[F], usize)>,
) -> Result<(), PrimError> {
    if offsets.len() != blocks.len() {
        return Err(PrimError::ShapeMismatch {
            operand: "stack_meta_offsets",
            rows: blocks.len(),
            cols: 1,
            len: offsets.len(),
        });
    }
    if width == 0 || n_meta > width {
        return Err(PrimError::ShapeMismatch {
            operand: "stack_meta_width",
            rows: n_meta,
            cols: 1,
            len: width,
        });
    }
    for (i, &(data, cols)) in blocks.iter().enumerate() {
        if data.len() != n_rows * cols {
            return Err(PrimError::ShapeMismatch {
                operand: "stack_meta_block",
                rows: n_rows,
                cols,
                len: data.len(),
            });
        }
        if offsets[i] + cols > n_meta {
            return Err(PrimError::ShapeMismatch {
                operand: "stack_meta_offset",
                rows: offsets[i],
                cols,
                len: n_meta,
            });
        }
    }
    let passthrough = width > n_meta;
    match (passthrough, x) {
        (true, None) => Err(PrimError::ShapeMismatch {
            operand: "stack_meta_passthrough_x",
            rows: n_rows,
            cols: width - n_meta,
            len: 0,
        }),
        (true, Some((data, cols))) => {
            if cols != width - n_meta || data.len() != n_rows * cols {
                return Err(PrimError::ShapeMismatch {
                    operand: "stack_meta_x",
                    rows: n_rows,
                    cols: width - n_meta,
                    len: data.len(),
                });
            }
            Ok(())
        }
        // A non-passthrough layout ignores `x` rather than rejecting it: the
        // shim passes `X` through unconditionally and lets the LAYOUT decide,
        // which is the same rule `concatenate_predictions` follows.
        (false, _) => Ok(()),
    }
}
