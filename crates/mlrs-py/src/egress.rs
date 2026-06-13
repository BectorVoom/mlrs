//! Device → host egress helpers (D-03).
//!
//! The Python surface returns **host buffers + shape** to the pure-Python shim,
//! which wraps them to the resolved `output_type` container (numpy / pyarrow) —
//! the numpy/pyarrow wrap is *shim-side*, never here (D-03). These helpers are
//! the thin Rust side: read a fitted [`DeviceArray`] back to a host `Vec` (via
//! the metered read path so the D-10 read-back counter stays honest) and pair it
//! with its `(rows, cols)` shape.
//!
//! Labels / neighbor indices are `i32` everywhere (the algos layer widens `u32`
//! → `i32` at the boundary); [`labels_to_py`] carries that contract so the shim
//! materializes numpy `int32`.

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;

/// A host-materialized float result + its `(rows, cols)` shape, handed to the
/// Python shim (which wraps it to the resolved `output_type`).
///
/// `rows * cols == values.len()` for a 2-D result; a 1-D result uses
/// `cols == 1` (or the shim flattens on `rows == 1`). The shape is carried
/// explicitly so the shim never re-derives geometry from a flat buffer.
pub type FloatResult<F> = (Vec<F>, (usize, usize));

/// A host-materialized `i32` label / index result + its `(rows, cols)` shape.
///
/// Used for `labels_` (clustering) and neighbor index arrays — `i32` at egress
/// (D-03), which the shim wraps to numpy `int32`.
pub type LabelResult = (Vec<i32>, (usize, usize));

/// Read a fitted float [`DeviceArray`] back to a host [`FloatResult`] (D-03).
///
/// Uses the *metered* read path
/// ([`DeviceArray::to_host_metered`]) so each terminal read-back bumps the pool's
/// `read_backs` counter (the D-10 memory gate observes real read-backs, not a
/// code-review claim). `shape` is the `(rows, cols)` the caller knows for this
/// fitted attribute (e.g. `(n_clusters, n_features)` for `cluster_centers_`).
pub fn vec_f_to_py<F: bytemuck::Pod>(
    array: &DeviceArray<ActiveRuntime, F>,
    shape: (usize, usize),
    pool: &mut BufferPool<ActiveRuntime>,
) -> FloatResult<F> {
    (array.to_host_metered(pool), shape)
}

/// Pair an already-host `Vec<i32>` of labels / indices with its `(rows, cols)`
/// shape for the shim (D-03).
///
/// Labels frequently originate host-side (the algos layer returns `Vec<i32>`
/// from its label accessor); this helper just attaches the shape so the egress
/// contract is uniform with [`vec_f_to_py`].
pub fn labels_to_py(values: Vec<i32>, shape: (usize, usize)) -> LabelResult {
    (values, shape)
}

/// Read a fitted `i32` [`DeviceArray`] back to a host [`LabelResult`] (D-03).
///
/// The device-resident variant of [`labels_to_py`], for label / index arrays
/// that live on the device until a terminal read. Uses the metered read path
/// (D-10), mirroring [`vec_f_to_py`].
pub fn vec_i32_to_py(
    array: &DeviceArray<ActiveRuntime, i32>,
    shape: (usize, usize),
    pool: &mut BufferPool<ActiveRuntime>,
) -> LabelResult {
    (array.to_host_metered(pool), shape)
}
