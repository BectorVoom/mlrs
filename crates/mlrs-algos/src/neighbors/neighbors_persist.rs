//! `neighbors_persist` (NEIGH-PERSIST, prototype) — the `mlrs-neighbors` half of
//! the mlrs model file format: the container discriminator, the aliases the
//! three neighbor estimators write and read through, and the training-set core
//! plus metric encoding they share.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## The on-disk shape
//!
//! | name | dtype | shape | held by |
//! |---|---|---|---|
//! | `_fit_X` | `F` (`F32`/`F64`) | `[n_samples, n_features]` | all three |
//! | `classes_` | `I64` | `[n_classes]` | `KNeighborsClassifier` |
//! | `_y` | `I64` | `[n_samples]` | `KNeighborsClassifier`, the ENCODED labels |
//! | `_y` | `F` | `[n_samples, n_outputs]` | `KNeighborsRegressor` |
//! | `param:n_neighbors` / `param:metric` / `param:device` | `__metadata__` | — | all three |
//! | `param:p` | `__metadata__`, Minkowski only | — | all three |
//! | `param:weights` | `__metadata__` | — | the two predictors |
//!
//! `n_samples` and `n_features` are recovered from `_fit_X`'s shape, `n_classes`
//! from `classes_`'s and `n_outputs` from `_y`'s — none is stored again.
//!
//! ## The training set IS the model
//!
//! A k-NN estimator has no parameters at all: `kneighbors` scans every training
//! row, so `_fit_X` is not a fitting artifact but the entire fitted state. These
//! files are therefore `n_samples × n_features` floats and a header, and the
//! only size levers are the ones the format already pulls — the stored dtype is
//! the model's own, and nothing derivable is written beside it. Storing a
//! space-partitioning index instead (a KD- or ball-tree) would shrink nothing
//! and would tie the file to one query strategy; mlrs's k-NN is a brute-force
//! device scan, and the file follows the compute path.
//!
//! ## `_y` carries the ENCODED labels, and `classes_` the decode
//!
//! `KNeighborsClassifier`'s core is integer-only: the Python shim
//! `np.unique`-encodes the training labels to a dense `0..K` before they reach
//! the kernel, because the gather indexes a per-class table by that dense value
//! (CR-02). The file stores BOTH — `_y` as the encoding the kernel consumes, and
//! `classes_` as the table that turns a prediction back into the label the
//! caller trained with.
//!
//! Storing only the original labels and re-encoding at load would work today and
//! is deliberately not done: the encoding is defined by sort order over the
//! distinct values, so it is only stable as long as that rule never changes, and
//! a file that had to be re-encoded could not be read without re-deriving it.
//! Storing only the encoding is the worse failure — a model that round-trips its
//! own coefficients perfectly and predicts `{0, 1, 2}` where training said
//! `{0, 2, 7}`.
//!
//! Tests live in `crates/mlrs-algos/tests/neighbors_persist_test.rs`
//! (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;
use mlrs_backend::device::Device;

use super::{Metric, Weights};

// The container is shared with every other family; only the discriminator and
// the neighbor-shaped helpers below are local. Re-exported (not just imported)
// so `neighbors::neighbors_persist::{AlignedBytes, SaveModel, …}` is the single
// import path for a neighbor estimator's `save`/`load`.
pub use crate::persist::{
    as_f64, as_floats, as_i64, expect_len, shape_1d, shape_2d, AlignedBytes, Container, LoadModel,
    ModelFile, ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The neighbors container discriminator (`format = "mlrs-neighbors"`).
pub struct NeighborsContainer;

impl Container for NeighborsContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-neighbors";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`NeighborsFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the training matrix, row-major `[n_samples, n_features]`.
///
/// Named `_fit_X` because that is sklearn's own attribute on
/// `KNeighborsMixin` — the bare tensor names in this format are sklearn's, so a
/// `safetensors.numpy.load_file(path)` in Python hands back a dict keyed the way
/// the sklearn estimator is, leading underscore and all.
pub const FIT_X_NAME: &str = "_fit_X";

/// The tensor holding the per-sample targets: `I64` encoded class ids for the
/// classifier, `F` regression targets for the regressor. sklearn calls it `_y`
/// in both cases.
pub const Y_NAME: &str = "_y";

/// The tensor holding the distinct sorted training labels, `[n_classes]`.
pub const CLASSES_NAME: &str = "classes_";

/// The `__metadata__` key holding the distance metric's sklearn name.
pub const METRIC_KEY: &str = "param:metric";
/// The `__metadata__` key holding the Minkowski exponent — written ONLY for
/// `metric='minkowski'`, which is sklearn's own split of the two arguments.
pub const P_KEY: &str = "param:p";
/// The `__metadata__` key holding the `device=` placement hyperparameter.
pub const DEVICE_KEY: &str = "param:device";

/// The neighbors writer: [`ModelWriter`] pinned to the `mlrs-neighbors`
/// container.
pub type NeighborsWriter<'a> = ModelWriter<'a, NeighborsContainer>;

/// The neighbors reader: [`ModelFile`] pinned to the `mlrs-neighbors`
/// container.
pub type NeighborsFile<'a> = ModelFile<'a, NeighborsContainer>;

/// Stage the training matrix, rejecting a degenerate geometry.
///
/// Written at `F`'s OWN width — the single biggest size lever available to this
/// family, since `_fit_X` is essentially the whole file.
pub fn write_fit_x<'a, F: Pod>(
    w: &mut NeighborsWriter<'a>,
    fit_x: &'a [F],
    n_samples: usize,
    n_features: usize,
) -> Result<(), PersistError> {
    if n_samples == 0 || n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{FIT_X_NAME}' would be [{n_samples}, {n_features}]; a fitted neighbor \
                 estimator has at least one training sample and one feature"
            ),
        });
    }
    w.tensor(
        FIT_X_NAME,
        TensorRef::floats(fit_x, vec![n_samples, n_features])?,
    );
    Ok(())
}

/// Read the training matrix back with its `(n_samples, n_features)`.
///
/// The returned [`Cow`] BORROWS the mapped file bytes when the dtype matches
/// `F`, so the largest tensor in the file reaches
/// [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray::from_host)
/// without a single copy.
pub fn read_fit_x<'a, F: Pod>(
    file: &NeighborsFile<'a>,
) -> Result<(Cow<'a, [F]>, usize, usize), PersistError> {
    let view = file.tensor(FIT_X_NAME)?;
    let (n_samples, n_features) = shape_2d(&view, FIT_X_NAME)?;
    if n_samples == 0 || n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{FIT_X_NAME}' declares shape [{n_samples}, {n_features}]; a \
                 fitted neighbor estimator has at least one training sample and one \
                 feature"
            ),
        });
    }
    Ok((as_floats::<F>(&view, FIT_X_NAME)?, n_samples, n_features))
}

/// Stage the metric as its sklearn `(name, p)` pair.
///
/// Two keys rather than one because that is sklearn's own split —
/// `metric='minkowski'` and `p=2` are separate constructor arguments — and
/// because [`Metric::Minkowski`] is the only variant carrying a payload. `p` is
/// written only for that variant, so key-presence expresses the `Option`
/// exactly and costs nothing for the other four.
pub fn write_metric(w: &mut NeighborsWriter<'_>, metric: Metric) {
    w.scalar_str(METRIC_KEY, metric.name());
    w.scalar_opt_f64(P_KEY, metric.p());
}

/// Read back what [`write_metric`] staged.
///
/// A `metric='minkowski'` with no `p` is REJECTED rather than defaulted to 2 —
/// [`Metric::from_name`] enforces that, and the reason is that a silent `p = 2`
/// is Euclidean, so a file that lost its exponent would load as a different
/// metric and every distance it computed would be wrong with nothing to signal
/// it. An unrecognised name is likewise a [`PersistError::BadMetadata`] naming
/// the key, not a fallback.
pub fn read_metric(file: &NeighborsFile<'_>) -> Result<Metric, PersistError> {
    let name = file.scalar_str(METRIC_KEY)?;
    let p = file.scalar_opt_f64(P_KEY)?;
    Metric::from_name(name, p).ok_or(PersistError::BadMetadata { key: METRIC_KEY })
}

/// Stage the `device=` placement hyperparameter (DEVICE-PARAM-01).
///
/// It is a genuine constructor argument, not a fitted attribute: it says which
/// arm a future `kneighbors` should run on, so dropping it would hand back a
/// model that silently reverts to the heuristic. What is NOT stored is which arm
/// actually ran at fit time — that is a property of a call, not of the model.
pub fn write_device(w: &mut NeighborsWriter<'_>, device: Device) {
    w.scalar_str(DEVICE_KEY, device.name());
}

/// Read back what [`write_device`] staged, rejecting an unrecognised name.
pub fn read_device(file: &NeighborsFile<'_>) -> Result<Device, PersistError> {
    Device::from_name(file.scalar_str(DEVICE_KEY)?)
        .ok_or(PersistError::BadMetadata { key: DEVICE_KEY })
}

/// Stage the `weights=` hyperparameter — the two predictors only.
pub fn write_weights(w: &mut NeighborsWriter<'_>, weights: Weights) {
    w.scalar_str("param:weights", weights.name());
}

/// Read back what [`write_weights`] staged, rejecting an unrecognised name.
///
/// Never defaulted to `uniform`: the two schemes produce different predictions
/// from the same neighbors, so a silent fallback would be a different model with
/// nothing to signal it.
pub fn read_weights(file: &NeighborsFile<'_>) -> Result<Weights, PersistError> {
    Weights::from_name(file.scalar_str("param:weights")?).ok_or(PersistError::BadMetadata {
        key: "param:weights",
    })
}

/// Validate `n_neighbors` against the training set it will be queried over.
///
/// `k > n_samples` is not merely odd, it is unanswerable — the scan cannot
/// return more neighbors than it holds — and the file is untrusted input
/// (T-04-01-01), so a hand-edited `param:n_neighbors` has to fail here rather
/// than produce an out-of-range gather on the first query.
pub fn expect_k_fits(n_neighbors: usize, n_samples: usize) -> Result<(), PersistError> {
    if n_neighbors == 0 || n_neighbors > n_samples {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'param:n_neighbors' is {n_neighbors}, but '{FIT_X_NAME}' holds \
                 {n_samples} training samples"
            ),
        });
    }
    Ok(())
}
