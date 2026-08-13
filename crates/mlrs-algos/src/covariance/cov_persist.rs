//! `cov_persist` (COV-PERSIST, prototype) — the `mlrs-cov` half of the mlrs
//! model file format: the container discriminator, the aliases the two
//! covariance estimators write and read through, and the location/scatter core
//! they share.
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
//! | `covariance_` | `F` (`F32`/`F64`) | `[n_features, n_features]` | both |
//! | `location_` | `F` | `[n_features]` | both |
//! | `precision_` | `F` | `[n_features, n_features]` | `EmpiricalCovariance`, optional |
//! | `param:assume_centered` | `__metadata__` | — | both |
//! | `param:store_precision` | `__metadata__` | — | `EmpiricalCovariance` |
//! | `shrinkage_` | `__metadata__` | — | `LedoitWolf` |
//!
//! `n_features` is recovered from `covariance_`'s shape, which is also where the
//! SQUARENESS check lives — a covariance that is not `d × d` is malformed in a
//! way no other tensor in mlrs can be, so [`read_scatter_core`] rejects it by
//! name rather than letting a downstream solve fail on it.
//!
//! ## Why the full matrix is stored rather than a triangle
//!
//! `covariance_` and `precision_` are SYMMETRIC, so a packed upper triangle
//! would be almost exactly half the size — the single largest size win available
//! anywhere in this format, since these files are `d²` floats and little else.
//! It is rejected on purpose, for the reason the rest of the format makes the
//! same call: the estimators hold the full `d × d` matrix device-resident, so a
//! packed file would have to be EXPANDED on every load — an `O(d²)` scatter into
//! a fresh allocation — where the full one is borrowed straight out of the file
//! buffer and handed to `DeviceArray::from_host` untouched. And
//! `safetensors.numpy.load_file(path)["covariance_"]` would stop being the
//! matrix the sklearn attribute is, becoming a packed vector no Python reader
//! could interpret without mlrs's private layout.
//!
//! That trade is worth revisiting for a genuinely large `d`, where halving the
//! file may outweigh an expansion pass. It is not worth making silently.
//!
//! ## Why `precision_` is optional and not always recomputed
//!
//! sklearn's `store_precision=False` means the inverse is computed ON DEMAND
//! rather than kept, and mlrs mirrors that: the file writes `precision_` only
//! when the model holds it, so the flag round-trips as key-presence AND as a
//! real size difference. Recomputing it at load instead would be an `O(d³)`
//! eigen-decomposition on a path that is otherwise a single sequential read, and
//! would silently convert a `store_precision=False` model into a
//! `store_precision=True` one.
//!
//! Tests live in `crates/mlrs-algos/tests/cov_persist_test.rs` (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

// The container is shared with every other family; only the discriminator and
// the covariance-shaped helpers below are local. Re-exported (not just
// imported) so `covariance::cov_persist::{AlignedBytes, SaveModel, …}` is the
// single import path for a covariance estimator's `save`/`load`.
pub use crate::persist::{
    as_floats, expect_len, shape_1d, shape_2d, AlignedBytes, Container, LoadModel, ModelFile,
    ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The covariance container discriminator (`format = "mlrs-cov"`).
pub struct CovContainer;

impl Container for CovContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-cov";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`CovFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the scatter matrix, row-major `[n_features, n_features]`.
pub const COVARIANCE_NAME: &str = "covariance_";
/// The tensor holding the estimated location (mean), `[n_features]`.
pub const LOCATION_NAME: &str = "location_";
/// The tensor holding the inverse covariance, `[n_features, n_features]` —
/// present only when the model was built with `store_precision = true`.
pub const PRECISION_NAME: &str = "precision_";

/// The covariance writer: [`ModelWriter`] pinned to the `mlrs-cov` container.
pub type CovWriter<'a> = ModelWriter<'a, CovContainer>;

/// The covariance reader: [`ModelFile`] pinned to the `mlrs-cov` container.
pub type CovFile<'a> = ModelFile<'a, CovContainer>;

/// The fitted state both covariance estimators hold identically: one square
/// scatter matrix and the location it was measured about.
///
/// Staged BORROWING for the write side, so the estimators' host readbacks flow
/// into the serializer without a second copy.
pub struct ScatterCoreRef<'a, F> {
    /// The scatter matrix, row-major `n_features × n_features`.
    pub covariance: &'a [F],
    /// The estimated location, length `n_features`.
    pub location: &'a [F],
    /// The feature count — the matrix's (equal) row and column extent.
    pub n_features: usize,
}

impl<'a, F: Pod> ScatterCoreRef<'a, F> {
    /// Stage both tensors into `w`, at `F`'s OWN width.
    ///
    /// The squareness of `covariance` is implied by the `[d, d]` shape this
    /// declares, and `TensorRef::floats` checks that against the payload length —
    /// so a caller that passed a `d × k` buffer fails HERE rather than writing a
    /// file whose header claims a covariance it does not hold.
    pub fn write_into(self, w: &mut CovWriter<'a>) -> Result<(), PersistError> {
        if self.n_features == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "'{COVARIANCE_NAME}' would be [0, 0]; a fitted covariance has at \
                     least one feature"
                ),
            });
        }
        expect_len(
            LOCATION_NAME,
            self.location.len(),
            self.n_features,
            "entries",
        )?;

        w.tensor(
            COVARIANCE_NAME,
            TensorRef::floats(self.covariance, vec![self.n_features, self.n_features])?,
        );
        w.tensor(
            LOCATION_NAME,
            TensorRef::floats(self.location, vec![self.n_features])?,
        );
        Ok(())
    }
}

/// The owned counterpart of [`ScatterCoreRef`], as recovered from a file.
///
/// Both arrays are [`Cow`]s rather than `Vec`s: when the file's dtype matches
/// `F` they BORROW the mapped file bytes, so they reach
/// [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray::from_host)
/// with no intervening allocation.
pub struct ScatterCore<'a, F: Clone> {
    /// The scatter matrix, row-major `n_features × n_features`.
    pub covariance: Cow<'a, [F]>,
    /// The estimated location, length `n_features`.
    pub location: Cow<'a, [F]>,
    /// Recovered from `covariance_`'s shape, not stored separately.
    pub n_features: usize,
}

/// Read back everything [`ScatterCoreRef::write_into`] staged.
///
/// The file is UNTRUSTED input (T-04-01-01), so `covariance_` defines the
/// geometry and `location_` is measured against it before any value is handed
/// back. The SQUARENESS check is the one this family needs that no other does: a
/// `[d, k]` covariance is malformed on its face, and without the check a
/// downstream Mahalanobis distance or precision solve would index out of range
/// rather than report a bad file.
pub fn read_scatter_core<'a, F: Pod>(
    file: &CovFile<'a>,
) -> Result<ScatterCore<'a, F>, PersistError> {
    let cov_v = file.tensor(COVARIANCE_NAME)?;
    let (rows, cols) = shape_2d(&cov_v, COVARIANCE_NAME)?;
    if rows == 0 || rows != cols {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{COVARIANCE_NAME}' declares shape [{rows}, {cols}]; a fitted \
                 covariance is square and has at least one feature"
            ),
        });
    }

    let loc_v = file.tensor(LOCATION_NAME)?;
    expect_len(
        LOCATION_NAME,
        shape_1d(&loc_v, LOCATION_NAME)?,
        rows,
        "entries",
    )?;

    // `as_floats` borrows the file buffer outright when the dtype matches `F`,
    // so both of these are views into the bytes `read_exact` landed.
    Ok(ScatterCore {
        covariance: as_floats::<F>(&cov_v, COVARIANCE_NAME)?,
        location: as_floats::<F>(&loc_v, LOCATION_NAME)?,
        n_features: rows,
    })
}

/// Read the OPTIONAL `precision_` matrix, checking it against the geometry
/// `covariance_` already established.
///
/// `Ok(None)` when the tensor is absent, which is exactly what a
/// `store_precision = false` model wrote. A present-but-misshapen `precision_`
/// is an error rather than a `None`: a truncated matrix is a corrupt file, not a
/// model that chose not to store one.
pub fn read_precision<'a, F: Pod>(
    file: &CovFile<'a>,
    n_features: usize,
) -> Result<Option<Cow<'a, [F]>>, PersistError> {
    let Some(view) = file.tensor_opt(PRECISION_NAME) else {
        return Ok(None);
    };
    let (rows, cols) = shape_2d(&view, PRECISION_NAME)?;
    if rows != n_features || cols != n_features {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{PRECISION_NAME}' declares shape [{rows}, {cols}], but \
                 '{COVARIANCE_NAME}' implies [{n_features}, {n_features}]"
            ),
        });
    }
    Ok(Some(as_floats::<F>(&view, PRECISION_NAME)?))
}
