//! `decomp_persist` (DECOMP-PERSIST, prototype) — the `mlrs-decomp` half of the
//! mlrs model file format: the container discriminator, the aliases the three
//! decomposition estimators write and read through, and the spectral core they
//! share.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast. This module holds only what is
//! specific to a decomposition, and re-exports the rest so estimator code and
//! the test suite reach everything through one path.
//!
//! ## The on-disk shape
//!
//! Every member of this family is one `components_` matrix plus three
//! per-component spectra, and `Pca`/`IncrementalPCA` add the column mean they
//! center by:
//!
//! | name | dtype | shape | held by |
//! |---|---|---|---|
//! | `components_` | `F` (`F32`/`F64`) | `[n_components, n_features]` | all three |
//! | `explained_variance_` | `F` | `[n_components]` | all three |
//! | `explained_variance_ratio_` | `F` | `[n_components]` | all three |
//! | `singular_values_` | `F` | `[n_components]` | all three |
//! | `mean_` | `F` | `[n_features]` | `Pca`, `IncrementalPCA` |
//! | `var_` | `F` | `[n_features]` | `IncrementalPCA` only |
//! | `param:n_components` | `__metadata__` | — | all three |
//!
//! and NOTHING else: `n_components` and `n_features` are BOTH recovered from
//! `components_`'s shape at load, so the `param:n_components` scalar is the
//! REQUESTED count and the shape is what the fit actually retained. Those two
//! are not always equal — a rank-deficient fit retains fewer — which is exactly
//! why the request is stored separately rather than inferred back off the
//! matrix.
//!
//! ### Why `components_` is one fused block
//!
//! The row-major `[n_components, n_features]` layout is EXACTLY the layout the
//! fitted [`DeviceArray`](mlrs_backend::device_array::DeviceArray) already
//! holds, so neither direction reshuffles anything: a save is one device
//! readback streamed straight into the file, and a load hands the file's own
//! bytes to `DeviceArray::from_host` with no intermediate `Vec` and no
//! per-element decode. Splitting it into one tensor per component would cost
//! `n_components` JSON header entries (~60 bytes each) and `n_components`
//! separate uploads, for no benefit — `transform` is a single GEMM against the
//! whole block and never reads a component on its own.
//!
//! ### Why the three spectra are stored rather than recomputed
//!
//! `explained_variance_` and `singular_values_` are related by
//! `var = s²/(n_samples − 1)`, so one is derivable from the other GIVEN
//! `n_samples` — a value this family does not otherwise store, because nothing
//! at transform time needs it. Storing the derived pair costs
//! `2 · n_components` floats on a file whose payload is
//! `n_components · n_features`; storing `n_samples` instead to recover them
//! would save those floats and add a scalar whose only purpose is a division,
//! plus a round-trip that is exact in neither direction at `f32`. The ratio is
//! not derivable from either without the total variance, which is likewise not
//! otherwise stored.
//!
//! Tests live in `crates/mlrs-algos/tests/decomp_persist_test.rs` (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

// The container is shared with every other family; only the discriminator and
// the spectral helpers below are local. Re-exported (not just imported) so
// `decomposition::decomp_persist::{AlignedBytes, SaveModel, …}` is the single
// import path for a decomposition's `save`/`load`.
pub use crate::persist::{
    as_floats, expect_len, shape_1d, shape_2d, AlignedBytes, Container, LoadModel, ModelFile,
    ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The decomposition container discriminator (`format = "mlrs-decomp"`).
///
/// A zero-sized marker, never constructed — it exists only to carry the two tags
/// [`ModelFile::parse`] validates, so `DecompWriter`/`DecompFile` cannot be
/// instantiated with another family's tag by accident.
pub struct DecompContainer;

impl Container for DecompContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`. Present so a `.safetensors` file produced
/// by some other project — or by another mlrs family — fails the check at
/// [`DecompFile::parse`].
pub const FORMAT_ID: &str = "mlrs-decomp";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`DecompFile::parse`] rejects anything else outright (a prototype
/// has no back-compat obligations, so this is a hard equality, not a range).
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the component matrix, row-major
/// `[n_components, n_features]`. Named for sklearn's fitted attribute, so
/// `safetensors.numpy.load_file(path)` in Python hands back a dict keyed the way
/// the sklearn estimator is.
pub const COMPONENTS_NAME: &str = "components_";
/// The tensor holding the per-component variance, `[n_components]`.
pub const EXPLAINED_VARIANCE_NAME: &str = "explained_variance_";
/// The tensor holding the per-component variance fraction, `[n_components]`.
pub const EXPLAINED_VARIANCE_RATIO_NAME: &str = "explained_variance_ratio_";
/// The tensor holding the singular values, `[n_components]`.
pub const SINGULAR_VALUES_NAME: &str = "singular_values_";
/// The tensor holding the column mean, `[n_features]` — `Pca` and
/// `IncrementalPCA` only, since `TruncatedSvd` does NOT center (that is the
/// whole difference between the two, and the reason it works on sparse input in
/// sklearn).
pub const MEAN_NAME: &str = "mean_";
/// The tensor holding the running per-feature variance, `[n_features]` —
/// `IncrementalPCA` only, which is the one member that keeps a running
/// `explained_variance_ratio_` denominator across `partial_fit` calls and so
/// must carry the variance forward rather than recompute it.
pub const VAR_NAME: &str = "var_";

/// The `__metadata__` key holding `IncrementalPCA`'s running sample count.
///
/// A FITTED attribute, so no [`PARAM_PREFIX`]: it is what `partial_fit`
/// accumulated, not what the constructor was given. It lives here rather than in
/// `incremental_pca` because it is the one piece of this family's on-disk
/// vocabulary that a reader has to know about to interpret the running
/// statistics at all.
pub const N_SAMPLES_SEEN_KEY: &str = "n_samples_seen_";

/// The decomposition writer: [`ModelWriter`] pinned to the `mlrs-decomp`
/// container, so an estimator's `save` names only itself (`DecompWriter::new`)
/// and never a format id.
pub type DecompWriter<'a> = ModelWriter<'a, DecompContainer>;

/// The decomposition reader: [`ModelFile`] pinned to the `mlrs-decomp`
/// container. Loading an `mlrs-prep` file through it fails with
/// [`PersistError::NotAnMlrsModel`] before any tensor is touched.
pub type DecompFile<'a> = ModelFile<'a, DecompContainer>;

// ---------------------------------------------------------------------------
// The shared spectral core
// ---------------------------------------------------------------------------

/// The fitted state every decomposition in this family holds identically: the
/// component matrix and its three per-component spectra.
///
/// Staged BORROWING for the write side, so the estimators' host readbacks flow
/// into the serializer without a second copy. Named fields rather than a
/// positional function because all three spectra are `&[F]` of the same length —
/// exactly the signature a transposed argument slips through unnoticed, and one
/// no geometry check could catch.
///
/// Each estimator keeps its OWN `save`/`load` and stages its extra state either
/// side of this (`Pca`'s `mean_`, `IncrementalPCA`'s `var_`/`n_samples_seen_`);
/// only this common middle is shared, the same split
/// [`linear_persist`](crate::linear::linear_persist) makes for the dense linear
/// core.
pub struct SpectralCoreRef<'a, F> {
    /// The component matrix, row-major `n_components × n_features`.
    pub components: &'a [F],
    /// The per-component variance, length `n_components`.
    pub explained_variance: &'a [F],
    /// The per-component variance fraction, length `n_components`.
    pub explained_variance_ratio: &'a [F],
    /// The singular values, length `n_components`.
    pub singular_values: &'a [F],
    /// The RETAINED component count (the matrix's row extent).
    pub n_components: usize,
    /// The fitted feature count (the matrix's column extent).
    pub n_features: usize,
}

impl<'a, F: Pod> SpectralCoreRef<'a, F> {
    /// Stage the four tensors into `w`.
    ///
    /// Every tensor is written at `F`'s OWN width, so an `f32`-fitted model
    /// produces a file half the size of the `f64` one for the same geometry —
    /// and the dtype tag keeps the file self-describing, so that is a storage
    /// decision, not a commitment about how it is loaded back.
    ///
    /// The three spectra are length-checked against `n_components` HERE rather
    /// than left to `TensorRef::new`'s shape check, so the error names the two
    /// fields that disagree instead of just a shape and a length.
    pub fn write_into(self, w: &mut DecompWriter<'a>) -> Result<(), PersistError> {
        if self.n_components == 0 || self.n_features == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "'{COMPONENTS_NAME}' would be [{}, {}]; a fitted decomposition \
                     retains at least one component over at least one feature",
                    self.n_components, self.n_features
                ),
            });
        }
        for (name, values) in [
            (EXPLAINED_VARIANCE_NAME, self.explained_variance),
            (EXPLAINED_VARIANCE_RATIO_NAME, self.explained_variance_ratio),
            (SINGULAR_VALUES_NAME, self.singular_values),
        ] {
            expect_len(name, values.len(), self.n_components, "entries")?;
        }

        w.tensor(
            COMPONENTS_NAME,
            TensorRef::floats(self.components, vec![self.n_components, self.n_features])?,
        );
        w.tensor(
            EXPLAINED_VARIANCE_NAME,
            TensorRef::floats(self.explained_variance, vec![self.n_components])?,
        );
        w.tensor(
            EXPLAINED_VARIANCE_RATIO_NAME,
            TensorRef::floats(self.explained_variance_ratio, vec![self.n_components])?,
        );
        w.tensor(
            SINGULAR_VALUES_NAME,
            TensorRef::floats(self.singular_values, vec![self.n_components])?,
        );
        Ok(())
    }
}

/// The owned counterpart of [`SpectralCoreRef`], as recovered from a file.
///
/// Every array is a [`Cow`] rather than a `Vec` on purpose: when the file's
/// dtype matches `F` they BORROW the mapped file bytes, so they reach
/// [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray::from_host)
/// with no intervening allocation. Owning them here would silently undo the
/// whole zero-copy read path.
///
/// The `F: Clone` bound is what `Cow<'a, [F]>` needs (`[F]: ToOwned`). It is
/// implied by the `F: Pod` every caller already has, but a struct carries no
/// bound from its constructor, so it is spelled here.
pub struct SpectralCore<'a, F: Clone> {
    /// The component matrix, row-major `n_components × n_features`.
    pub components: Cow<'a, [F]>,
    /// The per-component variance, length `n_components`.
    pub explained_variance: Cow<'a, [F]>,
    /// The per-component variance fraction, length `n_components`.
    pub explained_variance_ratio: Cow<'a, [F]>,
    /// The singular values, length `n_components`.
    pub singular_values: Cow<'a, [F]>,
    /// Recovered from `components_`'s row extent, not stored separately.
    pub n_components: usize,
    /// Recovered from `components_`'s column extent, not stored separately.
    pub n_features: usize,
}

/// Read back everything [`SpectralCoreRef::write_into`] staged, validating the
/// geometry against `components_`'s own shape.
///
/// The file is UNTRUSTED input (T-04-01-01), so `components_` defines the
/// geometry and every spectrum is measured against it before a single value is
/// handed back — a header edited to claim more components than
/// `singular_values_` holds fails here, rather than reading out of bounds the
/// first time the model transforms. A zero extent is rejected for the same
/// reason: a `[0, d]` or `[k, 0]` decomposition cannot transform anything, and
/// an empty upload is a landmine on the device backends.
pub fn read_spectral_core<'a, F: Pod>(
    file: &DecompFile<'a>,
) -> Result<SpectralCore<'a, F>, PersistError> {
    let components_v = file.tensor(COMPONENTS_NAME)?;
    let (n_components, n_features) = shape_2d(&components_v, COMPONENTS_NAME)?;
    if n_components == 0 || n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{COMPONENTS_NAME}' declares shape [{n_components}, {n_features}]; \
                 a fitted decomposition retains at least one component over at least \
                 one feature"
            ),
        });
    }

    let mut spectra = Vec::with_capacity(3);
    for name in [
        EXPLAINED_VARIANCE_NAME,
        EXPLAINED_VARIANCE_RATIO_NAME,
        SINGULAR_VALUES_NAME,
    ] {
        let view = file.tensor(name)?;
        expect_len(name, shape_1d(&view, name)?, n_components, "entries")?;
        // `as_floats` borrows the file buffer outright when the dtype matches
        // `F`, so this is a view into the bytes `read_exact` landed — no
        // intermediate `Vec`, no per-element decode.
        spectra.push(as_floats::<F>(&view, name)?);
    }
    // Popped in reverse so each `Cow` moves out rather than being cloned; the
    // order is the one the loop pushed in.
    let singular_values = spectra.pop().expect("three spectra were pushed");
    let explained_variance_ratio = spectra.pop().expect("three spectra were pushed");
    let explained_variance = spectra.pop().expect("three spectra were pushed");

    Ok(SpectralCore {
        components: as_floats::<F>(&components_v, COMPONENTS_NAME)?,
        explained_variance,
        explained_variance_ratio,
        singular_values,
        n_components,
        n_features,
    })
}

/// Read a `[n_features]` companion vector (`mean_`, `var_`), checking its length
/// against the geometry `components_` already established.
///
/// Separate from [`read_spectral_core`] because only two of the three estimators
/// have one, and because `TruncatedSvd`'s ABSENCE of `mean_` is meaningful — it
/// does not center, which is the whole difference between it and `Pca`. Folding
/// the mean into the core would force `TruncatedSvd` to write a zero vector it
/// never reads, and would let a `Pca` file whose `mean_` was dropped load as a
/// silently uncentered model.
pub fn read_feature_vec<'a, F: Pod>(
    file: &DecompFile<'a>,
    name: &'static str,
    n_features: usize,
) -> Result<Cow<'a, [F]>, PersistError> {
    let view = file.tensor(name)?;
    expect_len(name, shape_1d(&view, name)?, n_features, "entries")?;
    as_floats::<F>(&view, name)
}
