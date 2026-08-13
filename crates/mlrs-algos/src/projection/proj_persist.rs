//! `proj_persist` (PROJ-PERSIST, prototype) — the `mlrs-proj` half of the mlrs
//! model file format: the container discriminator, the aliases the two
//! random-projection estimators write and read through, and the
//! `n_components='auto'` encoding they share.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## The on-disk shape
//!
//! A fitted random projection is exactly one matrix plus its constructor
//! scalars:
//!
//! | name | dtype | shape | held by |
//! |---|---|---|---|
//! | `components_` | `F` (`F32`/`F64`) | `[n_components_, n_features]` | both |
//! | `param:n_components` | `__metadata__` | — | both, see [`write_n_components`] |
//! | `param:seed` / `param:eps` | `__metadata__` | — | both |
//! | `param:density` | `__metadata__` | — | `SparseRandomProjection`, optional |
//! | `density_` | `__metadata__` | — | `SparseRandomProjection` |
//!
//! `n_components_` and `n_features` are BOTH recovered from `components_`'s
//! shape, so neither is stored again.
//!
//! ## Why the matrix is stored rather than the seed
//!
//! The tempting move is to store `seed`, `eps` and the geometry and REGENERATE
//! `components_` at load — the projection matrix is a deterministic function of
//! those, and the file would shrink from `n_components_ × n_features` floats to
//! a few dozen bytes. It is rejected, for two reasons that both outrank size.
//!
//! It would make the file's meaning depend on the mlrs BUILD that reads it, not
//! on its own contents. The matrix comes from `prims::rng`'s SplitMix64 stream;
//! any change to that generator, to the order it fills the matrix in, or to the
//! Achlioptas thresholding in `SparseRandomProjection` would silently produce a
//! DIFFERENT projection from the same file — a saved model that transforms its
//! own training data differently than it did before the upgrade, with nothing to
//! signal it. Storing the matrix makes the file self-contained: the bytes are
//! the model.
//!
//! And it would be a load-time cost that grows with the model. Regeneration is
//! `n_components_ × n_features` RNG draws on every load, against a read that is
//! one sequential `read_exact` and, on the matching-dtype arm, no copy at all.
//! The seed is still written — a reloaded model must be able to report the
//! hyperparameter it was built with — but it is a record, not an instruction.
//!
//! ## Why a sparse projection is stored dense
//!
//! `SparseRandomProjection`'s matrix is mostly zeros (density defaults to
//! `1/sqrt(n_features)`), so a CSR or coordinate encoding would be far smaller.
//! mlrs holds it DENSE in memory by design (D-12: the projection is a single
//! dense GEMM, and a sparse operand would need a different kernel), and the file
//! follows the memory layout for the reason the rest of this format does — a
//! dense `components_` loads with no copy and no decode straight into
//! `DeviceArray::from_host`, while a sparse one would have to be expanded on
//! every load. It is a defensible thing to revisit for a very wide projection;
//! it is not defensible to do while the compute path stays dense.
//!
//! Tests live in `crates/mlrs-algos/tests/proj_persist_test.rs` (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

use crate::projection::gaussian::NComponents;

// The container is shared with every other family; only the discriminator and
// the projection-shaped helpers below are local. Re-exported (not just
// imported) so `projection::proj_persist::{AlignedBytes, SaveModel, …}` is the
// single import path for a projection's `save`/`load`.
pub use crate::persist::{
    as_floats, shape_2d, AlignedBytes, Container, LoadModel, ModelFile, ModelWriter, PersistError,
    SaveModel, TensorRef, PARAM_PREFIX,
};

/// The random-projection container discriminator (`format = "mlrs-proj"`).
///
/// A zero-sized marker, never constructed — it exists only to carry the two tags
/// [`ModelFile::parse`] validates, so `ProjWriter`/`ProjFile` cannot be
/// instantiated with another family's tag by accident. It is separate from
/// `mlrs-decomp` even though both families store one `components_` matrix,
/// because the two mean different things: a decomposition's rows are fitted
/// directions and a projection's are random ones, and no cross-load between them
/// could be anything but a mistake.
pub struct ProjContainer;

impl Container for ProjContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-proj";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`ProjFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the projection matrix, row-major
/// `[n_components_, n_features]`. Named for sklearn's fitted attribute.
pub const COMPONENTS_NAME: &str = "components_";

/// The `__metadata__` key holding the `n_components` constructor argument.
pub const N_COMPONENTS_KEY: &str = "param:n_components";

/// The random-projection writer: [`ModelWriter`] pinned to the `mlrs-proj`
/// container.
pub type ProjWriter<'a> = ModelWriter<'a, ProjContainer>;

/// The random-projection reader: [`ModelFile`] pinned to the `mlrs-proj`
/// container.
pub type ProjFile<'a> = ModelFile<'a, ProjContainer>;

/// Stage the `components_` matrix, rejecting a degenerate geometry.
///
/// Written at `F`'s OWN width, so an `f32`-fitted projection produces a file
/// half the size of the `f64` one — which matters more here than anywhere else
/// in mlrs, since `components_` is not merely the largest part of the file but
/// essentially all of it.
pub fn write_components<'a, F: Pod>(
    w: &mut ProjWriter<'a>,
    components: &'a [F],
    n_components: usize,
    n_features: usize,
) -> Result<(), PersistError> {
    if n_components == 0 || n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{COMPONENTS_NAME}' would be [{n_components}, {n_features}]; a fitted \
                 projection has at least one component and one feature"
            ),
        });
    }
    w.tensor(
        COMPONENTS_NAME,
        TensorRef::floats(components, vec![n_components, n_features])?,
    );
    Ok(())
}

/// Read the `components_` matrix back with its `(n_components_, n_features)`.
///
/// The shape IS the schema — both extents are read off it rather than stored
/// separately (decision 2 in [`crate::persist`]'s docs), so there is no second
/// copy for a hand-edited header to disagree with. A zero extent is rejected
/// because a projection with no components or no features cannot transform
/// anything, and an empty upload is a landmine on the device backends.
pub fn read_components<'a, F: Pod>(
    file: &ProjFile<'a>,
) -> Result<(Cow<'a, [F]>, usize, usize), PersistError> {
    let view = file.tensor(COMPONENTS_NAME)?;
    let (n_components, n_features) = shape_2d(&view, COMPONENTS_NAME)?;
    if n_components == 0 || n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{COMPONENTS_NAME}' declares shape [{n_components}, {n_features}]; \
                 a fitted projection has at least one component and one feature"
            ),
        });
    }
    Ok((
        as_floats::<F>(&view, COMPONENTS_NAME)?,
        n_components,
        n_features,
    ))
}

/// Stage the `n_components` constructor argument as its sklearn STRING —
/// `"auto"` or the decimal.
///
/// [`NComponents`] is a two-variant enum whose `Auto` arm has no numeric value,
/// so it needs an encoding that admits both. The alternatives were an optional
/// integer with absence meaning `auto` — which makes a dropped key and a
/// deliberate `auto` indistinguishable, exactly the confusion
/// [`ModelWriter::scalar_opt_usize`] is careful to avoid — or a separate boolean
/// flag, which is two keys that can contradict each other. One string reads the
/// way the sklearn constructor argument does (`n_components='auto'` vs
/// `n_components=64`) and cannot be internally inconsistent.
///
/// Note this stores the REQUEST, not the outcome. Under `auto` the fitted
/// `n_components_` came out of `johnson_lindenstrauss_min_dim(n_samples, eps)`
/// and is recovered from `components_`'s row extent; the two are different
/// facts, and a reloaded model reports both.
pub fn write_n_components(w: &mut ProjWriter<'_>, n_components: NComponents) {
    match n_components {
        NComponents::Auto => w.scalar_str(N_COMPONENTS_KEY, "auto"),
        NComponents::Fixed(k) => w.scalar_usize(N_COMPONENTS_KEY, k),
    }
}

/// Read back what [`write_n_components`] staged.
///
/// A value that is neither `"auto"` nor a non-negative decimal is a
/// [`PersistError::BadMetadata`] naming the key rather than a silent fallback to
/// `Auto`: the two arms size the embedding differently, so guessing would hand
/// back a model whose reported hyperparameter never produced its own
/// `components_`.
pub fn read_n_components(file: &ProjFile<'_>) -> Result<NComponents, PersistError> {
    let raw = file.scalar_str(N_COMPONENTS_KEY)?;
    if raw == "auto" {
        return Ok(NComponents::Auto);
    }
    raw.parse::<usize>()
        .map(NComponents::Fixed)
        .map_err(|_| PersistError::BadMetadata {
            key: N_COMPONENTS_KEY,
        })
}
