//! `manifold_persist` (MANIFOLD-PERSIST, prototype) — the `mlrs-manifold` half
//! of the mlrs model file format: the container discriminator and the aliases
//! `Tsne` and `Umap` write and read through.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## What a manifold model is, and what it is not
//!
//! Both estimators produce an `embedding_` of the rows they were FITTED on, and
//! for `Tsne` that is the whole of it: t-SNE has no out-of-sample extension —
//! sklearn's `TSNE` exposes `fit_transform` and no `transform` — so its file is
//! the embedding plus the diagnostics that describe how the descent went.
//! `Umap` does generalize, and keeps `x_train_` to do it, so its file carries
//! the training matrix too and is correspondingly larger.
//!
//! That asymmetry is why the two share a container but no core: they hold
//! different things for structural reasons, not incidental ones.
//!
//! | name | dtype | shape | held by |
//! |---|---|---|---|
//! | `embedding_` | `F` (`F32`/`F64`) | `[n_samples, n_components]` | both |
//! | `_raw_data` | `F` | `[n_samples, n_features]` | `Umap` |
//! | `param:init_array` | `F64` | `[n_samples, n_components]` | `Tsne`, explicit init only |
//! | `param:metric_v` / `param:metric_vi` | `F64` | `[n_features]` / `[n_features²]` | `Tsne`, optional |
//! | `kl_divergence_` / `n_iter_` / `device_` | `__metadata__` | — | `Tsne` |
//!
//! `n_samples`, `n_components` and `n_features` all come off tensor shapes.
//!
//! Tests live in `crates/mlrs-algos/tests/manifold_persist_test.rs`
//! (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

// The container is shared with every other family; only the discriminator and
// the embedding helpers below are local. Re-exported (not just imported) so
// `manifold::manifold_persist::{AlignedBytes, SaveModel, …}` is the single
// import path for a manifold estimator's `save`/`load`.
pub use crate::persist::{
    as_f64, as_floats, expect_len, shape_1d, shape_2d, AlignedBytes, Container, LoadModel,
    ModelFile, ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The manifold container discriminator (`format = "mlrs-manifold"`).
pub struct ManifoldContainer;

impl Container for ManifoldContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-manifold";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`ManifoldFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the fitted embedding, row-major
/// `[n_samples, n_components]` — sklearn's and umap-learn's `embedding_`.
pub const EMBEDDING_NAME: &str = "embedding_";

/// The tensor holding `Umap`'s retained training matrix, row-major
/// `[n_samples, n_features]`.
///
/// Named `_raw_data` because that is umap-learn's own attribute for it — the
/// bare tensor names in this format are the upstream library's, so a
/// `safetensors.numpy.load_file(path)` in Python hands back a dict keyed the way
/// the estimator is.
pub const RAW_DATA_NAME: &str = "_raw_data";

/// The manifold writer: [`ModelWriter`] pinned to the `mlrs-manifold`
/// container.
pub type ManifoldWriter<'a> = ModelWriter<'a, ManifoldContainer>;

/// The manifold reader: [`ModelFile`] pinned to the `mlrs-manifold` container.
pub type ManifoldFile<'a> = ModelFile<'a, ManifoldContainer>;

/// Stage the embedding, rejecting a degenerate geometry.
///
/// Written at `F`'s OWN width. For `Tsne` this is essentially the whole file, so
/// an `f32` fit halves it outright; for `Umap` it shares the file with the
/// retained training matrix.
pub fn write_embedding<'a, F: Pod>(
    w: &mut ManifoldWriter<'a>,
    embedding: &'a [F],
    n_samples: usize,
    n_components: usize,
) -> Result<(), PersistError> {
    if n_samples == 0 || n_components == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{EMBEDDING_NAME}' would be [{n_samples}, {n_components}]; a fitted \
                 embedding has at least one sample and one component"
            ),
        });
    }
    w.tensor(
        EMBEDDING_NAME,
        TensorRef::floats(embedding, vec![n_samples, n_components])?,
    );
    Ok(())
}

/// Read the embedding back with its `(n_samples, n_components)`.
///
/// The shape IS the schema — both extents come off it rather than being stored
/// separately (decision 2 in [`crate::persist`]'s docs). The returned [`Cow`]
/// borrows the mapped file bytes when the dtype matches `F`.
pub fn read_embedding<'a, F: Pod>(
    file: &ManifoldFile<'a>,
) -> Result<(Cow<'a, [F]>, usize, usize), PersistError> {
    let view = file.tensor(EMBEDDING_NAME)?;
    let (n_samples, n_components) = shape_2d(&view, EMBEDDING_NAME)?;
    if n_samples == 0 || n_components == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{EMBEDDING_NAME}' declares shape [{n_samples}, {n_components}]; a \
                 fitted embedding has at least one sample and one component"
            ),
        });
    }
    Ok((
        as_floats::<F>(&view, EMBEDDING_NAME)?,
        n_samples,
        n_components,
    ))
}

/// Read an OPTIONAL `f64` vector — the shape `Tsne`'s `metric_params` payloads
/// (`v`, `vi`, `w`) and its explicit-init array all take.
///
/// `Ok(None)` when the tensor is absent, which is exactly what a model that did
/// not carry one wrote. A present-but-non-rank-1 tensor is an error rather than
/// a `None`.
pub fn read_opt_f64_vec(
    file: &ManifoldFile<'_>,
    name: &'static str,
) -> Result<Option<Vec<f64>>, PersistError> {
    let Some(view) = file.tensor_opt(name) else {
        return Ok(None);
    };
    shape_1d(&view, name)?;
    Ok(Some(as_f64(&view, name)?.into_owned()))
}

/// Stage an OPTIONAL `f64` vector, mirroring [`read_opt_f64_vec`]. Absent means
/// no tensor at all rather than an empty one, so `Option` round-trips as
/// tensor-presence and costs zero bytes when `None`.
pub fn write_opt_f64_vec<'a>(
    w: &mut ManifoldWriter<'a>,
    name: &str,
    values: Option<&'a Vec<f64>>,
) -> Result<(), PersistError> {
    if let Some(v) = values {
        w.tensor(name, TensorRef::f64s(v, vec![v.len()])?);
    }
    Ok(())
}
