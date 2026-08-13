//! `prep_persist` (PREP-PERSIST, prototype) — the `mlrs-prep` half of the mlrs
//! model file format: the container discriminator, the aliases the six
//! preprocessing transformers write and read through, and the one shared shape
//! they all reduce to.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic; the Naive Bayes
//! ([`nb_persist`](crate::naive_bayes::nb_persist)) and dense-linear
//! ([`linear_persist`](crate::linear::linear_persist)) families sit on the same
//! machinery under their own tags. Read [`crate::persist`]'s docs for the four
//! decisions that make the files small and the loads fast. This module holds
//! only what is specific to a preprocessing transformer, and re-exports the rest
//! so estimator code and the test suite reach everything through one path.
//!
//! ## The on-disk shape
//!
//! A fitted transformer in this family is a handful of per-COLUMN statistic
//! vectors, every one of them length `n_features`, plus its constructor scalars:
//!
//! | estimator | tensors | extra `param:` scalars |
//! |---|---|---|
//! | `StandardScaler` | `mean_`, `var_`, `scale_` | `with_mean`, `with_std` |
//! | `MinMaxScaler` | `data_min_`, `data_max_`, `scale_`, `min_` | `feature_range_min`/`_max`, `clip` |
//! | `MaxAbsScaler` | `max_abs_`, `scale_` | — |
//! | `RobustScaler` | `center_`, `scale_` | `with_centering`, `with_scaling`, `quantile_min`/`_max`, `unit_variance` |
//! | `Binarizer` | — | `threshold` |
//! | `Normalizer` | — | `norm` |
//!
//! That uniformity is the whole reason this module is small: [`write_columns`]
//! and [`read_columns`] handle the four scalers between them, and the geometry
//! rule is one sentence — every vector in the file has the same length, and that
//! length IS `n_features_in_`.
//!
//! ## Why the last two store `n_features_in_` and the first four do not
//!
//! [`crate::persist`]'s decision 2 is "nothing derivable is stored": the four
//! scalers recover `n_features_in_` off their first tensor's shape, so writing it
//! again would only create a second copy that a hand-edited header could make
//! disagree with the first.
//!
//! `Binarizer` and `Normalizer` have no data-dependent state at all — their
//! `fit` learns exactly one thing, the column count it was shown, which
//! `transform` then enforces. With no tensor to read it off, the derivation is
//! unavailable and the scalar is the only place it can live, so those two write
//! [`N_FEATURES_KEY`] and the four scalers do not. The asymmetry is in the data,
//! not in the format: both halves store `n_features_in_` exactly once.
//!
//! Neither is written under [`PARAM_PREFIX`]. `n_features_in_` is sklearn's
//! FITTED attribute — an output of `fit`, not a constructor input — and the
//! prefix is what tells the two apart when a file is read from Python.
//!
//! Tests live in `crates/mlrs-algos/tests/prep_persist_test.rs` (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

// The container is shared with every other family; only the discriminator and
// the column-vector helpers below are local. Re-exported (not just imported) so
// `preprocessing::prep_persist::{AlignedBytes, SaveModel, …}` is the single
// import path for a transformer's `save`/`load`.
pub use crate::persist::{
    as_floats, expect_len, shape_1d, AlignedBytes, Container, LoadModel, ModelFile, ModelWriter,
    PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The preprocessing container discriminator (`format = "mlrs-prep"`).
///
/// A zero-sized marker, never constructed — it exists only to carry the two tags
/// [`ModelFile::parse`] validates, so `PrepWriter`/`PrepFile` cannot be
/// instantiated with another family's tag by accident, and a `Ridge` file cannot
/// reach a scaler's geometry checks at all.
pub struct PrepContainer;

impl Container for PrepContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`. Present so a `.safetensors` file produced
/// by some other project — or by another mlrs family — fails the check at
/// [`PrepFile::parse`].
pub const FORMAT_ID: &str = "mlrs-prep";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`PrepFile::parse`] rejects anything else outright (a prototype has
/// no back-compat obligations, so this is a hard equality, not a range).
pub const FORMAT_VERSION: &str = "1";

/// The `__metadata__` key holding the fitted column count, written ONLY by the
/// two transformers that have no tensor to recover it from. See the module docs
/// for why the asymmetry is the honest encoding rather than an inconsistency.
pub const N_FEATURES_KEY: &str = "n_features_in_";

/// The preprocessing writer: [`ModelWriter`] pinned to the `mlrs-prep`
/// container, so a transformer's `save` names only itself
/// (`PrepWriter::new("standard_scaler")`) and never a format id.
pub type PrepWriter<'a> = ModelWriter<'a, PrepContainer>;

/// The preprocessing reader: [`ModelFile`] pinned to the `mlrs-prep` container.
/// Loading an `mlrs-linear` file through it fails with
/// [`PersistError::NotAnMlrsModel`] before any tensor is touched.
pub type PrepFile<'a> = ModelFile<'a, PrepContainer>;

// ---------------------------------------------------------------------------
// The shared shape: k per-column vectors, all length n_features
// ---------------------------------------------------------------------------

/// Stage `columns` as one `[n_features]` tensor each, at the model's OWN float
/// width.
///
/// Every vector is required to be the same length, and that is checked HERE
/// rather than left to the reader: a `scale_` shorter than its `mean_` is a bug
/// on the save side, and catching it before the bytes reach disk is the
/// difference between a failed save and a corrupt file that only fails the next
/// time someone loads it.
///
/// Takes `&'a [(&str, &'a [F])]`-shaped pairs rather than a struct per estimator
/// because the four scalers differ ONLY in how many vectors they have and what
/// they are called — there is no per-estimator invariant left to encode once the
/// equal-length rule is enforced.
///
/// Writing each statistic as its own `[n_features]` tensor rather than one fused
/// `[k, n_features]` matrix costs ~60 bytes of JSON header per extra entry and
/// buys two things worth more than that: `safetensors.numpy.load_file(path)` in
/// Python hands back a dict keyed exactly the way the sklearn estimator is
/// (`d["scale_"]`, not `d["stats_"][2]`), and each vector loads back as its own
/// borrowed slice, so no upload has to be strided out of a shared block.
pub fn write_columns<'a, F: Pod>(
    w: &mut PrepWriter<'a>,
    columns: &[(&'static str, &'a [F])],
) -> Result<(), PersistError> {
    let Some((first_name, first)) = columns.first() else {
        return Ok(());
    };
    let n_features = first.len();
    if n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{first_name}' is empty; a fitted transformer has at least one feature"
            ),
        });
    }
    for (name, values) in columns {
        expect_len(name, values.len(), n_features, "entries")?;
        w.tensor(name, TensorRef::floats(*values, vec![n_features])?);
    }
    Ok(())
}

/// Read back the vectors [`write_columns`] staged, in the same order.
///
/// The FIRST name defines the geometry and every other is measured against it,
/// which is the whole guard against a tampered header: the file is untrusted
/// input (T-04-01-01), and a `scale_` one element short of `mean_` would
/// otherwise read out of bounds the first time the transformer is applied rather
/// than fail here.
///
/// Each result is a [`Cow`], not a `Vec`: when the file's dtype matches `F` it
/// BORROWS the bytes `read_exact` landed, so the vector reaches
/// [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray::from_host)
/// with no intervening allocation and no per-element decode. Owning them here
/// would silently undo the zero-copy read path for the whole family.
pub fn read_columns<'a, F: Pod>(
    file: &PrepFile<'a>,
    names: &[&'static str],
) -> Result<(Vec<Cow<'a, [F]>>, usize), PersistError> {
    let mut out = Vec::with_capacity(names.len());
    let mut n_features = 0usize;
    for (i, &name) in names.iter().enumerate() {
        let view = file.tensor(name)?;
        let len = shape_1d(&view, name)?;
        if i == 0 {
            if len == 0 {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!(
                        "tensor '{name}' declares shape [0]; a fitted transformer \
                         has at least one feature"
                    ),
                });
            }
            n_features = len;
        } else {
            expect_len(name, len, n_features, "entries")?;
        }
        out.push(as_floats::<F>(&view, name)?);
    }
    Ok((out, n_features))
}

/// Read the [`N_FEATURES_KEY`] scalar for the two transformers that carry no
/// tensor, rejecting a zero.
///
/// A `n_features_in_ = 0` transformer would accept only an empty matrix, which
/// no `transform` can produce anything useful from and which is a landmine on
/// the device backends; it is far more likely to be a truncated or hand-written
/// header. Rejecting it here keeps the two stateless transformers under the same
/// non-degeneracy rule [`read_columns`] applies to the four scalers.
pub fn read_n_features(file: &PrepFile<'_>) -> Result<usize, PersistError> {
    let n_features = file.scalar_usize(N_FEATURES_KEY)?;
    if n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{N_FEATURES_KEY}' is 0; a fitted transformer has at least one feature"
            ),
        });
    }
    Ok(n_features)
}

/// Stage a `(min, max)` pair as two scalars under `<key>_min` / `<key>_max`.
///
/// `MinMaxScaler`'s `feature_range` and `RobustScaler`'s `quantile_range` are
/// both `(f64, f64)` constructor arguments, and both would otherwise be spelled
/// out twice per estimator across `save`/`load` — four spellings of a key whose
/// halves are trivially transposable. Splitting the pair into two named scalars
/// rather than one `"0.0,1.0"` string keeps each half individually parsable by a
/// Python reader and makes a truncated value a
/// [`PersistError::BadMetadata`] naming the half that failed.
pub fn write_range(w: &mut PrepWriter<'_>, min_key: &str, max_key: &str, range: (f64, f64)) {
    w.scalar_f64(min_key, range.0);
    w.scalar_f64(max_key, range.1);
}

/// Read back the pair [`write_range`] staged. Both halves are REQUIRED: a
/// missing key is a corrupt file, not a request for the default — silently
/// substituting `(0, 1)` would hand back a transformer that maps its input to
/// different numbers than the one that was saved.
pub fn read_range(
    file: &PrepFile<'_>,
    min_key: &'static str,
    max_key: &'static str,
) -> Result<(f64, f64), PersistError> {
    Ok((file.scalar_f64(min_key)?, file.scalar_f64(max_key)?))
}
