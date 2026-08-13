//! `mixture_persist` (MIX-PERSIST, prototype) — the `mlrs-mixture` half of the
//! mlrs model file format: the container discriminator, the aliases the two
//! mixture estimators write and read through, and the helpers for the state
//! shapes they share.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## Everything here is `F64`, whatever the estimator's `F`
//!
//! Both mixtures run their EM loop entirely in `f64` on every backend
//! ([`gmm_host`](mlrs_backend::prims::gmm_host)): the loop is launch-bound and
//! `f64`-bound, with a serial `O(k·d³)` Cholesky tail that loses its
//! conditioning at `f32`. So [`MixtureParams`](super::gaussian_mixture::MixtureParams)
//! is `Vec<f64>` regardless of the estimator's `F`, and the file stores what the
//! model holds.
//!
//! This is the same call [`IncrementalPCA`](crate::decomposition::IncrementalPCA)
//! makes, and for the same reason: narrowing to `F` on save would not be a
//! storage decision but a MODEL change. A reloaded mixture is a valid
//! `warm_start` continuation point, and a `precisions_cholesky_` rounded to
//! `f32` would restart the EM loop from a factorization the saved model never
//! computed.
//!
//! ## The parameter block is stored, and the precisions are stored WITH it
//!
//! `precisions_cholesky_` is derivable from `covariances_` — it IS the Cholesky
//! factor of their inverses — so the tempting move is to store one and recover
//! the other. It is rejected for the reason this format rejects every
//! recompute-on-load: the derivation is an `O(k·d³)` factorization on a path
//! that is otherwise one sequential read, and it is the numerically delicate
//! step of the whole algorithm. Recovering it would mean a reloaded model
//! scoring subtly differently from the saved one whenever the factorization is
//! not reproduced bit for bit, which for a near-singular covariance it will not
//! be. sklearn stores both attributes for the same reason.
//!
//! The flat length of both blocks depends on `covariance_type` —
//! [`CovarianceType::param_len`] — which is why the parameter tensors are
//! written FLAT with their geometry checked against that function rather than
//! shaped `[k, d, d]`: `tied` shares one matrix across components and
//! `spherical` holds one scalar each, so a single rank-3 shape would be wrong
//! for three of the four parameterizations.
//!
//! Tests live in `crates/mlrs-algos/tests/mixture_persist_test.rs`
//! (AGENTS.md §2).

use mlrs_backend::prims::gmm_host::CovarianceType;

// The container is shared with every other family; only the discriminator and
// the mixture-shaped helpers below are local. Re-exported (not just imported)
// so `mixture::mixture_persist::{AlignedBytes, SaveModel, …}` is the single
// import path for a mixture's `save`/`load`.
pub use crate::persist::{
    as_f64, as_i64, expect_len, shape_1d, AlignedBytes, Container, LoadModel, ModelFile,
    ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The mixture container discriminator (`format = "mlrs-mixture"`).
pub struct MixtureContainer;

impl Container for MixtureContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-mixture";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`MixtureFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The mixture writer: [`ModelWriter`] pinned to the `mlrs-mixture` container.
pub type MixtureWriter<'a> = ModelWriter<'a, MixtureContainer>;

/// The mixture reader: [`ModelFile`] pinned to the `mlrs-mixture` container.
pub type MixtureFile<'a> = ModelFile<'a, MixtureContainer>;

/// The tensor holding the per-component mixing weights, `[n_components]`.
pub const WEIGHTS_NAME: &str = "weights_";
/// The tensor holding the component means, flat `[n_components * n_features]`.
pub const MEANS_NAME: &str = "means_";
/// The tensor holding the covariances, flat — length from
/// [`CovarianceType::param_len`].
pub const COVARIANCES_NAME: &str = "covariances_";
/// The tensor holding the Cholesky factors of the precisions, same flat length
/// as [`COVARIANCES_NAME`].
pub const PRECISIONS_CHOLESKY_NAME: &str = "precisions_cholesky_";
/// The tensor holding the per-iteration lower bound trace, `[n_iter]`.
pub const LOWER_BOUNDS_NAME: &str = "lower_bounds_";
/// The tensor holding the training-set hard assignment, `[n_samples]` — what
/// `fit_predict` returned, kept so a reloaded model can report it without a
/// second E-step.
pub const TRAIN_LABELS_NAME: &str = "train_labels";

/// Stage the four parameter blocks that make up a fitted mixture's density.
///
/// The geometry is validated against `covariance_type` HERE rather than left to
/// the reader: `weights_` fixes `n_components`, `means_` fixes `n_features`, and
/// the two parameter blocks must both be exactly
/// [`CovarianceType::param_len`] long for that pair. A mismatch on the save side
/// is a bug in the estimator, and catching it before the bytes reach disk is the
/// difference between a failed save and a corrupt file.
pub fn write_mixture_params<'a>(
    w: &mut MixtureWriter<'a>,
    names: &MixtureParamNames,
    weights: &'a [f64],
    means: &'a [f64],
    covariances: &'a [f64],
    precisions_cholesky: &'a [f64],
    covariance_type: CovarianceType,
) -> Result<(usize, usize), PersistError> {
    let k = weights.len();
    if k == 0 || means.is_empty() || means.len() % k != 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{}' holds {k} components and '{}' {} entries; the means must be a \
                 positive multiple of the component count",
                names.weights,
                names.means,
                means.len()
            ),
        });
    }
    let d = means.len() / k;
    let param_len = covariance_type.param_len(k, d);
    expect_len(names.covariances, covariances.len(), param_len, "entries")?;
    expect_len(
        names.precisions_cholesky,
        precisions_cholesky.len(),
        param_len,
        "entries",
    )?;

    w.tensor(names.weights, TensorRef::f64s(weights, vec![k])?);
    w.tensor(names.means, TensorRef::f64s(means, vec![k, d])?);
    w.tensor(
        names.covariances,
        TensorRef::f64s(covariances, vec![param_len])?,
    );
    w.tensor(
        names.precisions_cholesky,
        TensorRef::f64s(precisions_cholesky, vec![param_len])?,
    );
    Ok((k, d))
}

/// The four parameter blocks recovered from a file, with the geometry they
/// imply.
pub struct MixtureParamsRaw {
    /// Per-component mixing weights.
    pub weights: Vec<f64>,
    /// Component means, flat `k × d`.
    pub means: Vec<f64>,
    /// Covariances, flat.
    pub covariances: Vec<f64>,
    /// Cholesky factors of the precisions, flat.
    pub precisions_cholesky: Vec<f64>,
    /// Component count, from `weights_`'s length.
    pub n_components: usize,
    /// Feature count, from `means_`'s column extent.
    pub n_features: usize,
}

/// The four tensor names one parameter block is stored under.
///
/// A named set rather than a runtime prefix because [`PersistError`] carries
/// `&'static str` throughout — every error in this format names its tensor
/// without allocating — and because it is what lets the same reader serve two
/// blocks: the FITTED parameters and the `warm_start` ones are the same four
/// arrays under different names, and a mixture saved mid-`warm_start` carries
/// both.
pub struct MixtureParamNames {
    /// The mixing-weight tensor's name.
    pub weights: &'static str,
    /// The means tensor's name.
    pub means: &'static str,
    /// The covariances tensor's name.
    pub covariances: &'static str,
    /// The precision-Cholesky tensor's name.
    pub precisions_cholesky: &'static str,
}

/// The names a FITTED parameter block is stored under — sklearn's own attribute
/// names, so `safetensors.numpy.load_file(path)` in Python hands back a dict
/// keyed the way the sklearn estimator is.
pub const FITTED_NAMES: MixtureParamNames = MixtureParamNames {
    weights: WEIGHTS_NAME,
    means: MEANS_NAME,
    covariances: COVARIANCES_NAME,
    precisions_cholesky: PRECISIONS_CHOLESKY_NAME,
};

/// The names a `warm_start` parameter block is stored under. Prefixed rather
/// than suffixed so the two blocks sort apart in the header, and so a Python
/// reader sees at a glance which dict entries are the model and which are the
/// resumption point.
pub const WARM_NAMES: MixtureParamNames = MixtureParamNames {
    weights: "warm_weights_",
    means: "warm_means_",
    covariances: "warm_covariances_",
    precisions_cholesky: "warm_precisions_cholesky_",
};

/// Read back everything [`write_mixture_params`] staged, under `names`.
///
/// The file is UNTRUSTED input (T-04-01-01), so `covariance_type` is consulted
/// and both parameter blocks are measured against
/// [`CovarianceType::param_len`] before a single value is handed back: a `tied`
/// model whose `covariances_` is `full`-length would otherwise index past the
/// end of the shared matrix on the first `score_samples`.
pub fn read_mixture_params(
    file: &MixtureFile<'_>,
    names: &MixtureParamNames,
    covariance_type: CovarianceType,
) -> Result<MixtureParamsRaw, PersistError> {
    let (weights_n, means_n, cov_n, prec_n) = (
        names.weights,
        names.means,
        names.covariances,
        names.precisions_cholesky,
    );

    let weights_v = file.tensor(weights_n)?;
    let k = shape_1d(&weights_v, weights_n)?;
    if k == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{weights_n}' is empty; a fitted mixture has at least one component"
            ),
        });
    }

    let means_v = file.tensor(means_n)?;
    let (means_rows, d) = crate::persist::shape_2d(&means_v, means_n)?;
    expect_len(means_n, means_rows, k, "rows")?;
    if d == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{means_n}' declares 0 features; a fitted mixture has at least one"
            ),
        });
    }

    let param_len = covariance_type.param_len(k, d);
    let cov_v = file.tensor(cov_n)?;
    expect_len(cov_n, shape_1d(&cov_v, cov_n)?, param_len, "entries")?;
    let prec_v = file.tensor(prec_n)?;
    expect_len(prec_n, shape_1d(&prec_v, prec_n)?, param_len, "entries")?;

    Ok(MixtureParamsRaw {
        weights: as_f64(&weights_v, weights_n)?.into_owned(),
        means: as_f64(&means_v, means_n)?.into_owned(),
        covariances: as_f64(&cov_v, cov_n)?.into_owned(),
        precisions_cholesky: as_f64(&prec_v, prec_n)?.into_owned(),
        n_components: k,
        n_features: d,
    })
}

/// Stage an OPTIONAL `f64` vector — the shape every mixture `*_init`
/// hyperparameter and prior takes.
///
/// Absent means no tensor at all rather than an empty one, so `Option`
/// round-trips as tensor-presence and costs zero bytes when `None` — which is
/// the common case, since all of `weights_init`, `means_init`,
/// `precisions_init`, `mean_prior` and `covariance_prior` default to unset.
pub fn write_opt_vec<'a>(
    w: &mut MixtureWriter<'a>,
    name: &str,
    values: Option<&'a Vec<f64>>,
) -> Result<(), PersistError> {
    if let Some(v) = values {
        w.tensor(name, TensorRef::f64s(v, vec![v.len()])?);
    }
    Ok(())
}

/// Read back what [`write_opt_vec`] staged. `Ok(None)` when the tensor is
/// absent; a present-but-non-rank-1 tensor is an error rather than a `None`.
pub fn read_opt_vec(
    file: &MixtureFile<'_>,
    name: &'static str,
) -> Result<Option<Vec<f64>>, PersistError> {
    let Some(view) = file.tensor_opt(name) else {
        return Ok(None);
    };
    shape_1d(&view, name)?;
    Ok(Some(as_f64(&view, name)?.into_owned()))
}

/// Map the `device_` diagnostic back to a `&'static str`.
///
/// The estimators hold it as `&'static str` because it is one of exactly two
/// values, and a file cannot hand back a `'static` borrow of its own bytes — so
/// the string is matched against the known arms rather than leaked. An
/// unrecognised value is an error: `device_` reports which arm RAN, and a model
/// claiming an arm this build does not have is a file from a different build.
pub fn read_device_arm(
    file: &MixtureFile<'_>,
    key: &'static str,
) -> Result<Option<&'static str>, PersistError> {
    match file.metadata().get(key).map(String::as_str) {
        None => Ok(None),
        Some("cpu") => Ok(Some("cpu")),
        Some("gpu") => Ok(Some("gpu")),
        Some(_) => Err(PersistError::BadMetadata { key }),
    }
}
