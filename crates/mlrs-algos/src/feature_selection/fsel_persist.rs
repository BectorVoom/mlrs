//! `fsel_persist` (FSEL-PERSIST, prototype) — the `mlrs-fsel` half of the mlrs
//! model file format: the container discriminator, the aliases the six feature
//! selectors write and read through, and the support mask every one of them
//! reduces to.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## The whole family is one boolean mask
//!
//! Every selector here answers the same question — which columns survive — and
//! `transform` is a column gather driven by exactly that. So the fitted state
//! that MATTERS is a `[n_features]` `BOOL` tensor, and everything else in these
//! files is the evidence behind it: `variances_`, `scores_`, `pvalues_`,
//! `ranking_`, the resolved `threshold_`.
//!
//! That makes these the smallest files in mlrs after `Binarizer`'s, and it makes
//! the geometry rule one sentence: every array is length `n_features`, and
//! `support_`'s length IS `n_features`.
//!
//! ## The four meta-selectors cannot round-trip through [`LoadModel`]
//!
//! `SelectFromModel`, `Rfe`, `Rfecv` and `SequentialFeatureSelector` are
//! parameterized over a caller-supplied [`ImportanceEstimator`] or
//! [`FoldScorer`] — a trait object or closure that fits the inner model. A Rust
//! closure has no on-disk representation, and no amount of format design
//! changes that.
//!
//! mlrs does NOT paper over it. Those four implement [`SaveModel`] (the fitted
//! selection is perfectly serializable) but NOT [`LoadModel`], whose
//! `load(pool, path)` signature has no slot for the missing half. They provide
//! `load_with(pool, path, estimator)` instead, which takes it from the caller.
//!
//! The alternative — a `Default` bound on the estimator so `load` could
//! construct one — was rejected outright: it would hand back a selector whose
//! `support_` came from one model and whose `estimator` is a different one, and
//! nothing about that is visible until someone re-fits.
//!
//! For the same reason a selector fitted with an
//! [`ImportanceGetter::Custom`](super::meta::ImportanceGetter::Custom) records
//! that fact and REFUSES to load: the custom post-processor is a closure too, and
//! silently substituting `Auto` would change what a re-fit selects.
//!
//! Tests live in `crates/mlrs-algos/tests/fsel_persist_test.rs` (AGENTS.md §2).

// The container is shared with every other family; only the discriminator and
// the mask helpers below are local. Re-exported (not just imported) so
// `feature_selection::fsel_persist::{AlignedBytes, SaveModel, …}` is the single
// import path for a selector's `save`/`load`.
pub use crate::persist::{
    as_bools, as_f64, as_usizes, expect_len, pack_bools, shape_1d, AlignedBytes, Container,
    LoadModel, ModelFile, ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

use super::mutual_info::{DiscreteFeatures, MutualInfoParams};
use super::score::ScoreFunc;
use super::univariate::{KBest, SelectionMode};

/// The feature-selection container discriminator (`format = "mlrs-fsel"`).
pub struct FselContainer;

impl Container for FselContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-fsel";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`FselFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the boolean keep mask, `[n_features]` — sklearn's
/// `get_support()`.
///
/// `BOOL`, one byte per flag, so `safetensors.numpy.load_file(path)` in Python
/// hands back a plain `bool` array of the right length. See
/// [`TensorRef::bools`](crate::persist::TensorRef::bools) for why the format
/// does not bit-pack it.
pub const SUPPORT_NAME: &str = "support_";

/// The feature-selection writer: [`ModelWriter`] pinned to the `mlrs-fsel`
/// container.
pub type FselWriter<'a> = ModelWriter<'a, FselContainer>;

/// The feature-selection reader: [`ModelFile`] pinned to the `mlrs-fsel`
/// container.
pub type FselFile<'a> = ModelFile<'a, FselContainer>;

/// Stage the support mask, rejecting an empty one.
///
/// A selector fitted on zero features is not a degenerate model but an absent
/// one — every member of this family learns a mask over the columns it was
/// shown, so a zero-length mask means the fit never happened.
///
/// `packed` must come from [`pack_bools`] and be bound BEFORE the writer, which
/// borrows it. `bool` is not `Pod`, so the conversion is unavoidable; it is the
/// one copy this whole family's save path costs.
pub fn write_support<'a>(w: &mut FselWriter<'a>, packed: &'a [u8]) -> Result<(), PersistError> {
    if packed.is_empty() {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{SUPPORT_NAME}' is empty; a fitted selector has a mask over at least one \
                 feature"
            ),
        });
    }
    w.tensor(SUPPORT_NAME, TensorRef::bools(packed, vec![packed.len()])?);
    Ok(())
}

/// Read the support mask back, with the `n_features` it establishes.
///
/// The mask's length IS the geometry for this whole family, so it is read first
/// and every other array is measured against it.
pub fn read_support(file: &FselFile<'_>) -> Result<Vec<bool>, PersistError> {
    let view = file.tensor(SUPPORT_NAME)?;
    let n_features = shape_1d(&view, SUPPORT_NAME)?;
    if n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{SUPPORT_NAME}' is empty; a fitted selector has a mask over at \
                 least one feature"
            ),
        });
    }
    as_bools(&view, SUPPORT_NAME)
}

/// Stage a REQUIRED `[n_features]` `f64` evidence array (`variances_`,
/// `scores_`, …).
pub fn write_f64_vec<'a>(
    w: &mut FselWriter<'a>,
    name: &str,
    values: &'a [f64],
    n_features: usize,
) -> Result<(), PersistError> {
    expect_len(SUPPORT_NAME, values.len(), n_features, "entries")?;
    w.tensor(name, TensorRef::f64s(values, vec![n_features])?);
    Ok(())
}

/// Read a REQUIRED `[n_features]` `f64` evidence array back.
pub fn read_f64_vec(
    file: &FselFile<'_>,
    name: &'static str,
    n_features: usize,
) -> Result<Vec<f64>, PersistError> {
    let view = file.tensor(name)?;
    expect_len(name, shape_1d(&view, name)?, n_features, "entries")?;
    Ok(as_f64(&view, name)?.into_owned())
}

/// Read an OPTIONAL `[n_features]` `f64` array — `pvalues_`, which three of the
/// five score functions do not produce.
///
/// `Ok(None)` when the tensor is absent, which is exactly what a
/// `r_regression`-scored filter wrote. A present-but-misshapen array is an error
/// rather than a `None`.
pub fn read_opt_f64_vec(
    file: &FselFile<'_>,
    name: &'static str,
    n_features: usize,
) -> Result<Option<Vec<f64>>, PersistError> {
    let Some(view) = file.tensor_opt(name) else {
        return Ok(None);
    };
    expect_len(name, shape_1d(&view, name)?, n_features, "entries")?;
    Ok(Some(as_f64(&view, name)?.into_owned()))
}

/// The `__metadata__` key recording whether the selector's importance
/// post-processor was a caller-supplied closure.
///
/// A `true` here makes `load_with` refuse. See this module's docs: the closure
/// has no on-disk form, and substituting `Auto` would change what a re-fit
/// selects while leaving the stored `support_` looking authoritative.
pub const CUSTOM_GETTER_KEY: &str = "importance_getter_is_custom";

/// Refuse to load a selector whose importance getter was a closure.
///
/// Called by every meta-selector's `load_with` before any state is restored, so
/// the refusal names the reason rather than surfacing later as a re-fit that
/// selects different columns.
pub fn reject_custom_getter(file: &FselFile<'_>) -> Result<(), PersistError> {
    if file
        .metadata()
        .get(CUSTOM_GETTER_KEY)
        .is_some_and(|v| v == "true")
    {
        return Err(PersistError::InconsistentGeometry {
            reason: "this selector was fitted with a custom importance_getter, which is a \
                     closure and has no on-disk representation; re-fit it rather than \
                     loading a selector whose getter would silently differ"
                .to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The two enum-shaped hyperparameters the univariate filter carries
// ---------------------------------------------------------------------------

/// Stage a [`ScoreFunc`] as its sklearn name plus whatever payload the variant
/// carries.
///
/// The five built-in functions each take zero or a handful of scalars, and they
/// ride under `param:score_*` keys named for the field rather than positionally
/// — so a variant that grows a knob later adds a key instead of shifting an
/// existing one's meaning.
///
/// [`ScoreFunc::Custom`] has NO representation: it wraps a caller-supplied
/// closure. Rather than write a file that would load as `f_classif` and score
/// every column differently, `save` fails with a
/// [`PersistError::MissingState`]-shaped refusal — see
/// [`UnivariateFilter::save`](super::univariate::UnivariateFilter).
pub fn write_score_func(w: &mut FselWriter<'_>, f: &ScoreFunc) -> Result<(), PersistError> {
    let name = match f {
        ScoreFunc::FClassif => "f_classif",
        ScoreFunc::Chi2 => "chi2",
        ScoreFunc::RRegression { .. } => "r_regression",
        ScoreFunc::FRegression { .. } => "f_regression",
        ScoreFunc::MutualInfoClassif(_) => "mutual_info_classif",
        ScoreFunc::MutualInfoRegression(_) => "mutual_info_regression",
        ScoreFunc::Custom(_) => {
            return Err(PersistError::MissingState {
                estimator: "univariate_filter",
                field: "score_func (a custom closure has no on-disk representation)",
            })
        }
    };
    w.scalar_str("param:score_func", name);
    match f {
        ScoreFunc::RRegression {
            center,
            force_finite,
        }
        | ScoreFunc::FRegression {
            center,
            force_finite,
        } => {
            w.scalar_bool("param:score_center", *center);
            w.scalar_bool("param:score_force_finite", *force_finite);
        }
        ScoreFunc::MutualInfoClassif(p) | ScoreFunc::MutualInfoRegression(p) => {
            // `Mask` is the one arm carrying an array, so it names itself in the
            // scalar and its payload goes in a companion tensor. A caller that
            // wants the mask stored must pass it through
            // `write_discrete_mask` before the writer takes ownership.
            w.scalar_str("param:score_discrete_features", p.discrete_features.name());
            w.scalar_usize("param:score_n_neighbors", p.n_neighbors);
            w.scalar_bool("param:score_copy", p.copy);
            w.scalar_opt_u64("param:score_random_state", p.random_state);
            if let Some(j) = p.n_jobs {
                w.scalar_usize("param:score_n_jobs", j);
            }
        }
        _ => {}
    }
    Ok(())
}

/// Read back what [`write_score_func`] staged.
///
/// An unrecognised name is a [`PersistError::BadMetadata`] naming the key, never
/// a fallback to the `f_classif` default: the five functions score columns
/// completely differently, so a silent substitution would produce a filter that
/// selects other features than the saved one with nothing to signal it.
pub fn read_score_func(file: &FselFile<'_>) -> Result<ScoreFunc, PersistError> {
    let params = |file: &FselFile<'_>| -> Result<MutualInfoParams, PersistError> {
        Ok(MutualInfoParams {
            discrete_features: read_discrete_features(file)?,
            n_neighbors: file.scalar_usize("param:score_n_neighbors")?,
            copy: file.scalar_bool("param:score_copy")?,
            random_state: file.scalar_opt_u64("param:score_random_state")?,
            n_jobs: file.scalar_opt_usize("param:score_n_jobs")?,
        })
    };
    match file.scalar_str("param:score_func")? {
        "f_classif" => Ok(ScoreFunc::FClassif),
        "chi2" => Ok(ScoreFunc::Chi2),
        "r_regression" => Ok(ScoreFunc::RRegression {
            center: file.scalar_bool("param:score_center")?,
            force_finite: file.scalar_bool("param:score_force_finite")?,
        }),
        "f_regression" => Ok(ScoreFunc::FRegression {
            center: file.scalar_bool("param:score_center")?,
            force_finite: file.scalar_bool("param:score_force_finite")?,
        }),
        "mutual_info_classif" => Ok(ScoreFunc::MutualInfoClassif(params(file)?)),
        "mutual_info_regression" => Ok(ScoreFunc::MutualInfoRegression(params(file)?)),
        _ => Err(PersistError::BadMetadata {
            key: "param:score_func",
        }),
    }
}

/// The tensor holding a `discrete_features` mask, `[n_features]`.
pub const DISCRETE_MASK_NAME: &str = "param:score_discrete_mask";

/// Read the `discrete_features` argument back, taking the mask arm's payload
/// from its companion tensor.
///
/// A `"mask"` scalar with no tensor — or a tensor with any other scalar — is a
/// file whose two halves describe different arguments, and is rejected rather
/// than resolved in favour of either.
pub fn read_discrete_features(file: &FselFile<'_>) -> Result<DiscreteFeatures, PersistError> {
    let name = file.scalar_str("param:score_discrete_features")?;
    let mask = file.tensor_opt(DISCRETE_MASK_NAME);
    match (name, mask) {
        ("mask", Some(view)) => {
            shape_1d(&view, DISCRETE_MASK_NAME)?;
            Ok(DiscreteFeatures::Mask(as_bools(&view, DISCRETE_MASK_NAME)?))
        }
        ("mask", None) => Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'param:score_discrete_features' is 'mask' but the '{DISCRETE_MASK_NAME}' \
                 tensor is absent"
            ),
        }),
        (other, Some(_)) => Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'param:score_discrete_features' is '{other}' but a '{DISCRETE_MASK_NAME}' \
                 tensor is present"
            ),
        }),
        (other, None) => DiscreteFeatures::from_name(other).ok_or(PersistError::BadMetadata {
            key: "param:score_discrete_features",
        }),
    }
}

/// Stage a [`SelectionMode`] as its sklearn `mode` string plus the one scalar
/// the mode takes.
///
/// The five modes each carry exactly one number, and `k_best` alone admits the
/// non-numeric `"all"` — which rides as that literal string in the same key,
/// the encoding [`NComponents`](crate::projection::NComponents) uses for
/// `'auto'`.
pub fn write_selection_mode(w: &mut FselWriter<'_>, mode: &SelectionMode) {
    w.scalar_str("param:mode", mode.name());
    match mode {
        SelectionMode::Percentile(v)
        | SelectionMode::Fpr(v)
        | SelectionMode::Fdr(v)
        | SelectionMode::Fwe(v) => w.scalar_f64("param:mode_param", *v),
        SelectionMode::KBest(KBest::Count(k)) => w.scalar_usize("param:mode_param", *k),
        SelectionMode::KBest(KBest::All) => w.scalar_str("param:mode_param", "all"),
    }
}

/// Read back what [`write_selection_mode`] staged.
pub fn read_selection_mode(file: &FselFile<'_>) -> Result<SelectionMode, PersistError> {
    let bad = PersistError::BadMetadata {
        key: "param:mode_param",
    };
    let raw = file.scalar_str("param:mode_param")?;
    match file.scalar_str("param:mode")? {
        "percentile" => Ok(SelectionMode::Percentile(
            raw.parse::<f64>().map_err(|_| bad)?,
        )),
        "k_best" => Ok(SelectionMode::KBest(if raw == "all" {
            KBest::All
        } else {
            KBest::Count(raw.parse::<usize>().map_err(|_| bad)?)
        })),
        "fpr" => Ok(SelectionMode::Fpr(raw.parse::<f64>().map_err(|_| bad)?)),
        "fdr" => Ok(SelectionMode::Fdr(raw.parse::<f64>().map_err(|_| bad)?)),
        "fwe" => Ok(SelectionMode::Fwe(raw.parse::<f64>().map_err(|_| bad)?)),
        _ => Err(PersistError::BadMetadata { key: "param:mode" }),
    }
}

/// Stage the ranking vector RFE and RFECV produce, `[n_features]`.
///
/// `U64` rather than the mask's `BOOL`: a rank is `1` for a selected feature and
/// counts upward for the order features were eliminated in, so it carries strictly
/// more information than `support_` and is not derivable from it.
pub fn write_ranking<'a>(
    w: &mut FselWriter<'a>,
    ranking: &'a [u64],
    n_features: usize,
) -> Result<(), PersistError> {
    expect_len(RANKING_NAME, ranking.len(), n_features, "entries")?;
    w.tensor(RANKING_NAME, TensorRef::u64s(ranking, vec![n_features])?);
    Ok(())
}

/// The tensor holding the elimination ranking, `[n_features]`.
pub const RANKING_NAME: &str = "ranking_";

/// Read the ranking vector back, checked against the mask's geometry.
///
/// The ranking and the mask must AGREE: sklearn's contract is that rank `1`
/// means selected, so a file whose two halves disagree describes two different
/// selections. Only the cross-check catches it — each is individually
/// well-formed.
pub fn read_ranking(file: &FselFile<'_>, support: &[bool]) -> Result<Vec<usize>, PersistError> {
    let view = file.tensor(RANKING_NAME)?;
    expect_len(
        RANKING_NAME,
        shape_1d(&view, RANKING_NAME)?,
        support.len(),
        "entries",
    )?;
    let ranking = as_usizes(&view, RANKING_NAME)?;
    for (i, (&rank, &kept)) in ranking.iter().zip(support.iter()).enumerate() {
        if rank == 0 {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{RANKING_NAME}' holds rank 0 at feature {i}; ranks start at 1"
                ),
            });
        }
        if (rank == 1) != kept {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "feature {i} has rank {rank} but '{SUPPORT_NAME}' says it is {}; rank 1 \
                     means selected",
                    if kept { "kept" } else { "dropped" }
                ),
            });
        }
    }
    Ok(ranking)
}

// ---------------------------------------------------------------------------
// The meta-selectors' enum-shaped hyperparameters
// ---------------------------------------------------------------------------

use super::meta::{
    Cv, CvResults, Direction, ImportanceGetter, NFeatures, RfeStep, SfsTarget, Threshold,
};

/// Stage a [`Threshold`] as its sklearn spelling.
///
/// `None` (sklearn's estimator-dependent default) rides as the literal
/// `"default"` rather than as an absent key, because absence already means
/// something else in this format — an `Option` that was not supplied — and
/// `Threshold::Default` is a deliberate choice the caller made.
pub fn write_threshold(w: &mut FselWriter<'_>, t: &Threshold) {
    match t {
        Threshold::Default => w.scalar_str("param:threshold", "default"),
        Threshold::Value(v) => w.scalar_str("param:threshold", &format!("{v:?}")),
        Threshold::Scaled { scale, median } => w.scalar_str(
            "param:threshold",
            &format!("{scale:?}*{}", if *median { "median" } else { "mean" }),
        ),
    }
}

/// Read back what [`write_threshold`] staged, reusing the estimator's own
/// [`Threshold::parse`] for the scaled and numeric forms so the file and the
/// constructor cannot disagree about what a spelling means.
pub fn read_threshold(file: &FselFile<'_>) -> Result<Threshold, PersistError> {
    let raw = file.scalar_str("param:threshold")?;
    if raw == "default" {
        return Ok(Threshold::Default);
    }
    Threshold::parse(raw).map_err(|_| PersistError::BadMetadata {
        key: "param:threshold",
    })
}

/// Stage an [`NFeatures`] target as `"half"`, an integer, or a decimal.
///
/// The three arms are distinguishable because a fraction always carries a point
/// and a count never does — the same disambiguation
/// [`MaxFeatures`](crate::ensemble::MaxFeatures) uses.
pub fn write_n_features(w: &mut FselWriter<'_>, key: &str, n: &NFeatures) {
    match n {
        NFeatures::Half => w.scalar_str(key, "half"),
        NFeatures::Count(c) => w.scalar_usize(key, *c),
        NFeatures::Fraction(f) => w.scalar_str(key, &format!("{f:?}")),
    }
}

/// Read back what [`write_n_features`] staged.
pub fn read_n_features(file: &FselFile<'_>, key: &'static str) -> Result<NFeatures, PersistError> {
    let raw = file.scalar_str(key)?;
    if raw == "half" {
        return Ok(NFeatures::Half);
    }
    if let Ok(c) = raw.parse::<usize>() {
        return Ok(NFeatures::Count(c));
    }
    raw.parse::<f64>()
        .map(NFeatures::Fraction)
        .map_err(|_| PersistError::BadMetadata { key })
}

/// Stage an [`SfsTarget`], mirroring [`write_n_features`] over its own three
/// arms.
pub fn write_sfs_target(w: &mut FselWriter<'_>, t: &SfsTarget) {
    match t {
        SfsTarget::Auto => w.scalar_str("param:n_features_to_select", "auto"),
        SfsTarget::Count(c) => w.scalar_usize("param:n_features_to_select", *c),
        SfsTarget::Fraction(f) => w.scalar_str("param:n_features_to_select", &format!("{f:?}")),
    }
}

/// Read back what [`write_sfs_target`] staged.
pub fn read_sfs_target(file: &FselFile<'_>) -> Result<SfsTarget, PersistError> {
    let key = "param:n_features_to_select";
    let raw = file.scalar_str(key)?;
    if raw == "auto" {
        return Ok(SfsTarget::Auto);
    }
    if let Ok(c) = raw.parse::<usize>() {
        return Ok(SfsTarget::Count(c));
    }
    raw.parse::<f64>()
        .map(SfsTarget::Fraction)
        .map_err(|_| PersistError::BadMetadata { key })
}

/// Stage an [`RfeStep`] as an integer count or a decimal fraction, the same
/// point-vs-no-point disambiguation the targets above use.
pub fn write_rfe_step(w: &mut FselWriter<'_>, step: &RfeStep) {
    match step {
        RfeStep::Count(c) => w.scalar_usize("param:step", *c),
        RfeStep::Fraction(f) => w.scalar_str("param:step", &format!("{f:?}")),
    }
}

/// Read back what [`write_rfe_step`] staged.
pub fn read_rfe_step(file: &FselFile<'_>) -> Result<RfeStep, PersistError> {
    let raw = file.scalar_str("param:step")?;
    if let Ok(c) = raw.parse::<usize>() {
        return Ok(RfeStep::Count(c));
    }
    raw.parse::<f64>()
        .map(RfeStep::Fraction)
        .map_err(|_| PersistError::BadMetadata { key: "param:step" })
}

/// Stage a [`Direction`] as its sklearn string.
pub fn write_direction(w: &mut FselWriter<'_>, d: Direction) {
    w.scalar_str(
        "param:direction",
        match d {
            Direction::Forward => "forward",
            Direction::Backward => "backward",
        },
    );
}

/// Read back what [`write_direction`] staged, never defaulting: the two
/// directions select different feature sets from the same data.
pub fn read_direction(file: &FselFile<'_>) -> Result<Direction, PersistError> {
    match file.scalar_str("param:direction")? {
        "forward" => Ok(Direction::Forward),
        "backward" => Ok(Direction::Backward),
        _ => Err(PersistError::BadMetadata {
            key: "param:direction",
        }),
    }
}

/// Record whether the importance getter was a closure, so `load_with` can
/// refuse. See [`reject_custom_getter`].
pub fn write_importance_getter(w: &mut FselWriter<'_>, g: &ImportanceGetter) {
    w.scalar_bool(CUSTOM_GETTER_KEY, matches!(g, ImportanceGetter::Custom(_)));
}

/// The tensor holding an explicit CV split's train indices, CSR-style.
pub const CV_TRAIN_INDPTR: &str = "param:cv_train_indptr";
/// See [`CV_TRAIN_INDPTR`].
pub const CV_TRAIN_INDICES: &str = "param:cv_train_indices";
/// See [`CV_TRAIN_INDPTR`].
pub const CV_TEST_INDPTR: &str = "param:cv_test_indptr";
/// See [`CV_TRAIN_INDPTR`].
pub const CV_TEST_INDICES: &str = "param:cv_test_indices";

/// The widened, flattened form of an explicit [`Cv`] split list, bound so it
/// outlives the writer that borrows it.
///
/// An explicit split list is genuinely RAGGED — folds need not be equal-sized,
/// and a train fold is never the same size as its test fold — so it is stored
/// CSR-style rather than padded: two `indptr`/`indices` pairs, exactly the
/// encoding [`cluster_persist`](crate::cluster::cluster_persist) uses for a
/// sparse affinity, and for the same reason. Padding would need a sentinel index
/// that could collide with a real row.
pub struct CvStaging {
    /// Train-fold row starts, length `n_splits + 1`. Empty for `Cv::Folds`.
    pub train_indptr: Vec<u64>,
    /// Train-fold row indices, concatenated.
    pub train_indices: Vec<u64>,
    /// Test-fold row starts, length `n_splits + 1`.
    pub test_indptr: Vec<u64>,
    /// Test-fold row indices, concatenated.
    pub test_indices: Vec<u64>,
}

impl CvStaging {
    /// Flatten whatever `cv` carries, ahead of the writer.
    pub fn prepare(cv: &Cv) -> Self {
        let mut s = CvStaging {
            train_indptr: Vec::new(),
            train_indices: Vec::new(),
            test_indptr: Vec::new(),
            test_indices: Vec::new(),
        };
        if let Cv::Explicit(splits) = cv {
            s.train_indptr.push(0);
            s.test_indptr.push(0);
            for (train, test) in splits {
                s.train_indices.extend(train.iter().map(|&v| v as u64));
                s.test_indices.extend(test.iter().map(|&v| v as u64));
                s.train_indptr.push(s.train_indices.len() as u64);
                s.test_indptr.push(s.test_indices.len() as u64);
            }
        }
        s
    }

    /// Stage the split specification.
    ///
    /// `Cv::Folds` is two scalars; `Cv::Explicit` is the four CSR arrays plus a
    /// discriminator. The discriminator is explicit rather than inferred from
    /// tensor presence, for the reason the affinity layout is: inferring works
    /// today and would silently mis-read a file that grew a third form.
    pub fn write_into<'a>(&'a self, w: &mut FselWriter<'a>, cv: &Cv) -> Result<(), PersistError> {
        match cv {
            Cv::Folds {
                n_splits,
                stratified,
            } => {
                w.scalar_str("param:cv", "folds");
                w.scalar_usize("param:cv_n_splits", *n_splits);
                w.scalar_bool("param:cv_stratified", *stratified);
            }
            Cv::Explicit(_) => {
                w.scalar_str("param:cv", "explicit");
                for (name, values) in [
                    (CV_TRAIN_INDPTR, &self.train_indptr),
                    (CV_TRAIN_INDICES, &self.train_indices),
                    (CV_TEST_INDPTR, &self.test_indptr),
                    (CV_TEST_INDICES, &self.test_indices),
                ] {
                    w.tensor(name, TensorRef::u64s(values, vec![values.len()])?);
                }
            }
        }
        Ok(())
    }
}

/// Read back what [`CvStaging::write_into`] staged.
///
/// The two `indptr` arrays must start at 0, be non-decreasing, and end at their
/// respective index array's length. Those are the same CSR invariants the
/// affinity reader checks and they matter for the same reason: a violation is an
/// out-of-range slice rather than a wrong number.
pub fn read_cv(file: &FselFile<'_>) -> Result<Cv, PersistError> {
    match file.scalar_str("param:cv")? {
        "folds" => Ok(Cv::Folds {
            n_splits: file.scalar_usize("param:cv_n_splits")?,
            stratified: file.scalar_bool("param:cv_stratified")?,
        }),
        "explicit" => {
            let read = |ptr_name: &'static str,
                        idx_name: &'static str|
             -> Result<Vec<Vec<usize>>, PersistError> {
                let ptr_v = file.tensor(ptr_name)?;
                shape_1d(&ptr_v, ptr_name)?;
                let indptr = as_usizes(&ptr_v, ptr_name)?;
                let idx_v = file.tensor(idx_name)?;
                let nnz = shape_1d(&idx_v, idx_name)?;
                let indices = as_usizes(&idx_v, idx_name)?;
                if indptr.is_empty()
                    || indptr[0] != 0
                    || *indptr.last().expect("non-empty") != nnz
                    || indptr.windows(2).any(|w| w[0] > w[1])
                {
                    return Err(PersistError::InconsistentGeometry {
                        reason: format!(
                            "'{ptr_name}' is not a valid CSR row-start array over the {nnz} \
                             entries in '{idx_name}'"
                        ),
                    });
                }
                Ok(indptr
                    .windows(2)
                    .map(|w| indices[w[0]..w[1]].to_vec())
                    .collect())
            };
            let train = read(CV_TRAIN_INDPTR, CV_TRAIN_INDICES)?;
            let test = read(CV_TEST_INDPTR, CV_TEST_INDICES)?;
            if train.len() != test.len() {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!(
                        "the explicit CV split list has {} train folds and {} test folds",
                        train.len(),
                        test.len()
                    ),
                });
            }
            Ok(Cv::Explicit(train.into_iter().zip(test).collect()))
        }
        _ => Err(PersistError::BadMetadata { key: "param:cv" }),
    }
}

/// The tensor names [`CvResults`] flattens into.
pub const CVR_N_FEATURES: &str = "cv_results_n_features";
/// See [`CVR_N_FEATURES`].
pub const CVR_MEAN: &str = "cv_results_mean_test_score";
/// See [`CVR_N_FEATURES`].
pub const CVR_STD: &str = "cv_results_std_test_score";
/// See [`CVR_N_FEATURES`].
pub const CVR_SPLIT_SCORE: &str = "cv_results_split_test_score";
/// See [`CVR_N_FEATURES`].
pub const CVR_SPLIT_RANKING: &str = "cv_results_split_ranking";
/// See [`CVR_N_FEATURES`].
pub const CVR_SPLIT_SUPPORT: &str = "cv_results_split_support";

/// The flattened form of a [`CvResults`], bound so it outlives the writer.
///
/// Unlike an explicit CV split list, `cv_results_` is RECTANGULAR — every fold
/// evaluates the same subset ladder and every subset ranks the same feature
/// count — so it flattens into rank-2 and rank-3 tensors rather than needing a
/// CSR encoding: `[n_folds, n_subsets]` for the per-split scores and
/// `[n_folds, n_subsets, n_features]` for the per-split rankings and masks.
pub struct CvResultsStaging {
    /// Feature counts per subset, widened.
    pub n_features: Vec<u64>,
    /// Per-split scores, flattened `[n_folds, n_subsets]`.
    pub split_score: Vec<f64>,
    /// Per-split rankings, flattened `[n_folds, n_subsets, n_features]`.
    pub split_ranking: Vec<u64>,
    /// Per-split masks, packed one byte per flag.
    pub split_support: Vec<u8>,
    /// Fold count.
    pub n_folds: usize,
    /// Subset count.
    pub n_subsets: usize,
    /// Feature count.
    pub n_features_in: usize,
}

impl CvResultsStaging {
    /// Flatten `r`, ahead of the writer.
    pub fn prepare(r: &CvResults, n_features_in: usize) -> Self {
        CvResultsStaging {
            n_features: r.n_features.iter().map(|&v| v as u64).collect(),
            split_score: r.split_test_score.iter().flatten().copied().collect(),
            split_ranking: r
                .split_ranking
                .iter()
                .flatten()
                .flatten()
                .map(|&v| v as u64)
                .collect(),
            split_support: r
                .split_support
                .iter()
                .flatten()
                .flatten()
                .map(|&b| u8::from(b))
                .collect(),
            n_folds: r.split_test_score.len(),
            n_subsets: r.n_features.len(),
            n_features_in,
        }
    }

    /// Stage the flattened results.
    pub fn write_into<'a>(
        &'a self,
        w: &mut FselWriter<'a>,
        r: &'a CvResults,
    ) -> Result<(), PersistError> {
        let (f, s, d) = (self.n_folds, self.n_subsets, self.n_features_in);
        w.scalar_usize("cv_results_n_folds", f);
        w.tensor(CVR_N_FEATURES, TensorRef::u64s(&self.n_features, vec![s])?);
        w.tensor(CVR_MEAN, TensorRef::f64s(&r.mean_test_score, vec![s])?);
        w.tensor(CVR_STD, TensorRef::f64s(&r.std_test_score, vec![s])?);
        w.tensor(
            CVR_SPLIT_SCORE,
            TensorRef::f64s(&self.split_score, vec![f, s])?,
        );
        w.tensor(
            CVR_SPLIT_RANKING,
            TensorRef::u64s(&self.split_ranking, vec![f, s, d])?,
        );
        w.tensor(
            CVR_SPLIT_SUPPORT,
            TensorRef::bools(&self.split_support, vec![f, s, d])?,
        );
        Ok(())
    }
}

/// Read a [`CvResults`] back, validating every extent against the `n_features`
/// the support mask established.
pub fn read_cv_results(
    file: &FselFile<'_>,
    n_features_in: usize,
) -> Result<CvResults, PersistError> {
    let n_folds = file.scalar_usize("cv_results_n_folds")?;
    let nf_v = file.tensor(CVR_N_FEATURES)?;
    let n_subsets = shape_1d(&nf_v, CVR_N_FEATURES)?;
    let n_features = as_usizes(&nf_v, CVR_N_FEATURES)?;

    let mean_test_score = read_f64_vec(file, CVR_MEAN, n_subsets)?;
    let std_test_score = read_f64_vec(file, CVR_STD, n_subsets)?;

    let score_v = file.tensor(CVR_SPLIT_SCORE)?;
    let score_len: usize = score_v.shape().iter().product();
    expect_len(CVR_SPLIT_SCORE, score_len, n_folds * n_subsets, "entries")?;
    let flat_score = as_f64(&score_v, CVR_SPLIT_SCORE)?;

    let rank_v = file.tensor(CVR_SPLIT_RANKING)?;
    let rank_len: usize = rank_v.shape().iter().product();
    expect_len(
        CVR_SPLIT_RANKING,
        rank_len,
        n_folds * n_subsets * n_features_in,
        "entries",
    )?;
    let flat_rank = as_usizes(&rank_v, CVR_SPLIT_RANKING)?;

    let sup_v = file.tensor(CVR_SPLIT_SUPPORT)?;
    let sup_len: usize = sup_v.shape().iter().product();
    expect_len(
        CVR_SPLIT_SUPPORT,
        sup_len,
        n_folds * n_subsets * n_features_in,
        "entries",
    )?;
    let flat_sup = as_bools(&sup_v, CVR_SPLIT_SUPPORT)?;

    // The chunk widths are guarded with `.max(1)` because `chunks(0)` panics,
    // and a zero extent is reachable from a hand-edited header even though the
    // length checks above make the resulting vectors empty.
    let per_fold = n_subsets * n_features_in;
    Ok(CvResults {
        n_features,
        mean_test_score,
        std_test_score,
        split_test_score: flat_score
            .chunks(n_subsets.max(1))
            .map(<[f64]>::to_vec)
            .collect(),
        split_ranking: flat_rank
            .chunks(per_fold.max(1))
            .map(|fold| {
                fold.chunks(n_features_in.max(1))
                    .map(<[usize]>::to_vec)
                    .collect()
            })
            .collect(),
        split_support: flat_sup
            .chunks(per_fold.max(1))
            .map(|fold| {
                fold.chunks(n_features_in.max(1))
                    .map(<[bool]>::to_vec)
                    .collect()
            })
            .collect(),
    })
}
