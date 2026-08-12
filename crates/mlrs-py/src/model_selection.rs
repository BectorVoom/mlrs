//! PyO3 surface for `mlrs.model_selection` (MODSEL-BIND-01).
//!
//! Thin wrappers over `mlrs_algos::model_selection`. Like [`crate::metrics`]
//! this is a free-function surface over plain `Vec<i64>`/`Vec<f64>`, not the
//! Arrow capsule ingress: everything here is host-side index and schedule
//! bookkeeping over *integer* label codes and *float* scores, and the capsule
//! path is float-only (`crates/mlrs-py/src/ingress.rs`).
//!
//! ## Three conventions the whole file follows
//!
//! **1. Labels arrive as codes.** `y` and `groups` cross as
//! `np.unique(..., return_inverse=True)[1]` — sorted-unique factorization
//! codes. The shim does the factorization with numpy, which already handles
//! strings, objects, NaN ordering and multi-label rows; Rust never sees a
//! Python label object.
//!
//! **2. The RNG is a round-tripped state, not a seed.** sklearn passes a LIVE
//! `numpy.random.RandomState` through its splitters, and callers observe the
//! advancement (`RepeatedKFold` continues one stream across repeats;
//! `ParameterSampler` interleaves `scipy` `rvs` draws mlrs cannot host). So
//! [`NumpyRandomState`] is a mutable `#[pyclass]` the shim builds from
//! `rs.get_state()` and writes back with `rs.set_state()`. Passing a bare seed
//! would break both cases silently.
//!
//! **3. Parameter validation stays in Python.** sklearn raises
//! `InvalidParameterError` — a class that subclasses BOTH `ValueError` and
//! `TypeError` — for a malformed parameter, and a plain `ValueError` for a
//! parameter that is wrong for the data. A Rust `PyErr` cannot cheaply carry
//! that dual base, and the shim already owns the constraint messages, so
//! `ModelSelectionError` maps to `PyValueError` here and the shim raises the
//! `InvalidParameterError` cases before calling in. See
//! `crates/mlrs-py/python/mlrs/model_selection.py`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mlrs_algos::model_selection as ms;
use mlrs_algos::model_selection::param::{sample_parameter_grid, GridSpec, ParameterGrid};
use mlrs_algos::model_selection::rng::NumpyRandomState as CoreRng;
use mlrs_algos::model_selection::search as msearch;
use mlrs_algos::model_selection::split as msplit;
use mlrs_algos::model_selection::threshold as mthr;
use mlrs_algos::model_selection::validate as mval;
use mlrs_algos::model_selection::{ModelSelectionError, RandomStateSpec, SizeSpec, Splits};

/// Map a core error onto `ValueError` (see convention 3 in the module docs).
fn ms_err_to_py(err: ModelSelectionError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// The wire form of a splitter result: `(train_index_lists, test_index_lists,
/// warning_messages)`.
///
/// The warnings ride along rather than being emitted here because a Rust-side
/// `log::warn!` is invisible to `pytest.warns` and to a user's
/// `warnings.simplefilter("error")` — the shim re-raises them through
/// `warnings.warn` so sklearn-compatible warning behavior is preserved.
type SplitsOut = (Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<String>);

fn unpack(splits: Splits) -> SplitsOut {
    let mut trains = Vec::with_capacity(splits.splits.len());
    let mut tests = Vec::with_capacity(splits.splits.len());
    for s in splits.splits {
        trains.push(s.train);
        tests.push(s.test);
    }
    (trains, tests, splits.warnings)
}

/// Rebuild a [`SizeSpec`] from the shim's "exactly one of these is set" pair.
///
/// The int/float distinction is preserved across the boundary because sklearn
/// `ceil`s a float `test_size` and `floor`s a float `train_size` — collapsing
/// both to a float here would change the row counts for an integer argument.
fn size_spec(int_value: Option<usize>, float_value: Option<f64>) -> SizeSpec {
    match (int_value, float_value) {
        (Some(v), _) => SizeSpec::Int(v),
        (None, Some(v)) => SizeSpec::Float(v),
        (None, None) => SizeSpec::None,
    }
}

// =========================================================================
// the round-tripped generator
// =========================================================================

/// numpy's legacy `RandomState` (MT19937), reimplemented in Rust.
///
/// Built from `rs.get_state()[1:3]` and read back with
/// [`get_state`](NumpyRandomState::get_state) so the caller's own
/// `RandomState` can be advanced to exactly where sklearn would have left it.
// NOT `Clone`: this object is a *mutable cursor* into an MT19937 stream, and a
// silent Python-side copy would let two callers draw the same values while each
// believed it had advanced the shared generator.
#[pyclass(name = "NumpyRandomState", module = "mlrs._mlrs")]
pub struct PyNumpyRandomState {
    /// `pub(crate)` so a Rust-side driver that borrows the generator for a whole
    /// fit — `estimators::ransac`, whose trial loop draws once per iteration —
    /// can snapshot it out and write the advanced words back, rather than
    /// holding a `RefCell` borrow across a callback into Python.
    pub(crate) inner: CoreRng,
}

#[pymethods]
impl PyNumpyRandomState {
    /// Rebuild from raw `get_state()` words: the 624-element key and position.
    #[new]
    fn new(key: Vec<u32>, pos: usize) -> PyResult<Self> {
        let key: [u32; ms::rng::MT_N] = key.try_into().map_err(|v: Vec<u32>| {
            PyValueError::new_err(format!(
                "MT19937 state key must have {} words, got {}",
                ms::rng::MT_N,
                v.len()
            ))
        })?;
        Ok(Self {
            inner: CoreRng::from_key(key, pos),
        })
    }

    /// Seed as `numpy.random.RandomState(seed)` does.
    #[staticmethod]
    fn from_seed(seed: u32) -> Self {
        Self {
            inner: CoreRng::from_seed(seed),
        }
    }

    /// The advanced `(key, pos)`, for `rs.set_state(("MT19937", key, pos, 0, 0.0))`.
    fn get_state(&self) -> (Vec<u32>, usize) {
        (self.inner.key().to_vec(), self.inner.pos())
    }

    /// `rng.randint(n)` — used by the shim between `scipy` `rvs` draws.
    fn randint(&mut self, n: u64) -> PyResult<u64> {
        if n == 0 {
            return Err(PyValueError::new_err("randint: empty range"));
        }
        Ok(self.inner.randint(n))
    }

    /// `rng.permutation(n)`.
    fn permutation(&mut self, n: usize) -> Vec<i64> {
        self.inner.permutation(n)
    }

    /// `rng.random_sample()`.
    fn random_sample(&mut self) -> f64 {
        self.inner.random_sample()
    }
}

/// Borrow the generator out of an optional handle, or make a throwaway one.
///
/// A `None` handle means the splitter does not shuffle, so nothing will be
/// drawn — the throwaway generator exists only to satisfy the signature and is
/// never observed.
fn with_rng<T>(
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
    f: impl FnOnce(&mut CoreRng) -> T,
) -> PyResult<T> {
    match rng {
        Some(handle) => {
            let mut guard = handle.borrow_mut();
            Ok(f(&mut guard.inner))
        }
        None => {
            let mut throwaway = CoreRng::from_seed(0);
            Ok(f(&mut throwaway))
        }
    }
}

// =========================================================================
// splitters
// =========================================================================

#[pyfunction]
#[pyo3(signature = (n_samples, n_splits, shuffle, rng=None))]
pub fn kfold_split(
    n_samples: usize,
    n_splits: usize,
    shuffle: bool,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::KFold {
        n_splits,
        shuffle,
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(n_samples, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (groups, n_splits, shuffle, rng=None))]
pub fn group_kfold_split(
    groups: Vec<i64>,
    n_splits: usize,
    shuffle: bool,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::GroupKFold {
        n_splits,
        shuffle,
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(&groups, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y, n_splits, shuffle, rng=None))]
pub fn stratified_kfold_split(
    y: Vec<i64>,
    n_splits: usize,
    shuffle: bool,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::StratifiedKFold {
        n_splits,
        shuffle,
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(&y, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y, groups, n_splits, shuffle, rng=None))]
pub fn stratified_group_kfold_split(
    y: Vec<i64>,
    groups: Vec<i64>,
    n_splits: usize,
    shuffle: bool,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::StratifiedGroupKFold {
        n_splits,
        shuffle,
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(&y, &groups, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (n_samples, n_splits, max_train_size=None, test_size=None, gap=0))]
pub fn time_series_split(
    n_samples: usize,
    n_splits: usize,
    max_train_size: Option<usize>,
    test_size: Option<usize>,
    gap: usize,
) -> PyResult<SplitsOut> {
    msplit::TimeSeriesSplit {
        n_splits,
        max_train_size,
        test_size,
        gap,
    }
    .split(n_samples)
    .map(unpack)
    .map_err(ms_err_to_py)
}

/// One `LeaveOneOut` split, by index.
///
/// Streamed rather than materialized: `LeaveOneOut` on a million rows would
/// otherwise build a million train vectors of a million entries each.
#[pyfunction]
pub fn leave_one_out_split_at(n_samples: usize, i: usize) -> PyResult<(Vec<i64>, Vec<i64>)> {
    msplit::LeaveOneOut
        .split_at(n_samples, i)
        .map(|s| (s.train, s.test))
        .map_err(ms_err_to_py)
}

#[pyfunction]
pub fn leave_p_out_n_splits(n_samples: usize, p: usize) -> PyResult<u128> {
    msplit::LeavePOut::new(p)
        .get_n_splits(n_samples)
        .map_err(ms_err_to_py)
}

/// The `i`-th `LeavePOut` split, in `itertools.combinations` order.
#[pyfunction]
pub fn leave_p_out_split_at(n_samples: usize, p: usize, i: u128) -> PyResult<(Vec<i64>, Vec<i64>)> {
    msplit::LeavePOut::new(p)
        .split_at(n_samples, i)
        .map(|s| (s.train, s.test))
        .map_err(ms_err_to_py)
}

#[pyfunction]
pub fn leave_one_group_out_split(groups: Vec<i64>) -> PyResult<SplitsOut> {
    msplit::LeaveOneGroupOut
        .split(&groups)
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
pub fn leave_p_groups_out_n_splits(groups: Vec<i64>, n_groups: usize) -> PyResult<u128> {
    msplit::LeavePGroupsOut::new(n_groups)
        .get_n_splits(&groups)
        .map_err(ms_err_to_py)
}

#[pyfunction]
pub fn leave_p_groups_out_split_at(
    groups: Vec<i64>,
    n_groups: usize,
    i: u128,
) -> PyResult<(Vec<i64>, Vec<i64>)> {
    msplit::LeavePGroupsOut::new(n_groups)
        .split_at(&groups, i)
        .map(|s| (s.train, s.test))
        .map_err(ms_err_to_py)
}

#[pyfunction]
pub fn predefined_split(test_fold: Vec<i64>) -> PyResult<SplitsOut> {
    msplit::PredefinedSplit::new(test_fold)
        .split()
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (n_samples, n_splits, test_size_int=None, test_size_float=None,
                    train_size_int=None, train_size_float=None, rng=None))]
pub fn shuffle_split(
    n_samples: usize,
    n_splits: usize,
    test_size_int: Option<usize>,
    test_size_float: Option<f64>,
    train_size_int: Option<usize>,
    train_size_float: Option<f64>,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::ShuffleSplit {
        n_splits,
        test_size: size_spec(test_size_int, test_size_float),
        train_size: size_spec(train_size_int, train_size_float),
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(n_samples, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (groups, n_splits, test_size_int=None, test_size_float=None,
                    train_size_int=None, train_size_float=None, rng=None))]
pub fn group_shuffle_split(
    groups: Vec<i64>,
    n_splits: usize,
    test_size_int: Option<usize>,
    test_size_float: Option<f64>,
    train_size_int: Option<usize>,
    train_size_float: Option<f64>,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::GroupShuffleSplit {
        n_splits,
        test_size: size_spec(test_size_int, test_size_float),
        train_size: size_spec(train_size_int, train_size_float),
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(&groups, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y, n_splits, test_size_int=None, test_size_float=None,
                    train_size_int=None, train_size_float=None, rng=None))]
pub fn stratified_shuffle_split(
    y: Vec<i64>,
    n_splits: usize,
    test_size_int: Option<usize>,
    test_size_float: Option<f64>,
    train_size_int: Option<usize>,
    train_size_float: Option<f64>,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::StratifiedShuffleSplit {
        n_splits,
        test_size: size_spec(test_size_int, test_size_float),
        train_size: size_spec(train_size_int, train_size_float),
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(&y, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (n_samples, n_splits, n_repeats, rng=None))]
pub fn repeated_kfold_split(
    n_samples: usize,
    n_splits: usize,
    n_repeats: usize,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::RepeatedKFold {
        n_splits,
        n_repeats,
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(n_samples, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (y, n_splits, n_repeats, rng=None))]
pub fn repeated_stratified_kfold_split(
    y: Vec<i64>,
    n_splits: usize,
    n_repeats: usize,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<SplitsOut> {
    let splitter = msplit::RepeatedStratifiedKFold {
        n_splits,
        n_repeats,
        random_state: RandomStateSpec::Entropy,
    };
    with_rng(rng, |r| splitter.split_with_rng(&y, r))?
        .map(unpack)
        .map_err(ms_err_to_py)
}

#[pyfunction]
#[pyo3(signature = (n_samples, test_size_int=None, test_size_float=None,
                    train_size_int=None, train_size_float=None, shuffle=true,
                    stratify=None, rng=None))]
#[allow(clippy::too_many_arguments)]
pub fn train_test_split_indices(
    n_samples: usize,
    test_size_int: Option<usize>,
    test_size_float: Option<f64>,
    train_size_int: Option<usize>,
    train_size_float: Option<f64>,
    shuffle: bool,
    stratify: Option<Vec<i64>>,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let test_size = size_spec(test_size_int, test_size_float);
    let train_size = size_spec(train_size_int, train_size_float);
    with_rng(rng, |r| {
        msplit::train_test_split_indices(
            n_samples,
            test_size,
            train_size,
            shuffle,
            stratify.as_deref(),
            r,
        )
    })?
    .map(|s| (s.train, s.test))
    .map_err(ms_err_to_py)
}

/// `_validate_shuffle_split` — resolve `(n_train, n_test)` without splitting.
#[pyfunction]
#[pyo3(signature = (n_samples, test_size_int=None, test_size_float=None,
                    train_size_int=None, train_size_float=None,
                    default_test_size=0.25))]
pub fn validate_shuffle_split(
    n_samples: usize,
    test_size_int: Option<usize>,
    test_size_float: Option<f64>,
    train_size_int: Option<usize>,
    train_size_float: Option<f64>,
    default_test_size: f64,
) -> PyResult<(usize, usize)> {
    ms::validate_shuffle_split(
        n_samples,
        size_spec(test_size_int, test_size_float),
        size_spec(train_size_int, train_size_float),
        default_test_size,
    )
    .map_err(ms_err_to_py)
}

// =========================================================================
// ParameterGrid / ParameterSampler
// =========================================================================

/// Build the core grid from per-sub-grid value counts.
///
/// The counts arrive ALREADY in sorted-key order (the shim sorts with
/// Python's own `sorted`, which is what sklearn uses), so this must not
/// re-sort — hence [`GridSpec::from_sorted_counts`] rather than
/// `GridSpec::new`.
fn build_grid(value_counts: Vec<Vec<usize>>) -> PyResult<ParameterGrid> {
    let grids = value_counts
        .into_iter()
        .map(GridSpec::from_sorted_counts)
        .collect::<Result<Vec<_>, _>>()
        .map_err(ms_err_to_py)?;
    Ok(ParameterGrid::new(grids))
}

#[pyfunction]
pub fn parameter_grid_size(value_counts: Vec<Vec<usize>>) -> PyResult<usize> {
    Ok(build_grid(value_counts)?.len())
}

/// The `ind`-th candidate as `(sub_grid_index, value_index_per_key)`, or
/// `None` past the end.
#[pyfunction]
pub fn parameter_grid_nth(
    value_counts: Vec<Vec<usize>>,
    ind: usize,
) -> PyResult<Option<(usize, Vec<usize>)>> {
    Ok(build_grid(value_counts)?
        .nth(ind)
        .map(|c| (c.grid, c.value_indices)))
}

/// `ParameterSampler`'s all-lists draw: `(candidate_indices, warning)`.
#[pyfunction]
#[pyo3(signature = (value_counts, n_iter, rng=None))]
pub fn sample_parameter_grid_indices(
    value_counts: Vec<Vec<usize>>,
    n_iter: usize,
    rng: Option<&Bound<'_, PyNumpyRandomState>>,
) -> PyResult<(Vec<i64>, Option<String>)> {
    let grid = build_grid(value_counts)?;
    let sampled = with_rng(rng, |r| sample_parameter_grid(&grid, n_iter, r))?;
    Ok((sampled.indices, sampled.warning))
}

// =========================================================================
// search / aggregation
// =========================================================================

/// `(mean_test_score, std_test_score, rank_test_score)` from a row-major
/// `(n_candidates, n_splits)` score matrix.
#[pyfunction]
pub fn summarize_scores(
    scores: Vec<f64>,
    n_candidates: usize,
    n_splits: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<i32>)> {
    mval::summarize_scores(&scores, n_candidates, n_splits)
        .map(|s| (s.mean, s.std, s.rank))
        .map_err(ms_err_to_py)
}

#[pyfunction]
pub fn rank_scores(mean_scores: Vec<f64>) -> Vec<i32> {
    mval::rank_scores(&mean_scores)
}

#[pyfunction]
pub fn best_index(mean_scores: Vec<f64>) -> usize {
    msearch::best_index(&mean_scores)
}

#[pyfunction]
pub fn top_k(mean_scores: Vec<f64>, k: usize) -> Vec<usize> {
    msearch::top_k(&mean_scores, k)
}

/// The successive-halving schedule:
/// `(min_resources, n_required_iterations, n_possible_iterations,
///   per_iteration_n_resources, per_iteration_n_candidates)`.
///
/// `min_resources_kind` is `"smallest"`, `"exhaust"` or `"fixed"`;
/// `min_resources_value` is only read for `"fixed"`.
#[pyfunction]
#[pyo3(signature = (n_candidates, factor, min_resources_kind, min_resources_value,
                    max_resources, aggressive_elimination, smallest_resources))]
pub fn halving_schedule(
    n_candidates: usize,
    factor: usize,
    min_resources_kind: &str,
    min_resources_value: usize,
    max_resources: usize,
    aggressive_elimination: bool,
    smallest_resources: usize,
) -> PyResult<(usize, usize, usize, Vec<usize>, Vec<usize>)> {
    let min_resources = match min_resources_kind {
        "smallest" => msearch::MinResources::Smallest,
        "exhaust" => msearch::MinResources::Exhaust,
        "fixed" => msearch::MinResources::Fixed(min_resources_value),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown min_resources kind '{other}'"
            )))
        }
    };
    let schedule = msearch::halving_schedule(
        msearch::HalvingParams {
            factor,
            min_resources,
            max_resources,
            aggressive_elimination,
            smallest_resources,
        },
        n_candidates,
    )
    .map_err(ms_err_to_py)?;
    Ok((
        schedule.min_resources,
        schedule.n_required_iterations,
        schedule.n_possible_iterations,
        schedule.iterations.iter().map(|i| i.n_resources).collect(),
        schedule.iterations.iter().map(|i| i.n_candidates).collect(),
    ))
}

#[pyfunction]
pub fn exhaust_n_candidates(max_resources: usize, min_resources: usize) -> usize {
    msearch::exhaust_n_candidates(max_resources, min_resources)
}

// =========================================================================
// curves / permutation test / partitions
// =========================================================================

/// `_translate_train_sizes`. Exactly one of `fractions` / `absolute` is set.
#[pyfunction]
#[pyo3(signature = (n_max_training_samples, fractions=None, absolute=None))]
pub fn translate_train_sizes(
    n_max_training_samples: usize,
    fractions: Option<Vec<f64>>,
    absolute: Option<Vec<usize>>,
) -> PyResult<(Vec<usize>, Option<String>)> {
    let sizes = match (fractions, absolute) {
        (Some(f), None) => mval::TrainSizes::Fractions(f),
        (None, Some(a)) => mval::TrainSizes::Absolute(a),
        _ => {
            return Err(PyValueError::new_err(
                "translate_train_sizes: pass exactly one of fractions/absolute",
            ))
        }
    };
    mval::translate_train_sizes(&sizes, n_max_training_samples).map_err(ms_err_to_py)
}

#[pyfunction]
pub fn permutation_pvalue(score: f64, permutation_scores: Vec<f64>) -> f64 {
    mval::permutation_pvalue(score, &permutation_scores)
}

/// `cross_val_predict`'s scatter map: for each row, its position in the
/// fold-order concatenation of predictions. Errors if the splits are not a
/// partition.
#[pyfunction]
pub fn partition_inverse(test_sets: Vec<Vec<i64>>, n_samples: usize) -> PyResult<Vec<usize>> {
    mval::partition_inverse(&test_sets, n_samples).map_err(ms_err_to_py)
}

// =========================================================================
// decision thresholds
// =========================================================================

#[pyfunction]
pub fn apply_threshold(scores: Vec<f64>, threshold: f64) -> Vec<i64> {
    mthr::apply_threshold(&scores, threshold)
}

/// `TunedThresholdClassifierCV`'s reduction:
/// `(best_threshold, best_score, thresholds, scores)`.
///
/// `fold_thresholds` / `fold_scores` are one entry per CV fold. Exactly one of
/// `grid_count` / `grid_explicit` is set.
#[pyfunction]
#[pyo3(signature = (fold_thresholds, fold_scores, grid_count=None, grid_explicit=None))]
pub fn tune_threshold(
    fold_thresholds: Vec<Vec<f64>>,
    fold_scores: Vec<Vec<f64>>,
    grid_count: Option<usize>,
    grid_explicit: Option<Vec<f64>>,
) -> PyResult<(f64, f64, Vec<f64>, Vec<f64>)> {
    if fold_thresholds.len() != fold_scores.len() {
        return Err(PyValueError::new_err(
            "tune_threshold: fold_thresholds and fold_scores must have the same length",
        ));
    }
    let folds: Vec<mthr::FoldCurve> = fold_thresholds
        .into_iter()
        .zip(fold_scores)
        .map(|(thresholds, scores)| mthr::FoldCurve { thresholds, scores })
        .collect();
    let grid = match (grid_count, grid_explicit) {
        (Some(n), None) => mthr::ThresholdGrid::Count(n),
        (None, Some(values)) => mthr::ThresholdGrid::Explicit(values),
        _ => {
            return Err(PyValueError::new_err(
                "tune_threshold: pass exactly one of grid_count/grid_explicit",
            ))
        }
    };
    mthr::tune_threshold(&folds, &grid)
        .map(|t| (t.best_threshold, t.best_score, t.thresholds, t.scores))
        .map_err(ms_err_to_py)
}

/// Register every `model_selection` binding on the `_mlrs` module.
///
/// One call site in `lib.rs` keeps the 30-odd `add_function` lines out of the
/// module initializer, which every estimator family also appends to.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNumpyRandomState>()?;

    macro_rules! add {
        ($($f:ident),* $(,)?) => {
            $( m.add_function(wrap_pyfunction!($f, m)?)?; )*
        };
    }
    add!(
        kfold_split,
        group_kfold_split,
        stratified_kfold_split,
        stratified_group_kfold_split,
        time_series_split,
        leave_one_out_split_at,
        leave_p_out_n_splits,
        leave_p_out_split_at,
        leave_one_group_out_split,
        leave_p_groups_out_n_splits,
        leave_p_groups_out_split_at,
        predefined_split,
        shuffle_split,
        group_shuffle_split,
        stratified_shuffle_split,
        repeated_kfold_split,
        repeated_stratified_kfold_split,
        train_test_split_indices,
        validate_shuffle_split,
        parameter_grid_size,
        parameter_grid_nth,
        sample_parameter_grid_indices,
        summarize_scores,
        rank_scores,
        best_index,
        top_k,
        halving_schedule,
        exhaust_n_candidates,
        translate_train_sizes,
        permutation_pvalue,
        partition_inverse,
        apply_threshold,
        tune_threshold,
    );
    Ok(())
}
