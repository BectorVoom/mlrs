//! `mlrs::model_selection` — the sklearn `model_selection` surface in Rust
//! (MODSEL-RS-01..08).
//!
//! Everything in `sklearn.model_selection` that is *not* a call into a user's
//! estimator lives here: every cross-validation splitter's index generation,
//! the `ParameterGrid`/`ParameterSampler` combinatorics, the successive-halving
//! resource schedule, the cross-validation result aggregation, the
//! learning-/validation-curve schedules, the permutation-test p-value, and the
//! decision-threshold tuning math. The one thing Rust cannot own is *calling an
//! arbitrary Python estimator's* `fit`/`predict`, so the search drivers here
//! ([`search`]) take a **closure** evaluator: the caller supplies "given these
//! parameters and this train/test index pair, produce scores", and this crate
//! owns the schedule, the bookkeeping and the ranking around it.
//!
//! ## Parity contract (host-match, not merely distributional)
//!
//! Every splitter reproduces sklearn's *rows*, index for index, for the same
//! arguments — including `shuffle=True`, which is why [`rng`] reimplements
//! numpy's legacy MT19937 rather than picking a nicer Rust RNG. The gate is a
//! live comparison against an installed scikit-learn (see
//! `crates/mlrs-py/python/tests/test_oracle_model_selection.py`) plus stored
//! `.npz` fixtures for the no-Python Rust suite
//! (`crates/mlrs-algos/tests/model_selection_*_test.rs`).
//!
//! ## Labels and groups cross the boundary as *codes*
//!
//! `y` and `groups` may be strings, objects, floats — anything `np.unique`
//! accepts. Rather than teach Rust about Python object ordering, both arrive
//! here as **sorted-unique factorization codes**: `np.unique(y,
//! return_inverse=True)[1]`, i.e. `codes[i] == k` iff `y[i]` is the `k`-th
//! distinct value *in ascending order*. That is precisely the encoding every
//! sklearn splitter derives internally, so the codes carry all the information
//! the algorithms use and none of the representation they don't.
//! [`factorize`] produces the same codes for a Rust-native caller.
//!
//! ## Module index
//!
//! - [`rng`] — numpy legacy `RandomState` (MT19937) + `sample_without_replacement`
//! - [`split`] — the 16 splitters, `train_test_split`, `check_cv`
//! - [`param`] — `ParameterGrid` / `ParameterSampler` index engines
//! - [`search`] — grid / randomized / successive-halving drivers over a closure
//! - [`validate`] — CV aggregation, curve schedules, permutation p-value
//! - [`threshold`] — decision-threshold tuning math
//! - [`container`] — the `RowContainer` row-gather trait (+ `ndarray` / `polars`)
//!
//! Tests live in `crates/mlrs-algos/tests/` (no in-source `#[cfg(test)] mod tests`).

pub mod container;
pub mod param;
pub mod rng;
pub mod search;
pub mod split;
pub mod threshold;
pub mod validate;

pub use container::RowContainer;
pub use rng::NumpyRandomState;
pub use split::{Split, Splits};

/// A `model_selection` failure, split by the **exception class sklearn
/// raises** rather than by what went wrong.
///
/// That distinction is load-bearing at the Python boundary: sklearn's
/// `@validate_params` rejects a malformed *parameter* with
/// `InvalidParameterError`, which subclasses BOTH `ValueError` and
/// `TypeError`, while a parameter that is well-formed but wrong *for this
/// data* raises a plain `ValueError`. Code migrating from sklearn may guard
/// either one, so collapsing the two would silently stop an `except
/// TypeError` caller from catching a bad `test_size`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelSelectionError {
    /// A parameter violated its declared type/range constraint — maps to
    /// `sklearn.utils._param_validation.InvalidParameterError`.
    #[error("{0}")]
    InvalidParameter(String),
    /// A well-formed parameter was wrong for this data — maps to `ValueError`.
    #[error("{0}")]
    Value(String),
}

/// Convenience alias for this module's fallible operations.
pub type Result<T> = core::result::Result<T, ModelSelectionError>;

/// Build a [`ModelSelectionError::Value`] from a format string.
macro_rules! value_err {
    ($($arg:tt)*) => {
        $crate::model_selection::ModelSelectionError::Value(format!($($arg)*))
    };
}
pub(crate) use value_err;

/// Build a [`ModelSelectionError::InvalidParameter`] from a format string.
macro_rules! param_err {
    ($($arg:tt)*) => {
        $crate::model_selection::ModelSelectionError::InvalidParameter(format!($($arg)*))
    };
}
pub(crate) use param_err;

/// `np.unique(values, return_inverse=True)` — the sorted distinct values plus
/// the per-element code into them.
///
/// This is the encoding every splitter in [`split`] expects for `y` and
/// `groups`. A Rust-native caller with `Vec<String>` labels calls this; the
/// Python layer instead passes numpy's own `np.unique` codes, which are
/// identical by construction.
pub fn factorize<T: Ord + Clone>(values: &[T]) -> (Vec<T>, Vec<i64>) {
    let mut uniques: Vec<T> = values.to_vec();
    uniques.sort();
    uniques.dedup();
    let codes = values
        .iter()
        .map(|v| {
            uniques
                .binary_search(v)
                .expect("value came from `uniques` itself") as i64
        })
        .collect();
    (uniques, codes)
}

/// How a randomized splitter's `random_state` was specified.
///
/// [`Seed`](RandomStateSpec::Seed) is the reproducible case and the only one
/// with a parity contract. [`Entropy`](RandomStateSpec::Entropy) mirrors
/// sklearn's `random_state=None`, which draws from numpy's *global* singleton
/// generator — mlrs cannot (and should not) mirror another process's global
/// state, so it seeds from the OS clock instead. Nothing is reproducible in
/// either implementation under `None`, so there is no parity to lose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomStateSpec {
    Entropy,
    Seed(u32),
}

impl RandomStateSpec {
    /// Materialize the generator this spec describes.
    pub fn resolve(&self) -> NumpyRandomState {
        match self {
            RandomStateSpec::Seed(s) => NumpyRandomState::from_seed(*s),
            RandomStateSpec::Entropy => {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos() ^ (d.as_secs() as u32))
                    .unwrap_or(0);
                NumpyRandomState::from_seed(nanos)
            }
        }
    }
}

/// A size expressed as an absolute row count, a fraction of the data, or
/// "whatever is left" — sklearn's `test_size` / `train_size` parameter shape.
///
/// The `Int` / `Float` split is not cosmetic: sklearn `ceil`s a float
/// `test_size` and `floor`s a float `train_size`, so `0.25` and `25` on 100
/// rows are only accidentally the same answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizeSpec {
    None,
    Int(usize),
    Float(f64),
}

impl SizeSpec {
    fn is_none(&self) -> bool {
        matches!(self, SizeSpec::None)
    }
}

/// sklearn's `_validate_shuffle_split(n_samples, test_size, train_size,
/// default_test_size)` — resolve the `(n_train, n_test)` row counts.
///
/// `default_test_size` applies only when NEITHER size is given (0.25 for
/// `train_test_split`, 0.1 for `ShuffleSplit`, 0.2 for `GroupShuffleSplit` and
/// `StratifiedShuffleSplit`).
pub fn validate_shuffle_split(
    n_samples: usize,
    test_size: SizeSpec,
    train_size: SizeSpec,
    default_test_size: f64,
) -> Result<(usize, usize)> {
    let test_size = if test_size.is_none() && train_size.is_none() {
        SizeSpec::Float(default_test_size)
    } else {
        test_size
    };

    if let SizeSpec::Int(v) = test_size {
        if v >= n_samples || v == 0 {
            return Err(value_err!(
                "test_size={v} should be either positive and smaller than the \
                 number of samples {n_samples} or a float in the (0, 1) range"
            ));
        }
    }
    if let SizeSpec::Float(v) = test_size {
        if v <= 0.0 || v >= 1.0 {
            return Err(value_err!(
                "test_size={v} should be either positive and smaller than the \
                 number of samples {n_samples} or a float in the (0, 1) range"
            ));
        }
    }
    if let SizeSpec::Int(v) = train_size {
        if v >= n_samples || v == 0 {
            return Err(value_err!(
                "train_size={v} should be either positive and smaller than the \
                 number of samples {n_samples} or a float in the (0, 1) range"
            ));
        }
    }
    if let SizeSpec::Float(v) = train_size {
        if v <= 0.0 || v >= 1.0 {
            return Err(value_err!(
                "train_size={v} should be either positive and smaller than the \
                 number of samples {n_samples} or a float in the (0, 1) range"
            ));
        }
    }

    if let (SizeSpec::Float(tr), SizeSpec::Float(te)) = (train_size, test_size) {
        if tr + te > 1.0 {
            return Err(value_err!(
                "The sum of test_size and train_size = {}, should be in the \
                 (0, 1) range. Reduce test_size and/or train_size.",
                tr + te
            ));
        }
    }

    // sklearn computes both as f64 first so the "sum exceeds n_samples" check
    // below sees the same rounding it will hand back.
    let mut n_test = match test_size {
        SizeSpec::Float(v) => (v * n_samples as f64).ceil(),
        SizeSpec::Int(v) => v as f64,
        SizeSpec::None => 0.0,
    };
    let mut n_train = match train_size {
        SizeSpec::Float(v) => (v * n_samples as f64).floor(),
        SizeSpec::Int(v) => v as f64,
        SizeSpec::None => 0.0,
    };

    if train_size.is_none() {
        n_train = n_samples as f64 - n_test;
    } else if test_size.is_none() {
        n_test = n_samples as f64 - n_train;
    }

    if n_train + n_test > n_samples as f64 {
        return Err(value_err!(
            "The sum of train_size and test_size = {}, should be smaller than \
             the number of samples {n_samples}. Reduce test_size and/or \
             train_size.",
            (n_train + n_test) as i64
        ));
    }

    let (n_train, n_test) = (n_train as usize, n_test as usize);
    if n_train == 0 {
        return Err(value_err!(
            "With n_samples={n_samples}, test_size={} and train_size={}, the \
             resulting train set will be empty. Adjust any of the \
             aforementioned parameters.",
            describe_size(test_size),
            describe_size(train_size)
        ));
    }
    Ok((n_train, n_test))
}

/// Render a [`SizeSpec`] the way sklearn's error messages print the original
/// argument (`None` / an int / a float).
fn describe_size(size: SizeSpec) -> String {
    match size {
        SizeSpec::None => "None".to_string(),
        SizeSpec::Int(v) => v.to_string(),
        SizeSpec::Float(v) => v.to_string(),
    }
}
