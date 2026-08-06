//! `ParameterGrid` / `ParameterSampler` combinatorics (MODSEL-RS-04).
//!
//! Both classes are, underneath, pure index arithmetic over "how many values
//! does each parameter have" — the *values themselves* are arbitrary Python
//! objects (estimators, callables, scipy distributions) that Rust has no
//! business holding. So this module works on a [`GridSpec`]: the sorted
//! parameter names plus each one's value count. It hands back **value index
//! tuples**, and the caller looks them up in its own value lists.
//!
//! ## Why the key order is not an implementation detail
//!
//! sklearn sorts each grid's keys and iterates `itertools.product` over the
//! sorted values, so the **last key alphabetically varies fastest**. A search
//! that enumerated candidates in a different order would still visit the same
//! *set* of candidates, but `cv_results_` row `i` would hold different
//! parameters — and `RandomizedSearchCV`'s `sample_without_replacement` draws
//! *indices into this enumeration*, so a different order silently samples a
//! different subset from the same seed.

use super::rng::{sample_without_replacement, NumpyRandomState, SampleMethod};
use super::{param_err, Result};

/// One sub-grid: parameter names with the number of values each one offers.
///
/// Construct through [`GridSpec::new`], which sorts the keys — every downstream
/// index is relative to that sorted order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridSpec {
    keys: Vec<String>,
    value_counts: Vec<usize>,
}

impl GridSpec {
    /// Build a sub-grid from `(name, n_values)` pairs, sorting by name.
    ///
    /// Rejects an empty value list with sklearn's own message — a grid entry
    /// with nothing in it makes the whole product empty, which would read as a
    /// silent "searched zero candidates" rather than an error.
    pub fn new<I, S>(entries: I) -> Result<Self>
    where
        I: IntoIterator<Item = (S, usize)>,
        S: Into<String>,
    {
        let mut pairs: Vec<(String, usize)> =
            entries.into_iter().map(|(k, v)| (k.into(), v)).collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        for (key, count) in &pairs {
            if *count == 0 {
                return Err(param_err!(
                    "Parameter grid for parameter {key:?} need to be a \
                     non-empty sequence, got: []"
                ));
            }
        }
        let (keys, value_counts) = pairs.into_iter().unzip();
        Ok(Self { keys, value_counts })
    }

    /// Build a sub-grid from value counts that are ALREADY in sorted-key
    /// order, without knowing the key names.
    ///
    /// This is the PyO3 boundary's constructor: the shim sorts each grid's
    /// keys with Python's own `sorted` (which is what sklearn uses, and which
    /// orders Python strings by code point rather than by Rust's `str` `Ord` —
    /// the two agree for ASCII identifiers but not in general). Re-sorting
    /// here with [`new`](Self::new) could therefore reorder the counts away
    /// from the values the caller is holding, so this constructor deliberately
    /// preserves the given order and synthesizes placeholder names.
    pub fn from_sorted_counts(value_counts: Vec<usize>) -> Result<Self> {
        for (i, count) in value_counts.iter().enumerate() {
            if *count == 0 {
                return Err(param_err!(
                    "Parameter grid for parameter at position {i} need to be a \
                     non-empty sequence, got: []"
                ));
            }
        }
        let keys = (0..value_counts.len()).map(|i| i.to_string()).collect();
        Ok(Self { keys, value_counts })
    }

    /// The sorted parameter names.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Per-key value counts, in [`keys`](Self::keys) order.
    pub fn value_counts(&self) -> &[usize] {
        &self.value_counts
    }

    /// Number of candidates this sub-grid contributes.
    ///
    /// An EMPTY sub-grid (no keys at all) contributes exactly ONE candidate —
    /// the empty parameter dict — which is sklearn's documented behavior and
    /// the reason this is not simply `product(counts)`.
    pub fn len(&self) -> usize {
        if self.keys.is_empty() {
            1
        } else {
            self.value_counts.iter().product()
        }
    }

    /// Always `false`: even a keyless sub-grid yields one (empty) candidate.
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// `sklearn.model_selection.ParameterGrid` — the cartesian product of one or
/// more sub-grids, enumerated in sklearn's exact order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterGrid {
    grids: Vec<GridSpec>,
}

/// One enumerated candidate: which sub-grid it came from, and the value index
/// chosen for each of that sub-grid's (sorted) keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub grid: usize,
    pub value_indices: Vec<usize>,
}

impl ParameterGrid {
    pub fn new(grids: Vec<GridSpec>) -> Self {
        Self { grids }
    }

    /// Total number of candidates across every sub-grid.
    pub fn len(&self) -> usize {
        self.grids.iter().map(GridSpec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn grids(&self) -> &[GridSpec] {
        &self.grids
    }

    /// The `ind`-th candidate, or `None` past the end
    /// (`ParameterGrid.__getitem__`, which raises `IndexError` there).
    ///
    /// Within a sub-grid the LAST key varies fastest, matching
    /// `itertools.product` over the ascending-sorted keys.
    pub fn nth(&self, ind: usize) -> Option<Candidate> {
        let mut remaining = ind;
        for (grid_idx, grid) in self.grids.iter().enumerate() {
            let total = grid.len();
            if remaining >= total {
                remaining -= total;
                continue;
            }
            let mut value_indices = vec![0usize; grid.keys.len()];
            for (k, &size) in grid.value_counts.iter().enumerate().rev() {
                value_indices[k] = remaining % size;
                remaining /= size;
            }
            return Some(Candidate {
                grid: grid_idx,
                value_indices,
            });
        }
        None
    }

    /// Every candidate, in order.
    pub fn iter(&self) -> impl Iterator<Item = Candidate> + '_ {
        (0..self.len()).map(move |i| self.nth(i).expect("i < len"))
    }
}

/// What [`sample_parameter_grid`] decided to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampledGrid {
    /// Candidate indices into the [`ParameterGrid`] enumeration.
    pub indices: Vec<i64>,
    /// sklearn's "grid is smaller than n_iter" `UserWarning`, if it applies.
    pub warning: Option<String>,
}

/// `ParameterSampler`'s **all-lists** path: draw `n_iter` distinct candidates
/// from a finite grid without replacement.
///
/// sklearn caps `n_iter` at the grid size (with a `UserWarning`) rather than
/// erroring, and draws through `sample_without_replacement`, whose method
/// dispatch depends on the sampling *ratio* — see
/// [`sample_without_replacement`](super::rng::sample_without_replacement).
/// Reproducing that dispatch is what makes the drawn subset match sklearn's for
/// the same seed.
///
/// The mixed path (any value being a scipy distribution) is deliberately NOT
/// here: it interleaves `rvs(random_state=rng)` calls only Python can make, so
/// the Python layer drives that loop and reaches into
/// [`NumpyRandomState`](super::rng::NumpyRandomState) for the index draws
/// between them.
pub fn sample_parameter_grid(
    grid: &ParameterGrid,
    n_iter: usize,
    rng: &mut NumpyRandomState,
) -> SampledGrid {
    let grid_size = grid.len();
    let (n_iter, warning) = if grid_size < n_iter {
        (
            grid_size,
            Some(format!(
                "The total space of parameters {grid_size} is smaller than \
                 n_iter={n_iter}. Running {grid_size} iterations. For \
                 exhaustive searches, use GridSearchCV."
            )),
        )
    } else {
        (n_iter, None)
    };
    let indices = sample_without_replacement(grid_size, n_iter, SampleMethod::Auto, rng)
        .expect("n_iter was clamped to grid_size above");
    SampledGrid { indices, warning }
}
