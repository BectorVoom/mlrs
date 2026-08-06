# `model_selection` — splitters, search and validation

mlrs implements the whole of `sklearn.model_selection`. The algorithms live in
Rust (`mlrs_algos::model_selection`); the Python package
(`mlrs.model_selection`) wraps them in sklearn-compatible classes.

This document covers the two things the API reference does not: **what parity
means here**, and **how to use the Rust-native surface**, which has no Python
equivalent.

---

## 1. The parity contract: same rows, not similar rows

Every splitter reproduces scikit-learn's output *index for index* for the same
arguments — including under `shuffle=True`.

```python
import numpy as np
import mlrs.model_selection as ms
import sklearn.model_selection as skm

X = np.arange(200).reshape(50, 4)
mine = list(ms.KFold(5, shuffle=True, random_state=42).split(X))
theirs = list(skm.KFold(5, shuffle=True, random_state=42).split(X))
assert all(np.array_equal(a, b) for m, t in zip(mine, theirs) for a, b in zip(m, t))
```

This is not a distributional claim ("same sizes, same class balance"). It is
exact equality, and it is what makes mlrs a drop-in for code that already has a
scikit-learn baseline: swapping the import does not move a single row between
train and test.

### How

`random_state` is resolved through `sklearn.utils.check_random_state` into a
legacy `numpy.random.RandomState` — the MT19937 generator, not the modern
`Generator`. Its 624-word state is handed to Rust, advanced there by a
bit-exact reimplementation of numpy's `shuffle` / `permutation` / `randint`,
and **written back** into the caller's object.

Three consequences worth knowing:

| you pass | what happens |
|---|---|
| `random_state=7` | identical split to sklearn's, every time |
| `random_state=rs` (a live `RandomState`) | `rs` comes back advanced exactly as sklearn would have left it |
| `random_state=None` | draws from — and advances — numpy's global singleton, like sklearn. Nothing is reproducible on either side. |

The middle row is why `RepeatedKFold` (one generator shared across repeats) and
`ParameterSampler` (which interleaves `scipy.stats` `rvs` draws that only Python
can make) stay in step with sklearn rather than merely producing valid-looking
output.

### Index order is part of the contract

scikit-learn's splitters disagree with each other about the order of the indices
they return, and mlrs reproduces the disagreement:

| family | order | members |
|---|---|---|
| mask-based | **ascending** | `LeaveOneOut`, `LeavePOut`, `KFold`, `GroupKFold`, `StratifiedKFold`, `StratifiedGroupKFold`, `LeaveOneGroupOut`, `LeavePGroupsOut`, `PredefinedSplit` |
| permutation-based | **draw order** | `ShuffleSplit`, `StratifiedShuffleSplit` |
| both | ascending | `GroupShuffleSplit` (groups drawn in permutation order, rows recovered by mask), `TimeSeriesSplit` |

`KFold(shuffle=True)` is the one that surprises people: shuffling changes *which*
rows land in a fold, not the order they are reported in.

---

## 2. What runs where

| | Rust | Python |
|---|---|---|
| splitter index generation | ✅ | |
| `shuffle=True` randomness | ✅ | |
| `ParameterGrid` / `ParameterSampler` combinatorics | ✅ | values |
| search + successive-halving schedules | ✅ | |
| score mean / std / rank | ✅ | |
| learning-curve tick resolution | ✅ | |
| permutation-test p-value | ✅ | |
| decision-threshold tuning | ✅ | |
| `fit` / `predict` / scoring | | ✅ (it is *your* estimator) |
| container row gather | | ✅ (native pandas/polars/pyarrow takes) |

Calling an arbitrary Python estimator is the one thing Rust cannot own, so the
search drivers take an evaluator and own everything around it. In Rust that
evaluator is a closure (§4); in Python it is the estimator itself.

---

## 3. Containers (Python)

Each input is gathered with its own native row-take and comes back as the same
type it went in as:

```python
import polars as pl

frame = pl.DataFrame({"a": range(100), "b": range(100)})
labels = pl.Series("y", [i % 3 for i in range(100)])

f_train, f_test, y_train, y_test = ms.train_test_split(
    frame, labels, test_size=0.25, random_state=0, stratify=labels
)
assert isinstance(f_train, pl.DataFrame) and isinstance(y_train, pl.Series)
```

Supported: numpy, pandas (DataFrame / Series / Index), polars (DataFrame /
Series), pyarrow (Table / RecordBatch / Array / ChunkedArray), scipy.sparse,
and any plain Python sequence including `range`.

polars, pyarrow and scipy are detected through `sys.modules` and are **never
imported** by mlrs, so the check cannot pull in a library you do not have.

Splitters read only the *row count* of `X`, so `KFold(...).split(polars_frame)`
works and returns integer index arrays — gather with the frame's own take, or
with `mlrs.model_selection._safe_indexing`.

---

## 4. The Rust-native surface

Rust callers get the same splitters plus a closure-based search driver. Nothing
here goes through Python.

```toml
[dependencies]
mlrs-algos = { version = "0.1", features = ["ndarray", "polars"] }
```

Both container features are **optional and off by default** — slices and the
dependency-free `RowMajor` view always work, and the mlrs wheels never build
polars (the Python side reaches polars through polars' own Python API).

### Splitting

```rust
use mlrs_algos::model_selection::{RandomStateSpec, SizeSpec};
use mlrs_algos::model_selection::split::{KFold, StratifiedKFold, train_test_split_indices};
use mlrs_algos::model_selection::NumpyRandomState;

// y and groups cross the API as sorted-unique factorization codes;
// `factorize` produces them from any `Ord` label type.
let (_classes, y_codes) = mlrs_algos::model_selection::factorize(&labels);

let folds = StratifiedKFold {
    n_splits: 5,
    shuffle: true,
    random_state: RandomStateSpec::Seed(42),
}
.split(&y_codes)?;

for split in &folds.splits {
    // split.train / split.test are Vec<i64> row indices
}

let mut rng = NumpyRandomState::from_seed(0);
let holdout = train_test_split_indices(
    n_rows,
    SizeSpec::Float(0.25),
    SizeSpec::None,
    /* shuffle */ true,
    Some(&y_codes),
    &mut rng,
)?;
```

Splitters return a `Splits { splits, warnings }`. The `warnings` are the
`UserWarning` texts scikit-learn would have emitted (e.g. "the least populated
class in y has only N members") — returned rather than logged so a caller can
route them wherever it wants.

`LeaveOneOut`, `LeavePOut` and `LeavePGroupsOut` also expose `split_at(i)`,
which unranks the `i`-th combination directly. Use it: `LeavePOut { p: 3 }` on
100 rows is 161 700 splits, and materializing them is not an option.

### Gathering rows

```rust
use mlrs_algos::model_selection::container::{take_split, RowContainer, RowMajor};

// ndarray
let (x_train, x_test) = take_split(&x_array2, &holdout);   // Array2 -> Array2

// polars
let (df_train, df_test) = take_split(&frame, &holdout);    // DataFrame -> PolarsResult<DataFrame>

// no dependencies: a flat row-major buffer
let matrix = RowMajor { data: &values, n_cols: 8 };
let (train, test) = take_split(&matrix, &holdout);         // -> Vec<f64>
```

### Searching

The search drivers take an evaluator closure, so they work over any estimator —
an `mlrs_algos` one, a wrapper around something else, or a pure function.

```rust
use mlrs_algos::model_selection::search::{evaluate_candidates, run_halving,
                                          HalvingParams, MinResources};

let results = evaluate_candidates(&candidate_indices, n_splits, |candidate, split| {
    let params = &grid[candidate];
    let Split { train, test } = &folds.splits[split];
    fit_and_score(params, train, test)   // your code
})?;

println!("best candidate: {}", results.candidates[results.best]);
println!("ranks: {:?}", results.summary.rank);
```

Successive halving works the same way, with the resource level threaded through:

```rust
let rounds = run_halving(
    HalvingParams {
        factor: 3,
        min_resources: MinResources::Exhaust,
        max_resources: n_rows,
        aggressive_elimination: false,
        smallest_resources: n_splits * 2,
    },
    n_candidates,
    n_splits,
    |candidate, split, n_resources| fit_and_score_on(candidate, split, n_resources),
)?;
// the winner is the best of the LAST round — a candidate scored on 20 rows has
// not beaten one measured on 2000.
```

---

## 5. Testing

Two complementary gates, because neither is sufficient alone:

| suite | oracle | why |
|---|---|---|
| `crates/mlrs-algos/tests/model_selection_*_test.rs` | committed `.npz` fixtures (`tests/fixtures/model_selection_splits_seed42.npz`) | the Rust suite must run with **no Python in the loop** |
| `crates/mlrs-py/python/tests/test_oracle_model_selection.py` | a **live** scikit-learn, same process | re-checks parity against the *installed* sklearn on every run, so an upstream change fails here instead of drifting from a frozen fixture |

Regenerate the Rust fixtures with `python scripts/gen_oracle.py` (needs numpy +
scikit-learn).

---

## 6. Known differences from scikit-learn

Small, deliberate, and none of them change which rows a split selects:

* **`pandas.Index`** is gathered positionally and returned as an `Index`.
  scikit-learn raises `TypeError` on it.
* **`pyarrow.RecordBatch`** comes back as a `RecordBatch`. scikit-learn degrades
  it to a `StructArray`.
* **`learning_curve(exploit_incremental_learning=True)`** raises
  `NotImplementedError` rather than silently ignoring the flag — the whole point
  of the flag is a different (`partial_fit`-driven) cost profile.
* **`TunedThresholdClassifierCV(scoring=...)`** requires a scoring string or a
  `make_scorer` result, not a bare callable: the threshold sweep needs the
  underlying metric function, which a plain scorer callable does not expose.
* **Metadata routing** (`set_score_request` / `set_fit_request`) is not wired
  through. `params=` is forwarded to the estimator's `fit`, with row-aligned
  entries indexed per fold — the behavior you get from scikit-learn with routing
  disabled, which is its default.
