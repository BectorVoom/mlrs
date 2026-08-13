"""Live-sklearn oracle for the full ``mlrs.model_selection`` surface
(MODSEL-RS-01..08).

``test_model_selection.py`` covers ``train_test_split``, ``KFold`` and
``StratifiedKFold`` in depth. This file covers everything else — the remaining
13 splitters, ``check_cv``, ``ParameterGrid``/``ParameterSampler``, the four
search estimators, the five validation functions, the two threshold classifiers
and the two displays — against a **live scikit-learn**, called with the same
arguments in the same process.

## Why live, and not a stored fixture

The recorded parity decision for this surface is *host-match*: the same
arguments must select the same ROWS and rank the same CANDIDATES, index for
index. A live comparison is strictly stronger than a stored ``.npz`` here, for
two reasons:

* an index permutation is exactly reproducible, so there is no tolerance to
  tune and nothing a fixture would capture that the live call does not;
* the live call re-checks parity against the *installed* scikit-learn on every
  run, so an upstream behavior change surfaces as a failure here rather than
  silently drifting away from a fixture frozen at authoring time.

(The Rust suite does use stored fixtures — `tests/fixtures/
model_selection_splits_seed42.npz` — because it must run with no Python in the
loop. The two gates are complementary, not redundant.)

## What "the same" means per splitter

Index ORDER is asserted, not just membership. sklearn's mask-based splitters
report ascending indices while its permutation-based ones report draw order,
and a caller zipping a split against another array observes the difference —
so ``assert_array_equal`` is used throughout rather than a set comparison.
"""

import numpy as np
import pytest
import sklearn.model_selection as skm
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression, Ridge

import mlrs.model_selection as ms

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised on a pandas-free install
    pd = None
try:
    import polars as pl
except ImportError:  # pragma: no cover - exercised on a polars-free install
    pl = None

needs_pandas = pytest.mark.skipif(pd is None, reason="pandas is not installed")
needs_polars = pytest.mark.skipif(pl is None, reason="polars is not installed")


N = 37


@pytest.fixture
def data():
    """37 rows, an imbalanced 3-class ``y``, and 7 ragged groups.

    Deliberately awkward: 37 is coprime with every fold count below (so the
    uneven-fold branch runs), the class counts 17/13/7 give stratification real
    work, and the ragged groups give the group splitters' greedy balancing ties
    to break.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, 4))
    y = np.array([0] * 17 + [1] * 13 + [2] * 7)
    rng.shuffle(y)
    groups = np.array([i % 7 for i in range(N)])
    groups[:5] = 0
    return X, y, groups


def assert_same_splits(got, want, name):
    """Every split identical, in order, on both sides."""
    got, want = list(got), list(want)
    assert len(got) == len(want), f"{name}: {len(got)} splits vs sklearn's {len(want)}"
    for i, ((g_tr, g_te), (w_tr, w_te)) in enumerate(zip(got, want)):
        np.testing.assert_array_equal(g_te, w_te, err_msg=f"{name} split {i} test")
        np.testing.assert_array_equal(g_tr, w_tr, err_msg=f"{name} split {i} train")


# --------------------------------------------------------------------------- #
# 1. splitter parity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("n_splits", [2, 3, 5])
@pytest.mark.parametrize("seed", [0, 42])
def test_group_kfold_parity(data, shuffle, n_splits, seed):
    X, y, groups = data
    kwargs = {"shuffle": shuffle, "random_state": seed if shuffle else None}
    assert_same_splits(
        ms.GroupKFold(n_splits, **kwargs).split(X, y, groups),
        skm.GroupKFold(n_splits, **kwargs).split(X, y, groups),
        "GroupKFold",
    )


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("n_splits", [2, 3])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_stratified_group_kfold_parity(data, shuffle, n_splits, seed):
    X, y, groups = data
    kwargs = {"shuffle": shuffle, "random_state": seed if shuffle else None}
    assert_same_splits(
        ms.StratifiedGroupKFold(n_splits, **kwargs).split(X, y, groups),
        skm.StratifiedGroupKFold(n_splits, **kwargs).split(X, y, groups),
        "StratifiedGroupKFold",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_splits": 3},
        {"n_splits": 3, "gap": 2},
        {"n_splits": 3, "test_size": 5},
        {"n_splits": 3, "max_train_size": 10},
        {"n_splits": 4, "max_train_size": 8, "test_size": 4, "gap": 1},
    ],
)
def test_time_series_split_parity(data, kwargs):
    X, _, _ = data
    assert_same_splits(
        ms.TimeSeriesSplit(**kwargs).split(X),
        skm.TimeSeriesSplit(**kwargs).split(X),
        "TimeSeriesSplit",
    )


def test_leave_one_out_parity(data):
    X, _, _ = data
    assert_same_splits(
        ms.LeaveOneOut().split(X), skm.LeaveOneOut().split(X), "LeaveOneOut"
    )
    assert ms.LeaveOneOut().get_n_splits(X) == skm.LeaveOneOut().get_n_splits(X)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_leave_p_out_parity(p):
    # A small n keeps C(n, p) tractable while still exercising the combination
    # unranking that the streamed implementation depends on.
    X = np.arange(18).reshape(9, 2)
    assert_same_splits(
        ms.LeavePOut(p).split(X), skm.LeavePOut(p).split(X), f"LeavePOut(p={p})"
    )
    assert ms.LeavePOut(p).get_n_splits(X) == skm.LeavePOut(p).get_n_splits(X)


def test_leave_one_group_out_parity(data):
    X, y, groups = data
    assert_same_splits(
        ms.LeaveOneGroupOut().split(X, y, groups),
        skm.LeaveOneGroupOut().split(X, y, groups),
        "LeaveOneGroupOut",
    )


@pytest.mark.parametrize("n_groups", [1, 2, 3])
def test_leave_p_groups_out_parity(data, n_groups):
    X, y, groups = data
    assert_same_splits(
        ms.LeavePGroupsOut(n_groups).split(X, y, groups),
        skm.LeavePGroupsOut(n_groups).split(X, y, groups),
        "LeavePGroupsOut",
    )


def test_predefined_split_parity():
    test_fold = np.array([-1, 0, 1, 2] * 9 + [0])
    assert_same_splits(
        ms.PredefinedSplit(test_fold).split(),
        skm.PredefinedSplit(test_fold).split(),
        "PredefinedSplit",
    )
    assert ms.PredefinedSplit(test_fold).get_n_splits() == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_splits": 3, "test_size": 0.3},
        {"n_splits": 3, "test_size": 10},
        {"n_splits": 2, "train_size": 20, "test_size": 10},
        {"n_splits": 2, "train_size": 0.5},
    ],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_shuffle_split_parity(data, kwargs, seed):
    X, _, _ = data
    assert_same_splits(
        ms.ShuffleSplit(random_state=seed, **kwargs).split(X),
        skm.ShuffleSplit(random_state=seed, **kwargs).split(X),
        "ShuffleSplit",
    )


@pytest.mark.parametrize(
    "kwargs", [{}, {"n_splits": 3, "test_size": 0.4}, {"n_splits": 2, "test_size": 2}]
)
@pytest.mark.parametrize("seed", [0, 42])
def test_group_shuffle_split_parity(data, kwargs, seed):
    X, y, groups = data
    assert_same_splits(
        ms.GroupShuffleSplit(random_state=seed, **kwargs).split(X, y, groups),
        skm.GroupShuffleSplit(random_state=seed, **kwargs).split(X, y, groups),
        "GroupShuffleSplit",
    )


@pytest.mark.parametrize(
    "kwargs", [{}, {"n_splits": 5, "test_size": 0.25}, {"n_splits": 3, "train_size": 20}]
)
@pytest.mark.parametrize("seed", [0, 42])
def test_stratified_shuffle_split_parity(data, kwargs, seed):
    X, y, _ = data
    assert_same_splits(
        ms.StratifiedShuffleSplit(random_state=seed, **kwargs).split(X, y),
        skm.StratifiedShuffleSplit(random_state=seed, **kwargs).split(X, y),
        "StratifiedShuffleSplit",
    )


@pytest.mark.parametrize("n_repeats", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_repeated_kfold_parity(data, n_repeats, seed):
    X, y, _ = data
    assert_same_splits(
        ms.RepeatedKFold(n_splits=3, n_repeats=n_repeats, random_state=seed).split(X),
        skm.RepeatedKFold(n_splits=3, n_repeats=n_repeats, random_state=seed).split(X),
        "RepeatedKFold",
    )
    assert_same_splits(
        ms.RepeatedStratifiedKFold(
            n_splits=3, n_repeats=n_repeats, random_state=seed
        ).split(X, y),
        skm.RepeatedStratifiedKFold(
            n_splits=3, n_repeats=n_repeats, random_state=seed
        ).split(X, y),
        "RepeatedStratifiedKFold",
    )


def test_repeated_kfold_shares_one_generator(data):
    """Each repeat continues the previous repeat's stream.

    Re-seeding per repeat would make repeat 2 a copy of repeat 1 while still
    satisfying every per-repeat invariant — invisible unless two repeats are
    compared, which is what this does."""
    X, _, _ = data
    splits = list(ms.RepeatedKFold(n_splits=3, n_repeats=2, random_state=0).split(X))
    assert len(splits) == 6
    assert not np.array_equal(splits[0][1], splits[3][1])


def test_randomstate_instance_is_advanced_like_sklearn(data):
    """A live ``RandomState`` must come back advanced to the same point.

    This is the property that makes an mlrs splitter safe to drop into code
    that shares one generator across several calls — and it is the one a
    seed-only bridge would break."""
    X, y, _ = data
    mine = np.random.RandomState(0)
    theirs = np.random.RandomState(0)
    list(ms.StratifiedShuffleSplit(n_splits=3, random_state=mine).split(X, y))
    list(skm.StratifiedShuffleSplit(n_splits=3, random_state=theirs).split(X, y))
    np.testing.assert_array_equal(mine.get_state()[1], theirs.get_state()[1])
    assert mine.get_state()[2] == theirs.get_state()[2]
    # ...and a subsequent draw from each therefore agrees.
    assert mine.randint(1_000_000) == theirs.randint(1_000_000)


def test_string_and_object_labels_are_supported(data):
    """`y` and `groups` may be strings; the codes crossing into Rust must keep
    numpy's lexicographic ordering."""
    X, _, _ = data
    y = np.array(["beta", "alpha", "gamma"] * 12 + ["alpha"])
    groups = np.array([f"g{i % 5}" for i in range(N)])
    assert_same_splits(
        ms.StratifiedKFold(3).split(X, y), skm.StratifiedKFold(3).split(X, y), "str y"
    )
    assert_same_splits(
        ms.GroupKFold(3).split(X, y, groups),
        skm.GroupKFold(3).split(X, y, groups),
        "str groups",
    )


@needs_polars
def test_splitters_accept_a_polars_frame(data):
    """Only the ROW COUNT of X is read, so a polars frame splits identically."""
    X, y, _ = data
    frame = pl.DataFrame({f"c{i}": X[:, i] for i in range(X.shape[1])})
    assert_same_splits(
        ms.StratifiedKFold(3, shuffle=True, random_state=0).split(frame, y),
        skm.StratifiedKFold(3, shuffle=True, random_state=0).split(X, y),
        "polars X",
    )


@needs_pandas
def test_splitters_accept_a_pandas_frame(data):
    X, y, _ = data
    frame = pd.DataFrame(X, index=np.arange(100, 100 + N))
    assert_same_splits(
        ms.StratifiedKFold(3, shuffle=True, random_state=0).split(frame, pd.Series(y)),
        skm.StratifiedKFold(3, shuffle=True, random_state=0).split(X, y),
        "pandas X",
    )


# --------------------------------------------------------------------------- #
# 2. check_cv
# --------------------------------------------------------------------------- #


def test_check_cv_dispatch_matches_sklearn(data):
    _, y, _ = data
    assert isinstance(ms.check_cv(4), ms.KFold)
    assert ms.check_cv(4).n_splits == 4
    assert isinstance(ms.check_cv(4, y, classifier=True), ms.StratifiedKFold)
    # a classifier with a CONTINUOUS y still falls back to plain KFold
    assert isinstance(
        ms.check_cv(4, np.linspace(0, 1, len(y)), classifier=True), ms.KFold
    )
    # a splitter passes through untouched
    splitter = ms.GroupKFold(3)
    assert ms.check_cv(splitter) is splitter


def test_check_cv_wraps_an_iterable_of_index_pairs():
    pairs = [(np.array([0, 1]), np.array([2])), (np.array([1, 2]), np.array([0]))]
    checked = ms.check_cv(pairs)
    assert checked.get_n_splits() == 2
    assert_same_splits(checked.split(), pairs, "iterable cv")


def test_check_cv_rejects_a_string():
    with pytest.raises(ValueError, match="Expected `cv` as an integer"):
        ms.check_cv("kfold")


# --------------------------------------------------------------------------- #
# 3. ParameterGrid / ParameterSampler
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "grid",
    [
        {"a": [1, 2, 3]},
        {"a": [1, 2, 3], "b": ["x", "y"]},
        [{"a": [1, 2]}, {"b": [3, 4, 5]}],
        [{}],
        {"z": [1], "a": [1, 2], "m": [7, 8, 9]},
    ],
)
def test_parameter_grid_parity(grid):
    assert list(ms.ParameterGrid(grid)) == list(skm.ParameterGrid(grid))
    assert len(ms.ParameterGrid(grid)) == len(skm.ParameterGrid(grid))
    for i in range(len(ms.ParameterGrid(grid))):
        assert ms.ParameterGrid(grid)[i] == skm.ParameterGrid(grid)[i]


def test_parameter_grid_index_out_of_range():
    with pytest.raises(IndexError):
        ms.ParameterGrid({"a": [1, 2]})[5]


@pytest.mark.parametrize("n_iter", [1, 3, 5, 40])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_parameter_sampler_list_parity(n_iter, seed):
    dist = {"a": [1, 2, 3], "b": ["x", "y"], "c": [0, 1, 2, 3]}
    with pytest.warns(UserWarning) if n_iter > 24 else _nullcontext():
        mine = list(ms.ParameterSampler(dist, n_iter, random_state=seed))
    with pytest.warns(UserWarning) if n_iter > 24 else _nullcontext():
        theirs = list(skm.ParameterSampler(dist, n_iter, random_state=seed))
    assert mine == theirs


def _nullcontext():
    import contextlib

    return contextlib.nullcontext()


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_parameter_sampler_distribution_parity(seed):
    """The scipy path: mlrs hands the generator to ``rvs`` and takes it back.

    A bridge that merely re-seeded would produce valid-looking draws that
    diverge from sklearn's after the first ``rvs`` call — which is exactly what
    this compares."""
    from scipy.stats import expon, uniform

    dist = {"a": [1, 2, 3], "c": uniform(0, 1), "d": expon(scale=2)}
    mine = list(ms.ParameterSampler(dist, 6, random_state=seed))
    theirs = list(skm.ParameterSampler(dist, 6, random_state=seed))
    assert [sorted(p) for p in mine] == [sorted(p) for p in theirs]
    for m, t in zip(mine, theirs):
        assert m["a"] == t["a"]
        assert m["c"] == pytest.approx(t["c"])
        assert m["d"] == pytest.approx(t["d"])


def test_parameter_grid_validation_matches_sklearn():
    with pytest.raises(TypeError, match="needs to be a list or a numpy array"):
        ms.ParameterGrid({"a": 1})
    with pytest.raises(ValueError, match="non-empty sequence"):
        ms.ParameterGrid({"a": []})
    with pytest.raises(TypeError, match="not a dict"):
        ms.ParameterGrid([1])


# --------------------------------------------------------------------------- #
# 4. cross-validation drivers
# --------------------------------------------------------------------------- #


@pytest.fixture
def regression():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(scale=0.1, size=60)
    return X, y


@pytest.fixture
def classification():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=80) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("cv", [3, 5])
def test_cross_val_score_parity(regression, cv):
    X, y = regression
    np.testing.assert_allclose(
        ms.cross_val_score(Ridge(), X, y, cv=cv),
        skm.cross_val_score(Ridge(), X, y, cv=cv),
    )


def test_cross_validate_parity_multimetric(regression):
    X, y = regression
    scoring = ["r2", "neg_mean_squared_error"]
    mine = ms.cross_validate(
        Ridge(), X, y, cv=3, scoring=scoring, return_train_score=True
    )
    theirs = skm.cross_validate(
        Ridge(), X, y, cv=3, scoring=scoring, return_train_score=True
    )
    assert sorted(mine) == sorted(theirs)
    for key in theirs:
        if key.startswith(("test_", "train_")):
            np.testing.assert_allclose(mine[key], theirs[key], err_msg=key)


def test_cross_validate_returns_indices_and_estimators(regression):
    X, y = regression
    out = ms.cross_validate(
        Ridge(), X, y, cv=3, return_estimator=True, return_indices=True
    )
    assert len(out["estimator"]) == 3
    assert len(out["indices"]["test"]) == 3
    covered = np.sort(np.concatenate(out["indices"]["test"]))
    np.testing.assert_array_equal(covered, np.arange(len(y)))


def test_cross_validate_sample_weight_is_indexed_per_fold(regression):
    """A row-aligned ``params`` entry must be sliced to the fold, not passed whole."""
    X, y = regression
    weights = np.linspace(0.5, 1.5, len(y))
    mine = ms.cross_validate(Ridge(), X, y, cv=3, params={"sample_weight": weights})
    theirs = skm.cross_validate(
        Ridge(), X, y, cv=3, params={"sample_weight": weights}
    )
    np.testing.assert_allclose(mine["test_score"], theirs["test_score"])


@pytest.mark.parametrize("method", ["predict", "predict_proba", "decision_function"])
def test_cross_val_predict_parity(classification, method):
    X, y = classification
    np.testing.assert_allclose(
        ms.cross_val_predict(LogisticRegression(), X, y, cv=4, method=method),
        skm.cross_val_predict(LogisticRegression(), X, y, cv=4, method=method),
    )


def test_cross_val_predict_keeps_a_multi_output_response_a_list(classification):
    """A response method that returns one array PER TARGET stays a list.

    ``RandomForestClassifier.predict_proba`` on a multilabel ``y`` returns
    ``n_targets`` arrays of ``(n_test, n_classes)``. Concatenating the folds
    with a plain ``np.concatenate`` would glue the target axis onto the sample
    axis and hand back an array of the wrong rank — which is what
    ``StackingClassifier`` on a multilabel target consumes, so the defect would
    surface there as a shape error rather than here.
    """
    from sklearn.ensemble import RandomForestClassifier

    X, y = classification
    Y = np.column_stack([y, 1 - y, (X[:, 0] > 0).astype(int)])
    forest = RandomForestClassifier(n_estimators=5, random_state=0)
    mine = ms.cross_val_predict(forest, X, Y, cv=3, method="predict_proba")
    theirs = skm.cross_val_predict(forest, X, Y, cv=3, method="predict_proba")

    assert isinstance(mine, list) and len(mine) == len(theirs) == 3
    for got, want in zip(mine, theirs):
        assert got.shape == want.shape == (len(y), 2)
        np.testing.assert_allclose(got, want)


def test_cross_val_predict_rejects_a_non_partition(regression):
    X, y = regression
    with pytest.raises(ValueError, match="only works for partitions"):
        ms.cross_val_predict(Ridge(), X, y, cv=ms.ShuffleSplit(3, random_state=0))


def test_learning_curve_parity(regression):
    X, y = regression
    mine = ms.learning_curve(Ridge(), X, y, cv=3)
    theirs = skm.learning_curve(Ridge(), X, y, cv=3)
    np.testing.assert_array_equal(mine[0], theirs[0])
    np.testing.assert_allclose(mine[1], theirs[1])
    np.testing.assert_allclose(mine[2], theirs[2])


def test_learning_curve_absolute_sizes_and_times(regression):
    X, y = regression
    mine = ms.learning_curve(Ridge(), X, y, cv=3, train_sizes=[10, 20, 40], return_times=True)
    theirs = skm.learning_curve(
        Ridge(), X, y, cv=3, train_sizes=[10, 20, 40], return_times=True
    )
    np.testing.assert_array_equal(mine[0], theirs[0])
    np.testing.assert_allclose(mine[1], theirs[1])
    np.testing.assert_allclose(mine[2], theirs[2])
    # times are wall-clock, so only their SHAPE is comparable
    assert mine[3].shape == theirs[3].shape
    assert mine[4].shape == theirs[4].shape


def test_learning_curve_warns_on_duplicate_ticks(regression):
    X, y = regression
    with pytest.warns(RuntimeWarning, match="Removed duplicate entries"):
        sizes, _, _ = ms.learning_curve(Ridge(), X, y, cv=3, train_sizes=[0.021, 0.022])
    assert len(sizes) == 1


def test_validation_curve_parity(regression):
    X, y = regression
    mine = ms.validation_curve(
        Ridge(), X, y, param_name="alpha", param_range=[0.1, 1.0, 10.0], cv=3
    )
    theirs = skm.validation_curve(
        Ridge(), X, y, param_name="alpha", param_range=[0.1, 1.0, 10.0], cv=3
    )
    np.testing.assert_allclose(mine[0], theirs[0])
    np.testing.assert_allclose(mine[1], theirs[1])


@pytest.mark.parametrize("seed", [0, 3])
def test_permutation_test_score_parity(classification, seed):
    X, y = classification
    mine = ms.permutation_test_score(
        LogisticRegression(), X, y, cv=3, n_permutations=20, random_state=seed
    )
    theirs = skm.permutation_test_score(
        LogisticRegression(), X, y, cv=3, n_permutations=20, random_state=seed
    )
    assert mine[0] == pytest.approx(theirs[0])
    np.testing.assert_allclose(np.sort(mine[1]), np.sort(theirs[1]))
    assert mine[2] == pytest.approx(theirs[2])


def test_permutation_test_score_pvalue_floor(classification):
    """The best attainable p-value is ``1 / (n_permutations + 1)``, never 0."""
    X, y = classification
    _, _, pvalue = ms.permutation_test_score(
        LogisticRegression(), X, y, cv=3, n_permutations=9, random_state=0
    )
    assert pvalue >= 1 / 10


# --------------------------------------------------------------------------- #
# 5. search estimators
# --------------------------------------------------------------------------- #


def test_grid_search_parity(regression):
    X, y = regression
    grid = {"alpha": [0.01, 0.1, 1.0, 10.0]}
    mine = ms.GridSearchCV(Ridge(), grid, cv=3, return_train_score=True).fit(X, y)
    theirs = skm.GridSearchCV(Ridge(), grid, cv=3, return_train_score=True).fit(X, y)

    assert mine.best_params_ == theirs.best_params_
    assert mine.best_index_ == theirs.best_index_
    assert mine.best_score_ == pytest.approx(theirs.best_score_)
    for key in ("mean_test_score", "std_test_score", "mean_train_score"):
        np.testing.assert_allclose(
            mine.cv_results_[key], theirs.cv_results_[key], err_msg=key
        )
    np.testing.assert_array_equal(
        mine.cv_results_["rank_test_score"], theirs.cv_results_["rank_test_score"]
    )
    np.testing.assert_allclose(mine.predict(X), theirs.predict(X))


def test_grid_search_multiple_subgrids_use_masked_param_columns(regression):
    """A candidate from another sub-grid has no value for a parameter — that is
    a MASKED entry, not ``None``, which is a different (and legitimate) value."""
    X, y = regression
    grid = [{"alpha": [0.1, 1.0]}, {"fit_intercept": [True, False]}]
    mine = ms.GridSearchCV(Ridge(), grid, cv=3).fit(X, y)
    theirs = skm.GridSearchCV(Ridge(), grid, cv=3).fit(X, y)
    assert list(mine.cv_results_["params"]) == list(theirs.cv_results_["params"])
    np.testing.assert_array_equal(
        mine.cv_results_["param_alpha"].mask, theirs.cv_results_["param_alpha"].mask
    )


def test_grid_search_multimetric_refit(regression):
    X, y = regression
    grid = {"alpha": [0.1, 1.0, 10.0]}
    scoring = {"r2": "r2", "mse": "neg_mean_squared_error"}
    mine = ms.GridSearchCV(Ridge(), grid, cv=3, scoring=scoring, refit="r2").fit(X, y)
    theirs = skm.GridSearchCV(Ridge(), grid, cv=3, scoring=scoring, refit="r2").fit(X, y)
    assert mine.best_params_ == theirs.best_params_
    np.testing.assert_allclose(
        mine.cv_results_["mean_test_r2"], theirs.cv_results_["mean_test_r2"]
    )
    np.testing.assert_allclose(
        mine.cv_results_["mean_test_mse"], theirs.cv_results_["mean_test_mse"]
    )


def test_grid_search_refit_false_blocks_predict(regression):
    X, y = regression
    search = ms.GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3, refit=False).fit(X, y)
    with pytest.raises(AttributeError, match="refit=False"):
        search.predict(X)
    # ...but the results are still there
    assert "mean_test_score" in search.cv_results_


def test_grid_search_error_score_records_nan(regression):
    """A candidate that cannot fit is recorded as NaN and ranked LAST, not
    dropped — so the candidates that did work still report."""
    X, y = regression
    grid = {"alpha": [1.0, -5.0]}  # a negative alpha is invalid for Ridge
    with pytest.warns(Warning):
        mine = ms.GridSearchCV(Ridge(), grid, cv=3, error_score=np.nan).fit(X, y)
    scores = mine.cv_results_["mean_test_score"]
    assert np.isnan(scores).sum() == 1
    assert mine.best_params_ == {"alpha": 1.0}


@pytest.mark.parametrize("seed", [0, 5])
def test_randomized_search_parity(regression, seed):
    X, y = regression
    dist = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    mine = ms.RandomizedSearchCV(
        Ridge(), dist, n_iter=3, cv=3, random_state=seed
    ).fit(X, y)
    theirs = skm.RandomizedSearchCV(
        Ridge(), dist, n_iter=3, cv=3, random_state=seed
    ).fit(X, y)
    assert list(mine.cv_results_["params"]) == list(theirs.cv_results_["params"])
    assert mine.best_params_ == theirs.best_params_
    np.testing.assert_allclose(
        mine.cv_results_["mean_test_score"], theirs.cv_results_["mean_test_score"]
    )


@pytest.fixture
def big_regression():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 4))
    y = X @ np.array([1.0, 2.0, 3.0, 4.0]) + rng.normal(size=400)
    return X, y


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"factor": 2},
        {"min_resources": 20},
        {"min_resources": "smallest"},
        {"aggressive_elimination": True, "max_resources": 200},
    ],
)
def test_halving_grid_search_schedule_parity(big_regression, kwargs):
    """The schedule — rounds, resources per round, survivors per round — is the
    whole algorithm, and it is derived in Rust."""
    X, y = big_regression
    grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    mine = ms.HalvingGridSearchCV(Ridge(), grid, cv=3, random_state=0, **kwargs).fit(X, y)
    theirs = skm.HalvingGridSearchCV(
        Ridge(), grid, cv=3, random_state=0, **kwargs
    ).fit(X, y)
    assert mine.min_resources_ == theirs.min_resources_
    assert mine.max_resources_ == theirs.max_resources_
    assert mine.n_required_iterations_ == theirs.n_required_iterations_
    assert mine.n_possible_iterations_ == theirs.n_possible_iterations_
    assert mine.n_iterations_ == theirs.n_iterations_
    assert mine.n_resources_ == theirs.n_resources_
    assert mine.n_candidates_ == theirs.n_candidates_


def test_halving_random_search_schedule_parity(big_regression):
    X, y = big_regression
    dist = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    with pytest.warns(UserWarning):
        mine = ms.HalvingRandomSearchCV(Ridge(), dist, cv=3, random_state=0).fit(X, y)
    with pytest.warns(UserWarning):
        theirs = skm.HalvingRandomSearchCV(Ridge(), dist, cv=3, random_state=0).fit(X, y)
    assert mine.n_resources_ == theirs.n_resources_
    assert mine.n_candidates_ == theirs.n_candidates_
    assert mine.min_resources_ == theirs.min_resources_


def test_halving_rejects_both_exhaust(big_regression):
    X, y = big_regression
    with pytest.raises(ValueError, match="cannot be both set to 'exhaust'"):
        ms.HalvingRandomSearchCV(
            Ridge(), {"alpha": [0.1, 1.0]}, min_resources="exhaust",
            n_candidates="exhaust", cv=3,
        ).fit(X, y)


def test_halving_best_is_from_the_last_iteration(big_regression):
    """A candidate that scored well on 20 rows has not beaten one measured on
    400 — the winner must come from the final round."""
    X, y = big_regression
    grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
    search = ms.HalvingGridSearchCV(Ridge(), grid, cv=3, random_state=0).fit(X, y)
    last_iter = np.max(search.cv_results_["iter"])
    assert search.cv_results_["iter"][search.best_index_] == last_iter


# --------------------------------------------------------------------------- #
# 6. decision thresholds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("threshold", ["auto", 0.3, 0.5, 0.75])
def test_fixed_threshold_classifier_parity(classification, threshold):
    X, y = classification
    mine = ms.FixedThresholdClassifier(LogisticRegression(), threshold=threshold).fit(X, y)
    theirs = skm.FixedThresholdClassifier(
        LogisticRegression(), threshold=threshold
    ).fit(X, y)
    np.testing.assert_array_equal(mine.predict(X), theirs.predict(X))
    np.testing.assert_allclose(mine.predict_proba(X), theirs.predict_proba(X))


def test_fixed_threshold_tie_is_positive():
    """``>= threshold``: a row scoring exactly the threshold is POSITIVE."""
    import mlrs.model_selection as m

    assert m._ext().apply_threshold([0.49, 0.5, 0.51], 0.5) == [0, 1, 1]


@pytest.mark.parametrize("cv", [3, 5, 0.3])
def test_tuned_threshold_parity(classification, cv):
    X, y = classification
    mine = ms.TunedThresholdClassifierCV(
        LogisticRegression(), cv=cv, random_state=0, store_cv_results=True
    ).fit(X, y)
    theirs = skm.TunedThresholdClassifierCV(
        LogisticRegression(), cv=cv, random_state=0, store_cv_results=True
    ).fit(X, y)
    assert mine.best_threshold_ == pytest.approx(theirs.best_threshold_)
    assert mine.best_score_ == pytest.approx(theirs.best_score_)
    np.testing.assert_allclose(
        mine.cv_results_["thresholds"], theirs.cv_results_["thresholds"]
    )
    np.testing.assert_allclose(mine.cv_results_["scores"], theirs.cv_results_["scores"])
    np.testing.assert_array_equal(mine.predict(X), theirs.predict(X))


def test_tuned_threshold_custom_scoring_and_grid(classification):
    X, y = classification
    mine = ms.TunedThresholdClassifierCV(
        LogisticRegression(), scoring="f1", thresholds=[0.2, 0.5, 0.8], cv=3
    ).fit(X, y)
    theirs = skm.TunedThresholdClassifierCV(
        LogisticRegression(), scoring="f1", thresholds=[0.2, 0.5, 0.8], cv=3
    ).fit(X, y)
    assert mine.best_threshold_ == pytest.approx(theirs.best_threshold_)


def test_tuned_threshold_rejects_multiclass():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 3))
    y = rng.randint(0, 3, size=60)
    with pytest.raises(ValueError, match="Only binary classification"):
        ms.TunedThresholdClassifierCV(LogisticRegression(), cv=3).fit(X, y)


# --------------------------------------------------------------------------- #
# 7. displays
# --------------------------------------------------------------------------- #


def test_learning_curve_display(regression):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    X, y = regression
    display = ms.LearningCurveDisplay.from_estimator(Ridge(), X, y, cv=3)
    assert display.train_scores.shape == display.test_scores.shape
    assert display.ax_.get_xlabel() == "Number of samples in the training set"
    assert len(display.lines_) == 2


def test_validation_curve_display(regression):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    X, y = regression
    display = ms.ValidationCurveDisplay.from_estimator(
        Ridge(), X, y, param_name="alpha", param_range=[0.1, 1.0, 10.0], cv=3
    )
    assert display.train_scores.shape == (3, 3)
    assert display.ax_.get_xlabel() == "alpha"


# --------------------------------------------------------------------------- #
# 8. sklearn interop in the other direction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "splitter",
    [
        ms.KFold(3),
        ms.GroupKFold(3),
        ms.ShuffleSplit(3, random_state=0),
        ms.TimeSeriesSplit(3),
    ],
)
def test_mlrs_splitters_drive_sklearn_cross_val_score(regression, splitter):
    """sklearn duck-types ``cv=``; an mlrs splitter must work inside sklearn's
    own drivers, not just mlrs's."""
    X, y = regression
    groups = np.arange(len(y)) % 4
    scores = skm.cross_val_score(Ridge(), X, y, cv=splitter, groups=groups)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_mlrs_stratified_kfold_drives_sklearn_gridsearch(classification):
    """The stratified splitter needs a discrete target, so it gets its own case
    rather than riding the regression fixture above."""
    X, y = classification
    search = skm.GridSearchCV(
        LogisticRegression(),
        {"C": [0.1, 1.0]},
        cv=ms.StratifiedKFold(4, shuffle=True, random_state=0),
    ).fit(X, y)
    assert search.n_splits_ == 4
    assert len(search.cv_results_["mean_test_score"]) == 2


def test_sklearn_splitters_drive_mlrs_search(regression):
    """...and the converse: a sklearn splitter inside an mlrs search."""
    X, y = regression
    search = ms.GridSearchCV(
        Ridge(), {"alpha": [0.1, 1.0]}, cv=skm.KFold(4, shuffle=True, random_state=0)
    ).fit(X, y)
    assert search.n_splits_ == 4
    assert search.best_params_["alpha"] in (0.1, 1.0)


def test_mlrs_search_is_clonable_and_reports_params(regression):
    """``clone`` + ``get_params`` are what let an mlrs search sit inside a
    sklearn ``Pipeline`` or an outer cross-validation."""
    from sklearn.base import clone

    X, y = regression
    search = ms.GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=3)
    fresh = clone(search)
    assert fresh.get_params()["cv"] == 3
    assert fresh.get_params()["param_grid"] == {"alpha": [0.1, 1.0]}
    scores = ms.cross_val_score(search, X, y, cv=2)
    assert scores.shape == (2,)
