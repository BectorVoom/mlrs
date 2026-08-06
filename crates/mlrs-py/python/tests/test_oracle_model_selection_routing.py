"""Live-sklearn oracle for metadata routing through ``mlrs.model_selection``.

``test_oracle_model_selection.py`` covers the surface with routing OFF — the
default, and the configuration every other test in the tree runs under. This
file covers the same entry points with
``sklearn.set_config(enable_metadata_routing=True)``, where ``params`` stops
being "extra kwargs for ``fit``" and becomes a routed bundle: each consumer (the
estimator, the splitter, the scorers) receives exactly the metadata it asked for
through ``set_fit_request`` / ``set_split_request`` / ``set_score_request``, and
metadata nobody asked for is an error rather than a silent drop.

## What is asserted

Three kinds of claim, in descending order of how much they'd hurt to get wrong:

* **where the metadata went** — checked against a recording consumer, since two
  implementations can agree on a score while disagreeing about which rows the
  weights came from;
* **live parity with scikit-learn** — the same call, same arguments, same
  process, compared on the numbers *and* on the exception type;
* **the routing declarations themselves** (``get_metadata_routing``), because a
  wrong declaration is invisible until some caller nests the estimator in a
  pipeline that routes through it.

## The one deliberate divergence

``permutation_test_score`` constrains its permutation by the routed ``groups``;
scikit-learn constrains only the *split* by them, leaving the permutation
global. That is SK-002 in ``docs/upstream-sklearn-issues.md`` and it has its own
test at the bottom of this file, which pins BOTH behaviours so an upstream fix
shows up here as a failure rather than as drift.
"""

import copy

import numpy as np
import pytest
import sklearn
import sklearn.model_selection as skm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, make_scorer, r2_score

import mlrs.model_selection as ms

N = 40


@pytest.fixture
def regression():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(N, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(scale=0.1, size=N)
    return X, y


@pytest.fixture
def classification():
    rng = np.random.RandomState(1)
    X = rng.normal(size=(N, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=N) > 0).astype(int)
    return X, y


@pytest.fixture
def weights():
    return np.linspace(0.5, 1.5, N)


@pytest.fixture
def groups():
    return np.repeat(np.arange(4), N // 4)


@pytest.fixture
def routing_enabled():
    """Run the test body with sklearn's global routing switch on."""
    with sklearn.config_context(enable_metadata_routing=True):
        yield


class _Registry(list):
    """A list that survives ``clone``, so per-fold clones append to one log.

    ``clone`` deep-copies non-estimator constructor params, which would give
    every fold its own private (and unreachable) registry. Returning ``self``
    from both copy hooks is sklearn's own trick in its routing tests.
    """

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """A regressor that appends every ``fit``'s metadata to ``registry``.

    The metadata parameters are spelled out in the signature (rather than
    swallowed by ``**kwargs``) because that signature is what sklearn derives
    the default metadata request from.
    """

    def __init__(self, registry=None):
        self.registry = registry

    def fit(self, X, y, sample_weight=None, metadata=None):
        if self.registry is not None:
            self.registry.append(
                {"n": len(X), "sample_weight": sample_weight, "metadata": metadata}
            )
        self.n_features_in_ = X.shape[1]
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """A classifier counterpart to :class:`RecordingRegressor`."""

    def __init__(self, registry=None):
        self.registry = registry

    def fit(self, X, y, sample_weight=None, metadata=None):
        if self.registry is not None:
            self.registry.append(
                {"n": len(X), "sample_weight": sample_weight, "metadata": metadata}
            )
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        counts = np.bincount(np.searchsorted(self.classes_, y))
        self._majority = self.classes_[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(len(X), self._majority)

    def predict_proba(self, X):
        # A constant-but-not-degenerate response: the threshold sweep needs a
        # score to sweep, and a single repeated value would make every
        # threshold equivalent.
        p = 1.0 / (1.0 + np.exp(-X[:, 0]))
        return np.column_stack([1.0 - p, p])


class PlainRegressor(RegressorMixin, BaseEstimator):
    """A regressor whose ``fit`` takes no metadata at all.

    Used where a test needs a name that NO consumer knows: with a
    ``sample_weight``-aware estimator the same call raises the *unrequested*
    error instead, which is a different failure.
    """

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def recording_metric(registry):
    """A metric that logs the metadata it was called with, then scores as R²."""

    def metric(y_true, y_pred, sample_weight=None, metadata=None):
        registry.append(
            {"n": len(y_true), "sample_weight": sample_weight, "metadata": metadata}
        )
        return r2_score(y_true, y_pred, sample_weight=sample_weight)

    return metric


# --------------------------------------------------------------------------- #
# the routing declarations
# --------------------------------------------------------------------------- #


SPLITTERS = [
    ("KFold", {}),
    ("StratifiedKFold", {}),
    ("ShuffleSplit", {}),
    ("StratifiedShuffleSplit", {}),
    ("TimeSeriesSplit", {}),
    ("LeaveOneOut", {}),
    ("LeavePOut", {"p": 2}),
    ("PredefinedSplit", {"test_fold": [0, 1, 0, 1]}),
    ("RepeatedKFold", {}),
    ("RepeatedStratifiedKFold", {}),
    ("GroupKFold", {}),
    ("StratifiedGroupKFold", {}),
    ("GroupShuffleSplit", {}),
    ("LeaveOneGroupOut", {}),
    ("LeavePGroupsOut", {"n_groups": 2}),
]


@pytest.mark.parametrize("name,kwargs", SPLITTERS, ids=[n for n, _ in SPLITTERS])
def test_splitter_routing_declaration_matches_sklearn(name, kwargs):
    """Only the group splitters request ``groups`` — and the rest cannot opt in.

    The negative half is the load-bearing one: ``KFold`` accepts a ``groups``
    argument and ignores it, so a requestable ``groups`` would let a caller ask
    for a grouping the splitter never applies. sklearn marks it ``UNUSED``,
    which removes both the request and the ``set_split_request`` setter.
    """
    mine = getattr(ms, name)(**kwargs)
    theirs = getattr(skm, name)(**kwargs)
    assert str(mine.get_metadata_routing()) == str(theirs.get_metadata_routing())
    assert hasattr(mine, "set_split_request") == hasattr(theirs, "set_split_request")


def routed_names(router, method):
    """The metadata ``method`` will accept, aliases resolved — the caller's view."""
    return sorted(
        router._get_param_names(
            method=method, return_alias=True, ignore_self_request=False
        )
    )


@pytest.mark.parametrize("cv", ["groups", "int"])
def test_search_routing_declaration_matches_sklearn(cv):
    """The same metadata is accepted, for the same methods, as sklearn's search.

    Compared on the accepted NAMES rather than on the router's repr: mlrs keeps
    its scorers in a dict and so publishes them through one extra nesting level
    (:func:`_scorer_router`) where sklearn publishes a single scorer object.
    The nesting is invisible to a caller; what it can request is not.
    """
    mine_cv, their_cv = (ms.GroupKFold(n_splits=2), skm.GroupKFold(n_splits=2))
    if cv == "int":
        mine_cv = their_cv = 3
    mine = ms.GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=mine_cv)
    theirs = skm.GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=their_cv)
    for method in ("fit", "score"):
        assert routed_names(mine.get_metadata_routing(), method) == routed_names(
            theirs.get_metadata_routing(), method
        )


def test_threshold_classifier_routing_declarations_match_sklearn():
    fixed_mine = ms.FixedThresholdClassifier(LogisticRegression())
    fixed_theirs = skm.FixedThresholdClassifier(LogisticRegression())
    assert str(fixed_mine.get_metadata_routing()) == str(
        fixed_theirs.get_metadata_routing()
    )

    tuned_mine = ms.TunedThresholdClassifierCV(LogisticRegression(), cv=3)
    tuned_theirs = skm.TunedThresholdClassifierCV(LogisticRegression(), cv=3)
    # sklearn's scorer node is a `_CurveScorer` and mlrs's is the scorer the
    # sweep's metric comes from, so the OWNER names differ by construction; the
    # routed request — what a caller actually has to set — must not.
    assert routed_names(tuned_mine.get_metadata_routing(), "fit") == routed_names(
        tuned_theirs.get_metadata_routing(), "fit"
    )


# --------------------------------------------------------------------------- #
# cross_validate / cross_val_score / cross_val_predict
# --------------------------------------------------------------------------- #


def test_cross_validate_routes_requested_sample_weight(
    regression, weights, routing_enabled
):
    """A requested ``sample_weight`` reaches ``fit``, sliced to the fold.

    ``set_score_request(sample_weight=False)`` is not decoration: with
    ``scoring=None`` the scorer IS the estimator's own ``score``, which also
    takes a ``sample_weight`` — leaving that request unset makes the call raise
    (in sklearn too), because "weight the fit" and "weight the score" are
    genuinely different asks.
    """
    X, y = regression
    registry = _Registry()
    estimator = (
        RecordingRegressor(registry=registry)
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=False)
    )

    ms.cross_validate(estimator, X, y, cv=4, params={"sample_weight": weights})

    assert len(registry) == 4
    for call in registry:
        assert call["metadata"] is None
        assert len(call["sample_weight"]) == call["n"] == 30
        # every weight is one of the caller's, never a re-derived default
        assert np.all(np.isin(call["sample_weight"], weights))


def test_cross_validate_sample_weight_matches_sklearn(
    regression, weights, routing_enabled
):
    X, y = regression
    estimator = Ridge().set_fit_request(sample_weight=True).set_score_request(
        sample_weight=False
    )
    mine = ms.cross_validate(estimator, X, y, cv=4, params={"sample_weight": weights})
    theirs = skm.cross_validate(
        clone(estimator), X, y, cv=4, params={"sample_weight": weights}
    )
    np.testing.assert_allclose(mine["test_score"], theirs["test_score"])


def test_cross_validate_unrequested_metadata_raises_like_sklearn(
    regression, weights, routing_enabled
):
    """Passing metadata nobody requested is an error, not a silent drop."""
    X, y = regression
    with pytest.raises(UnsetMetadataPassedError) as mine:
        ms.cross_validate(Ridge(), X, y, cv=3, params={"sample_weight": weights})
    with pytest.raises(UnsetMetadataPassedError) as theirs:
        skm.cross_validate(Ridge(), X, y, cv=3, params={"sample_weight": weights})

    assert mine.value.unrequested_params == theirs.value.unrequested_params
    # the caller ran `cross_validate`, not `cross_validate.fit` — the message
    # must not name a method they never invoked
    assert "cross_validate.fit" not in str(mine.value)
    assert "cross_validate" in str(mine.value)


def test_cross_validate_unroutable_metadata_raises_like_sklearn(regression, routing_enabled):
    """A name no consumer declares is a TypeError, not an unrequested-metadata error."""
    X, y = regression
    payload = {"nonsense": np.arange(N)}
    with pytest.raises(TypeError, match="not routed to any object"):
        ms.cross_validate(Ridge(), X, y, cv=3, params=payload)
    with pytest.raises(TypeError, match="not routed to any object"):
        skm.cross_validate(Ridge(), X, y, cv=3, params=payload)


def test_cross_validate_routes_groups_to_a_group_splitter(regression, groups, routing_enabled):
    """``groups`` in ``params`` reaches a group splitter and picks the same folds."""
    X, y = regression
    mine = ms.cross_validate(
        Ridge(), X, y, cv=ms.GroupKFold(n_splits=4), params={"groups": groups},
        return_indices=True,
    )
    theirs = skm.cross_validate(
        Ridge(), X, y, cv=skm.GroupKFold(n_splits=4), params={"groups": groups},
        return_indices=True,
    )
    for a, b in zip(mine["indices"]["test"], theirs["indices"]["test"]):
        np.testing.assert_array_equal(np.sort(a), np.sort(b))
    np.testing.assert_allclose(mine["test_score"], theirs["test_score"])


@pytest.mark.parametrize(
    "call",
    [
        lambda mod, X, y, g: mod.cross_validate(Ridge(), X, y, cv=3, groups=g),
        lambda mod, X, y, g: mod.cross_val_score(Ridge(), X, y, cv=3, groups=g),
        lambda mod, X, y, g: mod.cross_val_predict(Ridge(), X, y, cv=3, groups=g),
        lambda mod, X, y, g: mod.learning_curve(Ridge(), X, y, cv=3, groups=g),
        lambda mod, X, y, g: mod.validation_curve(
            Ridge(), X, y, param_name="alpha", param_range=[0.1, 1.0], cv=3, groups=g
        ),
        lambda mod, X, y, g: mod.permutation_test_score(
            Ridge(), X, y, cv=3, groups=g, n_permutations=2
        ),
    ],
    ids=["cross_validate", "cross_val_score", "cross_val_predict", "learning_curve",
         "validation_curve", "permutation_test_score"],
)
def test_groups_argument_is_refused_while_routing_is_enabled(
    regression, groups, routing_enabled, call
):
    """``groups=`` and routed ``groups`` are two sources for one input."""
    X, y = regression
    with pytest.raises(ValueError, match="can only be passed if metadata routing"):
        call(ms, X, y, groups)
    with pytest.raises(ValueError, match="can only be passed if metadata routing"):
        call(skm, X, y, groups)


def test_cross_val_predict_has_no_scorer_to_route_to(
    regression, weights, routing_enabled
):
    """Nothing is scored here, so the router has no scorer node — as in sklearn.

    The estimator is one whose ``fit`` knows no metadata, so a ``sample_weight``
    has nowhere left to go: under ``cross_validate`` the scorer would still be a
    candidate consumer and the error would name it instead.
    """
    X, y = regression
    scorer_only = {"sample_weight": weights}
    with pytest.raises(TypeError, match="not routed to any object"):
        ms.cross_val_predict(PlainRegressor(), X, y, cv=3, params=scorer_only)
    with pytest.raises(TypeError, match="not routed to any object"):
        skm.cross_val_predict(PlainRegressor(), X, y, cv=3, params=scorer_only)


def test_cross_val_predict_routes_fit_metadata(regression, weights, routing_enabled):
    X, y = regression
    estimator = Ridge().set_fit_request(sample_weight=True)
    np.testing.assert_allclose(
        ms.cross_val_predict(estimator, X, y, cv=4, params={"sample_weight": weights}),
        skm.cross_val_predict(estimator, X, y, cv=4, params={"sample_weight": weights}),
    )


# --------------------------------------------------------------------------- #
# scorers
# --------------------------------------------------------------------------- #


def test_scorer_metadata_is_routed_to_the_scorer(regression, weights, routing_enabled):
    """A scorer's request is honoured independently of the estimator's."""
    X, y = regression
    registry = _Registry()
    scorer = make_scorer(recording_metric(registry)).set_score_request(sample_weight=True)
    estimator = Ridge().set_fit_request(sample_weight=False)

    ms.cross_validate(
        estimator, X, y, cv=4, scoring=scorer, params={"sample_weight": weights}
    )

    assert len(registry) == 4  # test folds only; no train scores requested
    for call in registry:
        assert call["n"] == 10
        assert len(call["sample_weight"]) == 10


def test_scorer_metadata_matches_sklearn(regression, weights, routing_enabled):
    X, y = regression
    scorer = make_scorer(r2_score).set_score_request(sample_weight=True)
    estimator = Ridge().set_fit_request(sample_weight=False)
    payload = {"sample_weight": weights}
    np.testing.assert_allclose(
        ms.cross_validate(estimator, X, y, cv=4, scoring=scorer, params=payload)["test_score"],
        skm.cross_validate(clone(estimator), X, y, cv=4, scoring=scorer, params=payload)[
            "test_score"
        ],
    )


def test_two_scorers_get_only_what_each_requested(regression, weights, routing_enabled):
    """The union is routed to the scorers; the split back out is per scorer."""
    X, y = regression
    weighted_log, plain_log = _Registry(), _Registry()
    scoring = {
        "weighted": make_scorer(recording_metric(weighted_log)).set_score_request(
            sample_weight=True
        ),
        # explicitly declined, not merely unset: an unset request would make the
        # whole call raise, which is the mechanism working as intended
        "plain": make_scorer(recording_metric(plain_log)).set_score_request(
            sample_weight=False
        ),
    }
    estimator = Ridge().set_fit_request(sample_weight=False)
    ms.cross_validate(
        estimator, X, y, cv=4, scoring=scoring, params={"sample_weight": weights}
    )

    assert [call["sample_weight"] is not None for call in weighted_log] == [True] * 4
    assert [call["sample_weight"] is None for call in plain_log] == [True] * 4


@pytest.mark.parametrize("multimetric", [False, True])
def test_unset_scorer_request_raises_instead_of_becoming_a_nan(
    regression, weights, routing_enabled, multimetric
):
    """An unset request is a caller error, so ``error_score`` must not absorb it.

    A scorer that raises mid-fold is reported as ``error_score`` — that is the
    point of ``error_score``. A scorer that was handed metadata it never asked
    for has not failed on the data; silently scoring the whole run as NaN would
    hide a routing mistake behind a plausible-looking result.
    """
    X, y = regression
    estimator = Ridge().set_fit_request(sample_weight=False)
    scoring = make_scorer(r2_score)  # knows `sample_weight`, requests nothing
    if multimetric:
        scoring = {
            "asked": make_scorer(r2_score).set_score_request(sample_weight=True),
            "did_not": scoring,
        }
    for mod in (ms, skm):
        with pytest.raises(UnsetMetadataPassedError):
            mod.cross_validate(
                estimator,
                X,
                y,
                cv=3,
                scoring=scoring,
                params={"sample_weight": weights},
                error_score=np.nan,
            )


def test_estimator_and_scorer_request_the_same_name(regression, weights, routing_enabled):
    """One ``sample_weight`` can legitimately reach two consumers at once."""
    X, y = regression
    fit_log, score_log = _Registry(), _Registry()
    estimator = RecordingRegressor(registry=fit_log).set_fit_request(sample_weight=True)
    scorer = make_scorer(recording_metric(score_log)).set_score_request(sample_weight=True)

    ms.cross_validate(
        estimator, X, y, cv=4, scoring=scorer, params={"sample_weight": weights}
    )

    assert len(fit_log) == len(score_log) == 4
    assert all(call["n"] == 30 for call in fit_log)  # train rows
    assert all(call["n"] == 10 for call in score_log)  # test rows


# --------------------------------------------------------------------------- #
# learning_curve / validation_curve / permutation_test_score
# --------------------------------------------------------------------------- #


def test_learning_curve_routes_like_sklearn(regression, weights, routing_enabled):
    X, y = regression
    estimator = Ridge().set_fit_request(sample_weight=True).set_score_request(
        sample_weight=False
    )
    payload = {"sample_weight": weights}
    mine = ms.learning_curve(estimator, X, y, cv=4, params=payload)
    theirs = skm.learning_curve(estimator, X, y, cv=4, params=payload)
    for a, b in zip(mine, theirs):
        np.testing.assert_allclose(a, b)


def test_learning_curve_routes_groups(regression, groups, routing_enabled):
    X, y = regression
    mine = ms.learning_curve(
        Ridge(), X, y, cv=ms.GroupKFold(n_splits=4), params={"groups": groups}
    )
    theirs = skm.learning_curve(
        Ridge(), X, y, cv=skm.GroupKFold(n_splits=4), params={"groups": groups}
    )
    for a, b in zip(mine, theirs):
        np.testing.assert_allclose(a, b)


def test_validation_curve_routes_like_sklearn(regression, weights, routing_enabled):
    X, y = regression
    estimator = Ridge().set_fit_request(sample_weight=True).set_score_request(
        sample_weight=False
    )
    payload = {"sample_weight": weights}
    mine = ms.validation_curve(
        estimator, X, y, param_name="alpha", param_range=[0.1, 1.0], cv=4, params=payload
    )
    theirs = skm.validation_curve(
        estimator, X, y, param_name="alpha", param_range=[0.1, 1.0], cv=4, params=payload
    )
    for a, b in zip(mine, theirs):
        np.testing.assert_allclose(a, b)


def test_permutation_test_score_routes_like_sklearn(regression, weights, routing_enabled):
    X, y = regression
    estimator = Ridge().set_fit_request(sample_weight=True).set_score_request(
        sample_weight=False
    )
    payload = {"sample_weight": weights}
    score_mine, _, _ = ms.permutation_test_score(
        estimator, X, y, cv=3, n_permutations=3, random_state=0, params=payload
    )
    score_theirs, _, _ = skm.permutation_test_score(
        estimator, X, y, cv=3, n_permutations=3, random_state=0, params=payload
    )
    np.testing.assert_allclose(score_mine, score_theirs)


def test_permutation_test_score_constrains_the_shuffle_by_routed_groups(regression, groups):
    """SK-002: routed ``groups`` bound the permutation, as ``groups=`` does.

    scikit-learn refuses ``groups=`` under routing and then permutes globally,
    so its grouped permutation test silently loses its grouping — the two
    spellings of the same input stop meaning the same thing. mlrs keeps them
    equivalent: the routed run reproduces the ``groups=``-with-routing-off run
    exactly. Both halves are pinned, so an upstream fix fails here loudly.
    """
    X, y = regression
    cv = ms.GroupKFold(n_splits=4)
    _, off_scores, _ = ms.permutation_test_score(
        Ridge(), X, y, cv=cv, groups=groups, n_permutations=5, random_state=7
    )
    with sklearn.config_context(enable_metadata_routing=True):
        _, on_scores, _ = ms.permutation_test_score(
            Ridge(), X, y, cv=cv, n_permutations=5, random_state=7,
            params={"groups": groups},
        )
        _, sk_on_scores, _ = skm.permutation_test_score(
            Ridge(), X, y, cv=skm.GroupKFold(n_splits=4), n_permutations=5,
            random_state=7, params={"groups": groups},
        )
    np.testing.assert_allclose(on_scores, off_scores)
    # ...and sklearn's routed run does NOT reproduce its own grouped run:
    _, sk_off_scores, _ = skm.permutation_test_score(
        Ridge(), X, y, cv=skm.GroupKFold(n_splits=4), groups=groups,
        n_permutations=5, random_state=7,
    )
    assert not np.allclose(sk_on_scores, sk_off_scores)


# --------------------------------------------------------------------------- #
# the search estimators
# --------------------------------------------------------------------------- #


def test_grid_search_routes_sample_weight_to_fit(regression, weights, routing_enabled):
    X, y = regression
    registry = _Registry()
    estimator = (
        RecordingRegressor(registry=registry)
        .set_fit_request(sample_weight=True)
        .set_score_request(sample_weight=False)
    )
    search = ms.GridSearchCV(estimator, {"registry": [registry]}, cv=4)
    search.fit(X, y, sample_weight=weights)

    # 4 folds + the refit on the whole dataset
    assert len(registry) == 5
    assert [call["n"] for call in registry] == [30, 30, 30, 30, 40]
    assert all(len(call["sample_weight"]) == call["n"] for call in registry)


def test_grid_search_sample_weight_matches_sklearn(regression, weights, routing_enabled):
    X, y = regression
    grid = {"alpha": [0.01, 0.1, 1.0]}
    estimator = Ridge().set_fit_request(sample_weight=True).set_score_request(
        sample_weight=False
    )
    mine = ms.GridSearchCV(estimator, grid, cv=4).fit(X, y, sample_weight=weights)
    theirs = skm.GridSearchCV(clone(estimator), grid, cv=4).fit(
        X, y, sample_weight=weights
    )
    assert mine.best_params_ == theirs.best_params_
    np.testing.assert_allclose(mine.best_score_, theirs.best_score_)


def test_grid_search_routes_groups_to_the_splitter(regression, groups, routing_enabled):
    X, y = regression
    grid = {"alpha": [0.01, 1.0]}
    mine = ms.GridSearchCV(Ridge(), grid, cv=ms.GroupKFold(n_splits=4)).fit(
        X, y, groups=groups
    )
    theirs = skm.GridSearchCV(Ridge(), grid, cv=skm.GroupKFold(n_splits=4)).fit(
        X, y, groups=groups
    )
    assert mine.best_params_ == theirs.best_params_
    np.testing.assert_allclose(
        mine.cv_results_["mean_test_score"], theirs.cv_results_["mean_test_score"]
    )


def test_halving_search_routes_groups_to_the_splitter(classification, groups, routing_enabled):
    X, y = classification
    grid = {"C": [0.1, 1.0, 10.0]}
    mine = ms.HalvingGridSearchCV(
        LogisticRegression(), grid, cv=ms.GroupKFold(n_splits=2), factor=2,
        min_resources=16, random_state=0,
    ).fit(X, y, groups=groups)
    theirs = skm.HalvingGridSearchCV(
        LogisticRegression(), grid, cv=skm.GroupKFold(n_splits=2), factor=2,
        min_resources=16, random_state=0,
    ).fit(X, y, groups=groups)
    assert mine.n_resources_ == theirs.n_resources_
    assert mine.n_candidates_ == theirs.n_candidates_


def test_grid_search_unrequested_metadata_raises_like_sklearn(
    regression, weights, routing_enabled
):
    X, y = regression
    grid = {"alpha": [0.1, 1.0]}
    with pytest.raises(UnsetMetadataPassedError) as mine:
        ms.GridSearchCV(Ridge(), grid, cv=3).fit(X, y, sample_weight=weights)
    with pytest.raises(UnsetMetadataPassedError) as theirs:
        skm.GridSearchCV(Ridge(), grid, cv=3).fit(X, y, sample_weight=weights)
    assert mine.value.unrequested_params == theirs.value.unrequested_params


def test_grid_search_score_routes_scorer_metadata(regression, weights, routing_enabled):
    X, y = regression
    scorer = make_scorer(r2_score).set_score_request(sample_weight=True)
    grid = {"alpha": [0.1, 1.0]}
    mine = ms.GridSearchCV(Ridge(), grid, cv=3, scoring=scorer).fit(X, y)
    theirs = skm.GridSearchCV(Ridge(), grid, cv=3, scoring=scorer).fit(X, y)
    np.testing.assert_allclose(
        mine.score(X, y, sample_weight=weights), theirs.score(X, y, sample_weight=weights)
    )


def test_grid_search_score_refuses_metadata_the_scorer_did_not_request(
    regression, weights, routing_enabled
):
    """``score`` enforces the scorer's request rather than forwarding blindly.

    The single-scorer node IS the scorer (see ``_scorer_router``), so an unset
    request is caught here — where the caller can act on it — instead of
    surfacing as an unexpected keyword deep inside the metric.
    """
    X, y = regression
    scorer = make_scorer(recording_metric(_Registry()))  # requests nothing
    mine = ms.GridSearchCV(Ridge(), {"alpha": [0.1]}, cv=3, scoring=scorer).fit(X, y)
    theirs = skm.GridSearchCV(Ridge(), {"alpha": [0.1]}, cv=3, scoring=scorer).fit(X, y)
    with pytest.raises(UnsetMetadataPassedError):
        mine.score(X, y, sample_weight=weights)
    with pytest.raises(UnsetMetadataPassedError):
        theirs.score(X, y, sample_weight=weights)


# --------------------------------------------------------------------------- #
# the routing-DISABLED path (the default) is unchanged
# --------------------------------------------------------------------------- #


def test_disabled_routing_still_forwards_params_to_fit(regression, weights):
    """No request set, routing off: ``params`` goes to ``fit`` wholesale."""
    X, y = regression
    registry = _Registry()
    ms.cross_validate(
        RecordingRegressor(registry=registry), X, y, cv=4,
        params={"sample_weight": weights},
    )
    assert len(registry) == 4
    assert all(len(call["sample_weight"]) == 30 for call in registry)


def test_disabled_routing_search_forwards_sample_weight_to_the_scorer(
    regression, weights
):
    """sklearn's legacy special case: fit weights are also scoring weights."""
    X, y = regression
    grid = {"alpha": [0.01, 0.1, 1.0]}
    mine = ms.GridSearchCV(Ridge(), grid, cv=4).fit(X, y, sample_weight=weights)
    theirs = skm.GridSearchCV(Ridge(), grid, cv=4).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(
        mine.cv_results_["mean_test_score"], theirs.cv_results_["mean_test_score"]
    )
    assert mine.best_params_ == theirs.best_params_


def test_disabled_routing_search_warns_when_the_scorer_ignores_weights(classification):
    """Fitting on weights while scoring without them is a silent statistical error."""
    X, y = classification
    scorer = make_scorer(lambda yt, yp: accuracy_score(yt, yp))  # no sample_weight
    search = ms.GridSearchCV(
        LogisticRegression(), {"C": [0.1, 1.0]}, cv=3, scoring=scorer
    )
    with pytest.warns(UserWarning, match="does not support sample_weight"):
        search.fit(X, y, sample_weight=np.linspace(0.5, 1.5, len(y)))


def test_disabled_routing_still_accepts_groups_positionally(regression, groups):
    X, y = regression
    mine = ms.cross_val_score(Ridge(), X, y, cv=ms.GroupKFold(n_splits=4), groups=groups)
    theirs = skm.cross_val_score(
        Ridge(), X, y, cv=skm.GroupKFold(n_splits=4), groups=groups
    )
    np.testing.assert_allclose(mine, theirs)


# --------------------------------------------------------------------------- #
# the threshold classifiers
# --------------------------------------------------------------------------- #


def test_fixed_threshold_classifier_routes_fit_metadata(
    classification, weights, routing_enabled
):
    X, y = classification
    registry = _Registry()
    estimator = RecordingClassifier(registry=registry).set_fit_request(
        sample_weight=True
    )
    ms.FixedThresholdClassifier(estimator, threshold=0.5).fit(X, y, sample_weight=weights)
    assert len(registry) == 1
    np.testing.assert_allclose(registry[0]["sample_weight"], weights)


def test_fixed_threshold_classifier_rejects_unrequested_metadata(
    classification, weights, routing_enabled
):
    X, y = classification
    with pytest.raises(UnsetMetadataPassedError):
        ms.FixedThresholdClassifier(LogisticRegression()).fit(X, y, sample_weight=weights)
    with pytest.raises(UnsetMetadataPassedError):
        skm.FixedThresholdClassifier(LogisticRegression()).fit(X, y, sample_weight=weights)


def test_tuned_threshold_routes_metadata_to_fit_and_to_the_metric(
    classification, weights, routing_enabled
):
    """The estimator's and the metric's buckets are indexed to different rows."""
    X, y = classification
    fit_log, score_log = _Registry(), _Registry()
    estimator = LogisticRegression().set_fit_request(sample_weight=True)

    def metric(y_true, y_pred, sample_weight=None):
        score_log.append({"n": len(y_true), "sample_weight": sample_weight})
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    tuned = ms.TunedThresholdClassifierCV(
        estimator,
        scoring=make_scorer(metric).set_score_request(sample_weight=True),
        cv=4,
        thresholds=5,
    )
    tuned.fit(X, y, sample_weight=weights)

    # 5 thresholds x 4 folds of 10 validation rows each
    assert len(score_log) == 20
    assert all(
        call["n"] == 10 and len(call["sample_weight"]) == 10 for call in score_log
    )
    assert 0.0 <= tuned.best_score_ <= 1.0


def test_tuned_threshold_disabled_routing_indexes_fit_params_per_fold(
    classification, weights
):
    """Row-aligned ``params`` are sliced to the fold even with routing off."""
    X, y = classification
    registry = _Registry()
    ms.TunedThresholdClassifierCV(
        RecordingClassifier(registry=registry), scoring="accuracy", cv=4, thresholds=5
    ).fit(X, y, sample_weight=weights)
    # 4 fold fits of 30 rows, then the refit on all 40
    assert [call["n"] for call in registry] == [30, 30, 30, 30, 40]
    assert all(len(call["sample_weight"]) == call["n"] for call in registry)


def test_deepcopy_of_a_routed_splitter_keeps_its_request():
    """A request must survive the clone/deepcopy a search subjects it to."""
    splitter = ms.GroupKFold(n_splits=3)
    assert str(copy.deepcopy(splitter).get_metadata_routing()) == str(
        splitter.get_metadata_routing()
    )
