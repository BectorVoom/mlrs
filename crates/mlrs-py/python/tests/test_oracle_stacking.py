"""StackingRegressor oracle harness (STACK-01: the full parameter surface).

Every test here compares :class:`mlrs.StackingRegressor` against a LIVE
:class:`sklearn.ensemble.StackingRegressor` built with the same arguments, on
the same data, in the same process. There are no committed ``.npz`` fixtures:
stacking has no numerics of its own — it composes other estimators — so what
needs gating is the *composition*, and the only reference for that is sklearn
itself. A stored blob would freeze one sklearn version's answer where a live
comparison tracks it.

## The string-valued parameter surface (the point of this file)

``StackingRegressor``'s declared constraints leave exactly two places a caller
supplies a STRING, and both change what ``fit`` does rather than merely how
fast it does it:

  ============================  ==========================================
  string                        what it selects
  ============================  ==========================================
  ``cv="prefit"``               skip cloning + refitting; meta features are
                                the base estimators' FULL-training-set
                                predictions, not out-of-fold ones
  ``estimators=[(name, "drop")]``  disable one entry: no fit, no meta column,
                                but the slot survives in
                                ``named_estimators_`` as the string
                                ``'drop'``
  ============================  ==========================================

Both are oracle-tested here in every combination that can interact with them
(``passthrough``, a surviving sibling, ``get_feature_names_out``,
``named_estimators_``, ``n_features_in_``), together with the rejection paths —
an unrecognized ``cv`` string, and an all-``'drop'`` list. The rejection
messages are asserted to be sklearn's own text, not merely "some ValueError":
they are the same strings the Rust core owns
(``crates/mlrs-algos/tests/stacking_test.rs``), so a divergence in either layer
fails here.

``n_jobs``, ``passthrough``, ``verbose``, ``cv`` as an int/splitter/iterable and
``final_estimator`` are covered too — those are the rest of the parameter
surface, and a value-neutral parameter that silently changes the answer is
exactly the defect an oracle suite exists to catch.

## Backend gating

Two designs run side by side:

* **sklearn-only sub-estimators** — pure host composition. The stacking layer
  itself is dtype-independent host bookkeeping, so these cells run identically
  (and must be EXACT, ``atol=0``) on cpu / wgpu / rocm / cuda. This is where
  the string-parameter coverage lives, so no backend can end up with a vacuous
  run of it.
* **mlrs sub-estimators** — the real deployment shape, where the base fits go to
  the device. These use ``conftest.default_float_dtype()`` /
  ``conftest.live_atol()`` rather than hardcoding float64, which is what keeps
  them from turning red at ingress on an f64-incapable backend (rocm / cuda)
  instead of comparing anything.

Req: STACK-01 (parameter surface), STACK-BIND-01 (the Rust structural core).
"""

import re

import numpy as np
import pytest
from sklearn.ensemble import StackingRegressor as SkStackingRegressor
from sklearn.linear_model import (
    LinearRegression as SkLinearRegression,
    LogisticRegression as SkLogisticRegression,
    Ridge as SkRidge,
)

import conftest

mlrs = pytest.importorskip("mlrs")


# --------------------------------------------------------------------------- #
# designs
# --------------------------------------------------------------------------- #

N_SAMPLES = 200
N_FEATURES = 5
SEED = 42


def host_design(dtype=np.float64, n_samples=N_SAMPLES):
    """A well-conditioned linear regression problem, host-side float64.

    Used with sklearn sub-estimators, so no mlrs ingress is involved and the
    dtype needs no backend gate.
    """
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, N_FEATURES)).astype(dtype)
    y = (X @ np.array([3.0, -1.5, 0.0, 0.75, 2.0], dtype=dtype)).astype(dtype)
    y = y + (0.05 * rng.standard_normal(n_samples)).astype(dtype)
    return X, y


def device_design():
    """The same problem at the BACKEND's float dtype, for mlrs sub-estimators."""
    return host_design(dtype=conftest.default_float_dtype())


def sk_estimators():
    """Two sklearn base regressors — the reference composition."""
    return [("lr", SkLinearRegression()), ("ridge", SkRidge(alpha=1.0))]


def mlrs_estimators():
    """The same two, as mlrs device estimators."""
    return [("lr", mlrs.LinearRegression()), ("ridge", mlrs.Ridge(alpha=1.0))]


def both(**kwargs):
    """``(mlrs_estimator, sklearn_estimator)`` built from identical arguments."""
    estimators = kwargs.pop("estimators", None)
    return (
        mlrs.StackingRegressor(
            estimators if estimators is not None else sk_estimators(), **kwargs
        ),
        SkStackingRegressor(
            estimators if estimators is not None else sk_estimators(), **kwargs
        ),
    )


def assert_same_fit(a, b, X, y, *, atol=0.0):
    """mlrs and sklearn agree on everything a fitted stack exposes."""
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=atol, rtol=0)
    np.testing.assert_allclose(a.transform(X), b.transform(X), atol=atol, rtol=0)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
    assert list(a.named_estimators_) == list(b.named_estimators_)
    assert a.stack_method_ == b.stack_method_
    assert a.n_features_in_ == b.n_features_in_
    assert a._n_feature_outs == b._n_feature_outs
    return a, b


# =========================================================================== #
# STRING PARAMETER 1 — cv="prefit"
# =========================================================================== #


def _prefit_estimators(X, y):
    """Two base regressors ALREADY fitted, as ``cv="prefit"`` requires."""
    return [
        ("lr", SkLinearRegression().fit(X, y)),
        ("ridge", SkRidge(alpha=1.0).fit(X, y)),
    ]


def test_cv_prefit_matches_sklearn_exactly():
    """``cv="prefit"``: same predictions, same meta features, same names."""
    X, y = host_design()
    fitted = _prefit_estimators(X, y)
    a, b = both(estimators=fitted, cv="prefit")
    assert_same_fit(a, b, X, y)


def test_cv_prefit_does_not_refit_the_given_estimators():
    """The caller's own objects are reused, not cloned — sklearn's contract.

    This is the observable difference between ``"prefit"`` and every other
    ``cv``: ``estimators_[i] is estimators[i][1]``. If mlrs cloned here, a
    caller's expensively-pretrained model would be silently refitted on the
    stacking data.
    """
    X, y = host_design()
    fitted = _prefit_estimators(X, y)
    coef_before = fitted[0][1].coef_.copy()

    a, b = both(estimators=fitted, cv="prefit")
    a.fit(X, y)
    b.fit(X, y)

    assert a.estimators_[0] is fitted[0][1]
    assert b.estimators_[0] is fitted[0][1]
    np.testing.assert_array_equal(fitted[0][1].coef_, coef_before)


def test_cv_prefit_uses_full_training_predictions_not_out_of_fold():
    """``"prefit"`` meta features differ from the 5-fold ones, as they must.

    A test that only asserted mlrs == sklearn would pass even if BOTH silently
    ignored the string. This pins the semantic difference itself: the in-sample
    predictions of a fitted base estimator are strictly closer to ``y`` than its
    out-of-fold ones.

    The base estimator is a 1-nearest-neighbour regressor precisely because it
    makes the gap unmissable — fitted on all of ``X`` its in-sample prediction is
    each row's own target, while its out-of-fold prediction is a genuinely
    different row's. A well-conditioned OLS base would NOT work here: on a linear
    problem its in-sample and out-of-fold predictions agree to ~15 digits, so the
    test would compare two numbers that are equal for reasons unrelated to
    whether ``"prefit"`` was honoured.

    Note WHERE the difference is observable. ``transform`` is identical on both
    routes — it always re-predicts through ``estimators_``, which under ``cv=5``
    are refitted on the full ``X`` — so the route shows up only in what
    ``final_estimator_`` was TRAINED on, i.e. in its coefficients and in the
    resulting predictions. Asserting on ``transform`` would have passed
    vacuously.
    """
    from sklearn.neighbors import KNeighborsRegressor

    X, y = host_design()
    fitted = [("nn", KNeighborsRegressor(n_neighbors=1).fit(X, y))]

    prefit = mlrs.StackingRegressor(fitted, cv="prefit").fit(X, y)
    kfold = mlrs.StackingRegressor(
        [("nn", KNeighborsRegressor(n_neighbors=1))], cv=5
    ).fit(X, y)

    assert not np.allclose(prefit.predict(X), kfold.predict(X))
    # `prefit` trained the meta learner on y itself, so it learns the identity;
    # `cv=5` trained it on noisier out-of-fold neighbours, so it needs a larger
    # coefficient to reach the same targets.
    assert prefit.final_estimator_.coef_[0] == pytest.approx(1.0, abs=1e-3)
    assert kfold.final_estimator_.coef_[0] > prefit.final_estimator_.coef_[0] + 0.05
    # The documented `"prefit"` hazard, made visible: a training-set score that
    # flatters the stack because the meta learner already saw these targets.
    assert prefit.score(X, y) > kfold.score(X, y)


def test_cv_prefit_with_an_unfitted_estimator_raises_not_fitted():
    """sklearn checks each entry with ``check_is_fitted`` before using it."""
    from sklearn.exceptions import NotFittedError

    X, y = host_design()
    a, b = both(estimators=sk_estimators(), cv="prefit")
    with pytest.raises(NotFittedError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(NotFittedError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


@pytest.mark.parametrize("passthrough", [False, True])
def test_cv_prefit_composes_with_passthrough(passthrough):
    """The two independent structural switches do not interfere."""
    X, y = host_design()
    fitted = _prefit_estimators(X, y)
    a, b = both(estimators=fitted, cv="prefit", passthrough=passthrough)
    assert_same_fit(a, b, X, y)
    expected_width = 2 + (N_FEATURES if passthrough else 0)
    assert a.transform(X).shape == (N_SAMPLES, expected_width)


@pytest.mark.parametrize("bad", ["Prefit", "PREFIT", "prefit ", "auto", ""])
def test_cv_rejects_every_other_string_with_sklearns_message(bad):
    """Only the exact literal ``"prefit"`` is accepted; the rest raise sklearn's
    ``StrOptions`` text, as ``InvalidParameterError`` (both a ``ValueError`` and
    a ``TypeError``, so either ``except`` clause catches it)."""
    from sklearn.utils._param_validation import (
        InvalidParameterError as SkInvalidParameterError,
    )

    X, y = host_design()
    a, b = both(cv=bad)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(SkInvalidParameterError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert isinstance(mlrs_exc.value, TypeError)


def test_cv_prefit_is_reported_verbatim_by_get_params():
    """``get_params`` round-trips the string — ``clone`` must not normalize it."""
    from sklearn.base import clone

    est = mlrs.StackingRegressor(sk_estimators(), cv="prefit")
    assert est.get_params()["cv"] == "prefit"
    assert clone(est).get_params()["cv"] == "prefit"


# =========================================================================== #
# STRING PARAMETER 2 — an estimator set to 'drop'
# =========================================================================== #


@pytest.mark.parametrize("dropped", ["lr", "ridge"])
def test_drop_via_constructor_matches_sklearn(dropped):
    """A ``'drop'`` entry written directly into ``estimators``."""
    X, y = host_design()
    estimators = [
        (name, "drop" if name == dropped else est) for name, est in sk_estimators()
    ]
    a, b = both(estimators=estimators, cv=3)
    assert_same_fit(a, b, X, y)
    assert a.transform(X).shape == (N_SAMPLES, 1)


@pytest.mark.parametrize("dropped", ["lr", "ridge"])
def test_drop_via_set_params_matches_sklearn(dropped):
    """``set_params(name="drop")`` — the documented way to disable an entry."""
    X, y = host_design()
    a, b = both(cv=3)
    a.set_params(**{dropped: "drop"})
    b.set_params(**{dropped: "drop"})
    assert_same_fit(a, b, X, y)


def test_dropped_slot_survives_in_named_estimators_as_the_string():
    """A dropped entry keeps its name, mapped to the literal ``'drop'``.

    It is absent from ``estimators_`` (which holds fitted objects only) but
    present in ``named_estimators_`` — the asymmetry is sklearn's, and code that
    iterates one or the other depends on it.
    """
    X, y = host_design()
    a, b = both(cv=3)
    a.set_params(lr="drop")
    b.set_params(lr="drop")
    a.fit(X, y)
    b.fit(X, y)

    assert a.named_estimators_["lr"] == "drop"
    assert b.named_estimators_["lr"] == "drop"
    assert list(a.named_estimators_) == ["lr", "ridge"]
    assert len(a.estimators_) == 1
    assert len(a.stack_method_) == 1


def test_dropped_entry_contributes_no_meta_column_or_name():
    """The meta matrix and ``get_feature_names_out`` both lose exactly one slot."""
    X, y = host_design()
    a, b = both(cv=3)
    full = a.fit(X, y).transform(X).shape[1]
    a.set_params(ridge="drop")
    b.set_params(ridge="drop")
    a.fit(X, y)
    b.fit(X, y)
    assert a.transform(X).shape[1] == full - 1
    assert list(a.get_feature_names_out()) == ["stackingregressor_lr"]
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


def test_drop_composes_with_passthrough():
    """Dropping an estimator does not disturb the passthrough block's position."""
    X, y = host_design()
    a, b = both(cv=3, passthrough=True)
    a.set_params(lr="drop")
    b.set_params(lr="drop")
    assert_same_fit(a, b, X, y)
    assert list(a.get_feature_names_out()) == [
        "stackingregressor_ridge",
        *[f"x{i}" for i in range(N_FEATURES)],
    ]


def test_all_estimators_dropped_raises_sklearns_message():
    """Nothing left to stack — sklearn's exact text, from the Rust core."""
    X, y = host_design()
    estimators = [(name, "drop") for name, _ in sk_estimators()]
    a, b = both(estimators=estimators)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert str(mlrs_exc.value) == (
        "All estimators are dropped. At least one is required to be an estimator."
    )


def test_drop_is_the_only_accepted_string_estimator():
    """Any OTHER string in an estimator slot is a non-estimator, not a sentinel.

    sklearn does not special-case it either — the string falls through to the
    is-it-a-regressor test and trips sklearn's tag lookup, raising
    ``AttributeError`` rather than ``ValueError``. mlrs reproduces that exactly
    (same class, same text) rather than "improving" it into a ValueError, since
    a caller's ``except`` clause is what would break.
    """
    X, y = host_design()
    a, b = both(estimators=[("lr", SkLinearRegression()), ("bad", "skip")])
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    assert type(mlrs_exc.value) is type(sk_exc.value)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_dropping_then_restoring_round_trips():
    """``set_params`` back to an estimator restores the column and the fit."""
    X, y = host_design()
    est = mlrs.StackingRegressor(sk_estimators(), cv=3)
    baseline = est.fit(X, y).predict(X)
    est.set_params(lr="drop")
    est.fit(X, y)
    est.set_params(lr=SkLinearRegression())
    np.testing.assert_allclose(est.fit(X, y).predict(X), baseline, atol=0, rtol=0)


# =========================================================================== #
# name validation — the remaining sklearn-parity error strings
# =========================================================================== #


@pytest.mark.parametrize(
    "estimators",
    [
        pytest.param(
            [("a", SkLinearRegression()), ("a", SkRidge())], id="duplicate-names"
        ),
        pytest.param([("cv", SkLinearRegression())], id="collides-with-ctor-arg"),
        pytest.param([("verbose", SkLinearRegression())], id="collides-with-verbose"),
        pytest.param([("a__b", SkLinearRegression())], id="contains-double-underscore"),
        pytest.param([], id="empty-list"),
        pytest.param([SkLinearRegression()], id="not-a-tuple"),
        pytest.param([(0, SkLinearRegression())], id="non-string-name"),
    ],
)
def test_invalid_estimators_raise_sklearns_exact_message(estimators):
    """Every structural rejection carries sklearn's own text, verbatim."""
    X, y = host_design()
    a, b = both(estimators=estimators)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_a_classifier_base_estimator_is_rejected():
    """Stacking a REGRESSOR requires regressor members."""
    X, y = host_design()
    a, b = both(estimators=[("clf", SkLogisticRegression())])
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_a_classifier_final_estimator_is_rejected():
    X, y = host_design()
    a, b = both(final_estimator=SkLogisticRegression())
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


# =========================================================================== #
# the non-string parameter surface
# =========================================================================== #


def test_default_final_estimator_is_sklearn_ridgecv():
    """``final_estimator=None`` resolves to sklearn's own ``RidgeCV()``.

    mlrs ships no ``RidgeCV``, and substituting ``Ridge(alpha=1.0)`` would give
    every default-constructed stack different predictions from the sklearn
    baseline users migrate from. The exact-equality assertion below is what
    makes that a contract rather than a comment.
    """
    from sklearn.linear_model import RidgeCV

    X, y = host_design()
    a, b = both()
    a.fit(X, y)
    b.fit(X, y)
    assert isinstance(a.final_estimator_, RidgeCV)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=0, rtol=0)


@pytest.mark.parametrize("cv", [2, 3, 5, 10])
def test_cv_integer_folds_match_sklearn(cv):
    """An int ``cv`` goes through mlrs's Rust ``KFold``; sklearn uses its own.

    They agree index-for-index (the model_selection parity contract), so the
    stacked predictions must agree bit-for-bit — this is the test that would
    catch a fold-boundary off-by-one on either side.
    """
    X, y = host_design()
    a, b = both(cv=cv)
    assert_same_fit(a, b, X, y)


def test_cv_none_is_five_fold():
    """``cv=None`` is 5-fold KFold — the same answer as ``cv=5``."""
    X, y = host_design()
    default = mlrs.StackingRegressor(sk_estimators(), cv=None).fit(X, y)
    five = mlrs.StackingRegressor(sk_estimators(), cv=5).fit(X, y)
    np.testing.assert_allclose(default.predict(X), five.predict(X), atol=0, rtol=0)


def test_cv_splitter_object_matches_sklearn():
    """An mlrs splitter instance is accepted and gives sklearn's answer."""
    from sklearn.model_selection import KFold as SkKFold

    from mlrs.model_selection import KFold

    X, y = host_design()
    a = mlrs.StackingRegressor(
        sk_estimators(), cv=KFold(n_splits=4, shuffle=True, random_state=0)
    )
    b = SkStackingRegressor(
        sk_estimators(), cv=SkKFold(n_splits=4, shuffle=True, random_state=0)
    )
    assert_same_fit(a, b, X, y)


def test_cv_iterable_of_index_pairs_matches_sklearn():
    """A raw iterable of ``(train, test)`` pairs is a legal ``cv``."""
    X, y = host_design()
    idx = np.arange(N_SAMPLES)
    splits = [
        (idx[idx % 2 == 0], idx[idx % 2 == 1]),
        (idx[idx % 2 == 1], idx[idx % 2 == 0]),
    ]
    a, b = both(cv=list(splits))
    assert_same_fit(a, b, X, y)


@pytest.mark.parametrize("passthrough", [False, True])
def test_passthrough_matches_sklearn(passthrough):
    X, y = host_design()
    a, b = both(cv=3, passthrough=passthrough)
    assert_same_fit(a, b, X, y)
    assert a.transform(X).shape[1] == 2 + (N_FEATURES if passthrough else 0)


def test_passthrough_appends_x_unchanged_after_the_meta_columns():
    """The passthrough block is the ORIGINAL X, byte for byte, and comes LAST."""
    X, y = host_design()
    a = mlrs.StackingRegressor(sk_estimators(), cv=3, passthrough=True).fit(X, y)
    meta = a.transform(X)
    np.testing.assert_allclose(meta[:, 2:], X, atol=0, rtol=0)


@pytest.mark.parametrize("n_jobs", [None, 1, 2])
def test_n_jobs_does_not_change_the_answer(n_jobs):
    """``n_jobs`` is a value-neutral parameter — a scheduling knob only."""
    X, y = host_design()
    a, b = both(cv=3, n_jobs=n_jobs)
    assert_same_fit(a, b, X, y)


@pytest.mark.parametrize("verbose", [0, 1, 3])
def test_verbose_does_not_change_the_answer(verbose):
    X, y = host_design()
    a, b = both(cv=3, verbose=verbose)
    assert_same_fit(a, b, X, y)


@pytest.mark.parametrize(
    "final_estimator",
    [
        pytest.param(SkRidge(alpha=0.5), id="ridge"),
        pytest.param(SkLinearRegression(), id="linear-regression"),
    ],
)
def test_final_estimator_variants_match_sklearn(final_estimator):
    from sklearn.base import clone

    X, y = host_design()
    a, b = both(cv=3, final_estimator=clone(final_estimator))
    assert_same_fit(a, b, X, y)


def test_sample_weight_reaches_every_sub_estimator():
    """``sample_weight`` is forwarded to base AND final fits (routing off)."""
    X, y = host_design()
    rng = np.random.default_rng(7)
    sw = rng.random(N_SAMPLES) + 0.1
    a, b = both(cv=3)
    a.fit(X, y, sample_weight=sw)
    b.fit(X, y, sample_weight=sw)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=0, rtol=0)
    # A weighted fit must actually differ from the unweighted one, or the
    # assertion above would hold vacuously with the weights dropped on both.
    unweighted = mlrs.StackingRegressor(sk_estimators(), cv=3).fit(X, y)
    assert not np.allclose(a.predict(X), unweighted.predict(X))


def test_extra_fit_kwargs_require_routing():
    """An unroutable kwarg is rejected with sklearn's message, not ignored."""
    X, y = host_design()
    a, b = both()
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y, unexpected=1)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y, unexpected=1)
    assert str(mlrs_exc.value) == str(sk_exc.value)


# =========================================================================== #
# sklearn API surface
# =========================================================================== #


def test_get_params_exposes_exactly_sklearns_parameter_names():
    """The shallow parameter set is sklearn's, name for name."""
    a, b = both()
    assert set(a.get_params(deep=False)) == set(b.get_params(deep=False))
    assert set(a.get_params(deep=False)) == {
        "estimators",
        "final_estimator",
        "cv",
        "n_jobs",
        "passthrough",
        "verbose",
    }


def test_get_params_deep_exposes_nested_sub_estimator_params():
    """``<name>`` and ``<name>__<param>`` keys, so ``GridSearchCV`` can reach in."""
    a, b = both()
    assert set(a.get_params(deep=True)) == set(b.get_params(deep=True))
    assert a.get_params(deep=True)["ridge__alpha"] == 1.0


def test_set_params_reaches_a_nested_sub_estimator_param():
    a, b = both()
    a.set_params(ridge__alpha=7.0)
    b.set_params(ridge__alpha=7.0)
    assert a.get_params()["ridge__alpha"] == 7.0 == b.get_params()["ridge__alpha"]


def test_clone_round_trips_the_whole_composition():
    from sklearn.base import clone

    a = mlrs.StackingRegressor(
        sk_estimators(), final_estimator=SkRidge(alpha=3.0), cv=4, passthrough=True
    )
    c = clone(a)
    assert c.get_params(deep=False).keys() == a.get_params(deep=False).keys()
    assert c.cv == 4 and c.passthrough is True
    assert c.get_params()["ridge__alpha"] == 1.0


def test_named_estimators_property_reads_the_unfitted_argument():
    a, b = both()
    assert list(a.named_estimators) == list(b.named_estimators) == ["lr", "ridge"]


def test_unfitted_attribute_access_raises_not_fitted():
    from sklearn.exceptions import NotFittedError

    est = mlrs.StackingRegressor(sk_estimators())
    with pytest.raises(NotFittedError):
        est.transform(host_design()[0])
    with pytest.raises(AttributeError):
        est.n_features_in_


def test_predict_is_hidden_while_final_estimator_is_none_and_unfitted():
    """sklearn's ``available_if`` gate: an unresolved ``final_estimator`` means
    ``hasattr(est, "predict")`` is ``False``, and a default-constructed stack has
    one. Reproduced because duck-typing code branches on it."""
    a, b = both()
    assert hasattr(a, "predict") is hasattr(b, "predict") is False
    a2, b2 = both(final_estimator=SkRidge())
    assert hasattr(a2, "predict") is hasattr(b2, "predict") is True


def test_fit_transform_equals_fit_then_transform():
    X, y = host_design()
    a, b = both(cv=3)
    np.testing.assert_allclose(
        a.fit_transform(X, y), b.fit_transform(X, y), atol=0, rtol=0
    )


def test_score_matches_sklearn():
    X, y = host_design()
    a, b = both(cv=3)
    a.fit(X, y)
    b.fit(X, y)
    assert a.score(X, y) == pytest.approx(b.score(X, y), abs=1e-12)


def test_get_feature_names_out_validates_input_features():
    """A wrong-length / mismatched ``input_features`` raises sklearn's message."""
    X, y = host_design()
    a, b = both(cv=3, passthrough=True)
    a.fit(X, y)
    b.fit(X, y)
    names = [f"f{i}" for i in range(N_FEATURES)]
    assert list(a.get_feature_names_out(names)) == list(b.get_feature_names_out(names))
    with pytest.raises(ValueError) as mlrs_exc:
        a.get_feature_names_out(names[:-1])
    with pytest.raises(ValueError) as sk_exc:
        b.get_feature_names_out(names[:-1])
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_get_metadata_routing_matches_sklearns_router():
    a, b = both()
    assert str(a.get_metadata_routing()) == str(b.get_metadata_routing())


def test_metadata_routing_honours_per_estimator_requests():
    """With routing ON, only the estimators that REQUESTED ``sample_weight`` get it."""
    import sklearn

    X, y = host_design()
    rng = np.random.default_rng(11)
    sw = rng.random(N_SAMPLES) + 0.1
    sklearn.set_config(enable_metadata_routing=True)
    try:
        estimators = [
            ("lr", SkLinearRegression().set_fit_request(sample_weight=True)),
            ("ridge", SkRidge().set_fit_request(sample_weight=False)),
        ]
        final = SkRidge().set_fit_request(sample_weight=True)
        a = mlrs.StackingRegressor(estimators, final_estimator=final, cv=3)
        b = SkStackingRegressor(estimators, final_estimator=final, cv=3)
        a.fit(X, y, sample_weight=sw)
        b.fit(X, y, sample_weight=sw)
        np.testing.assert_allclose(a.predict(X), b.predict(X), atol=0, rtol=0)
    finally:
        sklearn.set_config(enable_metadata_routing=False)


def test_sklearn_tags_are_derived_from_the_composed_estimators():
    """``allow_nan`` / ``sparse`` are the AND over members, as in sklearn."""
    a, b = both()
    assert a.__sklearn_tags__().input_tags.sparse == (
        b.__sklearn_tags__().input_tags.sparse
    )
    # An mlrs member ingests dense Arrow only, so the stack loses `sparse`.
    m = mlrs.StackingRegressor(mlrs_estimators())
    assert m.__sklearn_tags__().input_tags.sparse is False


def test_nests_inside_grid_search_and_pipeline():
    """The composition is reachable by a search and usable as a pipeline step."""
    from sklearn.pipeline import make_pipeline

    from mlrs.model_selection import GridSearchCV

    X, y = host_design()
    search = GridSearchCV(
        mlrs.StackingRegressor(sk_estimators(), cv=3),
        {"ridge__alpha": [0.1, 1.0]},
        cv=3,
    ).fit(X, y)
    assert search.best_params_["ridge__alpha"] in (0.1, 1.0)
    pipe = make_pipeline(mlrs.StackingRegressor(sk_estimators(), cv=3)).fit(X, y)
    assert pipe.predict(X[:2]).shape == (2,)


# =========================================================================== #
# device design — mlrs sub-estimators (backend-gated dtype + tolerance)
# =========================================================================== #


def test_mlrs_sub_estimators_match_sklearn_within_tolerance():
    """The real deployment shape: base fits on the device, meta fit on the device.

    sklearn answers in float64 regardless, so on an f32 backend this compares
    against a MORE precise reference — hence ``conftest.live_atol()`` rather than
    a hardcoded 1e-5.
    """
    X, y = device_design()
    a = mlrs.StackingRegressor(
        mlrs_estimators(), final_estimator=mlrs.Ridge(alpha=1.0), cv=5
    ).fit(X, y)
    b = SkStackingRegressor(
        sk_estimators(), final_estimator=SkRidge(alpha=1.0), cv=5
    ).fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    np.testing.assert_allclose(
        np.asarray(a.predict(X), dtype=np.float64),
        b.predict(np.asarray(X, dtype=np.float64)),
        atol=conftest.live_atol(),
        rtol=0,
    )


@pytest.mark.parametrize("passthrough", [False, True])
def test_mlrs_sub_estimators_with_passthrough(passthrough):
    X, y = device_design()
    a = mlrs.StackingRegressor(
        mlrs_estimators(),
        final_estimator=mlrs.Ridge(alpha=1.0),
        cv=3,
        passthrough=passthrough,
    ).fit(X, y)
    expected = 2 + (N_FEATURES if passthrough else 0)
    assert np.asarray(a.transform(X)).shape == (N_SAMPLES, expected)
    b = SkStackingRegressor(
        sk_estimators(),
        final_estimator=SkRidge(alpha=1.0),
        cv=3,
        passthrough=passthrough,
    ).fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    np.testing.assert_allclose(
        np.asarray(a.predict(X), dtype=np.float64),
        b.predict(np.asarray(X, dtype=np.float64)),
        atol=conftest.live_atol(),
        rtol=0,
    )


def test_mlrs_sub_estimators_honour_cv_prefit():
    """``cv="prefit"`` over ALREADY-fitted mlrs estimators."""
    X, y = device_design()
    fitted = [
        ("lr", mlrs.LinearRegression().fit(X, y)),
        ("ridge", mlrs.Ridge(alpha=1.0).fit(X, y)),
    ]
    a = mlrs.StackingRegressor(
        fitted, final_estimator=mlrs.Ridge(alpha=1.0), cv="prefit"
    ).fit(X, y)
    assert a.estimators_[0] is fitted[0][1]

    sk_fitted = [
        ("lr", SkLinearRegression().fit(np.asarray(X, np.float64), np.asarray(y, np.float64))),
        ("ridge", SkRidge(alpha=1.0).fit(np.asarray(X, np.float64), np.asarray(y, np.float64))),
    ]
    b = SkStackingRegressor(
        sk_fitted, final_estimator=SkRidge(alpha=1.0), cv="prefit"
    ).fit(np.asarray(X, np.float64), np.asarray(y, np.float64))
    np.testing.assert_allclose(
        np.asarray(a.predict(X), dtype=np.float64),
        b.predict(np.asarray(X, dtype=np.float64)),
        atol=conftest.live_atol(),
        rtol=0,
    )


def test_mlrs_sub_estimators_honour_drop():
    """``'drop'`` over mlrs members — one column, one fitted estimator."""
    X, y = device_design()
    a = mlrs.StackingRegressor(
        mlrs_estimators(), final_estimator=mlrs.Ridge(alpha=1.0), cv=3
    )
    a.set_params(lr="drop")
    a.fit(X, y)
    assert len(a.estimators_) == 1
    assert np.asarray(a.transform(X)).shape == (N_SAMPLES, 1)
    assert list(a.get_feature_names_out()) == ["stackingregressor_ridge"]


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_n_jobs_over_mlrs_members_warns_and_runs_serially(n_jobs):
    """A device-handle member forces serial fitting, loudly — never a crash.

    This is the one place mlrs deliberately does NOT do what ``n_jobs`` asks,
    and both alternatives were measured on real hardware before settling:

    * a process backend (joblib's default) raises ``TypeError: cannot pickle
      'builtins.Ridge' object`` — the fitted handle wraps device state;
    * the threading backend runs correctly (since ``mlrs_backend::stream_cap``
      capped CubeCL's per-OS-thread stream count) but barely helps: every device
      call holds the process-global pool mutex, so six members at ``cv=20`` on
      rocm went 1.584 s serial -> 1.343 s at ``n_jobs=4``.

    So the fit runs serially with a ``UserWarning``, and — asserted here — gives
    exactly the answer the serial fit gives.
    """
    X, y = device_design()
    with pytest.warns(UserWarning, match="n_jobs is ignored"):
        parallel = mlrs.StackingRegressor(
            mlrs_estimators(),
            final_estimator=mlrs.Ridge(alpha=1.0),
            cv=3,
            n_jobs=n_jobs,
        ).fit(X, y)
    serial = mlrs.StackingRegressor(
        mlrs_estimators(), final_estimator=mlrs.Ridge(alpha=1.0), cv=3
    ).fit(X, y)
    np.testing.assert_allclose(
        np.asarray(parallel.predict(X), dtype=np.float64),
        np.asarray(serial.predict(X), dtype=np.float64),
        atol=0,
        rtol=0,
    )


def test_n_jobs_over_host_members_does_not_warn():
    """The fallback is scoped to device members; host compositions keep n_jobs."""
    import warnings

    X, y = host_design()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mlrs.StackingRegressor(sk_estimators(), cv=3, n_jobs=2).fit(X, y)


def test_mixed_mlrs_and_sklearn_members_compose():
    """Nothing requires the members to come from the same library."""
    X, y = device_design()
    est = [("mlrs_lr", mlrs.LinearRegression()), ("sk_ridge", SkRidge(alpha=1.0))]
    a = mlrs.StackingRegressor(est, final_estimator=SkRidge(alpha=1.0), cv=3).fit(X, y)
    assert list(a.get_feature_names_out()) == [
        "stackingregressor_mlrs_lr",
        "stackingregressor_sk_ridge",
    ]
    assert np.asarray(a.predict(X)).shape == (N_SAMPLES,)


# =========================================================================== #
# the Rust structural core, reached directly
# =========================================================================== #


def test_rust_layout_matches_the_python_meta_matrix():
    """``stacking_meta_layout``'s width is the width ``transform`` produces."""
    ext = pytest.importorskip("mlrs")._load_ext()
    X, y = host_design()
    a = mlrs.StackingRegressor(sk_estimators(), cv=3, passthrough=True).fit(X, y)
    n_feature_outs, offsets, n_meta, width = ext.stacking_meta_layout(
        [1, 1], N_FEATURES, True
    )
    assert list(n_feature_outs) == a._n_feature_outs
    assert list(offsets) == [0, 1]
    assert n_meta == 2
    assert width == a.transform(X).shape[1]


def test_rust_feature_names_are_what_the_estimator_returns():
    ext = pytest.importorskip("mlrs")._load_ext()
    X, y = host_design()
    a = mlrs.StackingRegressor(sk_estimators(), cv=3).fit(X, y)
    assert list(a.get_feature_names_out()) == list(
        ext.stacking_feature_names("stackingregressor", ["lr", "ridge"], [1, 1], None)
    )


def test_rust_drop_sentinel_is_the_string_the_estimator_accepts():
    ext = pytest.importorskip("mlrs")._load_ext()
    assert ext.stacking_drop_sentinel() == "drop"


def test_rust_cv_is_prefit_classifies_the_string():
    ext = pytest.importorskip("mlrs")._load_ext()
    assert ext.stacking_cv_is_prefit("prefit") is True
    with pytest.raises(ValueError, match=re.escape("must be an int in the range")):
        ext.stacking_cv_is_prefit("nope")
