"""StackingClassifier oracle harness (STACK-CLF-01: the full parameter surface).

Every test here compares :class:`mlrs.StackingClassifier` against a LIVE
:class:`sklearn.ensemble.StackingClassifier` built with the same arguments, on
the same data, in the same process — the design
``test_oracle_stacking.py`` establishes for the regressor, and for the same
reason: stacking has no numerics of its own, so what needs gating is the
*composition*, and the only reference for that is sklearn itself.

## The string-valued parameter surface (the point of this file)

``StackingClassifier`` has THREE places a caller supplies a string, one more
than the regressor, and the extra one is the parameter that makes this class
different:

  ==================================  =====================================
  string                              what it selects
  ==================================  =====================================
  ``stack_method="auto"``             per estimator, the first of
                                      ``predict_proba`` /
                                      ``decision_function`` / ``predict`` it
                                      implements
  ``stack_method="predict_proba"``    probabilities; on a BINARY target the
                                      first column is dropped as collinear
  ``stack_method="decision_function"``  margins, never column-dropped
  ``stack_method="predict"``          one column of encoded labels
  ``cv="prefit"``                     skip cloning + refitting; meta features
                                      are FULL-training-set responses
  ``estimators=[(name, "drop")]``     disable one entry; the slot survives in
                                      ``named_estimators_`` as ``'drop'``
  ==================================  =====================================

Each is tested for (a) exact agreement with sklearn, (b) the semantics itself —
a value-neutral assertion would pass even if BOTH libraries ignored the string —
and (c) the rejection path, with the message compared against sklearn's own.

## Landmine: sklearn's ``StrOptions`` message is NOT deterministic

``The 'stack_method' parameter … must be a str among {…}`` renders its options
by iterating a Python ``set``, whose order for these strings changes with
``PYTHONHASHSEED``. Two runs of the SAME sklearn call produce
``{'decision_function', 'auto', …}`` and ``{'predict_proba', …}``. So the
message is compared with the option set parsed out (:func:`_split_options`);
comparing it as raw text would be a coin flip that passes locally and fails in
CI.

## Backend gating

Two designs run side by side, as in the regressor's suite:

* **sklearn-only sub-estimators** — pure host composition, dtype-independent
  bookkeeping, so these cells run identically (and EXACTLY, ``atol=0``) on
  cpu / wgpu / rocm / cuda. All string-parameter coverage lives here, so no
  backend gets a vacuous run of it.
* **mlrs sub-estimators** — the real deployment shape, where the base fits go to
  the device. These use ``conftest.default_float_dtype()`` /
  ``conftest.live_atol()`` rather than hardcoding float64, which is what keeps
  them from turning red at ingress on an f64-incapable backend.

Req: STACK-CLF-01 (parameter surface), STACK-BIND-01 (the Rust structural core).
"""

import re

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier as SkStackingClassifier
from sklearn.linear_model import (
    LinearRegression as SkLinearRegression,
    LogisticRegression as SkLogisticRegression,
)
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.svm import LinearSVC as SkLinearSVC

import conftest

mlrs = pytest.importorskip("mlrs")


# --------------------------------------------------------------------------- #
# designs
# --------------------------------------------------------------------------- #

N_SAMPLES = 200
N_FEATURES = 5
SEED = 42


def host_design(dtype=np.float64, n_classes=2, n_samples=N_SAMPLES):
    """A separable classification problem, host-side float64 by default.

    The target is a thresholded linear score, so every member — probabilistic,
    margin-based or a plain regressor — has something to learn, and the classes
    stay balanced enough for a 5-fold ``StratifiedKFold`` at ``n_samples``.
    """
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, N_FEATURES)).astype(dtype)
    score = X @ np.array([3.0, -1.5, 0.0, 0.75, 2.0], dtype=dtype)
    score = score + (0.5 * rng.standard_normal(n_samples)).astype(dtype)
    if n_classes == 2:
        y = (score > 0).astype(np.int64)
    else:
        cuts = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
        y = np.digitize(score, cuts).astype(np.int64)
    return X, y


def device_design(**kwargs):
    """The same problem at the BACKEND's float dtype, for mlrs sub-estimators."""
    return host_design(dtype=conftest.default_float_dtype(), **kwargs)


def sk_estimators():
    """Two sklearn base classifiers with DIFFERENT response surfaces.

    ``GaussianNB`` has ``predict_proba`` but no ``decision_function``;
    ``LinearSVC`` has ``decision_function`` but no ``predict_proba``. Under
    ``stack_method="auto"`` they therefore resolve to different methods, which
    is the mixed-stack case the resolution rule exists for.
    """
    return [("nb", SkGaussianNB()), ("svc", SkLinearSVC())]


def proba_estimators():
    """Two members that both implement ``predict_proba``."""
    return [("lr", SkLogisticRegression()), ("nb", SkGaussianNB())]


def decision_estimators():
    """Two members that both implement ``decision_function``."""
    return [("lr", SkLogisticRegression()), ("svc", SkLinearSVC())]


def mlrs_estimators():
    """Two mlrs device classifiers — the real deployment shape."""
    return [("nb", mlrs.GaussianNB()), ("knn", mlrs.KNeighborsClassifier())]


#: The member set each ``stack_method`` value can legally be applied to.
MEMBERS_FOR_METHOD = {
    "auto": sk_estimators,
    "predict": sk_estimators,
    "predict_proba": proba_estimators,
    "decision_function": decision_estimators,
}


def both(**kwargs):
    """``(mlrs_estimator, sklearn_estimator)`` built from identical arguments."""
    estimators = kwargs.pop("estimators", None)
    if estimators is None:
        estimators = sk_estimators()
    return (
        mlrs.StackingClassifier(estimators, **kwargs),
        SkStackingClassifier(estimators, **kwargs),
    )


def assert_same_fit(a, b, X, y, *, atol=0.0):
    """mlrs and sklearn agree on everything a fitted stack exposes."""
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_allclose(a.transform(X), b.transform(X), atol=atol, rtol=0)
    np.testing.assert_allclose(
        a.predict_proba(X), b.predict_proba(X), atol=max(atol, 1e-10), rtol=0
    )
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
    assert list(a.named_estimators_) == list(b.named_estimators_)
    assert a.stack_method_ == b.stack_method_
    assert a.n_features_in_ == b.n_features_in_
    assert a._n_feature_outs == b._n_feature_outs
    np.testing.assert_array_equal(a.classes_, b.classes_)
    return a, b


def _split_options(message):
    """The option set out of an sklearn ``StrOptions`` message, order-free.

    See this module's docstring: the ORDER inside the braces is Python set
    iteration order and is not reproducible across processes.
    """
    inner = re.search(r"\{(.*?)\}", message)
    assert inner is not None, message
    return frozenset(part.strip() for part in inner.group(1).split(","))


def _message_shape(message):
    """``(text with the option set blanked, the option set)``."""
    return re.sub(r"\{.*?\}", "{}", message), _split_options(message)


# =========================================================================== #
# STRING PARAMETER 1 — stack_method (the classifier's own parameter)
# =========================================================================== #


@pytest.mark.parametrize("stack_method", ["auto", "predict_proba",
                                          "decision_function", "predict"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_stack_method_matches_sklearn_exactly(stack_method, n_classes):
    """Every legal ``stack_method``, on a binary and a 3-class target.

    Exact (``atol=0``): this is host bookkeeping over host sub-estimators, so
    any difference at all is a difference in the composition, not in arithmetic.
    """
    X, y = host_design(n_classes=n_classes)
    a, b = both(estimators=MEMBERS_FOR_METHOD[stack_method](),
                stack_method=stack_method, cv=3)
    assert_same_fit(a, b, X, y)


def test_auto_resolves_per_estimator_not_per_stack():
    """``"auto"`` is a PER-MEMBER decision, and the choices are observable.

    A stack of a proba-only and a margin-only member must report two different
    methods — asserting only "mlrs == sklearn" would pass even if both resolved
    everything to ``predict``.
    """
    X, y = host_design()
    a, b = both(estimators=sk_estimators(), cv=3)
    a.fit(X, y)
    b.fit(X, y)
    assert a.stack_method_ == ["predict_proba", "decision_function"]
    assert a.stack_method_ == b.stack_method_


def test_auto_falls_back_to_predict_for_a_regressor_member():
    """A regressor first layer (sklearn's ordinal-regression case) is legal.

    It implements neither ``predict_proba`` nor ``decision_function``, so
    ``"auto"`` lands on ``predict`` — and the stack fits rather than rejecting
    the member for not being a classifier.
    """
    X, y = host_design()
    a, b = both(estimators=[("lin", SkLinearRegression())], cv=3)
    assert_same_fit(a, b, X, y)
    assert a.stack_method_ == ["predict"]


def test_binary_predict_proba_drops_the_collinear_first_column():
    """The rule that makes the classifier's layout differ from the regressor's.

    On a binary target a two-column ``predict_proba`` becomes ONE meta column,
    and it is the SECOND one — ``p(y=1)``. Pinned against the base estimator's
    own output under ``cv="prefit"``, where the meta features are exactly the
    members' full-training-set responses and the comparison is unambiguous.
    """
    X, y = host_design()
    fitted = [("nb", SkGaussianNB().fit(X, y))]
    a, b = both(estimators=fitted, cv="prefit", stack_method="predict_proba")
    a.fit(X, y)
    b.fit(X, y)

    assert a._n_feature_outs == [1]
    expected = fitted[0][1].predict_proba(X)[:, 1]
    np.testing.assert_allclose(a.transform(X)[:, 0], expected, atol=0, rtol=0)
    np.testing.assert_allclose(a.transform(X), b.transform(X), atol=0, rtol=0)


def test_multiclass_predict_proba_keeps_every_column():
    """Three classes, three columns per member — no drop, because they are not
    pairwise collinear (only their SUM is constrained)."""
    X, y = host_design(n_classes=3)
    a, b = both(estimators=proba_estimators(), stack_method="predict_proba", cv=3)
    a.fit(X, y)
    b.fit(X, y)
    assert a._n_feature_outs == [3, 3] == b._n_feature_outs
    assert a.transform(X).shape == (N_SAMPLES, 6)


def test_binary_decision_function_is_one_column_and_is_not_dropped():
    """A binary ``decision_function`` is 1-D, so it contributes one column —
    the margin itself, NOT a dropped-column artefact of a 2-column block."""
    X, y = host_design()
    fitted = [("svc", SkLinearSVC().fit(X, y))]
    a, _ = both(estimators=fitted, cv="prefit", stack_method="decision_function")
    a.fit(X, y)
    assert a._n_feature_outs == [1]
    np.testing.assert_allclose(
        a.transform(X)[:, 0], fitted[0][1].decision_function(X), atol=0, rtol=0
    )


def test_stack_method_changes_the_answer():
    """``stack_method`` is not value-neutral: a stack of probabilities and a
    stack of hard labels are different models.

    Without this, every "mlrs == sklearn" assertion above would still hold if
    the parameter were silently ignored on both sides.
    """
    X, y = host_design()
    proba = mlrs.StackingClassifier(proba_estimators(),
                                    stack_method="predict_proba", cv=3).fit(X, y)
    hard = mlrs.StackingClassifier(proba_estimators(),
                                   stack_method="predict", cv=3).fit(X, y)
    assert not np.allclose(proba.transform(X), hard.transform(X))
    assert not np.allclose(proba.predict_proba(X), hard.predict_proba(X))


def test_stack_method_rejects_an_unknown_string_like_sklearn():
    """The ``StrOptions`` rejection, compared modulo the option ORDER.

    ``InvalidParameterError`` subclasses both ``ValueError`` and ``TypeError``,
    so a caller migrating from sklearn catches it either way.
    """
    from sklearn.utils._param_validation import (
        InvalidParameterError as SkInvalidParameterError,
    )

    X, y = host_design()
    a, b = both(stack_method="proba")
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(SkInvalidParameterError) as sk_exc:
        b.fit(X, y)
    assert _message_shape(str(mlrs_exc.value)) == _message_shape(str(sk_exc.value))
    assert _split_options(str(mlrs_exc.value)) == frozenset(
        ["'auto'", "'predict_proba'", "'decision_function'", "'predict'"]
    )
    assert isinstance(mlrs_exc.value, ValueError)
    assert isinstance(mlrs_exc.value, TypeError)


def test_cv_is_also_validated_before_the_estimators_list():
    """The same ordering rule for the other string constructor parameter.

    Both are constructor parameters, and sklearn checks constructor parameters
    first; a stack with a colliding estimator name AND a bad ``cv`` must report
    the ``cv``.
    """
    X, y = host_design()
    a, b = both(estimators=[("cv", SkGaussianNB())], cv="nope")
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    assert "'cv' parameter" in str(mlrs_exc.value)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_stack_method_is_validated_before_the_estimators_list():
    """sklearn validates parameters (``@validate_params``) before it validates
    ``estimators``, so a caller who got BOTH wrong sees the same complaint from
    both libraries — not one library's name error and the other's."""
    X, y = host_design()
    bad = [("cv", SkGaussianNB())]  # a name colliding with a ctor argument
    a, b = both(estimators=bad, stack_method="nope")
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    assert "stack_method" in str(mlrs_exc.value)
    assert _message_shape(str(mlrs_exc.value)) == _message_shape(str(sk_exc.value))


@pytest.mark.parametrize(
    "members, method",
    [
        ([("svc", SkLinearSVC())], "predict_proba"),
        ([("nb", SkGaussianNB())], "decision_function"),
        ([("lr", SkLogisticRegression()), ("svc", SkLinearSVC())], "predict_proba"),
    ],
)
def test_a_member_lacking_the_named_method_is_sklearns_value_error(members, method):
    """``Underlying estimator {name} does not implement the method {method}.``"""
    X, y = host_design()
    a, b = both(estimators=members, stack_method=method, cv=3)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert str(mlrs_exc.value).endswith(f"does not implement the method {method}.")


class _NoResponseMethod(ClassifierMixin, BaseEstimator):
    """An estimator with none of the three response methods.

    Contrived on purpose: it is the only way to reach ``"auto"``'s failure
    branch, whose message interpolates the whole method LIST rather than one
    name.
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        return self


def test_auto_with_no_response_method_reports_the_whole_list():
    X, y = host_design()
    a, b = both(estimators=[("odd", _NoResponseMethod())], cv=3)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert str(mlrs_exc.value) == (
        "Underlying estimator odd does not implement the method "
        "['predict_proba', 'decision_function', 'predict']."
    )


def test_stack_method_survives_get_params_clone_and_set_params():
    """It is a plain constructor parameter, so ``clone`` must carry it."""
    from sklearn.base import clone

    est = mlrs.StackingClassifier(proba_estimators(), stack_method="predict")
    assert est.get_params()["stack_method"] == "predict"
    assert clone(est).stack_method == "predict"
    est.set_params(stack_method="predict_proba")
    assert est.stack_method == "predict_proba"


# =========================================================================== #
# STRING PARAMETER 2 — cv="prefit"
# =========================================================================== #


def _prefit_estimators(X, y):
    """Two base classifiers ALREADY fitted, as ``cv="prefit"`` requires."""
    return [("nb", SkGaussianNB().fit(X, y)), ("svc", SkLinearSVC().fit(X, y))]


@pytest.mark.parametrize("n_classes", [2, 3])
def test_cv_prefit_matches_sklearn_exactly(n_classes):
    X, y = host_design(n_classes=n_classes)
    a, b = both(estimators=_prefit_estimators(X, y), cv="prefit")
    assert_same_fit(a, b, X, y)


def test_cv_prefit_does_not_refit_the_given_estimators():
    """``estimators_[i] is estimators[i][1]`` — the caller's own objects."""
    X, y = host_design()
    fitted = _prefit_estimators(X, y)
    theta_before = fitted[0][1].theta_.copy()

    a, b = both(estimators=fitted, cv="prefit")
    a.fit(X, y)
    b.fit(X, y)

    assert a.estimators_[0] is fitted[0][1]
    assert b.estimators_[0] is fitted[0][1]
    np.testing.assert_array_equal(fitted[0][1].theta_, theta_before)


def test_cv_prefit_uses_full_training_responses_not_out_of_fold():
    """The semantic difference itself, not merely agreement with sklearn.

    A 1-nearest-neighbour member makes it unmissable: fitted on all of ``X`` its
    in-sample ``predict_proba`` is a one-hot of each row's OWN label, while its
    out-of-fold one comes from a genuinely different row. As in the regressor's
    suite, the route is invisible in ``transform`` (which always re-predicts
    through ``estimators_``) and shows up in what ``final_estimator_`` was
    TRAINED on.
    """
    from sklearn.neighbors import KNeighborsClassifier

    X, y = host_design()
    fitted = [("nn", KNeighborsClassifier(n_neighbors=1).fit(X, y))]

    prefit = mlrs.StackingClassifier(fitted, cv="prefit").fit(X, y)
    kfold = mlrs.StackingClassifier(
        [("nn", KNeighborsClassifier(n_neighbors=1))], cv=5
    ).fit(X, y)

    # The prefit meta learner was trained on a meta column that IS the label, so
    # it sees a perfectly separable problem and pushes its coefficient up; the
    # cross-validated one was trained on a noisier column and stays smaller.
    # Measured on this design: 5.63 vs 2.66.
    assert abs(prefit.final_estimator_.coef_[0, 0]) > 2 * abs(
        kfold.final_estimator_.coef_[0, 0]
    )
    # The confidence, not the label, is where it shows: both stacks classify
    # this separable design perfectly, so a `score` comparison would be a tie.
    assert not np.allclose(prefit.predict_proba(X), kfold.predict_proba(X))
    # …and NOT in `transform`, which re-predicts through `estimators_` on both
    # routes (the regressor suite's trap 1, restated for the classifier).
    np.testing.assert_allclose(prefit.transform(X), kfold.transform(X), atol=0)


def test_cv_prefit_with_an_unfitted_estimator_raises_not_fitted():
    from sklearn.exceptions import NotFittedError

    X, y = host_design()
    a, b = both(estimators=sk_estimators(), cv="prefit")
    with pytest.raises(NotFittedError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(NotFittedError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


@pytest.mark.parametrize("passthrough", [False, True])
def test_cv_prefit_with_passthrough(passthrough):
    X, y = host_design()
    a, b = both(estimators=_prefit_estimators(X, y), cv="prefit",
                passthrough=passthrough)
    a, b = assert_same_fit(a, b, X, y)
    width = 2 + (N_FEATURES if passthrough else 0)
    assert a.transform(X).shape == (N_SAMPLES, width)


def test_cv_rejects_an_unknown_string_like_sklearn():
    from sklearn.utils._param_validation import (
        InvalidParameterError as SkInvalidParameterError,
    )

    X, y = host_design()
    a, b = both(cv="prefitted")
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(SkInvalidParameterError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "StackingClassifier" in str(mlrs_exc.value)


# =========================================================================== #
# STRING PARAMETER 3 — estimators=[(name, "drop")]
# =========================================================================== #


@pytest.mark.parametrize("passthrough", [False, True])
def test_drop_disables_one_entry_and_keeps_its_slot(passthrough):
    X, y = host_design()
    members = [("nb", SkGaussianNB()), ("svc", "drop")]
    a, b = both(estimators=members, cv=3, passthrough=passthrough)
    a, b = assert_same_fit(a, b, X, y)

    assert a.named_estimators_["svc"] == "drop"
    assert len(a.estimators_) == 1
    assert a.stack_method_ == ["predict_proba"]
    assert list(a.get_feature_names_out())[0] == "stackingclassifier_nb"


def test_drop_is_not_asked_for_a_method_it_lacks():
    """A dropped entry is never response-resolved.

    sklearn computes ``_method_name`` for every entry but returns ``None`` for a
    dropped one BEFORE checking the method exists, so a stack whose only
    proba-less member is dropped fits fine under
    ``stack_method="predict_proba"``. mlrs resolves only the kept entries, which
    is the same rule stated once instead of twice.
    """
    X, y = host_design()
    members = [("lr", SkLogisticRegression()), ("svc", "drop")]
    a, b = both(estimators=members, stack_method="predict_proba", cv=3)
    assert_same_fit(a, b, X, y)


def test_all_dropped_is_rejected_with_sklearns_message():
    X, y = host_design()
    a, b = both(estimators=[("nb", "drop"), ("svc", "drop")])
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert str(mlrs_exc.value) == (
        "All estimators are dropped. At least one is required to be an estimator."
    )


def test_set_params_drop_round_trips():
    """``set_params(name='drop')`` is how sklearn users actually drop a member."""
    X, y = host_design()
    a, b = both(estimators=sk_estimators(), cv=3)
    a.set_params(svc="drop")
    b.set_params(svc="drop")
    assert_same_fit(a, b, X, y)
    assert a.named_estimators_["svc"] == "drop"


# =========================================================================== #
# the rest of the parameter surface
# =========================================================================== #


@pytest.mark.parametrize("cv", [2, 5])
def test_cv_as_an_int(cv):
    X, y = host_design()
    a, b = both(cv=cv)
    assert_same_fit(a, b, X, y)


def test_cv_as_a_splitter_object():
    from sklearn.model_selection import StratifiedKFold

    X, y = host_design()
    a, b = both(cv=StratifiedKFold(4))
    assert_same_fit(a, b, X, y)


def test_cv_as_an_iterable_of_index_pairs():
    X, y = host_design()
    rng = np.random.default_rng(0)
    order = rng.permutation(N_SAMPLES)
    folds = [
        (np.setdiff1d(order, part), part) for part in np.array_split(order, 4)
    ]
    a, b = both(cv=list(folds))
    assert_same_fit(a, b, X, y)


def test_cv_none_is_five_fold_stratified():
    """``cv=None`` stratifies for a classifier — the same folds sklearn picks.

    Exact agreement with sklearn's ``cv=None`` fit IS the assertion: a
    non-stratified 5-fold would produce different out-of-fold responses on this
    design and the meta features would differ.
    """
    X, y = host_design(n_classes=3)
    a, b = both(cv=None)
    assert_same_fit(a, b, X, y)


@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_passthrough_appends_the_design_columns_last(passthrough, n_classes):
    X, y = host_design(n_classes=n_classes)
    a, b = both(cv=3, passthrough=passthrough)
    a, b = assert_same_fit(a, b, X, y)

    per_member = 1 if n_classes == 2 else n_classes
    meta_width = per_member + (1 if n_classes == 2 else n_classes)
    width = meta_width + (N_FEATURES if passthrough else 0)
    assert a.transform(X).shape == (N_SAMPLES, width)
    if passthrough:
        np.testing.assert_allclose(a.transform(X)[:, -N_FEATURES:], X, atol=0)


def test_verbose_is_forwarded_and_value_neutral(capfd):
    X, y = host_design()
    a, b = both(cv=2, verbose=1)
    assert_same_fit(a, b, X, y)


def test_n_jobs_over_host_members_is_value_neutral():
    X, y = host_design()
    serial = mlrs.StackingClassifier(sk_estimators(), cv=3).fit(X, y)
    parallel = mlrs.StackingClassifier(sk_estimators(), cv=3, n_jobs=2).fit(X, y)
    np.testing.assert_allclose(
        serial.transform(X), parallel.transform(X), atol=0, rtol=0
    )


def test_final_estimator_default_is_sklearns_logistic_regression():
    """``None`` means sklearn's ``LogisticRegression()``, not an mlrs stand-in.

    Substituting mlrs's own would move every default-constructed stack off the
    sklearn baseline; the parity contract is the reason this default is
    deliberate rather than incidental.
    """
    X, y = host_design()
    a, b = both(cv=3)
    a.fit(X, y)
    b.fit(X, y)
    assert type(a.final_estimator_) is type(b.final_estimator_)
    assert type(a.final_estimator_).__name__ == "LogisticRegression"


def test_final_estimator_custom():
    X, y = host_design()
    a, b = both(final_estimator=SkGaussianNB(), cv=3)
    assert_same_fit(a, b, X, y)


def test_final_estimator_must_be_a_classifier():
    X, y = host_design()
    a, b = both(final_estimator=SkLinearRegression())
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_decision_function_is_delegated_to_the_final_estimator():
    """Available exactly when ``final_estimator_`` has it, and equal to it."""
    X, y = host_design(n_classes=3)
    a, b = both(final_estimator=SkLinearSVC(), cv=3)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(
        a.decision_function(X), b.decision_function(X), atol=1e-10, rtol=0
    )
    # `LinearSVC` has no `predict_proba`, and `available_if` must say so.
    assert not hasattr(a, "predict_proba")
    assert not hasattr(b, "predict_proba")


def test_predict_proba_is_delegated_to_the_final_estimator():
    X, y = host_design(n_classes=3)
    a, b = both(cv=3)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(
        a.predict_proba(X), b.predict_proba(X), atol=1e-10, rtol=0
    )


def test_fit_transform_equals_fit_then_transform():
    X, y = host_design()
    a, b = both(cv=3)
    np.testing.assert_allclose(
        a.fit_transform(X, y), b.fit_transform(X, y), atol=0, rtol=0
    )


# --------------------------------------------------------------------------- #
# labels: encoding, ordering, dtype
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["neg", "pos"]),
        np.array([-1, 1]),
        np.array([10, 20]),
        np.array([False, True]),
    ],
)
def test_classes_round_trip_in_the_callers_own_dtype(labels):
    """``predict`` answers in the caller's labels, whatever their dtype.

    The device (and the meta learner) only ever see ``0..n_classes-1``; the
    encoder is what makes that invisible.
    """
    X, y01 = host_design()
    y = labels[y01]
    a, b = both(cv=3)
    a, b = assert_same_fit(a, b, X, y)
    assert a.predict(X).dtype == b.predict(X).dtype
    assert set(np.unique(a.predict(X))).issubset(set(labels))


def test_string_labels_with_predict_stack_method():
    """``stack_method="predict"`` stacks ENCODED labels, not the strings.

    Otherwise the meta matrix would be an object array and the final estimator
    would refuse it — the encoding is what makes this combination work at all.
    """
    X, y01 = host_design(n_classes=3)
    y = np.array(["a", "b", "c"])[y01]
    a, b = both(stack_method="predict", cv=3)
    a, b = assert_same_fit(a, b, X, y)
    assert a.transform(X).dtype.kind in "iu"


def test_a_continuous_target_is_rejected_like_sklearn():
    X, _ = host_design()
    y = X @ np.ones(N_FEATURES)
    a, b = both(cv=3)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_multilabel_indicator_target():
    """A multilabel ``y``: ``classes_`` is a LIST and ``predict`` is 2-D.

    This is also the only path where one member contributes SEVERAL meta blocks
    (one per target, each with its first probability column dropped), so it
    exercises the multi-output branch of the Rust slice rule and the
    list-shaped ``cross_val_predict`` response together.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    X, y01 = host_design()
    Y = np.column_stack([y01, 1 - y01, (X[:, 1] > 0).astype(np.int64)])
    kwargs = dict(
        final_estimator=MultiOutputClassifier(SkLogisticRegression()), cv=2
    )
    members = [("rf", RandomForestClassifier(n_estimators=5, random_state=0))]
    a = mlrs.StackingClassifier(members, **kwargs).fit(X, Y)
    b = SkStackingClassifier(members, **kwargs).fit(X, Y)

    np.testing.assert_allclose(a.transform(X), b.transform(X), atol=0, rtol=0)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), atol=1e-10)
    assert a._n_feature_outs == [1, 1, 1] == b._n_feature_outs
    assert [list(c) for c in a.classes_] == [list(c) for c in b.classes_]
    # sklearn zips names with `_n_feature_outs`, so the extra blocks have no
    # names — the short list is the reference behaviour, not a bug in mlrs.
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


# --------------------------------------------------------------------------- #
# names, features and metadata
# --------------------------------------------------------------------------- #


def test_estimator_name_rules_match_sklearn():
    X, y = host_design()
    for members, fragment in [
        ([("a", SkGaussianNB()), ("a", SkLinearSVC())], "not unique"),
        ([("cv", SkGaussianNB())], "conflict with constructor arguments"),
        ([("a__b", SkGaussianNB())], "must not contain __"),
    ]:
        a, b = both(estimators=members)
        with pytest.raises(ValueError) as mlrs_exc:
            a.fit(X, y)
        with pytest.raises(ValueError) as sk_exc:
            b.fit(X, y)
        assert fragment in str(mlrs_exc.value)
        assert str(mlrs_exc.value) == str(sk_exc.value)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("passthrough", [False, True])
def test_get_feature_names_out(n_classes, passthrough):
    X, y = host_design(n_classes=n_classes)
    a, b = both(cv=3, passthrough=passthrough)
    a.fit(X, y)
    b.fit(X, y)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
    if n_classes == 3:
        # A multi-column block is suffixed with the within-block index and NO
        # separator.
        assert "stackingclassifier_nb0" in list(a.get_feature_names_out())


def test_feature_names_in_comes_from_a_fitted_member():
    pd = pytest.importorskip("pandas")

    X, y = host_design()
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    a, b = both(cv=3)
    a.fit(frame, y)
    b.fit(frame, y)
    np.testing.assert_array_equal(a.feature_names_in_, b.feature_names_in_)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


def test_n_features_in_is_the_base_estimators_width():
    X, y = host_design()
    a, b = both(cv=3, passthrough=True)
    a.fit(X, y)
    b.fit(X, y)
    assert a.n_features_in_ == b.n_features_in_ == N_FEATURES


def test_n_features_in_raises_attribute_error_before_fit():
    est = mlrs.StackingClassifier(sk_estimators())
    with pytest.raises(AttributeError):
        est.n_features_in_


def test_sample_weight_reaches_every_member():
    X, y = host_design()
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.5, 1.5, N_SAMPLES)
    a, b = both(estimators=proba_estimators(), cv=3)
    a.fit(X, y, sample_weight=weights)
    b.fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(a.transform(X), b.transform(X), atol=0, rtol=0)
    # …and it is not ignored: a different weighting is a different fit.
    unweighted = mlrs.StackingClassifier(proba_estimators(), cv=3).fit(X, y)
    assert not np.allclose(a.transform(X), unweighted.transform(X))


def test_extra_fit_params_need_routing_enabled():
    X, y = host_design()
    a, b = both(cv=3)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y, something=1)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y, something=1)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_get_metadata_routing_matches_sklearns_router():
    a, b = both()
    assert str(a.get_metadata_routing()) == str(b.get_metadata_routing())


def test_metadata_routing_honours_per_estimator_requests():
    """With routing ON, only the members that REQUESTED ``sample_weight`` get it."""
    import sklearn

    X, y = host_design()
    rng = np.random.default_rng(11)
    sw = rng.random(N_SAMPLES) + 0.1
    sklearn.set_config(enable_metadata_routing=True)
    try:
        estimators = [
            ("lr", SkLogisticRegression().set_fit_request(sample_weight=True)),
            ("nb", SkGaussianNB().set_fit_request(sample_weight=False)),
        ]
        final = SkLogisticRegression().set_fit_request(sample_weight=True)
        a = mlrs.StackingClassifier(estimators, final_estimator=final, cv=3)
        b = SkStackingClassifier(estimators, final_estimator=final, cv=3)
        a.fit(X, y, sample_weight=sw)
        b.fit(X, y, sample_weight=sw)
        np.testing.assert_allclose(a.transform(X), b.transform(X), atol=0, rtol=0)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))
    finally:
        sklearn.set_config(enable_metadata_routing=False)


def test_grid_search_over_a_member_parameter():
    """``<name>__<param>`` reaches into a member — what ``get_params(deep=True)``
    exists for."""
    from sklearn.model_selection import GridSearchCV

    X, y = host_design()
    est = mlrs.StackingClassifier(proba_estimators(), cv=2)
    search = GridSearchCV(est, {"lr__C": [0.1, 1.0]}, cv=2).fit(X, y)
    assert search.best_params_["lr__C"] in (0.1, 1.0)


def test_tags_are_the_and_over_members():
    est = mlrs.StackingClassifier(sk_estimators())
    sk = SkStackingClassifier(sk_estimators())
    from sklearn.utils import get_tags

    assert get_tags(est).input_tags.allow_nan == get_tags(sk).input_tags.allow_nan
    assert get_tags(est).input_tags.sparse == get_tags(sk).input_tags.sparse


# =========================================================================== #
# the device design — mlrs sub-estimators
# =========================================================================== #


@pytest.mark.parametrize("stack_method", ["auto", "predict", "predict_proba"])
def test_mlrs_members_match_sklearn_members_structurally(stack_method):
    """A stack of mlrs estimators composes exactly like a stack of sklearn ones.

    The numbers are the members' own (and are gated by their own oracle suites);
    what this asserts is that the COMPOSITION does not change when the members
    move to the device — same resolved methods, same meta width, same names, and
    predictions that agree with the sklearn-member stack within the backend's
    live tolerance.
    """
    X, y = device_design()
    device = mlrs.StackingClassifier(
        mlrs_estimators(), stack_method=stack_method, cv=3
    ).fit(X, y)
    host = SkStackingClassifier(
        [("nb", SkGaussianNB()), ("knn", _sk_knn())],
        stack_method=stack_method,
        cv=3,
    ).fit(X, y)

    assert device.stack_method_ == host.stack_method_
    assert device.transform(X).shape == host.transform(X).shape
    assert list(device.get_feature_names_out()) == list(host.get_feature_names_out())
    np.testing.assert_allclose(
        device.transform(X), host.transform(X), atol=conftest.live_atol(), rtol=0
    )


def _sk_knn():
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier()


def test_mlrs_members_with_passthrough_and_prefit():
    X, y = device_design()
    fitted = [(name, est.fit(X, y)) for name, est in mlrs_estimators()]
    est = mlrs.StackingClassifier(fitted, cv="prefit", passthrough=True).fit(X, y)
    assert est.transform(X).shape[1] == est._n_feature_outs[0] * 0 + sum(
        est._n_feature_outs
    ) + N_FEATURES
    assert est.estimators_[0] is fitted[0][1]


def test_mlrs_member_stack_warns_and_serializes_n_jobs():
    """``n_jobs`` over a device member is reduced to serial, with a warning.

    A process-based joblib backend cannot pickle a fitted mlrs estimator's
    device handle, so fanning out would raise rather than run.
    """
    X, y = device_design()
    est = mlrs.StackingClassifier(mlrs_estimators(), cv=2, n_jobs=2)
    with pytest.warns(UserWarning, match="n_jobs is ignored"):
        est.fit(X, y)
    assert est.predict(X).shape == (N_SAMPLES,)


# =========================================================================== #
# the Rust structural core, reached directly
# =========================================================================== #


def test_rust_stack_method_validation():
    ext = mlrs._load_ext()
    assert ext.stacking_stack_method("auto") == "auto"
    assert ext.stacking_stack_method("predict_proba") == "predict_proba"
    with pytest.raises(ValueError, match="must be a str among"):
        ext.stacking_stack_method("nope")


def test_rust_resolves_the_auto_chain():
    ext = mlrs._load_ext()
    # (has_predict_proba, has_decision_function, has_predict)
    assert ext.stacking_resolve_stack_methods(
        ["a", "b", "c"],
        "auto",
        [(True, True, True), (False, True, True), (False, False, True)],
    ) == ["predict_proba", "decision_function", "predict"]
    with pytest.raises(ValueError, match=re.escape("does not implement the method")):
        ext.stacking_resolve_stack_methods(["a"], "predict_proba", [(False, True, True)])


def test_rust_meta_slices_encode_the_drop_rule():
    ext = mlrs._load_ext()
    # kinds: 0 = 1-D, 1 = 2-D, 2 = list of 2-D
    assert ext.stacking_classifier_meta_slices(
        ["predict_proba"], [1], [[2]], 2
    ) == [(0, 0, 1, 1)]
    assert ext.stacking_classifier_meta_slices(
        ["predict_proba"], [1], [[3]], 3
    ) == [(0, 0, 0, 3)]
    assert ext.stacking_classifier_meta_slices(
        ["decision_function"], [0], [[]], 2
    ) == [(0, 0, 0, 1)]
    assert ext.stacking_classifier_meta_slices(
        ["predict_proba"], [2], [[2, 2]], 2
    ) == [(0, 0, 1, 1), (0, 1, 1, 1)]
