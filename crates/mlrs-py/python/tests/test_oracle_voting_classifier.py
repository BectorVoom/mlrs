"""VotingClassifier oracle harness (VOTE-CLF-01: the full parameter surface).

Every test here compares :class:`mlrs.VotingClassifier` against a LIVE
:class:`sklearn.ensemble.VotingClassifier` built with the same arguments, on the
same data, in the same process. There are no committed ``.npz`` fixtures, for
the reason ``test_oracle_voting_regressor.py`` gives: voting has no numerics of
its own — it fits other estimators and combines them — so what needs gating is
the *composition*, and the only reference for that is sklearn itself.

## The string-valued parameter surface (the point of this file)

``VotingClassifier``'s constructor is ``(estimators, *, voting='hard',
weights=None, n_jobs=None, flatten_transform=True, verbose=False)``. Two of
those take strings, and unlike the regressor one of them is a genuine scalar
parameter that forks the whole estimator:

  ===============================  =========================================
  string                           what it selects
  ===============================  =========================================
  ``voting='hard'``                weighted majority over the members'
                                   predicted LABELS. ``predict_proba`` does
                                   not exist; ``transform`` is the
                                   ``(n, k)`` label matrix;
                                   ``get_feature_names_out`` is one name per
                                   member
  ``voting='soft'``                weighted average of the members'
                                   ``predict_proba``. ``transform`` is
                                   ``np.hstack(probas)`` (or the raw 3-D
                                   stack under ``flatten_transform=False``),
                                   and there are ``n_classes`` feature names
                                   per member
  ``estimators=[(name, 'drop')]``  disable one entry: it is never fitted and
                                   contributes no column, but its slot
                                   survives in ``named_estimators_`` as the
                                   string ``'drop'`` AND its slot in
                                   ``weights`` still has to be supplied
  ===============================  =========================================

``weights`` is array-like or ``None``, ``n_jobs`` is an int or ``None``, and
``flatten_transform``/``verbose`` are booleans, so those two are the whole
string surface. Both are exercised here in every combination that can interact
with them: each ``voting`` value against a binary AND a multiclass target,
weighted and uniform, with ``'drop'`` at every position, through
``predict`` / ``predict_proba`` / ``transform`` / ``get_feature_names_out`` /
``named_estimators_`` / ``n_features_in_``, and on the rejection paths.

mlrs adds one further string surface of its own, ``MLRS_VOTING_ENGINE``
(``numpy`` / ``host`` / ``device``). It is an aggregation-arm A/B knob rather
than a constructor parameter, and it is oracle-tested per arm — including
``voting`` and ``'drop'`` again on every arm — in
``test_voting_classifier_engine.py``.

## Landmine: sklearn's ``StrOptions`` message is NOT deterministic

``The 'voting' parameter … must be a str among {…}`` renders its options by
iterating a Python ``set``, whose order for these strings changes with
``PYTHONHASHSEED``. Two runs of the SAME sklearn call produce ``{'hard',
'soft'}`` and ``{'soft', 'hard'}``. So the constraint messages are compared with
the option set PARSED OUT (:func:`options_in`) rather than as raw text —
comparing them literally would be a coin flip. This is the same trap
``test_oracle_stacking_classifier.py`` documents for ``stack_method``.

## Why the value assertions are EXACT

Hard voting is an integer/`f64` bincount argmax and soft voting reproduces
``np.average`` operation for operation, so mlrs and sklearn must agree BIT FOR
BIT on the sklearn-only cells rather than within 1e-5. That is a much stronger
assertion than the project's tolerance contract and is the only one that catches
a reassociated accumulation or a shifted tie-break. The ``mlrs``-sub-estimator
cells fall back to ``conftest.live_atol()``, because there the MEMBERS' own
device arithmetic is in the comparison.

## Backend gating

Two designs run side by side, exactly as in the regressor's harness:

* **sklearn-only sub-estimators** — pure host composition, dtype-independent, so
  these cells run identically (and EXACTLY) on cpu / wgpu / rocm / cuda. All the
  string-parameter coverage lives here, so no backend can end up with a vacuous
  run of it.
* **mlrs sub-estimators** — the real deployment shape, using
  ``conftest.default_float_dtype()`` / ``conftest.live_atol()`` so they do not
  turn red at ingress on an f64-incapable backend instead of comparing anything.

Req: VOTE-CLF-01 (parameter surface), VOTE-BIND-01 (the Rust structural core).
"""

import re

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier as SkVotingClassifier
from sklearn.linear_model import (
    LinearRegression as SkLinearRegression,
    LogisticRegression as SkLogisticRegression,
)
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier
from sklearn.svm import SVC as SkSVC
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier

import conftest

mlrs = pytest.importorskip("mlrs")


N_SAMPLES = 200
N_FEATURES = 5
SEED = 42

#: Every ``voting`` value sklearn accepts. Parametrizing on this rather than on
#: a literal pair means a future sklearn addition shows up as a failure here
#: instead of as silently missing coverage.
VOTING_VALUES = ["hard", "soft"]


# --------------------------------------------------------------------------- #
# designs
# --------------------------------------------------------------------------- #


def host_design(n_classes=3, dtype=np.float64, n_samples=N_SAMPLES):
    """A separable classification problem with ``n_classes`` well-populated
    classes.

    Deliberately NOT near-degenerate: soft voting's ``device`` arm rounds its
    reduction once where numpy rounds twice, so a design whose top two averaged
    probabilities differ by less than a ULP would make the argmax a coin flip and
    turn every engine comparison into a test of the hardware. Separated classes
    keep the assertions on the composition.
    """
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, N_FEATURES)).astype(dtype)
    score = X[:, 0] * 3.0 - X[:, 1] + 0.5 * X[:, 2]
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.searchsorted(edges, score)
    return X, y.astype(np.int64)


def sk_estimators():
    """Three sklearn classifiers that genuinely DISAGREE and all expose
    ``predict_proba``.

    Disagreement is load-bearing: an ensemble whose members always vote the same
    way cannot distinguish a correct weighting from a broken one. ``predict_proba``
    on all three is required because ``voting='soft'`` asks every member for it —
    a member without it is a separate, deliberately-tested rejection path.
    """
    return [
        ("lr", SkLogisticRegression(max_iter=500)),
        ("nb", SkGaussianNB()),
        ("knn", SkKNeighborsClassifier(n_neighbors=5)),
    ]


def both(estimators=None, **kwargs):
    """One mlrs estimator and one sklearn estimator with identical arguments."""
    if estimators is None:
        estimators = sk_estimators()
    return (
        mlrs.VotingClassifier(estimators, **kwargs),
        SkVotingClassifier(estimators, **kwargs),
    )


def fit_both(X, y, estimators=None, **kwargs):
    a, b = both(estimators, **kwargs)
    return a.fit(X, y), b.fit(X, y)


def assert_same(a, b, X):
    """Every fitted output agrees BIT FOR BIT.

    See the module docstring: both aggregations are reproduced operation for
    operation, so anything looser would let a reassociated accumulation or a
    shifted tie-break through.
    """
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    np.testing.assert_array_equal(a.classes_, b.classes_)
    if hasattr(b, "predict_proba"):
        assert hasattr(a, "predict_proba")
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    else:
        assert not hasattr(a, "predict_proba")


def assert_same_error(mlrs_exc, sk_exc):
    """Same exception CLASS NAME and same message.

    The name rather than the class itself: mlrs mirrors
    ``sklearn.utils._param_validation.InvalidParameterError`` with its own
    subclass of ``(ValueError, TypeError)`` rather than importing a private
    sklearn path, so an ``isinstance`` check against sklearn's would fail on an
    estimator that is behaving correctly. The NAME and the message are what a
    caller sees.

    A message carrying a ``StrOptions`` set is compared with that set PARSED OUT
    and the rest of the text compared literally — see the module docstring's
    landmine. Doing this in the shared helper rather than at each call site is
    not tidiness: a raw ``==`` on such a message PASSES on most
    ``PYTHONHASHSEED`` values and fails on the rest, so a call site that forgets
    is a test that goes red weeks later for no reason anyone changed. (Which is
    exactly what happened to this file before this helper learned the rule.)
    """
    assert type(mlrs_exc.value).__name__ == type(sk_exc.value).__name__
    mlrs_msg, sk_msg = str(mlrs_exc.value), str(sk_exc.value)
    if "{" in sk_msg and "}" in sk_msg:
        assert options_in(mlrs_msg) == options_in(sk_msg)
        assert mlrs_msg.split("{")[0] == sk_msg.split("{")[0]
        assert mlrs_msg.split("}")[-1] == sk_msg.split("}")[-1]
        return
    assert mlrs_msg == sk_msg


def options_in(message):
    """The option set out of an sklearn ``StrOptions`` message, order-free.

    See the module docstring's landmine section: the braces render a Python
    ``set``, whose iteration order moves with ``PYTHONHASHSEED``.
    """
    match = re.search(r"\{([^}]*)\}", message)
    assert match is not None, f"no option set in {message!r}"
    return frozenset(part.strip() for part in match.group(1).split(","))


# =========================================================================== #
# STRING PARAMETER 1 — `voting` (the parameter that forks the estimator)
# =========================================================================== #


@pytest.mark.parametrize("voting", VOTING_VALUES)
@pytest.mark.parametrize("n_classes", [2, 3, 5], ids=["binary", "3-class", "5-class"])
def test_every_voting_value_matches_sklearn_exactly(voting, n_classes):
    """The headline oracle: both modes, on a binary and two multiclass targets.

    The class count is parametrized because the two routes scale differently
    with it — hard voting's tally widens while soft voting's transform widens by
    ``n_classes`` per member — so a shape bug that cancels at ``n_classes == 2``
    (where a probability block is square with the binary special cases) survives
    into 3 and 5.
    """
    X, y = host_design(n_classes)
    a, b = fit_both(X, y, voting=voting)
    assert_same(a, b, X)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_voting_matches_sklearn_under_asymmetric_weights(voting, ):
    """The weighting has to reach BOTH routes.

    Under ``'hard'`` the weights become ``np.bincount``'s ``weights`` argument;
    under ``'soft'`` they become ``np.average``'s. Those are different code
    paths in sklearn and different Rust entry points in mlrs, so a weight vector
    that reached only one of them would pass a single-mode test.
    """
    X, y = host_design(3)
    a, b = fit_both(X, y, voting=voting, weights=[3.0, 1.0, 7.0])
    assert_same(a, b, X)


def test_hard_voting_is_a_weighted_majority_of_the_members_labels():
    """The value assertion, spelled out rather than delegated to sklearn.

    Recomputing the expected answer from the FITTED members' own predictions is
    what makes this test independent of sklearn's implementation: if both
    libraries changed the tie-break together, :func:`assert_same` would still
    pass and this would not.
    """
    X, y = host_design(3)
    weights = [3.0, 1.0, 7.0]
    a, _ = fit_both(X, y, voting="hard", weights=weights)

    columns = np.asarray([est.predict(X) for est in a.estimators_]).T
    expected = np.apply_along_axis(
        lambda row: np.argmax(np.bincount(row, weights=weights)), axis=1, arr=columns
    )
    np.testing.assert_array_equal(a.predict(X), a.le_.inverse_transform(expected))


def test_soft_voting_is_the_argmax_of_its_own_predict_proba():
    """``predict`` and ``predict_proba`` cannot disagree.

    The ``device`` arm FUSES these two into one kernel chain, so this identity is
    the one assertion that catches a fused path reading the accumulator at the
    wrong point. It holds on the ``numpy`` arm too, where it is nearly a
    tautology — which is exactly why it belongs in the shared harness rather than
    only in the engine suite.
    """
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="soft", weights=[3.0, 1.0, 7.0])
    proba = a.predict_proba(X)
    np.testing.assert_array_equal(a.predict(X), a.classes_[np.argmax(proba, axis=1)])
    np.testing.assert_array_equal(proba, b.predict_proba(X))


def test_predict_proba_does_not_exist_under_hard_voting():
    """sklearn hides it behind ``available_if``, and words the failure with the
    offending value in it. Both the ``hasattr`` answer and the message are part
    of the contract — a caller feature-detects with the former and a test suite
    matches on the latter."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="hard")
    assert not hasattr(a, "predict_proba")
    assert not hasattr(b, "predict_proba")
    with pytest.raises(AttributeError) as sk_exc:
        b.predict_proba(X)
    with pytest.raises(AttributeError) as mlrs_exc:
        a.predict_proba(X)
    # sklearn's `available_if` swallows the predicate's own AttributeError and
    # reports the generic "has no attribute" text, so THAT is the contract —
    # asserting on the predicate's wording would gate on an sklearn internal.
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "no attribute 'predict_proba'" in str(mlrs_exc.value)


def test_predict_proba_appears_when_voting_is_switched_to_soft():
    """``available_if`` is re-evaluated per access, so ``set_params`` flips it —
    including on an ALREADY-FITTED estimator, which is sklearn's behaviour and
    the reason the predicate reads ``self.voting`` rather than a fitted
    attribute."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="hard")
    a.set_params(voting="soft")
    b.set_params(voting="soft")
    assert hasattr(a, "predict_proba") and hasattr(b, "predict_proba")
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize(
    "bad", ["majority", "Hard", "SOFT", "", "hard ", "average", "weighted"]
)
def test_an_unrecognized_voting_value_is_sklearns_invalid_parameter_error(bad):
    """Case and whitespace matter — ``StrOptions`` is a plain set membership
    test — and the rejection must be sklearn's own exception CLASS, since a
    caller catching ``InvalidParameterError`` is catching a subclass of
    ``ValueError`` that mlrs must not widen."""
    X, y = host_design(3)
    a, b = both(voting=bad)
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    assert type(mlrs_exc.value).__name__ == type(sk_exc.value).__name__
    sk_msg, mlrs_msg = str(sk_exc.value), str(mlrs_exc.value)
    # Option SET, not raw text — see the module docstring's landmine.
    assert options_in(mlrs_msg) == options_in(sk_msg) == frozenset({"'hard'", "'soft'"})
    assert mlrs_msg.split("{")[0] == sk_msg.split("{")[0]
    assert mlrs_msg.split("}")[1] == sk_msg.split("}")[1]
    assert repr(bad) in mlrs_msg


def test_a_non_string_voting_value_is_rejected_the_same_way():
    """The constraint is a ``str`` membership test, so a non-string fails it
    rather than reaching the FFI (which cannot accept one at all)."""
    X, y = host_design(3)
    for bad in (1, None, ["soft"]):
        a, b = both(voting=bad)
        with pytest.raises(Exception) as sk_exc:
            b.fit(X, y)
        with pytest.raises(Exception) as mlrs_exc:
            a.fit(X, y)
        assert type(mlrs_exc.value).__name__ == type(sk_exc.value).__name__
        assert options_in(str(mlrs_exc.value)) == options_in(str(sk_exc.value))
        assert repr(bad) in str(mlrs_exc.value)


def test_the_voting_constraint_runs_before_anything_looks_at_y():
    """Order is observable: sklearn's ``@_fit_context`` validates parameters
    ahead of ``fit``'s body, so a caller who passed both a bad ``voting`` and a
    continuous ``y`` sees the ``voting`` complaint from both libraries."""
    X, _ = host_design(3)
    y = np.linspace(0.0, 1.0, X.shape[0])
    a, b = both(voting="majority")
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    assert_same_error(mlrs_exc, sk_exc)
    assert "'voting'" in str(mlrs_exc.value)


def test_the_rust_core_owns_the_voting_parse():
    """One place decides what ``'soft'`` means. If the shim compared the literal
    itself, an unrecognized value could silently mean ``'hard'`` in one branch
    and raise in another."""
    ext = mlrs._load_ext()
    assert ext.voting_mode("hard") == "hard"
    assert ext.voting_mode("soft") == "soft"
    with pytest.raises(ValueError):
        ext.voting_mode("majority")


# =========================================================================== #
# STRING PARAMETER 2 — the `'drop'` sentinel
# =========================================================================== #


def test_the_drop_sentinel_is_the_literal_sklearn_compares_against():
    assert mlrs._load_ext().stacking_drop_sentinel() == "drop"


@pytest.mark.parametrize("voting", VOTING_VALUES)
@pytest.mark.parametrize("position", [0, 1, 2])
def test_drop_in_the_constructor_at_every_position(voting, position):
    """A dropped entry contributes no column, wherever it sits, under either
    mode.

    Parametrized over the position because an off-by-one in the kept-index
    bookkeeping would still produce the right ANSWER when the dropped entry is
    last — the surviving columns happen to be a prefix there — and over
    ``voting`` because the two routes lay their columns out differently (one per
    member versus ``n_classes`` per member).
    """
    X, y = host_design(3)
    estimators = sk_estimators()
    estimators[position] = (estimators[position][0], "drop")

    a, b = fit_both(X, y, estimators, voting=voting)
    assert_same(a, b, X)
    assert len(a.estimators_) == len(b.estimators_) == 2
    expected_cols = 2 if voting == "hard" else 2 * 3
    assert a.transform(X).shape == (N_SAMPLES, expected_cols)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


@pytest.mark.parametrize("name", ["lr", "nb", "knn"])
def test_drop_via_set_params_matches_the_constructor_route(name):
    """``set_params(lr='drop')`` is how sklearn documents this, so it is gated
    separately from the constructor spelling — they reach different code
    (``_replace_estimator`` versus the raw list)."""
    X, y = host_design(3)
    a, b = both(voting="soft")
    a.set_params(**{name: "drop"})
    b.set_params(**{name: "drop"})
    a.fit(X, y)
    b.fit(X, y)
    assert_same(a, b, X)
    assert a.named_estimators_[name] == b.named_estimators_[name] == "drop"


def test_a_dropped_slot_survives_in_named_estimators_as_the_string():
    X, y = host_design(3)
    estimators = sk_estimators()
    estimators[1] = ("nb", "drop")
    a, b = fit_both(X, y, estimators)

    assert list(a.named_estimators_) == list(b.named_estimators_) == ["lr", "nb", "knn"]
    assert a.named_estimators_["nb"] == "drop"
    assert type(a.named_estimators_["lr"]) is SkLogisticRegression
    assert type(a.named_estimators_["knn"]) is SkKNeighborsClassifier


@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_drop_keeps_its_weight_slot_and_the_survivors_keep_theirs(voting):
    """``weights`` is indexed against the FULL list, so ``set_params(name='drop')``
    stays usable on a weighted ensemble.

    The weights are deliberately asymmetric AND the middle one is enormous —
    with ``[1, 1, 1]`` a misaligned weight vector would be invisible, and with a
    small dropped weight the shift would not change the winner.
    """
    X, y = host_design(3)
    estimators = sk_estimators()
    estimators[1] = ("nb", "drop")
    a, b = fit_both(X, y, estimators, voting=voting, weights=[3.0, 100.0, 1.0])
    assert_same(a, b, X)

    # And the answer is genuinely the 3:1 blend of the two SURVIVORS.
    if voting == "soft":
        blocks = np.asarray([e.predict_proba(X) for e in a.estimators_])
        np.testing.assert_array_equal(
            a.predict_proba(X), np.average(blocks, axis=0, weights=[3.0, 1.0])
        )
    else:
        columns = np.asarray([e.predict(X) for e in a.estimators_]).T
        expected = np.apply_along_axis(
            lambda row: np.argmax(np.bincount(row, weights=[3.0, 1.0])), 1, columns
        )
        np.testing.assert_array_equal(a.predict(X), a.le_.inverse_transform(expected))


def test_all_estimators_dropped_is_sklearns_error_text():
    X, y = host_design(3)
    estimators = [(name, "drop") for name, _ in sk_estimators()]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "All estimators are dropped" in str(mlrs_exc.value)


@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_a_single_surviving_estimator_is_legal_and_is_that_estimator(voting):
    """One member is not a degenerate case: a majority of one is that one, and
    the average of one probability block is that block."""
    X, y = host_design(3)
    estimators = sk_estimators()
    estimators[1] = ("nb", "drop")
    estimators[2] = ("knn", "drop")
    a, b = fit_both(X, y, estimators, voting=voting)
    assert_same(a, b, X)
    np.testing.assert_array_equal(a.predict(X), a.estimators_[0].predict(X))


def test_an_arbitrary_string_that_is_not_drop_is_rejected():
    """``'dropped'``, ``'DROP'`` and friends are NOT the sentinel.

    sklearn compares against the exact literal, so a near-miss falls through to
    the classifier type check rather than silently disabling the entry — which
    is the failure mode that matters, because silently disabling one member is
    unobservable in the output shape when a sibling survives.

    The rejection arrives as an ``AttributeError``, not a ``ValueError``:
    ``is_classifier`` asks the object for ``__sklearn_tags__`` and a ``str`` has
    none. That is sklearn's own behaviour, quirk included, so the exception TYPE
    is asserted alongside the text.
    """
    X, y = host_design(3)
    for near_miss in ("dropped", "DROP", "Drop", " drop"):
        estimators = sk_estimators()
        estimators[0] = ("lr", near_miss)
        a, b = both(estimators)
        with pytest.raises(Exception) as sk_exc:
            b.fit(X, y)
        with pytest.raises(Exception) as mlrs_exc:
            a.fit(X, y)
        assert type(mlrs_exc.value) is type(sk_exc.value), near_miss
        assert str(mlrs_exc.value) == str(sk_exc.value), near_miss


# =========================================================================== #
# `flatten_transform`
# =========================================================================== #


@pytest.mark.parametrize("flatten", [True, False])
@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_flatten_transform_shapes_match_sklearn(voting, flatten):
    """``flatten_transform`` is consulted ONLY under soft voting.

    Under ``'hard'`` sklearn ignores it entirely and returns the ``(n, k)`` label
    matrix either way — a shim that honoured it there would change a shape
    sklearn does not.
    """
    X, y = host_design(3)
    a, b = fit_both(X, y, voting=voting, flatten_transform=flatten)
    got, expected = a.transform(X), b.transform(X)
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)
    if voting == "soft" and not flatten:
        assert got.shape == (3, N_SAMPLES, 3)
    elif voting == "soft":
        assert got.shape == (N_SAMPLES, 9)
    else:
        assert got.shape == (N_SAMPLES, 3)


def test_the_unflattened_soft_transform_is_the_stack_the_flattened_one_packs():
    """The two shapes have to carry the same numbers, or one of them is wrong.

    ``np.hstack`` on a ``(k, n, C)`` stack is member-major, which is also the
    order ``get_feature_names_out`` names — so this identity is what ties the
    names to the columns.
    """
    X, y = host_design(3)
    a, _ = fit_both(X, y, voting="soft", flatten_transform=False)
    stacked = a.transform(X)
    a.set_params(flatten_transform=True)
    np.testing.assert_array_equal(a.transform(X), np.hstack(stacked))


def test_feature_names_are_rejected_for_an_unflattened_soft_transform():
    """A 3-D output has no columns to name, and sklearn says so verbatim."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="soft", flatten_transform=False)
    with pytest.raises(ValueError) as sk_exc:
        b.get_feature_names_out()
    with pytest.raises(ValueError) as mlrs_exc:
        a.get_feature_names_out()
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "flatten_transform=False" in str(mlrs_exc.value)


def test_flatten_transform_false_is_harmless_under_hard_voting():
    """The rejection above is soft-voting-specific; under ``'hard'`` the names
    are produced as usual."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="hard", flatten_transform=False)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


def test_a_non_boolean_flatten_transform_is_sklearns_constraint_error():
    X, y = host_design(3)
    a, b = both(flatten_transform="yes")
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    assert_same_error(mlrs_exc, sk_exc)


# =========================================================================== #
# `weights`
# =========================================================================== #


@pytest.mark.parametrize(
    "weights",
    [
        None,
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 3.0],
        [1, 2, 3],
        [0.5, 0.25, 0.25],
        [3.0, -1.0, 1.0],
        [1e-8, 1.0, 1e8],
    ],
    ids=["none", "uniform", "asymmetric", "int", "fractional", "negative", "wide-range"],
)
@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_weights_oracle_match_across_kinds(voting, weights):
    """Including a NEGATIVE weight, which is where the two routes diverge most.

    ``np.bincount``'s result is only as long as the row's largest label, so a
    class above that maximum is not a candidate even when its implicit zero count
    would beat a negative tally. mlrs reproduces that bound per row; a full-width
    tally would answer differently on exactly this parametrization.
    """
    X, y = host_design(3)
    a, b = fit_both(X, y, voting=voting, weights=weights)
    assert_same(a, b, X)


def test_weights_accept_a_numpy_array_and_a_tuple():
    """sklearn ``zip``s ``weights`` with ``estimators``, so any sequence works."""
    X, y = host_design(3)
    for weights in (np.array([2.0, 1.0, 3.0]), (2.0, 1.0, 3.0)):
        a, b = fit_both(X, y, voting="soft", weights=weights)
        assert_same(a, b, X)


@pytest.mark.parametrize("data_dtype", [np.float32, np.float64])
def test_the_weights_dtype_propagates_into_predict_proba_exactly_as_numpy_does(
    data_dtype,
):
    """A ``float32`` weight array keeps an f32 problem in f32; a Python-float
    list promotes it to f64.

    This is why the Rust surface answers with weight POSITIONS rather than
    values — passing them through as ``f64`` would erase the distinction, and
    ``np.average`` propagates it into ``predict_proba``'s dtype.
    """
    X, y = host_design(3, dtype=data_dtype)
    for weights in (np.array([2.0, 1.0, 3.0], dtype=np.float32), [2.0, 1.0, 3.0]):
        a, b = fit_both(X, y, voting="soft", weights=weights)
        got, expected = a.predict_proba(X), b.predict_proba(X)
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("n_weights", [1, 2, 4, 6])
def test_a_weight_count_mismatch_is_sklearns_message_verbatim(n_weights):
    X, y = host_design(3)
    a, b = both(weights=[1.0] * n_weights)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert f"got {n_weights} weights, 3 estimators" in str(mlrs_exc.value)


def test_the_weight_count_is_checked_against_the_full_list_not_the_kept_one():
    """A dropped entry still needs its slot — three weights for three entries,
    one of which is ``'drop'``. Filtering before checking would reject a fit
    sklearn completes."""
    X, y = host_design(3)
    estimators = sk_estimators()
    estimators[1] = ("nb", "drop")
    a, b = fit_both(X, y, estimators, weights=[1.0, 5.0, 2.0])
    assert_same(a, b, X)
    # …and two weights for the two SURVIVORS is what sklearn rejects.
    a2, b2 = both(estimators, weights=[1.0, 2.0])
    with pytest.raises(ValueError) as sk_exc:
        b2.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a2.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_soft_voting_weights_summing_to_zero_raise_numpys_zero_division_error():
    """``np.average`` divides by ``w.sum()``; sklearn does not intercept it, so
    the exception a caller catches is numpy's ``ZeroDivisionError`` — NOT a
    ``ValueError``, and a shim that normalized it would break a migrated
    ``except`` clause."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="soft", weights=[1.0, -1.0, 0.0])
    with pytest.raises(ZeroDivisionError) as sk_exc:
        b.predict_proba(X)
    with pytest.raises(ZeroDivisionError) as mlrs_exc:
        a.predict_proba(X)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_hard_voting_weights_summing_to_zero_do_not_raise():
    """The counterpart, and the reason the zero-sum rule cannot be hoisted into
    a shared check: hard voting never DIVIDES — ``np.bincount`` tallies and
    ``argmax`` picks — so a zero weight sum is a legal, if odd, ensemble there
    while it is a ``ZeroDivisionError`` under soft voting."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="hard", weights=[1.0, -1.0, 0.0])
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_weights_do_not_affect_transform(voting):
    """sklearn's ``transform`` returns the members' RAW responses; the weighting
    is ``predict``'s alone."""
    X, y = host_design(3)
    a, _ = fit_both(X, y, voting=voting)
    unweighted = a.transform(X)
    a.set_params(weights=[9.0, 1.0, 1.0])
    np.testing.assert_array_equal(a.transform(X), unweighted)


def test_transform_never_reads_weights_even_when_they_became_invalid():
    """A ``weights`` mutated after the fit is the only way it can be wrong at
    transform time, and sklearn completes the call regardless."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="soft")
    a.set_params(weights=[1.0])
    b.set_params(weights=[1.0])
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


# =========================================================================== #
# `n_jobs` and `verbose` — value-neutral parameters
# =========================================================================== #


@pytest.mark.parametrize("n_jobs", [None, 1, 2, -1])
@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_n_jobs_is_value_neutral_over_host_members(voting, n_jobs):
    """Parallelism changes the schedule, never the answer."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting=voting, n_jobs=n_jobs)
    assert_same(a, b, X)


def test_n_jobs_over_an_mlrs_member_warns_and_falls_back_to_serial():
    """A joblib fan-out over device-holding estimators is unsafe here
    (``mlrs-no-parallel-fanout-over-device-estimators``), so mlrs reduces it to
    serial and SAYS SO rather than crashing in a worker."""
    X, y = host_design(3, dtype=conftest.default_float_dtype())
    estimators = [("nb", mlrs.GaussianNB()), ("lr", SkLogisticRegression(max_iter=500))]
    est = mlrs.VotingClassifier(estimators, n_jobs=2)
    with pytest.warns(UserWarning, match="n_jobs"):
        est.fit(X, y)
    assert len(est.estimators_) == 2


@pytest.mark.parametrize("verbose", [False, True])
def test_verbose_is_value_neutral_and_prints_sklearns_line(verbose, capsys):
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="soft", verbose=verbose)
    assert_same(a, b, X)
    printed = capsys.readouterr().out
    if verbose:
        assert "Processing lr" in printed
        assert "(1 of 3)" in printed
    else:
        assert printed == ""


# =========================================================================== #
# the target: which `y` this estimator accepts, and what it says about the rest
# =========================================================================== #


def test_a_continuous_target_is_sklearns_value_error():
    """sklearn splits the target rejection across two exception classes on
    purpose. A continuous target is an unfittable one — a ``ValueError``."""
    X, _ = host_design(3)
    y = np.linspace(0.0, 1.0, X.shape[0])
    a, b = both()
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "Unknown label type: continuous" in str(mlrs_exc.value)


def test_a_multilabel_target_is_a_not_implemented_error_not_a_value_error():
    """…while a target this estimator merely does not support yet is a
    ``NotImplementedError``. The CLASS is the assertion: a caller can tell "you
    gave me nonsense" from "I have not built that", and collapsing the two would
    lose that."""
    X, _ = host_design(3)
    y = np.zeros((X.shape[0], 3), dtype=np.int64)
    y[::2, 0] = 1
    y[1::3, 1] = 1
    a, b = both()
    with pytest.raises(NotImplementedError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(NotImplementedError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "VotingClassifier only supports binary or multiclass" in str(mlrs_exc.value)


@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_string_class_labels_round_trip_through_classes(voting):
    """``predict`` returns the caller's own labels, not the encoded indices.

    String labels are the case that would catch a shim returning the argmax
    directly: the values would still be "valid" integers and only their TYPE
    would give it away.
    """
    X, y_int = host_design(3)
    y = np.array(["setosa", "versicolor", "virginica"], dtype=object)[y_int]
    a, b = fit_both(X, y, voting=voting)
    assert_same(a, b, X)
    assert list(a.classes_) == ["setosa", "versicolor", "virginica"]
    assert set(np.unique(a.predict(X))) <= set(a.classes_)


def test_classes_are_the_label_encoders_sorted_order():
    """``classes_`` is ``LabelEncoder``'s, so it is SORTED regardless of the
    order the labels first appear in — which is what makes the encoded targets
    the members see comparable across libraries."""
    X, y_int = host_design(3)
    y = np.array([30, 10, 20])[y_int]
    a, b = fit_both(X, y)
    np.testing.assert_array_equal(a.classes_, [10, 20, 30])
    np.testing.assert_array_equal(a.classes_, b.classes_)
    assert isinstance(a.le_.classes_, np.ndarray)


@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_a_binary_target_with_two_classes_is_not_a_special_case(voting):
    """Binary is where a probability block is narrowest and where an
    off-by-one in the transform width would still land inside the array."""
    X, y = host_design(2)
    a, b = fit_both(X, y, voting=voting)
    assert_same(a, b, X)
    assert a.transform(X).shape == (N_SAMPLES, 3 if voting == "hard" else 6)


# =========================================================================== #
# structural validation (shared with the regressor, re-run on this class)
# =========================================================================== #


def test_a_regressor_member_is_rejected_as_a_non_classifier():
    """Unlike ``StackingClassifier`` — which deliberately accepts regressors for
    ordinal problems — a ``VotingClassifier`` requires classifiers, and says so
    with the offending CLASS name."""
    X, y = host_design(3)
    estimators = sk_estimators()
    estimators[0] = ("lr", SkLinearRegression())
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "should be a classifier" in str(mlrs_exc.value)


def test_duplicate_names_are_sklearns_error_text():
    X, y = host_design(3)
    estimators = [("dup", SkGaussianNB()), ("dup", SkLogisticRegression())]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


@pytest.mark.parametrize(
    "name", ["voting", "weights", "n_jobs", "flatten_transform", "verbose", "estimators"]
)
def test_a_name_colliding_with_a_constructor_argument_is_rejected(name):
    """``get_params`` flattens the members into the same namespace as the
    constructor arguments, so a member called ``voting`` would shadow the
    parameter. ``flatten_transform`` and ``voting`` are the two names this class
    adds over the regressor, which is why the list is re-parametrized here rather
    than shared."""
    X, y = host_design(3)
    estimators = [(name, SkGaussianNB()), ("knn", SkKNeighborsClassifier())]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_a_name_containing_a_double_underscore_is_rejected():
    X, y = host_design(3)
    estimators = [("a__b", SkGaussianNB()), ("knn", SkKNeighborsClassifier())]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


@pytest.mark.parametrize(
    "estimators",
    [[], [SkGaussianNB()], [("nb",)], [(1, SkGaussianNB())]],
    ids=["empty", "bare-estimator", "one-tuple", "non-string-name"],
)
def test_a_malformed_estimators_list_is_sklearns_error_text(estimators):
    X, y = host_design(3)
    a, b = both(estimators)
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    assert type(mlrs_exc.value) is type(sk_exc.value)
    assert str(mlrs_exc.value) == str(sk_exc.value)


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("estimators", "not-a-list"),
        ("weights", 3),
        ("weights", "abc"),
        ("n_jobs", 1.5),
        ("verbose", -1),
        ("flatten_transform", 1),
    ],
)
def test_a_constraint_violation_is_sklearns_invalid_parameter_error(param, value):
    X, y = host_design(3)
    a, b = both(**{param: value})
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    assert_same_error(mlrs_exc, sk_exc)
    assert str(mlrs_exc.value).startswith(f"The {param!r} parameter of VotingClassifier")


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("weights", np.array([1.0, 2.0, 3.0])),
        ("n_jobs", -1),
        ("verbose", 0),
        ("verbose", 3),
        ("flatten_transform", np.bool_(False)),
    ],
)
def test_the_constraint_layer_accepts_what_sklearn_accepts(param, value):
    """The other half of the constraint contract — a shim that over-tightened
    would reject fits sklearn completes, and that is invisible in a
    rejection-only test."""
    X, y = host_design(3)
    a, b = fit_both(X, y, **{param: value})
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_the_constraints_run_before_the_structural_check():
    """A non-list ``estimators`` is a TYPE complaint, not the structural
    "non-empty list of (string, estimator) tuples" message — sklearn's
    ``@_fit_context`` gets there first, and the two are different classes."""
    X, y = host_design(3)
    a, b = both("nonsense")
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    assert_same_error(mlrs_exc, sk_exc)
    assert "must be an instance of 'list'" in str(mlrs_exc.value)
    # …while an EMPTY list is the structural message, from a plain ValueError.
    with pytest.raises(ValueError) as empty:
        mlrs.VotingClassifier([]).fit(X, y)
    assert "should be a non-empty list" in str(empty.value)


def test_a_soft_vote_over_a_member_without_predict_proba_fails_where_sklearn_fails():
    """sklearn does NOT check for ``predict_proba`` at fit time, so an
    ``SVC(probability=False)`` member fits fine and raises from ``predict``.
    Reproduced including the timing: moving the check into ``fit`` would reject
    an ensemble a caller could legitimately use with ``voting='hard'``."""
    X, y = host_design(3)
    estimators = [("svc", SkSVC()), ("nb", SkGaussianNB())]
    a, b = fit_both(X, y, estimators, voting="soft")
    with pytest.raises(AttributeError) as sk_exc:
        b.predict(X)
    with pytest.raises(AttributeError) as mlrs_exc:
        a.predict(X)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    # …and the same ensemble under hard voting is perfectly usable.
    a.set_params(voting="hard")
    b.set_params(voting="hard")
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


# =========================================================================== #
# composition parameter handling
# =========================================================================== #


def test_get_params_deep_exposes_every_member_and_its_parameters():
    a, b = both(voting="soft", weights=[1.0, 2.0, 3.0])
    pa, pb = a.get_params(deep=True), b.get_params(deep=True)
    assert set(pa) == set(pb)
    assert pa["voting"] == "soft"
    assert "lr__max_iter" in pa and "knn__n_neighbors" in pa


def test_get_params_shallow_is_the_constructor_signature():
    a, b = both()
    assert set(a.get_params(deep=False)) == set(b.get_params(deep=False))
    assert set(a.get_params(deep=False)) == {
        "estimators",
        "voting",
        "weights",
        "n_jobs",
        "flatten_transform",
        "verbose",
    }


def test_set_params_reaches_into_a_member():
    X, y = host_design(3)
    a, b = both(voting="soft")
    a.set_params(knn__n_neighbors=1, voting="hard")
    b.set_params(knn__n_neighbors=1, voting="hard")
    a.fit(X, y)
    b.fit(X, y)
    assert a.named_estimators_["knn"].n_neighbors == 1
    assert_same(a, b, X)


def test_named_estimators_reads_the_unfitted_list():
    a, _ = both()
    assert list(a.named_estimators) == ["lr", "nb", "knn"]


def test_clone_round_trips_every_parameter():
    a, _ = both(
        voting="soft", weights=[1.0, 2.0, 3.0], n_jobs=2, flatten_transform=False,
        verbose=True,
    )
    c = clone(a)
    assert c.voting == "soft"
    assert c.flatten_transform is False
    assert c.weights == [1.0, 2.0, 3.0]
    assert c.n_jobs == 2 and c.verbose is True
    assert [name for name, _ in c.estimators] == ["lr", "nb", "knn"]


# =========================================================================== #
# introspection
# =========================================================================== #


def test_n_features_in_matches_and_is_the_original_width():
    X, y = host_design(3)
    a, b = fit_both(X, y)
    assert a.n_features_in_ == b.n_features_in_ == N_FEATURES


def test_n_features_in_on_an_unfitted_estimator_is_sklearns_attribute_error():
    """An ``AttributeError``, so ``hasattr`` is ``False`` — and with the voting
    layer's own wording, which differs from the stacking layer's."""
    a, b = both()
    assert not hasattr(a, "n_features_in_")
    with pytest.raises(AttributeError) as sk_exc:
        b.n_features_in_
    with pytest.raises(AttributeError) as mlrs_exc:
        a.n_features_in_
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "has no n_features_in_ attribute." in str(mlrs_exc.value)


@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_feature_names_out_matches_sklearn_and_names_every_column(voting):
    """The names and the transform width have to agree, or the caller's
    DataFrame silently mislabels its columns."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting=voting)
    names = a.get_feature_names_out()
    assert list(names) == list(b.get_feature_names_out())
    assert names.dtype == object
    assert len(names) == a.transform(X).shape[1]
    if voting == "hard":
        assert list(names) == [
            "votingclassifier_lr",
            "votingclassifier_nb",
            "votingclassifier_knn",
        ]
    else:
        # Member-major, class index appended with NO separator.
        assert list(names[:4]) == [
            "votingclassifier_lr0",
            "votingclassifier_lr1",
            "votingclassifier_lr2",
            "votingclassifier_nb0",
        ]


def test_feature_names_out_validates_input_features_and_then_discards_them():
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="soft")
    good = [f"f{i}" for i in range(N_FEATURES)]
    assert list(a.get_feature_names_out(good)) == list(b.get_feature_names_out(good))
    # …and a wrong-length one is still rejected, which is what "validated" means.
    with pytest.raises(ValueError) as sk_exc:
        b.get_feature_names_out(["only-one"])
    with pytest.raises(ValueError) as mlrs_exc:
        a.get_feature_names_out(["only-one"])
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_feature_names_out_on_an_unfitted_estimator_raises_not_fitted():
    from sklearn.exceptions import NotFittedError

    a, b = both()
    with pytest.raises(NotFittedError):
        b.get_feature_names_out()
    with pytest.raises(NotFittedError):
        a.get_feature_names_out()


def test_feature_names_in_is_lifted_off_a_fitted_member():
    pd = pytest.importorskip("pandas")
    X, y = host_design(3)
    frame = pd.DataFrame(X, columns=[f"c{i}" for i in range(N_FEATURES)])
    a, b = fit_both(frame, y)
    np.testing.assert_array_equal(a.feature_names_in_, b.feature_names_in_)


def test_the_transformer_tag_clears_preserves_dtype():
    """``transform`` returns labels under hard voting and probabilities under
    soft, so sklearn clears the list outright — and sklearn's own estimator
    checks read this tag."""
    from sklearn.utils import get_tags

    a, b = both()
    assert get_tags(a).transformer_tags.preserves_dtype == []
    assert get_tags(b).transformer_tags.preserves_dtype == []


# =========================================================================== #
# mlrs sub-estimators — the real deployment shape
# =========================================================================== #


@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_mlrs_members_compose_and_match_an_all_sklearn_reference(voting):
    """The device path, end to end.

    The tolerance here is ``conftest.live_atol()`` rather than exact equality:
    the MEMBERS' own arithmetic is inside the comparison, and an mlrs
    ``GaussianNB`` on a device is not bit-identical to sklearn's. The voting
    layer itself is still gated exactly — that is what every cell above does.
    """
    dtype = conftest.default_float_dtype()
    X, y = host_design(3, dtype=dtype)
    estimators = [
        ("nb", mlrs.GaussianNB()),
        ("knn", mlrs.KNeighborsClassifier(n_neighbors=5)),
    ]
    reference = [
        ("nb", SkGaussianNB()),
        ("knn", SkKNeighborsClassifier(n_neighbors=5)),
    ]
    a = mlrs.VotingClassifier(estimators, voting=voting).fit(X, y)
    b = SkVotingClassifier(reference, voting=voting).fit(X, y)
    if voting == "soft":
        np.testing.assert_allclose(
            a.predict_proba(X), b.predict_proba(X), atol=conftest.live_atol()
        )
    # A handful of boundary rows may land differently once the members' own
    # arithmetic differs, so the LABELS are compared as an agreement rate rather
    # than exactly — the exact-label contract lives in the sklearn-only cells.
    agreement = np.mean(a.predict(X) == b.predict(X))
    assert agreement > 0.97, agreement


def test_an_mlrs_and_an_sklearn_member_can_be_mixed():
    """A composition is not required to be homogeneous, and this is the shape a
    caller migrating one estimator at a time actually has."""
    dtype = conftest.default_float_dtype()
    X, y = host_design(3, dtype=dtype)
    estimators = [
        ("mlrs_nb", mlrs.GaussianNB()),
        ("sk_tree", SkDecisionTreeClassifier(random_state=0)),
    ]
    est = mlrs.VotingClassifier(estimators, voting="soft", weights=[2.0, 1.0]).fit(X, y)
    proba = est.predict_proba(X)
    assert proba.shape == (N_SAMPLES, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(est.predict(X), est.classes_[proba.argmax(axis=1)])
