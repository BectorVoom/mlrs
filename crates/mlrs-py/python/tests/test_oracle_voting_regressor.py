"""VotingRegressor oracle harness (VOTE-01: the full parameter surface).

Every test here compares :class:`mlrs.VotingRegressor` against a LIVE
:class:`sklearn.ensemble.VotingRegressor` built with the same arguments, on the
same data, in the same process. There are no committed ``.npz`` fixtures:
voting has no numerics of its own — it fits other estimators and averages them
— so what needs gating is the *composition*, and the only reference for that is
sklearn itself. A stored blob would freeze one sklearn version's answer where a
live comparison tracks it.

## The string-valued parameter surface (the point of this file)

``VotingRegressor``'s constructor is ``(estimators, *, weights=None,
n_jobs=None, verbose=False)``. Unlike ``StackingRegressor`` — which has
``cv="prefit"`` on top — there is exactly ONE place a caller supplies a string,
and it is not a scalar parameter at all:

  ===============================  =========================================
  string                           what it selects
  ===============================  =========================================
  ``estimators=[(name, "drop")]``  disable one entry: it is never fitted and
                                   contributes no column, but its slot
                                   survives in ``named_estimators_`` as the
                                   string ``'drop'`` AND its slot in
                                   ``weights`` still has to be supplied
  ===============================  =========================================

``weights``, ``n_jobs`` and ``verbose`` take no strings — ``weights`` is
array-like or ``None``, ``n_jobs`` is an int or ``None``, ``verbose`` is a bool
— so this one sentinel is the whole string surface, and it is exercised here in
every combination that can interact with it: each position in the list, both
spellings (constructor argument and ``set_params``), with and without
``weights``, through ``get_feature_names_out`` / ``named_estimators_`` /
``n_features_in_`` / ``transform``, and on the rejection path where every entry
is dropped.

mlrs adds one further string surface of its own, ``MLRS_VOTING_ENGINE``
(``numpy`` / ``host`` / ``device``). It is an aggregation-arm A/B knob rather
than a constructor parameter, and it is oracle-tested per arm — including the
``'drop'`` sentinel again on every arm — in ``test_voting_engine.py``.

The rejection messages are asserted to be sklearn's own text, not merely "some
ValueError": they are the same strings the Rust core owns
(``crates/mlrs-algos/tests/voting_test.rs``), so a divergence in either layer
fails here.

## Why the value assertions are EXACT

``np.average`` is reproduced operation for operation (products, then a
left-to-right row sum, then a DIVISION by the weight sum, all in the input
dtype), so mlrs and sklearn must agree bit for bit rather than within 1e-5 on
the sklearn-only cells. That is a much stronger assertion than the project's
tolerance contract and is the only one that catches a reassociated
accumulation. The ``mlrs``-sub-estimator cells fall back to
``conftest.live_atol()``, because there the MEMBERS' own device arithmetic is in
the comparison.

## Backend gating

Two designs run side by side:

* **sklearn-only sub-estimators** — pure host composition. The voting layer is
  dtype-independent host bookkeeping, so these cells run identically (and
  EXACTLY) on cpu / wgpu / rocm / cuda. All the string-parameter coverage lives
  here, so no backend can end up with a vacuous run of it.
* **mlrs sub-estimators** — the real deployment shape, where the member fits go
  to the device. These use ``conftest.default_float_dtype()`` /
  ``conftest.live_atol()`` rather than hardcoding float64, which keeps them from
  turning red at ingress on an f64-incapable backend instead of comparing
  anything.

Req: VOTE-01 (parameter surface), VOTE-BIND-01 (the Rust structural core).
"""

import re

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import VotingRegressor as SkVotingRegressor
from sklearn.linear_model import (
    LinearRegression as SkLinearRegression,
    LogisticRegression as SkLogisticRegression,
    Ridge as SkRidge,
)
from sklearn.neighbors import KNeighborsRegressor as SkKNeighborsRegressor

import conftest

mlrs = pytest.importorskip("mlrs")


N_SAMPLES = 200
N_FEATURES = 5
SEED = 42


# --------------------------------------------------------------------------- #
# designs
# --------------------------------------------------------------------------- #


def host_design(dtype=np.float64, n_samples=N_SAMPLES):
    """A well-conditioned linear regression problem, plus noise."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, N_FEATURES)).astype(dtype)
    beta = np.array([3.0, -1.5, 0.0, 0.75, 2.0], dtype=dtype)
    y = (X @ beta).astype(dtype)
    return X, (y + (0.05 * rng.standard_normal(n_samples)).astype(dtype))


def sk_estimators():
    """Three sklearn members that genuinely DISAGREE.

    A voting ensemble whose members all predict the same thing cannot
    distinguish a correct weighting from a broken one — the average is the same
    either way. ``KNeighborsRegressor`` is deliberately in the list because it
    is not linear, so a mis-ordered weight vector moves the prediction.
    """
    return [
        ("lr", SkLinearRegression()),
        ("ridge", SkRidge(alpha=25.0)),
        ("knn", SkKNeighborsRegressor(n_neighbors=3)),
    ]


def both(estimators=None, **kwargs):
    """One mlrs estimator and one sklearn estimator with identical arguments."""
    if estimators is None:
        estimators = sk_estimators()
    return (
        mlrs.VotingRegressor(estimators, **kwargs),
        SkVotingRegressor(estimators, **kwargs),
    )


def fit_both(X, y, estimators=None, **kwargs):
    a, b = both(estimators, **kwargs)
    return a.fit(X, y), b.fit(X, y)


def assert_same(a, b, X):
    """``predict`` and ``transform`` agree BIT FOR BIT.

    See the module docstring: `np.average` is reproduced operation for
    operation, so anything looser would let a reassociated accumulation through.
    """
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


# =========================================================================== #
# `estimators` and the `'drop'` sentinel — THE string-valued surface
# =========================================================================== #


def test_the_drop_sentinel_is_the_literal_sklearn_compares_against():
    """The Rust core and sklearn must agree on the spelling itself."""
    assert mlrs._load_ext().stacking_drop_sentinel() == "drop"


@pytest.mark.parametrize("position", [0, 1, 2])
def test_drop_in_the_constructor_at_every_position(position):
    """A dropped entry contributes no column, wherever it sits.

    Parametrized over the position because an off-by-one in the kept-index
    bookkeeping would still produce the right ANSWER when the dropped entry is
    last — the surviving columns happen to be a prefix there.
    """
    X, y = host_design()
    estimators = sk_estimators()
    estimators[position] = (estimators[position][0], "drop")

    a, b = fit_both(X, y, estimators)
    assert_same(a, b, X)
    assert len(a.estimators_) == len(b.estimators_) == 2
    assert a.transform(X).shape == (N_SAMPLES, 2)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


@pytest.mark.parametrize("name", ["lr", "ridge", "knn"])
def test_drop_via_set_params_matches_the_constructor_route(name):
    """``set_params(lr='drop')`` is how sklearn documents this, so it is gated
    separately from the constructor spelling — they reach different code
    (``_replace_estimator`` versus the raw list)."""
    X, y = host_design()
    a, b = both()
    a.set_params(**{name: "drop"})
    b.set_params(**{name: "drop"})
    a.fit(X, y)
    b.fit(X, y)
    assert_same(a, b, X)
    assert a.named_estimators_[name] == b.named_estimators_[name] == "drop"


def test_a_dropped_slot_survives_in_named_estimators_as_the_string():
    """The slot is KEPT and holds the sentinel — it does not disappear."""
    X, y = host_design()
    estimators = sk_estimators()
    estimators[1] = ("ridge", "drop")
    a, b = fit_both(X, y, estimators)

    assert list(a.named_estimators_) == list(b.named_estimators_)
    assert list(a.named_estimators_) == ["lr", "ridge", "knn"]
    assert a.named_estimators_["ridge"] == "drop"
    # …and the surviving slots hold FITTED estimators, in list order.
    assert type(a.named_estimators_["lr"]) is SkLinearRegression
    assert type(a.named_estimators_["knn"]) is SkKNeighborsRegressor


def test_drop_keeps_its_weight_slot_and_the_survivors_keep_theirs():
    """The rule that makes ``set_params(name='drop')`` usable on a weighted
    ensemble: ``weights`` is indexed against the FULL list, and the dropped
    entry's weight is discarded rather than shifting the others along.

    The weights here are deliberately asymmetric — with ``[1, 1, 1]`` a
    misaligned weight vector would be invisible.
    """
    X, y = host_design()
    estimators = sk_estimators()
    estimators[1] = ("ridge", "drop")
    a, b = fit_both(X, y, estimators, weights=[3.0, 100.0, 1.0])
    assert_same(a, b, X)

    # And the answer is genuinely the 3:1 blend of the two SURVIVORS — not the
    # 3:100 or 100:1 pairing a shifted vector would produce.
    cols = np.asarray([e.predict(X) for e in a.estimators_]).T
    np.testing.assert_array_equal(
        a.predict(X), np.average(cols, axis=1, weights=[3.0, 1.0])
    )


def test_all_estimators_dropped_is_sklearns_error_text():
    X, y = host_design()
    estimators = [(name, "drop") for name, _ in sk_estimators()]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "All estimators are dropped" in str(mlrs_exc.value)


def test_a_single_surviving_estimator_is_legal_and_is_that_estimator():
    """One member is not a degenerate case — the average of one column is that
    column, and sklearn allows it."""
    X, y = host_design()
    estimators = sk_estimators()
    estimators[1] = ("ridge", "drop")
    estimators[2] = ("knn", "drop")
    a, b = fit_both(X, y, estimators)
    assert_same(a, b, X)
    np.testing.assert_array_equal(a.predict(X), a.estimators_[0].predict(X))


def test_an_arbitrary_string_that_is_not_drop_is_rejected():
    """``'dropped'``, ``'DROP'`` and friends are NOT the sentinel.

    sklearn compares against the exact literal, so a near-miss falls through to
    the regressor type check rather than silently disabling the entry — which is
    the failure mode that matters, because silently disabling one is unobservable
    in the output shape when a sibling survives.

    The rejection arrives as an ``AttributeError``, not a ``ValueError``:
    ``is_regressor`` asks the object for ``__sklearn_tags__`` and a ``str`` has
    none. That is sklearn's own behaviour, quirk included, and the assertion is
    on the exception TYPE as well as the text so a shim that "helpfully"
    normalized it to ``ValueError`` would fail here.
    """
    X, y = host_design()
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
def test_weights_oracle_match_across_kinds(weights):
    """Every shape of weight vector sklearn accepts.

    ``int`` is separate from ``float`` because numpy promotes an integer weight
    array to float64 during the average and a shim that cast it early could
    change the result on an f32 problem. The NEGATIVE entry is here because
    numpy permits it — only a zero SUM is an error — and a defensive guard on
    individual weights would reject a fit sklearn completes.
    """
    X, y = host_design()
    a, b = fit_both(X, y, weights=weights)
    assert_same(a, b, X)


@pytest.mark.parametrize("k", [1, 2, 5, 9, 16])
@pytest.mark.parametrize("weighted", [False, True], ids=["uniform", "weighted"])
def test_many_members_still_match_numpys_reduction_exactly(k, weighted):
    """The member count crosses numpy's pairwise-summation threshold.

    ``np.add.reduce`` blocks pairwise above 8 elements when it can, which
    REASSOCIATES the sum. mlrs accumulates left to right unconditionally, so if
    numpy's reduction along this axis were pairwise the two would diverge in the
    last bits above ``k = 8`` — and every other cell in this file uses ``k = 3``,
    which would never notice.

    It does not diverge, because the axis being reduced is the ``k`` axis of an
    ``(n, k)`` array numpy built by transposing ``(k, n)`` — strided, not
    contiguous. That is a fact about the shape this estimator hands numpy, not a
    guarantee numpy offers, so it is TESTED here (exactly, on both sides of the
    threshold) rather than reasoned about in a docstring. Anything that changes
    the shape handed to numpy will fail this first.
    """
    X, y = host_design()
    estimators = [
        (f"r{i}", SkRidge(alpha=float(i + 1) * 3.0)) for i in range(k)
    ]
    weights = [1.0 + i * 0.37 for i in range(k)] if weighted else None
    a, b = fit_both(X, y, estimators, weights=weights)
    assert_same(a, b, X)


def test_weights_accept_a_numpy_array_and_a_tuple():
    """sklearn declares ``weights`` as "array-like", so a list is not the only
    spelling; a shim that only handled lists would fail on the natural one."""
    X, y = host_design()
    for w in (np.array([2.0, 1.0, 3.0]), (2.0, 1.0, 3.0), np.array([2, 1, 3])):
        a, b = fit_both(X, y, weights=w)
        assert_same(a, b, X)


@pytest.mark.parametrize(
    "weight_dtype", [np.float32, np.float64, np.int32, np.int64]
)
@pytest.mark.parametrize("data_dtype", [np.float32, np.float64])
def test_the_weights_dtype_propagates_into_the_result_exactly_as_numpy_does(
    weight_dtype, data_dtype
):
    """``np.average`` infers its result dtype from the columns AND the weights.

    A ``float32`` weight array over ``float32`` predictions stays in
    ``float32``; make either one wider and the answer widens. A shim that
    normalized ``weights`` to Python floats on the way through — the obvious
    thing to do at an FFI boundary — would silently promote every f32 problem to
    f64 and be *wrong by one dtype* while still passing a 1e-5 comparison. So
    this asserts the dtype as well as the values.
    """
    X, y = host_design(data_dtype)
    w = np.array([2, 1, 3], dtype=weight_dtype)
    a, b = fit_both(X, y, weights=w)
    assert a.predict(X).dtype == b.predict(X).dtype
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize("n_weights", [1, 2, 4, 6])
def test_a_weight_count_mismatch_is_sklearns_message_verbatim(n_weights):
    X, y = host_design()
    weights = [1.0] * n_weights
    a, b = both(weights=weights)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert str(mlrs_exc.value) == (
        "Number of `estimators` and weights must be equal; got "
        f"{n_weights} weights, 3 estimators"
    )


def test_the_weight_count_is_checked_against_the_full_list_not_the_kept_one():
    """Three weights and three entries stays legal when one is dropped — and
    TWO weights (the kept count) is the error."""
    X, y = host_design()
    estimators = sk_estimators()
    estimators[1] = ("ridge", "drop")

    a, b = fit_both(X, y, estimators, weights=[1.0, 2.0, 3.0])
    assert_same(a, b, X)

    a, b = both(estimators, weights=[1.0, 3.0])
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_weights_summing_to_zero_raise_numpys_zero_division_error():
    """numpy answers a zero weight sum with ``ZeroDivisionError``, not
    ``ValueError`` — and the distinction is visible to any caller whose
    ``except`` clause came over from sklearn.

    The error surfaces from ``predict``, not ``fit``: sklearn's ``fit`` only
    checks the LENGTH, and the sum is not consulted until the average is taken.
    """
    X, y = host_design()
    a, b = fit_both(X, y, weights=[1.0, -1.0, 0.0])
    with pytest.raises(ZeroDivisionError) as sk_exc:
        b.predict(X)
    with pytest.raises(ZeroDivisionError) as mlrs_exc:
        a.predict(X)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert str(mlrs_exc.value) == "Weights sum to zero, can't be normalized"
    # `transform` does NOT weight, so it still answers.
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


def test_transform_never_reads_weights_even_when_they_became_invalid():
    """``weights`` mutated AFTER the fit is the only way it can be wrong at
    aggregation time, and sklearn's ``transform`` — which never reads them —
    completes anyway. A shim that resolved the weights on every aggregation
    would raise here."""
    X, y = host_design()
    a, b = fit_both(X, y, weights=[2.0, 1.0, 3.0])
    a.weights = [1.0]
    b.weights = [1.0]
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


def test_weights_do_not_affect_transform():
    """``transform`` returns the members' RAW predictions; only ``predict``
    weights them. A shim that applied the weights in both would be wrong in a
    way that is invisible from ``predict`` alone."""
    X, y = host_design()
    a, _ = fit_both(X, y, weights=[9.0, 1.0, 1.0])
    c, _ = fit_both(X, y, weights=None)
    np.testing.assert_array_equal(a.transform(X), c.transform(X))
    assert not np.array_equal(a.predict(X), c.predict(X))


# =========================================================================== #
# `n_jobs`
# =========================================================================== #


@pytest.mark.parametrize("n_jobs", [None, 1, 2, -1])
def test_n_jobs_is_value_neutral_over_host_members(n_jobs):
    """A parallelism parameter that changes the ANSWER is exactly the defect an
    oracle suite exists to catch. Host members only, because a joblib fan-out
    over an mlrs member is reduced to serial by design (see
    ``_effective_n_jobs``) and would test the warning rather than the fan-out.
    """
    X, y = host_design()
    a, b = fit_both(X, y, n_jobs=n_jobs)
    assert_same(a, b, X)


def test_n_jobs_over_an_mlrs_member_warns_and_falls_back_to_serial():
    """mlrs's documented divergence from sklearn: a fitted mlrs estimator owns a
    non-picklable device handle, so the fan-out is refused with a warning rather
    than crashing in the worker. The RESULT is still sklearn's."""
    X, y = host_design(conftest.default_float_dtype())
    estimators = [("lr", SkLinearRegression()), ("mlrs_ridge", mlrs.Ridge())]
    a = mlrs.VotingRegressor(estimators, n_jobs=2)
    with pytest.warns(UserWarning, match="n_jobs is ignored"):
        a.fit(X, y)
    serial = mlrs.VotingRegressor(estimators, n_jobs=None).fit(X, y)
    np.testing.assert_allclose(a.predict(X), serial.predict(X), atol=conftest.live_atol())


# =========================================================================== #
# `verbose`
# =========================================================================== #


@pytest.mark.parametrize("verbose", [False, True])
def test_verbose_is_value_neutral_and_prints_sklearns_line(verbose, capsys):
    """``verbose=True`` prints one ``[Voting] (i of n) Processing <name>`` line
    per member — the same text, in the same layout, sklearn emits."""
    X, y = host_design()
    a, b = both(verbose=verbose)
    b.fit(X, y)
    sk_out = capsys.readouterr().out
    a.fit(X, y)
    mlrs_out = capsys.readouterr().out
    assert_same(a, b, X)

    if not verbose:
        assert mlrs_out == sk_out == ""
        return
    # The elapsed times differ run to run, so compare the lines with the time
    # field and the dot padding normalized away.
    def shape(text):
        return [
            re.sub(r"\.+", ".", re.sub(r"total=\s*[\d.]+m?s", "total=T", line))
            for line in text.strip().splitlines()
        ]

    assert shape(mlrs_out) == shape(sk_out)
    assert len(shape(mlrs_out)) == 3
    assert shape(mlrs_out)[0].startswith("[Voting] ")
    assert "(1 of 3) Processing lr" in shape(mlrs_out)[0]


# =========================================================================== #
# name validation (shared with stacking, but reachable through THIS class)
# =========================================================================== #


def test_duplicate_names_are_sklearns_error_text():
    X, y = host_design()
    estimators = [("lr", SkLinearRegression()), ("lr", SkRidge())]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "Names provided are not unique" in str(mlrs_exc.value)


@pytest.mark.parametrize("name", ["weights", "n_jobs", "verbose", "estimators"])
def test_a_name_colliding_with_a_constructor_argument_is_rejected(name):
    """The collision set is THIS class's ``get_params`` keys — which are not
    stacking's. ``weights`` is a legal member name on a ``StackingRegressor``
    and an error here, and that asymmetry is the thing worth gating.
    """
    X, y = host_design()
    estimators = [(name, SkLinearRegression()), ("ridge", SkRidge())]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "conflict with constructor arguments" in str(mlrs_exc.value)


def test_a_name_containing_a_double_underscore_is_rejected():
    X, y = host_design()
    estimators = [("a__b", SkLinearRegression()), ("ridge", SkRidge())]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "must not contain __" in str(mlrs_exc.value)


@pytest.mark.parametrize(
    "estimators", [[], [SkLinearRegression()], [("lr",)]],
    ids=["empty", "bare-estimator", "one-tuple"],
)
def test_a_malformed_estimators_list_is_sklearns_error_text(estimators):
    X, y = host_design()
    a, b = both(estimators)
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    assert type(mlrs_exc.value) is type(sk_exc.value)
    assert str(mlrs_exc.value) == str(sk_exc.value)


# =========================================================================== #
# `_parameter_constraints` — sklearn's PRE-fit type layer
# =========================================================================== #
#
# sklearn applies these through `@_fit_context`, which runs BEFORE `fit`'s body.
# The order is observable: a non-list `estimators` reports the TYPE error, not
# the structural "should be a non-empty list of (string, estimator) tuples" one
# that the composition check would otherwise reach first — and they are
# different exception CLASSES.


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("estimators", "not-a-list"),
        ("estimators", ("lr", SkLinearRegression())),
        ("estimators", 5),
        ("weights", "abc"),
        ("weights", 3),
        ("n_jobs", "two"),
        ("n_jobs", 1.5),
        ("verbose", "loud"),
        ("verbose", -1),
        ("verbose", None),
    ],
)
def test_a_constraint_violation_is_sklearns_invalid_parameter_error(param, value):
    X, y = host_design()
    kwargs = {"estimators": sk_estimators()}
    kwargs[param] = value
    a = mlrs.VotingRegressor(**kwargs)
    b = SkVotingRegressor(**kwargs)
    with pytest.raises(Exception) as sk_exc:
        b.fit(X, y)
    with pytest.raises(Exception) as mlrs_exc:
        a.fit(X, y)
    assert type(mlrs_exc.value).__name__ == type(sk_exc.value).__name__
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert str(mlrs_exc.value).startswith(f"The {param!r} parameter of VotingRegressor")


@pytest.mark.parametrize(
    ("param", "value"),
    [("verbose", 2), ("verbose", 0), ("verbose", np.True_), ("n_jobs", 1)],
)
def test_the_constraint_layer_accepts_what_sklearn_accepts(param, value):
    """The mirror direction: a constraint that is TIGHTER than sklearn's would
    reject fits sklearn completes, which is the worse failure of the two."""
    X, y = host_design()
    kwargs = {"estimators": sk_estimators()}
    kwargs[param] = value
    a, b = both(**{param: value})
    a.fit(X, y)
    b.fit(X, y)
    assert_same(a, b, X)


def test_the_constraints_run_before_the_structural_check():
    """``estimators=[]`` is a structural ``ValueError``; ``estimators='x'`` is a
    constraint ``InvalidParameterError``. A shim that ran them in the other
    order would report the structural message for both."""
    X, y = host_design()
    with pytest.raises(ValueError) as empty:
        mlrs.VotingRegressor([]).fit(X, y)
    assert "should be a non-empty list" in str(empty.value)
    with pytest.raises(Exception) as wrong_type:
        mlrs.VotingRegressor("x").fit(X, y)
    assert "must be an instance of 'list'" in str(wrong_type.value)


def test_a_classifier_member_is_rejected_as_a_non_regressor():
    """Unlike ``StackingClassifier`` — which deliberately accepts regressors —
    a ``VotingRegressor`` averages numbers and has no use for a classifier."""
    X, y = host_design()
    estimators = [("lr", SkLinearRegression()), ("clf", SkLogisticRegression())]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "should be a regressor" in str(mlrs_exc.value)


# =========================================================================== #
# the composition parameter surface (`get_params` / `set_params` / `clone`)
# =========================================================================== #


def test_get_params_deep_exposes_every_member_and_its_parameters():
    a, b = both()
    pa, pb = a.get_params(deep=True), b.get_params(deep=True)
    assert set(pa) == set(pb)
    assert "lr" in pa and "ridge__alpha" in pa
    assert pa["ridge__alpha"] == pb["ridge__alpha"] == 25.0


def test_get_params_shallow_is_the_constructor_signature():
    a, b = both()
    assert set(a.get_params(deep=False)) == set(b.get_params(deep=False))
    assert set(a.get_params(deep=False)) == {
        "estimators",
        "weights",
        "n_jobs",
        "verbose",
    }


def test_set_params_reaches_into_a_member():
    """What makes a ``GridSearchCV`` over ``ridge__alpha`` work."""
    X, y = host_design()
    a, b = both()
    a.set_params(ridge__alpha=0.001)
    b.set_params(ridge__alpha=0.001)
    a.fit(X, y)
    b.fit(X, y)
    assert_same(a, b, X)
    assert a.named_estimators_["ridge"].alpha == 0.001


def test_named_estimators_reads_the_unfitted_list():
    a, b = both()
    assert list(a.named_estimators) == list(b.named_estimators)
    assert type(a.named_estimators["lr"]) is SkLinearRegression


def test_clone_round_trips_every_parameter():
    a = mlrs.VotingRegressor(sk_estimators(), weights=[2.0, 1.0, 3.0], n_jobs=2, verbose=True)
    c = clone(a)
    assert c.weights == [2.0, 1.0, 3.0]
    assert c.n_jobs == 2 and c.verbose is True
    assert [n for n, _ in c.estimators] == ["lr", "ridge", "knn"]
    # `clone` clones the members too, so they are equal-but-not-identical.
    assert c.estimators[0][1] is not a.estimators[0][1]


# =========================================================================== #
# fitted introspection
# =========================================================================== #


def test_n_features_in_matches_and_is_the_original_width():
    X, y = host_design()
    a, b = fit_both(X, y)
    assert a.n_features_in_ == b.n_features_in_ == N_FEATURES


def test_n_features_in_on_an_unfitted_estimator_is_sklearns_attribute_error():
    """sklearn's VOTING layer words this differently from its stacking layer
    (``has no n_features_in_ attribute.`` versus ``has no attribute
    n_features_in_``), so the two shims must not share one message."""
    a, b = both()
    with pytest.raises(AttributeError) as sk_exc:
        b.n_features_in_
    with pytest.raises(AttributeError) as mlrs_exc:
        a.n_features_in_
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert str(mlrs_exc.value) == "VotingRegressor object has no n_features_in_ attribute."
    assert not hasattr(a, "n_features_in_")


def test_feature_names_out_is_class_underscore_name_per_kept_member():
    X, y = host_design()
    a, b = fit_both(X, y)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
    assert list(a.get_feature_names_out()) == [
        "votingregressor_lr",
        "votingregressor_ridge",
        "votingregressor_knn",
    ]
    assert a.get_feature_names_out().dtype == b.get_feature_names_out().dtype


def test_feature_names_out_validates_input_features_and_then_discards_them():
    """sklearn calls ``_check_feature_names_in(..., generate_names=False)``: the
    argument is checked for LENGTH and then contributes nothing to the output.
    A shim that ignored it entirely would accept a wrong-width argument."""
    X, y = host_design()
    a, b = fit_both(X, y)
    good = [f"f{i}" for i in range(N_FEATURES)]
    assert list(a.get_feature_names_out(good)) == list(b.get_feature_names_out(good))
    assert list(a.get_feature_names_out(good)) == list(a.get_feature_names_out())

    bad = ["only", "two"]
    with pytest.raises(ValueError) as sk_exc:
        b.get_feature_names_out(bad)
    with pytest.raises(ValueError) as mlrs_exc:
        a.get_feature_names_out(bad)
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
    X, y = host_design()
    frame = pd.DataFrame(X, columns=[f"c{i}" for i in range(N_FEATURES)])
    a, b = fit_both(frame, y)
    np.testing.assert_array_equal(a.feature_names_in_, b.feature_names_in_)
    np.testing.assert_array_equal(
        a.get_feature_names_out(list(frame.columns)),
        b.get_feature_names_out(list(frame.columns)),
    )


class _ColumnRegressor(SkLinearRegression):
    """A regressor whose ``predict`` answers ``(n, 1)`` instead of ``(n,)``."""

    def predict(self, X):
        return super().predict(X).reshape(-1, 1)


def test_members_returning_2d_columns_reproduce_sklearns_odd_shape():
    """sklearn stacks the members' answers AS RETURNED and transposes, so
    members answering ``(n, 1)`` yield a 3-D ``(1, n, k)`` transform and a
    ``(1, k)`` predict rather than the usual shapes.

    That is surprising and it is observable. Ravelling the predictions here
    would look like a fix and would make mlrs disagree with sklearn on exactly
    the input where sklearn surprises people — so the shapes are reproduced
    instead. The Rust arms decline anything that is not 1-D, so numpy handles
    this on every arm.
    """
    X, y = host_design()
    estimators = [("c1", _ColumnRegressor()), ("c2", _ColumnRegressor())]
    a, b = fit_both(X, y, estimators)
    assert a.transform(X).shape == b.transform(X).shape == (1, N_SAMPLES, 2)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    assert a.predict(X).shape == b.predict(X).shape == (1, 2)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_mixed_1d_and_2d_members_fail_the_same_way_on_both():
    """A ragged stack. numpy refuses it, and mlrs must refuse it with numpy's
    own message rather than repairing it into something sklearn would not
    produce."""
    X, y = host_design()
    estimators = [("col", _ColumnRegressor()), ("lr", SkLinearRegression())]
    a, b = fit_both(X, y, estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.transform(X)
    with pytest.raises(ValueError) as mlrs_exc:
        a.transform(X)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_set_output_pandas_names_the_transform_columns():
    """``TransformerMixin.set_output`` still applies — this class overrides
    ``fit_transform``, and an override that bypassed the ``_SetOutputMixin``
    wrapper would silently drop the configured container."""
    pd = pytest.importorskip("pandas")
    X, y = host_design()
    a = mlrs.VotingRegressor(sk_estimators()).set_output(transform="pandas")
    b = SkVotingRegressor(sk_estimators()).set_output(transform="pandas")
    a.fit(X, y)
    b.fit(X, y)
    assert isinstance(a.transform(X), pd.DataFrame)
    assert list(a.transform(X).columns) == list(b.transform(X).columns)
    # ...through `fit_transform` too, which is the overridden path.
    fa = mlrs.VotingRegressor(sk_estimators()).set_output(transform="pandas")
    fb = SkVotingRegressor(sk_estimators()).set_output(transform="pandas")
    assert list(fa.fit_transform(X, y).columns) == list(fb.fit_transform(X, y).columns)


def test_transform_shape_is_n_by_kept_members():
    X, y = host_design()
    a, b = fit_both(X, y)
    assert a.transform(X).shape == b.transform(X).shape == (N_SAMPLES, 3)
    # And the columns are in `estimators` order, not fit-completion order.
    for j, est in enumerate(a.estimators_):
        np.testing.assert_array_equal(a.transform(X)[:, j], est.predict(X))


def test_fit_transform_equals_fit_then_transform():
    X, y = host_design()
    a, b = both()
    np.testing.assert_array_equal(a.fit_transform(X, y), b.fit_transform(X, y))


def test_score_is_the_regressor_mixins_r2():
    X, y = host_design()
    a, b = fit_both(X, y)
    assert a.score(X, y) == b.score(X, y)


# =========================================================================== #
# fit-time metadata
# =========================================================================== #


def test_sample_weight_reaches_every_member():
    """``KNeighborsRegressor`` takes no ``sample_weight``, so this cell uses
    members that do — a mixed list is the NEXT test, where the refusal is the
    point."""
    X, y = host_design()
    rng = np.random.default_rng(SEED)
    sw = rng.uniform(0.5, 2.0, size=N_SAMPLES)
    estimators = [("lr", SkLinearRegression()), ("ridge", SkRidge(alpha=25.0))]
    a, b = both(estimators)
    a.fit(X, y, sample_weight=sw)
    b.fit(X, y, sample_weight=sw)
    assert_same(a, b, X)
    # The weights genuinely reached the members: an unweighted fit differs.
    c, _ = fit_both(X, y, estimators)
    assert not np.array_equal(a.predict(X), c.predict(X))


def test_a_member_that_rejects_sample_weight_gets_sklearns_clearer_message():
    X, y = host_design()
    estimators = [("lr", SkLinearRegression()), ("knn", SkKNeighborsRegressor(3))]
    a, b = both(estimators)
    sw = np.ones(N_SAMPLES)
    with pytest.raises(TypeError) as sk_exc:
        b.fit(X, y, sample_weight=sw)
    with pytest.raises(TypeError) as mlrs_exc:
        a.fit(X, y, sample_weight=sw)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "does not support sample weights" in str(mlrs_exc.value)


def test_an_unrouted_extra_fit_param_is_sklearns_refusal():
    X, y = host_design()
    a, b = both()
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y, nonsense=1)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y, nonsense=1)
    assert str(mlrs_exc.value) == str(sk_exc.value)
    assert "enable_metadata_routing=True" in str(mlrs_exc.value)


def test_metadata_routing_describes_one_fit_node_per_member():
    """A voting ensemble has no second stage, so — unlike stacking's router —
    there is no ``final_estimator_`` node to route ``predict`` to."""
    a, b = both()
    ra = a.get_metadata_routing()._serialize()
    rb = b.get_metadata_routing()._serialize()
    assert set(ra) == set(rb) == {"lr", "ridge", "knn"}


def test_a_2d_column_y_warns_exactly_as_sklearn_does():
    """``column_or_1d(y, warn=True)`` — an ``(n, 1)`` target is accepted with a
    ``DataConversionWarning``, and the warning is part of the surface."""
    X, y = host_design()
    a, b = both()
    with pytest.warns(UserWarning) as sk_rec:
        b.fit(X, y.reshape(-1, 1))
    with pytest.warns(UserWarning) as mlrs_rec:
        a.fit(X, y.reshape(-1, 1))
    assert [str(w.message) for w in mlrs_rec] == [str(w.message) for w in sk_rec]
    assert_same(a, b, X)


# =========================================================================== #
# dtypes, and mlrs sub-estimators (the real deployment shape)
# =========================================================================== #


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_both_float_dtypes_are_exact_over_host_members(dtype):
    """The aggregation stays in the input dtype on both arms, so an f32 problem
    must agree with sklearn EXACTLY rather than after a wider accumulation."""
    X, y = host_design(dtype)
    a, b = fit_both(X, y)
    assert_same(a, b, X)
    a, b = fit_both(X, y, weights=[2.0, 1.0, 3.0])
    assert_same(a, b, X)


def test_mlrs_members_compose_and_track_sklearn():
    """The deployment shape: members whose fits go to the device.

    Compared against a sklearn ``VotingRegressor`` over the SKLEARN twins of the
    same estimators — so this cell gates the composition end to end (member
    numerics included) at the project's live tolerance, not at bit equality.
    """
    dtype = conftest.default_float_dtype()
    X, y = host_design(dtype)
    a = mlrs.VotingRegressor(
        [("lr", mlrs.LinearRegression()), ("ridge", mlrs.Ridge(alpha=25.0))],
        weights=[2.0, 1.0],
    ).fit(X, y)
    b = SkVotingRegressor(
        [("lr", SkLinearRegression()), ("ridge", SkRidge(alpha=25.0))],
        weights=[2.0, 1.0],
    ).fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=conftest.live_atol())
    np.testing.assert_allclose(a.transform(X), b.transform(X), atol=conftest.live_atol())
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


def test_a_mixed_mlrs_and_sklearn_ensemble_works():
    """Nothing about the composition requires the members to agree on a
    backend, and a user migrating estimator by estimator lands here first."""
    dtype = conftest.default_float_dtype()
    X, y = host_design(dtype)
    a = mlrs.VotingRegressor(
        [("mlrs_lr", mlrs.LinearRegression()), ("sk_ridge", SkRidge(alpha=25.0))]
    ).fit(X, y)
    b = SkVotingRegressor(
        [("mlrs_lr", SkLinearRegression()), ("sk_ridge", SkRidge(alpha=25.0))]
    ).fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=conftest.live_atol())
