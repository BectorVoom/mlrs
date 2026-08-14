"""The three aggregation arms of ``VotingClassifier`` (VOTE-CLF-01).

``mlrs.VotingClassifier`` runs FOUR distinct aggregations — one per
(``voting``, method) pair — on one of three arms, chosen by
``MLRS_VOTING_ENGINE``:

  ==========  =====================================================
  value       arm
  ==========  =====================================================
  unset       numpy — the DEFAULT
  ``numpy``   the same, forced
  ``host``    ``mlrs_algos::ensemble::voting``'s host loops
  ``device``  the CubeCL kernels in ``mlrs_backend::prims::voting``
  ==========  =====================================================

## What each aggregation is held to, and why they differ

  ==========================  ==========  ==============================
  aggregation                 arms        bound
  ==========================  ==========  ==============================
  hard ``predict``            all three   **EXACT**
  soft ``predict_proba``      numpy/host  **EXACT**
  soft ``predict_proba``      device      ``ULP_BUDGET`` relative
  soft ``predict``            all three   exact LABELS
  soft ``transform``          all three   **EXACT** (a pure copy)
  hard ``transform``          all three   **EXACT** (numpy on every arm)
  ==========================  ==========  ==============================

**Hard voting is exact on the device too, and that is a real claim rather than
an accident.** The regressor's average is `acc + pred·w`, which a GPU contracts
into a fused multiply-add and so rounds once where numpy rounds twice. The hard
tally has no such shape: it adds a scalar weight into a bin, one rounding
everywhere. If this ever drifts, the cause is the tally or the tie-break, not
the hardware — so a tolerance here would hide the only bugs the kernel can have.

**Soft voting inherits the regressor's caveat**, because it IS the regressor's
reduction with ``n * n_classes`` elements per member. ``predict_proba`` is
therefore held to a few ULP on the ``device`` arm; ``predict`` is held to exact
LABELS anyway, because the fixture's classes are separated by far more than a
ULP (a near-tie would be gating on the contraction gap rather than on the
aggregation).

**Hard ``transform`` runs in numpy on every arm, by design.** It returns the
members' integer labels, and the Rust aggregation arms are float-typed — they
exist to reproduce ``np.average`` bit for bit. ``_vote_via_rust`` declines an
integer column, so numpy answers. Asserting that here is what keeps the gap
documented rather than latent.

## Two distinct things are gated

1. **The arms agree with sklearn** on every shape the shim can produce (one
   member, several, weighted and uniform, binary and multiclass, f32 and f64),
   and the numpy FALLBACK still catches what the Rust arms cannot represent.
2. **Both string-valued parameters still oracle-match live sklearn on every
   arm.** ``voting`` selects WHICH arm entry point runs at all, and ``'drop'``
   changes WHICH columns are aggregated — so a bug in one route could plausibly
   show up on one arm and not the other. Running them per arm is what keeps the
   new engines from being covered only by synthetic column data.

Req: VOTE-CLF-01 (the Rust/CubeCL classifier engine and the parameter surface).
"""

import numpy as np
import pytest
from sklearn.ensemble import VotingClassifier as SkVotingClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.neighbors import KNeighborsClassifier as SkKNeighborsClassifier

import conftest

mlrs = pytest.importorskip("mlrs")

#: The arms that route through Rust. ``numpy`` is covered as the reference.
RUST_ENGINES = ["host", "device"]

KNOB = "MLRS_VOTING_ENGINE"

VOTING_VALUES = ["hard", "soft"]

N_SAMPLES = 200
N_FEATURES = 5
SEED = 42

#: Relative ULP budget for the ``device`` arm's soft reduction. Same value and
#: same reasoning as ``test_voting_engine.py``: one ULP is the measured
#: FMA-contraction gap, four leaves room for a second contraction without
#: admitting a real reassociation bug.
ULP_BUDGET = 4.0


def device_f64_available():
    """Can the ``device`` arm reduce f64 blocks on this backend?

    ``backend_f64_device_kernels()`` — the widest of the three f64 flags — NOT
    ``backend_supports_f64()``. rocm and cuda answer False to the advertised flag
    (their matmul rejects f64) while running plain f64 kernels fine, and gating
    on the narrow flag here would skip cells that do work (STACK-META-01).
    """
    return bool(mlrs._load_ext().backend_f64_device_kernels())


def skip_if_no_device_f64(engine, dtype):
    """Skip a ``device`` cell whose blocks are f64 on a backend without f64
    device kernels — the arm raises there BY DESIGN, and a silent host fallback
    would make the whole ``device`` column a copy of the ``host`` one."""
    if engine == "device" and dtype == np.float64 and not device_f64_available():
        pytest.skip("device arm cannot launch an f64 kernel on this backend")


def assert_proba_agrees(got, expected, engine):
    """Exact for every arm but ``device``, which gets [`ULP_BUDGET`] relative."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    if engine != "device":
        np.testing.assert_array_equal(got, expected, err_msg=f"{engine} predict_proba")
        return
    eps = np.finfo(expected.dtype).eps
    np.testing.assert_allclose(
        got,
        expected,
        rtol=ULP_BUDGET * eps,
        atol=ULP_BUDGET * eps,
        err_msg=f"{engine} predict_proba",
    )


@pytest.fixture
def engine(request, monkeypatch):
    """Force one arm for the duration of a test, and prove the knob took.

    The assertion is not ceremony: a knob that silently failed to reach Rust
    would make every comparison below a comparison of numpy against numpy, and
    the suite would pass while testing nothing
    (``mlrs-bench-verify-knob-is-live``).
    """
    value = request.param
    monkeypatch.setenv(KNOB, value)
    resolved = mlrs._load_ext().voting_engine()
    assert resolved == value, f"knob {KNOB}={value} resolved to {resolved!r}"
    return value


def host_design(n_classes=3, dtype=np.float64, n_samples=N_SAMPLES):
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, N_FEATURES)).astype(dtype)
    score = X[:, 0] * 3.0 - X[:, 1] + 0.5 * X[:, 2]
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    return X, np.searchsorted(edges, score).astype(np.int64)


def sk_estimators():
    return [
        ("lr", SkLogisticRegression(max_iter=500)),
        ("nb", SkGaussianNB()),
        ("knn", SkKNeighborsClassifier(n_neighbors=5)),
    ]


def both(estimators=None, **kwargs):
    if estimators is None:
        estimators = sk_estimators()
    return (
        mlrs.VotingClassifier(estimators, **kwargs),
        SkVotingClassifier(estimators, **kwargs),
    )


def fit_both(X, y, estimators=None, **kwargs):
    a, b = both(estimators, **kwargs)
    return a.fit(X, y), b.fit(X, y)


# =========================================================================== #
# The knob itself
# =========================================================================== #


def test_the_default_arm_is_numpy(monkeypatch):
    """Unset means numpy — the Rust arms are opt-in.

    This is a product decision backed by ``docs/voting.md``'s ladder, so it is
    asserted rather than left implicit.
    """
    monkeypatch.delenv(KNOB, raising=False)
    assert mlrs._load_ext().voting_engine() == "numpy"


def test_an_unknown_arm_falls_back_to_numpy(monkeypatch):
    """A typo in a sweep script must not surface as an exception from
    ``predict`` — and both routes have to survive it, not just one."""
    monkeypatch.setenv(KNOB, "gpu")
    assert mlrs._load_ext().voting_engine() == "numpy"
    X, y = host_design()
    for voting in VOTING_VALUES:
        a, b = fit_both(X, y, voting=voting)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))


# =========================================================================== #
# 1. The arms agree with sklearn
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "weights",
    [None, [2.0, 1.0, 3.0], [3.0, -1.0, 1.0]],
    ids=["uniform", "asymmetric", "negative"],
)
def test_hard_predict_agrees_with_sklearn_exactly_on_every_arm(engine, dtype, weights):
    """Hard voting's tally has no multiply-accumulate to contract, so this is
    EQUALITY on the device too — including under a negative weight, where the
    per-row ``np.bincount`` length bound is what decides the answer."""
    # The label columns are integers regardless of `X`'s dtype, so the f64 device
    # gate does not apply to the hard route at all — the tally's own width is
    # chosen inside Rust.
    X, y = host_design(3, dtype)
    a, b = fit_both(X, y, voting="hard", weights=weights)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "weights", [None, [2.0, 1.0, 3.0]], ids=["uniform", "asymmetric"]
)
def test_soft_predict_proba_agrees_with_sklearn_on_every_arm(engine, dtype, weights):
    skip_if_no_device_f64(engine, dtype)
    X, y = host_design(3, dtype)
    a, b = fit_both(X, y, voting="soft", weights=weights)
    assert_proba_agrees(a.predict_proba(X), b.predict_proba(X), engine)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_soft_predict_labels_agree_exactly_on_every_arm(engine, dtype):
    """The FUSED path: on the ``device`` arm ``predict`` never downloads the
    ``(n, C)`` average at all, so its labels are the only observable of that
    kernel chain."""
    skip_if_no_device_f64(engine, dtype)
    X, y = host_design(3, dtype)
    a, b = fit_both(X, y, voting="soft", weights=[2.0, 1.0, 3.0])
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_fused_soft_predict_is_the_argmax_of_the_same_arms_proba(engine, dtype):
    """``predict`` and ``predict_proba`` must not disagree ON THE SAME ARM.

    This is the assertion that catches a fused device path reading the
    accumulator before the divide, or with the wrong row stride — a bug that a
    comparison against sklearn could mask if it happened to preserve the order.
    """
    skip_if_no_device_f64(engine, dtype)
    X, y = host_design(3, dtype)
    a, _ = fit_both(X, y, voting="soft", weights=[2.0, 1.0, 3.0])
    proba = a.predict_proba(X)
    np.testing.assert_array_equal(a.predict(X), a.classes_[np.argmax(proba, axis=1)])


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_classes", [2, 3, 5], ids=["binary", "3-class", "5-class"])
def test_the_flattened_soft_transform_is_exact_on_every_arm(engine, dtype, n_classes):
    """A pure copy — so EQUALITY, not a tolerance. A block written at the wrong
    column offset or a stride off by one is precisely what a tolerance would
    hide, and the class count is parametrized because the offset is a multiple
    of it."""
    skip_if_no_device_f64(engine, dtype)
    X, y = host_design(n_classes, dtype)
    a, b = fit_both(X, y, voting="soft")
    got, expected = a.transform(X), b.transform(X)
    assert got.shape == expected.shape == (N_SAMPLES, 3 * n_classes)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_the_unflattened_soft_transform_bypasses_the_arms_entirely(engine):
    """``flatten_transform=False`` returns the raw 3-D stack, which is not an
    aggregation at all — no arm should touch it, and every arm must return the
    same object shape sklearn does."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="soft", flatten_transform=False)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    assert a.transform(X).shape == (3, N_SAMPLES, 3)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_hard_transform_is_numpys_answer_on_every_arm(engine):
    """The documented gap in arm coverage.

    Hard ``transform`` returns integer labels, which the float aggregation arms
    decline — so numpy answers on every arm, and the RESULT (including its
    integer dtype) is identical to sklearn's. Asserting it keeps the gap
    documented rather than latent: if a future change routed integers through a
    float arm, the dtype would change here.
    """
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="hard")
    got, expected = a.transform(X), b.transform(X)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype
    assert got.dtype.kind in "iu"


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("k", [1, 2, 5, 9], ids=lambda k: f"k={k}")
def test_the_member_count_does_not_change_the_answer_on_any_arm(engine, k):
    """A one-member ensemble and a nine-member one exercise different launch
    shapes (the accumulate chain is one launch per member) but must both match
    sklearn."""
    X, y = host_design(3)
    estimators = [
        (f"nb{j}", SkGaussianNB(var_smoothing=10.0 ** (-9 + j))) for j in range(k)
    ]
    weights = [float(j + 1) for j in range(k)]
    for voting in VOTING_VALUES:
        a, b = fit_both(X, y, estimators, voting=voting, weights=weights)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))


# =========================================================================== #
# 2. Both string-valued parameters, on every arm
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("voting", VOTING_VALUES)
@pytest.mark.parametrize("n_classes", [2, 3, 5], ids=["binary", "3-class", "5-class"])
def test_voting_oracle_matches_sklearn_on_every_arm(engine, voting, n_classes):
    """``voting`` selects which arm entry point runs at all, so the oracle for
    it has to be re-run per arm rather than only on the default one."""
    X, y = host_design(n_classes)
    a, b = fit_both(X, y, voting=voting)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
    if voting == "soft":
        assert_proba_agrees(a.predict_proba(X), b.predict_proba(X), engine)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("voting", VOTING_VALUES)
@pytest.mark.parametrize("position", [0, 1, 2])
def test_the_drop_sentinel_still_oracle_matches_on_every_arm(engine, voting, position):
    """``'drop'`` changes WHICH columns reach the aggregation, so a kept-index
    bug could plausibly show up on one arm and not the other."""
    X, y = host_design(3)
    estimators = sk_estimators()
    estimators[position] = (estimators[position][0], "drop")
    a, b = fit_both(X, y, estimators, voting=voting, weights=[3.0, 100.0, 1.0])
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_a_single_surviving_member_is_that_member_on_every_arm(engine):
    """``k = 1`` is the degenerate launch shape: one accumulate, no adds."""
    X, y = host_design(3)
    estimators = sk_estimators()
    estimators[1] = ("nb", "drop")
    estimators[2] = ("knn", "drop")
    for voting in VOTING_VALUES:
        a, b = fit_both(X, y, estimators, voting=voting)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))
        np.testing.assert_array_equal(a.predict(X), a.estimators_[0].predict(X))


# =========================================================================== #
# 3. The numpy fallback still catches what the Rust arms decline
# =========================================================================== #


class _EmptyTolerantNB(SkGaussianNB):
    """A member that answers an empty query instead of rejecting it.

    Every stock sklearn predictor refuses a zero-row ``X`` in ``check_array``
    (``ensure_min_samples=1``), so the empty-aggregation path is unreachable
    through an ordinary ensemble — which is exactly why it needs a member that
    lets it through rather than a test that would silently assert nothing.
    """

    def predict(self, X):
        X = np.asarray(X)
        if X.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        return super().predict(X)

    def predict_proba(self, X):
        X = np.asarray(X)
        if X.shape[0] == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        return super().predict_proba(X)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_an_empty_query_falls_back_to_numpy_rather_than_a_zero_byte_allocation(engine):
    """Zero rows is a zero-byte device allocation, which CubeCL has nothing to
    do with. The Rust arms decline it and numpy answers — so mlrs reproduces
    sklearn EXACTLY, including where sklearn itself fails.

    And it does fail, on BOTH routes, in two different ways:
    ``np.apply_along_axis`` refuses a zero-length iteration axis, and
    ``np.average`` divides by a zero count. Those are sklearn's behaviours, so
    they are mlrs's — and asserting the identical failures is what proves the
    fallback is running numpy's code rather than a Rust arm that would have
    returned an empty array and diverged.

    ``transform``, which does not reduce, succeeds on both routes and is
    compared as a value.
    """
    X, y = host_design(3)
    empty = X[:0]
    estimators = [("a", _EmptyTolerantNB()), ("b", _EmptyTolerantNB(var_smoothing=1e-5))]

    for voting in VOTING_VALUES:
        a, b = fit_both(X, y, estimators, voting=voting)
        with pytest.raises(Exception) as sk_exc:
            b.predict(empty)
        with pytest.raises(Exception) as mlrs_exc:
            a.predict(empty)
        assert type(mlrs_exc.value) is type(sk_exc.value), voting
        assert str(mlrs_exc.value) == str(sk_exc.value), voting
        got, expected = a.transform(empty), b.transform(empty)
        assert got.shape == expected.shape
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_a_member_returning_float_labels_falls_back_to_numpy(engine):
    """``np.bincount`` takes integers, so a member answering with floats is
    numpy's error to report — not a message this shim invented.

    ``_vote_labels_via_rust`` declines a non-integer column for exactly that
    reason, and both libraries then raise the same thing.
    """

    class FloatLabels(SkGaussianNB):
        def predict(self, X):
            return super().predict(X).astype(np.float64)

    X, y = host_design(3)
    estimators = [("f", FloatLabels()), ("nb", SkGaussianNB())]
    a, b = fit_both(X, y, estimators, voting="hard")
    with pytest.raises(Exception) as sk_exc:
        b.predict(X)
    with pytest.raises(Exception) as mlrs_exc:
        a.predict(X)
    assert type(mlrs_exc.value) is type(sk_exc.value)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_a_zero_weight_sum_still_raises_numpys_error_on_every_arm(engine):
    """``np.average``'s ``ZeroDivisionError`` reaches the caller from the Rust
    arms too — it is validated host-side before any launch, because a kernel
    cannot return an error and would produce infinities instead."""
    X, y = host_design(3)
    a, b = fit_both(X, y, voting="soft", weights=[1.0, -1.0, 0.0])
    with pytest.raises(ZeroDivisionError) as sk_exc:
        b.predict_proba(X)
    with pytest.raises(ZeroDivisionError) as mlrs_exc:
        a.predict_proba(X)
    assert str(mlrs_exc.value) == str(sk_exc.value)


def test_the_device_arm_refuses_f64_blocks_where_it_has_no_f64_kernels(monkeypatch):
    """On a backend without f64 device kernels the arm RAISES rather than
    silently falling back to the host loop.

    A silent fallback would make the whole ``device`` column of this suite a
    copy of the ``host`` one — the vacuous-sweep failure mode
    (``mlrs-bench-verify-knob-is-live``) — so the refusal is the contract, and
    the message names the way out.
    """
    if device_f64_available():
        pytest.skip("this backend has f64 device kernels; nothing to refuse")
    monkeypatch.setenv(KNOB, "device")
    X, y = host_design(3, np.float64)
    a, _ = fit_both(X, y, voting="soft")
    with pytest.raises(ValueError, match="no f64 device kernels"):
        a.predict_proba(X)


# =========================================================================== #
# 4. mlrs sub-estimators on the Rust arms
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("voting", VOTING_VALUES)
def test_mlrs_members_compose_on_every_arm(engine, voting):
    """The real deployment shape: device-backed members AND a device-backed
    aggregation in one call.

    The comparison is against the same mlrs estimator on the ``numpy`` arm
    rather than against sklearn, which isolates the ARM from the members' own
    arithmetic — the sklearn comparison lives in
    ``test_oracle_voting_classifier.py``.
    """
    dtype = conftest.default_float_dtype()
    skip_if_no_device_f64(engine, dtype)
    X, y = host_design(3, dtype)
    estimators = [
        ("nb", mlrs.GaussianNB()),
        ("knn", mlrs.KNeighborsClassifier(n_neighbors=5)),
    ]
    fitted = mlrs.VotingClassifier(estimators, voting=voting, weights=[2.0, 1.0])
    fitted.fit(X, y)
    got = fitted.predict(X)

    import os

    saved = os.environ[KNOB]
    try:
        os.environ[KNOB] = "numpy"
        assert mlrs._load_ext().voting_engine() == "numpy"
        reference = fitted.predict(X)
    finally:
        os.environ[KNOB] = saved
    np.testing.assert_array_equal(got, reference)
