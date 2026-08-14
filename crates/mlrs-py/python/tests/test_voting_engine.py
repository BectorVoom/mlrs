"""The three aggregation arms of ``VotingRegressor`` (VOTE-01).

``mlrs.VotingRegressor`` computes ``transform`` (the ``n x k`` column stack) and
``predict`` (the weighted row mean) on one of three arms, chosen by
``MLRS_VOTING_ENGINE``:

  ==========  =====================================================
  value       arm
  ==========  =====================================================
  unset       ``np.asarray(...).T`` / ``np.average`` — the DEFAULT
  ``numpy``   the same, forced
  ``host``    ``mlrs_algos::ensemble::voting::{stack_columns, weighted_average}``
  ``device``  the CubeCL kernels in ``mlrs_backend::prims::voting``
  ==========  =====================================================

## What each arm is held to, and why they differ

``transform`` carries no arithmetic — it is a transpose — so all three arms must
agree **bit for bit**. A tolerance would hide precisely the bugs a scatter can
have: a column at the wrong index, a row stride off by one, the matrix
transposed.

``predict`` reduces, and here the arms split:

* ``numpy`` and ``host`` are still **exact**. ``weighted_average`` reproduces
  ``np.average`` operation for operation — the products, then a left-to-right row
  sum, then a DIVISION by the weight sum, all in the input dtype — so anything
  looser would let a reassociated accumulation or a reciprocal-multiply through.
* ``device`` is held to a few ULP. A GPU contracts ``acc + pred*w`` into a fused
  multiply-add, rounding ONCE where numpy rounds twice; measured on rocm
  gfx1151 the gap is at most one ULP, and the cpu backend (which does not
  contract) passes the same bound at zero. This is more accurate than the
  reference rather than less, and two orders inside mlrs's 1e-5 contract. The
  bound here is ``4 * eps`` RELATIVE — not 1e-5, which is ~80 f32 ULP and would
  let a genuine accumulation bug through.

## Two distinct things are gated

1. **The arms agree with numpy** on every shape the shim can produce (one
   member, several, weighted and uniform, f32 and f64), and the numpy FALLBACK
   still catches what the Rust arms cannot represent.
2. **The string-valued parameter still oracle-matches live sklearn on every
   arm.** ``'drop'`` is the whole string surface of this estimator (see
   ``test_oracle_voting_regressor.py``), and it changes WHICH columns are
   aggregated — so a scatter bug could plausibly show up on one route and not
   the other. Running it per arm is what keeps the new engines from being
   covered only by synthetic column data.

Req: VOTE-01 (the Rust/CubeCL aggregation engine and the parameter surface).
"""

import numpy as np
import pytest
from sklearn.ensemble import VotingRegressor as SkVotingRegressor
from sklearn.linear_model import LinearRegression as SkLinearRegression, Ridge as SkRidge
from sklearn.neighbors import KNeighborsRegressor as SkKNeighborsRegressor

import conftest

mlrs = pytest.importorskip("mlrs")

#: The arms that route through Rust. ``numpy`` is covered as the reference.
RUST_ENGINES = ["host", "device"]

KNOB = "MLRS_VOTING_ENGINE"

N_SAMPLES = 200
N_FEATURES = 5
SEED = 42

#: Relative ULP budget for the ``device`` arm's reduction. See the module
#: docstring: one ULP is the measured FMA-contraction gap, and four leaves room
#: for a second contraction without admitting a real reassociation bug.
ULP_BUDGET = 4.0


def device_f64_available():
    """Can the ``device`` arm reduce f64 columns on this backend?

    ``backend_f64_device_kernels()`` — the widest of the three f64 flags — NOT
    ``backend_supports_f64()``. rocm and cuda answer False to the advertised flag
    (their matmul rejects f64) while running plain f64 kernels fine, and gating
    on the narrow flag here would skip cells that do work (STACK-META-01).
    """
    return bool(mlrs._load_ext().backend_f64_device_kernels())


def skip_if_no_device_f64(engine, dtype):
    """Skip a ``device`` cell whose columns are f64 on a backend without f64
    device kernels — the arm raises there BY DESIGN, and a silent host fallback
    would make the whole ``device`` column a copy of the ``host`` one."""
    if engine == "device" and dtype == np.float64 and not device_f64_available():
        pytest.skip("device arm cannot launch an f64 kernel on this backend")


def assert_agrees(got, expected, engine, what):
    """Exact for every arm but ``device``, which gets [`ULP_BUDGET`] relative."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    if engine != "device":
        np.testing.assert_array_equal(got, expected, err_msg=f"{engine} {what}")
        return
    eps = np.finfo(expected.dtype).eps
    np.testing.assert_allclose(
        got,
        expected,
        rtol=ULP_BUDGET * eps,
        atol=ULP_BUDGET * eps,
        err_msg=f"{engine} {what}",
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


def host_design(dtype=np.float64, n_samples=N_SAMPLES):
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, N_FEATURES)).astype(dtype)
    beta = np.array([3.0, -1.5, 0.0, 0.75, 2.0], dtype=dtype)
    y = (X @ beta).astype(dtype)
    return X, (y + (0.05 * rng.standard_normal(n_samples)).astype(dtype))


def sk_estimators():
    return [
        ("lr", SkLinearRegression()),
        ("ridge", SkRidge(alpha=25.0)),
        ("knn", SkKNeighborsRegressor(n_neighbors=3)),
    ]


def both(estimators=None, **kwargs):
    if estimators is None:
        estimators = sk_estimators()
    return (
        mlrs.VotingRegressor(estimators, **kwargs),
        SkVotingRegressor(estimators, **kwargs),
    )


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
    """A typo in a sweep script must not surface as an exception from ``predict``."""
    monkeypatch.setenv(KNOB, "gpu")
    assert mlrs._load_ext().voting_engine() == "numpy"
    X, y = host_design()
    a, b = both()
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_forcing_numpy_explicitly_is_the_same_as_leaving_it_unset(monkeypatch):
    monkeypatch.setenv(KNOB, "numpy")
    assert mlrs._load_ext().voting_engine() == "numpy"


# =========================================================================== #
# 1. The arms agree with numpy
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "weights",
    [None, [2.0, 1.0, 3.0], [3.0, -1.0, 1.0]],
    ids=["uniform", "asymmetric", "negative"],
)
def test_predict_agrees_with_sklearn_on_every_arm(engine, dtype, weights):
    skip_if_no_device_f64(engine, dtype)
    X, y = host_design(dtype)
    a, b = both(weights=weights)
    a.fit(X, y)
    b.fit(X, y)
    assert_agrees(a.predict(X), b.predict(X), engine, "predict")


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_transform_is_bit_identical_on_every_arm(engine, dtype):
    """No arithmetic, so this is an EQUALITY on the device arm too — the FMA
    exemption applies to the reduction and to nothing else."""
    skip_if_no_device_f64(engine, dtype)
    X, y = host_design(dtype)
    a, b = both()
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("n_members", [1, 2, 3])
def test_every_member_count_agrees(engine, n_members):
    """A one-member ensemble is the degenerate shape (the kernel's first launch
    is also its last); three is the common one."""
    X, y = host_design()
    estimators = sk_estimators()[:n_members]
    a, b = both(estimators)
    a.fit(X, y)
    b.fit(X, y)
    assert_agrees(a.predict(X), b.predict(X), engine, "predict")
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_a_large_row_count_agrees(engine):
    """Past one cube's worth of units, where the launch over-provisions and the
    kernel's bounds check is what keeps it in range."""
    X, y = host_design(n_samples=5000)
    a, b = both(weights=[2.0, 1.0, 3.0])
    a.fit(X, y)
    b.fit(X, y)
    assert_agrees(a.predict(X), b.predict(X), engine, "predict")


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_the_numpy_fallback_still_catches_what_rust_cannot_represent(engine):
    """A member returning an INTEGER column: ``np.average`` promotes it, the
    Rust arms decline it, and ``_vote_via_rust`` returns ``None`` so numpy
    handles it exactly as before the arm existed."""
    from mlrs.ensemble import _vote_via_rust

    cols = [np.arange(8, dtype=np.int64), np.arange(8, dtype=np.int64) * 2]
    assert _vote_via_rust(cols, "predict", None, engine) is None
    # …and an object column, and a ragged one.
    assert _vote_via_rust([np.array(["a", "b"], dtype=object)], "predict", None, engine) is None
    assert (
        _vote_via_rust(
            [np.zeros(4, dtype=np.float64), np.zeros(3, dtype=np.float64)],
            "predict",
            None,
            engine,
        )
        is None
    )
    # A zero-row aggregation is left to numpy too — there is nothing to hand a
    # kernel, and numpy already produces the right empty shape.
    assert _vote_via_rust([np.zeros(0, dtype=np.float64)], "predict", None, engine) is None


def test_the_device_arm_refuses_f64_on_a_backend_without_f64_kernels(monkeypatch):
    """The refusal is explicit rather than a silent host fallback: a fallback
    would make an A/B sweep compare the host arm against itself."""
    if device_f64_available():
        pytest.skip("this backend has f64 device kernels; nothing to refuse")
    from mlrs.ensemble import _vote_via_rust

    cols = [np.ones(8, dtype=np.float64), np.zeros(8, dtype=np.float64)]
    with pytest.raises(ValueError, match="no f64 device kernels"):
        _vote_via_rust(cols, "predict", None, "device")


# =========================================================================== #
# 2. The string-valued parameter still oracle-matches sklearn on every arm
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("position", [0, 1, 2])
def test_the_drop_sentinel_oracle_matches_on_every_arm(engine, position):
    """``'drop'`` changes WHICH columns are aggregated, so it is the one string
    whose interaction with a scatter bug is plausible."""
    X, y = host_design()
    estimators = sk_estimators()
    estimators[position] = (estimators[position][0], "drop")
    a, b = both(estimators, weights=[3.0, 100.0, 1.0])
    a.fit(X, y)
    b.fit(X, y)
    assert_agrees(a.predict(X), b.predict(X), engine, "predict")
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_the_all_dropped_rejection_is_unchanged_by_the_arm(engine):
    """A structural rejection happens before any aggregation, so the arm must
    not change the message — or the exception type."""
    X, y = host_design()
    estimators = [(name, "drop") for name, _ in sk_estimators()]
    a, b = both(estimators)
    with pytest.raises(ValueError) as sk_exc:
        b.fit(X, y)
    with pytest.raises(ValueError) as mlrs_exc:
        a.fit(X, y)
    assert str(mlrs_exc.value) == str(sk_exc.value)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_a_zero_weight_sum_is_a_zero_division_error_on_every_arm(engine):
    """numpy's exception, not a Rust ``ValueError`` and not an infinity — on the
    arms that never touch numpy either."""
    X, y = host_design()
    a, b = both(weights=[1.0, -1.0, 0.0])
    a.fit(X, y)
    b.fit(X, y)
    with pytest.raises(ZeroDivisionError) as sk_exc:
        b.predict(X)
    with pytest.raises(ZeroDivisionError) as mlrs_exc:
        a.predict(X)
    assert str(mlrs_exc.value) == str(sk_exc.value)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_mlrs_members_compose_on_every_arm(engine):
    """The deployment shape — members whose fits go to the device — reaching the
    aggregation arm as well."""
    dtype = conftest.default_float_dtype()
    skip_if_no_device_f64(engine, dtype)
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
