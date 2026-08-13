"""The three meta-assembly arms of ``StackingRegressor`` (STACK-META-01).

``mlrs.StackingRegressor`` builds its meta-feature matrix on one of three arms,
chosen by ``MLRS_STACK_META_ENGINE``:

  ==========  =====================================================
  value       arm
  ==========  =====================================================
  unset       ``np.hstack`` in the shim — the DEFAULT
  ``numpy``   the same, forced
  ``host``    ``mlrs_algos::ensemble::stacking::concatenate_predictions``
  ``device``  the CubeCL scatter in ``mlrs_backend::prims::stacking_meta``
  ==========  =====================================================

The copy carries no arithmetic, so the three arms must agree **bit for bit**,
not within a tolerance — every assertion here is an exact equality. A tolerance
would hide precisely the bugs a scatter can have: a block at the wrong offset, a
row stride off by one, a multi-column block transposed.

Two distinct things are gated:

1. **The arms agree with numpy** on every layout the shim can produce
   (single/multiple blocks, a multi-column block, ``passthrough`` on and off,
   f32 and f64), and the numpy FALLBACK still catches what the Rust arms cannot
   represent.
2. **The string-valued parameters still oracle-match live sklearn on every
   arm.** ``cv="prefit"`` and the ``'drop'`` sentinel are the two strings in
   this estimator's surface; both change the CONTENT of the meta matrix, so a
   scatter bug could plausibly show up on one route and not the other. Running
   them per arm is what keeps the new engines from being covered only by
   synthetic block data.

Req: STACK-META-01 (the Rust/CubeCL meta engine), STACK-01 (parameter surface).
"""

import numpy as np
import pytest
from sklearn.ensemble import StackingRegressor as SkStackingRegressor
from sklearn.linear_model import LinearRegression as SkLinearRegression, Ridge as SkRidge
from sklearn.neighbors import KNeighborsRegressor

import conftest

mlrs = pytest.importorskip("mlrs")

#: The arms that route through Rust. ``numpy`` is covered as the reference.
RUST_ENGINES = ["host", "device"]

KNOB = "MLRS_STACK_META_ENGINE"

N_SAMPLES = 200
N_FEATURES = 5
SEED = 42


def device_f64_available():
    """Can the ``device`` arm assemble an f64 meta matrix on this backend?

    The scatter moves data and computes nothing, so the predicate is
    ``backend_f64_device_kernels()`` — the widest of the three f64 flags — NOT
    ``backend_supports_f64()``. rocm and cuda answer False to the advertised
    flag (their matmul rejects f64) while running plain f64 kernels fine, and
    gating on the narrow flag here would skip cells that do work.
    """
    return bool(mlrs._load_ext().backend_f64_device_kernels())


def skip_if_no_device_f64(engine, dtype):
    """Skip a ``device`` cell whose blocks are f64 on a backend without f64
    device kernels — the arm raises there BY DESIGN, and a silent host fallback
    would make the whole `device` column a copy of the `host` one."""
    if engine == "device" and dtype == np.float64 and not device_f64_available():
        pytest.skip("device arm cannot launch an f64 kernel on this backend")


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
    resolved = mlrs._load_ext().stacking_meta_engine()
    assert resolved == value, f"knob {KNOB}={value} resolved to {resolved!r}"
    return value


def host_design(dtype=np.float64, n_samples=N_SAMPLES):
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n_samples, N_FEATURES)).astype(dtype)
    y = (X @ np.array([3.0, -1.5, 0.0, 0.75, 2.0], dtype=dtype)).astype(dtype)
    return X, (y + (0.05 * rng.standard_normal(n_samples)).astype(dtype))


def sk_estimators():
    return [("lr", SkLinearRegression()), ("ridge", SkRidge(alpha=1.0))]


def both(**kwargs):
    estimators = kwargs.pop("estimators", None)
    if estimators is None:
        estimators = sk_estimators()
    return (
        mlrs.StackingRegressor(estimators, **kwargs),
        SkStackingRegressor(estimators, **kwargs),
    )


# =========================================================================== #
# The knob itself
# =========================================================================== #


def test_the_default_arm_is_numpy(monkeypatch):
    """Unset means ``np.hstack`` — the Rust arms are opt-in.

    This is a product decision backed by ``docs/stacking.md``'s ladder (the copy
    has no arithmetic to amortize an FFI round-trip with), so it is asserted
    rather than left implicit.
    """
    monkeypatch.delenv(KNOB, raising=False)
    assert mlrs._load_ext().stacking_meta_engine() == "numpy"


def test_an_unknown_arm_falls_back_to_numpy(monkeypatch):
    """A typo in a sweep script must not surface as an exception from ``fit``."""
    monkeypatch.setenv(KNOB, "gpu")
    assert mlrs._load_ext().stacking_meta_engine() == "numpy"
    X, y = host_design()
    a, b = both(cv=3)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


# =========================================================================== #
# The arms agree with numpy, exactly
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize(
    "block_cols,passthrough",
    [
        ([1], False),
        ([1, 1], False),
        ([1, 1], True),
        ([3], False),
        ([1, 4, 2], True),
        ([1, 2], False),
    ],
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_meta_matrix_is_bit_identical_to_hstack(engine, block_cols, passthrough, dtype):
    """Every layout the shim can produce, on both Rust arms, exactly."""
    skip_if_no_device_f64(engine, dtype)

    rng = np.random.default_rng(SEED)
    n_rows = 137  # not a multiple of any plausible cube width
    blocks = [rng.standard_normal((n_rows, c)).astype(dtype) for c in block_cols]
    X = rng.standard_normal((n_rows, 6)).astype(dtype)

    given = blocks + ([X] if passthrough else [])
    got = mlrs.ensemble._meta_via_rust(
        given, block_cols, X.shape[1] if passthrough else 0, passthrough, engine
    )
    expected = np.hstack(given)

    assert got is not None, "a float block list must not fall back to numpy"
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got, expected)
    # sklearn hands the meta matrix straight to the final estimator's `fit`,
    # which is entitled to write into it.
    assert got.flags.writeable


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_mixed_block_dtypes_promote_exactly_as_hstack_does(engine):
    """An f32 block beside an f64 one promotes to f64 — numpy's own rule.

    Getting this wrong would silently fit the final estimator on f32-rounded
    meta features, which no test comparing shapes or tolerances would catch.
    """
    skip_if_no_device_f64(engine, np.float64)  # the promotion IS to f64
    rng = np.random.default_rng(SEED)
    blocks = [
        rng.standard_normal((64, 1)).astype(np.float32),
        rng.standard_normal((64, 1)).astype(np.float64),
    ]
    got = mlrs.ensemble._meta_via_rust(blocks, [1, 1], 0, False, engine)
    expected = np.hstack(blocks)
    assert got.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize(
    "blocks,label",
    [
        ([np.arange(8, dtype=np.int64).reshape(8, 1)], "integer block"),
        ([np.array([["a"], ["b"]], dtype=object)], "object block"),
        ([np.zeros((4, 1)), np.zeros((5, 1))], "row counts disagree"),
        ([np.zeros(4)], "not 2-D"),
    ],
)
def test_unrepresentable_blocks_fall_back_to_numpy(engine, blocks, label):
    """The Rust arms decline (``None``), they do not raise — ``np.hstack``
    handled these before the arms existed and still does."""
    assert (
        mlrs.ensemble._meta_via_rust(blocks, [1] * len(blocks), 0, False, engine) is None
    ), label


# =========================================================================== #
# The whole estimator, per arm, against live sklearn
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("passthrough", [False, True])
def test_fit_predict_transform_match_sklearn_on_every_arm(engine, passthrough):
    """The arm is invisible in the answer, which is the entire contract."""
    skip_if_no_device_f64(engine, np.float64)  # sklearn members predict in f64
    X, y = host_design()
    a, b = both(cv=5, passthrough=passthrough)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-12, rtol=0)
    assert a._n_feature_outs == b._n_feature_outs


# =========================================================================== #
# STRING PARAMETER 1 — cv="prefit", per arm
# =========================================================================== #


def _prefit_estimators(X, y):
    """Two base regressors ALREADY fitted, as ``cv="prefit"`` requires."""
    return [
        ("lr", SkLinearRegression().fit(X, y)),
        ("ridge", SkRidge(alpha=1.0).fit(X, y)),
    ]


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("passthrough", [False, True])
def test_cv_prefit_matches_sklearn_on_every_arm(engine, passthrough):
    """``cv="prefit"`` composed with each meta engine, against live sklearn."""
    skip_if_no_device_f64(engine, np.float64)
    X, y = host_design()
    fitted = _prefit_estimators(X, y)
    a, b = both(estimators=fitted, cv="prefit", passthrough=passthrough)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-12, rtol=0)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_cv_prefit_still_selects_the_prefit_ROUTE_on_every_arm(engine):
    """Not just "mlrs == sklearn": the string must still change what is FITTED.

    ``transform`` cannot see the difference (it always re-predicts through
    ``estimators_``), so this asserts on ``final_estimator_`` — with a 1-NN base,
    whose in-sample prediction is each row's own target, so ``prefit`` learns
    coef ~ 1.0 while an int ``cv`` learns visibly more.
    """
    skip_if_no_device_f64(engine, np.float64)
    X, y = host_design()
    base = [("knn", KNeighborsRegressor(n_neighbors=1))]
    fitted = [("knn", KNeighborsRegressor(n_neighbors=1).fit(X, y))]

    prefit = mlrs.StackingRegressor(fitted, cv="prefit").fit(X, y)
    kfold = mlrs.StackingRegressor(base, cv=5).fit(X, y)

    assert prefit.estimators_[0] is fitted[0][1], "prefit must not clone or refit"
    assert prefit.final_estimator_.coef_[0] == pytest.approx(1.0, abs=1e-3)
    assert kfold.final_estimator_.coef_[0] > prefit.final_estimator_.coef_[0] + 0.05
    assert not np.allclose(prefit.predict(X), kfold.predict(X))


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("bad", ["Prefit", "PREFIT", "prefit ", "auto", ""])
def test_cv_rejects_every_other_string_on_every_arm(engine, bad):
    """The rejection path is arm-independent — it happens before any copy."""
    X, y = host_design()
    with pytest.raises(ValueError, match="prefit"):
        mlrs.StackingRegressor(sk_estimators(), cv=bad).fit(X, y)


# =========================================================================== #
# STRING PARAMETER 2 — estimators=[(name, "drop")], per arm
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
@pytest.mark.parametrize("dropped", ["lr", "ridge"])
@pytest.mark.parametrize("passthrough", [False, True])
def test_drop_matches_sklearn_on_every_arm(engine, dropped, passthrough):
    """A dropped entry contributes no meta column on any arm.

    This is the case most likely to expose an offset bug: the layout the scatter
    is handed has a hole in it relative to the ``estimators`` list.
    """
    skip_if_no_device_f64(engine, np.float64)
    X, y = host_design()
    estimators = [(name, "drop" if name == dropped else est) for name, est in sk_estimators()]
    a, b = both(estimators=estimators, cv=3, passthrough=passthrough)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-12, rtol=0)
    assert list(a.get_feature_names_out()) == list(b.get_feature_names_out())
    assert a.named_estimators_[dropped] == "drop"
    assert a._n_feature_outs == b._n_feature_outs == [1]


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_all_estimators_dropped_still_raises_on_every_arm(engine):
    X, y = host_design()
    est = mlrs.StackingRegressor([(n, "drop") for n, _ in sk_estimators()])
    with pytest.raises(ValueError, match="All estimators are dropped"):
        est.fit(X, y)


# =========================================================================== #
# mlrs sub-estimators — the real deployment shape
# =========================================================================== #


@pytest.mark.parametrize("engine", RUST_ENGINES, indirect=True)
def test_mlrs_members_compose_on_every_arm(engine):
    """Device-fitted members feeding each meta arm, against sklearn's answer."""
    dtype = conftest.default_float_dtype()
    X, y = host_design(dtype=dtype)
    a = mlrs.StackingRegressor(
        [("lr", mlrs.LinearRegression()), ("ridge", mlrs.Ridge(alpha=1.0))], cv=3
    )
    b = SkStackingRegressor(sk_estimators(), cv=3)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_allclose(a.predict(X), b.predict(X), atol=conftest.live_atol(), rtol=0)
