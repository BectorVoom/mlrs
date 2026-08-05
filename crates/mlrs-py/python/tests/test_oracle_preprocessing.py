"""Preprocessing scaler oracle harness (PREP-01: full numpy->pyarrow->FFI->device path).

Re-validates the 1e-5 contract for the six PREP-01 estimators (StandardScaler /
MinMaxScaler / MaxAbsScaler / RobustScaler / Normalizer / Binarizer) through the
FULL Python binding path by replaying the committed ``tests/fixtures/*.npz``
sklearn-reference blobs (``scripts/gen_oracle.py``'s ``gen_standard_scaler`` /
``gen_min_max_scaler`` / ``gen_max_abs_scaler`` / ``gen_robust_scaler`` /
``gen_normalizer`` / ``gen_binarizer``). Every fixture pins one constant (or
all-zero / all-threshold) column/row, so this ALSO covers the degenerate
zero-scale gate end to end.

f64 fixtures are skipped-with-reason on an f64-incapable backend via
``_skip_unsupported_dtype``, PER FIXTURE.

Not the blanket ``conftest.requires_f64`` marker: that is a module-level
``skipif`` evaluated once at import, so decorating a function parametrized over
BOTH widths throws away the f32 half too — on a rocm wheel or a wgpu adapter
without ``SHADER_F64`` the whole PREP-01 suite would report "skipped" and
nothing would gate these six estimators on one of the primary correctness
backends. Same reasoning as ``test_oracle_linear.py::test_ridge_params_oracle``.
"""

import numpy as np
import pytest

import mlrs
from conftest import dtype_of, fixture_path


def _skip_unsupported_dtype(fixture):
    if dtype_of(fixture) == np.float64 and not mlrs.backend_supports_f64():
        pytest.skip("backend does not support f64")


def _atol(fixture):
    return 1e-5 if dtype_of(fixture) == np.float64 else 1e-4


# --- StandardScaler / MinMaxScaler / MaxAbsScaler / RobustScaler: fitted
#     attrs + transform + inverse_transform -------------------------------- #

SCALER_CASES = [
    ("standard_scaler_f32_seed42", lambda d: mlrs.StandardScaler(), ("mean_", "mean_"), ("var_", "var_"), ("scale_", "scale_")),
    ("standard_scaler_f64_seed42", lambda d: mlrs.StandardScaler(), ("mean_", "mean_"), ("var_", "var_"), ("scale_", "scale_")),
    ("min_max_scaler_f32_seed42", lambda d: mlrs.MinMaxScaler(feature_range=tuple(d["feature_range"])), ("data_min_", "data_min_"), ("data_max_", "data_max_"), ("scale_", "scale_")),
    ("min_max_scaler_f64_seed42", lambda d: mlrs.MinMaxScaler(feature_range=tuple(d["feature_range"])), ("data_min_", "data_min_"), ("data_max_", "data_max_"), ("scale_", "scale_")),
    ("max_abs_scaler_f32_seed42", lambda d: mlrs.MaxAbsScaler(), ("max_abs_", "max_abs_"), ("scale_", "scale_"), None),
    ("max_abs_scaler_f64_seed42", lambda d: mlrs.MaxAbsScaler(), ("max_abs_", "max_abs_"), ("scale_", "scale_"), None),
    ("robust_scaler_f32_seed42", lambda d: mlrs.RobustScaler(), ("center_", "center_"), ("scale_", "scale_"), None),
    ("robust_scaler_f64_seed42", lambda d: mlrs.RobustScaler(), ("center_", "center_"), ("scale_", "scale_"), None),
]


@pytest.mark.parametrize(
    "fixture,builder,attr1,attr2,attr3",
    SCALER_CASES,
    ids=[c[0] for c in SCALER_CASES],
)
def test_scaler_oracle(fixture, builder, attr1, attr2, attr3):
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    est = builder(d).fit(d["X"])
    atol = _atol(fixture)

    for attr in (attr1, attr2, attr3):
        if attr is None:
            continue
        est_name, npz_name = attr
        assert np.allclose(np.ravel(getattr(est, est_name)), np.ravel(d[npz_name]), atol=atol, rtol=1e-5)

    got_transform = est.transform(d["X"])
    assert np.allclose(got_transform, d["transform"], atol=atol, rtol=1e-5)

    got_inverse = est.inverse_transform(got_transform)
    assert np.allclose(got_inverse, d["inverse"], atol=atol, rtol=1e-5)


# --- Normalizer: no fitted stats, transform-only, three norms -------------- #

NORMALIZER_CASES = [
    (f"normalizer_{norm}_{dtype_tag}_seed42", norm)
    for norm in ("l1", "l2", "max")
    for dtype_tag in ("f32", "f64")
]


@pytest.mark.parametrize("fixture,norm", NORMALIZER_CASES, ids=[c[0] for c in NORMALIZER_CASES])
def test_normalizer_oracle(fixture, norm):
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    got = mlrs.Normalizer(norm=norm).fit_transform(d["X"])
    assert np.allclose(got, d["transform"], atol=_atol(fixture), rtol=1e-5)


# --- Binarizer: no fitted stats, transform-only, exact structural (0/1) --- #

BINARIZER_CASES = ["binarizer_f32_seed42", "binarizer_f64_seed42"]


@pytest.mark.parametrize("fixture", BINARIZER_CASES)
def test_binarizer_oracle(fixture):
    _skip_unsupported_dtype(fixture)
    d = np.load(fixture_path(fixture))
    threshold = float(d["threshold"][0])
    got = mlrs.Binarizer(threshold=threshold).fit_transform(d["X"])
    # Structural gate (PREP-02 convention applied here too): binarize is exact
    # 0/1, not a tolerance-band float compare.
    assert np.array_equal(got, d["transform"])


# --- sklearn API contract: get_params / clone / fit_transform / NotFittedError

def test_sklearn_api_contract():
    from sklearn.base import clone
    from sklearn.exceptions import NotFittedError

    est = mlrs.StandardScaler(with_mean=False)
    params = est.get_params()
    assert params["with_mean"] is False
    assert params["with_std"] is True

    cloned = clone(est)
    assert cloned.with_mean is False

    with pytest.raises(NotFittedError):
        mlrs.StandardScaler().transform(np.zeros((2, 2)))


def test_min_max_scaler_invalid_feature_range():
    with pytest.raises(ValueError):
        mlrs.MinMaxScaler(feature_range=(1.0, 0.0)).fit(np.zeros((4, 2)))


def test_robust_scaler_invalid_quantile_range():
    with pytest.raises(ValueError):
        mlrs.RobustScaler(quantile_range=(75.0, 25.0)).fit(np.zeros((4, 2)))


def test_normalizer_unknown_norm():
    with pytest.raises(ValueError):
        mlrs.Normalizer(norm="bogus").fit(np.zeros((4, 2)))


# --- Refit on a REUSED extension handle keeps the constructor's params ------ #

def test_ext_scaler_refit_keeps_hyperparameters():
    """A second ``fit`` on the same ``_mlrs`` object must not reset params.

    The Rust ``fit`` used to read its hyperparameters out of the ``Any*::Unfit``
    enum arm, which ``fit`` itself overwrites — so the second call fell through
    to a ``_`` arm holding sklearn's DEFAULTS and silently re-fitted with them
    (``feature_range`` back to ``(0, 1)``) while ``get_params()`` still reported
    what the caller asked for. The shim hides this today by constructing a fresh
    handle per ``fit``, so the test goes at the extension class directly: it is a
    public ``#[pyclass]``, and reusing the handle across fits is the obvious
    optimization to make later.
    """
    from mlrs import _io
    from mlrs.base import MlrsBase

    dtype = np.float64 if mlrs.backend_supports_f64() else np.float32
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=dtype)
    suffix = "f64" if dtype == np.float64 else "f32"

    # feature_range=(-2, 3), NOT the (0, 1) default.
    obj = MlrsBase._ext().MinMaxScaler(-2.0, 3.0, False)

    def fit_then_transform():
        xa, rows, cols = _io.normalize_X(X, dtype=dtype)
        obj.fit(xa, rows, cols)
        xa, rows, cols = _io.normalize_X(X, dtype=dtype)
        return np.asarray(getattr(obj, f"transform_{suffix}")(xa, rows, cols))

    first = fit_then_transform()
    second = fit_then_transform()

    assert np.allclose(first, second, atol=1e-6), (
        "refit changed the transform: the second fit fell back to the default "
        f"feature_range. first={first.tolist()} second={second.tolist()}"
    )
    # And the range really is the constructed one, not (0, 1).
    assert np.isclose(first.min(), -2.0, atol=1e-6)
    assert np.isclose(first.max(), 3.0, atol=1e-6)
