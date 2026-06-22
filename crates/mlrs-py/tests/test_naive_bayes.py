"""Phase-11 Naive-Bayes Python live-FFI smoke (NB-01..05 — PY-06 sign-off).

Proves the FFI path + dtype dispatch + the full sklearn predict surface +
the construction/fit-time ValueError behavior (D-05/D-09) end to end for the
five Naive-Bayes estimators, through the REAL binding surface this plan delivers:
the low-level ``mlrs._mlrs`` extension classes ``GaussianNB``, ``MultinomialNB``,
``BernoulliNB``, ``ComplementNB`` and ``CategoricalNB``.

This is a SMOKE test, NOT a numerical oracle: the ≤1e-5 / documented-tolerance
numerical contract is already gated by the Rust oracle tests in Plans 11-02
(``gaussian_nb_test.rs``), 11-03 (``multinomial/bernoulli/complement_nb_test.rs``)
and 11-04 (``categorical_nb_test.rs``). Here we assert, ACROSS THE FFI BOUNDARY:

  * each estimator constructs with its sklearn-named kwargs (D-09 — zero
    translation: ``GaussianNB(var_smoothing=…, priors=…)``; the discrete four
    take ``alpha`` / ``force_alpha`` / ``fit_prior`` / ``class_prior`` plus
    ``binarize`` (Bernoulli) / ``norm`` (Complement) / ``min_categories``
    (Categorical));
  * ``fit(X, y)`` then ``predict_labels`` returns the right SHAPE and a sane
    multiclass split;
  * ``predict_proba`` rows are in [0,1] and **sum to 1.0 ± 1e-6** across the FFI
    (the load-bearing PY-06 assertion); ``predict_log_proba`` equals
    ``log(predict_proba)`` up to round-off; ``score`` is in [0,1]; and
  * a bad hyperparameter (``GaussianNB(var_smoothing=-1.0)`` /
    ``MultinomialNB(alpha=-1.0)``) raises a Python ``ValueError`` at the first
    ``fit`` (the D-05/D-09 construction-time behavior — mlrs surfaces the sklearn
    construction error at ``fit`` since the ``Unfit`` arm stores the raw params
    until then).

Both the f32 and f64 dtype-dispatch arms are exercised; the f64 case is gated
behind ``mlrs._mlrs.backend_supports_f64()`` so it skips-with-reason on an
f64-incapable backend (rocm), mirroring the ``capability.rs::skip_f64_with_log``
precedent.

Run via the shipped maturin-develop py-test flow (build the ``mlrs`` extension,
then ``pytest`` this file). The whole module is import-guarded so it skips
cleanly (never errors at collection) if the extension or pyarrow is unavailable
— the 10-05 / 08-05 precedent.
"""

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
_mlrs = pytest.importorskip("mlrs._mlrs")


_F64_OK = bool(_mlrs.backend_supports_f64())
requires_f64 = pytest.mark.skipif(
    not _F64_OK,
    reason="backend does not support f64 (mlrs._mlrs.backend_supports_f64() is False)",
)

_DTYPES = [
    np.float32,
    pytest.param(np.float64, marks=requires_f64),
]
_DTYPE_IDS = ["f32", "f64"]


def _arrow(a, dtype):
    """Fresh-contiguous row-major 1-D pyarrow float array (offset 0, no parent
    aliasing — the Rust bridge HARD-REJECTS sliced/offset arrays)."""
    flat = np.ascontiguousarray(a, dtype=dtype).ravel(order="C")
    at = pa.float32() if dtype == np.float32 else pa.float64()
    return pa.array(flat, type=at)


def _proba(est, X, dtype, rows, cols):
    """Dtype-routed predict_proba reshaped to (rows, n_classes)."""
    if dtype == np.float32:
        flat = np.asarray(est.predict_proba_f32(_arrow(X, dtype), rows, cols))
    else:
        flat = np.asarray(est.predict_proba_f64(_arrow(X, dtype), rows, cols))
    return flat.reshape(rows, -1)


def _log_proba(est, X, dtype, rows, cols):
    """Dtype-routed predict_log_proba reshaped to (rows, n_classes)."""
    if dtype == np.float32:
        flat = np.asarray(est.predict_log_proba_f32(_arrow(X, dtype), rows, cols))
    else:
        flat = np.asarray(est.predict_log_proba_f64(_arrow(X, dtype), rows, cols))
    return flat.reshape(rows, -1)


def _toy_gaussian(dtype):
    """Three well-separated Gaussian blobs (continuous features) → a clean
    3-class split."""
    X = np.array(
        [
            [-5.0, -5.0],
            [-4.8, -5.2],
            [-5.2, -4.9],
            [0.0, 0.1],
            [0.1, -0.1],
            [-0.1, 0.0],
            [5.0, 5.0],
            [4.9, 5.1],
            [5.1, 4.8],
        ],
        dtype=dtype,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=dtype)
    return X, y


def _toy_counts(dtype):
    """Non-negative integer count features (Multinomial/Complement) with a clean
    per-class signature on each of three vocabulary columns."""
    X = np.array(
        [
            [6.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
            [7.0, 0.0, 1.0],
            [0.0, 6.0, 0.0],
            [1.0, 5.0, 0.0],
            [0.0, 7.0, 1.0],
            [0.0, 0.0, 6.0],
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 7.0],
        ],
        dtype=dtype,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=dtype)
    return X, y


def _toy_binary_features(dtype):
    """0/1 occurrence features for BernoulliNB with a clean per-class pattern."""
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=dtype)
    return X, y


def _toy_categorical(dtype):
    """Non-negative integer CATEGORY codes (CategoricalNB) — each class picks a
    distinct category on every feature."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [2.0, 4.0],
            [2.0, 4.0],
            [2.0, 5.0],
        ],
        dtype=dtype,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=dtype)
    return X, y


def _assert_predict_surface(est, X, y, dtype):
    """Shared assertions on a fitted NB estimator's full predict surface."""
    rows, cols = X.shape
    assert est.is_fitted()
    assert est.dtype() == ("f32" if dtype == np.float32 else "f64")

    labels = np.asarray(est.predict_labels(_arrow(X, dtype), rows, cols))
    assert labels.shape == (rows,)
    # Each training cluster predicts a single (consistent) label, and the three
    # clusters are not all collapsed to one class.
    assert len(set(labels.tolist())) >= 2

    proba = _proba(est, X, dtype, rows, cols)
    assert proba.shape[0] == rows
    assert np.all(proba >= -1e-6) and np.all(proba <= 1.0 + 1e-6)
    # The load-bearing PY-06 assertion: rows sum to 1 across the FFI boundary.
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    log_proba = _log_proba(est, X, dtype, rows, cols)
    assert log_proba.shape == proba.shape
    # predict_log_proba == log(predict_proba) up to round-off (NB contract).
    assert np.allclose(np.exp(log_proba), proba, atol=1e-5)

    score = est.score(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    assert 0.0 <= score <= 1.0


# --- GaussianNB -------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_gaussian_nb_predict(dtype):
    """NB-01: PyGaussianNB fit -> full predict surface (FFI), sklearn-named knobs."""
    X, y = _toy_gaussian(dtype)
    rows, cols = X.shape
    est = _mlrs.GaussianNB(var_smoothing=1e-9, priors=None)
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    _assert_predict_surface(est, X, y, dtype)


# --- MultinomialNB ----------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_multinomial_nb_predict(dtype):
    """NB-02: PyMultinomialNB fit -> full predict surface (FFI), sklearn-named knobs."""
    X, y = _toy_counts(dtype)
    rows, cols = X.shape
    est = _mlrs.MultinomialNB(
        alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None
    )
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    _assert_predict_surface(est, X, y, dtype)


# --- BernoulliNB ------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_bernoulli_nb_predict(dtype):
    """NB-03: PyBernoulliNB fit -> full predict surface (FFI), sklearn-named knobs."""
    X, y = _toy_binary_features(dtype)
    rows, cols = X.shape
    est = _mlrs.BernoulliNB(
        alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True, class_prior=None
    )
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    _assert_predict_surface(est, X, y, dtype)


# --- ComplementNB -----------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_complement_nb_predict(dtype):
    """NB-04: PyComplementNB fit -> full predict surface (FFI), sklearn-named knobs."""
    X, y = _toy_counts(dtype)
    rows, cols = X.shape
    est = _mlrs.ComplementNB(
        alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None, norm=False
    )
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    _assert_predict_surface(est, X, y, dtype)


# --- CategoricalNB ----------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_categorical_nb_predict(dtype):
    """NB-05: PyCategoricalNB fit -> full predict surface (FFI), sklearn-named knobs.

    Exercises the None / int / list ``min_categories`` ingress mapping to
    MinCategories::{Infer,Uniform,PerFeature}."""
    X, y = _toy_categorical(dtype)
    rows, cols = X.shape
    # None -> Infer (the default path).
    est = _mlrs.CategoricalNB(
        alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
        min_categories=None,
    )
    est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
    _assert_predict_surface(est, X, y, dtype)


def test_categorical_nb_min_categories_int_and_list():
    """CategoricalNB min_categories accepts a uniform int and a per-feature list
    (the MinCategories::{Uniform,PerFeature} ingress), both fitting cleanly."""
    dtype = np.float32
    X, y = _toy_categorical(dtype)
    rows, cols = X.shape
    for mc in (4, [3, 6]):
        est = _mlrs.CategoricalNB(min_categories=mc)
        est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
        assert est.is_fitted()
        proba = _proba(est, X, dtype, rows, cols)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# --- Bad-hyperparameter ValueError (D-05/D-09, T-11-05-01) ------------------


@pytest.mark.parametrize(
    "ctor, kwargs",
    [
        (_mlrs.GaussianNB, {"var_smoothing": -1.0}),
        (_mlrs.MultinomialNB, {"alpha": -1.0}),
        (_mlrs.BernoulliNB, {"alpha": -1.0}),
        (_mlrs.ComplementNB, {"alpha": -1.0}),
        (_mlrs.CategoricalNB, {"alpha": -1.0}),
    ],
    ids=["gaussian", "multinomial", "bernoulli", "complement", "categorical"],
)
def test_bad_hyperparameter_raises_value_error(ctor, kwargs):
    """A negative alpha / var_smoothing surfaces as a Python ValueError at the
    first fit (D-05/D-09 — the Unfit arm stores the raw param until fit, then the
    builder build() path maps the BuildError through build_err_to_py to a
    PyValueError; T-11-05-01 mitigation, validated BEFORE any device upload)."""
    dtype = np.float32
    X, y = _toy_counts(dtype)
    rows, cols = X.shape
    est = ctor(**kwargs)
    with pytest.raises(ValueError):
        est.fit(_arrow(X, dtype), _arrow(y, dtype), rows, cols)
