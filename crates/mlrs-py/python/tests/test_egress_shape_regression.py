"""Egress-shape + non-finite-y regression tests through the FULL binding path.

Post-review regression coverage (06-REVIEW CR-01 / WR-01) that the oracle suite
missed because it only ever requested ``output_type='numpy'``:

* **CR-01** — under ``output_type='pyarrow'`` (or a pyarrow input mirrored to
  pyarrow egress) a genuinely 2-D result (``PCA.transform`` ->
  ``(rows, n_components)``, ``LogisticRegression.predict_proba`` ->
  ``(rows, n_classes)``) MUST NOT come back silently flattened to a 1-D arrow
  array of lost geometry. The shim raises a clear ``ValueError`` instead (a 2-D
  matrix has no faithful 1-D columnar pyarrow form); the numpy path still
  preserves the full 2-D shape.
* **WR-01** — a supervised ``fit(X, y)`` with NaN/Inf in ``y`` is rejected with
  a sklearn-standard ``ValueError`` (``_io.normalize_y`` now runs
  ``check_array(ensure_all_finite=True)``), never uploaded to the device.

These require the compiled ``_mlrs`` extension (``maturin develop`` with a
backend feature); they are collected-and-skipped otherwise so the suite stays
green pre-build.
"""

import numpy as np
import pyarrow as pa
import pytest

pytest.importorskip("mlrs")

import mlrs  # noqa: E402  (after importorskip)


# --------------------------------------------------------------------------- #
# CR-01: pyarrow egress must not silently flatten a 2-D result
# --------------------------------------------------------------------------- #


def _pca_input(rows=20, cols=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float64)


def test_pca_transform_numpy_output_is_2d():
    # Baseline: the numpy path preserves the full (rows, n_components) geometry.
    X = _pca_input()
    est = mlrs.PCA(n_components=2, output_type="numpy").fit(X)
    out = est.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (X.shape[0], 2)


def test_pca_transform_pyarrow_output_does_not_flatten():
    # CR-01: a 2-D transform result under pyarrow egress must RAISE, not silently
    # ravel to a length rows*n_components flat arrow array of lost geometry.
    X = _pca_input()
    est = mlrs.PCA(n_components=2, output_type="pyarrow").fit(X)
    with pytest.raises(ValueError, match="2-D results"):
        est.transform(X)


def test_pca_transform_default_numpy_input_stays_2d():
    # No false positive: the default output_type='input' with a numpy input
    # mirrors to numpy egress, so a 2-D transform is preserved (not raised).
    X = _pca_input()
    est = mlrs.PCA(n_components=2).fit(X)  # output_type='input' (default)
    out = est.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (X.shape[0], 2)


def _classification_data(rows=30, cols=3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, cols)).astype(np.float64)
    # Two well-separated classes so the solver converges to a proper proba.
    y = (X[:, 0] > 0).astype(np.float64)
    return X, y


def test_logreg_predict_proba_numpy_output_is_2d():
    X, y = _classification_data()
    est = mlrs.LogisticRegression(output_type="numpy").fit(X, y)
    proba = est.predict_proba(X)
    assert isinstance(proba, np.ndarray)
    assert proba.ndim == 2
    assert proba.shape[0] == X.shape[0]
    assert proba.shape[1] >= 2  # (rows, n_classes)


def test_logreg_predict_proba_pyarrow_output_does_not_flatten():
    # CR-01: predict_proba -> (rows, n_classes) is a genuine matrix; pyarrow
    # egress must raise rather than flatten the probability matrix.
    X, y = _classification_data()
    est = mlrs.LogisticRegression(output_type="pyarrow").fit(X, y)
    with pytest.raises(ValueError, match="2-D results"):
        est.predict_proba(X)


def test_logreg_predict_labels_pyarrow_output_ok_1d():
    # A 1-D result (predict labels -> (rows,)) is faithfully representable as a
    # pyarrow Array and must NOT be affected by the 2-D guard.
    X, y = _classification_data()
    est = mlrs.LogisticRegression(output_type="pyarrow").fit(X, y)
    labels = est.predict(X)
    assert isinstance(labels, pa.Array)
    assert len(labels) == X.shape[0]


# --------------------------------------------------------------------------- #
# WR-01: non-finite y is rejected through the full fit path
# --------------------------------------------------------------------------- #


def test_fit_rejects_nan_y_linear():
    X = _pca_input(rows=10, cols=3)
    y = np.arange(10, dtype=np.float64)
    y[3] = np.nan
    with pytest.raises(ValueError):
        mlrs.LinearRegression().fit(X, y)


def test_fit_rejects_inf_y_linear():
    X = _pca_input(rows=10, cols=3)
    y = np.arange(10, dtype=np.float64)
    y[5] = np.inf
    with pytest.raises(ValueError):
        mlrs.Ridge().fit(X, y)


def test_fit_rejects_nan_y_logreg():
    X, y = _classification_data(rows=12, cols=3)
    y[2] = np.nan
    with pytest.raises(ValueError):
        mlrs.LogisticRegression().fit(X, y)


def test_fit_accepts_finite_y():
    # Sanity: a finite y still fits cleanly (no false rejection from WR-01).
    X = _pca_input(rows=10, cols=3)
    y = np.arange(10, dtype=np.float64)
    est = mlrs.LinearRegression().fit(X, y)
    assert est.n_features_in_ == 3
