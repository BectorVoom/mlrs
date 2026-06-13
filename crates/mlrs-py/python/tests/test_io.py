"""Ingress/egress + base-shim unit tests (Task 1 — D-02/D-03/D-05).

These exercise the *pure-Python* layer (``mlrs._io`` + ``mlrs.base``) WITHOUT the
compiled ``_mlrs`` extension: ``normalize_X`` (fresh-contiguous pyarrow + shape,
D-02 / Pitfall 3), ``pick_dtype`` (default-dtype selection, D-05 / Pitfall 5),
``resolve_output_type`` + ``to_output`` (output_type mirror routing, D-03), and
``MlrsBase`` NotFitted handling + sklearn tags. They import the shim modules
directly (``mlrs._io`` / ``mlrs.base``) so they run before ``maturin develop``
(the family delegate path is covered by the live-extension gate).

Req: PY-03 (contiguous pyarrow ingress), PY-05 (default-dtype), D-03 (egress).
"""

import numpy as np
import pyarrow as pa
import pytest

from mlrs import _io
from mlrs.base import MlrsBase


# --------------------------------------------------------------------------- #
# normalize_X — fresh-contiguous pyarrow + (rows, cols)  (D-02 / Pitfall 3)
# --------------------------------------------------------------------------- #


def test_normalize_X_dense_float32_shape_and_len():
    arr, rows, cols = _io.normalize_X(np.eye(3, dtype=np.float32))
    assert (rows, cols) == (3, 3)
    assert len(arr) == 9
    assert pa.types.is_float32(arr.type)


def test_normalize_X_is_row_major_flatten():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    arr, rows, cols = _io.normalize_X(X)
    assert (rows, cols) == (2, 2)
    assert arr.to_pylist() == [1.0, 2.0, 3.0, 4.0]  # C-order ravel


def test_normalize_X_sliced_view_becomes_fresh_contiguous():
    # A non-contiguous numpy view (every other column) must come out as a fresh
    # contiguous pyarrow array with a zero offset and no parent buffer aliasing.
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    view = base[:, ::2]  # shape (3, 2), non-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    arr, rows, cols = _io.normalize_X(view)
    assert (rows, cols) == (3, 2)
    assert arr.offset == 0
    assert len(arr) == 6
    assert arr.to_pylist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_normalize_X_accepts_python_list():
    arr, rows, cols = _io.normalize_X([[1.0, 2.0, 3.0]])
    assert (rows, cols) == (1, 3)
    assert len(arr) == 3


def test_normalize_X_rejects_non_finite():
    X = np.array([[1.0, np.nan]], dtype=np.float64)
    with pytest.raises(ValueError):
        _io.normalize_X(X)


# --------------------------------------------------------------------------- #
# pick_dtype — default-dtype selection  (D-05 / Pitfall 5)
# --------------------------------------------------------------------------- #


def test_pick_dtype_preserves_float32():
    assert _io.pick_dtype(np.zeros((2, 2), dtype=np.float32)) == np.float32


def test_pick_dtype_preserves_float64():
    assert _io.pick_dtype(np.zeros((2, 2), dtype=np.float64)) == np.float64


def test_pick_dtype_integer_defaults_to_backend_float(monkeypatch):
    # On an f64-capable backend integer input defaults to float64...
    monkeypatch.setattr(_io, "_backend_supports_f64", lambda: True)
    assert _io.pick_dtype(np.array([[1, 2], [3, 4]])) == np.float64
    # ...and float32 on an f64-incapable backend (rocm).
    monkeypatch.setattr(_io, "_backend_supports_f64", lambda: False)
    assert _io.pick_dtype(np.array([[1, 2], [3, 4]])) == np.float32


# --------------------------------------------------------------------------- #
# resolve_output_type — input mirror  (D-03, narrowed set)
# --------------------------------------------------------------------------- #


def test_resolve_output_type_numpy_input():
    assert _io.resolve_output_type(np.eye(2), "input") == "numpy"


def test_resolve_output_type_pyarrow_input():
    table = pa.array([1.0, 2.0])
    assert _io.resolve_output_type(table, "input") == "pyarrow"


def test_resolve_output_type_list_input_defaults_numpy():
    assert _io.resolve_output_type([[1.0]], "input") == "numpy"


def test_resolve_output_type_explicit_overrides_input():
    assert _io.resolve_output_type(np.eye(2), "pyarrow") == "pyarrow"
    assert _io.resolve_output_type(pa.array([1.0]), "numpy") == "numpy"


# --------------------------------------------------------------------------- #
# to_output — egress wrapping (D-03); labels/indices materialize as int32
# --------------------------------------------------------------------------- #


def test_to_output_numpy_int_is_int32():
    out = _io.to_output([0, 1, 2], (3,), "numpy", np.int32)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int32
    assert out.tolist() == [0, 1, 2]


def test_to_output_numpy_float_reshapes():
    out = _io.to_output([1.0, 2.0, 3.0, 4.0], (2, 2), "numpy", np.float64)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)
    assert out.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_to_output_pyarrow_is_arrow_array():
    out = _io.to_output([1.0, 2.0], (2,), "pyarrow", np.float64)
    assert isinstance(out, pa.Array)
    assert out.to_pylist() == [1.0, 2.0]


# --------------------------------------------------------------------------- #
# CR-01: pyarrow egress must NOT silently flatten a genuine 2-D result
# --------------------------------------------------------------------------- #


def test_to_output_pyarrow_2d_matrix_raises_not_flatten():
    # A genuine (rows, cols) matrix (e.g. predict_proba -> (2, 3)) has no
    # faithful 1-D pyarrow representation; to_output must raise, never ravel it
    # to a length-6 flat array of lost geometry (the CR-01 corruption).
    buf = [0.1, 0.2, 0.7, 0.3, 0.3, 0.4]
    with pytest.raises(ValueError, match="2-D results"):
        _io.to_output(buf, (2, 3), "pyarrow", np.float64)


def test_to_output_pyarrow_2d_with_inferred_axis_raises():
    # The decomposition shim passes (rows, -1) for transform; the -1 resolves to
    # a >1 column count here, so it is a matrix and must still raise.
    buf = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    with pytest.raises(ValueError, match="2-D results"):
        _io.to_output(buf, (2, -1), "pyarrow", np.float64)


def test_to_output_pyarrow_column_vector_is_flattened_ok():
    # A (rows, 1) single-column result IS a vector and may flatten to a 1-D
    # arrow array without losing geometry.
    out = _io.to_output([1.0, 2.0, 3.0], (3, 1), "pyarrow", np.float64)
    assert isinstance(out, pa.Array)
    assert out.to_pylist() == [1.0, 2.0, 3.0]


def test_to_output_numpy_2d_matrix_preserves_shape():
    # The numpy path must always preserve 2-D geometry (the safe container).
    out = _io.to_output(
        [0.1, 0.2, 0.7, 0.3, 0.3, 0.4], (2, 3), "numpy", np.float64
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 3)
    assert out.tolist() == [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]


# --------------------------------------------------------------------------- #
# WR-01: normalize_y must reject non-finite (NaN/Inf) targets
# --------------------------------------------------------------------------- #


def test_normalize_y_rejects_nan():
    y = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _io.normalize_y(y, dtype=np.float64)


def test_normalize_y_rejects_inf():
    y = np.array([1.0, np.inf, 3.0], dtype=np.float64)
    with pytest.raises(ValueError):
        _io.normalize_y(y, dtype=np.float64)


def test_normalize_y_accepts_finite():
    arr = _io.normalize_y(np.array([1.0, 2.0, 3.0]), dtype=np.float64)
    assert isinstance(arr, pa.Array)
    assert arr.to_pylist() == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# WR-03: f64 capability probe must FAIL CLOSED (False) on error
# --------------------------------------------------------------------------- #


def test_backend_supports_f64_fails_closed_on_probe_error(monkeypatch):
    # If the capability query throws, the probe must return False (assume
    # f64-incapable) so f64 is never defaulted onto an unknown/erroring backend.
    # `from . import _mlrs` resolves the `_mlrs` attribute on the mlrs package,
    # so patch that attribute to an object whose backend_supports_f64() raises.
    import types

    import mlrs as _pkg

    fake = types.SimpleNamespace()

    def _boom():
        raise RuntimeError("simulated capability-probe failure")

    fake.backend_supports_f64 = _boom
    monkeypatch.setattr(_pkg, "_mlrs", fake, raising=False)
    assert _io._backend_supports_f64() is False


def test_backend_supports_f64_missing_extension_fails_closed(monkeypatch):
    # When the extension cannot be imported at all, the probe must also fail
    # closed (False), not optimistically report f64-capable. Simulate an
    # unimportable extension by clearing the cached attr AND poisoning
    # sys.modules so `from . import _mlrs` re-imports and fails.
    import sys

    import mlrs as _pkg

    monkeypatch.delattr(_pkg, "_mlrs", raising=False)
    monkeypatch.setitem(sys.modules, "mlrs._mlrs", None)
    assert _io._backend_supports_f64() is False


# --------------------------------------------------------------------------- #
# MlrsBase — output_type purity, NotFitted, sklearn tags
# --------------------------------------------------------------------------- #


def test_mlrsbase_stores_output_type_verbatim():
    b = MlrsBase()
    assert b.output_type == "input"
    assert MlrsBase(output_type="numpy").output_type == "numpy"


def test_mlrsbase_check_fitted_raises_before_fit():
    from sklearn.exceptions import NotFittedError

    class _Dummy(MlrsBase):
        def __init__(self, output_type="input"):
            self.output_type = output_type

        def fit(self, X, y=None):  # noqa: D401 - placeholder for check_is_fitted
            return self

    with pytest.raises(NotFittedError):
        _Dummy()._check_fitted()


def test_mlrsbase_sklearn_tags_disable_sparse_nan_arrayapi():
    tags = MlrsBase().__sklearn_tags__()
    assert tags.input_tags.sparse is False
    assert tags.input_tags.allow_nan is False
    assert tags.array_api_support is False
