"""Input/output normalization for the mlrs shim (D-02 / D-03 / D-05).

The pure-Python host<->device boundary helpers the family-module wrappers call
around every ``_mlrs`` delegate:

* :func:`normalize_X` — sklearn ``check_array`` (shape/finite validation so the
  error messages match ``estimator_checks`` point 5) then a *freshly-contiguous*
  row-major 1-D pyarrow float array + ``(rows, cols)``. The Rust bridge
  (``validate_f32`` / ``validate_f64``) HARD-REJECTS sliced/offset arrays — it
  requires the values view to cover the whole backing buffer — so the shim hands
  a fresh ``pa.array(np.ascontiguousarray(X).ravel())``, never a zero-copy slice
  of a larger numpy buffer (RESEARCH 06 Pitfall 3 / T-06-11).
* :func:`pick_dtype` — preserve the input float dtype; for non-float inputs
  default to ``float64`` *where supported*, ``float32`` on an f64-incapable
  backend (rocm) via ``_mlrs.backend_supports_f64()`` so integer/list inputs
  don't hit the D-04 error (RESEARCH 06 Pitfall 5 / D-05).
* :func:`resolve_output_type` — map the default ``"input"`` to ``"numpy"`` or
  ``"pyarrow"`` by the input container (narrowed set — D-03 egress mirror).
* :func:`to_output` — wrap a host buffer back into numpy or pyarrow; integer
  labels/indices materialize as ``int32`` (D-03 / D-06).

``_mlrs`` is imported *lazily* (only inside :func:`_backend_supports_f64`) so the
pure-Python layer is importable and unit-testable before ``maturin develop``.
"""

import numpy as np
import pyarrow as pa
from sklearn.utils import check_array

# The narrowed output_type set (D-03): mlrs mirrors numpy and pyarrow only,
# unlike cuML's wider cudf/cupy/numba set.
_NUMPY = "numpy"
_PYARROW = "pyarrow"


def _backend_supports_f64():
    """Query the compiled backend's f64 capability (D-05 / D-04).

    Imported lazily so ``mlrs._io`` is importable without the ``_mlrs``
    extension (the pure-Python unit tests monkeypatch this). If the extension
    is unavailable we conservatively report ``True`` (f64-capable) — the real
    f64 guard still lives in the Rust ``guard_f64`` on the f64 dispatch arm.
    """
    try:
        from . import _mlrs

        return bool(_mlrs.backend_supports_f64())
    except Exception:
        return True


def pick_dtype(X):
    """Pick the float dtype to upload ``X`` as (D-05).

    Preserves an existing ``float32`` / ``float64`` input dtype; for any
    non-float input (integer, bool, python list already coerced upstream) it
    defaults to ``float64`` on an f64-capable backend and ``float32`` on an
    f64-incapable backend (rocm) so the default path never trips the D-04 guard.
    """
    dtype = getattr(np.asarray(X).dtype, "type", None)
    if dtype is np.float32:
        return np.float32
    if dtype is np.float64:
        return np.float64
    return np.float64 if _backend_supports_f64() else np.float32


def normalize_X(X, *, dtype=None):
    """Normalize ``X`` to a fresh-contiguous 1-D pyarrow float array + shape.

    Runs sklearn ``check_array`` first (2-D, finite — so dimension/NaN errors
    match ``estimator_checks``), resolves the upload dtype (:func:`pick_dtype`
    unless ``dtype`` is given), then ``np.ascontiguousarray(...).ravel()`` ->
    ``pa.array`` so the result is a FRESH contiguous buffer (offset 0, no parent
    aliasing — Pitfall 3 / T-06-11), never a numpy slice.

    Returns ``(pyarrow_array, rows, cols)`` where the array is row-major
    flattened (``rows * cols`` elements).
    """
    if dtype is None:
        dtype = pick_dtype(X)
    # check_array(force_all_finite is renamed ensure_all_finite in sklearn>=1.6;
    # pass ensure_all_finite for forward compat, fall back for older sklearn).
    try:
        arr = check_array(
            X,
            ensure_all_finite=True,
            ensure_2d=True,
            dtype=dtype,
            copy=False,
        )
    except TypeError:  # pragma: no cover - pre-1.6 sklearn fallback
        arr = check_array(
            X,
            force_all_finite=True,
            ensure_2d=True,
            dtype=dtype,
            copy=False,
        )
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    # FRESH contiguous row-major buffer — never a slice of a larger array.
    flat = np.ascontiguousarray(arr, dtype=dtype).ravel(order="C")
    return pa.array(flat, type=_arrow_float_type(dtype)), rows, cols


def normalize_y(y, *, dtype):
    """Normalize a 1-D target ``y`` to a fresh-contiguous pyarrow float array.

    Used by the supervised wrappers (LinearRegression/Ridge/.../KNNReg/Clf). The
    Rust ``fit`` uploads ``y`` as the same float dtype as ``X`` and infers class
    labels (for classifiers) from the float values, so ``y`` is uploaded as a
    fresh contiguous float buffer mirroring :func:`normalize_X`.
    """
    arr = np.ascontiguousarray(np.asarray(y), dtype=dtype).ravel(order="C")
    return pa.array(arr, type=_arrow_float_type(dtype))


def resolve_output_type(input_obj, output_type):
    """Resolve the egress ``output_type`` against the input container (D-03).

    With the default ``"input"`` the egress mirrors the container the data
    arrived in: a pyarrow input -> ``"pyarrow"``, anything else (numpy / list)
    -> ``"numpy"`` (the narrowed mlrs set). An explicit ``"numpy"`` / ``"pyarrow"``
    overrides the mirror.
    """
    if output_type in (_NUMPY, _PYARROW):
        return output_type
    # output_type == "input" (or anything unrecognized) -> mirror the container.
    if isinstance(input_obj, (pa.Array, pa.ChunkedArray, pa.Table)):
        return _PYARROW
    return _NUMPY


def to_output(buf, shape, output_type, dtype):
    """Wrap a host buffer ``buf`` into the resolved ``output_type`` (D-03 / D-06).

    ``buf`` is a flat python list / 1-D sequence of host values; ``shape`` is the
    target numpy shape (``(rows,)`` for vectors, ``(rows, cols)`` for matrices;
    one axis may be ``-1`` for caller-inferred dims). Integer labels/indices
    materialize as ``int32`` (D-06); floats keep ``dtype``.

    The host buffer is reshaped to ``shape`` FIRST so the geometry the egress
    contract carries (``egress.rs`` ``FloatResult`` ``(rows, cols)``) is honored
    on every path. For ``"numpy"`` the shaped array is returned directly. For
    ``"pyarrow"`` a 1-D arrow array is returned for genuine vectors (``ndim == 1``
    or a ``(rows, 1)`` single-column result); a genuine 2-D matrix (``ndim > 1``
    with more than one column) CANNOT be faithfully represented as a 1-D columnar
    pyarrow ``Array``, so this raises ``ValueError`` rather than silently
    flattening it (CR-01 - D-03 narrowed-set; request ``output_type='numpy'``).
    """
    np_dtype = np.dtype(dtype)
    flat = np.asarray(buf, dtype=np_dtype)
    arr = flat.reshape(shape)
    if output_type == _PYARROW:
        # pyarrow Array is 1-D / columnar; a matrix has no faithful 1-D form.
        # A (rows, 1) single-column result IS a vector and ravels safely; any
        # other 2-D shape is a genuine matrix - refuse to flatten it so a
        # predict_proba / transform / kneighbors result is never silently
        # corrupted into a flat array of lost geometry (CR-01).
        if arr.ndim > 1 and arr.shape[1] != 1:
            raise ValueError(
                "mlrs: pyarrow output_type is unsupported for 2-D results "
                f"(shape {arr.shape}); request output_type='numpy'."
            )
        return pa.array(arr.ravel(order="C"))
    return arr


def _arrow_float_type(dtype):
    """The pyarrow float type matching a numpy float dtype."""
    if np.dtype(dtype) == np.float32:
        return pa.float32()
    return pa.float64()
