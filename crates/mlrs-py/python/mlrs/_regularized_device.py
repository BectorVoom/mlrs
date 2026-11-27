"""Dispatch the dense design's Gram and cross-products to CubeCL."""

import numpy as np
import pyarrow as pa
from scipy import sparse


def fit_device(device, X, *, require_float64=False):
    """Resolve an explicit preference or use the host for small auto fits."""
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'.")
    if device == "cpu":
        return "cpu"
    rows, features = X.shape
    if device == "auto" and (
        sparse.issparse(X) or rows < features or rows * features**2 < 2**22
    ):
        return "cpu"
    if sparse.issparse(X):
        raise ValueError("The regularized linear GPU path requires dense input.")
    from . import _load_ext

    try:
        backend = _load_ext().regularized_gram_backend(
            features, require_float64 or X.dtype == np.float64
        )
    except (ImportError, AttributeError) as err:
        if device == "auto":
            return "cpu"
        raise ValueError("device='gpu' requires a GPU-enabled mlrs build.") from err
    if backend is None:
        if device == "auto":
            return "cpu"
        raise ValueError(
            "This backend, dtype, or feature count does not support the device Gram "
            "(requires a GPU backend and at most 256 features)."
        )
    return "gpu"


def device_gram(X, Y, dtype):
    """Return unnormalized X.T @ X and X.T @ Y for already-preprocessed inputs."""
    from . import _load_ext

    ext = _load_ext()
    is_f64 = dtype == np.float64
    compute = ext.regularized_gram_f64 if is_f64 else ext.regularized_gram_f32
    rows, features = X.shape
    Y = Y.reshape(rows, -1)
    outputs = Y.shape[1]
    gram, cross = compute(
        pa.array(np.ascontiguousarray(X, dtype=dtype).ravel()),
        pa.array(np.ascontiguousarray(Y, dtype=dtype).ravel()),
        rows,
        features,
        outputs,
    )
    gram = np.asarray(gram, dtype=np.float64).reshape(features, features)
    cross = np.asarray(cross, dtype=np.float64).reshape(features, outputs)
    if not np.isfinite(gram).all() or not np.isfinite(cross).all():
        raise ValueError(
            "The device Gram calculation overflowed. Use float64 or rescale the input."
        )
    return gram, cross
