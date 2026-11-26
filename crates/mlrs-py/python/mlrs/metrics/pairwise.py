"""``mlrs.metrics.pairwise`` — sklearn's ``sklearn.metrics.pairwise`` surface.

Every string metric and kernel is computed by the Rust pairwise engine
(``mlrs_algos::metrics::pairwise``: direct f64 evaluation, parallel over output
row blocks); the full matrix comes back as one zero-copy pyarrow column.
A CALLABLE metric is, as in sklearn, evaluated in Python once per pair.

Output dtype follows sklearn: ``float32`` when every input is ``float32`` and
the metric is one whose sklearn implementation preserves it (euclidean,
cosine, nan_euclidean, the paired distances, and every kernel except
laplacian), ``float64`` otherwise — sklearn routes manhattan, haversine,
laplacian and all of scipy's metrics through float64 code.
"""

import itertools
import warnings

import numpy as np

from ._base import _ext, _owned, _pa

__all__ = [
    "PAIRWISE_BOOLEAN_FUNCTIONS",
    "PAIRWISE_DISTANCE_FUNCTIONS",
    "PAIRWISE_KERNEL_FUNCTIONS",
    "additive_chi2_kernel",
    "check_paired_arrays",
    "check_pairwise_arrays",
    "chi2_kernel",
    "cosine_distances",
    "cosine_similarity",
    "distance_metrics",
    "euclidean_distances",
    "haversine_distances",
    "kernel_metrics",
    "laplacian_kernel",
    "linear_kernel",
    "manhattan_distances",
    "nan_euclidean_distances",
    "paired_cosine_distances",
    "paired_distances",
    "paired_euclidean_distances",
    "paired_manhattan_distances",
    "pairwise_distances",
    "pairwise_distances_argmin",
    "pairwise_distances_argmin_min",
    "pairwise_distances_chunked",
    "pairwise_kernels",
    "polynomial_kernel",
    "rbf_kernel",
    "sigmoid_kernel",
]

PAIRWISE_BOOLEAN_FUNCTIONS = ["dice", "jaccard", "rogerstanimoto", "russellrao", "sokalsneath", "yule"]

_NATIVE = {"euclidean", "l2", "l1", "manhattan", "cityblock", "cosine", "nan_euclidean", "haversine"}
# The metrics whose sklearn implementation returns float32 for float32 input.
_KEEPS_FLOAT32 = {"euclidean", "l2", "cosine", "nan_euclidean"}
_SCIPY = {
    "braycurtis",
    "canberra",
    "chebyshev",
    "correlation",
    "dice",
    "hamming",
    "jaccard",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalsneath",
    "sqeuclidean",
    "yule",
}
_VALID_METRICS = sorted(_NATIVE | _SCIPY)

_KERNEL_PARAMS = {
    "additive_chi2": (),
    "chi2": frozenset(["gamma"]),
    "cosine": (),
    "linear": (),
    "poly": frozenset(["gamma", "degree", "coef0"]),
    "polynomial": frozenset(["gamma", "degree", "coef0"]),
    "rbf": frozenset(["gamma"]),
    "laplacian": frozenset(["gamma"]),
    "sigmoid": frozenset(["gamma", "coef0"]),
}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


def _as_2d(a, name, allow_nan=False, finite=True):
    arr = np.asarray(a)
    if arr.ndim == 1:
        raise ValueError(
            "Expected 2D array, got 1D array instead:\narray=%r.\nReshape your data either using "
            "array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it "
            "contains a single sample." % (arr,)
        )
    if arr.ndim != 2:
        raise ValueError("Found array with dim %d. Expected <= 2." % arr.ndim)
    if finite:
        bad = ~np.isfinite(arr) if not allow_nan else np.isinf(arr)
        if arr.dtype.kind == "f" and np.any(bad):
            raise ValueError("Input %s contains NaN or infinity." % name if not allow_nan else "Input %s contains infinity." % name)
    return arr


def check_pairwise_arrays(
    X,
    Y,
    *,
    precomputed=False,
    dtype="infer_float",
    accept_sparse="csr",
    ensure_all_finite=True,
    ensure_2d=True,
    copy=False,
):
    """Matches ``sklearn.metrics.pairwise.check_pairwise_arrays`` for dense
    input: validates shapes and returns ``(X, Y)`` as float arrays (``Y is X``
    when ``Y`` is ``None``)."""
    allow_nan = ensure_all_finite == "allow-nan"
    finite = bool(ensure_all_finite)
    Xa = _as_2d(X, "X", allow_nan, finite)
    out_dtype = _float_dtype(Xa, None if Y is None else np.asarray(Y)) if dtype == "infer_float" else dtype
    Xa = np.array(Xa, dtype=out_dtype, copy=copy)
    if Y is None or Y is X:
        Ya = Xa
    else:
        Ya = np.array(_as_2d(Y, "Y", allow_nan, finite), dtype=out_dtype, copy=copy)
    if precomputed:
        if Xa.shape[1] != Ya.shape[0]:
            raise ValueError(
                "Precomputed metric requires shape (n_queries, n_indexed). Got (%d, %d) for %d "
                "indexed." % (Xa.shape[0], Xa.shape[1], Ya.shape[0])
            )
    elif Xa.shape[1] != Ya.shape[1]:
        raise ValueError(
            "Incompatible dimension for X and Y matrices: X.shape[1] == %d while Y.shape[1] == %d"
            % (Xa.shape[1], Ya.shape[1])
        )
    return Xa, Ya


def check_paired_arrays(X, Y):
    """Matches ``sklearn.metrics.pairwise.check_paired_arrays``."""
    Xa, Ya = check_pairwise_arrays(X, Y)
    if Xa.shape != Ya.shape:
        raise ValueError(
            "X and Y should be of same shape. They were respectively %r and %r long."
            % (Xa.shape, Ya.shape)
        )
    return Xa, Ya


def _float_dtype(X, Y):
    xs = [X] if Y is None else [X, Y]
    if all(np.asarray(a).dtype == np.float32 for a in xs):
        return np.float32
    return np.float64


def _flat(a):
    return _pa(np.ascontiguousarray(a, dtype=np.float64).ravel())


def _prepare(X, Y, allow_nan=False):
    """Validate a pair and return ``(x_pa, y_pa_or_None, n_x, n_y, d,
    out_dtype)``."""
    Xa = _as_2d(X, "X", allow_nan)
    Ya = None if Y is None else _as_2d(Y, "Y", allow_nan)
    if Ya is not None and Xa.shape[1] != Ya.shape[1]:
        raise ValueError(
            "Incompatible dimension for X and Y matrices: X.shape[1] == %d while Y.shape[1] == %d"
            % (Xa.shape[1], Ya.shape[1])
        )
    out_dtype = _float_dtype(Xa, Ya)
    n_y = Xa.shape[0] if Ya is None else Ya.shape[0]
    return Xa, Ya, n_y, out_dtype


def _metric_params(metric, X, Y, kwds):
    """Resolve a string metric and its scipy keyword arguments into the
    ``(name, p, w, V, VI)`` the Rust engine takes, deriving sklearn/scipy's
    ``V``/``VI`` defaults from ``X`` when ``Y`` is ``None``."""
    if metric not in _VALID_METRICS:
        raise ValueError(
            "Unknown metric %s. Valid metrics are %s, or 'precomputed', or a callable"
            % (metric, _VALID_METRICS)
        )
    p = w = v = vi = None
    if metric == "minkowski":
        p = float(kwds.get("p", 2))
        if kwds.get("w") is not None:
            w = [float(t) for t in np.asarray(kwds["w"]).ravel()]
    elif metric in ("seuclidean", "mahalanobis"):
        key = "V" if metric == "seuclidean" else "VI"
        given = kwds.get(key)
        if given is None:
            if Y is not None:
                raise ValueError("The '%s' parameter is required for the %s metric when Y is passed." % (key, metric))
            var, inv_cov = _ext().distance_defaults(_flat(X), np.asarray(X).shape[1])
            given = var if metric == "seuclidean" else inv_cov
        flat = [float(t) for t in np.asarray(given, dtype=np.float64).ravel()]
        if metric == "seuclidean":
            v = flat
        else:
            vi = flat
    if metric in PAIRWISE_BOOLEAN_FUNCTIONS and np.asarray(X).dtype != bool:
        from sklearn.exceptions import DataConversionWarning

        warnings.warn("Data was converted to boolean for metric %s" % metric, DataConversionWarning)
    return metric, p, w, v, vi


def _pairwise_callable(X, Y, metric, **kwds):
    """sklearn's ``_pairwise_callable``: one Python call per pair; ``Y = None``
    evaluates the upper triangle, mirrors it, then evaluates the diagonal."""
    if Y is None:
        n = X.shape[0]
        out = np.zeros((n, n), dtype=np.float64)
        for i, j in itertools.combinations(range(n), 2):
            out[i, j] = metric(X[i], X[j], **kwds)
        out = out + out.T
        for i in range(n):
            out[i, i] = metric(X[i], X[i], **kwds)
        return out
    out = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            out[i, j] = metric(X[i], Y[j], **kwds)
    return out


def _distances(X, Y, metric, kwds, allow_nan=False, missing=np.nan):
    Xa, Ya, n_y, out_dtype = _prepare(X, Y, allow_nan)
    name, p, w, v, vi = _metric_params(metric, Xa, Ya, kwds)
    flat = _ext().pairwise_distances(
        _flat(Xa), None if Ya is None else _flat(Ya), Xa.shape[1], name, p, w, v, vi, float(missing)
    )
    out = _owned(flat, shape=(Xa.shape[0], n_y))
    if name not in _KEEPS_FLOAT32:
        return out
    return out.astype(out_dtype, copy=False)


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #


def euclidean_distances(X, Y=None, *, Y_norm_squared=None, squared=False, X_norm_squared=None):
    """Matches ``sklearn.metrics.pairwise.euclidean_distances``. The
    ``*_norm_squared`` hints are accepted (and shape-checked) but not needed:
    mlrs evaluates ``Σ (x − y)²`` directly, which is never less accurate than
    the norm expansion they speed up."""
    if X_norm_squared is not None and np.asarray(X_norm_squared).size != np.asarray(X).shape[0]:
        raise ValueError("Incompatible dimensions for X and X_norm_squared")
    if Y_norm_squared is not None and Y is not None and np.asarray(Y_norm_squared).size != np.asarray(Y).shape[0]:
        raise ValueError("Incompatible dimensions for Y and Y_norm_squared")
    return _distances(X, Y, "sqeuclidean" if squared else "euclidean", {}).astype(
        _float_dtype(np.asarray(X), None if Y is None else np.asarray(Y)), copy=False
    )


def nan_euclidean_distances(X, Y=None, *, squared=False, missing_values=np.nan, copy=True):
    """Matches ``sklearn.metrics.pairwise.nan_euclidean_distances``."""
    out = _distances(X, Y, "nan_euclidean", {}, allow_nan=True, missing=missing_values)
    return out**2 if squared else out


def manhattan_distances(X, Y=None):
    """Matches ``sklearn.metrics.pairwise.manhattan_distances``."""
    return _distances(X, Y, "manhattan", {})


def cosine_distances(X, Y=None):
    """Matches ``sklearn.metrics.pairwise.cosine_distances`` (clipped to
    ``[0, 2]``, exact-zero diagonal when ``Y`` is ``None``)."""
    return _distances(X, Y, "cosine", {})


def haversine_distances(X, Y=None):
    """Matches ``sklearn.metrics.pairwise.haversine_distances`` (``(lat, lon)``
    in radians)."""
    if np.asarray(X).shape[1] != 2 or (Y is not None and np.asarray(Y).shape[1] != 2):
        raise ValueError("Haversine distance only valid in 2 dimensions")
    return _distances(X, Y, "haversine", {})


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #


def _kernel(X, Y, kernel, gamma=None, degree=3.0, coef0=1.0):
    Xa, Ya, n_y, out_dtype = _prepare(X, Y)
    if kernel == "laplacian":
        out_dtype = np.float64
    if gamma is None:
        gamma = 1.0 / Xa.shape[1]
    flat = _ext().pairwise_kernels(
        _flat(Xa), None if Ya is None else _flat(Ya), Xa.shape[1], kernel, float(gamma), float(degree), float(coef0)
    )
    return _owned(flat, shape=(Xa.shape[0], n_y)).astype(out_dtype, copy=False)


def linear_kernel(X, Y=None, dense_output=True):
    """Matches ``sklearn.metrics.pairwise.linear_kernel``."""
    return _kernel(X, Y, "linear")


def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """Matches ``sklearn.metrics.pairwise.polynomial_kernel``."""
    return _kernel(X, Y, "poly", gamma, degree, coef0)


def rbf_kernel(X, Y=None, gamma=None):
    """Matches ``sklearn.metrics.pairwise.rbf_kernel``."""
    return _kernel(X, Y, "rbf", gamma)


def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    """Matches ``sklearn.metrics.pairwise.sigmoid_kernel``."""
    return _kernel(X, Y, "sigmoid", gamma, coef0=coef0)


def laplacian_kernel(X, Y=None, gamma=None):
    """Matches ``sklearn.metrics.pairwise.laplacian_kernel``."""
    return _kernel(X, Y, "laplacian", gamma)


def cosine_similarity(X, Y=None, dense_output=True):
    """Matches ``sklearn.metrics.pairwise.cosine_similarity``."""
    return _kernel(X, Y, "cosine")


def _check_nonneg(X, Y):
    if np.any(np.asarray(X) < 0):
        raise ValueError("X contains negative values.")
    if Y is not None and np.any(np.asarray(Y) < 0):
        raise ValueError("Y contains negative values.")


def additive_chi2_kernel(X, Y=None):
    """Matches ``sklearn.metrics.pairwise.additive_chi2_kernel``."""
    _check_nonneg(X, Y)
    return _kernel(X, Y, "additive_chi2")


def chi2_kernel(X, Y=None, gamma=1.0):
    """Matches ``sklearn.metrics.pairwise.chi2_kernel``."""
    _check_nonneg(X, Y)
    return _kernel(X, Y, "chi2", gamma)


PAIRWISE_DISTANCE_FUNCTIONS = {
    "cityblock": manhattan_distances,
    "cosine": cosine_distances,
    "euclidean": euclidean_distances,
    "haversine": haversine_distances,
    "l2": euclidean_distances,
    "l1": manhattan_distances,
    "manhattan": manhattan_distances,
    "precomputed": None,
    "nan_euclidean": nan_euclidean_distances,
}

PAIRWISE_KERNEL_FUNCTIONS = {
    "additive_chi2": additive_chi2_kernel,
    "chi2": chi2_kernel,
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "poly": polynomial_kernel,
    "rbf": rbf_kernel,
    "laplacian": laplacian_kernel,
    "sigmoid": sigmoid_kernel,
    "cosine": cosine_similarity,
}


def distance_metrics():
    """Matches ``sklearn.metrics.pairwise.distance_metrics``."""
    return PAIRWISE_DISTANCE_FUNCTIONS


def kernel_metrics():
    """Matches ``sklearn.metrics.pairwise.kernel_metrics``."""
    return PAIRWISE_KERNEL_FUNCTIONS


# --------------------------------------------------------------------------- #
# the dispatchers
# --------------------------------------------------------------------------- #


def pairwise_distances(X, Y=None, metric="euclidean", *, n_jobs=None, ensure_all_finite=True, **kwds):
    """Matches ``sklearn.metrics.pairwise_distances`` for every metric in
    sklearn's ``_VALID_METRICS`` (bar the removed ``wminkowski``),
    ``'precomputed'``, and a callable. ``n_jobs`` is accepted; the Rust engine
    always uses every core."""
    if metric == "precomputed":
        Xa, _ = check_pairwise_arrays(X, Y, precomputed=True, ensure_all_finite=ensure_all_finite)
        if np.any(Xa < 0):
            raise ValueError("Negative values in data passed to precomputed distance matrix.")
        return Xa
    if callable(metric):
        Xa, Ya = check_pairwise_arrays(X, Y, ensure_all_finite=ensure_all_finite)
        return _pairwise_callable(Xa, None if Y is None else Ya, metric, **kwds)
    if metric == "nan_euclidean":
        return nan_euclidean_distances(X, Y, **kwds)
    if metric in ("euclidean", "l2"):
        return euclidean_distances(X, Y, **{k: kwds[k] for k in ("squared",) if k in kwds})
    return _distances(X, Y, metric, kwds)


def pairwise_kernels(X, Y=None, metric="linear", *, filter_params=False, n_jobs=None, **kwds):
    """Matches ``sklearn.metrics.pairwise_kernels`` (every named kernel,
    ``'precomputed'`` and a callable)."""
    if metric == "precomputed":
        Xa, _ = check_pairwise_arrays(X, Y, precomputed=True)
        return Xa
    if callable(metric):
        Xa, Ya = check_pairwise_arrays(X, Y)
        return _pairwise_callable(Xa, None if Y is None else Ya, metric, **kwds)
    if metric not in PAIRWISE_KERNEL_FUNCTIONS:
        raise ValueError("Unknown kernel %r" % metric)
    if filter_params:
        kwds = {k: kwds[k] for k in kwds if k in _KERNEL_PARAMS[metric]}
    return PAIRWISE_KERNEL_FUNCTIONS[metric](X, Y, **kwds)


def _working_rows(n_y, n_x, working_memory):
    if working_memory is None:
        from sklearn import get_config

        working_memory = get_config()["working_memory"]
    row_bytes = 8 * max(n_y, 1)
    rows = int(working_memory * (2**20) // row_bytes)
    if rows < 1:
        warnings.warn(
            "Could not adhere to working_memory config. Currently %.0fMiB, %.0fMiB required."
            % (working_memory, np.ceil(row_bytes * 2**-20))
        )
        rows = 1
    return min(rows, n_x)


def pairwise_distances_chunked(
    X, Y=None, *, reduce_func=None, metric="euclidean", n_jobs=None, working_memory=None, **kwds
):
    """Matches ``sklearn.metrics.pairwise_distances_chunked``: a generator of
    row blocks of the distance matrix (each optionally reduced by
    ``reduce_func(D_chunk, start)``), sized to ``working_memory`` MiB."""
    Xa = np.asarray(X)
    n_x = Xa.shape[0]
    n_y = n_x if Y is None else np.asarray(Y).shape[0]
    chunk = _working_rows(n_y, n_x, working_memory)
    Yref = Xa if Y is None else Y
    if not callable(metric) and metric in ("seuclidean", "mahalanobis") and Y is None:
        # Resolve V/VI against the FULL X once, as a single call would.
        name, p, w, v, vi = _metric_params(metric, Xa, None, kwds)
        kwds = dict(kwds, **({"V": v} if v is not None else {"VI": np.asarray(vi).reshape(Xa.shape[1], -1)}))
    for start in range(0, n_x, chunk):
        sl = slice(start, min(start + chunk, n_x))
        if metric == "precomputed":
            D = Xa[sl]
        else:
            D = pairwise_distances(Xa[sl], Yref, metric=metric, **kwds)
            if Y is None and metric in ("euclidean", "l2"):
                D.flat[sl.start :: n_x + 1] = 0
        if reduce_func is not None:
            out = reduce_func(D, sl.start)
            if out is None:
                raise TypeError("reduce_func must not return None")
            yield out
        else:
            yield D


def pairwise_distances_argmin_min(X, Y, *, axis=1, metric="euclidean", metric_kwargs=None):
    """Matches ``sklearn.metrics.pairwise_distances_argmin_min``: for each
    row of ``X`` (``axis=1``) the index of its closest ``Y`` row and the
    distance. Streams in blocks; the full matrix is never built."""
    if axis == 0:
        X, Y = Y, X
    kw = dict(metric_kwargs or {})
    if callable(metric):
        D = pairwise_distances(X, Y, metric=metric, **kw)
        idx = np.argmin(D, axis=1)
        return idx, D[np.arange(D.shape[0]), idx]
    Xa, Ya, _, _ = _prepare(X, Y)
    if metric == "euclidean" and kw.get("squared"):
        metric = "sqeuclidean"
    name, p, w, v, vi = _metric_params(metric if metric != "l2" else "euclidean", Xa, Ya, kw)
    idx, val = _ext().argmin_min(_flat(Xa), _flat(Ya), Xa.shape[1], name, p, w, v, vi)
    return _owned(idx, dtype=np.intp), _owned(val)


def pairwise_distances_argmin(X, Y, *, axis=1, metric="euclidean", metric_kwargs=None):
    """Matches ``sklearn.metrics.pairwise_distances_argmin``."""
    return pairwise_distances_argmin_min(X, Y, axis=axis, metric=metric, metric_kwargs=metric_kwargs)[0]


# --------------------------------------------------------------------------- #
# paired
# --------------------------------------------------------------------------- #


def _paired(X, Y, metric, p=None):
    Xa, Ya = check_paired_arrays(X, Y)
    out = _ext().paired_distances(_flat(Xa), _flat(Ya), Xa.shape[1], metric, p)
    return _owned(out).astype(_float_dtype(Xa, Ya), copy=False)


def paired_euclidean_distances(X, Y):
    """Matches ``sklearn.metrics.pairwise.paired_euclidean_distances``."""
    return _paired(X, Y, "euclidean")


def paired_manhattan_distances(X, Y):
    """Matches ``sklearn.metrics.pairwise.paired_manhattan_distances``."""
    return _paired(X, Y, "manhattan")


def paired_cosine_distances(X, Y):
    """Matches ``sklearn.metrics.pairwise.paired_cosine_distances``
    (``0.5·‖x̂ − ŷ‖²``)."""
    return _paired(X, Y, "cosine")


_PAIRED = {
    "cosine": paired_cosine_distances,
    "euclidean": paired_euclidean_distances,
    "l2": paired_euclidean_distances,
    "l1": paired_manhattan_distances,
    "manhattan": paired_manhattan_distances,
    "cityblock": paired_manhattan_distances,
}


def paired_distances(X, Y, *, metric="euclidean", **kwds):
    """Matches ``sklearn.metrics.pairwise.paired_distances``."""
    if metric in _PAIRED:
        return _PAIRED[metric](X, Y)
    if callable(metric):
        Xa, Ya = check_paired_arrays(X, Y)
        return np.array([metric(Xa[i], Ya[i]) for i in range(len(Xa))])
    raise ValueError("Unknown distance %s" % metric)
