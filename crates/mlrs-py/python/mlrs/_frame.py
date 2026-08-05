"""Container-native COLUMN operations for the mlrs shim (D-03, FSEL-01).

:mod:`mlrs._io` handles the ROW-major float ingress every estimator needs and the
flat-buffer egress most of them want. This module handles the other axis: taking a
subset of a container's COLUMNS and giving back the same kind of container, with
its column names and per-column dtypes intact.

That is what a feature selector's ``transform`` is, and it is why the operation
belongs here rather than in the compiled extension. ``mlrs.feature_selection``'s
Rust side returns a support MASK — the mask is the model — and this module applies
it with each container's own gather:

===========================  =========================================
input                        gather used
===========================  =========================================
``numpy.ndarray``            ``X[:, mask]``
``pandas`` DataFrame         ``X.iloc[:, indices]`` (POSITIONAL)
``polars`` DataFrame         ``X.select(selected_names)``
``pyarrow`` Table            ``X.select(selected_names)``
``python`` list of rows      a list comprehension
===========================  =========================================

so a polars frame in gives a polars frame out with only the kept columns, and a
pandas frame keeps its index and its mixed dtypes. Round-tripping through the
flat ``float64`` buffer the scores are computed on would lose the names, flatten
every dtype to one float, and drop a pandas index — which is exactly what
``sklearn``'s own selectors do by default (their ``transform`` returns numpy
unless you opt into ``set_output(transform="pandas")``), and is the behaviour
``output_type="numpy"`` still gives here.

## Detection is by DUCK TYPING and ``sys.modules``, never by import

``pandas``/``polars``/``pyarrow`` are detected exactly as
:mod:`mlrs.model_selection` detects them: by looking for an ALREADY-IMPORTED
module in ``sys.modules`` and, for pandas, by probing for ``.iloc``. This module
therefore never imports polars or pandas itself — a user who has neither
installed pays nothing, and a pandas-API frame that is not a pandas instance
(modin, cuDF) is still recognised through ``.iloc``. ``numpy`` is imported
unconditionally, as it is everywhere else in the package.
"""

import sys

import numpy as np


def _is_pandas(x):
    """``x`` is a pandas DataFrame (pandas already imported).

    Duck-typed on ``.iloc`` FIRST, which is sklearn's own pandas probe and what
    admits pandas-API frames that are not pandas instances. The ``sys.modules``
    check is the fallback for a frame without ``.iloc``.
    """
    if hasattr(x, "iloc") and hasattr(x, "columns"):
        return True
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(x, pd.DataFrame)


def _is_polars(x):
    """``x`` is a polars DataFrame (polars already imported)."""
    pl = sys.modules.get("polars")
    return pl is not None and isinstance(x, pl.DataFrame)


def _is_arrow_table(x):
    """``x`` is a pyarrow Table (pyarrow already imported)."""
    pa = sys.modules.get("pyarrow")
    return pa is not None and isinstance(x, pa.Table)


def column_names(X):
    """The container's column names, or ``None`` for a nameless container.

    This is what feeds sklearn's ``feature_names_in_`` fitted attribute and
    ``get_feature_names_out()``. ``None`` for numpy / lists, which have no names —
    sklearn sets ``feature_names_in_`` only "when `X` has feature names that are
    all strings", and returns generated ``x0, x1, …`` names otherwise.
    """
    if _is_pandas(X):
        return [str(c) for c in X.columns]
    if _is_polars(X) or _is_arrow_table(X):
        return list(X.columns) if _is_polars(X) else list(X.column_names)
    return None


def take_columns(X, mask):
    """Return ``X`` reduced to the columns where ``mask`` is true, SAME container.

    ``mask`` is a length-``n_features`` boolean sequence. An ALL-FALSE mask is
    legal and yields a container with zero columns and ``X``'s row count — which
    is what sklearn does (it warns, "No features were selected", and returns an
    ``n × 0`` array) rather than raising.

    The pandas gather is ``iloc``, i.e. POSITIONAL: the mask indexes column
    POSITIONS, not label values, and a frame whose column labels happen to be
    integers must not be reindexed by them.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.nonzero(mask)[0]

    if _is_pandas(X):
        return X.iloc[:, idx]
    if _is_polars(X):
        return X.select([X.columns[i] for i in idx])
    if _is_arrow_table(X):
        return X.select([X.column_names[i] for i in idx])
    if isinstance(X, np.ndarray):
        return X[:, mask]
    if isinstance(X, (list, tuple)):
        # A list of rows. Rebuilt as a list of lists so the result is the same
        # SHAPE of container that went in; a list input is not a typed container
        # with dtypes to preserve, so there is nothing subtler to do.
        return [[row[i] for i in idx] for row in X]
    # Anything else (a scipy sparse matrix, a memoryview, an unknown array-like)
    # goes through numpy, which is the narrowed set `_io.resolve_output_type`
    # documents. Deliberately not silently returning `X` unchanged.
    return np.asarray(X)[:, mask]


def restore_columns(Z, mask, names=None, mirror_container=True):
    """Inverse of :func:`take_columns`: widen ``Z`` back to ``len(mask)`` columns.

    The dropped columns come back as ZEROS, because that is what
    ``SelectorMixin.inverse_transform`` is defined to do — a selector discards
    information, so the inverse restores the geometry but not the values.

    ``Z`` is the REDUCED container (a selector's own ``transform`` output), so it
    is what decides the OUTPUT container type when ``mirror_container`` is set:
    a polars ``Z`` gives a polars frame back. ``names`` are the ORIGINAL
    ``n_features_in_`` column labels — the fitted ``feature_names_in_`` — which is
    the only place the dropped columns' names can come from, since ``Z`` no longer
    has them. Without usable names the result is numpy, because inventing labels
    for restored columns would be worse than returning an unlabelled array.
    """
    mask = np.asarray(mask, dtype=bool)
    idx = np.nonzero(mask)[0]
    z = np.asarray(to_numpy_2d(Z))
    out = np.zeros((z.shape[0], mask.shape[0]), dtype=z.dtype)
    out[:, idx] = z

    if not mirror_container or names is None or len(names) != mask.shape[0]:
        return out
    columns = {str(n): out[:, i] for i, n in enumerate(names)}
    if _is_polars(Z):
        return sys.modules["polars"].DataFrame(columns)
    if _is_arrow_table(Z):
        return sys.modules["pyarrow"].table(columns)
    if _is_pandas(Z):
        pd = sys.modules["pandas"]
        return pd.DataFrame(out, columns=list(names), index=getattr(Z, "index", None))
    return out


def to_numpy_2d(X):
    """A 2-D numpy view of any supported container, without validating it.

    Deliberately NOT ``check_array``: this is used by
    :func:`restore_columns`, whose caller has already validated, and a second
    validation pass would reject the NaNs ``VarianceThreshold`` legitimately
    carries.
    """
    if _is_polars(X) or _is_arrow_table(X):
        arr = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X)
    elif hasattr(X, "to_numpy"):
        arr = np.asarray(X.to_numpy())
    else:
        arr = np.asarray(X)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def feature_names_out(mask, names_in):
    """``get_feature_names_out()`` — the kept columns' names.

    Uses the fitted ``feature_names_in_`` when there was one, and sklearn's
    generated ``x0, x1, …`` positional names otherwise, which is what
    ``_check_feature_names_in`` does for a nameless input.
    """
    mask = np.asarray(mask, dtype=bool)
    if names_in is None:
        names_in = [f"x{i}" for i in range(mask.shape[0])]
    return np.asarray(
        [n for n, keep in zip(names_in, mask) if keep], dtype=object
    )
