"""Neighbors estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

NearestNeighbors (no scoring mixin — exposes ``kneighbors``, not ``predict``),
KNeighborsClassifier -> ``ClassifierMixin``, KNeighborsRegressor ->
``RegressorMixin``. sklearn-faithful ``__init__`` stores ``n_neighbors`` verbatim
(RESEARCH 06 §Hyperparameter Mapping). ``fit`` returns ``self``; the predict /
``kneighbors`` paths delegate to the matching ``_mlrs.Py*`` wrapper and wrap the
host output (D-03; neighbor indices are ``int32``, D-06).
"""

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from .base import MlrsBase

# The neighbor-search strategies this backend implements. mlrs is brute-force
# only (NEIGH-01); ``auto`` resolves to it, and sklearn's tree strategies are
# rejected rather than silently substituted.
_SUPPORTED_ALGORITHMS = ["auto", "brute"]


def _check_algorithm(algorithm):
    """Reject an unsupported ``algorithm`` with sklearn-shaped wording.

    Called from ``NearestNeighbors.fit`` and ``KNeighborsClassifier.fit``, which
    still accept only ``auto``/``brute``. ``KNeighborsRegressor`` validates its
    own wider set in ``_validate_params_for_fit`` (see
    ``_REGRESSOR_ALGORITHMS``) — bringing the other two up to that surface is
    follow-on work, not something this helper should straddle.
    """
    if algorithm not in _SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Algorithm is not supported: {algorithm}. "
            f"Supported algorithms are {_SUPPORTED_ALGORITHMS}"
        )


class NearestNeighbors(MlrsBase):
    """Brute-force k-NN search (NEIGH-01). Exposes ``kneighbors`` — no predict."""

    def __init__(self, n_neighbors=5, output_type="input", *, algorithm="auto"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.output_type = output_type

    def fit(self, X, y=None):
        _check_algorithm(self.algorithm)
        xa, rows, cols = self._normalize(X)
        obj = self._ext().NearestNeighbors(self.n_neighbors)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self.n_features_in_ = cols
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        self._check_fitted()
        if X is None:
            raise ValueError(
                "mlrs NearestNeighbors.kneighbors requires X (v1)"
            )
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        xa, rows, cols = self._check_predict_X(X)
        dist, idx = getattr(self._mlrs_obj, "kneighbors" + self._suffix())(
            xa, rows, cols, k
        )
        indices = self._to_output(idx, (rows, k), X, np.int32)
        if not return_distance:
            return indices
        distances = self._to_output(dist, (rows, k), X, self._np_float())
        return distances, indices


class KNeighborsClassifier(ClassifierMixin, MlrsBase):
    """k-NN classification by majority vote (NEIGH-02)."""

    def __init__(self, n_neighbors=5, output_type="input", *, algorithm="auto"):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.output_type = output_type

    def fit(self, X, y):
        _check_algorithm(self.algorithm)
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=self._x_float(xa))
        obj = self._ext().KNeighborsClassifier(self.n_neighbors)
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self.n_features_in_ = cols
        # classes_ are the core's DISTINCT sorted training labels, so a
        # non-contiguous target (e.g. {0, 2}) round-trips through predict (WR-01).
        self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    def predict_proba(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict_proba")(xa, rows, cols)
        n_classes = self._mlrs_obj.n_classes()
        return self._to_output(out, (rows, n_classes), X, self._np_float())

    @staticmethod
    def _x_float(xa):
        return np.float32 if xa.type.bit_width == 32 else np.float64


# ---------------------------------------------------------------------------
# KNeighborsRegressor parameter surface (KNN-REG-PARAMS)
# ---------------------------------------------------------------------------

# Every ``algorithm`` sklearn accepts. Unlike ``_SUPPORTED_ALGORITHMS`` above
# (which the two older shims still use), the regressor accepts the TREE names
# too and resolves them to the brute-force search.
#
# This is not a silent substitution of a different answer: ``algorithm`` selects
# how the neighbours are FOUND, never which ones they are, and sklearn's own
# ``'auto'`` already picks between the three by shape. A grid search over
# ``algorithm`` therefore gets identical predictions from mlrs for every value,
# which is what a drop-in replacement should do — rejecting ``'kd_tree'`` would
# fail a search that sklearn completes, over a parameter that cannot change the
# result.
_REGRESSOR_ALGORITHMS = ("auto", "ball_tree", "kd_tree", "brute")

# Metrics the device kernels implement, and the sklearn aliases that fold onto
# them. ``minkowski`` is handled separately because its resolution depends on
# ``p`` (see :func:`_resolve_metric`).
_SUPPORTED_METRICS = (
    "minkowski",
    "euclidean",
    "l2",
    "manhattan",
    "l1",
    "cityblock",
    "chebyshev",
    "infinity",
    "cosine",
)

# What sklearn's tree backends can actually index. ``cosine`` is not a metric
# either tree admits, so sklearn raises for that pair rather than falling back
# to brute; mirroring the rejection keeps an mlrs script's failure identical to
# sklearn's instead of quietly succeeding where sklearn does not.
_TREE_VALID_METRICS = (
    "minkowski",
    "euclidean",
    "l2",
    "manhattan",
    "l1",
    "cityblock",
    "chebyshev",
    "infinity",
)


def _resolve_metric(metric, p, metric_params):
    """sklearn's ``NeighborsBase._check_algorithm_metric`` + ``_fit`` resolution.

    Returns ``(effective_metric_, effective_metric_params_, effective_p)``.

    Reproduces sklearn's exact behaviour, including the parts that are easy to
    get subtly wrong:

      * a ``p`` inside ``metric_params`` OVERRIDES the ``__init__`` one and
        warns (``SyntaxWarning``) — sklearn does not merge them or prefer the
        constructor value;
      * ``minkowski`` collapses onto ``euclidean`` / ``manhattan`` /
        ``chebyshev`` at ``p = 2 / 1 / inf``, and ``p`` is then REMOVED from
        ``effective_metric_params_`` (a ``chebyshev`` with a stray ``p`` in its
        params would not compare equal to sklearn's);
      * when it does NOT collapse, ``effective_metric_params_`` carries BOTH
        ``p`` and ``w`` — sklearn puts the (usually ``None``) weight vector in
        there too, so a dict holding only ``p`` is not the dict sklearn
        produces.

    ``p < 1`` is rejected. sklearn's own parameter constraint allows ``p > 0``
    and only its brute+scipy path can evaluate ``0 < p < 1``; the device kernel
    follows the ``knn_graph`` contract (``p >= 1``, the range over which
    Minkowski is a metric at all), so this raises with sklearn's wording rather
    than computing a quasi-metric it was not asked for.

    A non-``None`` ``w`` (weighted Minkowski) is rejected outright: no device
    kernel takes a per-feature weight vector, and silently ignoring it would
    return unweighted distances under a weighted request.
    """
    params = {} if metric_params is None else dict(metric_params)

    if metric_params is not None and "p" in metric_params:
        import warnings

        warnings.warn(
            "Parameter p is found in metric_params. "
            "The corresponding parameter from __init__ is ignored.",
            SyntaxWarning,
            stacklevel=3,
        )

    effective_metric = metric
    effective_p = None

    if metric == "minkowski":
        effective_p = params.pop("p", p)
        w = params.pop("w", None)
        if effective_p is None or effective_p < 1:
            raise ValueError(
                "p must be greater or equal to one for minkowski metric"
            )
        if w is not None:
            raise ValueError(
                "weighted minkowski (metric_params['w']) is not supported"
            )
        if effective_p == 1:
            effective_metric = "manhattan"
        elif effective_p == 2:
            effective_metric = "euclidean"
        elif effective_p == float("inf"):
            effective_metric = "chebyshev"
        else:
            # Only the NON-collapsed case keeps them, and it keeps both.
            params["p"] = effective_p
            params["w"] = w

    return effective_metric, params, effective_p


def _get_weights(dist, weights):
    """sklearn ``neighbors._base._get_weights``.

    Returns the per-neighbour weight array, or ``None`` for the uniform case
    (which the callers read as "plain mean" rather than as "weights of 1", so
    the uniform path never allocates).

    The infinity handling is the subtle part and is sklearn's verbatim: a query
    that coincides with a training point gives ``1/0 = inf``, and normalising
    ``inf / inf`` is NaN. sklearn instead switches THAT ROW to an indicator
    weighting — the coincident neighbours get 1, everything else in the row gets
    0 — so the prediction becomes the mean of the coincident targets. The device
    kernel implements the same rule (``knn_regress_gather``); this host copy
    exists for the callable-weights / callable-metric paths.
    """
    if weights == "uniform":
        return None
    if weights == "distance":
        with np.errstate(divide="ignore"):
            dist = 1.0 / dist
        inf_mask = np.isinf(dist)
        inf_row = np.any(inf_mask, axis=1)
        dist[inf_row] = inf_mask[inf_row]
        return dist
    return weights(dist)


def _weighted_mean(y_neighbors, weights):
    """Average ``y_neighbors`` (``n_query × k [× n_outputs]``) over its `k` axis.

    ``weights=None`` is the plain mean. Shared by every host-side prediction
    path so the single-output and multi-output cases cannot diverge.
    """
    if weights is None:
        return np.mean(y_neighbors, axis=1)
    denom = np.sum(weights, axis=1)
    if y_neighbors.ndim == 3:
        num = np.sum(y_neighbors * weights[:, :, None], axis=1)
        return num / denom[:, None]
    return np.sum(y_neighbors * weights, axis=1) / denom


class KNeighborsRegressor(RegressorMixin, MlrsBase):
    """k-NN regression, matching sklearn's full parameter surface (NEIGH-03).

    ``KNeighborsRegressor(n_neighbors=5, output_type='input', *,
    weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
    metric_params=None, n_jobs=None)``.

    ``output_type`` keeps its mlrs-specific second-positional slot (shared with
    every sibling shim); everything after it mirrors sklearn's keyword-only
    signature verbatim, stored unvalidated in ``__init__`` per the sklearn
    contract (validation happens in ``fit``).

    Which parameters reach the device, and which stop here:

    ==================  ==========================================================
    ``n_neighbors``     device — the selection width
    ``weights``         device for ``'uniform'`` / ``'distance'``; a CALLABLE is
                        applied host-side to ``kneighbors`` distances
    ``metric`` / ``p``  device for the built-in metrics; a CALLABLE runs the
                        whole pairwise pass host-side
    ``metric_params``   resolution input only (its ``p`` overrides ``__init__``)
    ``algorithm``       accepted and resolved to brute force (see
                        ``_REGRESSOR_ALGORITHMS``) — cannot change the result
    ``leaf_size``       validated, then unused: it tunes a tree mlrs has no
                        equivalent of, and it cannot change the result either
    ``n_jobs``          validated, then unused: parallelism here is the device's,
                        not a host thread pool
    ==================  ==========================================================

    Multi-output targets are supported: a 2-D ``y`` (``n_samples × n_outputs``)
    makes ``predict`` return ``n_samples × n_outputs``.

    ``fit`` validates and stores; the device upload happens on the first query.
    Brute-force k-NN has no model to solve for, so that is where the work
    genuinely belongs — see ``PyKNeighborsRegressor`` in
    ``crates/mlrs-py/src/estimators/neighbors.rs`` for the measurement. Nothing
    observable changes: a non-finite training set still raises from ``fit``, and
    every fitted attribute is answerable without querying.
    """

    def __init__(
        self,
        n_neighbors=5,
        output_type="input",
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.output_type = output_type
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    # -- fit ---------------------------------------------------------------- #

    def _validate_params_for_fit(self):
        """Validate every constructor argument, sklearn-shaped, before any work.

        Kept out of ``__init__`` because sklearn requires constructors to store
        arguments verbatim and defer validation to ``fit``
        (``check_no_attributes_set_in_init`` /
        ``check_parameters_default_constructible``).
        """
        if self.algorithm not in _REGRESSOR_ALGORITHMS:
            raise ValueError(
                f"Algorithm is not supported: {self.algorithm}. "
                f"Supported algorithms are {list(_REGRESSOR_ALGORITHMS)}"
            )
        if not callable(self.weights) and self.weights not in (
            "uniform",
            "distance",
        ):
            raise ValueError(
                "weights not recognized: should be 'uniform', 'distance', "
                "or a callable function"
            )
        if not callable(self.metric) and self.metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Metric is not supported: {self.metric}. "
                f"Supported metrics are {list(_SUPPORTED_METRICS)}"
            )
        if not isinstance(self.leaf_size, (int, np.integer)) or self.leaf_size < 1:
            raise ValueError(
                f"leaf_size == {self.leaf_size}, must be >= 1."
            )
        if not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1:
            raise ValueError(
                f"n_neighbors == {self.n_neighbors}, must be >= 1."
            )
        if self.n_jobs is not None and not isinstance(self.n_jobs, (int, np.integer)):
            raise ValueError(f"n_jobs == {self.n_jobs}, must be an integer or None.")
        # A tree ALGORITHM restricts the metric even though mlrs runs brute
        # force for all of them: sklearn raises for the pair, so mlrs must too
        # or a script that sklearn rejects would silently succeed here.
        if self.algorithm in ("kd_tree", "ball_tree") and not callable(self.metric):
            if self.metric not in _TREE_VALID_METRICS:
                raise ValueError(
                    f"Metric '{self.metric}' not valid for algorithm "
                    f"'{self.algorithm}'"
                )

    def fit(self, X, y):
        self._validate_params_for_fit()
        effective_metric, effective_params, effective_p = _resolve_metric(
            self.metric if not callable(self.metric) else "euclidean",
            self.p,
            self.metric_params,
        )
        if callable(self.metric):
            # A Python callable cannot cross into a device kernel, so this path
            # is host-side end to end. sklearn reports the callable itself as
            # ``effective_metric_``.
            effective_metric = self.metric
            effective_params = (
                {} if self.metric_params is None else dict(self.metric_params)
            )

        # `ensure_all_finite=False` RELOCATES the NaN/inf rejection into the
        # Rust `fit`, it does not drop it: that call scans both operands itself
        # and raises `check_array`'s exact ValueError. Brute-force k-NN has no
        # model to solve for, so validation IS essentially all `fit` does — a
        # second full scan here would be a large fraction of it, and the Rust
        # one is the faster of the two.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = KNeighborsClassifier._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype, ensure_all_finite=False)

        # A callable weighting is applied to `kneighbors` distances host-side,
        # so the device object is built with the (unused) uniform weighting;
        # `predict` never asks it for a prediction in that case.
        device_weights = "uniform" if callable(self.weights) else self.weights
        # `p` is only meaningful to the device for a true Minkowski exponent;
        # every other metric ignores it. Pass 2.0 rather than a possibly-`None`
        # `self.p` so the Rust `minkowski` collapse is fed a valid number.
        device_p = float(effective_p) if effective_p is not None else 2.0
        obj = self._ext().KNeighborsRegressor(
            self.n_neighbors,
            device_weights,
            effective_metric if not callable(self.metric) else "euclidean",
            device_p,
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj

        self.n_features_in_ = cols
        self.n_samples_fit_ = rows
        self.effective_metric_ = effective_metric
        self.effective_metric_params_ = effective_params
        # sklearn keeps the training data on the estimator under `_fit_X` / `_y`
        # and several of its utilities read them; they also back this shim's own
        # host paths (callable metric, callable weights, `kneighbors(X=None)`).
        #
        # Only the ARROW buffers are retained here — the numpy views are built
        # on first access (see `_fit_X` / `_y` below). Materializing them eagerly
        # would put a second reshape+dtype pass over the whole training matrix on
        # every `fit`, and the common path (a built-in metric and weighting,
        # `predict` on new data) never reads either one.
        self._fit_arrow = (xa, ya, rows, cols, dtype)
        return self

    @property
    def _fit_X(self):
        """The fitted training matrix as `(n_samples_fit_, n_features_in_)` numpy.

        Materialized on first access from the arrow buffer `fit` already built —
        which is a zero-copy view for the dense float arrays that reach here, so
        this costs a reshape rather than a copy. Sourced from the NORMALIZED
        buffer rather than the caller's `X`, so it is by construction the exact
        matrix the device was given, without re-running `check_array` and the
        dtype coercion for every accepted container (list, pyarrow, DataFrame).
        """
        self._check_fitted()
        xa, _, rows, cols, dtype = self._fit_arrow
        return np.asarray(
            xa.to_numpy(zero_copy_only=False), dtype=dtype
        ).reshape(rows, cols)

    @property
    def _y(self):
        """The fitted targets: `(n_samples_fit_,)`, or 2-D when multi-output."""
        self._check_fitted()
        _, ya, rows, _, dtype = self._fit_arrow
        arr = np.asarray(ya.to_numpy(zero_copy_only=False), dtype=dtype)
        n_outputs = self._mlrs_obj.n_outputs()
        return arr.reshape(rows) if n_outputs == 1 else arr.reshape(rows, n_outputs)

    # -- neighbour queries -------------------------------------------------- #

    def _host_kneighbors(self, Xq, k):
        """Brute-force `kneighbors` under a CALLABLE metric, host-side.

        `n_query × n_train` Python-level metric calls — slow by construction,
        and no slower than sklearn's own callable-metric path, which also
        evaluates the callable once per pair. The selection uses a STABLE sort
        so the tie-break is lowest-index, matching the device `top_k` contract
        (a query equidistant from two training points must resolve to the same
        neighbour on both paths, or `predict` would disagree with itself across
        metrics).
        """
        train = self._fit_X
        d = np.empty((Xq.shape[0], train.shape[0]), dtype=np.float64)
        for i in range(Xq.shape[0]):
            for j in range(train.shape[0]):
                d[i, j] = self.metric(Xq[i], train[j])
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(d, order, axis=1), order.astype(np.int32)

    def _raw_kneighbors(self, X, k):
        """`(distances, indices)` as plain 2-D numpy — the INTERNAL query form.

        Every in-shim consumer (`predict`'s callable paths, `kneighbors_graph`,
        and `kneighbors` itself) goes through this rather than through the
        public `kneighbors`, so none of them has to un-wrap whatever container
        `output_type` chose. Routing a result out to pyarrow only to immediately
        `np.asarray` it back is both wasteful and lossy for the 2-D shapes here.
        Returns `(rows, cols)` alongside so the caller can size its own egress.
        """
        xa, rows, cols = self._check_predict_X(X)
        if callable(self.metric):
            query = np.ascontiguousarray(
                np.asarray(xa.to_numpy(zero_copy_only=False), dtype=self._np_float())
                .reshape(rows, cols)
            )
            dist, idx = self._host_kneighbors(query, k)
        else:
            flat_d, flat_i = getattr(
                self._mlrs_obj, "kneighbors" + self._suffix()
            )(xa, rows, cols, k)
            dist = np.asarray(flat_d, dtype=self._np_float()).reshape(rows, k)
            idx = np.asarray(flat_i, dtype=np.int32).reshape(rows, k)
        return dist, idx, rows

    def _resolve_k(self, n_neighbors, query_is_train):
        """Validate the effective `k` and the width the search must run at.

        `X=None` needs `k+1` internally so the self-match can be dropped, and it
        is the `k+1` — not `k` — that has to fit the training set.
        """
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"Expected n_neighbors > 0. Got {k}")
        k_query = k + 1 if query_is_train else k
        if k_query > self.n_samples_fit_:
            raise ValueError(
                f"Expected n_neighbors <= n_samples_fit, but "
                f"n_neighbors = {k}, n_samples_fit = {self.n_samples_fit_}"
            )
        return k, k_query

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """The `k` nearest training points of each row of `X`.

        `X=None` queries the TRAINING set with each point's own self-match
        removed, exactly as sklearn does: the search runs with `k+1` and the
        self column is dropped by INDEX IDENTITY, not by "first column" or
        "distance 0" — duplicated training points sit at distance 0 too, and
        dropping one of those instead of the true self would return a
        neighbour set sklearn does not.
        """
        self._check_fitted()
        query_is_train = X is None
        k, k_query = self._resolve_k(n_neighbors, query_is_train)
        query = self._fit_X if query_is_train else X

        dist, idx, rows = self._raw_kneighbors(query, k_query)
        if query_is_train:
            dist, idx = _drop_self_column(dist, idx, k)

        indices = self._to_output(idx.ravel(), (rows, k), query, np.int32)
        if not return_distance:
            return indices
        distances = self._to_output(
            dist.ravel(), (rows, k), query, self._np_float()
        )
        return distances, indices

    def kneighbors_graph(self, X=None, n_neighbors=None, mode="connectivity"):
        """The `(n_query, n_samples_fit_)` sparse neighbour graph.

        `mode='connectivity'` stores 1 at each neighbour, `mode='distance'`
        stores the distance. Returns a scipy CSR matrix, as sklearn does — this
        is the one method on the shim whose output type is NOT routed through
        `output_type`, because sklearn's callers (e.g. `SpectralClustering`,
        `Isomap`) require the sparse matrix specifically.
        """
        from scipy.sparse import csr_matrix

        self._check_fitted()
        if mode not in ("connectivity", "distance"):
            raise ValueError(
                f'Unsupported mode, must be one of "connectivity" or '
                f'"distance" but got "{mode}" instead'
            )
        query_is_train = X is None
        k, k_query = self._resolve_k(n_neighbors, query_is_train)
        query = self._fit_X if query_is_train else X

        dist, idx, n_query = self._raw_kneighbors(query, k_query)
        if query_is_train:
            dist, idx = _drop_self_column(dist, idx, k)

        data = (
            np.ones(n_query * k, dtype=np.float64)
            if mode == "connectivity"
            else np.asarray(dist, dtype=np.float64).ravel()
        )
        indptr = np.arange(0, n_query * k + 1, k)
        return csr_matrix(
            (data, idx.ravel(), indptr), shape=(n_query, self.n_samples_fit_)
        )

    # -- predict ------------------------------------------------------------ #

    def predict(self, X):
        """The weighted mean of each query's `k` neighbour targets.

        Returns `(n_samples,)` for a 1-D fitted target and
        `(n_samples, n_outputs)` for a 2-D one.
        """
        xa, rows, cols = self._check_predict_X(X)
        n_outputs = self._mlrs_obj.n_outputs()
        shape = (rows,) if n_outputs == 1 else (rows, n_outputs)

        if callable(self.weights) or callable(self.metric):
            # Host path: the device cannot evaluate a Python callable, so the
            # prediction is rebuilt from the (device- or host-computed)
            # neighbour set. It is the SAME neighbour set the device `predict`
            # would average — both go through `_raw_kneighbors` — so the two
            # paths agree wherever they overlap.
            dist, idx, _ = self._raw_kneighbors(X, self.n_neighbors)
            dist = np.asarray(dist, dtype=np.float64)
            idx = idx.astype(np.intp)
            w = _get_weights(dist, self.weights)
            out = _weighted_mean(self._y[idx], w)
            return self._to_output(
                np.ascontiguousarray(out).ravel(), shape, X, self._np_float()
            )

        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, shape, X, self._np_float())


def _drop_self_column(dist, idx, k):
    """Remove each query row's own index from a `k+1`-wide neighbour result.

    Used by `kneighbors(X=None)`. The column removed is the one whose neighbour
    INDEX equals the row number, never simply the first or the nearest: with
    duplicated training points several columns sit at distance 0, and dropping
    the wrong one returns a different neighbour set from sklearn's.

    If the self index is absent (which cannot happen for a genuine X-vs-X query,
    since a point is always at distance 0 from itself) the LAST column is
    dropped, so the result is still `k` wide.
    """
    rows = dist.shape[0]
    keep = np.ones(dist.shape, dtype=bool)
    self_col = np.argmax(idx == np.arange(rows)[:, None], axis=1)
    has_self = np.any(idx == np.arange(rows)[:, None], axis=1)
    self_col = np.where(has_self, self_col, dist.shape[1] - 1)
    keep[np.arange(rows), self_col] = False
    return (
        dist[keep].reshape(rows, k),
        idx[keep].reshape(rows, k),
    )
