"""Neighbors estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

NearestNeighbors (no scoring mixin — exposes ``kneighbors`` /
``radius_neighbors``, not ``predict``), KNeighborsClassifier ->
``ClassifierMixin``, KNeighborsRegressor -> ``RegressorMixin``. ``fit`` returns
``self``; the predict / ``kneighbors`` paths delegate to the matching
``_mlrs.Py*`` wrapper and wrap the host output (D-03; neighbor indices are
``int32``, D-06).

All three estimators carry sklearn's FULL parameter surface (NEIGH-PARAMS /
KNN-REG-PARAMS / KNN-CLF-PARAMS) and share it through :class:`_NeighborsBase`
— one copy of the validation, the metric resolution, and the ``kneighbors`` /
``kneighbors_graph`` / ``radius_neighbors`` / ``radius_neighbors_graph``
surface, so the three cannot drift on which parameter values they accept or on
which neighbours they report. ``NearestNeighbors`` additionally carries
``radius`` (sklearn's radius-query default) — the only hyperparameter the two
k-neighbours estimators do not have, since neither implements a
radius-neighbours predict.
"""

import warnings

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.multiclass import check_classification_targets

from .base import MlrsBase

# ---------------------------------------------------------------------------
# Shared neighbours parameter surface (NEIGH-PARAMS / KNN-REG-PARAMS /
# KNN-CLF-PARAMS)
# ---------------------------------------------------------------------------

# Every ``algorithm`` sklearn accepts, for all three estimators: they accept
# the TREE names too and resolve them all to the brute-force search.
#
# This is not a silent substitution of a different answer: ``algorithm`` selects
# how the neighbours are FOUND, never which ones they are, and sklearn's own
# ``'auto'`` already picks between the three by shape. A grid search over
# ``algorithm`` therefore gets identical predictions from mlrs for every value,
# which is what a drop-in replacement should do — rejecting ``'kd_tree'`` would
# fail a search that sklearn completes, over a parameter that cannot change the
# result.
_NEIGHBORS_ALGORITHMS = ("auto", "ball_tree", "kd_tree", "brute")

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

# ``sklearn.neighbors.VALID_METRICS``, restricted to the metrics this shim
# supports. Every value of ``algorithm`` admits a DIFFERENT metric set, and the
# two exclusions are not symmetric:
#
#   * ``cosine`` is a brute-force-only metric — neither tree can index it;
#   * ``infinity`` is a TREE-only spelling of Chebyshev — sklearn's brute path
#     goes through ``pairwise_distances``, which has never known that name.
#
# mlrs runs brute force for every ``algorithm``, so it could compute all nine in
# every case. It rejects the same pairs sklearn does anyway: a script that
# sklearn refuses must not quietly succeed here, or a user validating against
# sklearn would find the failure only after switching back.
_ALGORITHM_VALID_METRICS = {
    "brute": (
        "minkowski",
        "euclidean",
        "l2",
        "manhattan",
        "l1",
        "cityblock",
        "chebyshev",
        "cosine",
    ),
    "kd_tree": (
        "minkowski",
        "euclidean",
        "l2",
        "manhattan",
        "l1",
        "cityblock",
        "chebyshev",
        "infinity",
    ),
}
# Ball tree admits everything the k-d tree does, out of this shim's nine.
_ALGORITHM_VALID_METRICS["ball_tree"] = _ALGORITHM_VALID_METRICS["kd_tree"]
# ``algorithm='auto'`` is not a metric restriction at all: sklearn checks the
# metric against the BALL TREE's set when it is in it and against brute's
# otherwise, and every one of the nine is in one or the other. So auto accepts
# all nine — which is also why it is the only algorithm under which `'cosine'`
# and `'infinity'` can both be used.
_ALGORITHM_VALID_METRICS["auto"] = _SUPPORTED_METRICS


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
    0 — so the prediction becomes the mean (or the vote) of the coincident
    targets only. The device regressor kernel and the Rust classifier vote both
    implement the same rule; this host copy exists for the callable-weights /
    callable-metric paths.
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


class _NeighborsBase(MlrsBase):
    """The parameter surface + neighbour queries shared by all three estimators.

    Every subclass takes sklearn's keyword-only tail verbatim
    (``[weights]``/``algorithm``/``leaf_size``/``p``/``metric``/
    ``metric_params``/``n_jobs``, ``weights`` only on the two k-neighbours
    estimators) after mlrs's own second-positional ``output_type``, stores every
    argument unvalidated in ``__init__`` per the sklearn contract, and validates
    in ``fit`` through :meth:`_validate_params_for_fit`.

    Which parameters reach the device, and which stop here:

    ==================  ==========================================================
    ``n_neighbors``     device — the selection width (``kneighbors`` path)
    ``radius``          device — the selection threshold (``radius_neighbors``
                        path; ``NearestNeighbors`` only)
    ``weights``         device for ``'uniform'`` / ``'distance'``; a CALLABLE is
                        applied host-side to ``kneighbors`` distances (the two
                        k-neighbours estimators only — ``NearestNeighbors`` has
                        no vote/mean to weight)
    ``metric`` / ``p``  device for the built-in metrics; a CALLABLE runs the
                        whole pairwise pass host-side
    ``metric_params``   resolution input only (its ``p`` overrides ``__init__``)
    ``algorithm``       accepted and resolved to brute force (see
                        ``_NEIGHBORS_ALGORITHMS``) — cannot change the result,
                        but DOES restrict which ``metric`` values are legal,
                        exactly as sklearn's does
                        (see ``_ALGORITHM_VALID_METRICS``)
    ``leaf_size``       validated, then unused: it tunes a tree mlrs has no
                        equivalent of, and it cannot change the result either
    ``n_jobs``          validated, then unused: parallelism here is the device's,
                        not a host thread pool
    ==================  ==========================================================

    ``fit`` validates and stores; the device upload happens on the first query.
    Brute-force k-NN has no model to solve for, so that is where the work
    genuinely belongs — see ``PyKNeighborsRegressor`` in
    ``crates/mlrs-py/src/estimators/neighbors.rs`` for the measurement. Nothing
    observable changes: a non-finite training set still raises from ``fit``, and
    every fitted attribute is answerable without querying.
    """

    # -- validation --------------------------------------------------------- #

    def _validate_params_for_fit(self):
        """Validate every constructor argument, sklearn-shaped, before any work.

        Kept out of ``__init__`` because sklearn requires constructors to store
        arguments verbatim and defer validation to ``fit``
        (``check_no_attributes_set_in_init`` /
        ``check_parameters_default_constructible``).

        The ``weights`` check is gated on ``hasattr`` rather than living on a
        subclass override: ``NearestNeighbors`` has no ``weights`` attribute at
        all (there is no vote/mean to weight), and every other check here is
        identical across all three estimators — a subclass override would only
        exist to reorder this one line, at the cost of two nearly-identical
        copies of the rest of the method.
        """
        if self.algorithm not in _NEIGHBORS_ALGORITHMS:
            raise ValueError(
                f"Algorithm is not supported: {self.algorithm}. "
                f"Supported algorithms are {list(_NEIGHBORS_ALGORITHMS)}"
            )
        if hasattr(self, "weights") and not callable(self.weights) and self.weights not in (
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
        # The ALGORITHM restricts the metric even though mlrs runs brute force
        # for all of them (see `_ALGORITHM_VALID_METRICS`). A callable is exempt:
        # sklearn accepts one for every algorithm.
        if not callable(self.metric):
            if self.metric not in _ALGORITHM_VALID_METRICS[self.algorithm]:
                raise ValueError(
                    f"Metric '{self.metric}' not valid. Use "
                    f"sorted(sklearn.neighbors.VALID_METRICS['{self.algorithm}']) "
                    f"to get valid options. Metric can also be a callable "
                    f"function."
                )

    def _resolve_fit_metric(self):
        """``(effective_metric, effective_params, effective_p)`` for this ``fit``.

        A CALLABLE metric reports ITSELF as ``effective_metric_`` (as sklearn
        does) and carries ``metric_params`` through untouched; the resolution
        rules in :func:`_resolve_metric` only apply to the string spellings.
        """
        effective_metric, effective_params, effective_p = _resolve_metric(
            self.metric if not callable(self.metric) else "euclidean",
            self.p,
            self.metric_params,
        )
        if callable(self.metric):
            effective_metric = self.metric
            effective_params = (
                {} if self.metric_params is None else dict(self.metric_params)
            )
        return effective_metric, effective_params, effective_p

    def _device_metric_args(self, effective_metric, effective_p):
        """The ``(metric_name, p)`` pair handed to the ``_mlrs`` constructor.

        A callable metric is served entirely host-side, so the device object is
        built with the (then unused) Euclidean default rather than being asked
        for a name it cannot resolve. ``p`` is only meaningful for a genuine
        Minkowski exponent — every other metric ignores it — so a possibly-``None``
        ``effective_p`` becomes ``2.0`` rather than reaching Rust as a non-number.
        """
        name = "euclidean" if callable(self.metric) else effective_metric
        p = float(effective_p) if effective_p is not None else 2.0
        return name, p

    @staticmethod
    def _x_float(xa):
        """The numpy float dtype matching a normalized arrow ``X``."""
        return np.float32 if xa.type.bit_width == 32 else np.float64

    # -- fitted training data ------------------------------------------------ #

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
        xa, rows, cols, dtype = self._fit_arrow
        return np.asarray(
            xa.to_numpy(zero_copy_only=False), dtype=dtype
        ).reshape(rows, cols)

    # -- neighbour queries --------------------------------------------------- #

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

        Every in-shim consumer (`predict`'s callable and multi-output paths,
        `kneighbors_graph`, and `kneighbors` itself) goes through this rather
        than through the public `kneighbors`, so none of them has to un-wrap
        whatever container `output_type` chose. Routing a result out to pyarrow
        only to immediately `np.asarray` it back is both wasteful and lossy for
        the 2-D shapes here. Returns `(rows, cols)` alongside so the caller can
        size its own egress.
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

    # -- radius queries (NEIGH-RADIUS) --------------------------------------- #

    def _host_radius_neighbors(self, Xq, radius):
        """Brute-force `radius_neighbors` under a CALLABLE metric, host-side.

        `n_query × n_train` Python-level metric calls, mirroring
        `_host_kneighbors`. Matches are kept in ASCENDING TRAIN-INDEX order (a
        plain left-to-right column scan) rather than sorted by distance —
        the same unsorted convention the device path returns (see
        `RadiusNeighbors` in `crates/mlrs-algos/src/neighbors/nearest.rs`), so a
        callable metric cannot silently change the reported order.
        """
        train = self._fit_X
        dist_list, idx_list = [], []
        for i in range(Xq.shape[0]):
            d = np.array(
                [self.metric(Xq[i], train[j]) for j in range(train.shape[0])],
                dtype=np.float64,
            )
            mask = d <= radius
            idx_list.append(np.nonzero(mask)[0].astype(np.int32))
            dist_list.append(d[mask])
        return dist_list, idx_list

    def _raw_radius_neighbors(self, X, radius):
        """`(dist_list, idx_list, rows)` — the INTERNAL radius-query form.

        Each of `dist_list[i]` / `idx_list[i]` is a 1-D array of query row `i`'s
        matches (RAGGED — a different width per row), in ascending train-index
        order. `radius_neighbors` / `radius_neighbors_graph` both go through
        here, mirroring `_raw_kneighbors`.
        """
        xa, rows, cols = self._check_predict_X(X)
        if callable(self.metric):
            query = np.ascontiguousarray(
                np.asarray(xa.to_numpy(zero_copy_only=False), dtype=self._np_float())
                .reshape(rows, cols)
            )
            dist_list, idx_list = self._host_radius_neighbors(query, radius)
        else:
            flat_d, flat_i, counts = getattr(
                self._mlrs_obj, "radius_neighbors" + self._suffix()
            )(xa, rows, cols, float(radius))
            flat_d = np.asarray(flat_d, dtype=self._np_float())
            flat_i = np.asarray(flat_i, dtype=np.int32)
            offsets = np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))])
            dist_list = [flat_d[offsets[r]:offsets[r + 1]] for r in range(rows)]
            idx_list = [flat_i[offsets[r]:offsets[r + 1]] for r in range(rows)]
        return dist_list, idx_list, rows

    @staticmethod
    def _drop_self_ragged(dist_list, idx_list, rows):
        """Remove each query row's own index from a RAGGED `radius_neighbors`
        result (the radius-query sibling of `_drop_self_column`).

        Filters by INDEX IDENTITY, not "first" or "distance 0" — a genuine
        duplicate training point at distance 0 stays; only the row's own index
        is dropped, and only where it is actually present (a radius smaller
        than the point's own distance-0 match cannot happen, but a caller could
        still pass one, e.g. `radius=0` with floating-point jitter).
        """
        new_dist, new_idx = [], []
        for i in range(rows):
            d, idx = dist_list[i], idx_list[i]
            keep = idx != i
            new_dist.append(d[keep])
            new_idx.append(idx[keep])
        return new_dist, new_idx

    @staticmethod
    def _sort_ragged_by_distance(dist_list, idx_list, rows):
        """Sort each row's ragged matches by distance (`sort_results=True`)."""
        for i in range(rows):
            order = np.argsort(dist_list[i], kind="stable")
            dist_list[i] = dist_list[i][order]
            idx_list[i] = idx_list[i][order]
        return dist_list, idx_list

    def _resolve_radius(self, radius):
        """Validate the effective query `radius` (sklearn: must be non-negative)."""
        r = self.radius if radius is None else radius
        if not isinstance(r, (int, float, np.integer, np.floating)) or r < 0:
            raise ValueError(f"radius must be non-negative, got {r!r}")
        return float(r)

    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        """Every training point within `radius` of each row of `X`.

        The match count varies per row, so — unlike `kneighbors` — this returns
        two numpy OBJECT arrays (one ragged sub-array per query row), regardless
        of `output_type`: arrow/cudf have no ragged representation (the same
        exemption `kneighbors_graph` documents for its sparse-matrix return).

        `X=None` queries the TRAINING set with each point's own self-match
        removed by INDEX IDENTITY (see `_drop_self_ragged`), exactly as
        `kneighbors(X=None)` does. `sort_results=True` requires
        `return_distance=True` (sklearn's own restriction — an index-only,
        distance-sorted result is not orderable).
        """
        self._check_fitted()
        if sort_results and not return_distance:
            raise ValueError("return_distance must be True if sort_results is True.")
        r = self._resolve_radius(radius)
        query_is_train = X is None
        query = self._fit_X if query_is_train else X

        dist_list, idx_list, rows = self._raw_radius_neighbors(query, r)
        if query_is_train:
            dist_list, idx_list = self._drop_self_ragged(dist_list, idx_list, rows)
        if sort_results:
            dist_list, idx_list = self._sort_ragged_by_distance(dist_list, idx_list, rows)

        indices = np.empty(rows, dtype=object)
        for i in range(rows):
            indices[i] = idx_list[i]
        if not return_distance:
            return indices
        distances = np.empty(rows, dtype=object)
        for i in range(rows):
            distances[i] = dist_list[i]
        return distances, indices

    def radius_neighbors_graph(self, X=None, radius=None, mode="connectivity", sort_results=False):
        """The `(n_query, n_samples_fit_)` sparse radius-neighbour graph.

        `mode='connectivity'` stores 1 at each neighbour, `mode='distance'`
        stores the distance. Returns a scipy CSR matrix, as sklearn does — like
        `kneighbors_graph`, NOT routed through `output_type`.
        """
        from scipy.sparse import csr_matrix

        self._check_fitted()
        if mode not in ("connectivity", "distance"):
            raise ValueError(
                f'Unsupported mode, must be one of "connectivity" or '
                f'"distance" but got "{mode}" instead'
            )
        r = self._resolve_radius(radius)
        query_is_train = X is None
        query = self._fit_X if query_is_train else X

        dist_list, idx_list, rows = self._raw_radius_neighbors(query, r)
        if query_is_train:
            dist_list, idx_list = self._drop_self_ragged(dist_list, idx_list, rows)
        if sort_results:
            dist_list, idx_list = self._sort_ragged_by_distance(dist_list, idx_list, rows)

        counts = np.array([len(a) for a in idx_list], dtype=np.int64)
        indptr = np.concatenate([[0], np.cumsum(counts)])
        all_idx = (
            np.concatenate(idx_list) if rows else np.array([], dtype=np.int32)
        )
        if mode == "connectivity":
            data = np.ones(all_idx.shape[0], dtype=np.float64)
        else:
            data = (
                np.concatenate(dist_list).astype(np.float64)
                if rows
                else np.array([], dtype=np.float64)
            )
        return csr_matrix((data, all_idx, indptr), shape=(rows, self.n_samples_fit_))

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


# ---------------------------------------------------------------------------
# NearestNeighbors (NEIGH-PARAMS)
# ---------------------------------------------------------------------------


class NearestNeighbors(_NeighborsBase):
    """Brute-force neighbour search, matching sklearn's full parameter surface
    (NEIGH-PARAMS). No scoring mixin — exposes ``kneighbors`` /
    ``radius_neighbors`` (+ their ``_graph`` siblings), never ``predict``.

    ``NearestNeighbors(n_neighbors=5, output_type='input', *, radius=1.0,
    algorithm='auto', leaf_size=30, metric='minkowski', p=2,
    metric_params=None, n_jobs=None)``.

    See :class:`_NeighborsBase` for which parameters reach the device and which
    stop at this shim; ``radius`` is the one parameter unique to this estimator
    (the two k-neighbours estimators have no radius-based predict).
    """

    def __init__(
        self,
        n_neighbors=5,
        output_type="input",
        *,
        radius=1.0,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        n_jobs=None,
        device="auto",
    ):
        self.n_neighbors = n_neighbors
        self.output_type = output_type
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.device = device

    def _validate_params_for_fit(self):
        super()._validate_params_for_fit()
        if not isinstance(self.radius, (int, float, np.integer, np.floating)) or self.radius < 0:
            raise ValueError(f"radius must be non-negative, got {self.radius!r}")

    def fit(self, X, y=None):
        self._validate_params_for_fit()
        effective_metric, effective_params, effective_p = self._resolve_fit_metric()

        # `ensure_all_finite=False` RELOCATES the NaN/inf rejection into the
        # Rust `fit`, it does not drop it — see `KNeighborsRegressor.fit`.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = self._x_float(xa)

        metric_name, device_p = self._device_metric_args(
            effective_metric, effective_p
        )
        obj = self._ext().NearestNeighbors(
            self.n_neighbors, metric_name, device_p, self._device()
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj

        self.n_features_in_ = cols
        self.n_samples_fit_ = rows
        self.effective_metric_ = effective_metric
        self.effective_metric_params_ = effective_params
        # Only the ARROW buffer is retained; the numpy view of the training
        # matrix is built on first access (see `_fit_X` on the base).
        self._fit_arrow = (xa, rows, cols, dtype)
        return self


# ---------------------------------------------------------------------------
# KNeighborsClassifier (KNN-CLF-PARAMS)
# ---------------------------------------------------------------------------


class KNeighborsClassifier(ClassifierMixin, _NeighborsBase):
    """k-NN classification, matching sklearn's full parameter surface (NEIGH-02).

    ``KNeighborsClassifier(n_neighbors=5, output_type='input', *,
    weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
    metric_params=None, n_jobs=None)``.

    See :class:`_NeighborsBase` for which parameters reach the device and which
    stop at this shim.

    ## Labels are ENCODED here, not in Rust

    The Rust core votes over DENSE class indices and its label ingress is the
    shared float path, so it can only carry integer-valued targets. sklearn's
    ``classes_`` is ``np.unique(y)`` in ``y``'s own dtype — strings, booleans and
    non-contiguous integers included. This shim therefore does sklearn's
    ``np.unique(..., return_inverse=True)`` encoding itself, hands Rust the
    ``0..n_classes`` codes, and maps ``predict``'s result back through
    ``classes_``. A string target round-trips as strings, and the device never
    sees anything but a dense index.

    ## Multi-output targets

    A 2-D ``y`` makes ``classes_`` a LIST of per-column arrays, ``predict``
    return ``(n_samples, n_outputs)``, and ``predict_proba`` return a list of
    per-output probability matrices — sklearn's contract exactly. The vote for
    that case is rebuilt host-side from ``kneighbors``, which is the SAME
    neighbour set the device vote uses (both go through the core's
    ``neighbor_indices_metric`` with the same metric and ``k``), so the two paths
    cannot disagree about who the neighbours are.
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
        device="auto",
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
        self.device = device

    # -- fit ----------------------------------------------------------------- #

    def _encode_targets(self, y):
        """sklearn ``NeighborsBase._fit``'s classification-target block, verbatim.

        Sets ``outputs_2d_`` / ``classes_`` and returns the ``(n_samples,)`` or
        ``(n_samples, n_outputs)`` array of DENSE class codes. A single-column
        2-D ``y`` is treated as 1-D with sklearn's ``DataConversionWarning``,
        which is what makes ``clf.fit(X, y.reshape(-1, 1))`` behave the same in
        both libraries rather than silently becoming a one-output multi-output
        problem.
        """
        y = np.asarray(y)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            if y.ndim != 1:
                warnings.warn(
                    "A column-vector y was passed when a 1d array was "
                    "expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel().",
                    DataConversionWarning,
                    stacklevel=3,
                )
            self.outputs_2d_ = False
            y2 = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True
            y2 = y
        check_classification_targets(y)

        classes = []
        codes = np.empty(y2.shape, dtype=np.int64)
        for col in range(y2.shape[1]):
            classes_col, inverse = np.unique(y2[:, col], return_inverse=True)
            classes.append(classes_col)
            codes[:, col] = np.asarray(inverse).ravel()
        if not self.outputs_2d_:
            classes = classes[0]
            codes = codes.ravel()
        self.classes_ = classes
        return codes

    def fit(self, X, y):
        self._validate_params_for_fit()
        effective_metric, effective_params, effective_p = self._resolve_fit_metric()

        codes = self._encode_targets(y)
        # `ensure_all_finite=False` RELOCATES the NaN/inf rejection into the
        # Rust `fit`, it does not drop it: that call scans `X` itself and raises
        # `check_array`'s exact ValueError. `y` has already been through
        # `np.unique`, so its codes are finite by construction.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = self._x_float(xa)
        # Rust only ever needs ONE column of codes: it serves the single-output
        # vote, and for a multi-output target it is used purely as a fitted
        # `kneighbors` index (the vote is rebuilt per output column host-side).
        device_codes = codes if not self.outputs_2d_ else codes[:, 0]
        ya = self._normalize_y(device_codes, dtype=dtype, ensure_all_finite=False)

        metric_name, device_p = self._device_metric_args(
            effective_metric, effective_p
        )
        # A callable weighting is applied to `kneighbors` distances host-side, so
        # the device object is built with the (unused) uniform weighting.
        device_weights = "uniform" if callable(self.weights) else self.weights
        obj = self._ext().KNeighborsClassifier(
            self.n_neighbors, device_weights, metric_name, device_p, self._device()
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj

        self.n_features_in_ = cols
        self.n_samples_fit_ = rows
        self.effective_metric_ = effective_metric
        self.effective_metric_params_ = effective_params
        self._fit_codes = codes
        # Only the ARROW buffer is retained; the numpy view of the training
        # matrix is built on first access (see `_fit_X`). Materializing it
        # eagerly would put a second reshape+dtype pass over the whole matrix on
        # every `fit`, and the common path never reads it.
        self._fit_arrow = (xa, rows, cols, dtype)
        return self

    @property
    def _y(self):
        """The fitted DENSE class codes (sklearn's ``_y`` on a classifier)."""
        self._check_fitted()
        return self._fit_codes

    # -- predict ------------------------------------------------------------- #

    def _host_proba(self, X):
        """Per-output class probabilities, rebuilt host-side from `kneighbors`.

        Serves the three cases the device vote cannot: a multi-output target, a
        callable ``weights``, and a callable ``metric``. It is sklearn's
        ``predict_proba`` body term for term — weights summed into class columns,
        then the row divided by its total (leaving an all-zero row alone rather
        than dividing by zero).

        Returns ``(list_of_proba, n_queries)``, one ``(n_queries, n_classes_k)``
        matrix per output column, in ``classes_`` order.
        """
        dist, idx, n_queries = self._raw_kneighbors(X, self.n_neighbors)
        dist = np.asarray(dist, dtype=np.float64)
        idx = idx.astype(np.intp)

        weights = _get_weights(dist, self.weights)
        if weights is None:
            weights = np.ones_like(idx, dtype=np.float64)
        elif np.any(np.all(weights == 0, axis=1)):
            # sklearn refuses to normalize a row with no vote at all rather than
            # returning a uniform-looking 0/1 row.
            raise ValueError(
                "All neighbors of some sample is getting zero weights. "
                "Please modify 'weights' to avoid this case."
            )

        classes_ = self.classes_ if self.outputs_2d_ else [self.classes_]
        codes = self._y if self.outputs_2d_ else self._y.reshape((-1, 1))
        all_rows = np.arange(n_queries)

        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = codes[:, k][idx]
            proba_k = np.zeros((n_queries, classes_k.size), dtype=np.float64)
            for i, col in enumerate(pred_labels.T):
                proba_k[all_rows, col] += weights[:, i]
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer
            probabilities.append(proba_k)
        return probabilities, n_queries

    def _needs_host_vote(self):
        """Is this configuration outside what the device vote serves?"""
        return (
            self.outputs_2d_ or callable(self.weights) or callable(self.metric)
        )

    def predict(self, X):
        """The argmax of :meth:`predict_proba`, mapped back through ``classes_``.

        Ties resolve to the LOWEST class index on both paths — ``np.argmax``
        returns the first maximum and the Rust vote scans with a strict ``>`` —
        which is what sklearn's ``_mode`` (scipy returns the smallest tied value)
        and ``weighted_mode`` (argmax over the sorted class axis) both do.
        """
        self._check_fitted()
        if self._needs_host_vote():
            probabilities, n_queries = self._host_proba(X)
            classes_ = self.classes_ if self.outputs_2d_ else [self.classes_]
            out = np.empty((n_queries, len(classes_)), dtype=classes_[0].dtype)
            for k, classes_k in enumerate(classes_):
                out[:, k] = classes_k.take(np.argmax(probabilities[k], axis=1))
            if not self.outputs_2d_:
                return self._label_output(out.ravel(), (n_queries,), X)
            return out

        xa, rows, cols = self._check_predict_X(X)
        # The device returns the DENSE code (its own `classes_` is the identity
        # `0..n_classes` range, because this shim encoded the labels before
        # handing them over), so the mapping back to the user's label space
        # happens here and only here.
        dense = np.asarray(
            self._mlrs_obj.predict_labels(xa, rows, cols), dtype=np.intp
        )
        return self._label_output(self.classes_.take(dense), (rows,), X)

    def _label_output(self, labels, shape, X):
        """Route a predicted-label array through the `output_type` egress.

        Numeric label spaces go through the shared `_to_output` so `output_type`
        keeps working (and an integer `classes_` still egresses as the ints the
        user supplied). A NON-numeric one — strings, objects — has no arrow float
        or int form and sklearn returns plain numpy for it anyway, so it is
        handed back directly rather than being forced through a columnar egress
        that would either fail or lossily coerce it.
        """
        labels = np.asarray(labels)
        if labels.dtype.kind in "biuf":
            return self._to_output(labels, shape, X, labels.dtype)
        return labels.reshape(shape)

    def predict_proba(self, X):
        """Per-class probabilities: `(n_samples, n_classes)`, or a LIST of those.

        A multi-output fit returns one matrix per output column, in the order of
        ``classes_`` — sklearn's contract.
        """
        self._check_fitted()
        if self._needs_host_vote():
            probabilities, n_queries = self._host_proba(X)
            if self.outputs_2d_:
                return probabilities
            return self._to_output(
                np.ascontiguousarray(probabilities[0]).ravel(),
                (n_queries, self.classes_.size),
                X,
                self._np_float(),
            )

        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict_proba")(xa, rows, cols)
        n_classes = self._mlrs_obj.n_classes()
        return self._to_output(out, (rows, n_classes), X, self._np_float())


# ---------------------------------------------------------------------------
# KNeighborsRegressor (KNN-REG-PARAMS)
# ---------------------------------------------------------------------------


class KNeighborsRegressor(RegressorMixin, _NeighborsBase):
    """k-NN regression, matching sklearn's full parameter surface (NEIGH-03).

    ``KNeighborsRegressor(n_neighbors=5, output_type='input', *,
    weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
    metric_params=None, n_jobs=None)``.

    See :class:`_NeighborsBase` for which parameters reach the device and which
    stop at this shim. Multi-output targets are supported: a 2-D ``y``
    (``n_samples × n_outputs``) makes ``predict`` return ``n_samples ×
    n_outputs``.
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
        device="auto",
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
        self.device = device

    # -- fit ----------------------------------------------------------------- #

    def fit(self, X, y):
        self._validate_params_for_fit()
        effective_metric, effective_params, effective_p = self._resolve_fit_metric()

        # `ensure_all_finite=False` RELOCATES the NaN/inf rejection into the
        # Rust `fit`, it does not drop it: that call scans both operands itself
        # and raises `check_array`'s exact ValueError. Brute-force k-NN has no
        # model to solve for, so validation IS essentially all `fit` does — a
        # second full scan here would be a large fraction of it, and the Rust
        # one is the faster of the two.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = self._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype, ensure_all_finite=False)

        metric_name, device_p = self._device_metric_args(
            effective_metric, effective_p
        )
        # A callable weighting is applied to `kneighbors` distances host-side,
        # so the device object is built with the (unused) uniform weighting;
        # `predict` never asks it for a prediction in that case.
        device_weights = "uniform" if callable(self.weights) else self.weights
        obj = self._ext().KNeighborsRegressor(
            self.n_neighbors, device_weights, metric_name, device_p, self._device()
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
        # on first access (see `_fit_X` on the base / `_y` below).
        self._fit_arrow = (xa, rows, cols, dtype)
        self._fit_y_arrow = ya
        return self

    @property
    def _y(self):
        """The fitted targets: `(n_samples_fit_,)`, or 2-D when multi-output."""
        self._check_fitted()
        _, rows, _, dtype = self._fit_arrow
        arr = np.asarray(
            self._fit_y_arrow.to_numpy(zero_copy_only=False), dtype=dtype
        )
        n_outputs = self._mlrs_obj.n_outputs()
        return arr.reshape(rows) if n_outputs == 1 else arr.reshape(rows, n_outputs)

    # -- predict ------------------------------------------------------------- #

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
