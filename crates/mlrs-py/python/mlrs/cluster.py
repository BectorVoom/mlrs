"""Clustering estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

KMeans, DBSCAN -> ``ClusterMixin`` (gives ``fit_predict``). sklearn-faithful
``__init__`` stores every ctor arg verbatim (RESEARCH 06 §Hyperparameter
Mapping); KMeans stores sklearn ``random_state`` verbatim and maps it to the
Rust ``seed`` only at the ``_mlrs`` boundary inside ``fit``. DBSCAN has NO
standalone ``predict`` (algos D-08) — only ``fit`` + ``labels_``.
"""

import warnings

import numpy as np
from sklearn.base import ClusterMixin, TransformerMixin

from .base import MlrsBase


class KMeans(ClusterMixin, MlrsBase):
    """k-means clustering with sklearn's full parameter set (CLUSTER-01).

    Every ``sklearn.cluster.KMeans`` ctor parameter is supported:

    ``init``
        ``'k-means++'``, ``'random'``, an array-like of shape
        ``(n_clusters, n_features)``, or a callable
        ``init(X, k, random_state=...)``. The array and callable forms are
        resolved HERE (a callable is evaluated once, at ``fit``) and passed
        down as an explicit init array, which is exactly how sklearn's
        ``_init_centroids`` treats them.
    ``n_init``
        ``'auto'`` or a positive int. ``'auto'`` resolves as sklearn does:
        1 for ``k-means++`` / an explicit init, 10 for ``'random'``.
    ``algorithm``
        ``'lloyd'`` or ``'elkan'``. Elkan is an exact triangle-inequality
        acceleration — it returns the same fit as Lloyd, faster on
        well-separated data, at the cost of an ``n x k`` bounds matrix.
    ``verbose`` / ``copy_x``
        Accepted for signature compatibility. mlrs never prints from the
        library and never writes into the caller's ``X``, so neither has an
        observable effect.
    ``random_state``
        ``None`` maps to a fixed default seed, so a default-constructed fit is
        reproducible (sklearn's ``None`` draws from the global numpy RNG).

    Fitted attributes: ``cluster_centers_``, ``labels_``, ``inertia_``,
    ``n_iter_``.
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
        output_type="input",
    ):
        # sklearn-faithful: store every ctor arg VERBATIM (no validation, no
        # normalization) so ``get_params()``/``clone()`` round-trip exactly.
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.output_type = output_type

    def _resolve_init(self, X, rows, cols):
        """Split sklearn's polymorphic ``init`` into ``(init_str, init_array)``.

        The string strategies pass straight through; an array-like or callable
        becomes a flat row-major ``list[float]`` of length ``k * n_features``,
        which is what ``PyKMeans`` takes as ``init_array`` (and which wins over
        the string, mirroring ``_init_centroids``' own precedence).
        """
        init = self.init
        if isinstance(init, str):
            return init, None

        if callable(init):
            # sklearn calls ``init(X, k, random_state=rs)``. Pass a numpy
            # RandomState seeded from ``random_state`` so a user callable that
            # actually uses it stays reproducible.
            rs = np.random.RandomState(
                0 if self.random_state is None else int(self.random_state)
            )
            centers = init(X, self.n_clusters, random_state=rs)
        else:
            centers = init

        arr = np.ascontiguousarray(centers, dtype=np.float64)
        if arr.shape != (self.n_clusters, cols):
            raise ValueError(
                f"init has shape {arr.shape}, expected "
                f"({self.n_clusters}, {cols})"
            )
        return "k-means++", arr.ravel().tolist()

    def _resolve_n_init(self):
        """``'auto'`` -> ``None`` (the extension's 'auto'); an int passes through."""
        n_init = self.n_init
        if isinstance(n_init, str):
            if n_init != "auto":
                raise ValueError(
                    f"unknown n_init {n_init!r} (the only legal string is 'auto')"
                )
            return None
        return int(n_init)

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        # Normalize random_state at the boundary (WR-06): a numpy integer scalar
        # is not guaranteed to coerce to PyO3's u64 extractor, and a negative
        # value would fail with an opaque OverflowError. int() coercion mirrors
        # SpectralClustering.fit; None stays None (PyKMeans maps it to a default).
        seed = None if self.random_state is None else int(self.random_state)
        init_str, init_array = self._resolve_init(X, rows, cols)
        obj = self._ext().KMeans(
            self.n_clusters,
            init_str,
            init_array,
            self._resolve_n_init(),
            self.max_iter,
            self.tol,
            bool(self.verbose),
            seed,
            bool(self.copy_x),
            self.algorithm,
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    @property
    def cluster_centers_(self):
        out = self._suffixed("cluster_centers")()
        return self._to_output(
            out, (self.n_clusters, -1), None, self._np_float()
        )

    @property
    def labels_(self):
        self._check_fitted()
        return self._to_output(self._mlrs_obj.labels_(), (-1,), None, np.int32)

    @property
    def inertia_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "inertia" + self._suffix())()

    @property
    def n_iter_(self):
        """Iterations run by the restart that WON the ``n_init`` selection."""
        self._check_fitted()
        return self._mlrs_obj.n_iter_()

    @property
    def _algorithm(self):
        """The algorithm that actually ran (``'elkan'`` degrades to ``'lloyd'``
        at ``n_clusters == 1``). Private, matching sklearn's own spelling."""
        self._check_fitted()
        return self._mlrs_obj.algorithm_used()

    @property
    def _n_init(self):
        """The restart count that actually ran, after ``'auto'`` resolution and
        the explicit-init override. Private, matching sklearn's spelling."""
        self._check_fitted()
        return self._mlrs_obj.n_init_used()


class DBSCAN(ClusterMixin, MlrsBase):
    """Density-based clustering (CLUSTER-02). No standalone ``predict`` (D-08)."""

    def __init__(self, eps=0.5, min_samples=5, output_type="input"):
        self.eps = eps
        self.min_samples = min_samples
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().DBSCAN(self.eps, self.min_samples)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    @property
    def labels_(self):
        self._check_fitted()
        return self._to_output(self._mlrs_obj.labels_(), (-1,), None, np.int32)

    @property
    def core_sample_indices_(self):
        self._check_fitted()
        return self._to_output(
            self._mlrs_obj.core_sample_indices_(), (-1,), None, np.int32
        )


class AgglomerativeClustering(ClusterMixin, MlrsBase):
    """Single-linkage hierarchical clustering (AGGLO-01).

    cuML-parity scope: ``linkage='single'`` only (any other linkage raises).
    ``metric`` accepts the sklearn/cuML set ``{'euclidean', 'l2', 'manhattan',
    'l1', 'cityblock', 'cosine'}``. Labels/children are EXACT sklearn matches
    (line-for-line port of the unstructured single-linkage path). No standalone
    ``predict`` — ``fit`` + ``labels_``/``children_`` (sklearn parity).
    Defaults mirror ``PyAgglomerativeClustering`` ``#[new]``.
    """

    def __init__(
        self,
        n_clusters=2,
        metric="euclidean",
        linkage="single",
        output_type="input",
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.output_type = output_type

    def fit(self, X, y=None):
        # sklearn exposes ward/complete/average/single; the cuML/mlrs scope is
        # single-linkage only — reject the rest loudly, never silently degrade.
        if self.linkage != "single":
            raise ValueError(
                f"AgglomerativeClustering: unsupported linkage {self.linkage!r};"
                " mlrs supports linkage='single' only (the cuML scope)"
            )
        xa, rows, cols = self._normalize(X)
        obj = self._ext().AgglomerativeClustering(int(self.n_clusters), self.metric)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    @property
    def labels_(self):
        self._check_fitted()
        return self._to_output(self._mlrs_obj.labels_(), (-1,), None, np.int32)

    @property
    def children_(self):
        self._check_fitted()
        # The FFI returns children flattened row-major; reshape to (n-1, 2).
        return self._to_output(
            self._mlrs_obj.children_(), (-1, 2), None, np.int64
        )

    @property
    def n_leaves_(self):
        self._check_fitted()
        return self._mlrs_obj.n_leaves_()

    @property
    def n_connected_components_(self):
        self._check_fitted()
        return self._mlrs_obj.n_connected_components_()

    @property
    def n_clusters_(self):
        self._check_fitted()
        return int(self.n_clusters)


class SpectralClustering(ClusterMixin, MlrsBase):
    """Spectral clustering (SPECTRAL-02).

    Mirrors ``sklearn.cluster.SpectralClustering``'s parameter surface. Labels
    match sklearn up to a permutation of the label ids, which is inherent to
    clustering — score with ``adjusted_rand_score``, never by comparing ids.

    ``ClusterMixin`` supplies ``fit_predict``. There is no standalone
    ``predict``: like sklearn's, this estimator is transductive.

    ``kernel_params`` is deliberately not exposed. sklearn overwrites
    ``params["gamma"|"degree"|"coef0"]`` from the estimator's own attributes for
    any non-callable affinity and then calls ``pairwise_kernels`` with
    ``filter_params=True``, which drops every key outside those three — so for a
    string affinity the parameter is provably a no-op, and callable affinities
    are out of scope here.
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        eigen_solver=None,
        n_components=None,
        random_state=None,
        n_init=10,
        gamma=1.0,
        affinity="rbf",
        n_neighbors=10,
        eigen_tol="auto",
        assign_labels="kmeans",
        degree=3,
        coef0=1,
        n_jobs=None,
        verbose=False,
        output_type="input",
    ):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        # sklearn's ``eigen_tol`` is a ``float | "auto"`` union; the Rust side
        # takes one type, so "auto" maps to None here rather than being
        # re-parsed from a PyAny across the boundary.
        tol = None if self.eigen_tol == "auto" else self.eigen_tol
        seed = None if self.random_state is None else int(self.random_state)
        obj = self._ext().SpectralClustering(
            self.n_clusters,
            self.eigen_solver,
            self.n_components,
            seed,
            self.n_init,
            self.gamma,
            self.affinity,
            self.n_neighbors,
            tol,
            self.assign_labels,
            float(self.degree),
            float(self.coef0),
            self.n_jobs,
            bool(self.verbose),
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        self._n_rows = rows
        # sklearn emits this from ``_spectral_embedding`` and then changes
        # nothing at all; reproduce the warning and its advisory-only nature.
        if obj.n_graph_components() > 1:
            warnings.warn(
                "Graph is not fully connected, spectral embedding may not "
                "work as expected."
            )
        return self

    @property
    def labels_(self):
        self._check_fitted()
        return self._to_output(self._mlrs_obj.labels_(), (-1,), None, np.int32)

    @property
    def affinity_matrix_(self):
        """The fitted affinity graph.

        A ``scipy.sparse.csr_matrix`` for the kNN affinities and a dense
        ``numpy`` array for the kernel / precomputed ones — the same split
        sklearn produces, and the reason the kNN path can be fitted at sample
        counts where an ``n x n`` dense matrix would not fit in memory.
        """
        self._check_fitted()
        csr = self._mlrs_obj.affinity_matrix_csr()
        n = self._n_rows
        if csr is not None:
            from scipy.sparse import csr_matrix

            indptr, indices, data = csr
            return csr_matrix(
                (
                    np.asarray(data, dtype=np.float64),
                    np.asarray(indices, dtype=np.int32),
                    np.asarray(indptr, dtype=np.int32),
                ),
                shape=(n, n),
            )
        dense = self._mlrs_obj.affinity_matrix_dense()
        return np.asarray(dense, dtype=np.float64).reshape(n, n)


class SpectralEmbedding(TransformerMixin, MlrsBase):
    """Spectral embedding / Laplacian eigenmaps (SPECTRAL-01).

    Mirrors ``sklearn.manifold.SpectralEmbedding``'s full parameter surface.
    sklearn's estimator is transductive: it exposes ``fit`` + ``fit_transform``
    + ``embedding_`` and deliberately NO out-of-sample ``transform``, and
    neither does this.

    Two sklearn behaviors are worth calling out because the pre-rewrite shim got
    them wrong:

    * ``n_neighbors=None`` resolves to ``max(n_samples // 10, 1)`` at fit —
      truncating division, floored at 1. It is NOT a constant 10, so the graph
      this builds now differs from what the old shim built on any input with
      ``n_samples != 100``. Read the resolved value back from ``n_neighbors_``.
    * ``gamma=None`` resolves to ``1 / n_features``, and ``gamma=0`` is legal
      (sklearn's constraint is left-closed); only a negative or non-finite
      ``gamma`` is rejected.

    There is no longer any cap on ``n_samples``: the former implementation ran a
    dense ``n x n`` Jacobi eigendecomposition on the device, which capped
    ``n_samples <= 64``.
    """

    def __init__(
        self,
        n_components=2,
        *,
        affinity="nearest_neighbors",
        gamma=None,
        random_state=None,
        eigen_solver=None,
        eigen_tol="auto",
        n_neighbors=None,
        n_jobs=None,
        output_type="input",
    ):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        # sklearn's ``eigen_tol`` is a ``float | "auto"`` union; the Rust side
        # takes one type, so "auto" is mapped to None here rather than being
        # re-parsed from a PyAny across the boundary.
        tol = None if self.eigen_tol == "auto" else self.eigen_tol
        seed = None if self.random_state is None else int(self.random_state)
        obj = self._ext().SpectralEmbedding(
            self.n_components,
            self.affinity,
            self.gamma,
            seed,
            self.eigen_solver,
            tol,
            self.n_neighbors,
            self.n_jobs,
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        self._n_rows = rows
        # sklearn emits this from ``_spectral_embedding`` and then changes
        # nothing at all; reproduce the warning, and the fact that it is
        # advisory only.
        if obj.n_graph_components() > 1:
            warnings.warn(
                "Graph is not fully connected, spectral embedding may not "
                "work as expected."
            )
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.embedding_

    @property
    def embedding_(self):
        out = self._suffixed("embedding")()
        return self._to_output(
            out, (-1, self.n_components), None, self._np_float()
        )

    @property
    def affinity_matrix_(self):
        """The fitted affinity graph.

        Returns a ``scipy.sparse.csr_matrix`` for the kNN affinities and a dense
        ``numpy`` array for the kernel / precomputed ones — the same split
        sklearn produces, and the reason the kNN path can be fitted at sample
        counts where an ``n x n`` dense matrix would not fit in memory.
        """
        self._check_fitted()
        csr = self._mlrs_obj.affinity_matrix_csr()
        n = self._n_rows
        if csr is not None:
            from scipy.sparse import csr_matrix

            indptr, indices, data = csr
            return csr_matrix(
                (
                    np.asarray(data, dtype=np.float64),
                    np.asarray(indices, dtype=np.int32),
                    np.asarray(indptr, dtype=np.int32),
                ),
                shape=(n, n),
            )
        dense = self._mlrs_obj.affinity_matrix_dense()
        return np.asarray(dense, dtype=np.float64).reshape(n, n)

    @property
    def n_neighbors_(self):
        """The resolved neighbor count. sklearn sets this attribute only on the
        ``nearest_neighbors`` branch, so it raises ``AttributeError`` for the
        other affinities exactly as sklearn does."""
        self._check_fitted()
        v = self._mlrs_obj.n_neighbors_resolved()
        if v is None:
            raise AttributeError(
                "'SpectralEmbedding' object has no attribute 'n_neighbors_' "
                f"(affinity={self.affinity!r} does not build a kNN graph)"
            )
        return int(v)

    @property
    def gamma_(self):
        """The resolved kernel coefficient. Set only on a kernel affinity,
        matching sklearn's attribute presence."""
        self._check_fitted()
        v = self._mlrs_obj.gamma_resolved()
        if v is None:
            raise AttributeError(
                "'SpectralEmbedding' object has no attribute 'gamma_' "
                f"(affinity={self.affinity!r} is not a kernel)"
            )
        return float(v)


class HDBSCAN(ClusterMixin, MlrsBase):
    """Hierarchical density-based clustering (CLUSTER-03 / SHIM-01 pair).

    ``ClusterMixin`` provides ``fit_predict``; the shim forwards ``fit`` to the
    ``_mlrs.HDBSCAN`` wrapper and exposes ``labels_`` / ``probabilities_`` /
    ``outlier_scores_`` (the GLOSH scores surface as ``None`` until the
    feature-space front-end lands — see 16-10-SUMMARY). Defaults mirror
    ``PyHDBSCAN`` ``#[new]`` at cluster.rs:375-379.
    """

    def __init__(
        self,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom",
        metric="euclidean",
        alpha=1.0,
        max_cluster_size=0,
        output_type="input",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.alpha = alpha
        self.max_cluster_size = max_cluster_size
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().HDBSCAN(
            self.min_cluster_size,
            self.min_samples,
            self.cluster_selection_epsilon,
            self.cluster_selection_method,
            self.metric,
            self.alpha,
            self.max_cluster_size,
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    @property
    def labels_(self):
        self._check_fitted()
        return self._to_output(self._mlrs_obj.labels_(), (-1,), None, np.int32)

    @property
    def probabilities_(self):
        self._check_fitted()
        out = self._suffixed("probabilities")()
        if out is None:
            return None
        return self._to_output(out, (-1,), None, self._np_float())

    @property
    def outlier_scores_(self):
        self._check_fitted()
        out = self._suffixed("outlier_scores")()
        if out is None:
            return None
        return self._to_output(out, (-1,), None, self._np_float())
