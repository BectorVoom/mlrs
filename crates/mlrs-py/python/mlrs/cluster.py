"""Clustering estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

KMeans, DBSCAN -> ``ClusterMixin`` (gives ``fit_predict``). sklearn-faithful
``__init__`` stores every ctor arg verbatim (RESEARCH 06 §Hyperparameter
Mapping); KMeans stores sklearn ``random_state`` verbatim and maps it to the
Rust ``seed`` only at the ``_mlrs`` boundary inside ``fit``. DBSCAN has NO
standalone ``predict`` (algos D-08) — only ``fit`` + ``labels_``.
"""

import numpy as np
from sklearn.base import ClusterMixin, TransformerMixin

from .base import MlrsBase


class KMeans(ClusterMixin, MlrsBase):
    """Lloyd's k-means, k-means++ init (CLUSTER-01).

    ``init='k-means++'`` is the only supported value in v1; ``random_state`` is
    mapped to the Rust ``seed`` inside ``fit`` (``None`` -> a fixed default seed).
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None,
        output_type="input",
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        # Normalize random_state at the boundary (WR-06): a numpy integer scalar
        # is not guaranteed to coerce to PyO3's u64 extractor, and a negative
        # value would fail with an opaque OverflowError. int() coercion mirrors
        # SpectralClustering.fit; None stays None (PyKMeans maps it to a default).
        seed = None if self.random_state is None else int(self.random_state)
        obj = self._ext().KMeans(self.n_clusters, self.max_iter, self.tol, seed)
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

    sklearn ``random_state`` is stored verbatim and mapped to the Rust ``seed``
    only at the ``_mlrs`` boundary inside ``fit`` (``None`` -> a fixed default
    seed). No standalone ``predict`` — labels-only (``fit`` + ``labels_``).
    Defaults mirror ``PySpectralClustering`` ``#[new]`` at spectral.rs:313-314.
    """

    def __init__(
        self,
        n_clusters=8,
        n_components=None,
        affinity="rbf",
        gamma=1.0,
        n_neighbors=10,
        random_state=None,
        output_type="input",
    ):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        seed = 0 if self.random_state is None else int(self.random_state)
        obj = self._ext().SpectralClustering(
            self.n_clusters,
            self.n_components,
            self.affinity,
            self.gamma,
            self.n_neighbors,
            seed,
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    @property
    def labels_(self):
        self._check_fitted()
        return self._to_output(self._mlrs_obj.labels_(), (-1,), None, np.int32)


class SpectralEmbedding(TransformerMixin, MlrsBase):
    """Spectral embedding / Laplacian eigenmaps (SPECTRAL-01).

    sklearn's ``SpectralEmbedding`` supports ``fit`` + ``fit_transform`` only
    (no out-of-sample ``transform``); the embedding is materialized via the
    ``embedding_`` fitted attribute. Defaults mirror ``PySpectralEmbedding``
    ``#[new]`` at spectral.rs:121-122.
    """

    def __init__(
        self,
        n_components=2,
        affinity="nearest_neighbors",
        gamma=None,
        n_neighbors=10,
        output_type="input",
    ):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().SpectralEmbedding(
            self.n_components, self.affinity, self.gamma, self.n_neighbors
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        self._n_rows = rows
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
