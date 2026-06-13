"""Clustering estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

KMeans, DBSCAN -> ``ClusterMixin`` (gives ``fit_predict``). sklearn-faithful
``__init__`` stores every ctor arg verbatim (RESEARCH 06 §Hyperparameter
Mapping); KMeans stores sklearn ``random_state`` verbatim and maps it to the
Rust ``seed`` only at the ``_mlrs`` boundary inside ``fit``. DBSCAN has NO
standalone ``predict`` (algos D-08) — only ``fit`` + ``labels_``.
"""

import numpy as np
from sklearn.base import ClusterMixin

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
        obj = self._ext().KMeans(
            self.n_clusters, self.max_iter, self.tol, self.random_state
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
