"""Clustering estimator shells (PY-01/PY-02).

KMeans, DBSCAN. Pure-Python shells subclassing :class:`MlrsBase` +
``ClusterMixin`` (gives ``fit_predict``). sklearn-faithful ``__init__`` stores
every ctor arg verbatim (RESEARCH 06 §Hyperparameter Mapping); ``fit`` is a
Plan-04 placeholder. KMeans maps sklearn ``random_state`` → Rust ``seed`` at
the Rust boundary (Plan 04) but stores ``random_state`` verbatim here.
"""

from sklearn.base import ClusterMixin

from .base import MlrsBase


class KMeans(ClusterMixin, MlrsBase):
    """Lloyd's k-means, k-means++ init (CLUSTER-01).

    ``init='k-means++'`` is the only supported value in v1; ``random_state``
    is mapped to the Rust ``seed`` inside ``fit`` (Plan 04).
    """

    def __init__(
        self,
        n_clusters=8,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=None,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        raise NotImplementedError("mlrs KMeans.fit lands in Plan 04")


class DBSCAN(ClusterMixin, MlrsBase):
    """Density-based clustering (CLUSTER-02)."""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        raise NotImplementedError("mlrs DBSCAN.fit lands in Plan 04")
