"""``DistanceMetric`` — sklearn's metric-object interface over the mlrs
pairwise engine.

``DistanceMetric.get_metric(name, **kwargs).pairwise(X, Y)`` evaluates through
:func:`mlrs.metrics.pairwise_distances`; ``dist``/``rdist`` conversions follow
sklearn's definitions (the "reduced distance" is the squared distance for the
euclidean family, ``d**p`` for Minkowski, ``sin²(d/2)`` for haversine, and the
distance itself otherwise).
"""

import numpy as np

__all__ = ["DistanceMetric"]

_ALIASES = {
    "euclidean": "euclidean",
    "l2": "euclidean",
    "manhattan": "manhattan",
    "cityblock": "manhattan",
    "l1": "manhattan",
    "chebyshev": "chebyshev",
    "infinity": "chebyshev",
    "minkowski": "minkowski",
    "p": "minkowski",
    "seuclidean": "seuclidean",
    "mahalanobis": "mahalanobis",
    "hamming": "hamming",
    "canberra": "canberra",
    "braycurtis": "braycurtis",
    "haversine": "haversine",
    "jaccard": "jaccard",
    "dice": "dice",
    "rogerstanimoto": "rogerstanimoto",
    "russellrao": "russellrao",
    "sokalsneath": "sokalsneath",
    "matching": "hamming",
}


class DistanceMetric:
    """sklearn's ``DistanceMetric`` interface. Build with :meth:`get_metric`."""

    def __init__(self, metric, dtype=np.float64, **kwargs):
        self.metric = metric
        self.dtype = dtype
        self.kwargs = kwargs

    @classmethod
    def get_metric(cls, metric, dtype=np.float64, **kwargs):
        """Matches ``sklearn.metrics.DistanceMetric.get_metric``."""
        if isinstance(metric, DistanceMetric):
            return metric
        if callable(metric):
            return cls(metric, dtype, **kwargs)
        if metric not in _ALIASES:
            raise ValueError("Unrecognized metric '%s'" % metric)
        name = _ALIASES[metric]
        if name == "minkowski":
            p = kwargs.get("p", 2)
            if p == 1:
                name, kwargs = "manhattan", {}
            elif p == 2 and kwargs.get("w") is None:
                name, kwargs = "euclidean", {}
            elif np.isinf(p):
                name, kwargs = "chebyshev", {}
        return cls(name, dtype, **kwargs)

    def pairwise(self, X, Y=None):
        """The distance matrix between the rows of ``X`` (and ``Y``)."""
        from .pairwise import pairwise_distances

        kw = dict(self.kwargs)
        metric = self.metric
        if metric == "mahalanobis" and "VI" not in kw and "V" in kw:
            kw["VI"] = np.linalg.inv(np.asarray(kw.pop("V")))
        return np.asarray(pairwise_distances(X, Y, metric=metric, **kw), dtype=self.dtype)

    def _p(self):
        return self.kwargs.get("p", 2)

    def dist_to_rdist(self, dist):
        """Distance → reduced distance."""
        d = np.asarray(dist, dtype=np.float64)
        if self.metric in ("euclidean", "seuclidean", "mahalanobis"):
            return d**2
        if self.metric == "minkowski":
            return d ** self._p()
        if self.metric == "haversine":
            return np.sin(0.5 * d) ** 2
        return d

    def rdist_to_dist(self, rdist):
        """Reduced distance → distance."""
        r = np.asarray(rdist, dtype=np.float64)
        if self.metric in ("euclidean", "seuclidean", "mahalanobis"):
            return np.sqrt(r)
        if self.metric == "minkowski":
            return r ** (1.0 / self._p())
        if self.metric == "haversine":
            return 2 * np.arcsin(np.sqrt(r))
        return r
