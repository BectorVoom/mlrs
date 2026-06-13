"""Neighbors estimator shells (PY-01/PY-02).

NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor. Pure-Python shells
subclassing :class:`MlrsBase` (KNeighborsClassifier adds ``ClassifierMixin``,
KNeighborsRegressor adds ``RegressorMixin``; NearestNeighbors has no scoring
mixin — it exposes ``kneighbors``, not ``predict``). sklearn-faithful
``__init__`` stores ``n_neighbors`` verbatim (RESEARCH 06 §Hyperparameter
Mapping); ``fit`` / ``kneighbors`` / ``predict`` are Plan-04 placeholders.
"""

from sklearn.base import ClassifierMixin, RegressorMixin

from .base import MlrsBase


class NearestNeighbors(MlrsBase):
    """Brute-force k-NN search (NEIGH-01). Exposes ``kneighbors``."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        raise NotImplementedError("mlrs NearestNeighbors.fit lands in Plan 04")

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        raise NotImplementedError(
            "mlrs NearestNeighbors.kneighbors lands in Plan 04"
        )


class KNeighborsClassifier(ClassifierMixin, MlrsBase):
    """k-NN classification by majority vote (NEIGH-02)."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        raise NotImplementedError(
            "mlrs KNeighborsClassifier.fit lands in Plan 04"
        )

    def predict(self, X):
        raise NotImplementedError(
            "mlrs KNeighborsClassifier.predict lands in Plan 04"
        )


class KNeighborsRegressor(RegressorMixin, MlrsBase):
    """k-NN regression by neighbor mean (NEIGH-03)."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        raise NotImplementedError(
            "mlrs KNeighborsRegressor.fit lands in Plan 04"
        )

    def predict(self, X):
        raise NotImplementedError(
            "mlrs KNeighborsRegressor.predict lands in Plan 04"
        )
