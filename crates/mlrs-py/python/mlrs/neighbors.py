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


class NearestNeighbors(MlrsBase):
    """Brute-force k-NN search (NEIGH-01). Exposes ``kneighbors`` — no predict."""

    def __init__(self, n_neighbors=5, output_type="input"):
        self.n_neighbors = n_neighbors
        self.output_type = output_type

    def fit(self, X, y=None):
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

    def __init__(self, n_neighbors=5, output_type="input"):
        self.n_neighbors = n_neighbors
        self.output_type = output_type

    def fit(self, X, y):
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


class KNeighborsRegressor(RegressorMixin, MlrsBase):
    """k-NN regression by neighbor mean (NEIGH-03)."""

    def __init__(self, n_neighbors=5, output_type="input"):
        self.n_neighbors = n_neighbors
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=KNeighborsClassifier._x_float(xa))
        obj = self._ext().KNeighborsRegressor(self.n_neighbors)
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self.n_features_in_ = cols
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())
