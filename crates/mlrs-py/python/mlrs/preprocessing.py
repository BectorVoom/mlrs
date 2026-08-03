"""Preprocessing scaler shims (PREP-01) delegating to ``_mlrs``.

``StandardScaler`` / ``MinMaxScaler`` / ``MaxAbsScaler`` / ``RobustScaler`` ->
``TransformerMixin`` (gives ``fit_transform``) with ``inverse_transform``.
``Normalizer`` / ``Binarizer`` are also ``TransformerMixin`` but have no
``inverse_transform`` (matching sklearn — both maps are lossy). sklearn-faithful
``__init__``: every constructor argument is stored verbatim under the same name
(RESEARCH 06 §Hyperparameter Mapping); ``fit`` is unsupervised (``y=None``) and
returns ``self``.
"""

from sklearn.base import TransformerMixin

from .base import MlrsBase


class StandardScaler(TransformerMixin, MlrsBase):
    """`(X - mean) / std`, matching `sklearn.preprocessing.StandardScaler`."""

    def __init__(self, *, with_mean=True, with_std=True, copy=True, output_type="input"):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().StandardScaler(self.with_mean, self.with_std)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())

    def inverse_transform(self, X):
        self._check_fitted()
        xa, rows, cols = self._normalize(X, dtype=self._np_float())
        out = self._suffixed("inverse_transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())

    @property
    def mean_(self):
        return self._to_output(self._suffixed("mean")(), (-1,), None, self._np_float())

    @property
    def var_(self):
        return self._to_output(self._suffixed("var")(), (-1,), None, self._np_float())

    @property
    def scale_(self):
        return self._to_output(self._suffixed("scale")(), (-1,), None, self._np_float())


class MinMaxScaler(TransformerMixin, MlrsBase):
    """Linear map of each column into `feature_range`."""

    def __init__(self, feature_range=(0, 1), *, clip=False, copy=True, output_type="input"):
        self.feature_range = feature_range
        self.clip = clip
        self.copy = copy
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        lo, hi = self.feature_range
        obj = self._ext().MinMaxScaler(float(lo), float(hi), self.clip)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())

    def inverse_transform(self, X):
        self._check_fitted()
        xa, rows, cols = self._normalize(X, dtype=self._np_float())
        out = self._suffixed("inverse_transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())

    @property
    def data_min_(self):
        return self._to_output(self._suffixed("data_min")(), (-1,), None, self._np_float())

    @property
    def data_max_(self):
        return self._to_output(self._suffixed("data_max")(), (-1,), None, self._np_float())

    @property
    def data_range_(self):
        import numpy as np

        return np.asarray(self.data_max_) - np.asarray(self.data_min_)

    @property
    def scale_(self):
        return self._to_output(self._suffixed("scale")(), (-1,), None, self._np_float())

    @property
    def min_(self):
        return self._to_output(self._suffixed("min")(), (-1,), None, self._np_float())


class MaxAbsScaler(TransformerMixin, MlrsBase):
    """`X / max(|X|)` per column."""

    def __init__(self, *, copy=True, output_type="input"):
        self.copy = copy
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().MaxAbsScaler()
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())

    def inverse_transform(self, X):
        self._check_fitted()
        xa, rows, cols = self._normalize(X, dtype=self._np_float())
        out = self._suffixed("inverse_transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())

    @property
    def max_abs_(self):
        return self._to_output(self._suffixed("max_abs")(), (-1,), None, self._np_float())

    @property
    def scale_(self):
        return self._to_output(self._suffixed("scale")(), (-1,), None, self._np_float())


class RobustScaler(TransformerMixin, MlrsBase):
    """`(X - median) / IQR`, robust to outliers."""

    def __init__(
        self,
        *,
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
        copy=True,
        unit_variance=False,
        output_type="input",
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy
        self.unit_variance = unit_variance
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        q_min, q_max = self.quantile_range
        obj = self._ext().RobustScaler(
            self.with_centering,
            self.with_scaling,
            float(q_min),
            float(q_max),
            self.unit_variance,
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())

    def inverse_transform(self, X):
        self._check_fitted()
        xa, rows, cols = self._normalize(X, dtype=self._np_float())
        out = self._suffixed("inverse_transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())

    @property
    def center_(self):
        return self._to_output(self._suffixed("center")(), (-1,), None, self._np_float())

    @property
    def scale_(self):
        return self._to_output(self._suffixed("scale")(), (-1,), None, self._np_float())


class Normalizer(TransformerMixin, MlrsBase):
    """Per-row L1/L2/max unit-norm rescale. No `inverse_transform` (lossy)."""

    def __init__(self, norm="l2", *, copy=True, output_type="input"):
        self.norm = norm
        self.copy = copy
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().Normalizer(self.norm)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())


class Binarizer(TransformerMixin, MlrsBase):
    """`X > threshold ? 1 : 0`. No `inverse_transform` (lossy)."""

    def __init__(self, *, threshold=0.0, copy=True, output_type="input"):
        self.threshold = threshold
        self.copy = copy
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().Binarizer(float(self.threshold))
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(out, (rows, cols), X, self._np_float())
