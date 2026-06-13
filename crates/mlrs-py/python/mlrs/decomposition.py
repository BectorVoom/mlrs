"""Decomposition estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

PCA, TruncatedSVD -> ``TransformerMixin`` (gives ``fit_transform``).
sklearn-faithful ``__init__`` stores every ctor arg verbatim (RESEARCH 06
§Hyperparameter Mapping); v1 PCA requires an explicit int ``n_components`` (no
``None`` / ``'mle'``). ``fit`` is unsupervised (``y=None``); ``transform``
returns a ``rows × n_components`` array. PCA additionally exposes
``inverse_transform`` (TruncatedSVD does not).
"""

from sklearn.base import TransformerMixin

from .base import MlrsBase


class PCA(TransformerMixin, MlrsBase):
    """Principal component analysis, full SVD solver (DECOMP-01)."""

    def __init__(self, n_components, output_type="input"):
        self.n_components = n_components
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().PCA(self.n_components)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        return self

    def transform(self, X):
        xa, rows, cols = self._normalize(X, dtype=self._np_float())
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components), X, self._np_float()
        )

    def inverse_transform(self, Z):
        za, rows, k = self._normalize(Z, dtype=self._np_float())
        out = self._suffixed("inverse_transform")(za, rows, k)
        return self._to_output(out, (rows, -1), Z, self._np_float())

    @property
    def components_(self):
        out = self._suffixed("components")()
        return self._to_output(
            out, (self.n_components, -1), None, self._np_float()
        )

    @property
    def mean_(self):
        return self._to_output(
            self._suffixed("mean")(), (-1,), None, self._np_float()
        )

    @property
    def explained_variance_(self):
        return self._to_output(
            self._suffixed("explained_variance")(),
            (-1,),
            None,
            self._np_float(),
        )

    @property
    def explained_variance_ratio_(self):
        return self._to_output(
            self._suffixed("explained_variance_ratio")(),
            (-1,),
            None,
            self._np_float(),
        )


class TruncatedSVD(TransformerMixin, MlrsBase):
    """Truncated SVD (DECOMP-02)."""

    def __init__(self, n_components=2, output_type="input"):
        self.n_components = n_components
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().TruncatedSVD(self.n_components)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        return self

    def transform(self, X):
        xa, rows, cols = self._normalize(X, dtype=self._np_float())
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components), X, self._np_float()
        )

    @property
    def components_(self):
        out = self._suffixed("components")()
        return self._to_output(
            out, (self.n_components, -1), None, self._np_float()
        )

    @property
    def singular_values_(self):
        return self._to_output(
            self._suffixed("singular_values")(), (-1,), None, self._np_float()
        )

    @property
    def explained_variance_ratio_(self):
        return self._to_output(
            self._suffixed("explained_variance_ratio")(),
            (-1,),
            None,
            self._np_float(),
        )
