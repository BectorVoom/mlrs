"""Decomposition estimator shells (PY-01/PY-02).

PCA, TruncatedSVD. Pure-Python shells subclassing :class:`MlrsBase` +
``TransformerMixin`` (gives ``fit_transform``). sklearn-faithful ``__init__``
stores every ctor arg verbatim (RESEARCH 06 §Hyperparameter Mapping); ``fit``
and ``transform`` are Plan-04 placeholders. v1 requires an explicit int
``n_components`` (no ``None`` / ``'mle'``).
"""

from sklearn.base import TransformerMixin

from .base import MlrsBase


class PCA(TransformerMixin, MlrsBase):
    """Principal component analysis, full SVD solver (DECOMP-01)."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        raise NotImplementedError("mlrs PCA.fit lands in Plan 04")

    def transform(self, X):
        raise NotImplementedError("mlrs PCA.transform lands in Plan 04")


class TruncatedSVD(TransformerMixin, MlrsBase):
    """Truncated SVD (DECOMP-02)."""

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        raise NotImplementedError("mlrs TruncatedSVD.fit lands in Plan 04")

    def transform(self, X):
        raise NotImplementedError(
            "mlrs TruncatedSVD.transform lands in Plan 04"
        )
