"""Covariance estimator shims (COV-01/COV-02) delegating to ``_mlrs``.

``EmpiricalCovariance`` (MLE / ``ddof=0`` covariance) and ``LedoitWolf``
(shrinkage covariance) subclass :class:`MlrsBase` directly (no
``TransformerMixin`` — covariance estimators are ``fit``-only, they expose fitted
matrices/scalars, not a ``transform``). The sklearn-faithful ``__init__`` stores
every ctor arg verbatim (RESEARCH 06 §Hyperparameter Mapping); ``fit`` is
unsupervised (``y=None``). The ``@property`` accessors map the sklearn-named
attributes (``covariance_`` / ``location_`` / ``precision_`` / ``shrinkage_``) to
the dtype-suffixed ``_mlrs`` accessors via :meth:`MlrsBase._suffixed` /
:meth:`MlrsBase._to_output`; ``shrinkage_`` is a single ``float`` scalar (no
dtype suffix on the wrapper side — the Rust estimator keeps it in ``f64``).
"""

from .base import MlrsBase


class EmpiricalCovariance(MlrsBase):
    """Maximum-likelihood (``ddof=0``) covariance estimator (COV-01)."""

    def __init__(
        self, *, store_precision=True, assume_centered=False, output_type="input"
    ):
        self.store_precision = store_precision
        self.assume_centered = assume_centered
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().EmpiricalCovariance(
            self.store_precision, self.assume_centered
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    @property
    def covariance_(self):
        out = self._suffixed("covariance")()
        return self._to_output(out, (-1, self.n_features_in_), None, self._np_float())

    @property
    def location_(self):
        return self._to_output(
            self._suffixed("location")(), (-1,), None, self._np_float()
        )

    @property
    def precision_(self):
        out = self._suffixed("precision")()
        return self._to_output(out, (-1, self.n_features_in_), None, self._np_float())


class LedoitWolf(MlrsBase):
    """Ledoit-Wolf shrinkage covariance estimator (COV-02)."""

    def __init__(self, *, assume_centered=False, output_type="input"):
        self.assume_centered = assume_centered
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().LedoitWolf(self.assume_centered)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    @property
    def covariance_(self):
        out = self._suffixed("covariance")()
        return self._to_output(out, (-1, self.n_features_in_), None, self._np_float())

    @property
    def location_(self):
        return self._to_output(
            self._suffixed("location")(), (-1,), None, self._np_float()
        )

    @property
    def shrinkage_(self):
        """The optimal Ledoit-Wolf shrinkage intensity in ``[0, 1]`` (scalar).

        ``shrinkage_`` is a single ``float`` — kept in ``f64`` by the Rust
        estimator regardless of the fitted dtype arm — so it is read via the
        un-suffixed ``shrinkage_`` accessor rather than a dtype-suffixed one.
        """
        self._check_fitted()
        return float(self._mlrs_obj.shrinkage_())
