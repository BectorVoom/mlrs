"""Linear-model estimator shells (PY-01/PY-02).

LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression. Each is a pure
-Python shell subclassing :class:`MlrsBase` (+ the family sklearn mixin) with a
sklearn-faithful ``__init__`` that stores every ctor arg verbatim under the
SAME name (purity rule — no transformation in ``__init__``; RESEARCH 06
§Hyperparameter Mapping + Pitfall 4). ``fit`` is a placeholder that Plan 04
wires to the Rust ``_mlrs`` extension; it raises ``NotImplementedError`` here.

Note the sklearn name ``C`` for LogisticRegression (Rust field is ``c``) and
the sklearn defaults the shim advertises (LogReg max_iter=100/tol=1e-4 — the
sklearn-named floor, not the solver's internal headroom).
"""

from sklearn.base import ClassifierMixin, RegressorMixin

from .base import MlrsBase


class LinearRegression(RegressorMixin, MlrsBase):
    """Ordinary least squares (LINEAR-01). Plan 04 wires ``fit``."""

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        raise NotImplementedError("mlrs LinearRegression.fit lands in Plan 04")


class Ridge(RegressorMixin, MlrsBase):
    """L2-regularized least squares (LINEAR-02)."""

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        raise NotImplementedError("mlrs Ridge.fit lands in Plan 04")


class Lasso(RegressorMixin, MlrsBase):
    """L1-regularized least squares via coordinate descent (LINEAR-03)."""

    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        raise NotImplementedError("mlrs Lasso.fit lands in Plan 04")


class ElasticNet(RegressorMixin, MlrsBase):
    """Combined L1/L2 coordinate descent (LINEAR-04)."""

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        raise NotImplementedError("mlrs ElasticNet.fit lands in Plan 04")


class LogisticRegression(ClassifierMixin, MlrsBase):
    """Multinomial logistic regression (LINEAR-05).

    sklearn name ``C`` (inverse regularization); the Rust constructor field is
    ``c``. The shim stores it verbatim as ``self.C``.
    """

    def __init__(self, C=1.0, fit_intercept=True, max_iter=100, tol=1e-4):
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        raise NotImplementedError(
            "mlrs LogisticRegression.fit lands in Plan 04"
        )
