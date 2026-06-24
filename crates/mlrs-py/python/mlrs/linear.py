"""Linear-model estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

LinearRegression, Ridge, Lasso, ElasticNet -> ``RegressorMixin``;
LogisticRegression -> ``ClassifierMixin``. Each subclasses :class:`MlrsBase` +
the family sklearn mixin with a sklearn-faithful ``__init__`` storing every ctor
arg verbatim under the SAME name (purity rule — RESEARCH 06 §Hyperparameter
Mapping + Pitfall 4; LogisticRegression exposes sklearn ``C``, the Rust field is
``c``). ``fit`` normalizes via the base, constructs the matching ``_mlrs.Py*``
wrapper, stores the handle on ``self._mlrs_obj`` and returns ``self`` (PY-01).
Fitted-attr properties (``coef_`` / ``intercept_``) raise ``NotFittedError``
before ``fit`` and materialize via the dtype-suffixed wrapper accessor (D-03/D-06).
"""

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from .base import MlrsBase


class LinearRegression(RegressorMixin, MlrsBase):
    """Ordinary least squares (LINEAR-01)."""

    def __init__(self, fit_intercept=True, output_type="input"):
        self.fit_intercept = fit_intercept
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=self._x_float(xa))
        obj = self._ext().LinearRegression(self.fit_intercept)
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()

    @staticmethod
    def _x_float(xa):
        return np.float32 if xa.type.bit_width == 32 else np.float64


class Ridge(RegressorMixin, MlrsBase):
    """L2-regularized least squares (LINEAR-02)."""

    def __init__(self, alpha=1.0, fit_intercept=True, output_type="input"):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
        obj = self._ext().Ridge(self.alpha, self.fit_intercept)
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()


class Lasso(RegressorMixin, MlrsBase):
    """L1-regularized least squares via coordinate descent (LINEAR-03)."""

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        output_type="input",
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
        obj = self._ext().Lasso(
            self.alpha, self.fit_intercept, self.max_iter, self.tol
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()


class ElasticNet(RegressorMixin, MlrsBase):
    """Combined L1/L2 coordinate descent (LINEAR-04)."""

    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        output_type="input",
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
        obj = self._ext().ElasticNet(
            self.alpha,
            self.l1_ratio,
            self.fit_intercept,
            self.max_iter,
            self.tol,
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()


class LogisticRegression(ClassifierMixin, MlrsBase):
    """Multinomial logistic regression (LINEAR-05).

    sklearn name ``C`` (inverse regularization); the Rust ctor field is ``c``.
    The shim stores it verbatim as ``self.C`` (purity rule).
    """

    def __init__(
        self,
        C=1.0,
        fit_intercept=True,
        max_iter=100,
        tol=1e-4,
        output_type="input",
    ):
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
        obj = self._ext().LogisticRegression(
            self.C, self.fit_intercept, self.max_iter, self.tol
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        # classes_ exposed as int32 labels 0..n_classes-1 (v1 contiguous labels).
        self.classes_ = np.arange(obj.n_classes(), dtype=np.int32)
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

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        return self._to_output(
            self._suffixed("intercept")(), (-1,), None, self._np_float()
        )


class MBSGDRegressor(RegressorMixin, MlrsBase):
    """Mini-batch SGD regressor (LINEAR-06).

    sklearn-named ctor params stored verbatim (``seed`` is the Rust field for
    sklearn ``random_state``-style reproducibility; the wrap exposes ``seed``
    directly, matching PyMBSGDRegressor ``#[new]`` at linear.rs:1264-1300).
    """

    def __init__(
        self,
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        epsilon=0.1,
        batch_size=1,
        shuffle=True,
        seed=0,
        output_type="input",
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
        obj = self._ext().MBSGDRegressor(
            self.loss,
            self.penalty,
            self.alpha,
            self.l1_ratio,
            self.fit_intercept,
            self.max_iter,
            self.tol,
            self.learning_rate,
            self.eta0,
            self.power_t,
            self.epsilon,
            self.batch_size,
            self.shuffle,
            self.seed,
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()


class MBSGDClassifier(ClassifierMixin, MlrsBase):
    """Mini-batch SGD classifier (LINEAR-07).

    sklearn-named ctor params stored verbatim (matches PyMBSGDClassifier
    ``#[new]`` at linear.rs:991-1030).
    """

    def __init__(
        self,
        loss="hinge",
        penalty="l2",
        alpha=1e-4,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        learning_rate="optimal",
        eta0=0.01,
        power_t=0.5,
        batch_size=1,
        shuffle=True,
        seed=0,
        output_type="input",
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
        obj = self._ext().MBSGDClassifier(
            self.loss,
            self.penalty,
            self.alpha,
            self.l1_ratio,
            self.fit_intercept,
            self.max_iter,
            self.tol,
            self.learning_rate,
            self.eta0,
            self.power_t,
            self.batch_size,
            self.shuffle,
            self.seed,
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    def predict_proba(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict_proba")(xa, rows, cols)
        n_classes = int(self.classes_.shape[0])
        return self._to_output(out, (rows, n_classes), X, self._np_float())

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()


class LinearSVR(RegressorMixin, MlrsBase):
    """Linear support-vector regression (SVM-02).

    sklearn name ``C`` (the Rust field is ``c``); stored verbatim as ``self.C``
    (purity rule). Matches PyLinearSVR ``#[new]`` at linear.rs:1705-1745.
    """

    def __init__(
        self,
        loss="squared_epsilon_insensitive",
        penalty="l2",
        C=1.0,
        epsilon=0.0,
        intercept_scaling=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        output_type="input",
    ):
        self.loss = loss
        self.penalty = penalty
        self.C = C
        self.epsilon = epsilon
        self.intercept_scaling = intercept_scaling
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
        obj = self._ext().LinearSVR(
            self.loss,
            self.penalty,
            self.C,
            self.epsilon,
            self.intercept_scaling,
            self.fit_intercept,
            self.max_iter,
            self.tol,
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()


class LinearSVC(ClassifierMixin, MlrsBase):
    """Linear support-vector classification (SVM-01).

    sklearn name ``C`` (the Rust field is ``c``); stored verbatim as ``self.C``
    (purity rule). Matches PyLinearSVC ``#[new]`` at linear.rs:1501-1540.
    """

    def __init__(
        self,
        loss="squared_hinge",
        penalty="l2",
        C=1.0,
        intercept_scaling=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        output_type="input",
    ):
        self.loss = loss
        self.penalty = penalty
        self.C = C
        self.intercept_scaling = intercept_scaling
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=LinearRegression._x_float(xa))
        obj = self._ext().LinearSVC(
            self.loss,
            self.penalty,
            self.C,
            self.intercept_scaling,
            self.fit_intercept,
            self.max_iter,
            self.tol,
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()
