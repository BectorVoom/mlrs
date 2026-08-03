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


def _dense_linear_predict(est, X):
    """``predict`` for the four dense linear regressors (OLS/Ridge/Lasso/ENet).

    ``ensure_all_finite=False`` does NOT skip the NaN/inf rejection: the Rust
    predict path reads every element of ``X`` anyway, so it reports the same
    verdict from that pass and raises ``check_array``'s exact ``ValueError``
    itself (``errors.rs::nonfinite_input_err``). ``check_array``'s own scan is a
    second single-threaded trip over the whole matrix and was the single largest
    remaining cost of ``predict`` on the cpu backend — larger than the prediction
    itself. See ``prims/linear_predict.rs`` for the measurements, and
    ``base.MlrsBase._check_predict_X`` for how the error ORDER is preserved.

    All four estimators route here so the relocation lives in one place: a
    sibling left on ``ensure_all_finite=True`` would silently pay the extra pass,
    and one added later with the flag but WITHOUT the Rust-side scan would drop
    the validation entirely.
    """
    xa, rows, cols = est._check_predict_X(X, ensure_all_finite=False)
    out = est._suffixed("predict")(xa, rows, cols)
    return est._to_output(out, (rows,), X, est._np_float())


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
        return _dense_linear_predict(self, X)

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
    """L2-regularized least squares (LINEAR-02).

    ``Ridge(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None,
    tol=1e-4, solver='auto', positive=False, random_state=None)`` — the full
    ``sklearn.linear_model.Ridge`` parameter surface, including
    ``fit(X, y, sample_weight=...)`` and the ``n_iter_`` / ``solver_`` fitted
    attributes. See ``crates/mlrs-algos/src/linear/ridge.rs`` for the per-solver
    routing (and for why ``copy_X`` is a genuine no-op here: mlrs never writes
    into the caller's buffer).
    """

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        solver="auto",
        positive=False,
        random_state=None,
        output_type="input",
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
        self.output_type = output_type

    def _seed(self):
        """``random_state`` -> the ``u64`` seed the Rust ``sag``/``saga`` arm takes.

        ``check_random_state`` accepts all three sklearn spellings (``None`` /
        ``int`` / ``RandomState``); drawing the seed FROM it keeps an ``int``
        ``random_state`` reproducible and a ``RandomState`` instance usable,
        which a plain ``int(...)`` cast would not. ``None`` stays ``None`` so
        the Rust side applies its documented constant seed.
        """
        if self.random_state is None:
            return None
        from sklearn.utils import check_random_state

        return int(check_random_state(self.random_state).randint(0, 2**32 - 1))

    def fit(self, X, y, sample_weight=None):
        xa, rows, cols = self._normalize(X)
        dtype = LinearRegression._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype)
        swa = None if sample_weight is None else self._normalize_y(sample_weight, dtype=dtype)
        obj = self._ext().Ridge(
            self.alpha,
            self.fit_intercept,
            self.copy_X,
            self.max_iter,
            self.tol,
            self.solver,
            self.positive,
            self._seed(),
        )
        obj.fit(xa, ya, rows, cols, swa)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        return _dense_linear_predict(self, X)

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()

    @property
    def n_iter_(self):
        """sklearn's ``n_iter_``: ``None`` for the solvers sklearn leaves unset
        (``cholesky`` / ``svd`` / ``sparse_cg`` / ``lbfgs``)."""
        self._check_fitted()
        return self._mlrs_obj.n_iter()

    @property
    def solver_(self):
        """sklearn's ``solver_``: the solver that actually ran — ``auto``
        already resolved, and reflecting the singular-Gram ``cholesky``->``svd``
        fallback."""
        self._check_fitted()
        return self._mlrs_obj.solver_used()


def _seed_from_random_state(random_state):
    """``random_state`` -> the ``u64`` seed the Rust ``sag``/``saga`` arm takes
    (the :meth:`Ridge._seed` logic, shared with :class:`RidgeClassifier`).

    ``check_random_state`` accepts all three sklearn spellings (``None`` /
    ``int`` / ``RandomState``); drawing the seed FROM it keeps an ``int``
    ``random_state`` reproducible and a ``RandomState`` instance usable, which
    a plain ``int(...)`` cast would not. ``None`` stays ``None`` so the Rust
    side applies its documented constant seed.
    """
    if random_state is None:
        return None
    from sklearn.utils import check_random_state

    return int(check_random_state(random_state).randint(0, 2**32 - 1))


class RidgeClassifier(ClassifierMixin, MlrsBase):
    """Ridge regression used as a classifier (LINEAR-07).

    ``RidgeClassifier(alpha=1.0, fit_intercept=True, copy_X=True,
    max_iter=None, tol=1e-4, class_weight=None, solver='auto',
    positive=False, random_state=None)`` — the full
    ``sklearn.linear_model.RidgeClassifier`` parameter surface. The target is
    encoded as ``{-1, +1}`` (one-hot per class for K>2, sklearn's
    ``LabelBinarizer(neg_label=-1, pos_label=1)`` convention) and fitted as a
    multi-output Ridge regression; ``predict`` reads the sign (binary) or the
    ``argmax`` (multiclass) of ``decision_function``. Unlike
    :class:`LogisticRegression`, there is no ``predict_proba`` — sklearn's own
    ``RidgeClassifier`` does not expose one either.

    See ``crates/mlrs-algos/src/linear/ridge_classifier.rs`` for the cpu
    shared-Gram fast path (the whole point of a dedicated estimator rather
    than a `Ridge`-per-class-column loop) and for why ``copy_X`` is a genuine
    no-op here (mlrs never writes into the caller's buffer).
    """

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        class_weight=None,
        solver="auto",
        positive=False,
        random_state=None,
        output_type="input",
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
        self.output_type = output_type

    def fit(self, X, y, sample_weight=None):
        xa, rows, cols = self._normalize(X)
        dtype = LinearRegression._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype)
        swa = None if sample_weight is None else self._normalize_y(sample_weight, dtype=dtype)
        obj = self._ext().RidgeClassifier(
            self.alpha,
            self.fit_intercept,
            self.copy_X,
            self.max_iter,
            self.tol,
            self.class_weight,
            self.solver,
            self.positive,
            _seed_from_random_state(self.random_state),
        )
        obj.fit(xa, ya, rows, cols, swa)
        self._mlrs_obj = obj
        self._post_fit(cols)
        # classes_ are the core's DISTINCT sorted training labels, so a
        # non-contiguous target (e.g. {0, 2}) round-trips through predict
        # (WR-01, the LogisticRegression precedent).
        self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X, ensure_all_finite=False)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    def decision_function(self, X):
        """Confidence scores: length ``rows`` for binary (`>0` predicts
        ``classes_[1]``), or ``rows x n_classes`` for multiclass (`argmax`
        predicts), matching sklearn's own squeeze."""
        xa, rows, cols = self._check_predict_X(X, ensure_all_finite=False)
        out = self._suffixed("decision_function")(xa, rows, cols)
        n_targets = self._mlrs_obj.n_targets()
        shape = (rows,) if n_targets == 1 else (rows, n_targets)
        return self._to_output(out, shape, X, self._np_float())

    @property
    def coef_(self):
        n_targets = self._mlrs_obj.n_targets()
        shape = (-1,) if n_targets == 1 else (n_targets, -1)
        return self._to_output(self._suffixed("coef")(), shape, None, self._np_float())

    @property
    def intercept_(self):
        # sklearn's own `RidgeClassifier.intercept_` is an ndarray of shape
        # (1,) for binary and (n_classes,) for multiclass — EXCEPT when
        # `fit_intercept=False`, where `_set_intercept` stores the bare Python
        # scalar `0.0` regardless of `n_targets` (a shape quirk sklearn does
        # not special-case away; every target's intercept is 0 either way).
        if not self.fit_intercept:
            return 0.0
        return self._to_output(
            self._suffixed("intercept")(), (-1,), None, self._np_float()
        )

    @property
    def n_iter_(self):
        """sklearn's ``n_iter_``: ``None`` unless the resolved solver is
        ``lsqr`` / ``sag`` / ``saga`` (in which case it is length
        ``n_targets``)."""
        self._check_fitted()
        v = self._mlrs_obj.n_iter()
        return None if v is None else np.asarray(v)

    @property
    def solver_(self):
        """sklearn's ``solver_`` — the solver that actually ran."""
        self._check_fitted()
        return self._mlrs_obj.solver_used()


class BayesianRidge(RegressorMixin, MlrsBase):
    """Bayesian ridge regression with evidence-maximized precisions (LINEAR-06).

    ``BayesianRidge(max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6,
    lambda_1=1e-6, lambda_2=1e-6, alpha_init=None, lambda_init=None,
    compute_score=False, fit_intercept=True, copy_X=True, verbose=False)`` —
    the full ``sklearn.linear_model.BayesianRidge`` parameter surface, including
    ``fit(X, y, sample_weight=...)``, ``predict(X, return_std=True)`` and the
    ``alpha_`` / ``lambda_`` / ``sigma_`` / ``scores_`` / ``n_iter_`` /
    ``X_offset_`` / ``X_scale_`` fitted attributes.

    Unlike :class:`Ridge`, the L2 penalty here is not a hyperparameter — it is
    the fitted ratio ``lambda_ / alpha_``, re-estimated at every iteration
    alongside the noise precision. See
    ``crates/mlrs-algos/src/linear/bayesian_ridge.rs`` for the eigenbasis
    iteration that keeps each step ``O(n_features)`` (and for why ``copy_X`` is
    a genuine no-op here: mlrs never writes into the caller's buffer).
    """

    def __init__(
        self,
        max_iter=300,
        tol=1e-3,
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
        output_type="input",
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose
        self.output_type = output_type

    def fit(self, X, y, sample_weight=None):
        xa, rows, cols = self._normalize(X)
        dtype = LinearRegression._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype)
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=dtype)
        )
        obj = self._ext().BayesianRidge(
            self.max_iter,
            self.tol,
            self.alpha_1,
            self.alpha_2,
            self.lambda_1,
            self.lambda_2,
            self.alpha_init,
            self.lambda_init,
            self.compute_score,
            self.fit_intercept,
            self.copy_X,
            self.verbose,
        )
        obj.fit(xa, ya, rows, cols, swa)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X, return_std=False):
        """``predict(X)``, or ``(mean, std)`` when ``return_std``.

        The mean shares the four other dense regressors' no-upload path. The
        standard deviation is a SECOND call rather than a fused one because
        sklearn returns the mean whether or not ``return_std`` is set, so the
        common case must not pay for ``sigma_``'s quadratic form.
        """
        mean = _dense_linear_predict(self, X)
        if not return_std:
            return mean
        xa, rows, cols = self._check_predict_X(X, ensure_all_finite=False)
        std = self._suffixed("predict_std")(xa, rows, cols)
        import numpy as np

        return mean, self._to_output(std, (rows,), X, np.float64)

    @property
    def coef_(self):
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        return getattr(self._mlrs_obj, "intercept" + self._suffix())()

    @property
    def alpha_(self):
        """sklearn's ``alpha_``: the estimated precision of the noise."""
        self._check_fitted()
        return self._mlrs_obj.alpha_prec()

    @property
    def lambda_(self):
        """sklearn's ``lambda_``: the estimated precision of the weights."""
        self._check_fitted()
        return self._mlrs_obj.lambda_prec()

    @property
    def sigma_(self):
        """sklearn's ``sigma_``: the ``(n_features, n_features)`` posterior
        covariance of the weights.

        Always ``float64`` — it is accumulated in ``f64`` on both fitted arms
        (the evidence iteration does not run at the design's storage width).
        """
        self._check_fitted()
        import numpy as np

        d = self.n_features_in_
        return np.asarray(self._mlrs_obj.sigma(), dtype=np.float64).reshape(d, d)

    @property
    def scores_(self):
        """sklearn's ``scores_``: the log marginal likelihood at each iteration
        plus one final value, or ``None`` when ``compute_score`` is False (which
        is how sklearn leaves the attribute absent)."""
        self._check_fitted()
        import numpy as np

        s = self._mlrs_obj.scores()
        return np.asarray(s, dtype=np.float64) if s else None

    @property
    def n_iter_(self):
        """sklearn's ``n_iter_``: evidence iterations actually run."""
        self._check_fitted()
        return self._mlrs_obj.n_iter()

    @property
    def X_offset_(self):
        """sklearn's ``X_offset_``: the (possibly weighted) column means removed
        before the fit; zeros when ``fit_intercept=False``."""
        self._check_fitted()
        import numpy as np

        return np.asarray(self._mlrs_obj.x_offset(), dtype=np.float64)

    @property
    def X_scale_(self):
        """sklearn's ``X_scale_``: all ones (the attribute outlived the removed
        ``normalize`` parameter, and ``_set_intercept`` still divides by it)."""
        self._check_fitted()
        import numpy as np

        return np.asarray(self._mlrs_obj.x_scale(), dtype=np.float64)


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
        return _dense_linear_predict(self, X)

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
        return _dense_linear_predict(self, X)

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
        # A LinearSVR prediction is the same `X·coef_ + intercept_` matvec the
        # dense regressors compute, so it takes the same no-upload / no-list
        # path (and the same relocated NaN/inf scan).
        return _dense_linear_predict(self, X)

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
        # `ensure_all_finite=False` relocates the NaN/inf rejection into the
        # Rust call rather than dropping it — the reasoning in
        # :func:`_dense_linear_predict` applies verbatim, because a LinearSVC
        # decision function IS that same matvec over X and only adds the
        # sign -> ``classes_`` lookup on top (``linear.rs::predict_labels``).
        xa, rows, cols = self._check_predict_X(X, ensure_all_finite=False)
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
