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
from sklearn.base import ClassifierMixin, MetaEstimatorMixin, RegressorMixin

from .base import MlrsBase
# RANSACRegressor's parameter rejections raise the SAME exception type
# `mlrs.model_selection` already defines for the search estimators — sklearn's
# `InvalidParameterError` is private (`sklearn.utils._param_validation`), and
# the shim keeps one mirror of it rather than two.
from .model_selection import InvalidParameterError


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
        """``y`` may be a 1-D length-``n_samples`` target (the ORIGINAL,
        full-eight-solver contract, unchanged) or a 2-D ``n_samples ×
        n_targets`` array (RIDGE-MULTI-TARGET) — matching sklearn's own
        single-vs-multi-output ``Ridge`` split. Multi-target ``y`` is currently
        supported only for the DEFAULT solver (``auto``/``cholesky`` with
        ``positive=False``); every other ``solver``/``positive`` combination
        raises the same typed rejection the Rust side raises, rather than
        silently mis-fitting.
        """
        xa, rows, cols = self._normalize(X)
        dtype = LinearRegression._x_float(xa)
        y_arr = np.asarray(y)
        n_targets = int(y_arr.shape[1]) if y_arr.ndim == 2 and y_arr.shape[1] > 1 else 1
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
        obj.fit(xa, ya, rows, cols, swa, n_targets)
        self._mlrs_obj = obj
        self._n_targets_ = n_targets
        self._post_fit(cols)
        return self

    def predict(self, X):
        if getattr(self, "_n_targets_", 1) > 1:
            xa, rows, cols = self._check_predict_X(X, ensure_all_finite=False)
            out = self._suffixed("predict_multi")(xa, rows, cols)
            return self._to_output(
                out, (rows, self._n_targets_), X, self._np_float()
            )
        return _dense_linear_predict(self, X)

    @property
    def coef_(self):
        if getattr(self, "_n_targets_", 1) > 1:
            # Rust returns `n_features x n_targets` row-major; sklearn's
            # multi-output `coef_` is `(n_targets, n_features)`.
            d, t = self.n_features_in_, self._n_targets_
            flat = self._suffixed("coef_multi")()
            arr = self._to_output(flat, (d, t), None, self._np_float())
            return arr.T
        return self._to_output(
            self._suffixed("coef")(), (-1,), None, self._np_float()
        )

    @property
    def intercept_(self):
        self._check_fitted()
        if getattr(self, "_n_targets_", 1) > 1:
            return self._to_output(
                self._suffixed("intercept_multi")(), (-1,), None, self._np_float()
            )
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


class HuberRegressor(RegressorMixin, MlrsBase):
    """Robust linear regression with a fitted scale (HUBER-01).

    ``HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001,
    warm_start=False, fit_intercept=True, tol=1e-05)`` — the full
    ``sklearn.linear_model.HuberRegressor`` parameter surface, including
    ``fit(X, y, sample_weight=...)`` and the ``coef_`` / ``intercept_`` /
    ``scale_`` / ``n_iter_`` / ``outliers_`` fitted attributes. Every parameter
    is a float, an int or a bool — this estimator has no string-valued
    parameter, so there is no enum to validate at the boundary.

    Unlike :class:`Ridge`, samples whose scaled residual exceeds ``epsilon``
    contribute LINEARLY rather than quadratically, so a handful of gross
    outliers move the fit by a bounded amount. The scale ``sigma`` is fitted
    jointly with the coefficients, which is what makes ``epsilon`` meaningful
    without rescaling ``y`` — see ``crates/mlrs-algos/src/linear/huber.rs``.

    mlrs solves the objective TIGHTER than scikit-learn does: scikit-learn
    leaves scipy's ``factr`` at its default, so its fits stop on the relative-f
    criterion a measured ~1e-6 from the minimizer and ``tol`` cannot move that.
    Expect agreement at that scale rather than at ``tol``.
    """

    def __init__(
        self,
        epsilon=1.35,
        max_iter=100,
        alpha=1e-4,
        warm_start=False,
        fit_intercept=True,
        tol=1e-5,
        output_type="input",
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol
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
        # `warm_start` reuses the PREVIOUS wrapper so the packed
        # `[coef_, intercept_, scale_]` seed it holds survives — sklearn keeps
        # the same object and re-reads its own attributes, and the Rust `fit`
        # consumes the estimator, so the seed has to live on the wrapper.
        obj = getattr(self, "_mlrs_obj", None)
        if not (self.warm_start and obj is not None and obj.is_fitted()):
            obj = self._ext().HuberRegressor(
                self.epsilon,
                self.max_iter,
                self.alpha,
                self.warm_start,
                self.fit_intercept,
                self.tol,
            )
        obj.fit(xa, ya, rows, cols, swa)
        self._mlrs_obj = obj
        self._post_fit(cols)
        if not obj.converged():
            import warnings

            from sklearn.exceptions import ConvergenceWarning

            warnings.warn(
                "lbfgs failed to converge (max_iter=%d). Increase the number "
                "of iterations to improve the convergence." % self.max_iter,
                ConvergenceWarning,
                stacklevel=2,
            )
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
    def scale_(self):
        """sklearn's ``scale_``: the fitted ``sigma``.

        Always a Python float — the joint ``(w, sigma)`` iteration runs in
        ``f64`` whatever the design's storage width, so rounding it to the
        design dtype would only lose information.
        """
        self._check_fitted()
        return self._mlrs_obj.scale()

    @property
    def n_iter_(self):
        """sklearn's ``n_iter_``: L-BFGS iterations, capped at ``max_iter``."""
        self._check_fitted()
        return self._mlrs_obj.n_iter()

    @property
    def outliers_(self):
        """sklearn's ``outliers_``: the boolean mask
        ``|y - X @ coef_ - intercept_| > scale_ * epsilon`` over the TRAINING
        rows.
        """
        self._check_fitted()
        import numpy as np

        return np.asarray(self._mlrs_obj.outliers(), dtype=bool)


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
        n_iter_no_change=5,
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
        self.n_iter_no_change = n_iter_no_change
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
            self.n_iter_no_change,
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
        n_iter_no_change=5,
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
        self.n_iter_no_change = n_iter_no_change
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
            self.n_iter_no_change,
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
        """``(1, n_features)`` binary / ``(n_classes, n_features)`` multiclass.

        sklearn keeps the leading axis even in the binary case, where there is
        a single hyperplane — so the row count comes from the Rust side rather
        than from ``len(classes_)``, which would be wrong for binary (2
        classes, but 1 row).
        """
        self._check_fitted()
        k = self._mlrs_obj.n_coef_rows()
        return self._to_output(
            self._suffixed("coef")(), (k, -1), None, self._np_float()
        )

    @property
    def intercept_(self):
        """``(1,)`` binary / ``(n_classes,)`` multiclass — one per ``coef_`` row."""
        self._check_fitted()
        return self._to_output(
            getattr(self._mlrs_obj, "intercept" + self._suffix())(),
            (-1,),
            None,
            self._np_float(),
        )


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

    def decision_function(self, X):
        """Signed distance to the separating hyperplane(s).

        ``(n_samples,)`` for a binary fit and ``(n_samples, n_classes)`` for the
        one-vs-rest multiclass fit — sklearn's ``LinearSVC`` shape rule, the same
        asymmetry ``coef_`` has.
        """
        xa, rows, cols = self._check_predict_X(X, ensure_all_finite=False)
        out = self._mlrs_obj.decision_function(xa, rows, cols)
        k = self._mlrs_obj.n_coef_rows()
        shape = (rows,) if k == 1 else (rows, k)
        return self._to_output(out, shape, X, self._np_float())

    @property
    def coef_(self):
        """``(1, n_features)`` binary / ``(n_classes, n_features)`` multiclass.

        sklearn keeps the leading axis even in the binary case, where there is a
        single hyperplane — so the row count comes from the Rust side rather than
        from ``len(classes_)``, which would be wrong for binary (2 classes, but
        1 row).
        """
        self._check_fitted()
        k = self._mlrs_obj.n_coef_rows()
        return self._to_output(
            self._suffixed("coef")(), (k, -1), None, self._np_float()
        )

    @property
    def intercept_(self):
        """``(1,)`` binary / ``(n_classes,)`` multiclass — one per ``coef_`` row."""
        self._check_fitted()
        return self._to_output(
            getattr(self._mlrs_obj, "intercept" + self._suffix())(),
            (-1,),
            None,
            self._np_float(),
        )


# =========================================================================== #
# RANSACRegressor (RANSAC-01) — the outlier-EXCLUDING robust regressor.
# =========================================================================== #
#
# RANSAC is a meta-estimator, and that shapes this shim: sklearn's ``estimator=``
# takes ANY duck-typed regressor. mlrs answers that with two fit paths behind one
# public class, chosen per-fit:
#
#   ===============================  =======================================
#   configuration                    path
#   ===============================  =======================================
#   ``estimator=None``, or a         the Rust engine (`_fit_rust`)
#   ``LinearRegression`` with
#   ``positive=False`` — AND a
#   string ``loss``
#   anything else (another           `_fit_python`, a faithful
#   estimator class, ``positive=     transcription of sklearn's own loop
#   True``, a callable ``loss``)
#   ===============================  =======================================
#
# Both paths draw their sub-samples from the SAME numpy ``RandomState`` through
# the SAME ``sample_without_replacement``, so they agree index-for-index with
# sklearn — and with each other — for a given ``random_state``. What differs is
# only who fits the sub-model and who scans the residuals.
#
# ``is_data_valid`` / ``is_model_valid`` are supported on BOTH paths. They are
# RANSAC's own parameters, not the base estimator's, and they fire at most once
# per trial — so the Rust loop calls back up into Python rather than
# surrendering the whole configuration to the slow path (see
# ``crates/mlrs-py/src/estimators/ransac.rs``).
#
# A CALLABLE ``loss`` is the one RANSAC-own parameter the Rust path cannot take:
# its contract is ``loss(y_true, y_pred) -> array``, which needs the
# materialized ``y_pred`` that the fused scan deliberately never writes out.
# That configuration takes the Python path.


def _ransac_base_is_plain_ols(estimator):
    """Is ``estimator`` an ordinary least squares the Rust engine can host?

    ``None`` is sklearn's default (a fresh ``LinearRegression()``). An explicit
    ``LinearRegression`` — sklearn's or mlrs's — qualifies too, but only with
    ``positive=False``: the non-negative variant is a different solver
    (``scipy.optimize.nnls``), not a parameter of this one.

    Returns ``(usable, fit_intercept)``.
    """
    if estimator is None:
        return True, True
    from sklearn.linear_model import LinearRegression as _SkLinearRegression

    if isinstance(estimator, (_SkLinearRegression, LinearRegression)):
        if getattr(estimator, "positive", False):
            return False, True
        return True, bool(getattr(estimator, "fit_intercept", True))
    return False, True


def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """sklearn ``_ransac._dynamic_max_trials``, for the Python fit path.

    The Rust path has its own copy
    (``mlrs_backend::prims::ransac_host::dynamic_max_trials``); this one exists
    so the fallback loop is a transcription of sklearn's and not a call into
    sklearn's private module.
    """
    inlier_ratio = n_inliers / float(n_samples)
    nom = max(np.spacing(1), 1 - probability)
    denom = max(np.spacing(1), 1 - inlier_ratio**min_samples)
    if nom == 1:
        return 0
    if denom == 1:
        return float("inf")
    return abs(float(np.ceil(np.log(nom) / np.log(denom))))


class _PythonRansacHandle:
    """Stand-in for the compiled handle on the Python fit path.

    :class:`~mlrs.base.MlrsBase` keys ``check_is_fitted`` on ``_mlrs_obj`` and
    reads ``dtype()`` for the accessor suffix; the Python path has no compiled
    object, so this supplies both. It is never reached for arithmetic — every
    property that would use it branches on ``_from_rust`` first.
    """

    def __init__(self, estimator, n_targets):
        self._estimator = estimator
        self._n_targets = n_targets

    def dtype(self):
        return "f64"

    def n_targets(self):
        return self._n_targets


class RANSACRegressor(MetaEstimatorMixin, RegressorMixin, MlrsBase):
    """RANdom SAmple Consensus robust regression (RANSAC-01).

    ``RANSACRegressor(estimator=None, *, min_samples=None,
    residual_threshold=None, is_data_valid=None, is_model_valid=None,
    max_trials=100, max_skips=np.inf, stop_n_inliers=np.inf, stop_score=np.inf,
    stop_probability=0.99, loss='absolute_error', random_state=None)`` — the
    full ``sklearn.linear_model.RANSACRegressor`` parameter surface, including
    ``fit(X, y, sample_weight=...)`` and every fitted attribute (``estimator_``
    / ``inlier_mask_`` / ``n_trials_`` / the three ``n_skips_*``).

    Unlike :class:`HuberRegressor`, which bounds an outlier's influence, RANSAC
    EXCLUDES outliers: it searches random ``min_samples``-row sub-samples for
    the model with the largest consensus set and refits on that set alone.

    See ``crates/mlrs-algos/src/linear/ransac.rs`` for the trial loop and
    ``crates/mlrs-backend/src/prims/ransac_host.rs`` for the per-trial scan.
    """

    def __init__(
        self,
        estimator=None,
        *,
        min_samples=None,
        residual_threshold=None,
        is_data_valid=None,
        is_model_valid=None,
        max_trials=100,
        max_skips=np.inf,
        stop_n_inliers=np.inf,
        stop_score=np.inf,
        stop_probability=0.99,
        loss="absolute_error",
        random_state=None,
        output_type="input",
    ):
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.loss = loss
        self.random_state = random_state
        self.output_type = output_type

    # -- validation -------------------------------------------------------- #

    def _validate_ransac_params(self):
        """sklearn's ``_parameter_constraints``, reproduced with its wording.

        Raised HERE rather than in Rust for the same reason sklearn raises it in
        ``_fit_context``: the checks are data-independent and BOTH fit paths
        need them, and only one of the two has a Rust builder to raise from.
        """
        _bad = "The '{}' parameter of RANSACRegressor must be {}. Got {!r} instead."
        if not isinstance(self.loss, str) or self.loss not in (
            "absolute_error",
            "squared_error",
        ):
            if not callable(self.loss):
                raise InvalidParameterError(
                    _bad.format(
                        "loss",
                        "a str among {'absolute_error', 'squared_error'} or a "
                        "callable",
                        self.loss,
                    )
                )
        if not (0.0 <= self.stop_probability <= 1.0):
            raise InvalidParameterError(
                _bad.format(
                    "stop_probability",
                    "a float in the range [0, 1]",
                    self.stop_probability,
                )
            )
        if not isinstance(self.max_trials, (int, np.integer)) or self.max_trials < 1:
            raise InvalidParameterError(
                _bad.format(
                    "max_trials", "an int in the range [1, inf)", self.max_trials
                )
            )
        for name in ("max_skips", "stop_n_inliers"):
            value = getattr(self, name)
            if value < 0:
                raise InvalidParameterError(
                    _bad.format(name, "an int in the range [0, inf)", value)
                )
        if self.min_samples is not None and float(self.min_samples) < 0:
            raise InvalidParameterError(
                _bad.format(
                    "min_samples",
                    "an int in the range [1, inf), a float in the range [0, 1] "
                    "or None",
                    self.min_samples,
                )
            )

    def _loss_fn(self):
        """The per-row loss the Python path applies to ``(y_true, y_pred)``.

        sklearn sums over the target axis for a 2-D ``y`` and takes the bare
        value for a 1-D one — the ``n_targets == 1`` case of the same sum.
        """
        if callable(self.loss):
            return self.loss
        if self.loss == "absolute_error":
            return lambda yt, yp: (
                np.abs(yt - yp) if yt.ndim == 1 else np.sum(np.abs(yt - yp), axis=1)
            )
        return lambda yt, yp: (
            (yt - yp) ** 2 if yt.ndim == 1 else np.sum((yt - yp) ** 2, axis=1)
        )

    # -- fit --------------------------------------------------------------- #

    def fit(self, X, y, sample_weight=None):
        """Fit by RANSAC, then refit the base estimator on the consensus set.

        Raises ``ValueError`` when no consensus set was found at all — sklearn's
        two messages verbatim, chosen by whether the ``max_skips`` budget was
        blown. A consensus found DESPITE blowing that budget is sklearn's
        ``ConvergenceWarning``, and is warned about rather than raised.
        """
        from sklearn.utils import check_consistent_length, check_random_state

        self._validate_ransac_params()
        if y is None:
            raise ValueError(
                f"{type(self).__name__} requires y to be passed, but the target "
                "y is None"
            )

        # `ensure_all_finite=True` here, unlike `predict`. The relocation the
        # dense regressors make (let the Rust pass report the verdict it already
        # computes) does not apply to a FIT that walks the design a hundred
        # times: one validation scan is noise against that, and RANSAC's own
        # first pass is a sub-sample gather that never sees most of the rows —
        # so a NaN could survive into the consensus scan and silently classify
        # its row as an outlier instead of raising.
        xa, rows, cols = self._normalize(X)
        dtype = LinearRegression._x_float(xa)
        y_arr = np.asarray(y)
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            # sklearn's `column_or_1d(warn=True)`: an `(n, 1)` target is a 1-D
            # target that was reshaped by accident, and it says so rather than
            # treating it as a one-column multi-output problem.
            import warnings as _warnings

            from sklearn.exceptions import DataConversionWarning

            _warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples, ), for example "
                "using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )
        n_targets = (
            int(y_arr.shape[1]) if y_arr.ndim == 2 and y_arr.shape[1] > 1 else 1
        )
        check_consistent_length(np.empty(rows), y_arr)
        if sample_weight is not None:
            sw_arr = np.asarray(sample_weight, dtype=np.float64)
            # Both fit paths need this and only one of them has a Rust builder
            # to raise from, so the check lives here. The wording satisfies
            # sklearn's `check_all_zero_sample_weights_error` pattern
            # (`weight.*zero` / `zero.*weight`).
            if sw_arr.size and float(sw_arr.sum()) == 0.0:
                raise ValueError(
                    "RANSACRegressor: sample_weight sums to zero - at least one "
                    "sample must carry a non-zero weight"
                )

        rs = check_random_state(self.random_state)
        usable, base_fit_intercept = _ransac_base_is_plain_ols(self.estimator)

        # Reset any previous fit's materialized estimator; a refit must not
        # serve the old one out of the cache.
        self._estimator_cache = None
        if usable and isinstance(self.loss, str):
            self._fit_rust(
                y, xa, rows, cols, n_targets, dtype, sample_weight, rs,
                base_fit_intercept,
            )
        else:
            self._fit_python(X, y, rows, cols, n_targets, sample_weight, rs)

        self._n_targets_ = n_targets
        self._post_fit(cols)
        if self._exceeded_max_skips:
            import warnings

            from sklearn.exceptions import ConvergenceWarning

            warnings.warn(
                "RANSAC found a valid consensus set but exited early due to "
                "skipping more iterations than `max_skips`. See estimator "
                "attributes for diagnostics (n_skips*).",
                ConvergenceWarning,
            )
        return self

    def _fit_rust(
        self, y, xa, rows, cols, n_targets, dtype, sample_weight, rs,
        base_fit_intercept,
    ):
        """The native path: the whole trial loop runs in Rust.

        ``rs`` is BORROWED — its MT19937 words are lifted into the Rust
        generator and the advanced words are written back — so a caller who
        passed their own ``RandomState`` sees it advance exactly as sklearn's
        would. ``mlrs.model_selection._rust_rng`` is the one implementation of
        that borrow, shared with every splitter.
        """
        from .model_selection import _rust_rng

        ya = self._normalize_y(y, dtype=dtype)
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=np.float64)
        )
        obj = self._ext().RANSACRegressor(
            None if self.min_samples is None else float(self.min_samples),
            None if self.residual_threshold is None else float(self.residual_threshold),
            int(self.max_trials),
            float(self.max_skips),
            float(self.stop_n_inliers),
            float(self.stop_score),
            float(self.stop_probability),
            self.loss,
            base_fit_intercept,
        )
        with _rust_rng(rs) as bridge:
            obj.fit(
                xa,
                ya,
                rows,
                cols,
                n_targets,
                bridge.handle,
                swa,
                self._data_bridge(),
                self._model_bridge(base_fit_intercept),
            )
        self._mlrs_obj = obj
        self._from_rust = True

    def _data_bridge(self):
        """Adapt ``is_data_valid`` to the wrapper's flat-list signature."""
        if self.is_data_valid is None:
            return None

        def bridge(xs, ys, m, d, t):
            x_sub = np.asarray(xs, dtype=np.float64).reshape(m, d)
            y_sub = np.asarray(ys, dtype=np.float64)
            y_sub = y_sub.reshape(m) if t == 1 else y_sub.reshape(m, t)
            return bool(self.is_data_valid(x_sub, y_sub))

        return bridge

    def _model_bridge(self, base_fit_intercept):
        """Adapt ``is_model_valid`` to the wrapper's flat-list signature.

        sklearn hands the predicate the fitted ESTIMATOR OBJECT, so one is built
        here from the coefficients Rust computed: a real
        ``sklearn.linear_model.LinearRegression`` with ``coef_`` /
        ``intercept_`` / ``n_features_in_`` set, which is what a predicate that
        calls ``model.predict(...)`` or reads ``model.coef_`` needs. The Rust
        side deliberately does not model "a Python estimator".
        """
        if self.is_model_valid is None:
            return None
        from sklearn.linear_model import LinearRegression as _SkLinearRegression

        def bridge(coef, intercept, xs, ys, m, d, t):
            model = _SkLinearRegression(fit_intercept=base_fit_intercept)
            c = np.asarray(coef, dtype=np.float64)
            b = np.asarray(intercept, dtype=np.float64)
            model.coef_ = c.reshape(d) if t == 1 else c.reshape(t, d)
            model.intercept_ = b[0] if t == 1 else b
            model.n_features_in_ = d
            x_sub = np.asarray(xs, dtype=np.float64).reshape(m, d)
            y_sub = np.asarray(ys, dtype=np.float64)
            y_sub = y_sub.reshape(m) if t == 1 else y_sub.reshape(m, t)
            return bool(self.is_model_valid(model, x_sub, y_sub))

        return bridge

    def _fit_python(self, X, y, rows, cols, n_targets, sample_weight, rs):
        """sklearn's own trial loop, for a base estimator Rust cannot host.

        Statement for statement with ``sklearn.linear_model._ransac`` — the SAME
        ``sample_without_replacement`` off the SAME generator, so the draw
        sequence still matches index for index. What differs is only that the
        sub-model is the caller's estimator, fitted through its own ``fit``.
        """
        from sklearn.base import clone
        from sklearn.linear_model import LinearRegression as _SkLinearRegression
        from sklearn.utils import check_array
        from sklearn.utils.random import sample_without_replacement
        from sklearn.utils.validation import has_fit_parameter

        xa = check_array(X, dtype=np.float64, ensure_all_finite=False)
        ya = check_array(y, dtype=np.float64, ensure_2d=False)
        estimator = (
            clone(self.estimator)
            if self.estimator is not None
            else _SkLinearRegression()
        )

        if self.min_samples is None:
            if not isinstance(estimator, _SkLinearRegression):
                raise ValueError(
                    "`min_samples` needs to be explicitly set when estimator "
                    "is not a LinearRegression."
                )
            min_samples = cols + 1
        elif 0 < self.min_samples < 1:
            min_samples = int(np.ceil(self.min_samples * rows))
        else:
            min_samples = int(self.min_samples)
        if min_samples > rows:
            raise ValueError(
                "`min_samples` may not be larger than number of samples: "
                f"n_samples = {rows}."
            )

        if self.residual_threshold is None:
            residual_threshold = float(np.median(np.abs(ya - np.median(ya))))
        else:
            residual_threshold = float(self.residual_threshold)
        loss_function = self._loss_fn()

        # sklearn seeds the sub-estimator from the SAME generator when it has a
        # `random_state`; one that has not consumes nothing.
        try:
            estimator.set_params(random_state=rs)
        except ValueError:
            pass

        fit_params = {}
        if sample_weight is not None:
            if not has_fit_parameter(estimator, "sample_weight"):
                raise ValueError(
                    f"{type(estimator).__name__} does not support sample_weight. "
                    "Sample weights are only used for the calibration itself."
                )
            fit_params["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)

        n_inliers_best = 1
        score_best = -np.inf
        inlier_mask_best = None
        n_skips_no_inliers = n_skips_invalid_data = n_skips_invalid_model = 0
        sample_idxs = np.arange(rows)
        n_trials = 0
        max_trials = self.max_trials

        while n_trials < max_trials:
            n_trials += 1
            if (
                n_skips_no_inliers + n_skips_invalid_data + n_skips_invalid_model
            ) > self.max_skips:
                break
            subset_idxs = sample_without_replacement(
                rows, min_samples, random_state=rs
            )
            x_subset, y_subset = xa[subset_idxs], ya[subset_idxs]
            if self.is_data_valid is not None and not self.is_data_valid(
                x_subset, y_subset
            ):
                n_skips_invalid_data += 1
                continue
            sub_params = {k: v[subset_idxs] for k, v in fit_params.items()}
            estimator.fit(x_subset, y_subset, **sub_params)
            if self.is_model_valid is not None and not self.is_model_valid(
                estimator, x_subset, y_subset
            ):
                n_skips_invalid_model += 1
                continue
            residuals_subset = loss_function(ya, estimator.predict(xa))
            inlier_mask_subset = residuals_subset <= residual_threshold
            n_inliers_subset = int(np.sum(inlier_mask_subset))
            if n_inliers_subset < n_inliers_best:
                n_skips_no_inliers += 1
                continue
            inlier_idxs_subset = sample_idxs[inlier_mask_subset]
            score_subset = estimator.score(
                xa[inlier_idxs_subset], ya[inlier_idxs_subset]
            )
            if n_inliers_subset == n_inliers_best and score_subset < score_best:
                continue
            n_inliers_best = n_inliers_subset
            score_best = score_subset
            inlier_mask_best = inlier_mask_subset
            max_trials = min(
                max_trials,
                _dynamic_max_trials(
                    n_inliers_best, rows, min_samples, self.stop_probability
                ),
            )
            if n_inliers_best >= self.stop_n_inliers or score_best >= self.stop_score:
                break

        skips = n_skips_no_inliers + n_skips_invalid_data + n_skips_invalid_model
        if inlier_mask_best is None:
            if skips > self.max_skips:
                raise ValueError(
                    "RANSAC skipped more iterations than `max_skips` without "
                    "finding a valid consensus set. Iterations were skipped "
                    "because each randomly chosen sub-sample failed the passing "
                    "criteria. See estimator attributes for diagnostics "
                    "(n_skips*)."
                )
            raise ValueError(
                "RANSAC could not find a valid consensus set. All `max_trials` "
                "iterations were skipped because each randomly chosen "
                "sub-sample failed the passing criteria. See estimator "
                "attributes for diagnostics (n_skips*)."
            )

        best_idxs = sample_idxs[inlier_mask_best]
        best_params = {k: v[best_idxs] for k, v in fit_params.items()}
        estimator.fit(xa[best_idxs], ya[best_idxs], **best_params)

        self._from_rust = False
        self._py_estimator = estimator
        self._py_inlier_mask = inlier_mask_best
        self._py_counters = (
            n_trials,
            n_skips_no_inliers,
            n_skips_invalid_data,
            n_skips_invalid_model,
            skips > self.max_skips,
        )
        # `MlrsBase._check_fitted` keys on `_mlrs_obj`; the Python path has no
        # compiled handle, so this stand-in carries the contract.
        self._mlrs_obj = _PythonRansacHandle(estimator, n_targets)

    # -- fitted attributes ------------------------------------------------- #

    @property
    def _exceeded_max_skips(self):
        if self._from_rust:
            return bool(self._mlrs_obj.exceeded_max_skips())
        return bool(self._py_counters[4])

    @property
    def estimator_(self):
        """The base estimator refitted on the consensus set.

        On the Rust path there is no Python estimator object to hand back, so
        one is MATERIALIZED from the fitted coefficients — a real
        ``sklearn.linear_model.LinearRegression`` whose ``predict`` / ``score``
        behave as sklearn's do. It is cached, so repeated access is free.
        """
        self._check_fitted()
        if not self._from_rust:
            return self._py_estimator
        cached = getattr(self, "_estimator_cache", None)
        if cached is not None:
            return cached
        from sklearn.linear_model import LinearRegression as _SkLinearRegression

        _, base_fit_intercept = _ransac_base_is_plain_ols(self.estimator)
        est = _SkLinearRegression(fit_intercept=base_fit_intercept)
        d, t = self.n_features_in_, self._n_targets_
        coef = np.asarray(self._mlrs_obj.coef(), dtype=np.float64)
        icept = np.asarray(self._mlrs_obj.intercept(), dtype=np.float64)
        est.coef_ = coef.reshape(d) if t == 1 else coef.reshape(t, d)
        est.intercept_ = icept[0] if t == 1 else icept
        est.n_features_in_ = d
        self._estimator_cache = est
        return est

    @property
    def inlier_mask_(self):
        """sklearn ``inlier_mask_`` — the consensus set of the winning model."""
        self._check_fitted()
        if not self._from_rust:
            return self._py_inlier_mask
        return np.asarray(self._mlrs_obj.inlier_mask(), dtype=bool)

    @property
    def n_trials_(self):
        """sklearn ``n_trials_`` — always ``<= max_trials``."""
        self._check_fitted()
        return (
            self._mlrs_obj.n_trials() if self._from_rust else self._py_counters[0]
        )

    @property
    def n_skips_no_inliers_(self):
        """sklearn ``n_skips_no_inliers_``."""
        self._check_fitted()
        return (
            self._mlrs_obj.n_skips_no_inliers()
            if self._from_rust
            else self._py_counters[1]
        )

    @property
    def n_skips_invalid_data_(self):
        """sklearn ``n_skips_invalid_data_``."""
        self._check_fitted()
        return (
            self._mlrs_obj.n_skips_invalid_data()
            if self._from_rust
            else self._py_counters[2]
        )

    @property
    def n_skips_invalid_model_(self):
        """sklearn ``n_skips_invalid_model_``."""
        self._check_fitted()
        return (
            self._mlrs_obj.n_skips_invalid_model()
            if self._from_rust
            else self._py_counters[3]
        )

    # -- inference --------------------------------------------------------- #

    def predict(self, X):
        """``estimator_.predict(X)`` — sklearn delegates, and so does this.

        On the Rust path the matvec runs through the compiled HOST predict
        rather than through the materialized sklearn object, so a large
        ``predict`` is not paid in numpy (and, per
        [[mlrs-ridge-predict-cuda-vs-cpu]], not paid on a device either).
        """
        self._check_fitted()
        if not self._from_rust:
            return self._py_estimator.predict(np.asarray(X, dtype=np.float64))
        xa, rows, cols = self._check_predict_X(X, ensure_all_finite=False)
        out = self._suffixed("predict")(xa, rows, cols)
        t = self._n_targets_
        shape = (rows,) if t == 1 else (rows, t)
        return self._to_output(out, shape, X, self._np_float())

    def score(self, X, y, sample_weight=None):
        """``estimator_.score(X, y)`` — sklearn's delegation, spelled out.

        Explicit rather than inherited so the R² is computed on THIS class's
        ``predict`` (which routes to the compiled host path) instead of on a
        ``RegressorMixin`` default that would go through ``estimator_``.
        """
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X), sample_weight=sample_weight)

    @property
    def _from_rust(self):
        """Which fit path produced the current fitted state."""
        return getattr(self, "_fit_path_is_rust", False)

    @_from_rust.setter
    def _from_rust(self, v):
        self._fit_path_is_rust = bool(v)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags
