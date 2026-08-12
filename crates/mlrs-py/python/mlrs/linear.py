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
from sklearn.base import BaseEstimator as _SkBaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import MultiOutputMixin as _SkMultiOutputMixin
from sklearn.base import RegressorMixin

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
    tol=1e-4, solver='auto', positive=False, random_state=None,
    device='auto')`` — the full ``sklearn.linear_model.Ridge`` parameter
    surface, including ``fit(X, y, sample_weight=...)`` and the ``n_iter_`` /
    ``solver_`` fitted attributes.

    ``device`` (mlrs-only, DEVICE-PARAM-01) pins where the heavy phase runs:
    ``'cpu'`` takes the host arm — no upload of the design and no kernel launch
    — and ``'gpu'`` takes the ``cubecl`` arm; ``'auto'`` (the default) keeps the
    shape/backend heuristic and is the only value that consults the
    ``MLRS_RIDGE_GRAM_HOST`` A/B flag. It is a PREFERENCE: ``solver='lsqr'`` and
    friends have no host ingress, so read ``device_`` for the arm that actually
    ran. See ``crates/mlrs-algos/src/linear/ridge.rs`` for the per-solver
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
        device="auto",
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
        self.device = device
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
            self._device(),
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

    # ``device_`` (the execution arm that actually ran) is inherited from
    # :class:`~mlrs.base.MlrsBase` — it is identical for every estimator that
    # takes the parameter, so it is defined once there rather than per shim.


class _IdentityRegressor(RegressorMixin, _SkBaseEstimator):
    """A regressor whose ``predict`` IS its input.

    ``scoring`` is a scorer, and a scorer's signature is
    ``scorer(estimator, X, y)`` — it wants to call ``predict`` itself. The LOO
    predictions are already computed by the time we score them, so the "X" we
    hand the scorer is the prediction vector and the estimator is this. Exactly
    the trick sklearn's own ``_RidgeGCV._score`` uses, for the same reason.
    """

    def decision_function(self, y_predict):
        return y_predict

    def predict(self, y_predict):
        return y_predict


def _scorer_accepts_sample_weight(scorer, estimator):
    """sklearn's ``BaseSearchCV._check_scorers_accept_sample_weight``.

    A scorer built from a metric that has no ``sample_weight`` parameter
    (``max_error``, say) is NOT given the weights, and sklearn warns about it.
    Both halves are reproduced: silently weighting such a scorer would disagree
    with sklearn, and silently NOT warning would hide a real statistical
    caveat from the caller.
    """
    from inspect import signature

    if hasattr(scorer, "_accept_sample_weight"):
        accept = bool(scorer._accept_sample_weight())
    else:
        accept = "sample_weight" in signature(scorer).parameters
    if not accept:
        import warnings

        warnings.warn(
            f"The scoring {scorer} does not support sample_weight, which may "
            "lead to statistically incorrect results when fitting "
            f"{estimator} with sample_weight. "
        )
    return accept


def _argmax_first(scores):
    """``np.argmax`` with sklearn's ``_RidgeGCV`` tie/NaN semantics.

    sklearn updates its running winner with a strict ``alpha_score >
    best_score``, so the FIRST alpha of a tie wins and a NaN score never wins at
    all (``NaN > x`` is False). ``np.argmax`` agrees on ties but PICKS a NaN, so
    this is written out rather than delegated.
    """
    best = 0
    best_value = None
    for i, s in enumerate(scores):
        if np.isnan(s):
            continue
        if best_value is None or s > best_value:
            best_value, best = s, i
    if best_value is None:
        return 0
    return best


class RidgeCV(_SkMultiOutputMixin, RegressorMixin, MlrsBase):
    """Ridge regression with built-in cross-validation (RIDGECV-01).

    ``RidgeCV(alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, scoring=None,
    cv=None, gcv_mode=None, store_cv_results=False, alpha_per_target=False)`` —
    the full ``sklearn.linear_model.RidgeCV`` parameter surface, including
    ``fit(X, y, sample_weight=...)``, 2-D ``y``, and the ``alpha_`` /
    ``best_score_`` / ``cv_results_`` fitted attributes.

    Two engines, exactly as sklearn has:

    * ``cv=None`` (the DEFAULT) runs generalized (leave-one-out) CV in closed
      form off ONE symmetric eigendecomposition
      (``crates/mlrs-algos/src/linear/ridge_cv.rs``). sklearn re-forms an
      ``n x d`` product per alpha there; mlrs forms the eigenbasis projection
      once, which is why the whole fit is ``O(n*d^2) + O(n_alphas*n*d)`` rather
      than ``O(n_alphas*n*d^2)``.
    * any other ``cv`` runs the explicit ``GridSearchCV(Ridge(), {'alpha':
      alphas}, cv=cv)`` sklearn runs, with the train Gram hoisted out of the
      alpha loop, then refits :class:`Ridge` on the full data at the winner.

    ``scoring`` and ``cv`` may be arbitrary Python objects, so those two stay
    here: the shim resolves the splitter (``mlrs.model_selection.check_cv``) and
    applies the scorer, and Rust owns every ``O(n*d^2)`` pass either way.

    ``gcv_mode`` is accepted and validated (``'auto'`` / ``'svd'`` / ``'eigen'``,
    or ``None`` for ``'auto'``). mlrs derives all three from the SAME
    eigendecomposition — of whichever Gram is smaller — so unlike sklearn they
    are one code path and return identical values; see the Rust module docs for
    the derivation and for the conditioning caveat that comes with it.
    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        cv=None,
        gcv_mode=None,
        store_cv_results=False,
        alpha_per_target=False,
        output_type="input",
    ):
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scoring = scoring
        self.cv = cv
        self.gcv_mode = gcv_mode
        self.store_cv_results = store_cv_results
        self.alpha_per_target = alpha_per_target
        self.output_type = output_type

    # -- parameter resolution ------------------------------------------- #

    def _resolved_alphas(self):
        """``self.alphas`` as a 1-D float array, validated the way sklearn's
        ``_BaseRidgeCV.fit`` validates it.

        The boundary differs by engine and that is sklearn's rule, not a
        convenience: the GCV identity divides by ``alpha``, so ``cv=None``
        requires ``alpha > 0`` (``include_boundaries='neither'``), while an
        explicit ``cv`` refits a real ``Ridge`` per fold and ``alpha=0`` is a
        legitimate (unpenalized) grid point there.
        """
        alphas = np.atleast_1d(np.asarray(self.alphas, dtype=np.float64)).ravel()
        if alphas.size == 0:
            raise ValueError("alphas must contain at least one value")
        strict = self.cv is None
        for i, a in enumerate(alphas):
            name = "alphas" if alphas.size == 1 else f"alphas[{i}]"
            if not np.isfinite(a) or a < 0.0 or (strict and a <= 0.0):
                # sklearn's `check_scalar` phrasing, verbatim, so a caller
                # matching on the message keeps working.
                bound = "> 0.0" if strict else ">= 0.0"
                raise ValueError(f"{name} == {a}, must be {bound}.")
        return alphas

    def _scorer(self):
        """The resolved scorer, or ``None`` for ``scoring=None``.

        sklearn's ``_BaseRidgeCV`` calls ``check_scoring(..., allow_none=True)``
        and then explicitly RESETS the result to ``None`` when ``scoring`` is
        ``None`` ("reset `scorer` variable to original user-intend"), because
        ``check_scoring`` hands back a passthrough scorer rather than ``None``
        there. Skipping that reset silently scores the GCV arm with R² instead
        of ``-mean(looe^2)`` — two different winners on the same data — so the
        ``None`` short-circuit is written first and on purpose.

        On the explicit-``cv`` arm the reset is harmless in the other direction:
        that passthrough scorer IS ``RegressorMixin.score``, i.e. exactly the R²
        the Rust grid engine already computes, so ``scoring=None`` takes the
        fast in-Rust reduction rather than a Python callback per fold.
        """
        if self.scoring is None:
            return None
        from sklearn.metrics import check_scoring

        return check_scoring(self, scoring=self.scoring)

    # -- fit ------------------------------------------------------------- #

    def fit(self, X, y, sample_weight=None):
        alphas = self._resolved_alphas()
        scorer = self._scorer()

        xa, rows, cols = self._normalize(X)
        dtype = LinearRegression._x_float(xa)
        y_arr = np.asarray(y)
        y_ndim = int(y_arr.ndim)
        n_y = int(y_arr.shape[1]) if y_ndim == 2 else 1
        ya = self._normalize_y(y, dtype=dtype)
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=dtype)
        )
        dt = "f32" if dtype is np.float32 else "f64"

        obj = self._ext().RidgeCV(
            [float(a) for a in alphas],
            self.fit_intercept,
            "auto" if self.gcv_mode is None else self.gcv_mode,
        )

        if self.cv is None:
            coef, intercept, alpha_, best_score_, cv_results_ = self._fit_gcv(
                obj, xa, ya, swa, rows, cols, n_y, y_arr, alphas, scorer, dtype,
                sample_weight,
            )
        else:
            if self.store_cv_results:
                raise ValueError(
                    "cv!=None and store_cv_results=True are incompatible"
                )
            if self.alpha_per_target:
                raise ValueError("cv!=None and alpha_per_target=True are incompatible")
            coef, intercept, alpha_, best_score_, cv_results_ = self._fit_grid(
                obj, X, xa, ya, swa, rows, cols, n_y, y_arr, alphas, scorer,
                sample_weight,
            )

        obj.set_fitted(
            [float(v) for v in np.asarray(coef, dtype=np.float64).ravel(order="C")],
            [float(v) for v in np.asarray(intercept, dtype=np.float64).ravel()],
            cols,
            n_y,
            dt,
        )
        self._mlrs_obj = obj
        self._n_targets_ = n_y
        self._y_ndim_ = y_ndim
        self.alpha_ = alpha_
        self.best_score_ = best_score_
        if cv_results_ is not None:
            self.cv_results_ = cv_results_
        self._post_fit(cols)
        return self

    def _fit_gcv(
        self, obj, xa, ya, swa, rows, cols, n_y, y_arr, alphas, scorer, np_dtype,
        sample_weight,
    ):
        """sklearn's ``_RidgeGCV`` arm (``cv=None``)."""
        n_alphas = alphas.size
        want_pred = scorer is not None
        obj.gcv(
            xa, ya, rows, cols, n_y, swa, want_pred,
            bool(self.store_cv_results),
        )

        per_target = bool(self.alpha_per_target) and n_y > 1
        raw = None
        if want_pred or self.store_cv_results:
            # `n_samples x n_alphas x n_targets` row-major out of Rust (rows
            # outermost so each worker owned a contiguous slice).
            raw = np.asarray(obj.gcv_cv_values(), dtype=np.float64).reshape(
                rows, n_alphas, n_y
            )

        if not want_pred:
            scores = np.asarray(obj.gcv_scores(), dtype=np.float64).reshape(
                n_alphas, n_y
            )
            alpha_scores = scores if per_target else scores.mean(axis=1)
        else:
            preds = np.transpose(raw, (1, 0, 2))  # (n_alphas, n, n_y)
            truth = np.asarray(y_arr, dtype=np.float64)
            ident = _IdentityRegressor()
            # `_BaseRidgeCV` forwards `sample_weight` to the scorer here
            # UNCONDITIONALLY (no accepts-check, unlike its GridSearchCV arm),
            # so a scorer without the parameter raises -- which is sklearn's
            # behaviour and therefore this one's.
            sp = {} if sample_weight is None else {
                "sample_weight": np.asarray(sample_weight, dtype=np.float64)
            }
            if per_target:
                alpha_scores = np.empty((n_alphas, n_y), dtype=np.float64)
                for i in range(n_alphas):
                    for t in range(n_y):
                        alpha_scores[i, t] = scorer(
                            ident, preds[i, :, t], truth[:, t], **sp
                        )
            else:
                alpha_scores = np.empty(n_alphas, dtype=np.float64)
                for i in range(n_alphas):
                    p = preds[i] if truth.ndim == 2 else preds[i, :, 0]
                    alpha_scores[i] = scorer(ident, p, truth, **sp)

        coefs = np.asarray(obj.gcv_coefs(), dtype=np.float64).reshape(
            n_alphas, cols, n_y
        )
        if per_target:
            best_idx = np.array(
                [_argmax_first(alpha_scores[:, t]) for t in range(n_y)]
            )
            coef = np.empty((cols, n_y), dtype=np.float64)
            for t in range(n_y):
                coef[:, t] = coefs[best_idx[t], :, t]
            alpha_ = alphas[best_idx].copy()
            best_score_ = np.array(
                [alpha_scores[best_idx[t], t] for t in range(n_y)]
            )
        else:
            best_idx = _argmax_first(alpha_scores)
            coef = coefs[best_idx]
            alpha_ = float(alphas[best_idx])
            best_score_ = float(alpha_scores[best_idx])

        x_offset = np.asarray(obj.gcv_x_offset(), dtype=np.float64)
        y_offset = np.asarray(obj.gcv_y_offset(), dtype=np.float64)
        if self.fit_intercept:
            intercept = y_offset - x_offset @ coef
        else:
            intercept = np.zeros(n_y, dtype=np.float64)

        cv_results_ = None
        if self.store_cv_results:
            # sklearn: (n_samples, n_alphas) for 1-D y, (n_samples, n_targets,
            # n_alphas) for 2-D.
            out = np.transpose(raw, (0, 2, 1))
            if y_arr.ndim == 1:
                out = out[:, 0, :]
            cv_results_ = np.ascontiguousarray(out, dtype=np_dtype)
        return coef, intercept, alpha_, best_score_, cv_results_

    def _fit_grid(
        self, obj, X, xa, ya, swa, rows, cols, n_y, y_arr, alphas, scorer,
        sample_weight,
    ):
        """sklearn's ``GridSearchCV(Ridge(), {'alpha': alphas}, cv=cv)`` arm."""
        from .model_selection import check_cv

        n_alphas = alphas.size
        splitter = check_cv(self.cv, y_arr, classifier=False)
        # `X` is handed to the splitter UNCONVERTED. Splitters read it through
        # `_num_samples`, which already understands numpy / pandas / polars /
        # pyarrow / plain sequences; forcing `np.asarray` first would break a
        # pyarrow Table for no gain (`_normalize` above has already produced the
        # Arrow buffer the Rust side actually consumes).
        splits = [
            (np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64))
            for tr, te in splitter.split(X, y_arr)
        ]
        if not splits:
            raise ValueError("No fits were performed. Was the CV iterator empty?")
        want_pred = scorer is not None
        # sklearn's GridSearchCV forwards `sample_weight` to the TEST-fold
        # scorer too, when the scorer takes it -- which the default regressor
        # scorer (`RegressorMixin.score`) does. Dropping that makes the held-out
        # R^2 unweighted, which moves `best_score_` in the fourth decimal and
        # can move `alpha_` outright.
        weighted = sample_weight is not None and (
            scorer is None or _scorer_accepts_sample_weight(scorer, self)
        )
        obj.grid(
            xa,
            ya,
            rows,
            cols,
            n_y,
            [tr.tolist() for tr, _ in splits],
            [te.tolist() for _, te in splits],
            swa,
            want_pred,
            weighted,
        )

        if not want_pred:
            scores = np.asarray(obj.grid_scores(), dtype=np.float64).reshape(
                len(splits), n_alphas
            )
        else:
            flat = np.asarray(obj.grid_predictions(), dtype=np.float64)
            truth = np.asarray(y_arr, dtype=np.float64)
            ident = _IdentityRegressor()
            scores = np.empty((len(splits), n_alphas), dtype=np.float64)
            sw_all = (
                None
                if not weighted
                else np.asarray(sample_weight, dtype=np.float64)
            )
            base = 0
            for s, (_, te) in enumerate(splits):
                block = flat[
                    base * n_alphas * n_y : (base + te.size) * n_alphas * n_y
                ].reshape(n_alphas, te.size, n_y)
                yt = truth[te]
                sp = {} if sw_all is None else {"sample_weight": sw_all[te]}
                for a in range(n_alphas):
                    p = block[a] if truth.ndim == 2 else block[a, :, 0]
                    scores[s, a] = scorer(ident, p, yt, **sp)
                base += te.size

        mean = scores.mean(axis=0)
        best_idx = _argmax_first(mean)
        alpha_ = float(alphas[best_idx])
        best_score_ = float(mean[best_idx])

        # sklearn's GridSearchCV(refit=True): the reported coef_/intercept_ come
        # from a FRESH Ridge fit on the whole design at the winning alpha, not
        # from any fold. Delegating to `Ridge` reuses its validated host arm
        # rather than re-deriving the same solve here.
        best = Ridge(
            alpha=alpha_, fit_intercept=self.fit_intercept, solver="auto"
        ).fit(X, y_arr, sample_weight=sample_weight)
        coef = np.asarray(best.coef_, dtype=np.float64)
        coef = coef.reshape(n_y, cols).T if coef.ndim == 2 else coef.reshape(cols, 1)
        intercept = np.atleast_1d(
            np.asarray(best.intercept_, dtype=np.float64)
        ).reshape(n_y)
        return coef, intercept, alpha_, best_score_, None

    # -- predict / fitted attributes ------------------------------------- #

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X, ensure_all_finite=False)
        out = self._suffixed("predict")(xa, rows, cols)
        shape = (rows,) if self._n_targets_ == 1 else (rows, self._n_targets_)
        return self._to_output(out, shape, X, self._np_float())

    @property
    def coef_(self):
        flat = self._suffixed("coef")()
        d, t = self.n_features_in_, self._n_targets_
        arr = self._to_output(flat, (d, t), None, self._np_float())
        # sklearn ravels a single-target coef_ and transposes a multi-target one
        # to `(n_targets, n_features)`.
        return arr.reshape(d) if t == 1 else arr.T

    @property
    def intercept_(self):
        self._check_fitted()
        flat = self._suffixed("intercept")()
        arr = np.asarray(flat, dtype=self._np_float())
        if self._n_targets_ == 1 and getattr(self, "_y_ndim_", 1) == 1:
            return arr.reshape(())[()]
        return arr


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
        device="auto",
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
        self.device = device
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
            self._device(),
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
        device="auto",
        output_type="input",
    ):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.device = device
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
                self._device(),
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
        device="auto",
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
        self.device = device
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
            self._device(),
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
        device="auto",
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
        self.device = device
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
            self._device(),
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
        device="auto",
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
        self.device = device
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
            self._device(),
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
