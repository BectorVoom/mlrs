"""Least-angle Lasso and model selection by cross-validation or AIC/BIC."""

import warnings
from functools import lru_cache
from numbers import Integral

import numpy as np
import pyarrow as pa
from joblib import Parallel, delayed
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.linalg.lapack import dpocon
from scipy.optimize import nnls
from sklearn.base import MultiOutputMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import check_cv
from sklearn.utils import check_random_state
from sklearn.utils.metadata_routing import MetadataRouter, MethodMapping
from sklearn.utils.validation import validate_data

from ._coordinate_descent import _count, _real, _RegularizedRegressor, _split_params
from ._coordinate_descent import _products as _cross_products
from ._regularized_device import device_gram, fit_device

_EPS = np.finfo(float).eps


def _center(X, y, fit_intercept):
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x_mean = X.mean(axis=0) if fit_intercept else np.zeros(X.shape[1])
    y_mean = y.mean(axis=0) if fit_intercept else np.zeros(y.shape[1:])
    if fit_intercept:
        X = X - x_mean
        y = y - y_mean
    return X, y, x_mean, y_mean


def _gram(X, precompute):
    if isinstance(precompute, (str, bool, np.bool_)):
        if precompute not in ("auto", True, False):
            raise ValueError("precompute must be 'auto', a boolean, or a Gram matrix.")
        if isinstance(precompute, (bool, np.bool_)) and precompute:
            return X.T @ X
        if precompute == "auto" and X.shape[0] > X.shape[1]:
            return X.T @ X
        return None
    G = np.asarray(precompute, dtype=float)
    if G.shape != (X.shape[1], X.shape[1]) or not np.isfinite(G).all():
        raise ValueError(
            "precompute must have shape (n_features, n_features) and be finite."
        )
    return G


def _products(X, y, precompute, device):
    if device == "cpu":
        return _gram(X, precompute), None
    # Knot ordering and rank decisions need accurate inner products even
    # when fitted coefficients will be returned as float32.
    gram, cross = device_gram(X, y, np.float64)
    if not isinstance(precompute, (str, bool, np.bool_)):
        gram = _gram(X, precompute)
    elif precompute not in ("auto", True, False):
        raise ValueError("precompute must be 'auto', a boolean, or a Gram matrix.")
    return gram, cross


@lru_cache(maxsize=1)
def _native_lars():
    from . import _load_ext

    try:
        return _load_ext().regularized_lars_gram
    except (ImportError, AttributeError):
        return None


def _knots(X, y, gram, minimum, max_iter, eps, positive, cross=None, keep_path=True):
    """Follow the active-set equicorrelation direction to each entry or drop."""
    n, d = X.shape
    cross = X.T @ y if cross is None else np.asarray(cross, dtype=float)
    if cross.shape != (d,) or not np.isfinite(cross).all():
        raise ValueError("Xy must contain one finite correlation per feature.")
    native = _native_lars() if gram is not None and d <= 1024 else None
    if native is not None and np.array_equal(gram, gram.T):
        alphas, active, values, skipped = native(
            pa.array(gram.ravel()),
            pa.array(cross),
            n,
            minimum,
            max_iter,
            eps,
            positive,
            keep_path,
        )
        if skipped:
            warnings.warn(
                "A degenerate feature was skipped on the Lasso LARS path.",
                ConvergenceWarning,
                stacklevel=3,
            )
        iterations = len(alphas) - 1
        if keep_path:
            coefficients = np.asarray(values).reshape(-1, d).T.copy()
        else:
            alphas = alphas[-1:]
            coefficients = np.asarray(values).copy()
        return np.asarray(alphas), active, coefficients, iterations
    coef = np.zeros(d)
    correlations = cross.copy()
    available = np.ones(d, dtype=bool)
    active = []
    signs = []
    min_level = minimum * n
    scores = correlations if positive else np.abs(correlations)
    level = max(0.0, scores.max())
    tolerance = 64 * _EPS * max(level, np.finfo(float).tiny)
    if level <= min_level + tolerance:
        return np.array([minimum]), active, coef[:, None] if keep_path else coef, 0
    alphas = [level / n]
    path = [coef.copy()] if keep_path else None
    enter = int(np.argmax(scores))
    dropped = []
    iterations = 0

    while iterations < max_iter and level > min_level + tolerance:
        if enter is not None:
            active.append(enter)
            signs.append(1.0 if positive else np.sign(correlations[enter]))
            available[enter] = False
        if not active:
            break

        if gram is None:
            design = X[:, active]
            active_gram = design.T @ design
        else:
            active_gram = gram[np.ix_(active, active)]
        try:
            factor = cho_factor(active_gram, lower=True, check_finite=False)
            rank_floor = np.sqrt(
                _EPS * max(1, len(active)) * active_gram.diagonal().max()
            )
            if factor[0].diagonal().min() <= max(eps, rank_floor):
                raise LinAlgError("singular active design")
            direction = cho_solve(factor, signs, check_finite=False)
        except LinAlgError:
            if enter is None:
                break
            warnings.warn(
                "A degenerate feature was skipped on the Lasso LARS path.",
                ConvergenceWarning,
                stacklevel=3,
            )
            active.pop()
            signs.pop()
            candidates = np.where(available, scores, -np.inf)
            enter = int(np.argmax(candidates))
            if candidates[enter] < level - tolerance:
                enter = None
            continue

        slopes = (
            X.T @ (design @ direction) if gram is None else gram[:, active] @ direction
        )
        step = level - min_level
        next_enter = None
        next_drop = []
        for sign in (1.0,) if positive else (1.0, -1.0):
            denominator = 1 - sign * slopes
            candidates = available & (denominator > 16 * _EPS)
            numerator = level - sign * correlations
            candidates &= numerator >= -tolerance
            candidates[dropped] &= numerator[dropped] > tolerance
            crossings = np.full(d, np.inf)
            np.divide(
                np.maximum(numerator, 0), denominator, out=crossings, where=candidates
            )
            candidate = int(np.argmin(crossings))
            if crossings[candidate] < step - tolerance:
                step = crossings[candidate]
                next_enter = candidate

        active_coef = coef[active]
        crossing = np.full(len(active), np.inf)
        np.divide(-active_coef, direction, out=crossing, where=direction != 0)
        crossing[crossing <= tolerance] = np.inf
        first_drop = crossing.min()
        if first_drop < step - tolerance:
            step = first_drop
            next_drop = np.flatnonzero(
                np.abs(crossing - first_drop) <= tolerance
            ).tolist()
            next_enter = None

        coef[active] += step * direction
        correlations -= step * slopes
        scores = correlations if positive else np.abs(correlations)
        level = max(min_level, level - step)
        dropped = [active[index] for index in next_drop]
        for index in reversed(next_drop):
            feature = active.pop(index)
            signs.pop(index)
            coef[feature] = 0
            available[feature] = True
        if step > tolerance:
            alphas.append(level / n)
            if keep_path:
                path.append(coef.copy())
            iterations += 1
        enter = next_enter

    if keep_path:
        return np.asarray(alphas), active, np.asarray(path).T, iterations
    return np.asarray(alphas[-1:]), active, coef, iterations


class _LassoLarsBase(_RegularizedRegressor):
    _multi = True

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.target_tags.single_output = True
        return tags

    def _fit_data(self, X, y):
        return validate_data(
            self,
            X,
            y,
            dtype=[np.float64, np.float32],
            multi_output=self._multi,
            y_numeric=True,
        )

    def _options(self):
        if not isinstance(self.max_iter, Integral) or self.max_iter < 0:
            raise ValueError("max_iter must be an integer of at least 0.")
        return self.max_iter, _real(self.eps, "eps", 0), self.positive

    def _fit_at(self, X, y, alpha, Xy, keep_path, device):
        dtype = X.dtype
        X, y, x_mean, y_mean = _center(X, y, self.fit_intercept)
        gram, cross = _products(X, y, self.precompute, device)
        targets = y.reshape(len(y), -1)
        if Xy is not None:
            Xy = np.asarray(Xy)
            if Xy.ndim == 1:
                Xy = Xy[:, None]
            if Xy.shape != (X.shape[1], targets.shape[1]):
                raise ValueError("Xy must have shape (n_features, n_targets).")
        cross = cross if Xy is None else Xy
        options = self._options()
        results = [
            _knots(
                X,
                target,
                gram,
                alpha,
                *options,
                None if cross is None else cross[:, index],
                keep_path=keep_path,
            )
            for index, target in enumerate(targets.T)
        ]
        alphas, active, paths, iterations = map(list, zip(*results))
        coefficients = [path[:, -1] for path in paths] if keep_path else paths
        coef = np.asarray(coefficients, dtype=dtype)
        paths = [path.astype(dtype) for path in paths]
        intercept = np.asarray(y_mean - coef @ x_mean, dtype=dtype)
        single = targets.shape[1] == 1
        self.coef_ = coef[0] if single else coef
        self.intercept_ = intercept[0] if single else intercept
        self.alphas_ = alphas[0] if single else alphas
        self.n_iter_ = iterations[0] if single else iterations
        self.device_ = device
        if keep_path:
            self.active_ = active[0] if single else active
            self.coef_path_ = paths[0] if single else paths
        else:
            vars(self).pop("active_", None)
            vars(self).pop("coef_path_", None)
        return self

    def _set_lars_params(
        self, fit_intercept, verbose, precompute, max_iter, eps, copy_X, positive
    ):
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.precompute = precompute
        self.max_iter = max_iter
        self.eps = eps
        self.copy_X = copy_X
        self.positive = positive


class LassoLars(MultiOutputMixin, _LassoLarsBase):
    """Lasso trained along the least-angle path.

    With ``device="gpu"``, CubeCL forms the Gram matrix and target
    cross-products in float64. The active-set updates run on the host.
    """

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        verbose=False,
        precompute="auto",
        max_iter=500,
        eps=_EPS,
        copy_X=True,
        fit_path=True,
        positive=False,
        jitter=None,
        random_state=None,
        device="cpu",
    ):
        self.device = device
        self.alpha = alpha
        self.fit_path = fit_path
        self.jitter = jitter
        self.random_state = random_state
        self._set_lars_params(
            fit_intercept, verbose, precompute, max_iter, eps, copy_X, positive
        )

    def fit(self, X, y, Xy=None):
        alpha = _real(self.alpha, "alpha", 0)
        X, y = self._fit_data(X, y)
        if self.jitter is not None:
            jitter = _real(self.jitter, "jitter", 0)
            y = y + check_random_state(self.random_state).uniform(0, jitter, y.shape)
        return self._fit_at(
            X,
            y,
            alpha,
            Xy,
            self.fit_path,
            fit_device(self.device, X, require_float64=True),
        )


def _squared_path(X, y, path, gram, cross):
    energy = y @ y
    squares = energy + np.einsum("ij,ij->j", path, gram @ path) - 2 * (cross @ path)
    # Recompute fits near the rounding floor before comparing AIC/BIC.
    near_zero = squares <= 128 * _EPS * energy
    if np.any(near_zero):
        residual = y[:, None] - X @ path[:, near_zero]
        squares[near_zero] = np.einsum("ij,ij->j", residual, residual)
    return squares


def _fold_knots(X, y, train, test, fit_intercept, precompute, options, device):
    design, target, x_mean, y_mean = _center(X[train], y[train], fit_intercept)
    gram, cross = _products(design, target, precompute, device)
    alphas, _, path, _ = _knots(
        design, target, gram, 0, *options, None if cross is None else cross[:, 0]
    )
    design = X[test] - x_mean
    target = y[test] - y_mean
    path = path[:, ::-1]
    if len(test) >= path.shape[0]:
        gram, cross, _ = _cross_products(design, target)
        squares = _squared_path(design, target, path, gram, cross)
        steps = np.diff(path, axis=1)
        distances = np.einsum("ij,ij->j", steps, gram @ steps)
    else:
        residuals = design @ path - target[:, None]
        squares = np.einsum("ij,ij->j", residuals, residuals)
        steps = np.diff(residuals, axis=1)
        distances = np.einsum("ij,ij->j", steps, steps)
    return alphas[::-1], squares / len(test), distances / len(test)


def _interpolated_mse(grid, alphas, squares, distances):
    if len(alphas) == 1:
        return np.full(len(grid), squares[0])
    right = np.clip(np.searchsorted(alphas, grid, side="right"), 1, len(alphas) - 1)
    left = right - 1
    bounded = np.clip(grid, alphas[0], alphas[-1])
    step = (bounded - alphas[left]) / (alphas[right] - alphas[left])
    # Squared norm of the linear interpolation between adjacent residuals.
    return np.maximum(
        (1 - step) * squares[left]
        + step * squares[right]
        - step * (1 - step) * distances[left],
        0,
    )


class LassoLarsCV(_LassoLarsBase):
    """Select the LARS knot by cross-validated prediction error."""

    _multi = False

    def __init__(
        self,
        *,
        fit_intercept=True,
        verbose=False,
        max_iter=500,
        precompute="auto",
        cv=None,
        max_n_alphas=1000,
        n_jobs=None,
        eps=_EPS,
        copy_X=True,
        positive=False,
        device="cpu",
    ):
        self.device = device
        self.cv = cv
        self.max_n_alphas = max_n_alphas
        self.n_jobs = n_jobs
        self._set_lars_params(
            fit_intercept, verbose, precompute, max_iter, eps, copy_X, positive
        )

    def get_metadata_routing(self):
        return MetadataRouter(owner=type(self).__name__).add(
            splitter=check_cv(self.cv),
            method_mapping=MethodMapping().add(caller="fit", callee="split"),
        )

    def fit(self, X, y, **params):
        X, y = self._fit_data(X, y)
        splitter = check_cv(self.cv)
        split_params = _split_params(self, params)
        maximum = _count(self.max_n_alphas, "max_n_alphas")
        options = self._options()
        device = fit_device(self.device, X, require_float64=True)
        precompute = self.precompute
        if not isinstance(precompute, (str, bool, np.bool_)):
            warnings.warn(
                "LassoLarsCV computes separate Gram matrices for each fold.",
                UserWarning,
                stacklevel=2,
            )
            precompute = "auto"
        folds = Parallel(
            n_jobs=self.n_jobs, verbose=int(self.verbose), prefer="threads"
        )(
            delayed(_fold_knots)(
                X, y, train, test, self.fit_intercept, precompute, options, device
            )
            for train, test in splitter.split(X, y, **split_params)
        )
        if not folds:
            raise ValueError("cv must yield at least one train/test split.")
        grid = np.unique(np.concatenate([fold[0] for fold in folds]))
        grid = grid[:: max(1, len(grid) // maximum)]
        mse = np.column_stack([_interpolated_mse(grid, *fold) for fold in folds])
        alpha = grid[mse.mean(axis=1).argmin()]
        self._fit_at(X, y, alpha, None, True, device)
        self.alpha_ = alpha
        self.cv_alphas_ = grid
        self.mse_path_ = mse
        return self


def _least_squares(X, y, gram, cross, positive):
    if positive:
        return nnls(X, y)[0]
    if gram is not None:
        try:
            factor = cho_factor(gram, lower=True, check_finite=False)
            rcond, status = dpocon(factor[0], np.linalg.norm(gram, 1), uplo="L")
            if status == 0 and rcond > np.sqrt(_EPS):
                return cho_solve(factor, cross, check_finite=False)
        except LinAlgError:
            pass
    return np.linalg.lstsq(X, y, rcond=None)[0]


class LassoLarsIC(_LassoLarsBase):
    """Select a LARS knot using the Akaike or Bayesian information criterion."""

    _multi = False

    def __init__(
        self,
        criterion="aic",
        *,
        fit_intercept=True,
        verbose=False,
        precompute="auto",
        max_iter=500,
        eps=_EPS,
        copy_X=True,
        positive=False,
        noise_variance=None,
        device="cpu",
    ):
        self.device = device
        self.criterion = criterion
        self.noise_variance = noise_variance
        self._set_lars_params(
            fit_intercept, verbose, precompute, max_iter, eps, copy_X, positive
        )

    def fit(self, X, y, copy_X=None):
        if self.criterion not in ("aic", "bic"):
            raise ValueError("criterion must be 'aic' or 'bic'.")
        X, y = self._fit_data(X, y)
        dtype = X.dtype
        device = fit_device(self.device, X, require_float64=True)
        X, y, x_mean, y_mean = _center(X, y, self.fit_intercept)
        n, d = X.shape
        gram, cross = _products(X, y, self.precompute, device)
        cross = X.T @ y if cross is None else cross[:, 0]
        # User-supplied Gram data pertains to the path only.
        observed_gram = (
            gram if isinstance(self.precompute, (str, bool, np.bool_)) else None
        )
        variance = self.noise_variance
        if variance is None:
            dof = n - d - int(self.fit_intercept)
            if dof <= 0:
                raise ValueError(
                    "Noise variance cannot be estimated. Supply noise_variance explicitly."
                )
            least_squares = _least_squares(X, y, observed_gram, cross, self.positive)
            variance = np.sum((y - X @ least_squares) ** 2) / dof
        else:
            variance = _real(variance, "noise_variance", np.finfo(float).tiny)
        alphas, _, path, iterations = _knots(X, y, gram, 0, *self._options(), cross)
        if observed_gram is None:
            residual = y[:, None] - X @ path
            squares = np.einsum("ij,ij->j", residual, residual)
        else:
            squares = _squared_path(X, y, path, observed_gram, cross)
        degrees = np.count_nonzero(np.abs(path) > np.finfo(dtype).eps, axis=0)
        penalty = 2 if self.criterion == "aic" else np.log(n)
        # A zero estimated variance is the limit where the residual term dominates.
        safe_variance = max(variance, np.finfo(float).tiny)
        criterion = (
            n * np.log(2 * np.pi * safe_variance)
            + squares / safe_variance
            + penalty * degrees
        )
        best = int(criterion.argmin())
        self.coef_ = path[:, best].astype(dtype)
        self.intercept_ = dtype.type(y_mean - x_mean @ self.coef_)
        self.alphas_ = alphas
        self.alpha_ = alphas[best]
        self.criterion_ = criterion
        self.noise_variance_ = variance
        self.n_iter_ = iterations
        self.device_ = device
        return self
