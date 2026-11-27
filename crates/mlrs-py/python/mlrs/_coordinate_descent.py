"""Coordinate-descent paths and joint-target regularized linear regressors."""

import warnings
from copy import copy
from functools import lru_cache
from numbers import Integral, Real

import numpy as np
import pyarrow as pa
from joblib import Parallel, delayed
from scipy import sparse
from sklearn import get_config
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, check_cv
from sklearn.utils import check_random_state
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
)
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from ._regularized_device import device_gram, fit_device


def _real(value, name, lower, upper=np.inf):
    if (
        not isinstance(value, Real)
        or not np.isfinite(value)
        or not lower <= value <= upper
    ):
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}].")
    return float(value)


def _count(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer of at least 1.")
    return int(value)


def _weights(sample_weight, n):
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim == 0:
        weights = np.full(n, weights)
    if (
        weights.shape != (n,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isfinite(weights.sum())
        or weights.sum() <= 0
    ):
        raise ValueError(
            "sample_weight must contain one nonnegative weight per sample and have a positive finite sum."
        )
    return weights


def _options(max_iter, tol, selection):
    if selection not in ("cyclic", "random"):
        raise ValueError("selection must be 'cyclic' or 'random'.")
    return _count(max_iter, "max_iter"), _real(tol, "tol", 0), selection


def _warn_nonconvergence(max_iter, gap):
    warnings.warn(
        f"Objective did not converge after {max_iter} iterations. Dual gap: {gap:.6g}.",
        ConvergenceWarning,
        stacklevel=4,
    )


@lru_cache(maxsize=1)
def _native_cd():
    from . import _load_ext

    try:
        return _load_ext().regularized_cd_gram
    except (ImportError, AttributeError):
        return None


def _use_host_gram(X, options, precompute="auto"):
    rows, features = X.shape
    return (
        options[2] == "cyclic"
        and not sparse.issparse(X)
        and (not isinstance(precompute, (bool, np.bool_)) or precompute)
        and features <= 512
        and (precompute is True or rows >= max(64, features))
    )


def _products(X, Y):
    return X.T @ X, X.T @ Y, np.vdot(Y, Y).real


class _Problem:
    """Centered, weighted least squares with a residual shared across features."""

    def __init__(
        self, X, y, fit_intercept, weights=None, device="cpu", host_gram=False
    ):
        self.n, self.d = X.shape
        self.multi = y.ndim == 2
        self.dtype = X.dtype
        self.sparse = sparse.issparse(X)
        self.device = device
        self.gram = None
        self.scale = (
            None if weights is None else np.sqrt(weights * (self.n / weights.sum()))
        )
        Y = y.reshape(self.n, -1).astype(np.float64)
        self.y_mean = (
            np.average(Y, axis=0, weights=weights)
            if fit_intercept
            else np.zeros(Y.shape[1])
        )
        self.Y = Y - self.y_mean
        self.x_mean = np.zeros(self.d)
        if self.sparse:
            self.X = X.astype(np.float64).tocsc(copy=True)
            self.X.sum_duplicates()
            if fit_intercept:
                average_weights = (
                    np.full(self.n, 1 / self.n)
                    if weights is None
                    else weights / weights.sum()
                )
                self.x_mean = self.X.T @ average_weights
            squared = self.X.power(2)
            if weights is None:
                self.norms = (
                    np.asarray(squared.sum(axis=0)).ravel() - self.n * self.x_mean**2
                )
            else:
                self.norms = squared.T @ (self.scale**2) - self.n * self.x_mean**2
        else:
            order = "C" if device == "gpu" else "F"
            self.X = np.array(X, dtype=np.float64, order=order, copy=True)
            if fit_intercept:
                self.x_mean = np.average(self.X, axis=0, weights=weights)
                self.X -= self.x_mean
            if self.scale is not None:
                self.X *= self.scale[:, None]
            if device == "cpu" and not host_gram:
                self.norms = np.einsum("ij,ij->j", self.X, self.X)
        if self.scale is not None:
            self.Y *= self.scale[:, None]
        self.shape = (self.d, self.Y.shape[1])
        if device == "gpu":
            gram, cross = device_gram(self.X, self.Y, self.dtype)
            # The Gram and target energy must use the same rounded input.
            stored_Y = self.Y.astype(self.dtype, copy=False).astype(
                np.float64, copy=False
            )
            self._set_gram(gram, cross, np.vdot(stored_Y, stored_Y).real)
            self.X = None
            self.Y = None
        elif host_gram:
            self._set_gram(*_products(self.X, self.Y))
        else:
            self.target_energy = np.vdot(self.Y, self.Y).real / self.n
            self.norms = np.maximum(self.norms / self.n, 0)

    def _set_gram(self, gram, cross, energy):
        self.gram = np.ascontiguousarray(gram / self.n)
        self.cross = np.ascontiguousarray(cross / self.n)
        self.norms = np.maximum(self.gram.diagonal(), 0)
        self.target_energy = energy / self.n

    def native_path(self, alphas, ratio, initial, positive, options):
        if self.gram is None or options[2] != "cyclic":
            return None
        native = _native_cd()
        if native is None:
            return None
        max_iter, tol, _ = options
        if initial is None:
            initial = np.zeros(self.shape)
        values, gaps, iterations, converged = native(
            pa.array(self.gram.ravel()),
            pa.array(self.cross.ravel()),
            pa.array(np.ravel(initial)),
            *self.shape,
            self.target_energy,
            alphas.tolist(),
            ratio,
            max_iter,
            tol,
            positive,
        )
        for gap, success in zip(gaps, converged):
            if not success:
                _warn_nonconvergence(max_iter, gap)
        return np.asarray(values).reshape(-1, *self.shape), gaps, iterations

    def column(self, j):
        if not self.sparse:
            return self.X[:, j]
        start, stop = self.X.indptr[j : j + 2]
        column = np.full(self.n, -self.x_mean[j])
        column[self.X.indices[start:stop]] += self.X.data[start:stop]
        if self.scale is not None:
            column *= self.scale
        return column

    def correlate(self, residual):
        if not self.sparse:
            return (self.X.T @ residual) / self.n
        if self.scale is not None:
            residual = self.scale[:, None] * residual
        return (
            self.X.T @ residual - self.x_mean[:, None] * residual.sum(axis=0)
        ) / self.n

    def solve(self, alpha, ratio, initial, positive, options, rng):
        native = self.native_path(np.array([alpha]), ratio, initial, positive, options)
        if native is not None:
            coefs, gaps, iterations = native
            return coefs[0].copy(), gaps[0], iterations[0]
        max_iter, tol, selection = options
        l1 = alpha * ratio
        l2 = alpha * (1 - ratio)
        coef = np.zeros(self.shape) if initial is None else initial.copy()
        coef[self.norms == 0] = 0
        use_gram = self.gram is not None
        if use_gram:
            residual = self.cross - self.gram @ coef
        else:
            predicted = self.X @ coef
            if self.sparse:
                predicted -= self.x_mean @ coef
                if self.scale is not None:
                    predicted *= self.scale[:, None]
            residual = self.Y - predicted
        scale = self.target_energy
        gap = np.inf
        for iteration in range(1, max_iter + 1):
            order = (
                range(self.d)
                if selection == "cyclic"
                else rng.randint(self.d, size=self.d)
            )
            max_update = 0.0
            for j in order:
                if self.norms[j] == 0:
                    continue
                old = coef[j].copy()
                if use_gram:
                    column = self.gram[:, j]
                    target = residual[j] + self.norms[j] * old
                else:
                    column = self.column(j)
                    target = column @ residual / self.n + self.norms[j] * old
                if positive:
                    target = np.maximum(target, 0)
                norm = np.linalg.norm(target)
                coef[j] = (
                    target * ((1 - l1 / norm) / (self.norms[j] + l2))
                    if norm > l1
                    else 0
                )
                update = coef[j] - old
                max_update = max(max_update, np.max(np.abs(update)))
                if np.any(update):
                    residual -= column[:, None] * update

            if iteration == max_iter or max_update <= tol * np.max(np.abs(coef)):
                cross = residual if use_gram else self.correlate(residual)
                correlation = cross - l2 * coef
                projected = np.maximum(correlation, 0) if positive else correlation
                dual_norm = np.linalg.norm(projected, axis=1).max()
                if l1 == 0:
                    # The L2-only objective has a smooth gradient. For ridge,
                    # its squared norm divided by 2*l2 is the primal-dual gap.
                    if l2 > 0:
                        dual_coef = np.maximum(cross, 0) if positive else cross
                        gap = np.sum((dual_coef - l2 * coef) ** 2) / (2 * l2)
                        if positive:
                            gap += np.sum(np.maximum(-cross, 0) * coef)
                        converged = gap <= tol * scale
                    else:
                        gap = 0.0
                        converged = dual_norm <= tol * np.sqrt(scale)
                else:
                    factor = min(1.0, l1 / dual_norm) if dual_norm else 1.0
                    if use_gram:
                        # With R = Y - XW, cross contains X.T @ R / n.
                        residual_target = scale - np.sum(self.cross * coef)
                        residual_energy = max(
                            0.0, residual_target - np.sum(cross * coef)
                        )
                    else:
                        residual_target = np.sum(residual * self.Y) / self.n
                        residual_energy = np.sum(residual**2) / self.n
                    gap = max(
                        0.0,
                        (
                            (1 + factor**2)
                            * (residual_energy + l2 * np.sum(coef**2))
                            / 2
                            + l1 * np.linalg.norm(coef, axis=1).sum()
                            - factor * residual_target
                        ),
                    )
                    converged = gap <= tol * scale
                if converged:
                    break
        else:
            _warn_nonconvergence(max_iter, gap)
        return coef, gap, iteration

    def grid(self, ratio, eps, count, cross=None):
        if ratio == 0:
            raise ValueError(
                "Automatic alpha grids require l1_ratio > 0. Supply explicit alphas."
            )
        if cross is None:
            cross = self.cross if self.gram is not None else self.correlate(self.Y)
        maximum = np.linalg.norm(cross, axis=1).max() / ratio
        resolution = np.finfo(np.float64).resolution
        if maximum <= resolution:
            return np.full(count, resolution)
        return np.geomspace(maximum, maximum * eps, count)

    def intercept(self, coef):
        return self.y_mean - self.x_mean @ coef


def _alpha_values(alphas):
    values = np.asarray(alphas, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(
            "alphas must be a nonempty one-dimensional sequence of finite nonnegative numbers."
        )
    return np.sort(values)[::-1]


def _alpha_grid_count(alphas, n_alphas):
    if alphas is None or (isinstance(alphas, str) and alphas == "warn"):
        if n_alphas is None or n_alphas == "deprecated":
            return 100
        return _count(n_alphas, "n_alphas")
    if isinstance(alphas, Integral):
        return _count(alphas, "alphas")
    return None


def _validate_precompute(precompute, n_features, *, cross_validation=False):
    if isinstance(precompute, (bool, np.bool_)) or (
        isinstance(precompute, str) and precompute == "auto"
    ):
        return
    if not cross_validation and np.shape(precompute) == (n_features, n_features):
        return
    extra = "" if cross_validation else ", or a Gram matrix"
    raise ValueError(f"precompute must be 'auto' or a boolean{extra}.")


def enet_path(
    X,
    y,
    *,
    l1_ratio=0.5,
    eps=1e-3,
    n_alphas=100,
    alphas=None,
    precompute="auto",
    Xy=None,
    copy_X=True,
    coef_init=None,
    verbose=False,
    return_n_iter=False,
    positive=False,
    check_input=True,
    max_iter=1000,
    tol=1e-4,
    random_state=None,
    selection="cyclic",
    sample_weight=None,
    device="cpu",
):
    """Compute an Elastic Net path in descending alpha order.

    Center the input beforehand if an intercept is required. A 2-D target
    selects a joint row-wise L2,1 penalty. With ``device="gpu"``, CubeCL
    computes the dense design's Gram matrix and target cross-products.
    The CPU residual solver accepts ``precompute`` and ``copy_X`` for
    compatibility and does not modify its inputs.
    """
    if check_input:
        X = check_array(X, accept_sparse="csc", dtype=[np.float64, np.float32])
        y = check_array(y, ensure_2d=False, dtype=X.dtype)
    else:
        if not sparse.issparse(X):
            X = np.asarray(X)
        y = np.asarray(y, dtype=X.dtype)
    if y.ndim not in (1, 2) or len(y) != X.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if y.ndim == 2 and positive:
        raise ValueError("positive=True is only supported for one-dimensional y.")
    if check_input:
        _validate_precompute(precompute, X.shape[1])
    weights = _weights(sample_weight, X.shape[0])
    options = _options(max_iter, tol, selection)
    device = fit_device(device, X)
    problem = _Problem(
        X, y, False, weights, device, _use_host_gram(X, options, precompute)
    )
    ratio = _real(l1_ratio, "l1_ratio", 0, 1)
    count = _alpha_grid_count(alphas, n_alphas)
    if count is not None:
        cross = (
            None if Xy is None else np.asarray(Xy).reshape(problem.shape) / problem.n
        )
        alphas = problem.grid(
            ratio,
            _real(eps, "eps", np.finfo(float).tiny, 1),
            count,
            cross,
        )
    else:
        alphas = _alpha_values(alphas)

    coef = None
    if coef_init is not None:
        coef = np.asarray(coef_init, dtype=np.float64)
        expected = problem.shape[::-1] if problem.multi else (problem.d,)
        if coef.shape != expected or not np.isfinite(coef).all():
            raise ValueError(
                f"coef_init must have shape {expected} and contain finite values."
            )
        coef = coef.reshape(problem.shape[1], problem.d).T
        if positive:
            coef = np.maximum(coef, 0)
    rng = check_random_state(random_state)
    native = problem.native_path(alphas, ratio, coef, positive, options)
    if native is not None:
        coefs, gaps, iterations = native
        path = coefs.transpose(2, 1, 0) if problem.multi else coefs[:, :, 0].T
        result = (
            alphas,
            path.astype(problem.dtype),
            np.asarray(gaps, dtype=problem.dtype),
        )
        return (*result, iterations) if return_n_iter else result
    shape = (problem.shape[1], problem.d) if problem.multi else (problem.d,)
    path = np.empty((*shape, len(alphas)), dtype=problem.dtype)
    gaps = np.empty(len(alphas), dtype=problem.dtype)
    iterations = []
    for index, alpha in enumerate(alphas):
        coef, gaps[index], n_iter = problem.solve(
            alpha, ratio, coef, positive, options, rng
        )
        path[..., index] = coef.T if problem.multi else coef[:, 0]
        iterations.append(n_iter)
    return (alphas, path, gaps, iterations) if return_n_iter else (alphas, path, gaps)


def lasso_path(X, y, **kwargs):
    """Compute a Lasso path with an L1 or joint L2,1 penalty."""
    return enet_path(X, y, l1_ratio=1.0, **kwargs)


class _RegularizedRegressor(RegressorMixin, BaseEstimator):
    _multi = False

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = not self._multi
        tags.target_tags.multi_output = self._multi
        tags.target_tags.single_output = not self._multi
        return tags

    def _fit_data(self, X, y):
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False if self._multi else "csc",
            dtype=[np.float64, np.float32],
            multi_output=self._multi,
            y_numeric=True,
        )
        if self._multi and y.ndim != 2:
            raise ValueError(
                "Joint multi-task regression requires y with shape (n_samples, n_targets)."
            )
        return X, y

    def _record(self, problem, coef, gap, iterations):
        self.coef_ = (coef.T if problem.multi else coef[:, 0]).astype(problem.dtype)
        intercept = problem.intercept(coef).astype(problem.dtype)
        self.intercept_ = intercept if problem.multi else intercept[0]
        self.dual_gap_ = problem.dtype.type(gap)
        self.n_iter_ = iterations
        self.device_ = problem.device

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = validate_data(
            self,
            X,
            reset=False,
            dtype=self.coef_.dtype,
            accept_sparse=False if self._multi else "csr",
        )
        return X @ self.coef_.T + self.intercept_


class _MultiTaskRegressor(MultiOutputMixin, _RegularizedRegressor):
    _multi = True
    path = staticmethod(lasso_path)

    @property
    def sparse_coef_(self):
        check_is_fitted(self, "coef_")
        return sparse.csr_matrix(self.coef_)

    def fit(self, X, y):
        options = _options(self.max_iter, self.tol, self.selection)
        alpha = _real(self.alpha, "alpha", 0)
        ratio = _real(getattr(self, "l1_ratio", 1.0), "l1_ratio", 0, 1)
        rng = check_random_state(self.random_state)
        X, y = self._fit_data(X, y)
        device = fit_device(self.device, X)
        problem = _Problem(
            X,
            y,
            self.fit_intercept,
            device=device,
            host_gram=_use_host_gram(X, options),
        )
        initial = None
        if (
            self.warm_start
            and hasattr(self, "coef_")
            and self.coef_.T.shape == problem.shape
        ):
            initial = self.coef_.T.astype(np.float64)
        coef, gap, iterations = problem.solve(
            alpha, ratio, initial, False, options, rng
        )
        self._record(problem, coef, gap, iterations)
        return self


class MultiTaskElasticNet(_MultiTaskRegressor):
    """Elastic Net with a shared feature penalty across targets."""

    path = staticmethod(enet_path)

    def __init__(
        self,
        alpha=1.0,
        *,
        l1_ratio=0.5,
        fit_intercept=True,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        random_state=None,
        selection="cyclic",
        device="cpu",
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection
        self.device = device


class MultiTaskLasso(_MultiTaskRegressor):
    """Lasso with row-wise group shrinkage across targets."""

    def __init__(
        self,
        alpha=1.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=1000,
        tol=1e-4,
        warm_start=False,
        random_state=None,
        selection="cyclic",
        device="cpu",
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.selection = selection
        self.device = device


def _validation_loss(problem, validation, coefs):
    X, Y, weights, gram, cross, energy = validation
    count, _, targets = coefs.shape
    batch = coefs.transpose(1, 0, 2).reshape(problem.d, count * targets)
    if len(X) < problem.d:
        return _residual_losses(X, Y, weights, batch, targets)
    quadratic = (batch * (gram @ batch)).reshape(problem.d, count, targets)
    losses = (
        energy + quadratic.sum(axis=(0, 2)) - 2 * np.einsum("mdk,dk->m", coefs, cross)
    ) / targets
    # When the model is nearly exact, cancellation in the quadratic formula
    # could change grid selection. Score those candidates from the residual.
    near_zero = losses <= 128 * np.finfo(float).eps * energy
    if np.any(near_zero):
        subset = batch.reshape(problem.d, count, targets)[:, near_zero].reshape(
            problem.d, -1
        )
        losses[near_zero] = _residual_losses(X, Y, weights, subset, targets)
    return losses


def _residual_losses(X, Y, weights, coefficients, targets):
    errors = Y[:, None, :] - (X @ coefficients).reshape(len(X), -1, targets)
    return np.average(np.mean(errors**2, axis=2), axis=0, weights=weights)


def _kfold_moments(full, train, test, fit_intercept):
    # KFold's shuffled indices are sorted too. A contiguous validation fold can
    # be a view of the globally centered design.
    index = (
        slice(test[0], test[-1] + 1) if test[-1] - test[0] + 1 == len(test) else test
    )
    X, Y = full.X[index], full.Y[index]
    test_n = len(test)
    train_n = full.n - test_n
    gram, cross, energy = _products(X, Y)
    x_mean = X.mean(axis=0)
    y_mean = Y.mean(axis=0)
    x_shift = -test_n / train_n * x_mean if fit_intercept else np.zeros_like(x_mean)
    y_shift = -test_n / train_n * y_mean if fit_intercept else np.zeros_like(y_mean)

    problem = copy(full)
    problem.n = train_n
    problem.x_mean = full.x_mean + x_shift
    problem.y_mean = full.y_mean + y_shift
    whole_gram = full.gram * full.n
    whole_energy = full.target_energy * full.n
    train_gram = whole_gram - gram - train_n * np.outer(x_shift, x_shift)
    train_cross = full.cross * full.n - cross - train_n * np.outer(x_shift, y_shift)
    train_energy = whole_energy - energy - train_n * np.vdot(y_shift, y_shift).real
    floor = 128 * np.finfo(float).eps
    diag = whole_gram.diagonal()
    if (0 < whole_energy and train_energy <= floor * whole_energy) or np.any(
        (diag > 0) & (train_gram.diagonal() <= floor * diag)
    ):
        # Subtracting nearly equal moments can turn a feature that is
        # constant within the fold into a tiny nonzero diagonal.
        train_X = full.X[train]
        train_Y = full.Y[train]
        if fit_intercept:
            x_shift = train_X.mean(axis=0)
            y_shift = train_Y.mean(axis=0)
            problem.x_mean = full.x_mean + x_shift
            problem.y_mean = full.y_mean + y_shift
            train_X -= x_shift
            train_Y -= y_shift
        train_gram, train_cross, train_energy = _products(train_X, train_Y)
    problem._set_gram(train_gram, train_cross, train_energy)
    problem.X = problem.Y = None
    test_gram = (
        gram / test_n
        - np.outer(x_mean, x_mean)
        + np.outer(x_mean - x_shift, x_mean - x_shift)
    )
    test_cross = (
        cross / test_n
        - np.outer(x_mean, y_mean)
        + np.outer(x_mean - x_shift, y_mean - y_shift)
    )
    test_energy = (
        energy / test_n
        - np.vdot(y_mean, y_mean).real
        + np.vdot(y_mean - y_shift, y_mean - y_shift).real
    )
    return problem, (X - x_shift, Y - y_shift, None, test_gram, test_cross, test_energy)


def _fold_path(
    X,
    y,
    weights,
    train,
    test,
    grid,
    ratios,
    intercept,
    positive,
    options,
    seeds,
    device,
    precompute,
    shared,
):
    validation = None
    if shared is not None:
        problem, validation = _kfold_moments(shared, train, test, intercept)
    else:
        train_weights = None if weights is None else weights[train]
        test_weights = None if weights is None else weights[test]
        X_train, X_test = X[train], X[test]
        problem = _Problem(
            X_train,
            y[train],
            intercept,
            _weights(train_weights, X_train.shape[0]),
            device,
            _use_host_gram(X_train, options, precompute),
        )
        target = y[test].reshape(X_test.shape[0], -1)
        if problem.gram is not None and not problem.sparse and len(test) >= problem.d:
            design = X_test - problem.x_mean
            centered = target - problem.y_mean
            if test_weights is None:
                gram, cross, energy = _products(design, centered)
                mass = len(test)
            else:
                scale = np.sqrt(test_weights[:, None])
                gram, cross, energy = _products(design * scale, centered * scale)
                mass = test_weights.sum()
                if mass == 0:
                    raise ZeroDivisionError("Weights sum to zero, can't be normalized")
            validation = (
                design,
                centered,
                test_weights,
                gram / mass,
                cross / mass,
                energy / mass,
            )
    losses = np.empty(grid.shape)
    for row, (alphas, ratio, seed) in enumerate(zip(grid, ratios, seeds)):
        rng = check_random_state(seed)
        native = problem.native_path(alphas, ratio, None, positive, options)
        if native is not None and validation is not None:
            losses[row] = _validation_loss(problem, validation, native[0])
            continue
        if validation is not None:
            # A source-only checkout can still solve from compact Gram inputs.
            X_val, Y_val, val_weights, _, _, _ = validation
        coef = None
        for index, alpha in enumerate(alphas):
            if native is None:
                coef, _, _ = problem.solve(alpha, ratio, coef, positive, options, rng)
            else:
                coef = native[0][index]
            if validation is not None:
                errors = Y_val - X_val @ coef
                masses = val_weights
            else:
                errors = target - (X_test @ coef + problem.intercept(coef))
                masses = test_weights
            losses[row, index] = np.average(np.mean(errors**2, axis=1), weights=masses)
    return losses


def _split_params(estimator, params):
    if params and not get_config()["enable_metadata_routing"]:
        raise ValueError(
            "Passing parameters to the CV splitter requires metadata routing."
        )
    return process_routing(estimator, "fit", **params).splitter.split


class _RegularizedCV(_RegularizedRegressor):
    path = staticmethod(lasso_path)

    def get_metadata_routing(self):
        return (
            MetadataRouter(owner=type(self).__name__)
            .add(
                splitter=check_cv(self.cv),
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
            .add_self_request(self)
        )

    def _fit_cv(self, X, y, sample_weight, params):
        options = _options(self.max_iter, self.tol, self.selection)
        ratios = np.atleast_1d(getattr(self, "l1_ratio", 1.0))
        if ratios.ndim != 1 or len(ratios) == 0:
            raise ValueError(
                "l1_ratio must be a number or a nonempty sequence of numbers."
            )
        ratios = [_real(ratio, "l1_ratio", 0, 1) for ratio in ratios]
        positive = getattr(self, "positive", False)
        rng = check_random_state(self.random_state)
        splitter = check_cv(self.cv)
        split_params = _split_params(self, params)
        X, y = self._fit_data(X, y)
        weights = _weights(sample_weight, X.shape[0])
        device = fit_device(self.device, X)
        precompute = getattr(self, "precompute", "auto")
        _validate_precompute(precompute, X.shape[1], cross_validation=True)
        host_gram = _use_host_gram(X, options, precompute)
        problem = _Problem(X, y, self.fit_intercept, weights, device, host_gram)
        shared = (
            problem
            if type(splitter) is KFold
            and host_gram
            and weights is None
            and device == "cpu"
            else None
        )
        count = _alpha_grid_count(self.alphas, self.n_alphas)
        if count is not None:
            eps = _real(self.eps, "eps", np.finfo(float).tiny, 1)
            grid = np.array([problem.grid(ratio, eps, count) for ratio in ratios])
        else:
            grid = np.tile(_alpha_values(self.alphas), (len(ratios), 1))
        # CSC makes column validation cheap. CSR makes repeated row slices cheap.
        if sparse.issparse(X):
            X = X.tocsr()
        folds = list(splitter.split(X, y, **split_params))
        if not folds:
            raise ValueError("cv must yield at least one train/test split.")
        seeds = rng.randint(np.iinfo(np.int32).max, size=(len(ratios), len(folds)))
        paths = Parallel(
            n_jobs=self.n_jobs, verbose=int(self.verbose), prefer="threads"
        )(
            delayed(_fold_path)(
                X,
                y,
                weights,
                train,
                test,
                grid,
                ratios,
                self.fit_intercept,
                positive,
                options,
                seeds[:, index],
                device,
                precompute,
                shared,
            )
            for index, (train, test) in enumerate(folds)
        )
        mse = np.asarray(paths).transpose(1, 2, 0)
        best_ratio, best_alpha = np.unravel_index(
            np.argmin(mse.mean(axis=2)), mse.shape[:2]
        )
        alpha = grid[best_ratio, best_alpha]
        ratio = ratios[best_ratio]
        coef, gap, iterations = problem.solve(
            alpha, ratio, None, positive, options, rng
        )
        self._record(problem, coef, gap, iterations)
        self.alpha_ = alpha
        if hasattr(self, "l1_ratio"):
            self.l1_ratio_ = ratio
        self.alphas_ = grid[0] if len(ratios) == 1 or count is None else grid
        self.mse_path_ = mse[0] if len(ratios) == 1 else mse
        return self

    def _set_cv_params(
        self,
        eps,
        n_alphas,
        alphas,
        fit_intercept,
        max_iter,
        tol,
        cv,
        copy_X,
        verbose,
        n_jobs,
        random_state,
        selection,
    ):
        self.eps = eps
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.cv = cv
        self.copy_X = copy_X
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.selection = selection


class LassoCV(_RegularizedCV):
    """Select the Lasso alpha by cross-validation and refit on all samples.

    ``device="gpu"`` forms one Gram matrix and target cross-product per
    dense fold on the device. The CPU residual solver accepts
    ``precompute`` and ``copy_X`` without changing the caller's design.
    """

    def __init__(
        self,
        *,
        eps=1e-3,
        n_alphas=None,
        alphas=None,
        fit_intercept=True,
        precompute="auto",
        max_iter=1000,
        tol=1e-4,
        copy_X=True,
        cv=None,
        verbose=False,
        n_jobs=None,
        positive=False,
        random_state=None,
        selection="cyclic",
        device="cpu",
    ):
        self.device = device
        self.precompute = precompute
        self.positive = positive
        self._set_cv_params(
            eps,
            n_alphas,
            alphas,
            fit_intercept,
            max_iter,
            tol,
            cv,
            copy_X,
            verbose,
            n_jobs,
            random_state,
            selection,
        )

    def fit(self, X, y, sample_weight=None, **params):
        return self._fit_cv(X, y, sample_weight, params)


class ElasticNetCV(LassoCV):
    """Select both alpha and the L1 ratio by cross-validation."""

    path = staticmethod(enet_path)

    def __init__(
        self,
        *,
        l1_ratio=0.5,
        eps=1e-3,
        n_alphas=None,
        alphas=None,
        fit_intercept=True,
        precompute="auto",
        max_iter=1000,
        tol=1e-4,
        cv=None,
        copy_X=True,
        verbose=0,
        n_jobs=None,
        positive=False,
        random_state=None,
        selection="cyclic",
        device="cpu",
    ):
        self.l1_ratio = l1_ratio
        super().__init__(
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            fit_intercept=fit_intercept,
            precompute=precompute,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            copy_X=copy_X,
            verbose=verbose,
            n_jobs=n_jobs,
            positive=positive,
            random_state=random_state,
            selection=selection,
            device=device,
        )


class MultiTaskLassoCV(MultiOutputMixin, _RegularizedCV):
    """Cross-validated joint multi-task Lasso."""

    _multi = True

    def __init__(
        self,
        *,
        eps=1e-3,
        n_alphas=None,
        alphas=None,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        copy_X=True,
        cv=None,
        verbose=False,
        n_jobs=None,
        random_state=None,
        selection="cyclic",
        device="cpu",
    ):
        self.device = device
        self._set_cv_params(
            eps,
            n_alphas,
            alphas,
            fit_intercept,
            max_iter,
            tol,
            cv,
            copy_X,
            verbose,
            n_jobs,
            random_state,
            selection,
        )

    def fit(self, X, y, **params):
        return self._fit_cv(X, y, None, params)


class MultiTaskElasticNetCV(MultiTaskLassoCV):
    """Cross-validated joint multi-task Elastic Net."""

    path = staticmethod(enet_path)

    def __init__(
        self,
        *,
        l1_ratio=0.5,
        eps=1e-3,
        n_alphas=None,
        alphas=None,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-4,
        cv=None,
        copy_X=True,
        verbose=0,
        n_jobs=None,
        random_state=None,
        selection="cyclic",
        device="cpu",
    ):
        self.l1_ratio = l1_ratio
        super().__init__(
            eps=eps,
            n_alphas=n_alphas,
            alphas=alphas,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            cv=cv,
            copy_X=copy_X,
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
            selection=selection,
            device=device,
        )
