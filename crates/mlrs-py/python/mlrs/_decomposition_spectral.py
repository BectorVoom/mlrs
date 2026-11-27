"""Spectral decompositions and independent component analysis."""

import warnings

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._decomposition_common import (
    DecompositionTransformer,
    choice,
    count,
    flip_vectors,
    matrix,
)


def randomized_svd(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose="auto",
    flip_sign=True,
    random_state=None,
    svd_lapack_driver="gesdd",
):
    """Approximate a truncated SVD with a randomized range finder."""
    M = matrix(M, accept_sparse=True)
    k = count(n_components, "n_components")
    extra = count(n_oversamples, "n_oversamples", minimum=0)
    steps = (
        (7 if k < 0.1 * min(M.shape) else 4)
        if n_iter == "auto"
        else count(n_iter, "n_iter", minimum=0)
    )
    normalizer = choice(
        power_iteration_normalizer,
        "power_iteration_normalizer",
        ("auto", "none", "LU", "QR"),
    )
    if normalizer == "auto":
        normalizer = "LU" if steps > 2 else "none"
    transposed = M.shape[0] < M.shape[1] if transpose == "auto" else transpose
    if transposed:
        M = M.T
    Q = check_random_state(random_state).standard_normal((M.shape[1], k + extra))
    Q = Q.astype(M.dtype, copy=False)
    for _ in range(steps):
        Q = _range_normalize(M @ Q, normalizer)
        Q = _range_normalize(M.T @ Q, normalizer)
    Q = linalg.qr(M @ Q, mode="economic")[0]
    U, singular, Vt = linalg.svd(
        (M.T @ Q).T, full_matrices=False, lapack_driver=svd_lapack_driver
    )
    U = Q @ U
    if flip_sign:
        U, Vt = flip_vectors(U, Vt, use_u=not transposed)
    if transposed:
        return Vt[:k].T, singular[:k], U[:, :k].T
    return U[:, :k], singular[:k], Vt[:k]


def _range_normalize(X, mode):
    if mode == "QR":
        return linalg.qr(X, mode="economic")[0]
    if mode == "LU":
        return linalg.lu(X, permute_l=True)[0]
    return X


class KernelPCA(DecompositionTransformer):
    """Principal component analysis in a centered kernel feature space."""

    _accept_sparse = True

    def __init__(
        self,
        n_components=None,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        random_state=None,
        copy_X=True,
        n_jobs=None,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    def _kernel(self, X, Y=None):
        from .metrics.pairwise import pairwise_kernels

        if self.kernel == "precomputed":
            return X.toarray() if sparse.issparse(X) else X
        if self.kernel == "linear":
            result = X @ (X if Y is None else Y).T
            return result.toarray() if sparse.issparse(result) else result
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}
        if sparse.issparse(X):
            X = X.toarray()
        if sparse.issparse(Y):
            Y = Y.toarray()
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def fit(self, X, y=None):
        X = self._fit_matrix(X)
        n = X.shape[0]
        k = (
            n
            if self.n_components is None
            else min(n, count(self.n_components, "n_components"))
        )
        solver = choice(
            self.eigen_solver, "eigen_solver", ("auto", "dense", "arpack", "randomized")
        )
        if self.fit_inverse_transform and self.kernel == "precomputed":
            raise ValueError("An inverse transform requires an explicit kernel.")
        if (
            self.alpha < 0
            or self.tol < 0
            or (self.gamma is not None and self.gamma < 0)
        ):
            raise ValueError("alpha, gamma, and tol must be non-negative.")
        self.gamma_ = 1 / X.shape[1] if self.gamma is None else self.gamma
        if solver == "auto":
            solver = "arpack" if n > 200 and k < 10 else "dense"
        linear = None
        if (
            self.kernel == "linear"
            and self.eigen_solver in ("auto", "dense")
            and not sparse.issparse(X)
            and k <= X.shape[1]
            and n >= 2 * X.shape[1]
        ):
            linear = self._linear_eigenpairs(X, k)
        if linear is None:
            K = self._kernel(X)
            if K.shape != (n, n):
                raise ValueError("The precomputed training kernel must be square.")
            self._kernel_mean = K.mean(axis=0)
            self._kernel_grand_mean = self._kernel_mean.mean()
            K = self._center(K)
        if linear is not None:
            values, vectors = linear
        elif solver == "arpack" and k < n:
            initial = check_random_state(self.random_state).uniform(-1, 1, n)
            values, vectors = eigsh(
                K, k, which="LA", tol=self.tol, maxiter=self.max_iter, v0=initial
            )
        elif solver == "randomized":
            vectors, _, _ = randomized_svd(
                K, k, n_iter=self.iterated_power, random_state=self.random_state
            )
            values, basis = linalg.eigh(vectors.T @ K @ vectors)
            vectors = vectors @ basis
        else:
            values, vectors = linalg.eigh(K, subset_by_index=(n - k, n - 1))
        scale = max(values.max(), 0)
        relative, absolute = (
            (5e-3, 1e-6) if values.dtype == np.float32 else (1e-5, 1e-10)
        )
        if values.max() < 0 or values.min() < -max(absolute, relative * scale):
            raise ValueError("The kernel matrix has significant negative eigenvalues.")
        order = np.argsort(values)[::-1]
        values = values[order]
        values[values < (2e-7 if values.dtype == np.float32 else 1e-12) * scale] = 0
        np.maximum(values, 0, out=values)
        vectors = vectors[:, order]
        vectors, _ = flip_vectors(vectors, vectors.T, use_u=True)
        if self.remove_zero_eig or self.n_components is None:
            vectors = vectors[:, values > 0]
            values = values[values > 0]
        if self.fit_inverse_transform:
            if sparse.issparse(X):
                raise ValueError("The inverse transform requires dense training data.")
            projected = vectors * np.sqrt(values)
            inverse_kernel = self._kernel(projected)
            inverse_kernel.flat[:: n + 1] += self.alpha
            self.dual_coef_ = linalg.solve(inverse_kernel, X, assume_a="pos")
            self.X_transformed_fit_ = projected
        else:
            self.__dict__.pop("dual_coef_", None)
            self.__dict__.pop("X_transformed_fit_", None)
        self.eigenvalues_, self.eigenvectors_ = values, vectors
        self.X_fit_ = X.copy() if self.copy_X else X
        self._record_fit(X, len(values))
        return self

    def _linear_eigenpairs(self, X, k):
        # Centered X Xᵀ and Xᵀ X share their nonzero eigenvalues. Recover the
        # sample-space eigenvectors without constructing an n-by-n kernel.
        mean = X.mean(axis=0)
        centered = X - mean
        width = X.shape[1]
        values, vectors = linalg.eigh(
            centered.T @ centered, subset_by_index=(width - k, width - 1)
        )
        relative_floor = 2e-7 if values.dtype == np.float32 else 1e-12
        if values[0] <= relative_floor * values[-1]:
            # Sample-space null vectors cannot be recovered by division.
            return None
        self._kernel_mean = X @ mean
        self._kernel_grand_mean = self._kernel_mean.mean()
        return values, (centered @ vectors) / np.sqrt(values)

    def _center(self, K):
        return (
            K
            - self._kernel_mean
            - K.mean(axis=1, keepdims=True)
            + self._kernel_grand_mean
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.eigenvectors_ * np.sqrt(self.eigenvalues_)

    def transform(self, X):
        X = self._validate_transform(X)
        scale = np.sqrt(self.eigenvalues_)
        scaled = np.divide(
            self.eigenvectors_,
            scale,
            out=np.zeros_like(self.eigenvectors_),
            where=scale > 0,
        )
        return self._center(self._kernel(X, self.X_fit_)) @ scaled

    def inverse_transform(self, X):
        check_is_fitted(self, "X_fit_")
        if not hasattr(self, "dual_coef_"):
            raise ValueError("Set fit_inverse_transform=True before fitting KernelPCA.")
        return self._kernel(matrix(X), self.X_transformed_fit_) @ self.dual_coef_


class FactorAnalysis(DecompositionTransformer):
    """Maximum-likelihood Gaussian factor analysis with diagonal noise."""

    def __init__(
        self,
        n_components=None,
        *,
        tol=0.01,
        copy=True,
        max_iter=1000,
        noise_variance_init=None,
        svd_method="randomized",
        iterated_power=3,
        rotation=None,
        random_state=0,
    ):
        self.n_components = n_components
        self.tol = tol
        self.copy = copy
        self.max_iter = max_iter
        self.noise_variance_init = noise_variance_init
        self.svd_method = svd_method
        self.iterated_power = iterated_power
        self.rotation = rotation
        self.random_state = random_state

    def fit(self, X, y=None):
        X = self._fit_matrix(X).astype(np.float64, copy=False)
        n, p = X.shape
        k = (
            p
            if self.n_components is None
            else count(self.n_components, "n_components", minimum=0)
        )
        steps = count(self.max_iter, "max_iter")
        choice(self.svd_method, "svd_method", ("lapack", "randomized"))
        choice(self.rotation, "rotation", (None, "varimax", "quartimax"))
        mean = X.mean(axis=0)
        Xc = X.copy() if self.copy or not X.flags.writeable else X
        Xc -= mean
        variance = np.mean(Xc**2, axis=0)
        noise = (
            np.ones(p)
            if self.noise_variance_init is None
            else np.asarray(self.noise_variance_init, dtype=float).copy()
        )
        if noise.shape != (p,) or not np.isfinite(noise).all() or (noise <= 0).any():
            raise ValueError(
                "noise_variance_init must contain one positive finite value per feature."
            )
        rng = check_random_state(self.random_state)
        loglike = []
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")
        for _ in range(steps):
            root = np.sqrt(noise) + 1e-12
            adjusted = Xc / (root * np.sqrt(n))
            if self.svd_method == "lapack" or k == 0:
                _, spectrum, vectors = linalg.svd(adjusted, full_matrices=False)
                singular = spectrum[:k]
                residual = np.sum(spectrum[k:] ** 2)
            else:
                _, singular, vectors = randomized_svd(
                    adjusted, min(k, n, p), n_iter=self.iterated_power, random_state=rng
                )
                residual = np.sum(adjusted**2) - np.sum(singular**2)
            spectral = singular**2
            loadings = (
                np.sqrt(np.maximum(spectral - 1, 0))[:, None] * vectors[:k] * root
            )
            loglike.append(
                -n
                / 2
                * (
                    p * np.log(2 * np.pi)
                    + k
                    + np.log(spectral).sum()
                    + residual
                    + np.log(noise).sum()
                )
            )
            if len(loglike) > 1 and (
                loglike[-1] == loglike[-2] or loglike[-1] - loglike[-2] < self.tol
            ):
                break
            noise = np.maximum(variance - (loadings**2).sum(axis=0), 1e-12)
        else:
            warnings.warn(
                "FactorAnalysis did not converge.", ConvergenceWarning, stacklevel=2
            )
        if self.rotation is not None and len(loadings):
            loadings = _rotate(loadings, self.rotation == "varimax")
        self.mean_, self.components_, self.noise_variance_ = mean, loadings, noise
        self.loglike_, self.n_iter_ = loglike, len(loglike)
        self._record_fit(X, len(loadings))
        return self

    def transform(self, X):
        X = self._validate_transform(X)
        weighted = self.components_ / self.noise_variance_
        if len(weighted) == 0:
            return np.empty((len(X), 0), dtype=X.dtype)
        precision = np.eye(len(weighted)) + weighted @ self.components_.T
        return linalg.solve(
            precision, ((X - self.mean_) @ weighted.T).T, assume_a="pos"
        ).T

    def get_covariance(self):
        check_is_fitted(self, "components_")
        return self.components_.T @ self.components_ + np.diag(self.noise_variance_)

    def get_precision(self):
        covariance = self.get_covariance()
        return linalg.solve(covariance, np.eye(len(covariance)), assume_a="pos")

    def score_samples(self, X):
        Xc = self._validate_transform(X) - self.mean_
        precision = self.get_precision()
        return -0.5 * (
            np.sum((Xc @ precision) * Xc, axis=1)
            + Xc.shape[1] * np.log(2 * np.pi)
            - np.linalg.slogdet(precision)[1]
        )

    def score(self, X, y=None):
        return self.score_samples(X).mean()


def _rotate(components, varimax):
    loadings = components.T
    rotation = np.eye(loadings.shape[1])
    previous = 0
    for _ in range(100):
        projected = loadings @ rotation
        target = projected**3
        if varimax:
            target -= projected * (projected**2).mean(axis=0)
        U, values, Vt = linalg.svd(loadings.T @ target, full_matrices=False)
        rotation = U @ Vt
        if values.sum() <= previous * (1 + 1e-6):
            break
        previous = values.sum()
    return (loadings @ rotation).T


class FastICA(DecompositionTransformer):
    """Independent components estimated by fixed-point iteration."""

    def __init__(
        self,
        n_components=None,
        *,
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        fun_args=None,
        max_iter=200,
        tol=0.0001,
        w_init=None,
        whiten_solver="svd",
        random_state=None,
    ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.whiten_solver = whiten_solver
        self.random_state = random_state

    def _fit(self, X, *, compute_sources):
        X = self._fit_matrix(X)
        choice(self.algorithm, "algorithm", ("parallel", "deflation"))
        choice(self.whiten, "whiten", (False, "unit-variance", "arbitrary-variance"))
        choice(self.whiten_solver, "whiten_solver", ("svd", "eigh"))
        n, p = X.shape
        if n < 2:
            raise ValueError("FastICA requires at least two samples.")
        steps = count(self.max_iter, "max_iter")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")
        k = (
            min(n, p)
            if self.n_components is None
            else count(self.n_components, "n_components")
        )
        if not self.whiten:
            if self.n_components is not None:
                warnings.warn("Ignoring n_components with whiten=False.", stacklevel=3)
            k = p
            whitening = None
            centered = X
            working = X.T
        else:
            if k > min(n, p):
                k = min(n, p)
                warnings.warn(
                    f"n_components is too large. It will be set to {k}.", stacklevel=3
                )
            mean = X.mean(axis=0)
            centered = X - mean
            if self.whiten_solver == "eigh":
                eigenvalues, vectors = linalg.eigh(centered.T @ centered)
                values = np.sqrt(
                    np.maximum(eigenvalues[::-1], np.finfo(X.dtype).eps * 10)
                )
                vectors = vectors[:, ::-1]
            else:
                vectors, values, _ = linalg.svd(centered.T, full_matrices=False)
            signs = np.where(vectors[0, :k] < 0, -1, 1).astype(vectors.dtype)
            whitening = (
                vectors[:, :k]
                * signs
                / np.maximum(values[:k], np.finfo(X.dtype).eps * max(values[0], 1))
            ).T
            working = np.dot(whitening, centered.T)
            working *= np.sqrt(n)
        rng = check_random_state(self.random_state)
        weights = (
            rng.normal(size=(k, k)).astype(X.dtype)
            if self.w_init is None
            else matrix(self.w_init).astype(X.dtype, copy=True)
        )
        if weights.shape != (k, k):
            raise ValueError(f"w_init must have shape ({k}, {k}).")
        weights, iterations, converged = self._iterate(working, weights, steps)
        if not converged:
            warnings.warn("FastICA did not converge.", ConvergenceWarning, stacklevel=3)
        components = weights if whitening is None else weights @ whitening
        sources = None
        if compute_sources or self.whiten == "unit-variance":
            sources = centered @ components.T
        if self.whiten == "unit-variance":
            scale = sources.std(axis=0)
            scale[scale == 0] = 1
            sources /= scale
            components /= scale[:, None]
            weights /= scale[:, None]
        self.components_ = components
        self.mixing_ = linalg.pinv(components)
        self._unmixing = weights
        self.n_iter_ = iterations
        if whitening is not None:
            self.whitening_, self.mean_ = whitening, mean
        else:
            self.__dict__.pop("whitening_", None)
            self.__dict__.pop("mean_", None)
        self._record_fit(X, k)
        return sources if compute_sources else None

    def _nonlinearity(self, X):
        args = self.fun_args or {}
        if callable(self.fun):
            return self.fun(X, **args)
        choice(self.fun, "fun", ("logcosh", "exp", "cube"))
        if self.fun == "cube":
            return X**3, (3 * X**2).mean(axis=-1)
        if self.fun == "exp":
            exp = np.exp(-(X**2) / 2)
            return X * exp, ((1 - X**2) * exp).mean(axis=-1)
        alpha = args.get("alpha", 1)
        if not 1 <= alpha <= 2:
            raise ValueError("alpha must be in [1, 2].")
        g = np.tanh(alpha * X)
        return g, (alpha * (1 - g**2)).mean(axis=-1)

    def _iterate(self, X, weights, steps):
        n = X.shape[1]
        if self.algorithm == "parallel":
            weights = _decorrelate(weights)
            for step in range(1, steps + 1):
                g, prime = self._nonlinearity(np.dot(weights, X))
                updated = _decorrelate(np.dot(g, X.T) / n - prime[:, None] * weights)
                limit = np.max(
                    np.abs(np.abs(np.einsum("ij,ij->i", updated, weights)) - 1)
                )
                weights = updated
                if limit < self.tol:
                    return weights, step, True
            return weights, steps, False
        max_steps, converged = 0, True
        for j in range(len(weights)):
            w = _unit(weights[j])
            for step in range(1, steps + 1):
                g, prime = self._nonlinearity(w @ X)
                updated = (X * g).mean(axis=1) - np.mean(prime) * w
                basis = weights[:j]
                updated = _unit(updated - (basis @ updated) @ basis)
                limit = abs(abs(updated @ w) - 1)
                w = updated
                if limit < self.tol:
                    break
            else:
                converged = False
            max_steps = max(max_steps, step)
            weights[j] = w
        return weights, max_steps, converged

    def fit(self, X, y=None):
        self._fit(X, compute_sources=False)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X, compute_sources=True)

    def transform(self, X, copy=True):
        X = self._validate_transform(X)
        if hasattr(self, "mean_"):
            X = X.copy() if copy or not X.flags.writeable else X
            X -= self.mean_
        return X @ self.components_.T

    def inverse_transform(self, X, copy=True):
        check_is_fitted(self, "components_")
        return matrix(X) @ self.mixing_.T + getattr(self, "mean_", 0)


def _unit(vector):
    return vector / max(linalg.norm(vector), np.finfo(vector.dtype).tiny)


def _decorrelate(weights):
    values, vectors = linalg.eigh(weights @ weights.T)
    roots = np.sqrt(np.maximum(values, np.finfo(weights.dtype).tiny))
    return np.linalg.multi_dot([vectors * (1 / roots), vectors.T, weights])


def fastica(
    X,
    n_components=None,
    *,
    algorithm="parallel",
    whiten="unit-variance",
    fun="logcosh",
    fun_args=None,
    max_iter=200,
    tol=0.0001,
    w_init=None,
    whiten_solver="svd",
    random_state=None,
    return_X_mean=False,
    compute_sources=True,
    return_n_iter=False,
):
    """Fit FastICA and return whitening, unmixing, and optional sources."""
    model = FastICA(
        n_components,
        algorithm=algorithm,
        whiten=whiten,
        fun=fun,
        fun_args=fun_args,
        max_iter=max_iter,
        tol=tol,
        w_init=w_init,
        whiten_solver=whiten_solver,
        random_state=random_state,
    )
    sources = model._fit(X, compute_sources=compute_sources)
    result = (
        getattr(model, "whitening_", None),
        model._unmixing,
        sources,
    )
    if return_X_mean:
        result += (getattr(model, "mean_", None),)
    if return_n_iter:
        result += (model.n_iter_,)
    return result
