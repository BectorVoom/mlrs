"""Non-negative matrix factorization with coordinate and multiplicative updates."""

import warnings

import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

from ._decomposition_common import (
    DecompositionTransformer,
    choice,
    coordinate_descent,
    count,
    matrix,
)
from ._decomposition_spectral import randomized_svd


def _initialize(X, W, H, k, init, rng, update_H):
    n, p = X.shape
    limit = min(n, p)
    if init is None:
        init = "nndsvda" if k <= limit else "random"
    choice(init, "init", ("random", "nndsvd", "nndsvda", "nndsvdar", "custom"))
    if init == "custom" or not update_H:
        H = matrix(H, nonnegative=True).astype(X.dtype, copy=True)
        if H.shape != (k, p) or not np.any(H):
            raise ValueError(
                f"H must have shape ({k}, {p}) and contain positive entries."
            )
        if update_H:
            W = matrix(W, nonnegative=True).astype(X.dtype, copy=True)
            if W.shape != (n, k) or not np.any(W):
                raise ValueError(
                    f"W must have shape ({n}, {k}) and contain positive entries."
                )
        else:
            W = np.zeros((n, k), dtype=X.dtype)
        return W, H
    if init == "random":
        scale = np.sqrt(X.mean() / k)
        H = np.abs(rng.standard_normal((k, p))).astype(X.dtype) * scale
        W = np.abs(rng.standard_normal((n, k))).astype(X.dtype) * scale
        return W, H
    if k > limit:
        raise ValueError("NNDSVD initialization requires n_components <= min(X.shape).")
    U, s, Vt = randomized_svd(X, k, random_state=rng)
    W, H = np.zeros_like(U), np.zeros_like(Vt)
    W[:, 0] = np.sqrt(s[0]) * np.abs(U[:, 0])
    H[0] = np.sqrt(s[0]) * np.abs(Vt[0])
    for j in range(1, k):
        positive = (np.maximum(U[:, j], 0), np.maximum(Vt[j], 0))
        negative = (np.maximum(-U[:, j], 0), np.maximum(-Vt[j], 0))
        pos_norm = np.linalg.norm(positive[0]) * np.linalg.norm(positive[1])
        neg_norm = np.linalg.norm(negative[0]) * np.linalg.norm(negative[1])
        u, v = positive if pos_norm >= neg_norm else negative
        scale = np.sqrt(s[j] * max(pos_norm, neg_norm))
        if scale > 0:
            W[:, j] = scale * u / np.linalg.norm(u)
            H[j] = scale * v / np.linalg.norm(v)
    W[W < 1e-6] = 0
    H[H < 1e-6] = 0
    for factor in (W, H):
        zeros = factor == 0
        if init == "nndsvda":
            factor[zeros] = X.mean()
        elif init == "nndsvdar":
            factor[zeros] = np.abs(rng.standard_normal(zeros.sum())) * X.mean() / 100
    return W, H


def _beta(value):
    if isinstance(value, str):
        return {"frobenius": 2, "kullback-leibler": 1, "itakura-saito": 0}[
            choice(
                value, "beta_loss", ("frobenius", "kullback-leibler", "itakura-saito")
            )
        ]
    if not np.isfinite(value):
        raise ValueError("beta_loss must be finite.")
    return value


def _penalty(alpha, ratio, scale):
    return alpha * ratio * scale, alpha * (1 - ratio) * scale


def _cd(X, W, H, penalty, rng, shuffle):
    l1, l2 = penalty
    gram = H @ H.T
    gram.flat[:: len(H) + 1] += l2
    order = rng.permutation(len(H)).tolist() if shuffle else None
    return coordinate_descent(
        gram, X @ H.T, W, alpha=l1, positive=True, max_iter=1, tol=0, order=order
    )


def _row_batches(X):
    size = max(1, 1_000_000 // X.shape[1])
    for start in range(0, X.shape[0], size):
        yield slice(start, start + size)


def _mu_terms(X, W, H, beta, penalty):
    l1, l2 = penalty
    if beta == 2:
        numerator = X @ H.T
        denominator = W @ (H @ H.T)
    elif sparse.issparse(X):
        X = X.tocsr()
        numerator = np.empty_like(W)
        denominator = np.empty_like(W)
        for batch in _row_batches(X):
            entries = X[batch].tocoo()
            estimate = W[batch] @ H
            fitted_entries = estimate[entries.row, entries.col]
            if beta < 2:
                fitted_entries = np.maximum(fitted_entries, np.finfo(np.float32).eps)
            entries.data *= fitted_entries ** (beta - 2)
            numerator[batch] = entries.tocsr() @ H.T
            if beta < 1:
                np.maximum(estimate, np.finfo(np.float32).eps, out=estimate)
            denominator[batch] = estimate ** (beta - 1) @ H.T
    else:
        estimate = W @ H
        safe = np.maximum(estimate, np.finfo(np.float32).eps) if beta < 2 else estimate
        numerator = (X * safe ** (beta - 2)) @ H.T
        denominator = (safe if beta < 1 else estimate) ** (beta - 1) @ H.T
    return numerator, denominator + l1 + l2 * W


def _mu_power(beta):
    return 1 / (2 - beta) if beta < 1 else (1 / (beta - 1) if beta > 2 else 1)


def _mu_apply(W, numerator, denominator, beta):
    safe = np.where(denominator == 0, np.finfo(np.float32).eps, denominator)
    return W * (numerator / safe) ** _mu_power(beta)


def _loss(X, W, H, beta):
    if sparse.issparse(X):
        X = X.tocsr()
        return np.sqrt(
            sum(
                _loss(X[batch].toarray(), W[batch], H, beta) ** 2
                for batch in _row_batches(X)
            )
        )
    estimate = W @ H
    if beta == 2:
        return np.linalg.norm(X - estimate) / np.sqrt(2)
    estimate = np.maximum(estimate, np.finfo(np.float32).eps)
    nonzero = X > 0
    x, y = X[nonzero], estimate[nonzero]
    if beta == 1:
        return np.sqrt(max((x * np.log(x / y)).sum() - x.sum() + estimate.sum(), 0))
    if beta == 0:
        ratio = x / y
        return np.sqrt(max((ratio - np.log(ratio) - 1).sum(), 0))
    divergence = (
        (x**beta).sum()
        + (beta - 1) * (estimate**beta).sum()
        - beta * (x * y ** (beta - 1)).sum()
    ) / (beta * (beta - 1))
    return np.sqrt(max(divergence, 0))


def _fit_nmf(
    X,
    W,
    H,
    k,
    *,
    init,
    update_H,
    solver,
    beta_loss,
    tol,
    max_iter,
    alpha_W,
    alpha_H,
    l1_ratio,
    random_state,
    verbose,
    shuffle,
):
    X = matrix(X, nonnegative=True, accept_sparse=True)
    choice(solver, "solver", ("cd", "mu"))
    beta = _beta(beta_loss)
    if solver == "cd" and beta != 2:
        raise ValueError("The coordinate descent solver supports only Frobenius loss.")
    if beta <= 0 and (
        (sparse.issparse(X) and X.nnz < np.prod(X.shape)) or X.min() <= 0
    ):
        raise ValueError("beta_loss <= 0 requires strictly positive data.")
    n, p = X.shape
    if k == "auto":
        k = np.shape(H)[0] if H is not None else np.shape(W)[1] if W is not None else p
    k = count(k, "n_components")
    steps = count(max_iter, "max_iter", minimum=0)
    alpha_H = alpha_W if alpha_H == "same" else alpha_H
    if min(alpha_W, alpha_H, tol) < 0 or not 0 <= l1_ratio <= 1:
        raise ValueError("Invalid regularization or tolerance.")
    rng = check_random_state(random_state)
    W, H = _initialize(X, W, H, k, init, rng, update_H)
    if solver == "cd" and shuffle:
        rng = check_random_state(random_state)
    if not update_H and solver == "mu":
        W.fill(np.sqrt(X.mean() / k))
    w_penalty = _penalty(alpha_W, l1_ratio, p)
    h_penalty = _penalty(alpha_H, l1_ratio, n)
    previous = initial = _loss(X, W, H, beta) if solver == "mu" and tol else 0
    iteration = 0
    for iteration in range(1, steps + 1):
        if solver == "cd":
            W, violation = _cd(X, W, H, w_penalty, rng, shuffle)
            if update_H:
                transposed, extra = _cd(X.T, H.T, W.T, h_penalty, rng, shuffle)
                H = transposed.T
                violation += extra
            if iteration == 1:
                initial = violation
            if verbose:
                print(f"violation: {violation / initial if initial else 0}")
            if initial == 0 or violation <= tol * initial:
                break
        else:
            W = _mu_apply(W, *_mu_terms(X, W, H, beta, w_penalty), beta)
            if update_H:
                H = _mu_apply(H.T, *_mu_terms(X.T, H.T, W.T, beta, h_penalty), beta).T
            if tol and iteration % 10 == 0:
                error = _loss(X, W, H, beta)
                if verbose:
                    print(f"Epoch {iteration}, error: {error}")
                if previous - error < tol * initial:
                    break
                previous = error
    else:
        if steps and tol:
            warnings.warn(
                f"Maximum number of iterations {steps} reached.",
                ConvergenceWarning,
                stacklevel=3,
            )
    return W, H, iteration


def non_negative_factorization(
    X,
    W=None,
    H=None,
    n_components="auto",
    *,
    init=None,
    update_H=True,
    solver="cd",
    beta_loss="frobenius",
    tol=0.0001,
    max_iter=200,
    alpha_W=0.0,
    alpha_H="same",
    l1_ratio=0.0,
    random_state=None,
    verbose=0,
    shuffle=False,
):
    """Factorize a non-negative matrix into W and H."""
    return _fit_nmf(
        X,
        W,
        H,
        n_components,
        init=init,
        update_H=update_H,
        solver=solver,
        beta_loss=beta_loss,
        tol=tol,
        max_iter=max_iter,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        random_state=random_state,
        verbose=verbose,
        shuffle=shuffle,
    )


class NMF(DecompositionTransformer):
    """Non-negative matrix factorization."""

    _accept_sparse = True
    _nonnegative = True

    def __init__(
        self,
        n_components="auto",
        *,
        init=None,
        solver="cd",
        beta_loss="frobenius",
        tol=0.0001,
        max_iter=200,
        random_state=None,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
    ):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle

    def fit(self, X, y=None, **params):
        self.fit_transform(X, y, **params)
        return self

    def fit_transform(self, X, y=None, W=None, H=None):
        X = self._fit_matrix(X)
        W, H, iterations = non_negative_factorization(
            X,
            W,
            H,
            self.n_components,
            init=self.init,
            solver=self.solver,
            beta_loss=self.beta_loss,
            tol=self.tol,
            max_iter=self.max_iter,
            alpha_W=self.alpha_W,
            alpha_H=self.alpha_H,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state,
            verbose=self.verbose,
            shuffle=self.shuffle,
        )
        self._finish_fit(X, W, H, iterations)
        return W

    def _finish_fit(self, X, W, H, iterations):
        self.components_ = H
        self.n_components_, self.n_iter_ = len(H), iterations
        self.reconstruction_err_ = np.sqrt(2) * _loss(X, W, H, _beta(self.beta_loss))
        self._record_fit(X, len(H))

    def transform(self, X):
        X = self._validate_transform(X)
        W, _, _ = non_negative_factorization(
            X,
            H=self.components_,
            n_components=self.n_components_,
            update_H=False,
            solver=self.solver,
            beta_loss=self.beta_loss,
            tol=self.tol,
            max_iter=self.max_iter,
            alpha_W=self.alpha_W,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )
        return W

    def inverse_transform(self, X=None, *, Xt=None):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self, "components_")
        if X is not None and Xt is not None:
            raise ValueError("Pass either X or Xt.")
        return matrix(X if Xt is None else Xt, accept_sparse=True) @ self.components_


class MiniBatchNMF(NMF):
    """Non-negative factorization with online multiplicative updates."""

    def __init__(
        self,
        n_components="auto",
        *,
        init=None,
        batch_size=1024,
        beta_loss="frobenius",
        tol=0.0001,
        max_no_improvement=10,
        max_iter=200,
        alpha_W=0.0,
        alpha_H="same",
        l1_ratio=0.0,
        forget_factor=0.7,
        fresh_restarts=False,
        fresh_restarts_max_iter=30,
        transform_max_iter=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            n_components,
            init=init,
            solver="mu",
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            verbose=verbose,
        )
        self.batch_size = batch_size
        self.max_no_improvement = max_no_improvement
        self.forget_factor = forget_factor
        self.fresh_restarts = fresh_restarts
        self.fresh_restarts_max_iter = fresh_restarts_max_iter
        self.transform_max_iter = transform_max_iter

    def _solve_weights(self, X, H, steps):
        k = len(H)
        W = np.full((X.shape[0], k), np.sqrt(X.mean() / k), dtype=X.dtype)
        penalty = _penalty(self.alpha_W, self.l1_ratio, X.shape[1])
        beta = _beta(self.beta_loss)
        for _ in range(steps):
            previous = W
            W = _mu_apply(W, *_mu_terms(X, W, H, beta, penalty), beta)
            if self.tol and np.linalg.norm(W - previous) <= self.tol * np.linalg.norm(
                W
            ):
                break
        return W

    def _batch(self, X, W, H, A, B, rho):
        beta = _beta(self.beta_loss)
        w_penalty = _penalty(self.alpha_W, self.l1_ratio, X.shape[1])
        h_alpha = self.alpha_W if self.alpha_H == "same" else self.alpha_H
        h_penalty = _penalty(h_alpha, self.l1_ratio, X.shape[0])
        if self.fresh_restarts or W is None:
            W = self._solve_weights(
                X, H, count(self.fresh_restarts_max_iter, "fresh_restarts_max_iter")
            )
        else:
            W = _mu_apply(W, *_mu_terms(X, W, H, beta, w_penalty), beta)
        if beta < 1:
            W[W < np.finfo(float).eps] = 0
        cost = (
            _loss(X, W, H, beta) ** 2
            + w_penalty[0] * W.sum()
            + h_penalty[0] * H.sum()
            + w_penalty[1] * (W**2).sum()
            + h_penalty[1] * (H**2).sum()
        ) / X.shape[0]
        numerator, denominator = _mu_terms(X.T, H.T, W.T, beta, h_penalty)
        power = _mu_power(beta)
        A = rho * A + numerator.T * H ** (1 / power)
        B = rho * B + np.where(
            denominator.T == 0, np.finfo(np.float32).eps, denominator.T
        )
        H = (A / B) ** power
        if beta <= 1:
            H[H < np.finfo(float).eps] = 0
        return W, H, A, B, cost

    def _setup(self, X, W, H):
        if not 0 <= self.forget_factor <= 1:
            raise ValueError("forget_factor must be in [0, 1].")
        k = self.n_components
        if k == "auto":
            k = (
                np.shape(H)[0]
                if H is not None
                else np.shape(W)[1]
                if W is not None
                else X.shape[1]
            )
        beta = _beta(self.beta_loss)
        if beta <= 0 and X.min() <= 0:
            raise ValueError("beta_loss <= 0 requires strictly positive data.")
        return _initialize(
            X,
            W,
            H,
            count(k, "n_components"),
            self.init,
            check_random_state(self.random_state),
            True,
        )

    def fit_transform(self, X, y=None, W=None, H=None):
        X = self._fit_matrix(X)
        W, H = self._setup(X, W, H)
        batch = min(X.shape[0], count(self.batch_size, "batch_size"))
        rho = self.forget_factor ** (batch / X.shape[0])
        A, B = H.copy(), np.ones_like(H)
        average, best, stale = None, np.inf, 0
        steps_per_epoch = (X.shape[0] + batch - 1) // batch
        epochs = count(self.max_iter, "max_iter")
        step = 0
        for step in range(1, epochs * steps_per_epoch + 1):
            start = ((step - 1) % steps_per_epoch) * batch
            idx = slice(start, start + batch)
            previous = H
            W[idx], H, A, B, cost = self._batch(X[idx], W[idx], H, A, B, rho)
            if self.verbose:
                print(f"Minibatch step {step}, error: {cost}")
            if step == 1:
                continue
            weight = min(1, W[idx].shape[0] / (X.shape[0] + 1))
            average = (
                cost if average is None else (1 - weight) * average + weight * cost
            )
            stale = 0 if average < best else stale + 1
            best = min(best, average)
            if self.tol and np.linalg.norm(previous - H) <= self.tol * np.linalg.norm(
                H
            ):
                break
            if self.max_no_improvement is not None and stale >= self.max_no_improvement:
                break
        iterations = (step + steps_per_epoch - 1) // steps_per_epoch
        if iterations == epochs and self.tol:
            warnings.warn(
                f"Maximum number of iterations {epochs} reached.",
                ConvergenceWarning,
                stacklevel=2,
            )
        if self.fresh_restarts:
            W = self._solve_weights(X, H, self._transform_steps())
        self.n_steps_ = step
        self._A, self._B, self._rho = A, B, rho
        self._finish_fit(X, W, H, iterations)
        return W

    def partial_fit(self, X, y=None, W=None, H=None):
        fitted = hasattr(self, "components_")
        X = self._validate_transform(X) if fitted else self._fit_matrix(X)
        if fitted:
            H, A, B, rho = self.components_, self._A, self._B, self._rho
        else:
            _, H = self._setup(X, W, H)
            A, B = H.copy(), np.ones_like(H)
            batch = min(X.shape[0], count(self.batch_size, "batch_size"))
            rho = self.forget_factor ** (batch / X.shape[0])
        _, H, A, B, _ = self._batch(X, None, H, A, B, rho)
        self._A, self._B, self._rho = A, B, rho
        self.n_steps_ = getattr(self, "n_steps_", 0) + 1
        self.components_, self.n_components_ = H, len(H)
        self._record_fit(X, len(H))
        return self

    def _transform_steps(self):
        iterations = (
            self.max_iter
            if self.transform_max_iter is None
            else self.transform_max_iter
        )
        return count(iterations, "transform_max_iter", minimum=0)

    def transform(self, X):
        X = self._validate_transform(X)
        if _beta(self.beta_loss) <= 0 and X.min() <= 0:
            raise ValueError("beta_loss <= 0 requires strictly positive data.")
        return self._solve_weights(X, self.components_, self._transform_steps())
