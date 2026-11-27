"""Sparse coding, dictionary learning, and sparse principal components."""

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy import linalg
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._decomposition_common import (
    DecompositionTransformer,
    choice,
    coordinate_descent,
    count,
    flip_vectors,
    matrix,
    native_buffer,
    ridge_codes,
)
from ._decomposition_spectral import randomized_svd


def sparse_encode(
    X,
    dictionary,
    *,
    gram=None,
    cov=None,
    algorithm="lasso_lars",
    n_nonzero_coefs=None,
    alpha=None,
    copy_cov=True,
    init=None,
    max_iter=1000,
    n_jobs=None,
    check_input=True,
    verbose=0,
    positive=False,
):
    """Encode rows of X as sparse linear combinations of dictionary atoms."""
    if check_input:
        X, dictionary = matrix(X), matrix(dictionary)
    else:
        X, dictionary = np.asarray(X), np.asarray(dictionary)
    if X.ndim != 2 or dictionary.ndim != 2 or X.shape[1] != dictionary.shape[1]:
        raise ValueError("X and dictionary must have the same number of features.")
    choice(
        algorithm, "algorithm", ("lasso_lars", "lasso_cd", "lars", "omp", "threshold")
    )
    if positive and algorithm in ("omp", "lars"):
        raise ValueError(f"Positive coding is unavailable for algorithm={algorithm!r}.")
    n, p = X.shape
    k = len(dictionary)
    if not k:
        raise ValueError("The dictionary must contain at least one atom.")
    alpha = 1 if alpha is None else alpha
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    cross = X @ dictionary.T if cov is None else np.asarray(cov).T
    if cross.shape != (n, k):
        raise ValueError("cov must have shape (n_components, n_samples).")
    if algorithm == "threshold":
        code = np.sign(cross) * np.maximum(np.abs(cross) - alpha, 0)
        return np.maximum(code, 0) if positive else code
    gram = dictionary @ dictionary.T if gram is None else matrix(gram)
    if gram.shape != (k, k):
        raise ValueError("gram must have shape (n_components, n_components).")
    if algorithm == "lasso_cd":
        initial = np.zeros_like(cross) if init is None else matrix(init)
        if initial.shape != cross.shape:
            raise ValueError("init must have shape (n_samples, n_components).")
        return coordinate_descent(
            gram,
            cross,
            initial,
            alpha=alpha,
            positive=positive,
            max_iter=count(max_iter, "max_iter"),
            tol=1e-8,
        )[0]
    limit = min(
        k,
        max(p // 10, 1)
        if n_nonzero_coefs is None
        else count(n_nonzero_coefs, "n_nonzero_coefs"),
    )
    from . import _mlrs

    steps = count(max_iter, "max_iter") if algorithm == "lasso_lars" else limit
    gram = native_buffer(gram)

    def solve(batch):
        codes = _mlrs.decomp_sparse_code(
            gram, native_buffer(batch), k, steps, algorithm, alpha, positive
        )
        return np.frombuffer(codes, dtype=np.float64).reshape(len(batch), k)

    jobs = min(effective_n_jobs(n_jobs), n)
    if jobs <= 1:
        return solve(cross).astype(cross.dtype, copy=False)
    batches = np.array_split(cross, jobs)
    result = Parallel(n_jobs=jobs, prefer="threads", verbose=verbose)(
        delayed(solve)(batch) for batch in batches
    )
    return np.concatenate(result).astype(cross.dtype, copy=False)


class _DictionaryTransform(DecompositionTransformer):
    def transform(self, X):
        if not hasattr(self, "components_") and hasattr(self, "dictionary"):
            dictionary = matrix(self.dictionary)
            X = matrix(X)
        else:
            X = self._validate_transform(X)
            dictionary = self.components_
        alpha = self.transform_alpha
        if alpha is None:
            alpha = getattr(self, "alpha", 1)
        code = sparse_encode(
            X,
            dictionary,
            algorithm=self.transform_algorithm,
            n_nonzero_coefs=self.transform_n_nonzero_coefs,
            alpha=alpha,
            max_iter=self.transform_max_iter,
            n_jobs=self.n_jobs,
            positive=self.positive_code,
        )
        if self.split_sign:
            return np.concatenate((np.maximum(code, 0), np.maximum(-code, 0)), axis=1)
        return code

    def inverse_transform(self, X):
        check_is_fitted(self)
        dictionary = (
            matrix(self.dictionary) if hasattr(self, "dictionary") else self.components_
        )
        X = matrix(X)
        if self.split_sign:
            if X.shape[1] != 2 * len(dictionary):
                raise ValueError(
                    "Split codes must have twice as many columns as dictionary atoms."
                )
            X = X[:, : len(dictionary)] - X[:, len(dictionary) :]
        return X @ dictionary

    def _record_dictionary_fit(self, X, dictionary):
        self.components_ = dictionary
        self._record_fit(X, len(dictionary) * (2 if self.split_sign else 1))


class SparseCoder(_DictionaryTransform):
    """Transform signals using an existing dictionary."""

    def __init__(
        self,
        dictionary,
        *,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        split_sign=False,
        n_jobs=None,
        positive_code=False,
        transform_max_iter=1000,
    ):
        self.dictionary = dictionary
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.split_sign = split_sign
        self.n_jobs = n_jobs
        self.positive_code = positive_code
        self.transform_max_iter = transform_max_iter

    @property
    def n_features_in_(self):
        return np.shape(self.dictionary)[1]

    @property
    def n_components_(self):
        return np.shape(self.dictionary)[0]

    @property
    def _n_features_out(self):
        return self.n_components_ * (2 if self.split_sign else 1)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        matrix(self.dictionary)
        return self


def _initial_dictionary(X, k, rng, dictionary=None, code=None, online=False):
    if dictionary is not None:
        dictionary = matrix(dictionary).astype(X.dtype, copy=True)
        if dictionary.shape != (k, X.shape[1]):
            raise ValueError("dict_init must have shape (n_components, n_features).")
    if code is not None:
        code = matrix(code).astype(X.dtype, copy=True)
        if code.shape != (X.shape[0], k):
            raise ValueError("code_init must have shape (n_samples, n_components).")
    if dictionary is not None and (online or code is not None):
        return code, dictionary
    if online:
        U, singular, Vt = randomized_svd(X, min(k, *X.shape), random_state=rng)
    else:
        U, singular, Vt = linalg.svd(X, full_matrices=False)
        U, Vt = flip_vectors(U, Vt, use_u=True)
    rank = min(k, len(singular))
    if code is None:
        code = np.zeros((len(X), k), dtype=X.dtype)
        code[:, :rank] = U[:, :rank]
    if dictionary is None:
        dictionary = np.zeros((k, X.shape[1]), dtype=X.dtype)
        dictionary[:rank] = singular[:rank, None] * Vt[:rank]
    return code, dictionary


def _dictionary_step(X, dictionary, code, gram, cross, rng, positive):
    for j, atom in enumerate(dictionary):
        if gram[j, j] > 1e-6:
            atom += (cross[j] - gram[j] @ dictionary) / gram[j, j]
        else:
            row = X[rng.randint(len(X))]
            atom[:] = row + rng.normal(0, 0.01 * (row.std() or 1), row.shape)
            code[:, j] = 0
        if positive:
            np.maximum(atom, 0, out=atom)
        atom /= max(linalg.norm(atom), 1)


def _dictionary_cost(X, code, dictionary, alpha):
    residual = X - code @ dictionary
    return 0.5 * np.sum(residual**2) + alpha * np.abs(code).sum()


def dict_learning(
    X,
    n_components,
    *,
    alpha,
    max_iter=100,
    tol=1e-8,
    method="lars",
    n_jobs=None,
    dict_init=None,
    code_init=None,
    callback=None,
    verbose=False,
    random_state=None,
    return_n_iter=False,
    positive_dict=False,
    positive_code=False,
    method_max_iter=1000,
):
    """Learn a dictionary by alternating sparse encoding and atom updates."""
    X = matrix(X)
    k = count(n_components, "n_components")
    steps = count(max_iter, "max_iter", minimum=0)
    choice(method, "method", ("lars", "cd"))
    if tol < 0 or alpha < 0:
        raise ValueError("alpha and tol must be non-negative.")
    rng = check_random_state(random_state)
    code, dictionary = _initial_dictionary(X, k, rng, dict_init, code_init)
    errors = []
    for ii in range(steps):
        code = sparse_encode(
            X,
            dictionary,
            algorithm="lasso_" + method,
            alpha=alpha,
            n_jobs=n_jobs,
            init=code,
            positive=positive_code,
            max_iter=method_max_iter,
        )
        _dictionary_step(
            X, dictionary, code, code.T @ code, code.T @ X, rng, positive_dict
        )
        errors.append(_dictionary_cost(X, code, dictionary, alpha))
        if verbose and ii % 5 == 0:
            print(f"Iteration {ii}, cost: {errors[-1]}")
        if ii and errors[-2] - errors[-1] < tol * errors[-1]:
            break
        if callback is not None and ii % 5 == 0:
            callback(dict(locals()))
    result = (code, dictionary, errors)
    return result + (len(errors),) if return_n_iter else result


class DictionaryLearning(_DictionaryTransform):
    """Learn a dictionary that admits sparse encodings of the training data."""

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        max_iter=1000,
        tol=1e-8,
        fit_algorithm="lars",
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        n_jobs=None,
        code_init=None,
        dict_init=None,
        callback=None,
        verbose=False,
        split_sign=False,
        random_state=None,
        positive_code=False,
        positive_dict=False,
        transform_max_iter=1000,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.n_jobs = n_jobs
        self.code_init = code_init
        self.dict_init = dict_init
        self.callback = callback
        self.verbose = verbose
        self.split_sign = split_sign
        self.random_state = random_state
        self.positive_code = positive_code
        self.positive_dict = positive_dict
        self.transform_max_iter = transform_max_iter

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        X = self._fit_matrix(X)
        k = X.shape[1] if self.n_components is None else self.n_components
        code, dictionary, self.error_, self.n_iter_ = dict_learning(
            X,
            k,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            method=self.fit_algorithm,
            n_jobs=self.n_jobs,
            dict_init=self.dict_init,
            code_init=self.code_init,
            callback=self.callback,
            verbose=self.verbose,
            random_state=self.random_state,
            return_n_iter=True,
            positive_dict=self.positive_dict,
            positive_code=self.positive_code,
            method_max_iter=self.transform_max_iter,
        )
        self._record_dictionary_fit(X, dictionary)
        return code


class MiniBatchDictionaryLearning(_DictionaryTransform):
    """Learn a dictionary with minibatch updates to sufficient statistics."""

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        max_iter=1000,
        fit_algorithm="lars",
        n_jobs=None,
        batch_size=256,
        shuffle=True,
        dict_init=None,
        transform_algorithm="omp",
        transform_n_nonzero_coefs=None,
        transform_alpha=None,
        verbose=False,
        split_sign=False,
        random_state=None,
        positive_code=False,
        positive_dict=False,
        transform_max_iter=1000,
        callback=None,
        tol=0.001,
        max_no_improvement=10,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dict_init = dict_init
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.verbose = verbose
        self.split_sign = split_sign
        self.random_state = random_state
        self.positive_code = positive_code
        self.positive_dict = positive_dict
        self.transform_max_iter = transform_max_iter
        self.callback = callback
        self.tol = tol
        self.max_no_improvement = max_no_improvement

    def _setup(self, X):
        choice(self.fit_algorithm, "fit_algorithm", ("lars", "cd"))
        if self.alpha < 0 or self.tol < 0:
            raise ValueError("alpha and tol must be non-negative.")
        k = (
            X.shape[1]
            if self.n_components is None
            else count(self.n_components, "n_components")
        )
        self._rng = check_random_state(self.random_state)
        _, dictionary = _initial_dictionary(
            X, k, self._rng, self.dict_init, online=True
        )
        self._A = np.zeros((k, k), dtype=X.dtype)
        self._B = np.zeros_like(dictionary)
        self.n_steps_ = 0
        return dictionary

    def _step(self, X, dictionary):
        code = sparse_encode(
            X,
            dictionary,
            algorithm="lasso_" + self.fit_algorithm,
            alpha=self.alpha,
            n_jobs=self.n_jobs,
            positive=self.positive_code,
            max_iter=self.transform_max_iter,
        )
        size = len(X)
        step = self.n_steps_
        theta = (step + 1) * size if step < size - 1 else size**2 + step + 1 - size
        beta = (theta + 1 - size) / (theta + 1)
        A = beta * self._A + code.T @ code / size
        B = beta * self._B + code.T @ X / size
        cost = _dictionary_cost(X, code, dictionary, self.alpha) / size
        _dictionary_step(X, dictionary, code, A, B, self._rng, self.positive_dict)
        self._A, self._B = A, B
        self.n_steps_ += 1
        return cost

    def fit(self, X, y=None):
        X = self._fit_matrix(X)
        dictionary = self._setup(X)
        batch_size = min(len(X), count(self.batch_size, "batch_size"))
        ordered = X[self._rng.permutation(len(X))] if self.shuffle else X
        batches = [ordered[i : i + batch_size] for i in range(0, len(X), batch_size)]
        average, best, stale = None, np.inf, 0
        steps = count(self.max_iter, "max_iter", minimum=0) * len(batches)
        for step in range(steps):
            X_batch = batches[step % len(batches)]
            old = dictionary.copy()
            cost = self._step(X_batch, dictionary)
            if self.verbose:
                print(f"Step {step + 1}/{steps}, cost: {cost}")
            if self.callback is not None:
                self.callback(dict(locals()))
            if step + 1 <= min(100, len(X) / len(X_batch)):
                continue
            weight = min(1, len(X_batch) / (len(X) + 1))
            average = (
                cost if average is None else (1 - weight) * average + weight * cost
            )
            stale = 0 if average < best else stale + 1
            best = min(best, average)
            if self.tol and linalg.norm(dictionary - old) / len(dictionary) <= self.tol:
                break
            if self.max_no_improvement is not None and stale >= self.max_no_improvement:
                break
        self.n_iter_ = (self.n_steps_ + len(batches) - 1) // len(batches)
        self._record_dictionary_fit(X, dictionary)
        return self

    def partial_fit(self, X, y=None):
        if hasattr(self, "components_"):
            X = self._validate_transform(X)
            dictionary = self.components_.copy()
        else:
            X = self._fit_matrix(X)
            dictionary = self._setup(X)
        self._step(X, dictionary)
        self._record_dictionary_fit(X, dictionary)
        return self


def dict_learning_online(
    X,
    n_components=2,
    *,
    alpha=1,
    max_iter=100,
    return_code=True,
    dict_init=None,
    callback=None,
    batch_size=256,
    verbose=False,
    shuffle=True,
    n_jobs=None,
    method="lars",
    random_state=None,
    positive_dict=False,
    positive_code=False,
    method_max_iter=1000,
    tol=0.001,
    max_no_improvement=10,
):
    """Learn a dictionary incrementally from minibatches."""
    model = MiniBatchDictionaryLearning(
        n_components,
        alpha=alpha,
        max_iter=max_iter,
        fit_algorithm=method,
        n_jobs=n_jobs,
        batch_size=batch_size,
        shuffle=shuffle,
        dict_init=dict_init,
        transform_algorithm="lasso_" + method,
        verbose=verbose,
        random_state=random_state,
        positive_code=positive_code,
        positive_dict=positive_dict,
        transform_max_iter=method_max_iter,
        callback=callback,
        tol=tol,
        max_no_improvement=max_no_improvement,
    ).fit(X)
    return (model.transform(X), model.components_) if return_code else model.components_


class _SparsePCATransform(DecompositionTransformer):
    def _finish_fit(self, X, code, mean, iterations):
        code = code.T
        norms = linalg.norm(code, axis=1)
        self.components_ = code / np.where(norms > 0, norms, 1)[:, None]
        self.mean_ = mean
        self.n_iter_, self.n_components_ = iterations, len(code)
        self._record_fit(X, len(code))
        return self

    def transform(self, X):
        X = self._validate_transform(X)
        if self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be non-negative.")
        return ridge_codes(X - self.mean_, self.components_, self.ridge_alpha)

    def inverse_transform(self, X):
        check_is_fitted(self, "components_")
        return matrix(X) @ self.components_ + self.mean_


class SparsePCA(_SparsePCATransform):
    """Extract sparse components with batch dictionary learning."""

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        ridge_alpha=0.01,
        max_iter=1000,
        tol=1e-8,
        method="lars",
        n_jobs=None,
        U_init=None,
        V_init=None,
        verbose=False,
        random_state=None,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.U_init = U_init
        self.V_init = V_init
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        X = self._fit_matrix(X)
        mean = X.mean(axis=0)
        k = X.shape[1] if self.n_components is None else self.n_components
        code, dictionary, self.error_, iterations = dict_learning(
            (X - mean).T,
            k,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            return_n_iter=True,
            code_init=None if self.V_init is None else np.asarray(self.V_init).T,
            dict_init=None if self.U_init is None else np.asarray(self.U_init).T,
        )
        code, _ = flip_vectors(code, dictionary, use_u=True)
        return self._finish_fit(X, code, mean, iterations)


class MiniBatchSparsePCA(_SparsePCATransform):
    """Extract sparse components with minibatch dictionary learning."""

    def __init__(
        self,
        n_components=None,
        *,
        alpha=1,
        ridge_alpha=0.01,
        max_iter=1000,
        callback=None,
        batch_size=3,
        verbose=False,
        shuffle=True,
        n_jobs=None,
        method="lars",
        random_state=None,
        tol=0.001,
        max_no_improvement=10,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.callback = callback
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.method = method
        self.random_state = random_state
        self.tol = tol
        self.max_no_improvement = max_no_improvement

    def fit(self, X, y=None):
        X = self._fit_matrix(X)
        mean = X.mean(axis=0)
        k = X.shape[1] if self.n_components is None else self.n_components
        centered = (X - mean).T
        model = MiniBatchDictionaryLearning(
            k,
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_algorithm=self.method,
            n_jobs=self.n_jobs,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            verbose=self.verbose,
            callback=self.callback,
            random_state=self.random_state,
            tol=self.tol,
            max_no_improvement=self.max_no_improvement,
            transform_algorithm="lasso_" + self.method,
        ).fit(centered)
        return self._finish_fit(X, model.transform(centered), mean, model.n_iter_)
