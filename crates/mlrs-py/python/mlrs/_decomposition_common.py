"""Validation and array handling shared by the decomposition estimators."""

from numbers import Integral

import numpy as np
from scipy import linalg, sparse
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.validation import check_array, check_is_fitted, validate_data


class DecompositionTransformer(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Shared feature names and sklearn transformer protocol."""

    _accept_sparse = False
    _nonnegative = False

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = self._accept_sparse
        tags.input_tags.positive_only = self._nonnegative
        return tags

    def _record_fit(self, X, n_outputs):
        self.n_features_in_ = X.shape[1]
        self._n_features_out = n_outputs

    def _fit_matrix(self, X):
        return matrix(
            X,
            accept_sparse=self._accept_sparse,
            nonnegative=self._nonnegative,
            estimator=self,
            reset=True,
        )

    def _validate_transform(self, X):
        check_is_fitted(self, "_n_features_out")
        return matrix(
            X,
            accept_sparse=self._accept_sparse,
            nonnegative=self._nonnegative,
            estimator=self,
            reset=False,
        )


def matrix(X, *, accept_sparse=False, nonnegative=False, estimator=None, reset=True):
    options = {
        "accept_sparse": "csr" if accept_sparse else False,
        "dtype": [np.float64, np.float32],
    }
    if estimator is None:
        X = check_array(X, **options)
    else:
        X = validate_data(estimator, X, reset=reset, **options)
    if nonnegative:
        values = X.data if sparse.issparse(X) else X
        if np.any(values < 0):
            raise ValueError(
                "Negative values in data passed to a non-negative decomposition."
            )
    return X


def count(value, name, *, minimum=1):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"Invalid {name}={value!r}.")
    return int(value)


def choice(value, name, choices):
    if value not in choices:
        raise ValueError(f"Invalid {name}={value!r}. Expected one of {choices}.")
    return value


def flip_vectors(U, Vt, *, use_u=False):
    if use_u:
        pivots = U[np.argmax(np.abs(U), axis=0), np.arange(U.shape[1])]
    else:
        pivots = Vt[np.arange(Vt.shape[0]), np.argmax(np.abs(Vt), axis=1)]
    signs = np.where(pivots < 0, -1, 1).astype(U.dtype)
    return U * signs, Vt * signs[:, None]


def ridge_codes(X, dictionary, alpha):
    gram = dictionary @ dictionary.T
    gram.flat[:: len(dictionary) + 1] += alpha
    cross = X @ dictionary.T
    if alpha > 0:
        return linalg.solve(gram, cross.T, assume_a="pos").T
    return linalg.lstsq(dictionary.T, X.T)[0].T


def coordinate_descent(
    gram, cross, initial, *, alpha, positive, max_iter, tol, order=None
):
    from . import _mlrs

    if order is None:
        order = list(range(len(gram)))
    values, violation = _mlrs.decomp_coordinate_descent(
        gram.ravel().tolist(),
        cross.ravel().tolist(),
        initial.ravel().tolist(),
        order,
        alpha,
        positive,
        max_iter,
        tol,
    )
    return np.asarray(values, dtype=cross.dtype).reshape(cross.shape), violation
