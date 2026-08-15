"""Kernel-ridge estimator shim (PY-01/PY-02) delegating to ``_mlrs``.

KernelRidge -> ``RegressorMixin``. sklearn-faithful ``__init__`` stores every
ctor arg verbatim under the SAME name (purity rule — the AST gate enforces it).
``fit`` normalizes via the base, constructs ``_mlrs.KernelRidge``, stores the
handle on ``self._mlrs_obj`` and returns ``self`` (PY-01). ``predict`` forwards
to the dtype-suffixed accessor (D-06); ``dual_coef_`` materializes the fitted
dual coefficients.

The parameter surface is ``sklearn.kernel_ridge.KernelRidge``'s in full:
``alpha`` (scalar or per-target), ``kernel`` (all nine strings AND a callable),
``gamma``, ``degree``, ``coef0``, ``kernel_params``, and
``fit(X, y, sample_weight=...)``.

## Where each parameter is honoured

Everything that is arithmetic lives in Rust — the nine string kernels, the
scalar-vs-per-target ``alpha`` split, the sample weighting. Two parameters
cannot: a CALLABLE ``kernel`` and the ``kernel_params`` that feed it are a
Python object and a Python call, so this module evaluates them (through
sklearn's own ``pairwise_kernels``, so a callable sees exactly the arguments it
would from sklearn) and hands the resulting matrix to the Rust engine through
the ``precomputed`` path. That is what ``precomputed`` is for, and it means the
solve — the part that is ``O(n³)`` — is the same code for a callable kernel as
for a named one.

``kernel_params`` is IGNORED for a string kernel, as it is in sklearn:
``_get_kernel`` reads it only on the callable branch, and the named kernels take
their coefficients from ``gamma``/``degree``/``coef0``.

Defaults mirror ``PyKernelRidge`` ``#[new]`` at
``crates/mlrs-py/src/estimators/kernel.rs``.
"""

import warnings

import numpy as np
from sklearn.base import RegressorMixin

from .base import MlrsBase

# The kernel families that need no kernel evaluation on this side. Everything
# else is either a named kernel the Rust engine computes, or a callable this
# module evaluates INTO this arm.
_PRECOMPUTED = "precomputed"

# The named kernels the Rust engine implements — sklearn's
# `PAIRWISE_KERNEL_FUNCTIONS` keys plus `precomputed`. Kept here only for the
# ctor-time error message; the authoritative rejection is Rust's
# `KernelKind::from_name`, which this list must not drift from.
_KERNEL_NAMES = (
    "additive_chi2",
    "chi2",
    "cosine",
    "laplacian",
    "linear",
    "poly",
    "polynomial",
    "precomputed",
    "rbf",
    "sigmoid",
)


class KernelRidge(RegressorMixin, MlrsBase):
    """Kernel ridge regression (KERNEL-01).

    ``KernelRidge(alpha=1.0, kernel="linear", gamma=None, degree=3.0,
    coef0=1.0, kernel_params=None)`` — the full
    ``sklearn.kernel_ridge.KernelRidge`` parameter surface.

    The argument ORDER is sklearn's (``alpha`` first), which it was not before
    the full surface landed; a caller passing ``kernel`` positionally as the
    first argument is the one incompatibility, and it now fails loudly at the
    kernel-name check rather than quietly fitting the wrong penalty.
    """

    def __init__(
        self,
        alpha=1.0,
        kernel="linear",
        gamma=None,
        degree=3.0,
        coef0=1.0,
        kernel_params=None,
        output_type="input",
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.output_type = output_type

    # -- parameter resolution --------------------------------------------- #

    def _alphas(self):
        """``self.alpha`` as the list of floats the Rust ctor takes.

        A scalar becomes a one-element list (the "one penalty for every target"
        case the Rust side keeps a fast path for); an array-like becomes one
        entry per target. The length is NOT checked here — it is checked
        against ``y`` in Rust, which is the only place the target count is
        known for certain.
        """
        alpha = self.alpha
        if np.isscalar(alpha) or (
            isinstance(alpha, np.ndarray) and alpha.ndim == 0
        ):
            return [float(alpha)]
        arr = np.asarray(alpha, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"KernelRidge: alpha must be a scalar or a 1-D array-like, got "
                f"an array with {arr.ndim} dimensions."
            )
        return [float(v) for v in arr]

    def _kernel_is_callable(self):
        return callable(self.kernel)

    def _validate_kernel(self):
        """Reject a ``kernel`` that is neither a known name nor a callable.

        Rust rejects the unknown STRING too; this exists so a caller who passed
        an ``int`` — which would reach the Rust ctor as a type error about a
        ``String`` argument — reads about ``kernel`` instead.
        """
        if self._kernel_is_callable():
            return
        if not isinstance(self.kernel, str):
            raise ValueError(
                f"KernelRidge: kernel must be a string or a callable, got "
                f"{type(self.kernel).__name__}."
            )
        if self.kernel not in _KERNEL_NAMES:
            raise ValueError(
                f"KernelRidge: unknown kernel {self.kernel!r} (expected one of "
                f"{', '.join(_KERNEL_NAMES)}, or a callable)."
            )

    def _pairwise(self, X, Y=None):
        """``K`` for a CALLABLE ``self.kernel``, via sklearn's own dispatcher.

        Deliberately sklearn's ``pairwise_kernels`` rather than a hand-rolled
        double loop: the callable's contract — what it is handed, in what order,
        with which of ``kernel_params`` filtered out — is sklearn's to define,
        and reimplementing it here would be reimplementing the one part of this
        estimator a user can replace.
        """
        from sklearn.metrics.pairwise import pairwise_kernels

        params = self.kernel_params or {}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, **params
        )

    # -- fit / predict ----------------------------------------------------- #

    def fit(self, X, y, sample_weight=None):
        self._validate_kernel()
        xa, rows, cols = self._normalize(X)
        float_dtype = self._x_float(xa)
        ya = self._normalize_y(y, dtype=float_dtype)

        # A 1-D `y` gives a 1-D `dual_coef_` and a 1-D `predict`, exactly as
        # sklearn's `ravel` flag does. Recorded here because the flat row-major
        # buffer the Rust side returns cannot say which it was.
        y_arr = np.asarray(y)
        self._y_1d = y_arr.ndim == 1
        self._n_targets = 1 if self._y_1d else int(y_arr.shape[1])

        if self._kernel_is_callable():
            # Evaluate the callable HERE and fit on the resulting Gram through
            # the precomputed path. The design matrix has to be kept on THIS
            # side: `predict` re-applies the callable against it, and what the
            # Rust estimator holds under this route is the Gram, not `X`.
            self._callable_x_fit_ = np.ascontiguousarray(
                np.asarray(X, dtype=float_dtype)
            )
            k = np.ascontiguousarray(
                self._pairwise(self._callable_x_fit_), dtype=float_dtype
            )
            ka, krows, kcols = self._normalize(k)
            kernel_name, fit_a, fit_rows, fit_cols = (
                _PRECOMPUTED,
                ka,
                krows,
                kcols,
            )
        else:
            self._callable_x_fit_ = None
            kernel_name, fit_a, fit_rows, fit_cols = (
                self.kernel,
                xa,
                rows,
                cols,
            )

        obj = self._ext().KernelRidge(
            kernel_name, self._alphas(), self.gamma, self.degree, self.coef0
        )
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=float_dtype)
        )
        obj.fit(
            fit_a, ya, fit_rows, fit_cols, self._n_targets, swa
        )
        self._mlrs_obj = obj
        if obj.used_lstsq_fallback():
            # sklearn's wording, verbatim: an indefinite Gram — which
            # `additive_chi2` produces at every alpha, and `sigmoid` at most
            # coefficient choices — makes the Cholesky inapplicable, and the
            # least-squares answer that replaces it is a different guarantee.
            warnings.warn(
                "Singular matrix in solving dual problem. Using "
                "least-squares solution instead.",
                stacklevel=2,
            )
        # `n_features_in_` is the width of the X the CALLER passed, which for
        # `precomputed` is the training-sample count (sklearn does the same) and
        # for a callable is the real feature count — not the Gram width the Rust
        # estimator was handed.
        self._post_fit(cols)
        return self

    def predict(self, X):
        # The feature-count guard runs against the CALLER's X, before the
        # callable route replaces it with a Gram — a shape error should name the
        # matrix the caller passed, not the one this method built.
        xa, rows, cols = self._check_predict_X(X)
        if self._kernel_is_callable():
            dtype = self._np_float()
            k = self._pairwise(
                np.asarray(X, dtype=dtype), self._callable_x_fit_
            )
            xa, rows, cols = self._normalize(
                np.ascontiguousarray(k, dtype=dtype)
            )
        out = self._suffixed("predict")(xa, rows, cols)
        shape = (rows,) if self._y_1d else (rows, self._n_targets)
        return self._to_output(out, shape, X, self._np_float())

    @property
    def dual_coef_(self):
        self._check_fitted()
        n = -1 if self._y_1d else self._n_targets
        shape = (n,) if self._y_1d else (-1, n)
        return self._to_output(
            self._suffixed("dual_coef")(), shape, None, self._np_float()
        )

    # sklearn also exposes the training matrix as `X_fit_`. mlrs does not: the
    # fitted training set stays device-resident (the base's whole egress
    # convention), and the copy `_callable_x_fit_` holds is an implementation
    # detail of the callable route, not a promise about the others. Exposing it
    # only where it happens to be on this side would be a fitted attribute that
    # exists for one value of `kernel` and not the rest.

    # -- sklearn tags ------------------------------------------------------ #

    def __sklearn_tags__(self):
        """Mark the estimator PAIRWISE when ``kernel='precomputed'``.

        Without this, sklearn's cross-validation splits a precomputed Gram by
        rows only, handing the estimator an ``(n_train, n_total)`` matrix and a
        confusing shape error. The tag is what tells the splitters to take the
        same subset of the columns.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.kernel == _PRECOMPUTED
        return tags

    @staticmethod
    def _x_float(xa):
        return np.float32 if xa.type.bit_width == 32 else np.float64
