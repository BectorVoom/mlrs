"""Random-projection estimator shims (PROJ-01/PROJ-02) delegating to ``_mlrs``.

``GaussianRandomProjection`` (dense ``N(0, 1/n_components)`` matrix) and
``SparseRandomProjection`` (Achlioptas, stored dense) subclass
``TransformerMixin`` + :class:`MlrsBase` (the mixin gives ``fit_transform``).
The sklearn-faithful ``__init__`` stores every ctor arg verbatim; ``n_components``
defaults to ``'auto'`` (JL-sized) and is mapped to the ``_mlrs`` ``None`` sentinel
at ``fit``. ``random_state`` maps to the Rust ``seed`` (a documented ``u64``
SplitMix64 source).

CRITICAL (D-12 / PROJ-02): :meth:`SparseRandomProjection.fit` /
:meth:`SparseRandomProjection.transform` DENSIFY a scipy-sparse input at the
Python ingress boundary (``X.toarray()``) BEFORE :meth:`MlrsBase._normalize` —
the device path is dense-only. ``GaussianRandomProjection`` densifies too for
symmetry (sklearn accepts sparse input for both).

The module-level :func:`johnson_lindenstrauss_min_dim` delegates to the ``_mlrs``
``#[pyfunction]`` of the same name (value-matched to sklearn at 1e-5).
"""

from sklearn.base import TransformerMixin

from .base import MlrsBase


def _densify(X):
    """Densify a scipy-sparse ``X`` to a dense numpy array at ingress (D-12).

    The device random-projection path is dense-only (``components_`` are stored
    dense even for the Achlioptas sparse projection), so a sparse input is
    materialized to a dense array BEFORE :meth:`MlrsBase._normalize`. A
    non-sparse input is returned unchanged. ``scipy`` is imported lazily so the
    shim does not hard-depend on it when no sparse input is ever passed.
    """
    try:
        from scipy.sparse import issparse
    except ImportError:  # pragma: no cover - scipy always present with sklearn
        return X
    if issparse(X):
        return X.toarray()
    return X


def johnson_lindenstrauss_min_dim(n_samples, eps=0.1):
    """Minimum safe embedding dimension per the Johnson-Lindenstrauss lemma.

    Delegates to the ``_mlrs`` ``johnson_lindenstrauss_min_dim`` ``#[pyfunction]``
    (value-matched to ``sklearn.random_projection.johnson_lindenstrauss_min_dim``
    at 1e-5). ``eps`` must lie in ``(0, 1)``; an out-of-range ``eps`` raises a
    ``ValueError``. Returns a python ``int``.

    ``n_samples`` may be a scalar or an array-like; for an array-like the result
    is computed per element (mirroring sklearn), returned as a numpy ``int64``
    array.
    """
    import numpy as np

    from . import _load_ext

    fn = _load_ext().johnson_lindenstrauss_min_dim
    arr = np.asarray(n_samples)
    if arr.ndim == 0:
        return int(fn(float(arr), float(eps)))
    return np.array(
        [int(fn(float(n), float(eps))) for n in arr.ravel()], dtype=np.int64
    ).reshape(arr.shape)


class GaussianRandomProjection(TransformerMixin, MlrsBase):
    """Gaussian random projection (PROJ-01)."""

    def __init__(
        self,
        n_components="auto",
        *,
        eps=0.1,
        random_state=None,
        output_type="input",
    ):
        self.n_components = n_components
        self.eps = eps
        self.random_state = random_state
        self.output_type = output_type

    def _resolved_n_components(self):
        """``'auto'`` -> ``None`` (``_mlrs`` JL sentinel); an int -> the int."""
        return None if self.n_components == "auto" else int(self.n_components)

    def _resolved_seed(self):
        """``random_state`` -> the Rust ``u64`` seed (``None`` -> ``0``)."""
        return 0 if self.random_state is None else int(self.random_state)

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(_densify(X))
        obj = self._ext().GaussianRandomProjection(
            self._resolved_n_components(), self.eps, self._resolved_seed()
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(_densify(X))
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components_), X, self._np_float()
        )

    @property
    def n_components_(self):
        """Resolved embedding dimension after fit (``'auto'`` -> JL value)."""
        self._check_fitted()
        return int(self._mlrs_obj.n_components_())

    @property
    def components_(self):
        out = self._suffixed("components")()
        return self._to_output(
            out, (self.n_components_, -1), None, self._np_float()
        )


class SparseRandomProjection(TransformerMixin, MlrsBase):
    """Sparse (Achlioptas) random projection, stored dense (PROJ-02)."""

    def __init__(
        self,
        n_components="auto",
        *,
        density="auto",
        eps=0.1,
        random_state=None,
        output_type="input",
    ):
        self.n_components = n_components
        self.density = density
        self.eps = eps
        self.random_state = random_state
        self.output_type = output_type

    def _resolved_n_components(self):
        return None if self.n_components == "auto" else int(self.n_components)

    def _resolved_density(self):
        """``'auto'`` -> ``None`` (``_mlrs`` ``1/sqrt(n_features)`` default)."""
        return None if self.density == "auto" else float(self.density)

    def _resolved_seed(self):
        return 0 if self.random_state is None else int(self.random_state)

    def fit(self, X, y=None):
        # D-12 / PROJ-02: densify sparse input at ingress (dense-only device).
        xa, rows, cols = self._normalize(_densify(X))
        obj = self._ext().SparseRandomProjection(
            self._resolved_n_components(),
            self.eps,
            self._resolved_seed(),
            self._resolved_density(),
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        # D-12 / PROJ-02: densify sparse input at ingress before the device GEMM.
        xa, rows, cols = self._check_predict_X(_densify(X))
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components_), X, self._np_float()
        )

    @property
    def n_components_(self):
        self._check_fitted()
        return int(self._mlrs_obj.n_components_())

    @property
    def density_(self):
        """Resolved density after fit (``'auto'`` -> ``1/sqrt(n_features)``)."""
        self._check_fitted()
        return float(self._mlrs_obj.density_())

    @property
    def components_(self):
        out = self._suffixed("components")()
        return self._to_output(
            out, (self.n_components_, -1), None, self._np_float()
        )
