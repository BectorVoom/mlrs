"""Decomposition estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

PCA, TruncatedSVD, IncrementalPCA -> ``TransformerMixin`` (gives
``fit_transform``). sklearn-faithful ``__init__`` stores every ctor arg verbatim
(RESEARCH 06 §Hyperparameter Mapping); v1 PCA requires an explicit int
``n_components`` (no ``None`` / ``'mle'``). ``fit`` is unsupervised (``y=None``);
``transform`` returns a ``rows × n_components`` array. PCA + IncrementalPCA
additionally expose ``inverse_transform`` (TruncatedSVD does not). IncrementalPCA
additionally exposes ``partial_fit`` (the first v2 streaming ``partial_fit``).
"""

from sklearn.base import TransformerMixin

from .base import MlrsBase


class PCA(TransformerMixin, MlrsBase):
    """Principal component analysis, full SVD solver (DECOMP-01)."""

    def __init__(self, n_components, output_type="input"):
        self.n_components = n_components
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().PCA(self.n_components)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components), X, self._np_float()
        )

    def inverse_transform(self, Z):
        self._check_fitted()
        za, rows, k = self._normalize(Z, dtype=self._np_float())
        out = self._suffixed("inverse_transform")(za, rows, k)
        return self._to_output(out, (rows, -1), Z, self._np_float())

    @property
    def components_(self):
        out = self._suffixed("components")()
        return self._to_output(
            out, (self.n_components, -1), None, self._np_float()
        )

    @property
    def mean_(self):
        return self._to_output(
            self._suffixed("mean")(), (-1,), None, self._np_float()
        )

    @property
    def explained_variance_(self):
        return self._to_output(
            self._suffixed("explained_variance")(),
            (-1,),
            None,
            self._np_float(),
        )

    @property
    def explained_variance_ratio_(self):
        return self._to_output(
            self._suffixed("explained_variance_ratio")(),
            (-1,),
            None,
            self._np_float(),
        )


class IncrementalPCA(TransformerMixin, MlrsBase):
    """Streaming principal component analysis via incremental SVD (DECOMP-03).

    Exposes ``partial_fit`` — the first v2 streaming fit. ``fit`` is the
    sklearn-faithful one-shot path (the Rust estimator loops ``partial_fit`` over
    ``gen_batches`` internally); ``partial_fit`` merges a single batch into the
    running state, accumulating ``n_samples_seen_``. The compiled ``_mlrs``
    object is constructed ONCE (on ``fit`` or the first ``partial_fit``) and
    reused across a ``partial_fit`` stream.
    """

    def __init__(
        self, n_components, *, whiten=False, batch_size=None, output_type="input"
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.batch_size = batch_size
        self.output_type = output_type

    def fit(self, X, y=None):
        """Fit the decomposition on ``X`` in one shot.

        RESETS all running state: a FRESH ``_mlrs`` object is built on every
        call, so ``fit`` is NOT cumulative and ``n_samples_seen_`` reflects only
        the rows of THIS ``X`` (sklearn ``IncrementalPCA.fit`` semantics).

        Note the asymmetry with :meth:`partial_fit`, which CONTINUES the stream:
        calling ``fit(X1)`` then ``partial_fit(X2)`` merges ``X2`` into the
        ``X1``-fitted running state and bumps ``n_samples_seen_`` past
        ``len(X1)`` — matching sklearn, but easy to misjudge when mixing the two
        entry points.
        """
        xa, rows, cols = self._normalize(X)
        obj = self._ext().IncrementalPCA(
            self.n_components, self.whiten, self.batch_size
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def partial_fit(self, X, y=None):
        """Merge a single batch into the running decomposition.

        Constructs the compiled ``_mlrs`` object on the FIRST call (from the
        stored hyperparameters) and reuses it on subsequent calls so the running
        SVD state accumulates across the stream.

        Unlike :meth:`fit`, ``partial_fit`` does NOT reset: it CONTINUES the
        existing stream when ``_mlrs_obj`` is already present. A ``fit(X1)``
        followed by ``partial_fit(X2)`` therefore merges ``X2`` into the
        ``X1``-fitted state and advances ``n_samples_seen_`` to
        ``len(X1) + len(X2)`` (sklearn-faithful, but asymmetric with ``fit``).
        """
        xa, rows, cols = self._normalize(X)
        if getattr(self, "_mlrs_obj", None) is None:
            self._mlrs_obj = self._ext().IncrementalPCA(
                self.n_components, self.whiten, self.batch_size
            )
        self._mlrs_obj.partial_fit(xa, rows, cols)
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components), X, self._np_float()
        )

    def inverse_transform(self, Z):
        self._check_fitted()
        za, rows, k = self._normalize(Z, dtype=self._np_float())
        out = self._suffixed("inverse_transform")(za, rows, k)
        return self._to_output(out, (rows, -1), Z, self._np_float())

    @property
    def components_(self):
        out = self._suffixed("components")()
        return self._to_output(
            out, (self.n_components, -1), None, self._np_float()
        )

    @property
    def explained_variance_(self):
        return self._to_output(
            self._suffixed("explained_variance")(),
            (-1,),
            None,
            self._np_float(),
        )

    @property
    def explained_variance_ratio_(self):
        return self._to_output(
            self._suffixed("explained_variance_ratio")(),
            (-1,),
            None,
            self._np_float(),
        )

    @property
    def singular_values_(self):
        return self._to_output(
            self._suffixed("singular_values")(), (-1,), None, self._np_float()
        )

    @property
    def mean_(self):
        return self._to_output(
            self._suffixed("mean")(), (-1,), None, self._np_float()
        )

    @property
    def var_(self):
        return self._to_output(
            self._suffixed("var")(), (-1,), None, self._np_float()
        )

    @property
    def n_samples_seen_(self):
        """Total samples merged so far across ``partial_fit`` calls (scalar)."""
        self._check_fitted()
        return int(self._mlrs_obj.n_samples_seen())


class TruncatedSVD(TransformerMixin, MlrsBase):
    """Truncated SVD (DECOMP-02)."""

    def __init__(self, n_components=2, output_type="input"):
        self.n_components = n_components
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().TruncatedSVD(self.n_components)
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components), X, self._np_float()
        )

    @property
    def components_(self):
        out = self._suffixed("components")()
        return self._to_output(
            out, (self.n_components, -1), None, self._np_float()
        )

    @property
    def singular_values_(self):
        return self._to_output(
            self._suffixed("singular_values")(), (-1,), None, self._np_float()
        )

    @property
    def explained_variance_ratio_(self):
        return self._to_output(
            self._suffixed("explained_variance_ratio")(),
            (-1,),
            None,
            self._np_float(),
        )
