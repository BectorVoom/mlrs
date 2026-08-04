"""Gaussian mixture model shim (MIX-01) delegating to ``_mlrs``.

``GaussianMixture`` subclasses :class:`MlrsBase` and mirrors
``sklearn.mixture.GaussianMixture`` ctor-for-ctor: every hyperparameter is
stored verbatim (RESEARCH 06 §Hyperparameter Mapping), and the fitted
attributes (``weights_`` / ``means_`` / ``covariances_`` / ``precisions_`` /
``precisions_cholesky_`` / ``converged_`` / ``n_iter_`` / ``lower_bound_``) map
onto the dtype-suffixed ``_mlrs`` accessors.

Two shapes are worth knowing about at this layer:

* ``covariances_`` and ``precisions_`` have a DIFFERENT shape per
  ``covariance_type`` — ``(k, d, d)`` / ``(d, d)`` / ``(k, d)`` / ``(k,)``. The
  Rust side reports the shape through ``covariance_shape()`` rather than the
  shim re-deriving the rule, so the two can never disagree.
* ``random_state`` seeds a Rust ``SplitMix64``, not numpy's ``Generator``. A
  given seed is reproducible run-to-run within mlrs but does NOT reproduce
  sklearn's initialization — the same D-09 concession ``KMeans`` makes. Fitted
  RESULTS still match sklearn on a well-separated problem, which is what the
  oracle suite asserts.
"""

import numpy as np

from .base import MlrsBase


class GaussianMixture(MlrsBase):
    """EM fit of a ``n_components``-component Gaussian mixture (MIX-01).

    Supports all four ``covariance_type`` parameterizations and all four
    ``init_params`` routes.
    """

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        output_type="input",
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.output_type = output_type

    # -- fit ------------------------------------------------------------- #

    def _build(self):
        """Construct the ``_mlrs`` object from the stored hyperparameters.

        The injected inits cross the boundary as flat ``float64`` lists: PyO3
        extracts ``Vec<f64>``, and the Rust side keeps every parameter in
        ``f64`` regardless of the design's dtype, so flattening here loses
        nothing.
        """

        def flat(v):
            if v is None:
                return None
            return np.asarray(v, dtype=np.float64).ravel().tolist()

        seed = None if self.random_state is None else int(self.random_state)
        return self._ext().GaussianMixture(
            self.n_components,
            self.covariance_type,
            self.tol,
            self.reg_covar,
            self.max_iter,
            self.n_init,
            self.init_params,
            flat(self.weights_init),
            flat(self.means_init),
            flat(self.precisions_init),
            seed,
            self.warm_start,
            int(self.verbose),
            int(self.verbose_interval),
        )

    def fit(self, X, y=None):
        """Fit the mixture. With ``warm_start=True`` a second call RESUMES.

        The resume works by reusing the same underlying ``_mlrs`` object, which
        is what carries the previous fit's parameter block. One consequence is
        worth knowing: on that reuse path the hyperparameters are the ones the
        object was BUILT with, so changing them via ``set_params`` between two
        warm-started fits does not take effect until a cold fit rebuilds it.
        Every other path (``warm_start=False``, or the first fit) reads the
        current values.
        """
        xa, rows, cols = self._normalize(X)
        obj = getattr(self, "_mlrs_obj", None)
        if obj is None or not self.warm_start:
            obj = self._build()
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def fit_predict(self, X, y=None):
        """``fit`` then the training-set assignment — for free.

        ``fit`` already runs sklearn's terminal E-step (the one that makes the
        returned labels reflect the FINAL parameters rather than the last
        iteration's), so this reads the stored result instead of scoring again.
        """
        self.fit(X, y)
        return self._to_output(self._mlrs_obj.labels_(), (-1,), X, np.int32)

    # -- inference -------------------------------------------------------- #

    def _predict_X(self, X):
        """Normalize a query design to the FITTED dtype arm.

        Every scoring method on the Rust side dispatches on the design's dtype
        and must land on the same arm the estimator was fitted as, so a model
        fitted on `float64` scored with a `float32` query would otherwise fail
        with a confusing 'not fitted'. sklearn simply accepts the mixed case, so
        the coercion happens here.
        """
        return self._check_predict_X(X, dtype=self._np_float())

    def predict(self, X):
        xa, rows, cols = self._predict_X(X)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    def predict_proba(self, X):
        xa, rows, cols = self._predict_X(X)
        out = self._suffixed("predict_proba")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components), X, self._np_float()
        )

    def predict_log_proba(self, X):
        xa, rows, cols = self._predict_X(X)
        out = self._suffixed("predict_log_proba")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components), X, self._np_float()
        )

    def score_samples(self, X):
        xa, rows, cols = self._predict_X(X)
        out = self._mlrs_obj.score_samples(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.float64)

    def score(self, X, y=None):
        xa, rows, cols = self._predict_X(X)
        return float(self._mlrs_obj.score(xa, rows, cols))

    def bic(self, X):
        xa, rows, cols = self._predict_X(X)
        return float(self._mlrs_obj.bic(xa, rows, cols))

    def aic(self, X):
        xa, rows, cols = self._predict_X(X)
        return float(self._mlrs_obj.aic(xa, rows, cols))

    def sample(self, n_samples=1, seed=None):
        """Draw ``n_samples`` from the fitted mixture, returning ``(X, y)``.

        Defaults to the estimator's own ``random_state``, exactly as sklearn's
        ``sample`` does — ``seed`` is an mlrs-only override for callers who want
        a different draw without rebuilding the estimator. Either way the stream
        is a Rust ``SplitMix64``, so a given seed is reproducible within mlrs but
        does NOT reproduce numpy's draw (the D-09 concession the whole package
        makes about RNG).
        """
        self._check_fitted()
        if seed is None:
            seed = 0 if self.random_state is None else int(self.random_state)
        flat, y = self._suffixed("sample")(int(n_samples), int(seed))
        x = np.asarray(flat, dtype=self._np_float()).reshape(
            n_samples, self.n_features_in_
        )
        return x, np.asarray(y, dtype=np.int32)

    # -- fitted attributes ------------------------------------------------ #

    @property
    def weights_(self):
        return self._to_output(
            self._suffixed("weights")(), (-1,), None, self._np_float()
        )

    @property
    def means_(self):
        return self._to_output(
            self._suffixed("means")(),
            (self.n_components, -1),
            None,
            self._np_float(),
        )

    @property
    def covariances_(self):
        self._check_fitted()
        shape = tuple(self._mlrs_obj.covariance_shape())
        return self._to_output(
            self._suffixed("covariances")(), shape, None, self._np_float()
        )

    @property
    def precisions_(self):
        self._check_fitted()
        shape = tuple(self._mlrs_obj.covariance_shape())
        return self._to_output(
            self._suffixed("precisions")(), shape, None, self._np_float()
        )

    @property
    def precisions_cholesky_(self):
        self._check_fitted()
        shape = tuple(self._mlrs_obj.covariance_shape())
        return self._to_output(
            self._suffixed("precisions_cholesky")(),
            shape,
            None,
            self._np_float(),
        )

    @property
    def converged_(self):
        self._check_fitted()
        return bool(self._mlrs_obj.converged())

    @property
    def n_iter_(self):
        self._check_fitted()
        return int(self._mlrs_obj.n_iter())

    @property
    def lower_bound_(self):
        self._check_fitted()
        return float(self._mlrs_obj.lower_bound())

    @property
    def lower_bounds_(self):
        """Per-iteration ``lower_bound_`` trace of the WINNING restart.

        Length ``n_iter_``. With ``n_init > 1`` this is the trace of the restart
        that produced ``lower_bound_``, not of the last one run — so plotting it
        shows the ascent that was actually adopted.
        """
        self._check_fitted()
        return np.asarray(self._mlrs_obj.lower_bounds(), dtype=np.float64)

    def _n_parameters(self):
        """sklearn's private free-parameter count, used by ``bic``/``aic``.

        Exposed with sklearn's own (underscored) spelling because downstream
        model-selection code reads it directly.
        """
        self._check_fitted()
        return int(self._mlrs_obj.n_parameters())
