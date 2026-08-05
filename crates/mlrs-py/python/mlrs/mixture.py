"""Gaussian mixture model shims (MIX-01, MIX-02) delegating to ``_mlrs``.

Two estimators: ``GaussianMixture`` (maximum likelihood) and
``BayesianGaussianMixture`` (variational Bayes over the same model, with a
conjugate prior on every block and an ``n_components`` that acts as an upper
bound rather than a count). They share the Rust compute engine, and everything
below about shapes and RNG applies to both.

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


class BayesianGaussianMixture(MlrsBase):
    """Variational-Bayes fit of a Gaussian mixture (MIX-02).

    The variational sibling of :class:`GaussianMixture`: same four
    ``covariance_type`` parameterizations and same four ``init_params`` routes,
    but with a conjugate prior on every block. ``n_components`` is an UPPER
    BOUND rather than a count — with
    ``weight_concentration_prior_type='dirichlet_process'`` (the default) and a
    small ``weight_concentration_prior``, components that explain nothing are
    driven to near-zero weight instead of splitting real clusters.

    Two shape asymmetries are inherited from sklearn rather than smoothed over,
    because downstream code reads them:

    - ``weight_concentration_`` is a 2-TUPLE of arrays under
      ``dirichlet_process`` (the two Beta parameters of each stick break) and a
      single array under ``dirichlet_distribution``.
    - ``degrees_of_freedom_`` is a scalar under ``covariance_type='tied'``
      (one shared Wishart) and an ``n_components`` array otherwise.

    There is no ``bic`` / ``aic``: sklearn defines them on
    :class:`GaussianMixture` only, the variational model having no well-defined
    free-parameter count.
    """

    def __init__(
        self,
        *,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        output_type="input",
    ):
        # `n_components` is KEYWORD-ONLY here, matching sklearn's signature
        # (`GaussianMixture` takes it positionally; this one does not).
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.output_type = output_type

    # -- fit ------------------------------------------------------------- #

    def _build(self):
        """Construct the ``_mlrs`` object from the stored hyperparameters.

        ``mean_prior`` / ``covariance_prior`` cross the boundary as flat
        ``float64`` lists: PyO3 extracts ``Vec<f64>``, and the Rust side keeps
        every parameter in ``f64`` regardless of the design's dtype, so
        flattening here loses nothing. A scalar ``covariance_prior`` (the
        ``spherical`` case) flattens to a one-element list.
        """

        def flat(v):
            if v is None:
                return None
            return np.asarray(v, dtype=np.float64).ravel().tolist()

        def scalar(v):
            return None if v is None else float(v)

        seed = None if self.random_state is None else int(self.random_state)
        return self._ext().BayesianGaussianMixture(
            self.n_components,
            self.covariance_type,
            self.tol,
            self.reg_covar,
            self.max_iter,
            self.n_init,
            self.init_params,
            self.weight_concentration_prior_type,
            scalar(self.weight_concentration_prior),
            scalar(self.mean_precision_prior),
            flat(self.mean_prior),
            scalar(self.degrees_of_freedom_prior),
            flat(self.covariance_prior),
            seed,
            self.warm_start,
            int(self.verbose),
            int(self.verbose_interval),
        )

    def fit(self, X, y=None):
        """Fit the mixture. With ``warm_start=True`` a second call RESUMES.

        Same reuse rule as :meth:`GaussianMixture.fit`: the resume works by
        reusing the underlying ``_mlrs`` object, so on that path the
        hyperparameters are the ones the object was BUILT with and a
        ``set_params`` between two warm-started fits does not take effect until
        a cold fit rebuilds it.
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
        """``fit`` then the training-set assignment — for free."""
        self.fit(X, y)
        return self._to_output(self._mlrs_obj.labels_(), (-1,), X, np.int32)

    # -- inference -------------------------------------------------------- #

    def _predict_X(self, X):
        """Normalize a query design to the FITTED dtype arm.

        Same reason as :meth:`GaussianMixture._predict_X`: every scoring method
        dispatches on the design's dtype and must land on the arm the estimator
        was fitted as.
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

    def sample(self, n_samples=1, seed=None):
        """Draw ``n_samples`` from the fitted mixture, returning ``(X, y)``.

        Defaults to the estimator's own ``random_state``; ``seed`` is an
        mlrs-only override. Either way the stream is a Rust ``SplitMix64``, so a
        given seed is reproducible within mlrs but does NOT reproduce numpy's
        draw (the D-09 concession the whole package makes about RNG).
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
    def weight_concentration_(self):
        """The weight posterior, in sklearn's two SHAPES.

        A ``(a, b)`` tuple of arrays under ``dirichlet_process`` — the Beta
        parameters of each stick break — and a single array under
        ``dirichlet_distribution``, where there is no second parameter. The Rust
        side always returns the pair and leaves ``b`` empty in the second case,
        so the branch happens here rather than in two accessors.
        """
        self._check_fitted()
        a, b = self._mlrs_obj.weight_concentration()
        a = np.asarray(a, dtype=np.float64)
        if len(b) == 0:
            return a
        return (a, np.asarray(b, dtype=np.float64))

    @property
    def mean_precision_(self):
        self._check_fitted()
        return np.asarray(self._mlrs_obj.mean_precision(), dtype=np.float64)

    @property
    def degrees_of_freedom_(self):
        """``ν``: a SCALAR under ``covariance_type='tied'``, else an array.

        sklearn's asymmetry, reproduced rather than normalized — under ``tied``
        all components share one Wishart, so there is one value and downstream
        code indexes it as a scalar.
        """
        self._check_fitted()
        v = np.asarray(self._mlrs_obj.degrees_of_freedom(), dtype=np.float64)
        if len(self._mlrs_obj.degrees_of_freedom_shape()) == 0:
            return v[0]
        return v

    def _prior_tuple(self):
        self._check_fitted()
        return self._mlrs_obj.priors()

    @property
    def weight_concentration_prior_(self):
        return float(self._prior_tuple()[0])

    @property
    def mean_precision_prior_(self):
        return float(self._prior_tuple()[1])

    @property
    def mean_prior_(self):
        return np.asarray(self._prior_tuple()[2], dtype=np.float64)

    @property
    def degrees_of_freedom_prior_(self):
        return float(self._prior_tuple()[3])

    @property
    def covariance_prior_(self):
        """``W₀``, in the ``covariance_type``'s shape.

        ``(d, d)`` for ``full``/``tied``, ``(d,)`` for ``diag``, and a plain
        scalar for ``spherical`` — the same shapes sklearn stores.
        """
        v = np.asarray(self._prior_tuple()[4], dtype=np.float64)
        d = self.n_features_in_
        if self.covariance_type in ("full", "tied"):
            return v.reshape(d, d)
        if self.covariance_type == "diag":
            return v
        return v[0]

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
        """The evidence lower bound — NOT a log-likelihood.

        sklearn drops every constant term from the bound, so this is not
        comparable to :attr:`GaussianMixture.lower_bound_` and is not
        necessarily negative. It is monotone in the variational objective, which
        is what the convergence test needs.
        """
        self._check_fitted()
        return float(self._mlrs_obj.lower_bound())

    @property
    def lower_bounds_(self):
        """Per-iteration ``lower_bound_`` trace of the WINNING restart."""
        self._check_fitted()
        return np.asarray(self._mlrs_obj.lower_bounds(), dtype=np.float64)
