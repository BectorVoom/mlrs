"""Naive-Bayes estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB ->
``ClassifierMixin``. Each subclasses :class:`MlrsBase` + ``ClassifierMixin`` with
a sklearn-faithful ``__init__`` storing every ctor arg verbatim under the SAME
name (purity rule — the AST gate at ``tests/test_params.py`` enforces this).
``fit`` normalizes via the base, constructs the matching ``_mlrs.Py*NB``
wrapper, stores the handle on ``self._mlrs_obj`` and returns ``self`` (PY-01);
``classes_`` is materialized from the wrapper ``classes_()`` getter. ``predict``
forwards to the dtype-agnostic ``predict_labels``; ``predict_proba`` /
``predict_log_proba`` to the dtype-suffixed accessors (D-06).

The defaults mirror each ``Py*NB`` ``#[new]`` signature in
``crates/mlrs-py/src/estimators/naive_bayes.rs`` (D-02 sklearn defaults).
"""

import numpy as np
from sklearn.base import ClassifierMixin

from .base import MlrsBase


class _BaseNB(ClassifierMixin, MlrsBase):
    """Shared predict/predict_proba surface for the NB family.

    Subclasses provide a pure ``__init__`` and a ``fit`` that builds the matching
    ``_mlrs`` wrapper. All NB wrappers expose the same accessor surface
    (``predict_labels`` / ``predict_proba_f{32,64}`` / ``predict_log_proba_f{32,64}``
    / ``classes_()``), so the predict-side methods live here once.
    """

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    def predict_proba(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict_proba")(xa, rows, cols)
        n_classes = int(self.classes_.shape[0])
        return self._to_output(out, (rows, n_classes), X, self._np_float())

    def predict_log_proba(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict_log_proba")(xa, rows, cols)
        n_classes = int(self.classes_.shape[0])
        return self._to_output(out, (rows, n_classes), X, self._np_float())

    def _store_fit(self, obj, cols):
        """Common post-fit bookkeeping for the NB wrappers."""
        self._mlrs_obj = obj
        self._post_fit(cols)
        self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)


class GaussianNB(_BaseNB):
    """Gaussian naive Bayes (NB-01). ``GaussianNB(var_smoothing=1e-9, priors=None)``."""

    def __init__(self, var_smoothing=1e-9, priors=None, output_type="input"):
        self.var_smoothing = var_smoothing
        self.priors = priors
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=self._x_float(xa))
        obj = self._ext().GaussianNB(self.var_smoothing, self.priors)
        obj.fit(xa, ya, rows, cols)
        self._store_fit(obj, cols)
        return self

    @staticmethod
    def _x_float(xa):
        return np.float32 if xa.type.bit_width == 32 else np.float64


class MultinomialNB(_BaseNB):
    """Multinomial naive Bayes (NB-02).

    ``MultinomialNB(alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None)``.
    """

    def __init__(
        self,
        alpha=1.0,
        force_alpha=True,
        fit_prior=True,
        class_prior=None,
        output_type="input",
    ):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=GaussianNB._x_float(xa))
        obj = self._ext().MultinomialNB(
            self.alpha, self.force_alpha, self.fit_prior, self.class_prior
        )
        obj.fit(xa, ya, rows, cols)
        self._store_fit(obj, cols)
        return self


class BernoulliNB(_BaseNB):
    """Bernoulli naive Bayes (NB-03).

    ``BernoulliNB(alpha=1.0, force_alpha=True, binarize=0.0, fit_prior=True,
    class_prior=None)``.
    """

    def __init__(
        self,
        alpha=1.0,
        force_alpha=True,
        binarize=0.0,
        fit_prior=True,
        class_prior=None,
        output_type="input",
    ):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=GaussianNB._x_float(xa))
        obj = self._ext().BernoulliNB(
            self.alpha,
            self.force_alpha,
            self.binarize,
            self.fit_prior,
            self.class_prior,
        )
        obj.fit(xa, ya, rows, cols)
        self._store_fit(obj, cols)
        return self


class ComplementNB(_BaseNB):
    """Complement naive Bayes (NB-04).

    ``ComplementNB(alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
    norm=False)``.
    """

    def __init__(
        self,
        alpha=1.0,
        force_alpha=True,
        fit_prior=True,
        class_prior=None,
        norm=False,
        output_type="input",
    ):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=GaussianNB._x_float(xa))
        obj = self._ext().ComplementNB(
            self.alpha,
            self.force_alpha,
            self.fit_prior,
            self.class_prior,
            self.norm,
        )
        obj.fit(xa, ya, rows, cols)
        self._store_fit(obj, cols)
        return self


class CategoricalNB(_BaseNB):
    """Categorical naive Bayes (NB-05).

    ``CategoricalNB(alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
    min_categories=None)``. ``min_categories`` is stored verbatim and resolved
    (int / per-feature array / ``None``) by the ``_mlrs`` ctor.
    """

    def __init__(
        self,
        alpha=1.0,
        force_alpha=True,
        fit_prior=True,
        class_prior=None,
        min_categories=None,
        output_type="input",
    ):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.min_categories = min_categories
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=GaussianNB._x_float(xa))
        obj = self._ext().CategoricalNB(
            self.alpha,
            self.force_alpha,
            self.fit_prior,
            self.class_prior,
            self.min_categories,
        )
        obj.fit(xa, ya, rows, cols)
        self._store_fit(obj, cols)
        return self
