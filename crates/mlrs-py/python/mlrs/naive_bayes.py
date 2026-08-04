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

    # -- sklearn tags: the family's DOMAIN, declared (RESEARCH §estimator_checks)

    #: Whether this variant rejects negative feature values. sklearn's own NB
    #: classes set ``input_tags.positive_only`` for exactly the three that call
    #: ``check_non_negative`` on X (Multinomial / Complement / Categorical) and
    #: leave it False for Gaussian (unrestricted) and Bernoulli (which binarizes,
    #: so a negative value is meaningful input, not an error). Subclasses
    #: override; the default is the unrestricted one.
    _POSITIVE_ONLY = False

    #: Whether to exempt this variant from ``check_classifiers_train``'s
    #: ``accuracy > 0.83`` floor. sklearn sets it on the four DISCRETE variants
    #: and leaves it off for Gaussian, which scores well on that fixture; the
    #: default here matches the discrete majority and Gaussian overrides back.
    _POOR_SCORE = True

    def __sklearn_tags__(self):
        """Declare the two family-wide tags sklearn's own NB classes carry.

        Both were previously left at their defaults, and each silently enrolled
        the estimator in checks it cannot pass — the same failure mode as an
        estimator that under-declares any other capability:

        * ``input_tags.positive_only`` tells ``_enforce_estimator_tags_X`` to
          shift a fixture to be non-negative. Without it the harness feeds
          negative X to a variant that (correctly, and exactly as sklearn does)
          rejects it, and every check that fits reports the rejection as a
          failure.
        * ``classifier_tags.poor_score`` exempts the estimator from
          ``check_classifiers_train``'s ``accuracy > 0.83`` floor. Naive Bayes on
          that check's blob fixture scores ~0.79 — mlrs reproduces sklearn's
          predictions EXACTLY there, so the floor is a statement about the
          algorithm, not about this implementation, which is why sklearn sets
          the tag on all four discrete variants rather than weakening the check.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.positive_only = self._POSITIVE_ONLY
        tags.classifier_tags.poor_score = self._POOR_SCORE
        return tags


class GaussianNB(_BaseNB):
    """Gaussian naive Bayes (NB-01). ``GaussianNB(var_smoothing=1e-9, priors=None)``."""

    #: sklearn's GaussianNB carries neither discrete-family tag: it accepts
    #: negative features and clears the accuracy floor.
    _POOR_SCORE = False

    def __init__(self, var_smoothing=1e-9, priors=None, output_type="input"):
        self.var_smoothing = var_smoothing
        self.priors = priors
        self.output_type = output_type

    def fit(self, X, y, sample_weight=None):
        # ``sample_weight`` (optional, length n_samples) weights each row's
        # contribution to the per-class counts, matching sklearn's
        # ``Y *= sample_weight.T``. It rides the same 1-D float ingress as ``y``
        # and is validated in RUST (length, finite, non-negative, not all zero
        # -- ``linear/ridge.rs::validate_sample_weight``), so a bad weight
        # raises the same ``ValueError`` class sklearn raises.
        # ``ensure_all_finite=False`` does NOT skip the NaN/inf rejection: the
        # Rust fit's fused count sweep reads every element of ``X`` anyway, so it
        # reports the same verdict from that sweep and the PyO3 arm raises
        # ``check_array``'s exact ``ValueError`` itself
        # (``estimators/naive_bayes.rs::nb_host_fit_err``). ``check_array``'s own
        # scan is a second single-threaded trip over the whole matrix. ``y``
        # keeps its scan: it is 1-D and the label decode has no equivalent
        # hand-off.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = self._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype)
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=dtype)
        )
        obj = self._ext().GaussianNB(self.var_smoothing, self.priors)
        obj.fit(xa, ya, rows, cols, swa)
        self._store_fit(obj, cols)
        return self

    @staticmethod
    def _x_float(xa):
        return np.float32 if xa.type.bit_width == 32 else np.float64


class MultinomialNB(_BaseNB):
    """Multinomial naive Bayes (NB-02).

    ``MultinomialNB(alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None)``.
    """

    #: sklearn's MultinomialNB calls ``check_non_negative`` on X.
    _POSITIVE_ONLY = True

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

    def fit(self, X, y, sample_weight=None):
        # ``sample_weight`` (optional, length n_samples) weights each row's
        # contribution to the per-class counts, matching sklearn's
        # ``Y *= sample_weight.T``. It rides the same 1-D float ingress as ``y``
        # and is validated in RUST (length, finite, non-negative, not all zero
        # -- ``linear/ridge.rs::validate_sample_weight``), so a bad weight
        # raises the same ``ValueError`` class sklearn raises.
        # ``ensure_all_finite=False`` does NOT skip the NaN/inf rejection: the
        # Rust fit's fused count sweep reads every element of ``X`` anyway, so it
        # reports the same verdict from that sweep and the PyO3 arm raises
        # ``check_array``'s exact ``ValueError`` itself
        # (``estimators/naive_bayes.rs::nb_host_fit_err``). ``check_array``'s own
        # scan is a second single-threaded trip over the whole matrix. ``y``
        # keeps its scan: it is 1-D and the label decode has no equivalent
        # hand-off.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = GaussianNB._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype)
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=dtype)
        )
        obj = self._ext().MultinomialNB(
            self.alpha, self.force_alpha, self.fit_prior, self.class_prior
        )
        obj.fit(xa, ya, rows, cols, swa)
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

    def fit(self, X, y, sample_weight=None):
        # ``sample_weight`` (optional, length n_samples) weights each row's
        # contribution to the per-class counts, matching sklearn's
        # ``Y *= sample_weight.T``. It rides the same 1-D float ingress as ``y``
        # and is validated in RUST (length, finite, non-negative, not all zero
        # -- ``linear/ridge.rs::validate_sample_weight``), so a bad weight
        # raises the same ``ValueError`` class sklearn raises.
        # ``ensure_all_finite=False`` does NOT skip the NaN/inf rejection: the
        # Rust fit's fused count sweep reads every element of ``X`` anyway (it
        # has to — every value is thresholded into a per-class count), so it
        # reports the same verdict from that sweep and the PyO3 arm raises
        # ``check_array``'s exact ``ValueError`` itself
        # (``estimators/naive_bayes.rs::nb_host_fit_err``). ``check_array``'s own
        # scan is a second single-threaded trip over the whole matrix — one of
        # the largest remaining costs of a fit once the counting went
        # single-pass. ``y`` keeps its scan: it is 1-D and the label decode has
        # no equivalent hand-off.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = GaussianNB._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype)
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=dtype)
        )
        obj = self._ext().BernoulliNB(
            self.alpha,
            self.force_alpha,
            self.binarize,
            self.fit_prior,
            self.class_prior,
        )
        obj.fit(xa, ya, rows, cols, swa)
        self._store_fit(obj, cols)
        return self


class ComplementNB(_BaseNB):
    """Complement naive Bayes (NB-04).

    ``ComplementNB(alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
    norm=False)``.
    """

    #: sklearn's ComplementNB calls ``check_non_negative`` on X.
    _POSITIVE_ONLY = True

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

    def fit(self, X, y, sample_weight=None):
        # ``sample_weight`` (optional, length n_samples) weights each row's
        # contribution to the per-class counts, matching sklearn's
        # ``Y *= sample_weight.T``. It rides the same 1-D float ingress as ``y``
        # and is validated in RUST (length, finite, non-negative, not all zero
        # -- ``linear/ridge.rs::validate_sample_weight``), so a bad weight
        # raises the same ``ValueError`` class sklearn raises.
        # ``ensure_all_finite=False`` does NOT skip the NaN/inf rejection: the
        # Rust fit's fused count sweep reads every element of ``X`` anyway, so it
        # reports the same verdict from that sweep and the PyO3 arm raises
        # ``check_array``'s exact ``ValueError`` itself
        # (``estimators/naive_bayes.rs::nb_host_fit_err``). ``check_array``'s own
        # scan is a second single-threaded trip over the whole matrix. ``y``
        # keeps its scan: it is 1-D and the label decode has no equivalent
        # hand-off.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = GaussianNB._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype)
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=dtype)
        )
        obj = self._ext().ComplementNB(
            self.alpha,
            self.force_alpha,
            self.fit_prior,
            self.class_prior,
            self.norm,
        )
        obj.fit(xa, ya, rows, cols, swa)
        self._store_fit(obj, cols)
        return self


class CategoricalNB(_BaseNB):
    """Categorical naive Bayes (NB-05).

    ``CategoricalNB(alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None,
    min_categories=None)``. ``min_categories`` is stored verbatim and resolved
    (int / per-feature array / ``None``) by the ``_mlrs`` ctor.
    """

    #: sklearn's CategoricalNB calls ``check_non_negative`` on X.
    _POSITIVE_ONLY = True

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

    def fit(self, X, y, sample_weight=None):
        # ``sample_weight`` (optional, length n_samples) weights each row's
        # contribution to the per-class counts, matching sklearn's
        # ``Y *= sample_weight.T``. It rides the same 1-D float ingress as ``y``
        # and is validated in RUST (length, finite, non-negative, not all zero
        # -- ``linear/ridge.rs::validate_sample_weight``), so a bad weight
        # raises the same ``ValueError`` class sklearn raises.
        # ``ensure_all_finite=False`` does NOT skip the NaN/inf rejection: the
        # Rust fit's validation pass reads every element of ``X`` anyway (it has
        # to — a category index must be a non-negative integer), so it reports
        # the same verdict from that pass and the PyO3 arm raises
        # ``check_array``'s exact ``ValueError`` itself
        # (``estimators/naive_bayes.rs::categorical_fit_err``). ``check_array``'s
        # own scan is a second single-threaded trip over the whole matrix — the
        # largest remaining cost of a CategoricalNB fit once the tabulation went
        # row-major. ``y`` keeps its scan: it is 1-D and the label decode has no
        # equivalent hand-off.
        xa, rows, cols = self._normalize(X, ensure_all_finite=False)
        dtype = GaussianNB._x_float(xa)
        ya = self._normalize_y(y, dtype=dtype)
        swa = (
            None
            if sample_weight is None
            else self._normalize_y(sample_weight, dtype=dtype)
        )
        obj = self._ext().CategoricalNB(
            self.alpha,
            self.force_alpha,
            self.fit_prior,
            self.class_prior,
            self.min_categories,
        )
        obj.fit(xa, ya, rows, cols, swa)
        self._store_fit(obj, cols)
        return self
