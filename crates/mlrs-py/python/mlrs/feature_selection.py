"""``mlrs.feature_selection`` — the sklearn-faithful selector surface (FSEL-01).

============================  ============================================
name                          provenance
============================  ============================================
:func:`f_classif`             ``_mlrs`` (Rust)
:func:`f_oneway`              ``_mlrs``
:func:`chi2`                  ``_mlrs``
:func:`r_regression`          ``_mlrs``
:func:`f_regression`          ``_mlrs``
:func:`mutual_info_classif`   ``_mlrs``
:func:`mutual_info_regression`  ``_mlrs``
:class:`VarianceThreshold`    ``_mlrs`` fit + container-native transform
:class:`SelectKBest`          ``_mlrs``
:class:`SelectPercentile`     ``_mlrs``
:class:`SelectFpr`            ``_mlrs``
:class:`SelectFdr`            ``_mlrs``
:class:`SelectFwe`            ``_mlrs``
:class:`GenericUnivariateSelect`  ``_mlrs``
:class:`SelectFromModel`      sklearn fit logic + mlrs egress (see below)
:class:`RFE`                  ditto
:class:`RFECV`                ditto
:class:`SequentialFeatureSelector`  ditto
:class:`SelectorMixin`        :class:`MlrsSelectorMixin`, here
============================  ============================================

## Container support: numpy, pandas, polars, pyarrow, python lists

Every selector here accepts all five and — under the default
``output_type="input"`` — returns ``transform``'s result as the SAME kind of
container, with column names and per-column dtypes intact
(:mod:`mlrs._frame`). A polars frame in gives a polars frame out.

This is a deliberate divergence from sklearn, which returns numpy from
``transform`` unless you opt into ``set_output(transform="pandas")``. It is the
package-wide ``output_type`` contract (D-03, "egress mirrors ingress"), and
``output_type="numpy"`` restores sklearn's exact behaviour for a caller who wants
it. Numpy input gives numpy output either way, which is why the sklearn
``estimator_checks`` harness sees no difference.

The SCORES are always computed on a ``float64`` view of the data
(:func:`mlrs._io.normalize_X` widens once at the boundary), because the Rust
score functions accumulate in ``f64`` — their p-values reach 1e-27 and the 1e-5
contract is relative. So a mixed-dtype frame is scored as float64 and
``transform``ed in its original dtypes, which is both correct and what a caller
expects.

## Why the four META-selectors reuse sklearn's fit

``SelectFromModel`` / ``RFE`` / ``RFECV`` / ``SequentialFeatureSelector`` take an
arbitrary ``estimator``, and their own arithmetic is a handful of ``argsort``s
over ``n_features`` values — ALL of their cost is the inner estimator's ``fit``,
which is whatever the caller passed (an mlrs estimator running device kernels, a
sklearn one, anything duck-typed). There is nothing there to accelerate, and
re-implementing sklearn's elimination bookkeeping in Python would add a second
place for it to be subtly wrong without making anything faster.

So these four subclass sklearn's classes for the FIT and mix in
:class:`MlrsSelectorMixin` for the container-aware egress. That is the same
"passthrough with provenance recorded" decision :mod:`mlrs.model_selection` makes
for ``GridSearchCV`` / ``cross_val_score``. mlrs's Rust layer has its own native
implementations of all four (``mlrs_algos::feature_selection::meta``) for Rust
callers, where there is no duck typing to lean on.

``mlrs``'s own Rust implementations are NOT used from Python for these four
precisely because they would need to call a Python estimator back from Rust per
elimination step, re-acquiring the GIL inside the fit — strictly worse than
letting Python drive.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE as _SkRFE
from sklearn.feature_selection import RFECV as _SkRFECV
from sklearn.feature_selection import SelectFromModel as _SkSelectFromModel
from sklearn.feature_selection import (
    SequentialFeatureSelector as _SkSequentialFeatureSelector,
)
from sklearn.utils.validation import check_is_fitted

from . import _frame, _io

# ------------------------------------------------------------------------- #
# Score functions
# ------------------------------------------------------------------------- #


def _ext():
    """The compiled extension (lazy, so this module imports without it)."""
    from . import _load_ext

    return _load_ext()


def _as_xy(X, y, *, ensure_all_finite=True):
    """``(flat float64 X, flat float64 y, rows, cols)`` for the Rust call.

    ``float64`` unconditionally, not the input's dtype: the Rust scores accumulate
    in ``f64`` regardless (their p-values reach 1e-27 and the oracle contract is
    RELATIVE), so widening at the boundary is the only place it can happen once.
    """
    xa, rows, cols = _io.normalize_X(
        X, dtype=np.float64, ensure_all_finite=ensure_all_finite
    )
    flat = np.asarray(xa).astype(np.float64, copy=False).ravel()
    if y is None:
        return flat, None, rows, cols
    yv = np.asarray(_io.normalize_y(y, dtype=np.float64)).astype(
        np.float64, copy=False
    )
    return flat, yv, rows, cols


def f_classif(X, y):
    """ANOVA F-value of each feature against the class label.

    Returns ``(f_statistic, p_values)``, matching
    ``sklearn.feature_selection.f_classif``.
    """
    x, yv, rows, cols = _as_xy(X, y)
    scores, pvalues = _ext().f_classif(x, yv, rows, cols)
    return np.asarray(scores), np.asarray(pvalues)


def f_oneway(*args):
    """One-way ANOVA over two or more sample groups.

    Variadic like sklearn's: each argument is an ``(n_k, n_features)`` group. The
    groups are concatenated in the order given, which is the order
    :func:`f_classif` produces them in (``np.unique(y)``), so the two agree.
    """
    if len(args) < 2:
        raise ValueError("f_oneway needs at least 2 sample groups")
    blocks = [np.ascontiguousarray(np.asarray(a, dtype=np.float64)) for a in args]
    cols = blocks[0].shape[1]
    if any(b.shape[1] != cols for b in blocks):
        raise ValueError("f_oneway: every group must have the same n_features")
    sizes = [int(b.shape[0]) for b in blocks]
    flat = np.concatenate([b.ravel() for b in blocks])
    scores, pvalues = _ext().f_oneway(flat, sizes, int(cols))
    return np.asarray(scores), np.asarray(pvalues)


def chi2(X, y):
    """Chi-squared statistic of each NON-NEGATIVE feature against the class."""
    x, yv, rows, cols = _as_xy(X, y)
    scores, pvalues = _ext().chi2(x, yv, rows, cols)
    return np.asarray(scores), np.asarray(pvalues)


def r_regression(X, y, *, center=True, force_finite=True):
    """Pearson's r between each feature and the target. Scores only."""
    x, yv, rows, cols = _as_xy(X, y)
    return np.asarray(_ext().r_regression(x, yv, rows, cols, center, force_finite))


def f_regression(X, y, *, center=True, force_finite=True):
    """Univariate linear-regression F-statistic and p-value per feature."""
    x, yv, rows, cols = _as_xy(X, y)
    scores, pvalues = _ext().f_regression(x, yv, rows, cols, center, force_finite)
    return np.asarray(scores), np.asarray(pvalues)


def _discrete_args(discrete_features, n_features):
    """Resolve sklearn's ``discrete_features`` into the ``(all, mask)`` pair Rust takes.

    sklearn accepts ``"auto"``, a bool, a boolean MASK, or an INDEX ARRAY. The
    index-array form is resolved to a mask here, where numpy is already imported —
    resolving it on both sides would let the two layers disagree about an edge
    case like an empty index array.
    """
    if isinstance(discrete_features, str):
        if discrete_features != "auto":
            raise ValueError("Invalid string value for discrete_features.")
        return None, None
    if isinstance(discrete_features, (bool, np.bool_)):
        return bool(discrete_features), None
    arr = np.asarray(discrete_features)
    if arr.dtype == bool:
        if arr.shape[0] != n_features:
            raise ValueError(
                f"discrete_features has {arr.shape[0]} entries but X has "
                f"{n_features} features"
            )
        return None, [bool(v) for v in arr]
    mask = np.zeros(n_features, dtype=bool)
    mask[arr.astype(int)] = True
    return None, [bool(v) for v in mask]


def _mi_jobs(n_jobs):
    """sklearn's ``n_jobs``, with ``-1`` resolved to the machine's cpu count.

    ``-1`` cannot cross into Rust's ``usize``, and "all processors" is a host-side
    question anyway.
    """
    if n_jobs is None:
        return None
    n_jobs = int(n_jobs)
    if n_jobs < 0:
        import os

        return max(1, os.cpu_count() or 1)
    return n_jobs


def mutual_info_classif(
    X,
    y,
    *,
    discrete_features="auto",
    n_neighbors=3,
    copy=True,
    random_state=None,
    n_jobs=None,
):
    """Mutual information between each feature and a DISCRETE target.

    ``random_state`` reproduces ``numpy.random.RandomState(seed)`` bit-for-bit, so
    a given seed gives the same tie-breaking noise sklearn would draw. ``None``
    seeds ``0`` here rather than drawing from numpy's process-global stream — a
    deliberate divergence documented on the Rust side, chosen because a
    reproducible score is worth more than reproducing non-reproducible behaviour.
    """
    x, yv, rows, cols = _as_xy(X, y)
    d_all, d_mask = _discrete_args(discrete_features, cols)
    return np.asarray(
        _ext().mutual_info_classif(
            x,
            yv,
            rows,
            cols,
            d_all,
            d_mask,
            int(n_neighbors),
            bool(copy),
            None if random_state is None else int(random_state),
            _mi_jobs(n_jobs),
        )
    )


def mutual_info_regression(
    X,
    y,
    *,
    discrete_features="auto",
    n_neighbors=3,
    copy=True,
    random_state=None,
    n_jobs=None,
):
    """Mutual information between each feature and a CONTINUOUS target.

    See :func:`mutual_info_classif` on ``random_state``. Note that mlrs's k-NN
    mutual information is known to disagree with sklearn on columns containing
    MANY EXACTLY-TIED values; the Rust test
    ``mutual_info_on_tied_columns_diverges_from_sklearn`` records the measured
    size and scope of that gap.
    """
    x, yv, rows, cols = _as_xy(X, y)
    d_all, d_mask = _discrete_args(discrete_features, cols)
    return np.asarray(
        _ext().mutual_info_regression(
            x,
            yv,
            rows,
            cols,
            d_all,
            d_mask,
            int(n_neighbors),
            bool(copy),
            None if random_state is None else int(random_state),
            _mi_jobs(n_jobs),
        )
    )


#: The built-in score functions the compiled side can evaluate itself, keyed by
#: the Python callable so a user writing ``SelectKBest(chi2)`` — the sklearn
#: idiom — is recognised without having to pass a string. Anything NOT in here is
#: treated as a custom callable: the shim calls it and hands the resulting scores
#: to the Rust selection rule.
_BUILTIN_SCORE_FUNCS = {
    f_classif: "f_classif",
    chi2: "chi2",
    r_regression: "r_regression",
    f_regression: "f_regression",
    mutual_info_classif: "mutual_info_classif",
    mutual_info_regression: "mutual_info_regression",
}


def _builtin_name(score_func):
    """The Rust name for ``score_func``, or ``None`` if it is a custom callable.

    sklearn's own module-level functions are recognised too, by NAME: a user
    porting code will have ``from sklearn.feature_selection import chi2`` at the
    top of the file, and mlrs's ``chi2`` computes the same thing, so routing it to
    the compiled path is right — and strictly faster than calling back into
    sklearn.
    """
    if score_func in _BUILTIN_SCORE_FUNCS:
        return _BUILTIN_SCORE_FUNCS[score_func]
    name = getattr(score_func, "__name__", None)
    module = getattr(score_func, "__module__", "") or ""
    if name in _BUILTIN_SCORE_FUNCS.values() and module.startswith(
        ("sklearn.feature_selection", "mlrs.feature_selection")
    ):
        return name
    return None


# ------------------------------------------------------------------------- #
# The selector mixin
# ------------------------------------------------------------------------- #


class MlrsSelectorMixin(TransformerMixin):
    """``sklearn.feature_selection.SelectorMixin`` with mlrs's container egress.

    Like sklearn's mixin, everything derives from one method — here
    ``_get_support_mask()`` — so a selector supplies its mask and gets
    ``get_support`` / ``transform`` / ``inverse_transform`` /
    ``get_feature_names_out`` for free.

    The difference is the gather: sklearn's ``transform`` runs ``check_array`` and
    returns numpy, while this one dispatches to the input container's own column
    take (:func:`mlrs._frame.take_columns`) under the default
    ``output_type="input"``, so names and dtypes survive. ``output_type="numpy"``
    takes sklearn's route.
    """

    def get_support(self, indices=False):
        """Boolean mask (default) or integer indices of the selected features."""
        mask = self._get_support_mask()
        return np.nonzero(mask)[0] if indices else mask

    def _get_support_mask(self):  # pragma: no cover - abstract
        raise NotImplementedError(
            "a selector must implement _get_support_mask()"
        )

    def transform(self, X):
        """Reduce ``X`` to the selected features.

        Warns exactly as sklearn does when the mask is empty, and returns a
        zero-column container rather than raising.
        """
        check_is_fitted(self, attributes="n_features_in_")
        mask = np.asarray(self._get_support_mask(), dtype=bool)
        n = _n_features(X)
        if n != mask.shape[0]:
            raise ValueError(
                f"X has {n} features, but {type(self).__name__} is expecting "
                f"{mask.shape[0]} features as input."
            )
        if not mask.any():
            import warnings

            warnings.warn(
                "No features were selected: either the data is too noisy or the"
                " selection test too strict.",
                UserWarning,
                stacklevel=2,
            )
        if getattr(self, "output_type", "input") == "numpy":
            # sklearn's own behaviour: a plain numpy result regardless of the
            # input container. `to_numpy_2d` rather than `np.asarray` so a
            # pandas/polars frame goes through its own `to_numpy`, which handles
            # a mixed-dtype frame that `np.asarray` would turn into `object`.
            return _frame.to_numpy_2d(X)[:, mask]
        return _frame.take_columns(X, mask)

    def inverse_transform(self, Z):
        """Widen ``Z`` back to ``n_features_in_`` columns, dropped ones ZERO."""
        check_is_fitted(self, attributes="n_features_in_")
        mask = np.asarray(self._get_support_mask(), dtype=bool)
        return _frame.restore_columns(
            Z,
            mask,
            # The dropped columns' labels can only come from the FITTED
            # `feature_names_in_`: `Z` is the reduced frame and no longer carries
            # them.
            names=getattr(self, "feature_names_in_", None),
            mirror_container=getattr(self, "output_type", "input") != "numpy",
        )

    def get_feature_names_out(self, input_features=None):
        """Names of the selected features."""
        check_is_fitted(self, attributes="n_features_in_")
        names_in = (
            list(input_features)
            if input_features is not None
            else getattr(self, "feature_names_in_", None)
        )
        return _frame.feature_names_out(self._get_support_mask(), names_in)


def _n_features(X):
    """Column count of any supported container, without materialising it."""
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    names = getattr(X, "column_names", None)
    if names is not None:
        return len(names)
    return len(X[0])


class _MlrsSelectorBase(MlrsSelectorMixin, BaseEstimator):
    """Shared plumbing for the selectors mlrs fits itself (not the meta ones).

    ``__init__`` purity is preserved (every ctor argument stored verbatim under the
    same name, nothing else — sklearn's ``check_no_attributes_set_in_init``); the
    fitted state is a support mask plus whatever attributes the selector exposes.
    """

    def _record_fit(self, X, mask, n_features):
        self.n_features_in_ = int(n_features)
        names = _frame.column_names(X)
        if names is not None and all(isinstance(n, str) for n in names):
            self.feature_names_in_ = np.asarray(names, dtype=object)
        self.support_ = np.asarray(mask, dtype=bool)
        return self

    def _get_support_mask(self):
        check_is_fitted(self, attributes="support_")
        return self.support_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.array_api_support = False
        return tags


# ------------------------------------------------------------------------- #
# VarianceThreshold
# ------------------------------------------------------------------------- #


class VarianceThreshold(_MlrsSelectorBase):
    """Drop features whose variance is at or below ``threshold``.

    The one selector that ACCEPTS NaN input, as sklearn's does: it validates with
    ``ensure_all_finite="allow-nan"`` and computes ``np.nanvar``. At
    ``threshold == 0`` sklearn compares the peak-to-peak RANGE instead of the
    variance (to avoid a constant column surviving on `1e-17` of cancellation) and
    OVERWRITES ``variances_`` with the smaller of the two — both reproduced.
    """

    def __init__(self, threshold=0.0, *, output_type="input"):
        self.threshold = threshold
        self.output_type = output_type

    def fit(self, X, y=None):
        # `"allow-nan"` — NOT `False`. sklearn's `VarianceThreshold` validates
        # with exactly that string, so an INFINITY is still rejected while a NaN
        # passes through to the NaN-skipping Rust sweep. Passing `False` would
        # also admit infinities, which `np.nanvar` does not handle and sklearn
        # does not accept.
        x, _, rows, cols = _as_xy(X, None, ensure_all_finite="allow-nan")
        variances, mask = _ext().variance_threshold(
            x, rows, cols, float(self.threshold)
        )
        self.variances_ = np.asarray(variances)
        return self._record_fit(X, mask, cols)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.target_tags.required = False
        return tags


# ------------------------------------------------------------------------- #
# The univariate filters
# ------------------------------------------------------------------------- #


class _BaseFilter(_MlrsSelectorBase):
    """sklearn's ``_BaseFilter``: run ``score_func(X, y)``, then apply a mode.

    Subclasses supply ``_mode()`` and ``_param()``; everything else — including the
    custom-callable route — is here, mirroring how sklearn's subclasses supply only
    ``_get_support_mask``.
    """

    def _mode(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _param(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _score_kwargs(self):
        """Per-score-function options forwarded to the compiled call.

        Only meaningful for the score functions that HAVE options
        (``f_regression`` / ``r_regression``'s ``center`` and ``force_finite``,
        ``mutual_info_*``'s neighbours and seed). sklearn has no way to pass them
        through a selector either — a user who needs non-default options wraps the
        function in a lambda, which lands on the custom-callable route below — so
        these stay at sklearn's defaults.
        """
        return {}

    def fit(self, X, y=None):
        x, yv, rows, cols = _as_xy(X, y)
        builtin = _builtin_name(self.score_func)
        mode, param = self._mode(), self._param()

        if builtin is not None:
            if yv is None:
                raise ValueError(
                    f"{type(self).__name__}: score_func {builtin!r} is supervised "
                    "and requires y"
                )
            self._warn_if_k_exceeds(cols)
            scores, pvalues, mask = _ext().univariate_select(
                x, yv, rows, cols, mode, param, builtin, **self._score_kwargs()
            )
        else:
            # A custom callable: evaluate it HERE (Rust must never call back into
            # Python mid-fit) and hand the scores to the same Rust selection rule
            # the built-in path uses, so tie-breaking / percentile interpolation /
            # Benjamini-Hochberg stay in one implementation.
            ret = self.score_func(X, y)
            if isinstance(ret, (list, tuple)):
                scores, pvalues = ret
                pvalues = [float(v) for v in np.asarray(pvalues).ravel()]
            else:
                scores, pvalues = ret, None
            scores = [float(v) for v in np.asarray(scores).ravel()]
            self._warn_if_k_exceeds(cols)
            mask = _ext().univariate_select_from_scores(scores, pvalues, mode, param)

        self.scores_ = np.asarray(scores, dtype=np.float64)
        self.pvalues_ = None if pvalues is None else np.asarray(pvalues, dtype=np.float64)
        return self._record_fit(X, mask, cols)

    def _warn_if_k_exceeds(self, n_features):
        """sklearn's ``SelectKBest._check_params`` warning.

        Raised in Python rather than logged in Rust: sklearn's own tests assert it
        with ``pytest.warns``, and a ``log::warn!`` would not reach that.
        """
        k = getattr(self, "k", None)
        if isinstance(k, int) and not isinstance(k, bool) and k > n_features:
            import warnings

            warnings.warn(
                f"k={k} is greater than n_features={n_features}. "
                "All the features will be returned.",
                UserWarning,
                stacklevel=3,
            )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags


class SelectKBest(_BaseFilter):
    """Select the ``k`` highest-scoring features. ``k="all"`` keeps everything."""

    def __init__(self, score_func=f_classif, *, k=10, output_type="input"):
        self.score_func = score_func
        self.k = k
        self.output_type = output_type

    def _mode(self):
        return "k_best"

    def _param(self):
        # `None` is how the compiled side spells sklearn's `"all"`.
        return None if isinstance(self.k, str) else float(self.k)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        return tags


class SelectPercentile(_BaseFilter):
    """Select the top ``percentile`` percent of features by score."""

    def __init__(self, score_func=f_classif, *, percentile=10, output_type="input"):
        self.score_func = score_func
        self.percentile = percentile
        self.output_type = output_type

    def _mode(self):
        return "percentile"

    def _param(self):
        return float(self.percentile)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        return tags


class SelectFpr(_BaseFilter):
    """Select features with a p-value below ``alpha`` (false positive rate)."""

    def __init__(self, score_func=f_classif, *, alpha=5e-2, output_type="input"):
        self.score_func = score_func
        self.alpha = alpha
        self.output_type = output_type

    def _mode(self):
        return "fpr"

    def _param(self):
        return float(self.alpha)


class SelectFdr(_BaseFilter):
    """Benjamini-Hochberg false-discovery-rate selection at level ``alpha``."""

    def __init__(self, score_func=f_classif, *, alpha=5e-2, output_type="input"):
        self.score_func = score_func
        self.alpha = alpha
        self.output_type = output_type

    def _mode(self):
        return "fdr"

    def _param(self):
        return float(self.alpha)


class SelectFwe(_BaseFilter):
    """Bonferroni family-wise-error selection at level ``alpha``."""

    def __init__(self, score_func=f_classif, *, alpha=5e-2, output_type="input"):
        self.score_func = score_func
        self.alpha = alpha
        self.output_type = output_type

    def _mode(self):
        return "fwe"

    def _param(self):
        return float(self.alpha)


class GenericUnivariateSelect(_BaseFilter):
    """Univariate selection with a configurable ``mode`` and its ``param``."""

    _MODES = ("percentile", "k_best", "fpr", "fdr", "fwe")

    def __init__(
        self,
        score_func=f_classif,
        *,
        mode="percentile",
        param=1e-5,
        output_type="input",
    ):
        self.score_func = score_func
        self.mode = mode
        self.param = param
        self.output_type = output_type

    def _mode(self):
        if self.mode not in self._MODES:
            raise ValueError(
                f"mode={self.mode!r} is not one of {self._MODES}"
            )
        return self.mode

    def _param(self):
        return None if isinstance(self.param, str) else float(self.param)


# ------------------------------------------------------------------------- #
# The four meta-selectors: sklearn's fit, mlrs's egress
# ------------------------------------------------------------------------- #


class _MlrsMetaSelector(MlrsSelectorMixin):
    """Mix mlrs's container egress over a sklearn meta-selector's fit.

    ``_get_support_mask`` comes from the sklearn base (it is the whole reason these
    subclass rather than reimplement); only ``transform`` /
    ``inverse_transform`` / ``get_feature_names_out`` are replaced, and only to
    route them through :mod:`mlrs._frame`. See this module's docstring for why the
    fit itself is sklearn's.
    """

    def _get_support_mask(self):
        # Resolve against the sklearn base explicitly rather than relying on MRO
        # order, so the mixin cannot accidentally shadow the real implementation.
        for base in type(self).__mro__:
            if base.__module__.startswith("sklearn.feature_selection"):
                return base._get_support_mask(self)
        raise TypeError(  # pragma: no cover - structural
            f"{type(self).__name__} has no sklearn selector base"
        )


class SelectFromModel(_MlrsMetaSelector, _SkSelectFromModel):
    """Select features by an estimator's ``coef_`` / ``feature_importances_``.

    Full sklearn parameter surface (``threshold``, ``prefit``, ``norm_order``,
    ``max_features``, ``importance_getter``), plus mlrs's ``output_type``.
    """

    def __init__(
        self,
        estimator,
        *,
        threshold=None,
        prefit=False,
        norm_order=1,
        max_features=None,
        importance_getter="auto",
        output_type="input",
    ):
        super().__init__(
            estimator,
            threshold=threshold,
            prefit=prefit,
            norm_order=norm_order,
            max_features=max_features,
            importance_getter=importance_getter,
        )
        self.output_type = output_type


class RFE(_MlrsMetaSelector, _SkRFE):
    """Recursive feature elimination."""

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select=None,
        step=1,
        verbose=0,
        importance_getter="auto",
        output_type="input",
    ):
        super().__init__(
            estimator,
            n_features_to_select=n_features_to_select,
            step=step,
            verbose=verbose,
            importance_getter=importance_getter,
        )
        self.output_type = output_type


class RFECV(_MlrsMetaSelector, _SkRFECV):
    """Recursive feature elimination with cross-validated subset selection."""

    def __init__(
        self,
        estimator,
        *,
        step=1,
        min_features_to_select=1,
        cv=None,
        scoring=None,
        verbose=0,
        n_jobs=None,
        importance_getter="auto",
        output_type="input",
    ):
        super().__init__(
            estimator,
            step=step,
            min_features_to_select=min_features_to_select,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
            importance_getter=importance_getter,
        )
        self.output_type = output_type


class SequentialFeatureSelector(_MlrsMetaSelector, _SkSequentialFeatureSelector):
    """Greedy forward / backward selection scored by cross-validation."""

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select="auto",
        tol=None,
        direction="forward",
        scoring=None,
        cv=5,
        n_jobs=None,
        output_type="input",
    ):
        super().__init__(
            estimator,
            n_features_to_select=n_features_to_select,
            tol=tol,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
        self.output_type = output_type


#: Alias so ``from mlrs.feature_selection import SelectorMixin`` works for a
#: caller writing their own selector, as it does from sklearn.
SelectorMixin = MlrsSelectorMixin

__all__ = [
    "GenericUnivariateSelect",
    "RFE",
    "RFECV",
    "SelectFdr",
    "SelectFpr",
    "SelectFromModel",
    "SelectFwe",
    "SelectKBest",
    "SelectPercentile",
    "SelectorMixin",
    "SequentialFeatureSelector",
    "VarianceThreshold",
    "chi2",
    "f_classif",
    "f_oneway",
    "f_regression",
    "mutual_info_classif",
    "mutual_info_regression",
    "r_regression",
]
