"""Ensemble estimator shims (PY-ENS-01/02/03/04, RF-IMP-02, RF-OOB-02)
delegating to ``_mlrs``.

``RandomForestClassifier``/``HistGradientBoostingClassifier`` -> ``ClassifierMixin``;
``RandomForestRegressor``/``HistGradientBoostingRegressor`` -> ``RegressorMixin``.
Each subclasses :class:`MlrsBase` + the family sklearn mixin with a
sklearn-faithful ``__init__`` storing every ctor arg verbatim under the SAME
name (purity rule — matches ``naive_bayes.py``/``linear.py``'s established
pattern). ``fit`` normalizes via the base, constructs the matching
``_mlrs.Py{RandomForest,HistGradientBoosting}*`` wrapper, stores the handle on
``self._mlrs_obj`` and returns ``self`` (PY-01). ``classes_`` (classifiers
only) is materialized from the wrapper ``classes_()`` getter, mirroring
``LogisticRegression``/``MBSGDClassifier``.

``feature_importances_`` (RF-IMP-02, RandomForest only) mirrors ``coef_``'s
dtype-suffixed-accessor shape (``linear.py:41-45``) — always present once
fitted, no constructor gate. ``oob_score_`` (RF-OOB-02, RandomForest only)
reads the same-shaped ``Optional[float]`` accessor; when the estimator was
constructed with ``oob_score=False`` (the default), the underlying accessor
returns ``None`` and this property translates that into an ``AttributeError``
in the PYTHON shim layer (sklearn parity: ``hasattr(model, "oob_score_")`` is
``False`` unless ``oob_score=True`` was passed at construction) — NOT a silent
``None`` return. Neither ``feature_importances_`` nor ``oob_score_`` exists on
``HistGradientBoostingClassifier``/``Regressor`` — sklearn's own HGB
estimators do not expose them either (SPEC §2 non-goal, boosting is not a
bagging/OOB scheme); this is not an oversight.

The RandomForest defaults mirror ``PyRandomForestClassifier``/
``PyRandomForestRegressor``'s ``#[new]`` signatures in
``crates/mlrs-py/src/estimators/ensemble.rs`` (D-02/D-08 sklearn-default
single-source rule): ``n_estimators=100, max_depth=10, n_bins=32,
min_samples_split=2.0, min_samples_leaf=1.0, bootstrap=True, oob_score=False,
seed=42``; ``max_features`` defaults to ``"sqrt"`` for the classifier and
``1.0`` (sklearn's "all features" encoding) for the regressor. The Rust
``parse_max_features`` helper recognizes the strings ``"sqrt"``/``"log2"``/
``"all"``, an int, a float in ``(0.0, 1.0]``, or ``None``. Matching sklearn,
``max_features=None`` resolves to "all features" (the classifier default of
``"sqrt"`` applies only when the argument is OMITTED); ``"all"`` and ``1.0``
are equivalent explicit spellings for all-features.

The HistGradientBoosting defaults mirror ``PyHistGradientBoostingClassifier``/
``PyHistGradientBoostingRegressor``'s ``#[new]`` signatures in the same Rust
file (D-02/D-08): ``max_iter=100, learning_rate=0.1, max_depth=6, n_bins=64,
l2_regularization=0.0, min_samples_leaf=20``.
"""

import sys as _sys
import timeit as _timeit
import warnings as _warnings
from contextlib import contextmanager as _contextmanager, suppress as _suppress
from copy import deepcopy as _deepcopy
from numbers import Integral as _Integral

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch, get_tags
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, column_or_1d

from . import _io
from .base import MlrsBase

# STACK-01: `StackingRegressor` composes the Rust-backed CV surface rather than
# reimplementing folds. `_parallel` / `_delayed` are that module's joblib
# helpers; importing them keeps one definition of "how mlrs spawns workers".
from .model_selection import (  # noqa: E402
    InvalidParameterError,
    _delayed,
    _parallel,
    check_cv,
    cross_val_predict,
)


def _max_features_for_ext(value):
    """Translate the shim-level ``max_features`` to the value forwarded to the
    ``_mlrs`` FFI constructor. sklearn's ``max_features=None`` means "use all
    features"; the FFI's ``Option`` cannot distinguish an omitted argument from
    an explicit ``None`` (both collapse to the estimator's omitted default), so
    the shim — which CAN tell them apart, since its own ``__init__`` default is
    a non-``None`` value (``"sqrt"``/``1.0``) — forwards an explicit ``None`` as
    the ``"all"`` sentinel string, giving full sklearn ``None``-means-all
    parity at the user-facing layer. Every other value passes through untouched
    (``get_params()`` still reports the caller's original ``None``, so
    ``clone()`` round-trips faithfully)."""
    return "all" if value is None else value


class RandomForestClassifier(ClassifierMixin, MlrsBase):
    """Random forest classification (PY-ENS-01).

    ``RandomForestClassifier(n_estimators=100, max_depth=10, n_bins=32,
    max_features="sqrt", min_samples_split=2.0, min_samples_leaf=1.0,
    bootstrap=True, oob_score=False, seed=42)``.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        n_bins=32,
        max_features="sqrt",
        min_samples_split=2.0,
        min_samples_leaf=1.0,
        bootstrap=True,
        oob_score=False,
        seed=42,
        output_type="input",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.seed = seed
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=self._x_float(xa))
        obj = self._ext().RandomForestClassifier(
            self.n_estimators,
            self.max_depth,
            self.n_bins,
            _max_features_for_ext(self.max_features),
            self.min_samples_split,
            self.min_samples_leaf,
            self.bootstrap,
            self.oob_score,
            self.seed,
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        # classes_ are the core's DISTINCT sorted training labels, so a
        # non-contiguous target (e.g. {0, 2}) round-trips through predict.
        self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    def predict_proba(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict_proba")(xa, rows, cols)
        n_classes = int(self.classes_.shape[0])
        return self._to_output(out, (rows, n_classes), X, self._np_float())

    @property
    def feature_importances_(self):
        self._check_fitted()
        return self._to_output(
            self._suffixed("feature_importances")(), (-1,), None, self._np_float()
        )

    @property
    def oob_score_(self):
        self._check_fitted()
        score = self._suffixed("oob_score")()
        if score is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                "'oob_score_' (oob_score=False)"
            )
        return score

    @staticmethod
    def _x_float(xa):
        return np.float32 if xa.type.bit_width == 32 else np.float64

    def shap_values(self, X_train, X):
        """SHAP-01: path-dependent TreeSHAP values (self-consistency-gated
        — see the Rust ``tree_shap`` module docs; a native mlrs forest has
        no external oracle, unlike :meth:`ForestInference.shap_values`).

        ``X_train`` is the reference dataset cover is derived from (a
        re-route of every row through the fitted tree — typically the
        training set). Returns ``(phi, expected_value)``: ``phi`` is
        ``(n_query, n_features, n_classes)``; ``expected_value`` is
        ``(n_classes,)``. ``phi.sum(axis=1) + expected_value ==
        predict_proba(X)`` for every row (additive efficiency).
        """
        self._check_fitted()
        return _shap_values_helper(self._mlrs_obj, "classifier", X_train, X, True)


class RandomForestRegressor(RegressorMixin, MlrsBase):
    """Random forest regression (PY-ENS-02).

    ``RandomForestRegressor(n_estimators=100, max_depth=10, n_bins=32,
    max_features=1.0, min_samples_split=2.0, min_samples_leaf=1.0,
    bootstrap=True, oob_score=False, seed=42)``. ``max_features`` default is
    ``1.0`` ("all features"), NOT the classifier's ``"sqrt"`` — matches
    sklearn's own ``RandomForestRegressor`` default.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        n_bins=32,
        max_features=1.0,
        min_samples_split=2.0,
        min_samples_leaf=1.0,
        bootstrap=True,
        oob_score=False,
        seed=42,
        output_type="input",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.seed = seed
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=RandomForestClassifier._x_float(xa))
        obj = self._ext().RandomForestRegressor(
            self.n_estimators,
            self.max_depth,
            self.n_bins,
            _max_features_for_ext(self.max_features),
            self.min_samples_split,
            self.min_samples_leaf,
            self.bootstrap,
            self.oob_score,
            self.seed,
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())

    @property
    def feature_importances_(self):
        self._check_fitted()
        return self._to_output(
            self._suffixed("feature_importances")(), (-1,), None, self._np_float()
        )

    @property
    def oob_score_(self):
        self._check_fitted()
        score = self._suffixed("oob_score")()
        if score is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute "
                "'oob_score_' (oob_score=False)"
            )
        return score

    def shap_values(self, X_train, X):
        """SHAP-01: path-dependent TreeSHAP values (self-consistency-gated).
        Returns ``(phi, expected_value)``: ``phi`` is ``(n_query,
        n_features)``; ``expected_value`` is a scalar. ``phi.sum(axis=1) +
        expected_value == predict(X)`` for every row.
        """
        self._check_fitted()
        return _shap_values_helper(self._mlrs_obj, "regressor", X_train, X, True)


class HistGradientBoostingClassifier(ClassifierMixin, MlrsBase):
    """Histogram-based gradient boosting classification (PY-ENS-03).

    ``HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1,
    max_depth=6, n_bins=64, l2_regularization=0.0, min_samples_leaf=20)``.

    No ``feature_importances_``/``oob_score_`` — not applicable to boosting
    (SPEC §2 non-goal, matches sklearn's own ``HistGradientBoostingClassifier``
    public attribute shape).
    """

    def __init__(
        self,
        max_iter=100,
        learning_rate=0.1,
        max_depth=6,
        n_bins=64,
        l2_regularization=0.0,
        min_samples_leaf=20,
        device="auto",
        output_type="input",
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.l2_regularization = l2_regularization
        self.min_samples_leaf = min_samples_leaf
        self.device = device
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=self._x_float(xa))
        obj = self._ext().HistGradientBoostingClassifier(
            self.max_iter,
            self.learning_rate,
            self.max_depth,
            self.n_bins,
            self.l2_regularization,
            self.min_samples_leaf,
            self._device(),
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        # classes_ are the core's DISTINCT sorted training labels, so a
        # non-contiguous target (e.g. {0, 2}) round-trips through predict.
        self.classes_ = np.asarray(obj.classes_(), dtype=np.int32)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._mlrs_obj.predict_labels(xa, rows, cols)
        return self._to_output(out, (rows,), X, np.int32)

    def predict_proba(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict_proba")(xa, rows, cols)
        n_classes = int(self.classes_.shape[0])
        return self._to_output(out, (rows, n_classes), X, self._np_float())

    @staticmethod
    def _x_float(xa):
        return np.float32 if xa.type.bit_width == 32 else np.float64


class HistGradientBoostingRegressor(RegressorMixin, MlrsBase):
    """Histogram-based gradient boosting regression (PY-ENS-04).

    ``HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1,
    max_depth=6, n_bins=64, l2_regularization=0.0, min_samples_leaf=20)``.

    No ``feature_importances_``/``oob_score_`` — not applicable to boosting
    (SPEC §2 non-goal, matches sklearn's own ``HistGradientBoostingRegressor``
    public attribute shape).
    """

    def __init__(
        self,
        max_iter=100,
        learning_rate=0.1,
        max_depth=6,
        n_bins=64,
        l2_regularization=0.0,
        min_samples_leaf=20,
        device="auto",
        output_type="input",
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.l2_regularization = l2_regularization
        self.min_samples_leaf = min_samples_leaf
        self.device = device
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=HistGradientBoostingClassifier._x_float(xa))
        obj = self._ext().HistGradientBoostingRegressor(
            self.max_iter,
            self.learning_rate,
            self.max_depth,
            self.n_bins,
            self.l2_regularization,
            self.min_samples_leaf,
            self._device(),
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())


class ForestInference:
    """Batched device inference over an IMPORTED forest (FIL-01 — the cuML
    ``ForestInference`` parity surface, Phase 20).

    Load an externally-trained sklearn forest and serve GPU predictions::

        fil = mlrs.ForestInference.load_from_sklearn(sk_model)
        y = fil.predict(X)          # regressor import
        p = fil.predict_proba(X)    # classifier import

    Supported sources: fitted ``sklearn.ensemble.RandomForestClassifier`` /
    ``RandomForestRegressor`` (and any estimator exposing the same
    ``estimators_[i].tree_`` arrays). Leaf routing is EXACTLY sklearn's
    (``x <= threshold`` — thresholds are ``next_up``-bumped for the mlrs
    ``<`` comparator on import); trees deeper than 16 raise (retrain the
    source with ``max_depth <= 16``).

    Not an sklearn estimator shim (no ``fit``/``get_params``) — the model
    arrives fitted.
    """

    def __init__(self, obj, kind, classes, n_features, dtype):
        self._mlrs_obj = obj
        self._kind = kind
        self.classes_ = classes
        self._n_features = n_features
        self._dtype = dtype

    @classmethod
    def load_from_sklearn(cls, model, dtype=np.float32):
        """Import a fitted sklearn forest. ``dtype`` picks the device arm."""
        from . import _load_ext

        est_list = getattr(model, "estimators_", None)
        if est_list is None:
            raise ValueError(
                "ForestInference.load_from_sklearn: model has no estimators_ "
                "(expected a fitted sklearn RandomForest*)"
            )
        is_classifier = hasattr(model, "classes_")
        classes = np.asarray(model.classes_) if is_classifier else None
        n_values = int(classes.shape[0]) if is_classifier else 1
        kind = "classifier" if is_classifier else "regressor"

        # mlrs ForestInference is single-output only in v1: a multi-output
        # regressor (n_outputs_ > 1) would have per-node value rows of shape
        # (1, n_outputs) that this importer cannot represent — silently
        # keeping only output 0 would produce wrong predictions for the rest,
        # so reject it loudly instead. (Multi-class classifiers are fine —
        # their per-node rows ARE the n_values distribution.)
        n_outputs = int(getattr(model, "n_outputs_", 1))
        if n_outputs > 1:
            raise ValueError(
                "ForestInference.load_from_sklearn: multi-output forests "
                f"(n_outputs_={n_outputs}) are not supported; the per-node "
                "value rows carry only output 0's leaf values in this import "
                "path — fit one ForestInference per output column"
            )

        children_left, children_right, feature, threshold, value, nsw, counts = (
            [], [], [], [], [], [], []
        )
        for est in est_list:
            t = est.tree_
            counts.append(int(t.node_count))
            children_left.append(np.asarray(t.children_left, dtype=np.int64))
            children_right.append(np.asarray(t.children_right, dtype=np.int64))
            feature.append(np.asarray(t.feature, dtype=np.int64))
            threshold.append(np.asarray(t.threshold, dtype=np.float64))
            nsw.append(np.asarray(t.weighted_n_node_samples, dtype=np.float64))
            v = np.asarray(t.value, dtype=np.float64)  # (n_nodes, 1, n_values)
            if is_classifier:
                # sklearn >=1.3 stores value rows already normalized OR raw
                # counts depending on version/weighting; the Rust import
                # normalizes each leaf row, so either form round-trips.
                value.append(v[:, 0, :].reshape(-1))
            else:
                value.append(v[:, 0, 0].reshape(-1))

        dt = "f32" if np.dtype(dtype) == np.float32 else "f64"
        obj = _load_ext().ForestInference.load_from_arrays(
            np.concatenate(children_left).tolist(),
            np.concatenate(children_right).tolist(),
            np.concatenate(feature).tolist(),
            np.concatenate(threshold).tolist(),
            np.concatenate(value).tolist(),
            np.concatenate(nsw).tolist(),
            counts,
            n_values,
            kind,
            int(model.n_features_in_),
            dt,
        )
        return cls(obj, kind, classes, int(model.n_features_in_), dt)

    @property
    def n_trees(self):
        return self._mlrs_obj.n_trees()

    def _normalize_query(self, X):
        dtype = np.float32 if self._dtype == "f32" else np.float64
        arr = np.ascontiguousarray(np.asarray(X, dtype=dtype))
        if arr.ndim != 2:
            raise ValueError("ForestInference: X must be 2-D")
        xa, rows, cols = _io.normalize_X(arr)
        return xa, rows, cols

    def predict(self, X):
        """Regressor: forest-mean predictions. Classifier: ``classes_``-mapped
        argmax labels (sklearn ``predict`` parity)."""
        xa, rows, cols = self._normalize_query(X)
        if self._kind == "classifier":
            idx = np.asarray(
                self._mlrs_obj.predict_class_indices(xa, rows, cols), dtype=np.int64
            )
            return self.classes_[idx]
        out = getattr(self._mlrs_obj, f"predict_{self._dtype}")(xa, rows, cols)
        return np.asarray(out)

    def predict_proba(self, X):
        """Classifier: ``rows × n_classes`` mean-of-tree-distributions."""
        if self._kind != "classifier":
            raise ValueError("ForestInference: predict_proba on a regressor import")
        xa, rows, cols = self._normalize_query(X)
        out = getattr(self._mlrs_obj, f"predict_proba_{self._dtype}")(xa, rows, cols)
        return np.asarray(out).reshape(rows, -1)

    def shap_values(self, X):
        """SHAP-01: path-dependent TreeSHAP values using the import's OWN
        cover (``tree_.weighted_n_node_samples`` from the source sklearn
        model) — the ≤1e-5-vs-``shap.TreeExplainer``-gated path (see the
        Rust ``tree_shap`` module docs). Raises if the import carried no
        cover (built from raw arrays without ``node_sample_weight``).

        Returns ``(phi, expected_value)`` — classifier: ``phi`` is
        ``(n_query, n_features, n_classes)``, ``expected_value`` is
        ``(n_classes,)``; regressor: ``phi`` is ``(n_query, n_features)``,
        ``expected_value`` is a scalar.
        """
        xq, qr, qc = self._normalize_query(X)
        phi, ev = self._mlrs_obj.shap_values(xq, qr, qc)
        n_values = len(ev)
        phi = np.asarray(phi).reshape(qr, qc, n_values)
        ev = np.asarray(ev)
        if self._kind == "regressor":
            return phi[:, :, 0], ev[0]
        return phi, ev


def _shap_values_helper(mlrs_obj, kind, x_train, x_query, has_train_arg):
    """Shared SHAP-values plumbing for RandomForest*/ForestInference (SHAP-01)."""
    # The Rust `shap_values` dispatches on the FITTED model's dtype arm and
    # then reads the Arrow query capsule with `as_f32`/`as_f64` — a capsule
    # of the wrong dtype (e.g. a float64 query against a float32-fit model,
    # the common cross-dtype case) fails that downcast with an opaque
    # "unsupported dtype" error. Coerce both inputs to the fitted dtype here,
    # BEFORE normalize_X, so the capsule dtype always matches the arm.
    fit_dtype = np.float32 if mlrs_obj.dtype() == "f32" else np.float64
    if has_train_arg:
        xt, tr, tc = _io.normalize_X(x_train, dtype=fit_dtype)
        xq, qr, qc = _io.normalize_X(x_query, dtype=fit_dtype)
        phi, ev = mlrs_obj.shap_values(xt, tr, tc, xq, qr, qc)
    else:
        xq, qr, qc = _io.normalize_X(x_query, dtype=fit_dtype)
        phi, ev = mlrs_obj.shap_values(xq, qr, qc)
    n_values = len(ev)
    phi = np.asarray(phi).reshape(qr, qc, n_values)
    ev = np.asarray(ev)
    if kind == "regressor":
        return phi[:, :, 0], ev[0]
    return phi, ev


# =========================================================================== #
# StackingRegressor (STACK-01) — the stacked-generalization meta-estimator.
# =========================================================================== #
#
# Unlike every other class in this module, `StackingRegressor` is NOT an
# `MlrsBase` shim over a `_mlrs` `#[pyclass]`: it owns no device buffers and
# fits nothing itself. It is a *composition* — base regressors produce
# out-of-fold predictions, those become the columns of a meta-feature matrix,
# and a final regressor is fitted on it. All the arithmetic already happens
# inside the composed estimators (on the device, when those are mlrs
# estimators), and all the cross-validation index generation already happens in
# Rust via `mlrs.model_selection`.
#
# What the meta-estimator itself owns is structure, and that IS in Rust
# (`mlrs_algos::ensemble::stacking`, reached through the `_mlrs`
# `stacking_*` free functions):
#
#   =============================  =========================================
#   what                           where the work happens
#   =============================  =========================================
#   estimator-name validation      Rust (`stacking_validate_names`)
#   `'drop'` bookkeeping           Rust (`stacking_kept_indices`)
#   `cv="prefit"` classification   Rust (`stacking_cv_is_prefit`)
#   meta-column layout             Rust (`stacking_meta_layout`)
#   `get_feature_names_out`        Rust (`stacking_feature_names`)
#   fold index generation          Rust (`mlrs.model_selection`)
#   the meta-matrix copy           numpy by default (see below)
#   base/final `fit` + `predict`   the composed estimators
#   =============================  =========================================
#
# ## The meta-matrix copy has three arms (STACK-META-01)
#
# `MLRS_STACK_META_ENGINE` chooses between `np.hstack` (the default), the Rust
# host copy (`mlrs_algos::ensemble::stacking::concatenate_predictions`), and the
# CubeCL scatter (`mlrs_backend::prims::stacking_meta`). All three produce
# BYTE-IDENTICAL matrices — the operation carries no arithmetic — so the knob
# moves work and nothing else.
#
# numpy stays the default, and that is a measurement rather than an assumption:
# this is one `n x width` copy of data that is already in host memory, so the
# host arm pays an Arrow capsule crossing each way and the device arm an upload
# plus a download on top of that. `docs/stacking.md` carries the ladder and
# `scripts/bench_stacking_meta.py` re-runs it. `_meta_via_rust` is also the
# fallback boundary: it DECLINES (returns None) for anything the Rust arms
# cannot represent, leaving `np.hstack` to handle it exactly as before.
#
# ## The default `final_estimator` is sklearn's `RidgeCV`
#
# sklearn's default is `RidgeCV()`, which selects `alpha` from `(0.1, 1.0, 10.0)`
# by efficient leave-one-out generalized cross-validation. mlrs ships `Ridge`,
# not `RidgeCV`, and substituting `Ridge(alpha=1.0)` would silently change every
# default-constructed stack's predictions relative to the sklearn baseline users
# are migrating from — exactly the divergence the 1e-5 parity contract exists to
# prevent. So the default is `sklearn.linear_model.RidgeCV()`, constructed
# lazily inside `fit`. sklearn is already a hard runtime dependency of this
# package (every shim subclasses its mixins). Pass `final_estimator=mlrs.Ridge()`
# — or any other regressor — to put the meta-fit on the device.
#
# ## `n_features_in_` is the BASE estimators' feature count
#
# sklearn reads it off `estimators_[0]`, i.e. the width of the ORIGINAL `X`, not
# of the meta matrix. `get_feature_names_out(input_features=...)` is validated
# against that same width.


def _stack_ext():
    """The compiled ``_mlrs`` extension (lazy, mirroring ``MlrsBase._ext``)."""
    from . import _load_ext

    return _load_ext()


def _is_sparse(X):
    """``X`` is a scipy sparse matrix/array — WITHOUT importing scipy.

    Detected through ``sys.modules`` exactly as ``mlrs.model_selection`` does,
    so the check cannot pull in a library the user does not have installed.
    """
    sparse_mod = _sys.modules.get("scipy.sparse")
    return sparse_mod is not None and sparse_mod.issparse(X)


def _generate_input_feature_names(estimator, input_features):
    """sklearn ``_check_feature_names_in(..., generate_names=True)``.

    Reimplemented rather than imported: it is a dozen lines of pure bookkeeping
    and lives behind a private sklearn path. Same three branches, same two error
    messages.
    """
    feature_names_in_ = getattr(estimator, "feature_names_in_", None)
    n_features_in_ = getattr(estimator, "n_features_in_", None)

    if input_features is not None:
        input_features = np.asarray(input_features, dtype=object)
        if feature_names_in_ is not None and not np.array_equal(
            feature_names_in_, input_features
        ):
            raise ValueError("input_features is not equal to feature_names_in_")
        if n_features_in_ is not None and len(input_features) != n_features_in_:
            raise ValueError(
                "input_features should have length equal to number of "
                f"features ({n_features_in_}), got {len(input_features)}"
            )
        return input_features

    if feature_names_in_ is not None:
        return feature_names_in_
    if n_features_in_ is None:
        raise ValueError("Unable to generate feature names without n_features_in_")
    return np.asarray([f"x{i}" for i in range(n_features_in_)], dtype=object)


def _raise_for_extra_fit_params(params, owner, method, allow=()):
    """sklearn ``_raise_for_params``: extra kwargs need routing enabled.

    Message reproduced verbatim — a caller that greps for it (or a doctest that
    prints it) must not see a different string here than under sklearn.
    """
    if not _routing_enabled() and (params.keys() - set(allow)):
        raise ValueError(
            f"Passing extra keyword arguments to {type(owner).__name__}.{method} is"
            " only supported if enable_metadata_routing=True, which you can set"
            " using `sklearn.set_config`. See the User Guide"
            " <https://scikit-learn.org/stable/metadata_routing.html> for more"
            f" details. Extra parameters passed are: {set(params)}"
        )


@_contextmanager
def _elapsed_time(source, message=None):
    """sklearn ``_print_elapsed_time``: print ``message`` and the elapsed time.

    Reimplemented rather than imported for the reason
    :func:`_generate_input_feature_names` gives — it is a dozen lines behind a
    private sklearn path. The LINE it prints is user-visible under
    ``verbose=True``, so the padding rule (dots to column 70) and the two time
    formats are reproduced character for character.
    """
    if message is None:
        yield
        return
    start = _timeit.default_timer()
    yield
    elapsed = _timeit.default_timer() - start
    head = "[%s] " % source
    if elapsed > 60:
        time_str = "%4.1fmin" % (elapsed / 60)
    else:
        time_str = " %5.1fs" % elapsed
    tail = " %s, total=%s" % (message, time_str)
    print("%s%s%s" % (head, max(70 - len(head) - len(tail), 0) * ".", tail))


def _fit_one(estimator, X, y, fit_params, source=None, message=None):
    """sklearn ``_fit_single_estimator``: fit one composed estimator.

    With routing off, a ``sample_weight`` in ``fit_params`` is passed
    positionally-by-name and a ``TypeError`` from an estimator that does not
    accept it is re-raised as sklearn's clearer message.

    ``source``/``message`` are :class:`VotingRegressor`'s ``verbose=True``
    progress line, which sklearn emits from inside this same function; they are
    ``None`` for every stacking caller, whose ``verbose`` is an int forwarded to
    ``cross_val_predict`` instead.
    """
    if not _routing_enabled() and "sample_weight" in fit_params:
        try:
            with _elapsed_time(source, message):
                estimator.fit(X, y, sample_weight=fit_params["sample_weight"])
        except TypeError as exc:
            if "unexpected keyword argument 'sample_weight'" in str(exc):
                raise TypeError(
                    "Underlying estimator {} does not support sample weights.".format(
                        type(estimator).__name__
                    )
                ) from exc
            raise
    else:
        with _elapsed_time(source, message):
            estimator.fit(X, y, **fit_params)
    return estimator


def _effective_n_jobs(n_jobs, members, owner="StackingRegressor"):
    """``n_jobs``, reduced to serial when a member holds a device handle.

    Neither joblib fan-out is worth taking over a fitted mlrs estimator:

    * **Process backends** (``loky``, the joblib default; ``multiprocessing``;
      ``dask``) return each worker's result by PICKLING it. A fitted mlrs
      estimator owns ``self._mlrs_obj``, a compiled ``#[pyclass]`` wrapping
      device state, and that is not picklable — ``n_jobs=2`` raises
      ``TypeError: cannot pickle 'builtins.Ridge' object``. This is
      unconditional.
    * **The threading backend** works, and barely helps. Every device call runs
      while holding the process-global ``Mutex<BufferPool>``
      (``crates/mlrs-py/src/lib.rs``), so the fan-out cannot overlap the work it
      is fanning out. Measured on rocm gfx1151, six members at ``cv=20``:
      1.584 s serial -> 1.343 s at ``n_jobs=4`` (1.18x), bit-identical. Choosing
      a backend on the caller's behalf for ~18% is a bad trade; finer-grained
      locking, not a scheduler switch, is what would make this parameter pay.

    HISTORY: until the CubeCL stream cap landed (``mlrs_backend::stream_cap``,
    STREAM-CAP-01) the threading route did not merely underperform — it aborted
    the process. CubeCL allocates one stream per OS thread and one memory arena
    per stream, so a thread fan-out exhausted the device heap. That is fixed; the
    reason this function still reduces ``n_jobs`` is the mutex above, not the
    crash.

    So the composition runs serially and says so, once. ``n_jobs`` still works
    normally over host (sklearn) members — measured at n=100000, d=64, cv=5:
    1.61 s -> 0.96 s at ``n_jobs=4``.
    """
    if n_jobs in (None, 1):
        return n_jobs
    if not any(isinstance(est, MlrsBase) for est in members):
        return n_jobs
    _warnings.warn(
        f"{owner}: n_jobs is ignored because at least one composed "
        "estimator is an mlrs estimator holding a device handle. Process-based "
        "joblib backends cannot pickle that handle, and mlrs serializes device "
        "work behind one pool lock, so a threaded fan-out measures ~1.2x at "
        "best. Fitting serially. Pass host (e.g. scikit-learn) sub-estimators "
        "to use n_jobs.",
        UserWarning,
        stacklevel=3,
    )
    return 1


def _meta_via_rust(blocks, pred_cols, n_features, passthrough, engine):
    """The meta matrix from ``_mlrs.stacking_concatenate``, or ``None`` for numpy.

    ``blocks`` is the prediction blocks in kept order, with ``X`` appended when
    ``passthrough``; ``pred_cols`` is the prediction blocks' column counts (the
    layout's ``n_feature_outs``, so ``X`` is excluded).

    Returns ``None`` — deliberately, rather than raising — for every input the
    Rust arms cannot represent, leaving ``np.hstack`` to handle it exactly as it
    did before this arm existed:

    * a non-float block (an integer or object array, or a duck-typed ``X`` like
      ``estimator_checks``' ``_NotAnArray``, which ``np.hstack`` passes straight
      through);
    * a block that is not 2-D, or whose row count disagrees with the others —
      numpy's own error message for that is the one users already know.

    The dtype handed over is ``np.result_type`` of the blocks, which is the
    promotion ``np.hstack`` would have applied, so the meta matrix the final
    estimator is fitted on is bit-identical across all three arms rather than
    merely close.
    """
    import pyarrow as pa

    arrays = [np.asarray(b) for b in blocks]
    if any(a.ndim != 2 for a in arrays):
        return None
    try:
        dtype = np.result_type(*[a.dtype for a in arrays])
    except TypeError:
        return None
    if dtype != np.float32 and dtype != np.float64:
        return None
    n_rows = int(arrays[0].shape[0])
    if any(int(a.shape[0]) != n_rows for a in arrays):
        return None
    if n_rows == 0:
        # A zero-row meta matrix is a zero-byte device allocation on the device
        # arm; numpy already produces the right empty shape, and there is
        # nothing to gain by handing an empty buffer to CubeCL.
        return None

    arrow_type = pa.float32() if dtype == np.float32 else pa.float64()
    flats, capsules = [], []
    for a in arrays:
        # `ascontiguousarray` is a no-op view when `a` is already C-contiguous in
        # the promoted dtype, and `py_buffer` does not copy — so a block reaches
        # Rust without a staging copy, and the arms are compared on the copy
        # they actually perform rather than on ingress overhead.
        flat = np.ascontiguousarray(a, dtype=dtype).ravel(order="C")
        flats.append(flat)
        capsules.append(
            pa.Array.from_buffers(arrow_type, flat.size, [None, pa.py_buffer(flat)])
        )

    x = capsules.pop() if passthrough else None
    flat_out, width = _stack_ext().stacking_concatenate(
        capsules,
        [int(c) for c in pred_cols],
        n_rows,
        int(n_features),
        bool(passthrough),
        x,
        engine,
    )
    # `flats` is referenced until here on purpose: `py_buffer` borrows those
    # numpy buffers, and letting one be collected mid-call would free memory
    # Rust is still reading.
    del flats
    return _io.to_output(flat_out, (n_rows, width), "numpy", dtype)


def _final_estimator_has(attr):
    """``available_if`` predicate over ``final_estimator_`` then ``final_estimator``.

    Mirrors sklearn's ``_estimator_has(attr, delegates=("final_estimator_",
    "final_estimator"))``, including the observable consequence that
    ``hasattr(StackingRegressor(...), "predict")`` is ``False`` on an UNFITTED
    estimator left at ``final_estimator=None`` — ``getattr(None, "predict")``
    raises, and ``available_if`` reads a raise as "not available".
    """

    def check(self):
        for delegate in ("final_estimator_", "final_estimator"):
            try:
                delegated = getattr(self, delegate)
            except AttributeError:
                continue
            getattr(delegated, attr)
            return True
        raise AttributeError

    return check


class _HeterogeneousComposition:
    """The ``estimators``-list mechanics EVERY mlrs meta-ensemble shares.

    This is sklearn's `_BaseComposition` / `_BaseHeterogeneousEnsemble` half —
    the rules that care only that there is a list of ``(name, estimator)``
    pairs, and not at all what the members predict or what is done with their
    predictions:

    * the parameter surface over ``estimators`` (``get_params`` /
      ``set_params`` / ``named_estimators``), which is what makes
      ``set_params(lr="drop")`` and a ``GridSearchCV`` over ``lr__C`` work;
    * the structural validation — the shape check, the name rules, the
      ``'drop'`` bookkeeping;
    * the ``named_estimators_`` bookkeeping after a fit;
    * the ``allow_nan`` / ``sparse`` tags derived from the members.

    :class:`_StackComposition` adds the stacking-specific half on top;
    :class:`VotingRegressor` mixes this in directly, because its aggregation
    (`np.average` over one column per member) has nothing in common with a meta
    matrix beyond these three bullets.

    Not exported and not an estimator base class in its own right: it carries no
    ``__init__``, so ``get_params`` still reads each concrete class's own
    signature.
    """

    # -- composition parameter handling (sklearn `_BaseComposition`) ------- #

    def get_params(self, deep=True):
        """Constructor params, plus every named sub-estimator and its params.

        ``deep=True`` adds one ``<name>`` key per entry in ``estimators`` and a
        ``<name>__<param>`` key per sub-estimator parameter, so ``clone`` and a
        ``GridSearchCV`` over ``lr__alpha`` both work.
        """
        out = super().get_params(deep=deep)
        if not deep:
            return out
        try:
            out.update(self.estimators)
        except (TypeError, ValueError):
            # A malformed `estimators` must not break `get_params` — `set_params`
            # calls it, and the real complaint belongs to `fit`'s validation.
            return out
        for name, estimator in self.estimators:
            if hasattr(estimator, "get_params"):
                for key, value in estimator.get_params(deep=True).items():
                    out[f"{name}__{key}"] = value
        return out

    def set_params(self, **params):
        """Set params, including replacing a sub-estimator by name.

        ``set_params(lr="drop")`` disables an entry; ``set_params(lr__alpha=2)``
        reaches into one. Order matters and is sklearn's: the whole
        ``estimators`` list first, then whole estimators by name, then
        individual nested parameters.
        """
        if "estimators" in params:
            self.estimators = params.pop("estimators")
        items = self.estimators
        if isinstance(items, list) and items:
            with _suppress(TypeError):
                item_names, _ = zip(*items)
                for name in list(params.keys()):
                    if "__" not in name and name in item_names:
                        self._replace_estimator(name, params.pop(name))
        super().set_params(**params)
        return self

    def _replace_estimator(self, name, new_val):
        replaced = list(self.estimators)
        for i, (estimator_name, _) in enumerate(replaced):
            if estimator_name == name:
                replaced[i] = (name, new_val)
                break
        self.estimators = replaced

    @property
    def named_estimators(self):
        """``name -> estimator`` for the UNFITTED ``estimators`` argument."""
        return Bunch(**dict(self.estimators))

    # -- validation -------------------------------------------------------- #

    def _validate_composition(self):
        """``(names, estimators, kept_indices)`` — the Rust-backed structural check.

        The shape check, the name rules and the ``'drop'`` bookkeeping, which
        both classes run identically. What each class adds on top (a regressor
        type check for one, nothing for the other — sklearn's
        ``StackingClassifier`` deliberately accepts regressors for ordinal
        problems) stays with the class.
        """
        estimators = self.estimators
        if len(estimators) == 0 or not all(
            isinstance(item, (tuple, list)) and isinstance(item[0], str)
            for item in estimators
        ):
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a "
                "non-empty list of (string, estimator) tuples."
            )
        names, values = zip(*estimators)
        ext = _stack_ext()
        ext.stacking_validate_names(list(names), list(self.get_params(deep=False)))

        drop = ext.stacking_drop_sentinel()
        # The `== 'drop'` comparison itself has to happen here: the value is an
        # arbitrary object whose `__eq__` may be overloaded. Its CONSEQUENCES
        # (which slots survive, and whether any do) are Rust's.
        is_drop = [_is_drop(est, drop) for est in values]
        kept = ext.stacking_kept_indices(is_drop)
        return list(names), list(values), list(kept)

    # -- fitted bookkeeping -------------------------------------------------- #

    def _record_named_estimators(self, names, kept):
        """``named_estimators_``, with a dropped slot kept as the string ``'drop'``.

        Also lifts ``feature_names_in_`` off whichever fitted member exposes it,
        which is where sklearn takes it from too.
        """
        self.named_estimators_ = Bunch()
        kept_set = set(kept)
        fitted_idx = 0
        for i, name in enumerate(names):
            if i in kept_set:
                current = self.estimators_[fitted_idx]
                self.named_estimators_[name] = current
                fitted_idx += 1
                if hasattr(current, "feature_names_in_"):
                    self.feature_names_in_ = current.feature_names_in_
            else:
                self.named_estimators_[name] = _stack_ext().stacking_drop_sentinel()

    # -- derived tags -------------------------------------------------------- #

    def __sklearn_tags__(self):
        """``allow_nan`` / ``sparse`` are the AND over the composed estimators.

        A composition is only as permissive as its least permissive member, so —
        as in sklearn's ``_BaseHeterogeneousEnsemble``, where this lives too —
        these are derived rather than declared. Composing mlrs estimators
        therefore reports ``sparse=False`` (they ingest dense Arrow), while
        composing sklearn estimators can report ``True``.

        This is NOT cosmetic: sklearn's ``check_estimator_sparse_tag`` asserts
        that an estimator declaring ``sparse=False`` actually REJECTS sparse
        input, and neither of these meta-estimators touches ``X`` itself — it
        goes straight to the members — so declaring ``False`` over members that
        accept sparse would be a tag the estimator contradicts.
        """
        tags = super().__sklearn_tags__()
        try:
            drop = _stack_ext().stacking_drop_sentinel()
            tags.input_tags.allow_nan = all(
                True if _is_drop(est, drop) else get_tags(est).input_tags.allow_nan
                for _, est in self.estimators
            )
            tags.input_tags.sparse = all(
                True if _is_drop(est, drop) else get_tags(est).input_tags.sparse
                for _, est in self.estimators
            )
        except Exception:
            # A malformed `estimators` must not break tag computation; `fit`'s
            # own validation reports it properly.
            pass
        return tags


class _StackComposition(_HeterogeneousComposition):
    """The composition mechanics :class:`StackingRegressor` and
    :class:`StackingClassifier` share (STACK-CLF-01).

    Everything here is identical between the two by sklearn's own construction —
    it is the `_BaseStacking` half, sitting on the list mechanics
    :class:`_HeterogeneousComposition` already provides:

    * the ``cv="prefit"`` classification, which asks Rust and re-raises under
      ``InvalidParameterError``;
    * the meta-matrix assembly from already-shaped 2-D blocks — layout in Rust,
      copy on the arm ``MLRS_STACK_META_ENGINE`` names;
    * ``n_features_in_``, ``get_feature_names_out`` and ``get_metadata_routing``
      (the derived tags moved down to :class:`_HeterogeneousComposition`, which
      is where sklearn keeps them too).

    What is NOT here is everything the two classes genuinely disagree about:
    which sub-estimator type is legal, the default ``final_estimator``, the
    response method each member is asked for (``stack_method``, classifier
    only), the label encoding, and the shape of the blocks handed to
    :meth:`_assemble_meta`. Those live on the two classes, where a reader
    comparing mlrs against sklearn's source expects to find them.
    """

    # -- validation --------------------------------------------------------- #

    def _cv_is_prefit(self):
        """Is ``cv`` the ``"prefit"`` string? Classified in Rust.

        A non-string ``cv`` never reaches Rust — it stays here and goes to
        :func:`mlrs.model_selection.check_cv`. A string that is not ``"prefit"``
        raises sklearn's ``StrOptions`` message (naming THIS class), re-raised as
        ``InvalidParameterError`` so that both ``except ValueError`` and
        ``except TypeError`` callers migrating from sklearn still catch it.
        """
        if not isinstance(self.cv, str):
            return False
        try:
            return _stack_ext().stacking_cv_is_prefit(self.cv, type(self).__name__)
        except ValueError as exc:
            raise InvalidParameterError(str(exc)) from None

    # -- meta-feature assembly ---------------------------------------------- #

    def _assemble_meta(self, X, blocks):
        """One meta matrix from the per-estimator 2-D blocks, in kept order.

        The column LAYOUT (widths, offsets, total width, and the
        ``_n_feature_outs`` that ``get_feature_names_out`` reads) is computed in
        Rust; the copy runs on the arm ``MLRS_STACK_META_ENGINE`` names, which
        is ``np.hstack`` by default. Callers hand over blocks that are already
        2-D and already sliced — reshaping a 1-D ``predict`` output is the
        regressor's business and dropping a collinear probability column is the
        classifier's, and both are decided in Rust before they get here.
        """
        # X's width only enters the layout under `passthrough`; skip reading it
        # otherwise, since `X` here may be any array-LIKE the base estimators
        # accepted (sklearn's check harness passes duck-typed objects with no
        # `.shape` at all) and a needless probe would be the only thing that
        # rejected it.
        n_features = _n_columns(X) if self.passthrough else 0
        n_feature_outs, _offsets, _n_meta, _width = _stack_ext().stacking_meta_layout(
            [int(b.shape[1]) for b in blocks], n_features, bool(self.passthrough)
        )
        self._n_feature_outs = list(n_feature_outs)

        if self.passthrough:
            blocks = list(blocks) + [X]
            if _is_sparse(X):
                return _sys.modules["scipy.sparse"].hstack(blocks, format=X.format)

        # STACK-META-01: `np.hstack` is the default arm and the fallback for
        # everything the Rust arms cannot represent (see `_meta_via_rust`).
        engine = _stack_ext().stacking_meta_engine()
        if engine != "numpy":
            meta = _meta_via_rust(
                blocks, n_feature_outs, n_features, self.passthrough, engine
            )
            if meta is not None:
                return meta
        return np.hstack(blocks)

    # -- introspection ------------------------------------------------------ #

    @property
    def n_features_in_(self):
        """Feature count of the ORIGINAL ``X`` (read off ``estimators_[0]``)."""
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute n_features_in_"
            ) from nfe
        return self.estimators_[0].n_features_in_

    def get_feature_names_out(self, input_features=None):
        """Meta-feature names: ``<classname>_<name>`` per kept estimator.

        A multi-column block (a multiclass ``predict_proba``, say) is suffixed
        with the within-block index and no separator —
        ``stackingclassifier_lr0``. Under ``passthrough=True`` the input feature
        names follow. Generated in Rust so the naming rule has one definition.
        """
        check_is_fitted(self, "n_features_in_")
        kept_names = [
            name
            for name, est in self.estimators
            if not _is_drop(est, _stack_ext().stacking_drop_sentinel())
        ]
        if self.passthrough:
            inputs = [
                str(f) for f in _generate_input_feature_names(self, input_features)
            ]
        else:
            # sklearn still VALIDATES a supplied `input_features` here, it just
            # does not emit it (`generate_names=False`).
            if input_features is not None:
                _generate_input_feature_names(self, input_features)
            inputs = None
        names = _stack_ext().stacking_feature_names(
            type(self).__name__.lower(),
            kept_names,
            [int(n) for n in self._n_feature_outs],
            inputs,
        )
        return np.asarray(names, dtype=object)

    def get_metadata_routing(self):
        """Route ``fit`` metadata to each named estimator, ``predict`` to the final one."""
        router = MetadataRouter(owner=type(self).__name__)
        for name, estimator in self.estimators:
            router.add(
                **{name: estimator},
                method_mapping=MethodMapping().add(callee="fit", caller="fit"),
            )
        try:
            final_estimator_ = self.final_estimator_
        except AttributeError:
            final_estimator_ = self.final_estimator
        router.add(
            final_estimator_=final_estimator_,
            method_mapping=MethodMapping().add(caller="predict", callee="predict"),
        )
        return router


class StackingRegressor(
    RegressorMixin, TransformerMixin, MetaEstimatorMixin, _StackComposition, BaseEstimator
):
    """Stack of regressors with a final regressor (STACK-01).

    ``StackingRegressor(estimators, final_estimator=None, *, cv=None,
    n_jobs=None, passthrough=False, verbose=0)`` — the full
    :class:`sklearn.ensemble.StackingRegressor` parameter surface, with the
    cross-validation index generation and the composition bookkeeping in Rust.

    Base estimators are fitted on the whole of ``X`` and exposed as
    ``estimators_``; ``final_estimator_`` is fitted on their **out-of-fold**
    predictions (:func:`mlrs.model_selection.cross_val_predict`), so the meta
    learner never sees a base estimator's in-sample fit.

    Parameters
    ----------
    estimators : list of (str, estimator)
        The base regressors. An entry's estimator may be the string ``'drop'``
        (usually via ``set_params(name='drop')``) to disable it; a dropped entry
        keeps its slot in ``named_estimators_`` (as ``'drop'``) but contributes
        no meta column and is never fitted.
    final_estimator : estimator, default=None
        The regressor fitted on the meta features. ``None`` means
        ``sklearn.linear_model.RidgeCV()``, sklearn's own default — see this
        module's ``StackingRegressor`` section for why mlrs does not substitute
        its own :class:`mlrs.Ridge` here.
    cv : int, cross-validation generator, iterable, or "prefit", default=None
        Passed to :func:`mlrs.model_selection.cross_val_predict`. ``None`` is
        5-fold :class:`mlrs.model_selection.KFold` (stacking a regressor never
        stratifies). ``"prefit"`` assumes every entry in ``estimators`` is
        ALREADY fitted: nothing is cloned or refitted, and the meta features are
        the base estimators' predictions on the FULL training set rather than
        out-of-fold ones — which is much cheaper and, if those estimators were
        fitted on this same data, badly overfit.
    n_jobs : int, default=None
        joblib parallelism for the base-estimator fits and for the inner
        ``cross_val_predict`` calls. ``None`` means 1.
    passthrough : bool, default=False
        Append the original ``X`` columns to the meta features.
    verbose : int, default=0
        Forwarded to the inner ``cross_val_predict`` calls.

    Attributes
    ----------
    estimators_ : list of estimator
        The fitted base estimators, dropped entries excluded. Under
        ``cv="prefit"`` these are the caller's own objects, not clones.
    named_estimators_ : :class:`sklearn.utils.Bunch`
        ``name -> fitted estimator`` (or the string ``'drop'``).
    final_estimator_ : estimator
        The fitted meta regressor.
    stack_method_ : list of str
        ``"predict"`` per kept estimator (a regressor has no other response
        method; the attribute exists for sklearn parity).
    n_features_in_ : int
        Feature count of the ORIGINAL ``X``, read off ``estimators_[0]``.
    feature_names_in_ : ndarray of str
        Present only when a fitted base estimator exposes it.

    Examples
    --------
    >>> import numpy as np
    >>> import mlrs
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 5)).astype(np.float32)
    >>> y = (X[:, 0] * 3.0 - X[:, 1]).astype(np.float32)
    >>> reg = mlrs.StackingRegressor(
    ...     estimators=[("lr", mlrs.LinearRegression()), ("ridge", mlrs.Ridge())],
    ...     final_estimator=mlrs.Ridge(),
    ...     cv=3,
    ... )
    >>> reg.fit(X, y).predict(X[:2]).shape
    (2,)
    """

    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose

    # -- validation -------------------------------------------------------- #

    def _validate_estimators(self):
        """``(names, estimators, kept_indices)`` — the Rust-backed structural check.

        Raises sklearn's own ``ValueError`` texts for a malformed list, a
        duplicate/colliding/``__``-containing name, an all-``'drop'`` list, and a
        non-regressor entry. Only that last rule is the regressor's own: unlike
        :class:`StackingClassifier`, which accepts regressors as members, a
        stack of regressors has no use for a classifier.
        """
        names, values, kept = self._validate_composition()
        for i in kept:
            if not is_regressor(values[i]):
                raise ValueError(
                    "The estimator {} should be a regressor.".format(
                        type(values[i]).__name__
                    )
                )
        return names, values, kept

    def _resolve_final_estimator(self):
        """Clone ``final_estimator`` (or sklearn's ``RidgeCV()`` default) into
        ``final_estimator_``, then require it to be a regressor."""
        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            from sklearn.linear_model import RidgeCV

            self.final_estimator_ = clone(RidgeCV())
        if not is_regressor(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a regressor. Got {}".format(
                    self.final_estimator_
                )
            )

    # -- fit --------------------------------------------------------------- #

    def fit(self, X, y, **fit_params):
        """Fit the base estimators, then the final estimator on their predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        **fit_params : dict
            ``sample_weight`` is forwarded to every sub-estimator's ``fit``.
            Anything else requires ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        self : object
        """
        _raise_for_extra_fit_params(fit_params, self, "fit", allow=["sample_weight"])
        y = column_or_1d(y, warn=True)

        names, all_estimators, kept = self._validate_estimators()
        self._resolve_final_estimator()
        prefit = self._cv_is_prefit()

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch()
            for name in names:
                routed_params[name] = Bunch(fit={})
                if "sample_weight" in fit_params:
                    routed_params[name].fit["sample_weight"] = fit_params[
                        "sample_weight"
                    ]

        # One decision covers BOTH fan-outs below AND the inner
        # `cross_val_predict` calls' own `n_jobs`, so it is resolved once here
        # rather than re-derived at each `Parallel`.
        members = [all_estimators[i] for i in kept] + [self.final_estimator_]
        n_jobs = _effective_n_jobs(self.n_jobs, members)
        self._fit_members(
            X, y, names, all_estimators, kept, routed_params, prefit, fit_params, n_jobs
        )
        return self

    def _fit_members(
        self,
        X,
        y,
        names,
        all_estimators,
        kept,
        routed_params,
        prefit,
        fit_params,
        n_jobs,
    ):
        """The body of :meth:`fit`, at the resolved (device-safe) ``n_jobs``."""
        if prefit:
            self.estimators_ = []
            for i in kept:
                check_is_fitted(all_estimators[i])
                self.estimators_.append(all_estimators[i])
        else:
            self.estimators_ = _parallel(n_jobs, "2*n_jobs")(
                _delayed(_fit_one)(
                    clone(all_estimators[i]), X, y, routed_params[names[i]]["fit"]
                )
                for i in kept
            )

        self._record_named_estimators(names, kept)

        # A regressor's only response method is `predict`; sklearn still stores
        # the list so `stack_method_` reads the same on both classes.
        for i in kept:
            if not hasattr(all_estimators[i], "predict"):
                raise ValueError(
                    f"Underlying estimator {names[i]} does not implement the "
                    f"method predict."
                )
        self.stack_method_ = ["predict"] * len(kept)

        if prefit:
            predictions = [all_estimators[i].predict(X) for i in kept]
        else:
            cv = check_cv(self.cv, y=y, classifier=False)
            if hasattr(cv, "random_state") and cv.random_state is None:
                # sklearn pins a concrete RandomState so every base estimator
                # sees the SAME folds even under `shuffle=True`. `deepcopy`
                # below then keeps the per-estimator calls from advancing a
                # shared generator.
                cv.random_state = np.random.RandomState()
            predictions = _parallel(n_jobs, "2*n_jobs")(
                _delayed(cross_val_predict)(
                    clone(all_estimators[i]),
                    X,
                    y,
                    cv=_deepcopy(cv),
                    method="predict",
                    n_jobs=n_jobs,
                    params=routed_params[names[i]]["fit"],
                    verbose=self.verbose,
                )
                for i in kept
            )

        X_meta = self._concatenate_predictions(X, predictions)
        _fit_one(self.final_estimator_, X_meta, y, fit_params)

    def fit_transform(self, X, y, **fit_params):
        """``fit(X, y).transform(X)`` — the meta features for the training rows."""
        _raise_for_extra_fit_params(fit_params, self, "fit", allow=["sample_weight"])
        return self.fit(X, y, **fit_params).transform(X)

    # -- meta-feature assembly --------------------------------------------- #

    def _concatenate_predictions(self, X, predictions):
        """The meta matrix: one block per kept estimator, then ``X`` if passthrough.

        A regressor's ``predict`` is 1-D, so every block is one column; that
        reshape is the only thing this adds to the shared
        :meth:`_StackComposition._assemble_meta`.
        """
        blocks = []
        for pred in predictions:
            pred = np.asarray(pred)
            blocks.append(pred.reshape(-1, 1) if pred.ndim == 1 else pred)
        return self._assemble_meta(X, blocks)

    # -- transform / predict ------------------------------------------------ #

    def transform(self, X):
        """The base estimators' predictions for ``X``, as the meta-feature matrix.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_estimators [+ n_features])
        """
        check_is_fitted(self)
        predictions = [est.predict(X) for est in self.estimators_]
        return self._concatenate_predictions(X, predictions)

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Predict ``X`` by feeding the base predictions to ``final_estimator_``.

        ``**predict_params`` reach the FINAL estimator only (e.g. ``return_std``
        on a :class:`mlrs.BayesianRidge` meta learner), so any uncertainty they
        return accounts for the meta step alone.
        """
        check_is_fitted(self)
        return self.final_estimator_.predict(self.transform(X), **predict_params)


# =========================================================================== #
# StackingClassifier (STACK-CLF-01)
# =========================================================================== #
#
# The classifier shares every structural rule with `StackingRegressor` — the
# `'drop'` bookkeeping, the name validation, the `cv="prefit"` route, the column
# layout, the feature names — through `_StackComposition`, and adds the three
# things a classifier needs. All three are decided in Rust:
#
#   ==============================  ========================================
#   what                            where the work happens
#   ==============================  ========================================
#   `stack_method` validation       Rust (`stacking_stack_method`)
#   `"auto"` fallback chain         Rust (`stacking_resolve_stack_methods`)
#   the dropped-column rule         Rust (`stacking_classifier_meta_slices`)
#   ==============================  ========================================
#
# ## The dropped column is the whole reason the layout differs
#
# A binary `predict_proba` returns two columns that sum to one, so sklearn hands
# the meta learner only the second — otherwise `final_estimator_` gets two
# perfectly collinear features. A multiclass one is passed whole; a
# `decision_function` is passed whole (a signed margin has no collinear twin);
# `predict` is one column of ENCODED labels. Which of those applies is a
# function of (resolved method, response shape, len(classes_)), and that
# function is `mlrs_algos::ensemble::stacking::classifier_meta_slices`. The shim
# only performs the slice it is told to.
#
# ## The default `final_estimator` is sklearn's `LogisticRegression`
#
# Same reasoning as the regressor's `RidgeCV` default: sklearn's default is
# `LogisticRegression()`, mlrs ships a `LogisticRegression` whose defaults are
# its own, and silently substituting it would move every default-constructed
# stack off the baseline users are migrating from. Pass
# `final_estimator=mlrs.LogisticRegression()` to put the meta fit on the device.


class StackingClassifier(
    ClassifierMixin, TransformerMixin, MetaEstimatorMixin, _StackComposition, BaseEstimator
):
    """Stack of classifiers with a final classifier (STACK-CLF-01).

    ``StackingClassifier(estimators, final_estimator=None, *, cv=None,
    stack_method="auto", n_jobs=None, passthrough=False, verbose=0)`` — the full
    :class:`sklearn.ensemble.StackingClassifier` parameter surface, with the
    cross-validation index generation and the composition bookkeeping in Rust.

    Base estimators are fitted on the whole of ``X`` and exposed as
    ``estimators_``; ``final_estimator_`` is fitted on their **out-of-fold**
    responses (:func:`mlrs.model_selection.cross_val_predict`), so the meta
    learner never sees a base estimator's in-sample fit.

    Parameters
    ----------
    estimators : list of (str, estimator)
        The base estimators. Classifiers normally, but a **regressor is
        accepted** — sklearn allows it for ordinal problems, and mlrs matches
        that. An entry's estimator may be the string ``'drop'`` (usually via
        ``set_params(name='drop')``) to disable it; a dropped entry keeps its
        slot in ``named_estimators_`` (as ``'drop'``) but contributes no meta
        column and is never fitted.
    final_estimator : estimator, default=None
        The classifier fitted on the meta features. ``None`` means
        ``sklearn.linear_model.LogisticRegression()``, sklearn's own default —
        see this module's ``StackingClassifier`` section for why mlrs does not
        substitute its own.
    cv : int, cross-validation generator, iterable, or "prefit", default=None
        Passed to :func:`mlrs.model_selection.cross_val_predict`. ``None`` is
        5-fold :class:`mlrs.model_selection.StratifiedKFold` (a classifier
        stratifies). ``"prefit"`` assumes every entry in ``estimators`` is
        ALREADY fitted: nothing is cloned or refitted, and the meta features are
        the base estimators' responses on the FULL training set rather than
        out-of-fold ones — much cheaper and, if those estimators were fitted on
        this same data, badly overfit.
    stack_method : {"auto", "predict_proba", "decision_function", "predict"}, \
default="auto"
        The response method each base estimator is asked for. ``"auto"`` takes
        the first of ``predict_proba``, ``decision_function``, ``predict`` that
        the estimator implements — resolved per estimator, so a stack may mix
        methods; the choices land in ``stack_method_``. A NAMED method every
        member must implement, or ``fit`` raises.
    n_jobs : int, default=None
        joblib parallelism for the base-estimator fits and for the inner
        ``cross_val_predict`` calls. ``None`` means 1. Reduced to serial (with a
        warning) when a member is an mlrs estimator — see
        :func:`_effective_n_jobs`.
    passthrough : bool, default=False
        Append the original ``X`` columns to the meta features.
    verbose : int, default=0
        Forwarded to the inner ``cross_val_predict`` calls.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The class labels, in the order the encoded targets use. A list, one
        array per column, when ``y`` is a multilabel indicator.
    estimators_ : list of estimator
        The fitted base estimators, dropped entries excluded. Under
        ``cv="prefit"`` these are the caller's own objects, not clones.
    named_estimators_ : :class:`sklearn.utils.Bunch`
        ``name -> fitted estimator`` (or the string ``'drop'``).
    final_estimator_ : estimator
        The fitted meta classifier.
    stack_method_ : list of str
        The response method resolved for each kept estimator.
    n_features_in_ : int
        Feature count of the ORIGINAL ``X``, read off ``estimators_[0]``.
    feature_names_in_ : ndarray of str
        Present only when a fitted base estimator exposes it.

    Notes
    -----
    When a base estimator contributes ``predict_proba`` on a **binary** problem,
    its first column is dropped: ``p(y=0) = 1 - p(y=1)``, so both columns would
    be perfectly collinear. This is sklearn's rule and is why a binary stack's
    meta matrix has one column per estimator while a 3-class one has three.

    Examples
    --------
    >>> import numpy as np
    >>> import mlrs
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 5)).astype(np.float32)
    >>> y = (X[:, 0] > 0).astype(np.int64)
    >>> clf = mlrs.StackingClassifier(
    ...     estimators=[("nb", mlrs.GaussianNB()), ("knn", mlrs.KNeighborsClassifier())],
    ...     cv=3,
    ... )
    >>> clf.fit(X, y).predict(X[:2]).shape
    (2,)
    """

    def __init__(
        self,
        estimators,
        final_estimator=None,
        *,
        cv=None,
        stack_method="auto",
        n_jobs=None,
        passthrough=False,
        verbose=0,
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose

    # -- validation -------------------------------------------------------- #

    def _validate_estimators(self):
        """``(names, estimators, kept_indices)`` — the Rust-backed structural check.

        Deliberately does NOT require the members to be classifiers: sklearn's
        ``StackingClassifier`` overrides the base check for exactly that reason
        (a regressor first layer is how ordinal regression is stacked), and a
        parity shim that tightened it would reject fits sklearn completes.
        """
        return self._validate_composition()

    def _resolve_final_estimator(self):
        """Clone ``final_estimator`` (or sklearn's ``LogisticRegression()``
        default) into ``final_estimator_``, then require it to be a classifier."""
        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            from sklearn.linear_model import LogisticRegression

            self.final_estimator_ = clone(LogisticRegression())
        if not is_classifier(self.final_estimator_):
            raise ValueError(
                "'final_estimator' parameter should be a classifier. Got {}".format(
                    self.final_estimator_
                )
            )

    def _resolve_stack_methods(self, names, all_estimators, kept):
        """``stack_method_`` — one resolved response method per KEPT estimator.

        The ``hasattr`` probes stay here because an ``available_if`` descriptor
        decides them at access time (``SVC.predict_proba`` exists only under
        ``probability=True``); the preference order, the ``"auto"`` fallback and
        the rejection message are Rust's.

        Only the KEPT estimators are resolved. A dropped entry is never asked
        for a method — sklearn returns ``None`` for it before checking that the
        method exists — so a stack whose only proba-less member is ``'drop'``
        fits under ``stack_method="predict_proba"``.

        The ``ValueError`` this can raise is sklearn's *"Underlying estimator
        {name} does not implement the method {method}."* and propagates as-is;
        the other thing Rust rejects here, an unrecognized ``stack_method``
        string, has already been reported by :meth:`_check_stack_method` under
        ``InvalidParameterError``.
        """
        implements = [
            (
                hasattr(all_estimators[i], "predict_proba"),
                hasattr(all_estimators[i], "decision_function"),
                hasattr(all_estimators[i], "predict"),
            )
            for i in kept
        ]
        return _stack_ext().stacking_resolve_stack_methods(
            [names[i] for i in kept], self.stack_method, implements
        )

    def _check_stack_method(self):
        """Validate the ``stack_method`` string, at sklearn's point in ``fit``.

        sklearn's ``@validate_params`` runs before ``_validate_estimators``, so
        an unrecognized ``stack_method`` is reported ahead of anything about the
        ``estimators`` list — a caller who got both wrong sees the same
        complaint from both libraries.
        """
        try:
            _stack_ext().stacking_stack_method(self.stack_method)
        except ValueError as exc:
            raise InvalidParameterError(str(exc)) from None

    # -- fit --------------------------------------------------------------- #

    def fit(self, X, y, **fit_params):
        """Fit the base estimators, then the final estimator on their responses.

        ``y`` is label-encoded before anything else — the meta learner and every
        base estimator see ``0..n_classes-1``, and ``predict`` maps back through
        ``classes_``. A multilabel-indicator ``y`` is encoded column by column
        and ``classes_`` becomes a list.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        **fit_params : dict
            ``sample_weight`` is forwarded to every sub-estimator's ``fit``.
            Anything else requires
            ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        self : object
        """
        _raise_for_extra_fit_params(fit_params, self, "fit", allow=["sample_weight"])
        check_classification_targets(y)
        # Both string-valued CONSTRUCTOR parameters are validated here, at the
        # point sklearn's `@validate_params` runs — before anything about the
        # `estimators` list is looked at. A caller who got both wrong therefore
        # sees the same complaint from both libraries.
        self._check_stack_method()
        prefit = self._cv_is_prefit()

        if type_of_target(y) == "multilabel-indicator":
            y = np.asarray(y)
            self._label_encoder = [LabelEncoder().fit(col) for col in y.T]
            self.classes_ = [le.classes_ for le in self._label_encoder]
            y_encoded = np.array(
                [le.transform(col) for le, col in zip(self._label_encoder, y.T)]
            ).T
        else:
            y = column_or_1d(y, warn=True)
            self._label_encoder = LabelEncoder().fit(y)
            self.classes_ = self._label_encoder.classes_
            y_encoded = self._label_encoder.transform(y)

        names, all_estimators, kept = self._validate_estimators()
        self._resolve_final_estimator()

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch()
            for name in names:
                routed_params[name] = Bunch(fit={})
                if "sample_weight" in fit_params:
                    routed_params[name].fit["sample_weight"] = fit_params[
                        "sample_weight"
                    ]

        # One decision covers BOTH fan-outs below AND the inner
        # `cross_val_predict` calls' own `n_jobs`, so it is resolved once here
        # rather than re-derived at each `Parallel`.
        members = [all_estimators[i] for i in kept] + [self.final_estimator_]
        n_jobs = _effective_n_jobs(self.n_jobs, members, type(self).__name__)
        self._fit_members(
            X,
            y_encoded,
            names,
            all_estimators,
            kept,
            routed_params,
            prefit,
            fit_params,
            n_jobs,
        )
        return self

    def _fit_members(
        self,
        X,
        y,
        names,
        all_estimators,
        kept,
        routed_params,
        prefit,
        fit_params,
        n_jobs,
    ):
        """The body of :meth:`fit`, on the ENCODED ``y`` and at the resolved
        (device-safe) ``n_jobs``."""
        if prefit:
            self.estimators_ = []
            for i in kept:
                check_is_fitted(all_estimators[i])
                self.estimators_.append(all_estimators[i])
        else:
            self.estimators_ = _parallel(n_jobs, "2*n_jobs")(
                _delayed(_fit_one)(
                    clone(all_estimators[i]), X, y, routed_params[names[i]]["fit"]
                )
                for i in kept
            )

        self._record_named_estimators(names, kept)
        self.stack_method_ = self._resolve_stack_methods(names, all_estimators, kept)

        if prefit:
            responses = [
                getattr(all_estimators[i], method)(X)
                for i, method in zip(kept, self.stack_method_)
            ]
        else:
            cv = check_cv(self.cv, y=y, classifier=True)
            if hasattr(cv, "random_state") and cv.random_state is None:
                # sklearn pins a concrete RandomState so every base estimator
                # sees the SAME folds even under `shuffle=True`. `deepcopy`
                # below then keeps the per-estimator calls from advancing a
                # shared generator.
                cv.random_state = np.random.RandomState()
            responses = _parallel(n_jobs, "2*n_jobs")(
                _delayed(cross_val_predict)(
                    clone(all_estimators[i]),
                    X,
                    y,
                    cv=_deepcopy(cv),
                    method=method,
                    n_jobs=n_jobs,
                    params=routed_params[names[i]]["fit"],
                    verbose=self.verbose,
                )
                for i, method in zip(kept, self.stack_method_)
            )

        X_meta = self._concatenate_predictions(X, responses)
        _fit_one(self.final_estimator_, X_meta, y, fit_params)

    def fit_transform(self, X, y, **fit_params):
        """``fit(X, y).transform(X)`` — the meta features for the training rows."""
        _raise_for_extra_fit_params(fit_params, self, "fit", allow=["sample_weight"])
        return self.fit(X, y, **fit_params).transform(X)

    # -- meta-feature assembly --------------------------------------------- #

    def _concatenate_predictions(self, X, responses):
        """The meta matrix, with the collinear probability columns dropped.

        Which columns of which response block survive is
        ``mlrs_algos::ensemble::stacking::classifier_meta_slices``' answer, a
        function of the resolved method, the response's shape and
        ``len(classes_)``. This method performs the slices it is handed and
        nothing else; the layout and copy are the shared
        :meth:`_StackComposition._assemble_meta`.
        """
        kinds, cols, arrays = [], [], []
        for response in responses:
            if isinstance(response, list):
                # A multilabel `predict_proba`: one `(n, n_classes_j)` block per
                # target. `np.asarray` on the list would build a 3-D array and
                # lose the per-target shapes, so each is kept separate.
                blocks = [np.asarray(part) for part in response]
                kinds.append(2)
                cols.append([int(b.shape[1]) for b in blocks])
                arrays.append(blocks)
            else:
                arr = np.asarray(response)
                arrays.append([arr])
                if arr.ndim == 1:
                    kinds.append(0)
                    cols.append([])
                else:
                    kinds.append(1)
                    cols.append([int(arr.shape[1])])

        n_classes = len(self.classes_)
        slices = _stack_ext().stacking_classifier_meta_slices(
            list(self.stack_method_), kinds, cols, n_classes
        )

        blocks = []
        for block, sub, start_col, n_cols in slices:
            arr = arrays[block][sub]
            if arr.ndim == 1:
                blocks.append(arr.reshape(-1, 1))
            else:
                blocks.append(arr[:, start_col : start_col + n_cols])
        return self._assemble_meta(X, blocks)

    # -- transform / predict ------------------------------------------------ #

    def transform(self, X):
        """The base estimators' responses for ``X``, as the meta-feature matrix.

        Returns
        -------
        y_preds : ndarray of shape (n_samples, n_meta [+ n_features])
            One column per estimator when every member contributes ``predict``,
            a ``decision_function`` on a binary problem, or a binary
            ``predict_proba``; ``n_classes`` columns per estimator otherwise.
        """
        check_is_fitted(self)
        responses = [
            getattr(est, method)(X)
            for est, method in zip(self.estimators_, self.stack_method_)
        ]
        return self._concatenate_predictions(X, responses)

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **predict_params):
        """Predict class labels for ``X``, mapped back through ``classes_``.

        ``**predict_params`` reach the FINAL estimator only.
        """
        check_is_fitted(self)
        if _routing_enabled():
            routed_params = process_routing(self, "predict", **predict_params)
            predict_params = routed_params.final_estimator_["predict"]
        y_pred = self.final_estimator_.predict(self.transform(X), **predict_params)
        if isinstance(self._label_encoder, list):
            return np.array(
                [
                    le.inverse_transform(col)
                    for le, col in zip(self._label_encoder, np.asarray(y_pred).T)
                ]
            ).T
        return self._label_encoder.inverse_transform(y_pred)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Class probabilities from ``final_estimator_``.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            For a multilabel ``y`` this is ``(n_samples, n_outputs)`` — the
            positive-class probability of each output, which is what sklearn
            reduces the meta learner's per-output list to.
        """
        check_is_fitted(self)
        y_pred = self.final_estimator_.predict_proba(self.transform(X))
        if isinstance(self._label_encoder, list):
            return np.array([preds[:, 0] for preds in y_pred]).T
        return y_pred

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X):
        """``final_estimator_``'s decision function over the meta features."""
        check_is_fitted(self)
        return self.final_estimator_.decision_function(self.transform(X))


# =========================================================================== #
# VotingRegressor (VOTE-01)
# =========================================================================== #
#
# A voting regressor is the SIMPLEST heterogeneous ensemble sklearn ships: fit
# every member on the whole of `X`, then answer `predict` with the weighted mean
# of their predictions. There is no meta learner, no cross-validation, and no
# `passthrough` — which is exactly why its Rust surface looks different from
# `StackingRegressor`'s.
#
# ## What is in Rust, and why the split falls where it does
#
#   ============================  ============================================
#   what                          where the work happens
#   ============================  ============================================
#   name / `'drop'` validation    Rust, SHARED with stacking
#                                 (`stacking_validate_names`,
#                                 `stacking_kept_indices`) — these are
#                                 sklearn's own `_BaseHeterogeneousEnsemble`
#                                 rules, not stacking's
#   `weights` length rule         Rust (`voting_check_weights`)
#   `_weights_not_none`           Rust (`voting_active_weight_slots` — POSITIONS,
#                                 not values, so the weight objects' dtype
#                                 survives; see `_active_weights`)
#   `get_feature_names_out`       Rust (`voting_feature_names`)
#   the aggregation itself        Rust/CubeCL (`voting_aggregate`) OR numpy,
#                                 chosen by `MLRS_VOTING_ENGINE`
#   ============================  ============================================
#
# ## The aggregation is the only part that carries data — and it REDUCES
#
# `transform` is the `n x k` column stack `np.asarray([...]).T`; `predict` is
# `np.average(that, axis=1, weights=w)`. Stacking's equivalent operation is a
# pure copy, which is why `docs/stacking.md` concluded that `np.hstack` wins on
# every backend. `predict` here is different in kind: it consumes `n * k` and
# emits `n`, so the device arm's download shrinks by a factor of `k` and there
# is real arithmetic (`k` multiplies, `k - 1` adds per row) to amortise the
# crossing against. That made the arms worth building and measuring rather than
# assuming; `docs/voting.md` carries the ladder, and the DEFAULT is whatever it
# says.
#
# ## `np.average`'s evaluation order is reproduced exactly, not approximated
#
# numpy forms `mat * w`, reduces the row, and then DIVIDES by `w.sum()`. All
# three arms do those three things in that order, in the input dtype — so the
# host and device arms are bit-identical to numpy, not merely within 1e-5, for
# any `k` below numpy's pairwise-summation cutoff. That is what lets
# `test_voting_engine.py` assert EXACT equality for the numpy and host arms,
# which is the only assertion strong enough to catch an accumulation-order
# regression. The `device` arm is the one exception, and a documented one: a GPU
# contracts `acc + pred*w` into a fused multiply-add, rounding once where numpy
# rounds twice, so it lands within one ULP (measured on rocm gfx1151) and is
# gated to a few ULP rather than to equality. That is more accurate than the
# reference, not less, and two orders inside the project's 1e-5 contract.
#
# ## `weights` is indexed against the FULL `estimators` list
#
# sklearn requires `len(weights) == len(estimators)` and only THEN drops the
# weights of `'drop'`ped entries. So `set_params(lr='drop')` on a weighted
# ensemble keeps working without the caller re-writing `weights` — and a shim
# that filtered before checking would reject a fit sklearn completes.


def _is_array_like_not_scalar(value):
    """sklearn ``_is_arraylike_not_scalar``, for the ``weights`` constraint.

    Reimplemented rather than imported for the reason
    :func:`_generate_input_feature_names` gives — it is three lines behind a
    private sklearn path. The ``np.isscalar`` half is what rejects a bare
    ``3`` and a bare ``"abc"``, both of which have the duck-typed shape the
    first half looks for.
    """
    array_like = (
        hasattr(value, "__len__") or hasattr(value, "shape") or hasattr(value, "__array__")
    )
    return array_like and not np.isscalar(value)


def _vote_via_rust(columns, mode, weights, engine):
    """The aggregation from ``_mlrs.voting_aggregate``, or ``None`` for numpy.

    ``columns`` is one 1-D prediction per kept member. Returns ``None`` —
    deliberately, rather than raising — for every input the Rust arms cannot
    represent, leaving numpy to handle it exactly as it did before this arm
    existed:

    * a non-float column (an integer or object array — a member is free to
      return one, and ``np.average`` promotes it);
    * a column that is not 1-D, or whose length disagrees with the others.

    The dtype handed over is ``np.result_type`` of the columns AND the weights,
    which is the promotion numpy would have applied — so the ``host`` arm is
    bit-identical to numpy rather than merely close, and the ``device`` arm is
    within a few ULP of it (a GPU contracts ``acc + pred*w`` into one FMA, which
    rounds once where numpy rounds twice; see ``mlrs_kernels::voting``).
    """
    import pyarrow as pa

    arrays = [np.asarray(c) for c in columns]
    if any(a.ndim != 1 for a in arrays):
        return None
    dtypes = [a.dtype for a in arrays]
    if weights is not None:
        dtypes.append(np.asarray(weights).dtype)
    try:
        dtype = np.result_type(*dtypes)
    except TypeError:
        return None
    if dtype != np.float32 and dtype != np.float64:
        return None
    n_rows = int(arrays[0].shape[0])
    if any(int(a.shape[0]) != n_rows for a in arrays):
        return None
    if n_rows == 0:
        # A zero-row aggregation is a zero-byte device allocation; numpy already
        # produces the right empty shape and there is nothing to hand CubeCL.
        return None

    arrow_type = pa.float32() if dtype == np.float32 else pa.float64()
    flats, capsules = [], []
    for a in arrays:
        # `ascontiguousarray` is a no-op view when `a` is already contiguous in
        # the promoted dtype, and `py_buffer` does not copy — so a column reaches
        # Rust without a staging copy, and the arms are compared on the work they
        # actually do rather than on ingress overhead.
        flat = np.ascontiguousarray(a, dtype=dtype)
        flats.append(flat)
        capsules.append(
            pa.Array.from_buffers(arrow_type, flat.size, [None, pa.py_buffer(flat)])
        )

    flat_out, n_cols = _stack_ext().voting_aggregate(
        capsules,
        n_rows,
        mode,
        None if weights is None else [float(w) for w in weights],
        engine,
    )
    # `flats` is referenced until here on purpose: `py_buffer` borrows those
    # numpy buffers, and letting one be collected mid-call would free memory
    # Rust is still reading.
    del flats
    shape = (n_rows,) if mode == "predict" else (n_rows, n_cols)
    return _io.to_output(flat_out, shape, "numpy", dtype)


class _VoteComposition(_HeterogeneousComposition):
    """The ``_BaseVoting`` half :class:`VotingRegressor` and
    :class:`VotingClassifier` share (VOTE-CLF-01).

    Everything here is identical between the two by sklearn's own construction —
    it lives on ``_BaseVoting``, one level below the two concrete classes, and
    sits on the ``estimators``-list mechanics
    :class:`_HeterogeneousComposition` already provides:

    * ``_weights_not_none`` (:meth:`_active_weights`), the ``verbose`` line, and
      the member fan-out that IS ``_BaseVoting.fit``;
    * ``n_features_in_`` and the metadata router, both of which read only the
      composed members;
    * :meth:`_member_predictions`, the ``est.predict(X)`` column collection that
      feeds hard voting and the regressor's average alike.

    What each class adds on top is exactly what ``voting`` decides — the
    classifier's label encoding, its probability route, its `flatten_transform`
    — and none of that reaches this far down.

    Not an estimator base class in its own right: no ``__init__``, so
    ``get_params`` still reads each concrete class's own signature.
    """

    # -- validation --------------------------------------------------------- #

    def _active_weights(self, is_drop):
        """``_weights_not_none``: the surviving members' weights, or ``None``.

        ``None`` in, ``None`` out — the uniform case never materialises a weight
        vector, so ``np.average`` takes its own ``weights=None`` fast path (which
        is measurably faster; see ``docs/voting.md``) and the Rust arms take
        theirs.

        The weight OBJECTS are never converted here, only selected: sklearn
        returns them as they were given, and numpy infers ``predict``'s result
        dtype from them — a ``float32`` weight array keeps an f32 problem in f32
        where a Python-float list would promote it to f64. So Rust answers with
        POSITIONS (the length rule and the drop filter are still its), and this
        indexes the caller's own list.
        """
        if self.weights is None:
            return None
        # `list(...)`, matching sklearn's `zip(self.estimators, self.weights)`:
        # any sequence works, and a numpy array yields its scalars with their
        # dtype intact.
        weights = list(self.weights)
        slots = _stack_ext().voting_active_weight_slots(len(weights), list(is_drop))
        return [weights[i] for i in slots]

    def _weights_for_predict(self):
        """:meth:`_active_weights` against the CURRENT ``estimators`` list.

        Resolved at predict time rather than cached at fit time because that is
        where sklearn reads it — ``_weights_not_none`` is a property over
        ``self.weights`` and ``self.estimators``, so a ``set_params`` between
        ``fit`` and ``predict`` is visible in both libraries or in neither.
        """
        drop = _stack_ext().stacking_drop_sentinel()
        return self._active_weights([_is_drop(est, drop) for _, est in self.estimators])

    def _log_message(self, name, idx, total):
        """sklearn ``_BaseVoting._log_message``: the ``verbose=True`` line."""
        if not self.verbose:
            return None
        return f"({idx} of {total}) Processing {name}"

    # -- fit ---------------------------------------------------------------- #

    def _fit_members(self, X, y, names, all_estimators, kept, fit_params):
        """``_BaseVoting.fit``'s body: the weights rule, routing, and the fan-out.

        ``y`` is whatever the caller's ``fit`` decided the members should see —
        the raw target for a regressor, the LABEL-ENCODED one for a classifier —
        so the encoding stays with the class that owns it and the fan-out stays
        here.

        The weights length rule runs against the FULL list and BEFORE the drop
        filter, which is what makes ``set_params(name='drop')`` work on a
        weighted ensemble without the caller rewriting ``weights``.
        """
        if self.weights is not None:
            # `len(self.weights)`, exactly as sklearn writes it — so a weights
            # object without a length raises the same TypeError there and here.
            _stack_ext().voting_check_weights(len(self.weights), len(self.estimators))

        if _routing_enabled():
            routed_params = process_routing(self, "fit", **fit_params)
        else:
            routed_params = Bunch()
            for name in names:
                routed_params[name] = Bunch(fit={})
                if "sample_weight" in fit_params:
                    routed_params[name].fit["sample_weight"] = fit_params[
                        "sample_weight"
                    ]

        n_jobs = _effective_n_jobs(
            self.n_jobs, [all_estimators[i] for i in kept], owner=type(self).__name__
        )
        total = len(kept)
        self.estimators_ = _parallel(n_jobs, "2*n_jobs")(
            _delayed(_fit_one)(
                clone(all_estimators[i]),
                X,
                y,
                routed_params[names[i]]["fit"],
                "Voting",
                self._log_message(names[i], idx + 1, total),
            )
            for idx, i in enumerate(kept)
        )

        self._record_named_estimators(names, kept)
        return self

    # -- aggregation -------------------------------------------------------- #

    def _member_predictions(self, X):
        """Each fitted member's prediction, in kept order, AS RETURNED.

        Deliberately not reshaped or ravelled. sklearn's ``_BaseVoting._predict``
        is ``np.asarray([est.predict(X) for est in self.estimators_]).T``, so a
        member that answers with an ``(n, 1)`` column produces a 3-D stack there
        — odd, but observable, and flattening it here would make mlrs disagree
        with sklearn on exactly the input where sklearn is surprising. The Rust
        arms decline anything that is not 1-D (:func:`_vote_via_rust` and
        :func:`_vote_labels_via_rust` both return ``None``), so numpy handles
        that shape on every arm.
        """
        return [np.asarray(est.predict(X)) for est in self.estimators_]

    def _kept_names(self):
        """The names of the members that survived the ``'drop'`` filter."""
        drop = _stack_ext().stacking_drop_sentinel()
        return [name for name, est in self.estimators if not _is_drop(est, drop)]

    # -- introspection ------------------------------------------------------ #

    @property
    def n_features_in_(self):
        """Feature count of ``X``, read off ``estimators_[0]``.

        sklearn's voting layer words this AttributeError differently from its
        stacking layer (``has no n_features_in_ attribute.`` versus ``has no
        attribute n_features_in_``). Both texts are reproduced where they
        belong rather than unified, because a caller matching on one of them is
        matching on the class it actually uses.
        """
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(type(self).__name__)
            ) from nfe
        return self.estimators_[0].n_features_in_

    def get_metadata_routing(self):
        """Route ``fit`` metadata to each named member.

        No ``final_estimator`` node, unlike stacking's router — a voting
        ensemble has no second stage for anything to be routed to.
        """
        router = MetadataRouter(owner=type(self).__name__)
        for name, estimator in self.estimators:
            router.add(
                **{name: estimator},
                method_mapping=MethodMapping().add(callee="fit", caller="fit"),
            )
        return router


class VotingRegressor(
    RegressorMixin, TransformerMixin, MetaEstimatorMixin, _VoteComposition, BaseEstimator
):
    """Prediction voting regressor (VOTE-01).

    ``VotingRegressor(estimators, *, weights=None, n_jobs=None, verbose=False)``
    — the full :class:`sklearn.ensemble.VotingRegressor` parameter surface, with
    the composition bookkeeping and the prediction aggregation in Rust.

    Every member is fitted on the whole of ``X`` (there is no cross-validation
    and no meta learner) and ``predict`` returns the weighted mean of their
    predictions.

    Parameters
    ----------
    estimators : list of (str, estimator)
        The regressors to average. An entry's estimator may be the string
        ``'drop'`` (usually via ``set_params(name='drop')``) to disable it; a
        dropped entry keeps its slot in ``named_estimators_`` (as ``'drop'``)
        and its slot in ``weights``, but is never fitted and contributes no
        column.
    weights : array-like of shape (n_estimators,), default=None
        Per-member weights for the average. ``None`` is uniform. Indexed against
        the FULL ``estimators`` list — a dropped entry still needs its slot, and
        its weight is discarded afterwards, so ``set_params(name='drop')`` does
        not require rewriting ``weights``.
    n_jobs : int, default=None
        joblib parallelism for the member fits. ``None`` means 1. Reduced to
        serial (with a warning) when a member is an mlrs estimator — see
        :func:`_effective_n_jobs`.
    verbose : bool, default=False
        Print each member's fit time as it completes.

    Attributes
    ----------
    estimators_ : list of estimator
        The fitted members, dropped entries excluded.
    named_estimators_ : :class:`sklearn.utils.Bunch`
        ``name -> fitted estimator`` (or the string ``'drop'``).
    n_features_in_ : int
        Feature count of ``X``, read off ``estimators_[0]``.
    feature_names_in_ : ndarray of str
        Present only when a fitted member exposes it.

    Notes
    -----
    ``allow_nan`` and ``sparse`` are DERIVED from the members (the AND over
    them), because this estimator never touches ``X`` itself — it hands it
    straight to the members. So a ``VotingRegressor`` over sklearn estimators
    accepts sparse input and one over mlrs estimators does not.

    The aggregation runs on the arm ``MLRS_VOTING_ENGINE`` names — ``numpy``
    (the default), ``host`` (Rust), or ``device`` (CubeCL). ``numpy`` and
    ``host`` produce bit-identical values; ``device`` agrees to within a few ULP
    (a GPU fuses the multiply-add, rounding once where numpy rounds twice). See
    ``docs/voting.md`` for the measured ladder behind the default.

    Examples
    --------
    >>> import numpy as np
    >>> import mlrs
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 5)).astype(np.float32)
    >>> y = (X[:, 0] * 3.0 - X[:, 1]).astype(np.float32)
    >>> reg = mlrs.VotingRegressor(
    ...     estimators=[("lr", mlrs.LinearRegression()), ("ridge", mlrs.Ridge())],
    ...     weights=[2.0, 1.0],
    ... )
    >>> reg.fit(X, y).predict(X[:2]).shape
    (2,)
    """

    def __init__(self, estimators, *, weights=None, n_jobs=None, verbose=False):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    # -- validation --------------------------------------------------------- #

    def _validate_params(self):
        """sklearn's ``_parameter_constraints`` for this class, reproduced with
        its wording.

        sklearn applies these through ``@_fit_context``, which runs BEFORE
        ``fit``'s body — so a non-list ``estimators`` is an
        ``InvalidParameterError`` about the TYPE, not the structural
        "``'estimators'`` should be a non-empty list of (string, estimator)
        tuples" message that the composition check would otherwise reach first.
        The two are different exception classes and a caller can see which.

        These stay in Python rather than joining the rest of the surface in
        Rust, for the reason ``linear.py``'s ``_validate_params`` gives: every
        rule here is a predicate on an arbitrary PYTHON object
        (``isinstance``, ``np.isscalar``, "has ``__len__``"), which is not a
        question Rust can be asked. Only the message templates could cross, and
        four format strings are not worth an FFI call per ``fit``.

        Unlike ``StackingClassifier``'s ``stack_method``, none of these render a
        ``StrOptions`` set, so there is no ``PYTHONHASHSEED`` ordering trap here
        and the texts are compared literally by the oracle.
        """

        def fail(name, expected, value):
            raise InvalidParameterError(
                f"The {name!r} parameter of {type(self).__name__} must be "
                f"{expected}. Got {value!r} instead."
            )

        if not isinstance(self.estimators, list):
            fail("estimators", "an instance of 'list'", self.estimators)
        if self.weights is not None and not _is_array_like_not_scalar(self.weights):
            fail("weights", "an array-like or None", self.weights)
        if self.n_jobs is not None and not isinstance(self.n_jobs, _Integral):
            fail("n_jobs", "None or an instance of 'int'", self.n_jobs)
        verbose_ok = isinstance(self.verbose, (bool, np.bool_)) or (
            isinstance(self.verbose, _Integral) and self.verbose >= 0
        )
        if not verbose_ok:
            fail(
                "verbose",
                "an int in the range [0, inf), an instance of 'bool' or an "
                "instance of 'numpy.bool'",
                self.verbose,
            )

    def _validate_estimators(self):
        """``(names, estimators, kept_indices)`` — the Rust-backed structural check.

        The list rules are :meth:`_HeterogeneousComposition._validate_composition`
        (shared with stacking, because they are sklearn's shared base-class
        rules). What voting adds is the regressor type check, which sklearn's
        ``_BaseHeterogeneousEnsemble._validate_estimators`` applies through
        ``is_regressor`` for a ``VotingRegressor``.
        """
        names, values, kept = self._validate_composition()
        for i in kept:
            if not is_regressor(values[i]):
                raise ValueError(
                    "The estimator {} should be a regressor.".format(
                        type(values[i]).__name__
                    )
                )
        return names, values, kept

    # -- fit ---------------------------------------------------------------- #

    def fit(self, X, y, **fit_params):
        """Fit every member on the whole of ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        **fit_params : dict
            ``sample_weight`` is forwarded to every member's ``fit``. Anything
            else requires ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        self : object
        """
        # FIRST, before anything else touches the arguments: sklearn's
        # `@_fit_context` runs the constraint checks ahead of `fit`'s body, and
        # the order is observable (a non-list `estimators` reports a TYPE error,
        # not the structural one).
        self._validate_params()
        _raise_for_extra_fit_params(fit_params, self, "fit", allow=["sample_weight"])
        y = column_or_1d(y, warn=True)

        names, all_estimators, kept = self._validate_estimators()
        return self._fit_members(X, y, names, all_estimators, kept, fit_params)

    # -- aggregation -------------------------------------------------------- #

    def _aggregate(self, columns, mode):
        """Run ``mode`` on the arm ``MLRS_VOTING_ENGINE`` names.

        ``numpy`` is the default arm AND the fallback for everything the Rust
        arms cannot represent (see :func:`_vote_via_rust`), so this method always
        has an answer.

        ``weights`` is resolved only for ``"predict"``. sklearn's ``transform``
        never reads them, and reading them here would let a ``weights`` mutated
        AFTER the fit — the only way it can be wrong at this point — break a
        ``transform`` that sklearn completes.
        """
        weights = self._weights_for_predict() if mode == "predict" else None
        engine = _stack_ext().voting_engine()
        if engine != "numpy":
            out = _vote_via_rust(columns, mode, weights, engine)
            if out is not None:
                return out
        if mode == "predict":
            return np.average(np.asarray(columns).T, axis=1, weights=weights)
        return np.asarray(columns).T

    # -- introspection ------------------------------------------------------ #

    def get_feature_names_out(self, input_features=None):
        """``votingregressor_<name>`` per kept member.

        One column per member and no ``passthrough``, so — unlike stacking —
        there is no within-block index and no input-name tail. ``input_features``
        is still VALIDATED and then discarded, which is sklearn's
        ``_check_feature_names_in(..., generate_names=False)``.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is not None:
            _generate_input_feature_names(self, input_features)
        names = _stack_ext().voting_feature_names(
            type(self).__name__.lower(), self._kept_names()
        )
        return np.asarray(names, dtype=object)

    # -- transform / predict ------------------------------------------------ #

    def transform(self, X):
        """Every member's prediction for ``X``, one column each.

        Returns
        -------
        predictions : ndarray of shape (n_samples, n_estimators)
        """
        check_is_fitted(self)
        return self._aggregate(self._member_predictions(X), "transform")

    def fit_transform(self, X, y=None, **fit_params):
        """``fit(X, y).transform(X)`` — the members' training-row predictions."""
        return self.fit(X, y, **fit_params).transform(X)

    def predict(self, X):
        """The weighted mean of the members' predictions.

        Returns
        -------
        y : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        return self._aggregate(self._member_predictions(X), "predict")


# =========================================================================== #
# VotingClassifier (VOTE-CLF-01)
# =========================================================================== #
#
# `VotingClassifier` is `VotingRegressor` plus ONE string-valued parameter, and
# that parameter forks the estimator so completely that the two halves share no
# data path at all:
#
#   =========  ==========================  ===================================
#   method     voting='hard'               voting='soft'
#   =========  ==========================  ===================================
#   predict    argmax of a weighted        argmax of the weighted probability
#              BINCOUNT over the members'  average
#              labels
#   proba      absent (`available_if`)     np.average(probas, axis=0)
#   transform  the (n, k) label matrix     np.hstack(probas), or the raw
#                                          (k, n, C) stack when
#                                          flatten_transform=False
#   =========  ==========================  ===================================
#
# ## What is in Rust
#
#   ============================  ============================================
#   what                          where the work happens
#   ============================  ============================================
#   name / `'drop'` validation    Rust, SHARED with stacking and the regressor
#   `weights` length rule         Rust (`voting_check_weights`)
#   `_weights_not_none`           Rust (`voting_active_weight_slots`)
#   the `voting` constraint       Rust (`voting_mode`) — one parse, so the
#                                 shim's own branch reads Rust's answer rather
#                                 than re-comparing the literal
#   `get_feature_names_out`       Rust (`voting_classifier_feature_names` and
#                                 `voting_check_feature_names`)
#   hard predict                  Rust/CubeCL (`voting_hard_predict`) OR numpy
#   soft predict / proba          Rust/CubeCL (`voting_soft_predict`,
#                                 `voting_soft_proba`) OR numpy
#   soft transform (flattened)    Rust/CubeCL (`voting_hstack`) OR numpy
#   ============================  ============================================
#
# ## Why hard voting is the interesting one to move
#
# sklearn's hard route is
# `np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=w)), 1, preds)`
# — a PYTHON-LEVEL loop over `n` rows, allocating a fresh `bincount` array per
# row. That is the one place in either voting estimator where sklearn is not
# already running vectorised numpy, and it is why the hard route's Rust arms win
# by margins the regressor's aggregation never sees. `docs/voting.md` carries
# the ladder.
#
# The soft route, by contrast, is `np.average` over a 3-D stack — the SAME
# reduction the regressor performs, with `n * n_classes` elements per member
# instead of `n` — so it reuses `mlrs_algos::ensemble::voting::weighted_average`
# unchanged and inherits that ladder's conclusions. Its one genuinely new
# opportunity is `predict`, where the device arm fuses the argmax into the
# reduction and never downloads the `(n, C)` average at all.
#
# ## `transform` under hard voting stays in numpy on every arm
#
# It returns the members' LABELS, and labels are integers. The Rust aggregation
# arms are float-typed (they exist to reproduce `np.average` bit for bit), so
# `_vote_via_rust` declines an integer column and numpy answers — deliberately,
# because the alternative is a float round-trip that would change the dtype
# sklearn returns. This is a documented gap in arm coverage, not an oversight:
# see `test_voting_classifier_engine.py`, which asserts the numpy answer is
# what all three arms produce there.


def _vote_labels_via_rust(columns, weights, engine, n_classes):
    """The hard-voting majority from ``_mlrs.voting_hard_predict``, or ``None``.

    ``columns`` is one 1-D ENCODED label column per kept member. Returns
    ``None`` — deliberately, rather than raising — for every input the Rust arms
    cannot represent, leaving numpy to handle it exactly as it did before this
    arm existed:

    * a non-integer column (a member is free to return floats, and
      ``np.bincount`` would raise its own ``TypeError`` for them);
    * a NEGATIVE label, so that ``np.bincount``'s own *"'list' argument must have
      no negative elements"* is what the caller sees rather than a message this
      shim invented;
    * a column that is not 1-D, or whose length disagrees with the others;
    * an empty ``X``.

    ``n_bins`` is one past the largest label PRESENT, not ``len(classes_)``:
    ``np.bincount`` sizes itself from the data, and a member is not actually
    forbidden from returning a label outside the fitted range. Sizing from the
    data is both what numpy does and the only bound that cannot overflow the
    tally.
    """
    import pyarrow as pa

    arrays = [np.asarray(c) for c in columns]
    if any(a.ndim != 1 for a in arrays):
        return None
    if any(a.dtype.kind not in "iu" for a in arrays):
        return None
    n_rows = int(arrays[0].shape[0])
    if any(int(a.shape[0]) != n_rows for a in arrays):
        return None
    if n_rows == 0:
        return None
    lo = min(int(a.min()) for a in arrays)
    hi = max(int(a.max()) for a in arrays)
    if lo < 0 or hi >= 2**31:
        return None
    # `n_classes` is a FLOOR, not the answer: sklearn's `bincount` never sees
    # fewer bins than the labels need, but a row whose members all voted for
    # class 0 still has to be argmax-able against a tally sized for the whole
    # call. Taking the max of the two keeps every row's own `hi` bound in range.
    n_bins = max(hi + 1, int(n_classes))

    flats, capsules = [], []
    for a in arrays:
        flat = np.ascontiguousarray(a, dtype=np.uint32)
        flats.append(flat)
        capsules.append(
            pa.Array.from_buffers(pa.uint32(), flat.size, [None, pa.py_buffer(flat)])
        )

    out = _stack_ext().voting_hard_predict(
        capsules,
        n_rows,
        n_bins,
        None if weights is None else [float(w) for w in weights],
        engine,
    )
    # `flats` is referenced until here on purpose: `py_buffer` borrows those
    # numpy buffers, and letting one be collected mid-call would free memory
    # Rust is still reading.
    del flats
    return _io.to_output(out, (n_rows,), "numpy", np.uint32).astype(np.intp, copy=False)


def _probability_blocks(blocks, weights):
    """``(flat_arrays, capsules, dtype, n_rows, n_cols)`` for a soft-voting call,
    or ``None`` when numpy has to take it.

    The dtype handed over is ``np.result_type`` of the blocks AND the weights,
    which is the promotion ``np.average`` would have applied — so the ``host``
    arm is bit-identical to numpy rather than merely close.

    Declines, so that numpy answers instead of this raising:

    * a block that is not 2-D, or whose shape disagrees with the others (a
      member whose ``predict_proba`` disagrees on ``n_classes`` is sklearn's
      error to report, from inside ``np.average``);
    * a non-float promotion (an object-dtype block, say);
    * an empty ``X`` or a zero-class problem, both of which are zero-byte device
      allocations numpy already shapes correctly.
    """
    import pyarrow as pa

    arrays = [np.asarray(b) for b in blocks]
    if any(a.ndim != 2 for a in arrays):
        return None
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        return None
    dtypes = [a.dtype for a in arrays]
    if weights is not None:
        dtypes.append(np.asarray(weights).dtype)
    try:
        dtype = np.result_type(*dtypes)
    except TypeError:
        return None
    if dtype != np.float32 and dtype != np.float64:
        return None
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows == 0 or n_cols == 0:
        return None

    arrow_type = pa.float32() if dtype == np.float32 else pa.float64()
    flats, capsules = [], []
    for a in arrays:
        # `ascontiguousarray` is a no-op view when `a` is already contiguous in
        # the promoted dtype, and `py_buffer` does not copy — so a block reaches
        # Rust without a staging copy, and the arms are compared on the work they
        # actually do rather than on ingress overhead.
        flat = np.ascontiguousarray(a, dtype=dtype).reshape(-1)
        flats.append(flat)
        capsules.append(
            pa.Array.from_buffers(arrow_type, flat.size, [None, pa.py_buffer(flat)])
        )
    return flats, capsules, dtype, n_rows, n_cols


def _vote_proba_via_rust(blocks, mode, weights, engine):
    """One of the three soft-voting aggregations from ``_mlrs``, or ``None``.

    ``mode`` is ``"proba"`` (the ``(n, C)`` weighted average), ``"predict"`` (its
    row argmax, FUSED on the device arm so the average never crosses the bus) or
    ``"hstack"`` (the ``(n, k * C)`` flattened transform).

    ``None`` means numpy must take this call; see :func:`_probability_blocks`
    for what is declined and why.
    """
    prepared = _probability_blocks(blocks, None if mode == "hstack" else weights)
    if prepared is None:
        return None
    flats, capsules, dtype, n_rows, n_cols = prepared

    w = None if weights is None else [float(x) for x in weights]
    if mode == "proba":
        flat = _stack_ext().voting_soft_proba(capsules, n_rows, n_cols, w, engine)
        out = _io.to_output(flat, (n_rows, n_cols), "numpy", dtype)
    elif mode == "predict":
        flat = _stack_ext().voting_soft_predict(capsules, n_rows, n_cols, w, engine)
        out = _io.to_output(flat, (n_rows,), "numpy", np.uint32).astype(
            np.intp, copy=False
        )
    else:
        flat, out_cols = _stack_ext().voting_hstack(capsules, n_rows, n_cols, engine)
        out = _io.to_output(flat, (n_rows, out_cols), "numpy", dtype)
    # See `_vote_labels_via_rust`: the borrowed numpy buffers stay alive until
    # the call has returned.
    del flats
    return out


class VotingClassifier(
    ClassifierMixin, TransformerMixin, MetaEstimatorMixin, _VoteComposition, BaseEstimator
):
    """Soft voting / majority rule classifier (VOTE-CLF-01).

    ``VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None,
    flatten_transform=True, verbose=False)`` — the full
    :class:`sklearn.ensemble.VotingClassifier` parameter surface, with the
    composition bookkeeping and both aggregations in Rust.

    Every member is fitted on the whole of ``X`` (there is no cross-validation
    and no meta learner) against a LABEL-ENCODED target, and ``predict`` maps the
    winning index back through ``classes_``.

    Parameters
    ----------
    estimators : list of (str, estimator)
        The classifiers to combine. An entry's estimator may be the string
        ``'drop'`` (usually via ``set_params(name='drop')``) to disable it; a
        dropped entry keeps its slot in ``named_estimators_`` (as ``'drop'``)
        and its slot in ``weights``, but is never fitted and contributes no
        column.
    voting : {'hard', 'soft'}, default='hard'
        ``'hard'`` takes the weighted majority of the members' predicted labels.
        ``'soft'`` averages their ``predict_proba`` outputs and takes the argmax,
        which requires every member to implement ``predict_proba`` — sklearn
        does not check that up front, and neither does mlrs, so a member without
        it raises from ``predict`` rather than from ``fit``.
    weights : array-like of shape (n_estimators,), default=None
        Per-member weights. ``None`` is uniform. Indexed against the FULL
        ``estimators`` list — a dropped entry still needs its slot, and its
        weight is discarded afterwards, so ``set_params(name='drop')`` does not
        require rewriting ``weights``.
    n_jobs : int, default=None
        joblib parallelism for the member fits. ``None`` means 1. Reduced to
        serial (with a warning) when a member is an mlrs estimator — see
        :func:`_effective_n_jobs`.
    flatten_transform : bool, default=True
        Only consulted under ``voting='soft'``. ``True`` returns
        ``(n_samples, n_classifiers * n_classes)`` from ``transform``; ``False``
        returns the raw ``(n_classifiers, n_samples, n_classes)`` stack, which
        has no column names and so makes ``get_feature_names_out`` raise.
    verbose : bool, default=False
        Print each member's fit time as it completes.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The class labels, in ``LabelEncoder`` order.
    le_ : :class:`sklearn.preprocessing.LabelEncoder`
        The fitted encoder. Public, and named as sklearn names it, because
        sklearn's own ``predict`` reads it back.
    estimators_ : list of estimator
        The fitted members, dropped entries excluded.
    named_estimators_ : :class:`sklearn.utils.Bunch`
        ``name -> fitted estimator`` (or the string ``'drop'``).
    n_features_in_ : int
        Feature count of ``X``, read off ``estimators_[0]``.
    feature_names_in_ : ndarray of str
        Present only when a fitted member exposes it.

    Notes
    -----
    Only binary and multiclass targets are supported, which is sklearn's own
    limit: a continuous target is a ``ValueError`` and a multilabel one is a
    ``NotImplementedError``, and mlrs reproduces both including the choice of
    exception class.

    ``allow_nan`` and ``sparse`` are DERIVED from the members (the AND over
    them), because this estimator never touches ``X`` itself.

    The aggregations run on the arm ``MLRS_VOTING_ENGINE`` names — ``numpy``
    (the default), ``host`` (Rust), or ``device`` (CubeCL). Hard voting agrees
    EXACTLY across all three (its tally is an f64 bin sum, with no
    multiply-accumulate for a GPU to contract); soft voting's ``device`` arm
    agrees to within a few ULP, exactly as the regressor's does. See
    ``docs/voting.md``.

    Examples
    --------
    >>> import numpy as np
    >>> import mlrs
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 5)).astype(np.float32)
    >>> y = (X[:, 0] > 0).astype(np.int64)
    >>> clf = mlrs.VotingClassifier(
    ...     estimators=[("nb", mlrs.GaussianNB()), ("knn", mlrs.KNeighborsClassifier())],
    ...     voting="soft",
    ... )
    >>> clf.fit(X, y).predict(X[:2]).shape
    (2,)
    """

    def __init__(
        self,
        estimators,
        *,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
    ):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.verbose = verbose

    # -- validation --------------------------------------------------------- #

    def _validate_params(self):
        """sklearn's ``_parameter_constraints`` for this class, reproduced with
        its wording.

        sklearn applies these through ``@_fit_context``, which runs BEFORE
        ``fit``'s body — so a non-list ``estimators`` is an
        ``InvalidParameterError`` about the TYPE, not the structural message the
        composition check would otherwise reach first, and a bad ``voting`` is
        reported ahead of anything about ``y``.

        The ``voting`` rule is the one that crosses into Rust: it renders a
        ``StrOptions`` set, whose ORDER in sklearn's message is
        ``PYTHONHASHSEED``-dependent, so the text lives in one place
        (``mlrs_algos::ensemble::voting::voting_mode``) and the oracle test
        compares the option sets rather than the raw strings.
        """

        def fail(name, expected, value):
            raise InvalidParameterError(
                f"The {name!r} parameter of {type(self).__name__} must be "
                f"{expected}. Got {value!r} instead."
            )

        if not isinstance(self.estimators, list):
            fail("estimators", "an instance of 'list'", self.estimators)
        self._check_voting()
        if self.weights is not None and not _is_array_like_not_scalar(self.weights):
            fail("weights", "an array-like or None", self.weights)
        if self.n_jobs is not None and not isinstance(self.n_jobs, _Integral):
            fail("n_jobs", "None or an instance of 'int'", self.n_jobs)
        if not isinstance(self.flatten_transform, (bool, np.bool_)):
            fail(
                "flatten_transform",
                "an instance of 'bool' or an instance of 'numpy.bool'",
                self.flatten_transform,
            )
        verbose_ok = isinstance(self.verbose, (bool, np.bool_)) or (
            isinstance(self.verbose, _Integral) and self.verbose >= 0
        )
        if not verbose_ok:
            fail(
                "verbose",
                "an int in the range [0, inf), an instance of 'bool' or an "
                "instance of 'numpy.bool'",
                self.verbose,
            )

    def _check_voting(self):
        """Validate ``voting`` and return the canonical spelling.

        Rust owns the parse AND the message; the shim only re-raises it under
        ``InvalidParameterError``, which is a Python class this crate cannot
        construct. Every branch on ``voting`` in this class goes through here, so
        an unrecognized value can never reach a ``== 'soft'`` comparison that
        would silently mean "hard".
        """
        try:
            return _stack_ext().voting_mode(self.voting)
        except (ValueError, TypeError) as exc:
            if isinstance(exc, TypeError):
                # A non-string `voting` cannot cross the FFI at all; sklearn
                # reports the same constraint for it, with the repr of whatever
                # was passed.
                raise InvalidParameterError(
                    f"The 'voting' parameter of {type(self).__name__} must be a "
                    f"str among {{'hard', 'soft'}}. Got {self.voting!r} instead."
                ) from None
            raise InvalidParameterError(str(exc)) from None

    def _validate_estimators(self):
        """``(names, estimators, kept_indices)`` — the Rust-backed structural check.

        The list rules are :meth:`_HeterogeneousComposition._validate_composition`.
        What voting adds is the type check, which sklearn's
        ``_BaseHeterogeneousEnsemble._validate_estimators`` applies through
        ``is_classifier`` for a ``VotingClassifier`` — a regressor member is
        rejected here, unlike in ``StackingClassifier`` where sklearn
        deliberately allows one.
        """
        names, values, kept = self._validate_composition()
        for i in kept:
            if not is_classifier(values[i]):
                raise ValueError(
                    "The estimator {} should be a classifier.".format(
                        type(values[i]).__name__
                    )
                )
        return names, values, kept

    # -- fit ---------------------------------------------------------------- #

    def fit(self, X, y, **fit_params):
        """Fit every member on the whole of ``X``, against an encoded ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary or multiclass. A continuous target is a ``ValueError``; a
            multilabel or multi-output one is a ``NotImplementedError``.
        **fit_params : dict
            ``sample_weight`` is forwarded to every member's ``fit``. Anything
            else requires ``sklearn.set_config(enable_metadata_routing=True)``.

        Returns
        -------
        self : object
        """
        # FIRST, before anything else touches the arguments: sklearn's
        # `@_fit_context` runs the constraint checks ahead of `fit`'s body, and
        # the order is observable (a bad `voting` is reported before a bad `y`).
        self._validate_params()
        _raise_for_extra_fit_params(fit_params, self, "fit", allow=["sample_weight"])

        # sklearn splits the target rejection across TWO exception classes on
        # purpose, and a caller can see which: an unfittable target (continuous,
        # or something `type_of_target` cannot name) is a `ValueError`, while a
        # target this estimator merely does not implement yet — multilabel,
        # multi-output — is a `NotImplementedError`.
        y_type = type_of_target(y, input_name="y")
        if y_type in ("unknown", "continuous"):
            raise ValueError(
                f"Unknown label type: {y_type}. Maybe you are trying to fit a "
                "classifier, which expects discrete classes on a "
                "regression target with continuous values."
            )
        elif y_type not in ("binary", "multiclass"):
            raise NotImplementedError(
                f"{type(self).__name__} only supports binary or multiclass "
                "classification. Multilabel and multi-output classification are "
                "not supported."
            )

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        names, all_estimators, kept = self._validate_estimators()
        return self._fit_members(
            X, transformed_y, names, all_estimators, kept, fit_params
        )

    def fit_transform(self, X, y=None, **fit_params):
        """``fit(X, y).transform(X)`` — the members' training-row responses."""
        return self.fit(X, y, **fit_params).transform(X)

    # -- aggregation -------------------------------------------------------- #

    def _collect_probas(self, X):
        """Each fitted member's ``predict_proba(X)``, in kept order.

        A list rather than sklearn's ``np.asarray([...])`` stack: the Rust arms
        want the blocks separately (each is one contiguous upload), and the
        numpy fallbacks re-stack it themselves where they need to. A member whose
        block disagrees on ``n_classes`` therefore reaches ``np.average``, which
        is where sklearn reports it too.
        """
        return [np.asarray(clf.predict_proba(X)) for clf in self.estimators_]

    def _engine(self):
        """The arm ``MLRS_VOTING_ENGINE`` names, asked once per call."""
        return _stack_ext().voting_engine()

    def _hard_predict(self, X):
        """``argmax(bincount(row, weights))`` per row — the encoded winner.

        The numpy branch is sklearn's own expression, kept verbatim rather than
        vectorised, because it is also the FALLBACK for every input the Rust
        arms decline (:func:`_vote_labels_via_rust`) and a re-derivation would be
        a second definition of the tie-break and of ``bincount``'s per-row
        length.
        """
        columns = self._member_predictions(X)
        weights = self._weights_for_predict()
        engine = self._engine()
        if engine != "numpy":
            out = _vote_labels_via_rust(columns, weights, engine, len(self.classes_))
            if out is not None:
                return out
        predictions = np.asarray(columns).T
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=weights)),
            axis=1,
            arr=predictions,
        )

    def _soft_aggregate(self, blocks, mode):
        """Run one soft-voting ``mode`` on the arm ``MLRS_VOTING_ENGINE`` names.

        ``numpy`` is the default arm AND the fallback for everything the Rust
        arms cannot represent, so this method always has an answer.

        ``weights`` is resolved only for the two ``predict`` modes. sklearn's
        ``transform`` never reads them, and reading them here would let a
        ``weights`` mutated AFTER the fit — the only way it can be wrong at this
        point — break a ``transform`` that sklearn completes.
        """
        weights = self._weights_for_predict() if mode != "hstack" else None
        engine = self._engine()
        if engine != "numpy":
            out = _vote_proba_via_rust(blocks, mode, weights, engine)
            if out is not None:
                return out
        if mode == "hstack":
            return np.hstack(blocks)
        avg = np.average(np.asarray(blocks), axis=0, weights=weights)
        return avg if mode == "proba" else np.argmax(avg, axis=1)

    # -- introspection ------------------------------------------------------ #

    def get_feature_names_out(self, input_features=None):
        """``votingclassifier_<name>``, plus a class index under soft voting.

        ``voting='hard'`` gives one name per kept member; ``voting='soft'`` gives
        ``n_classes`` per member with the class index appended and NO separator
        (``votingclassifier_lr0``), matching ``np.hstack(probas)``' layout.

        ``voting='soft'`` with ``flatten_transform=False`` names a 3-D output and
        is rejected, which is sklearn's behaviour and Rust's message.
        ``input_features`` is otherwise VALIDATED and then discarded, which is
        sklearn's ``_check_feature_names_in(..., generate_names=False)``.
        """
        check_is_fitted(self, "n_features_in_")
        voting = self._check_voting()
        _stack_ext().voting_check_feature_names(voting, bool(self.flatten_transform))
        if input_features is not None:
            _generate_input_feature_names(self, input_features)
        names = _stack_ext().voting_classifier_feature_names(
            type(self).__name__.lower(),
            self._kept_names(),
            voting,
            len(self.classes_),
        )
        return np.asarray(names, dtype=object)

    def __sklearn_tags__(self):
        """The members' tags, plus sklearn's own ``preserves_dtype = []``.

        ``transform`` returns labels under hard voting and probabilities under
        soft, neither of which preserves the input dtype, so sklearn clears the
        list outright rather than naming a subset.
        """
        tags = super().__sklearn_tags__()
        tags.transformer_tags.preserves_dtype = []
        return tags

    # -- transform / predict ------------------------------------------------ #

    def transform(self, X):
        """The members' responses for ``X``.

        Returns
        -------
        probabilities_or_labels : ndarray
            ``(n_samples, n_classifiers)`` labels under ``voting='hard'``;
            ``(n_samples, n_classifiers * n_classes)`` probabilities under
            ``voting='soft'`` with ``flatten_transform=True``; and the raw
            ``(n_classifiers, n_samples, n_classes)`` stack when
            ``flatten_transform=False``.
        """
        check_is_fitted(self)
        if self._check_voting() == "soft":
            probas = self._collect_probas(X)
            if not self.flatten_transform:
                # sklearn's `_collect_probas` builds the 3-D array itself; this
                # shim keeps the blocks apart for the Rust arms, so the stack is
                # materialised here — same object, same shape.
                return np.asarray(probas)
            return self._soft_aggregate(probas, "hstack")
        # Hard voting's transform is the integer label matrix, which the float
        # aggregation arms decline by design — see this section's header.
        return np.asarray(self._member_predictions(X)).T

    def predict(self, X):
        """Predict class labels for ``X``, mapped back through ``classes_``.

        Returns
        -------
        maj : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        if self._check_voting() == "soft":
            maj = self._soft_aggregate(self._collect_probas(X), "predict")
        else:
            maj = self._hard_predict(X)
        return self.le_.inverse_transform(maj)

    def _voting_is_soft(self):
        """``available_if`` predicate for :meth:`predict_proba`.

        sklearn hides ``predict_proba`` behind ``voting='soft'`` — ``hasattr``
        is ``False`` under hard voting — and words the failure as an
        ``AttributeError`` naming the value that disabled it. Reproduced
        literally, including reading ``self.voting`` rather than the validated
        spelling: the descriptor runs before ``fit`` may ever have validated
        anything, so an unrecognized value must not raise a DIFFERENT error here
        than sklearn's.
        """
        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={self.voting!r}"
            )
        return True

    @available_if(_voting_is_soft)
    def predict_proba(self, X):
        """The weighted average of the members' class probabilities.

        Returns
        -------
        avg : ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        return self._soft_aggregate(self._collect_probas(X), "proba")


def _n_columns(X):
    """``X``'s column count, for any 2-D container the base estimators accept.

    ``.shape`` covers numpy / pandas / polars / scipy-sparse; the ``asarray``
    fallback covers a duck-typed array-like (sklearn's own check harness passes
    one) and a plain list of rows.
    """
    shape = getattr(X, "shape", None)
    if shape is not None and len(shape) == 2:
        return int(shape[1])
    return int(np.asarray(X).shape[1])


def _is_drop(estimator, drop):
    """``estimator == 'drop'``, guarded against an overloaded ``__eq__``.

    A numpy-array-valued (or otherwise broadcasting) ``__eq__`` would return an
    array here rather than a bool; sklearn's bare ``est != "drop"`` would then
    raise. Comparing identity/str first keeps the common cases exact.
    """
    if isinstance(estimator, str):
        return estimator == drop
    return False
