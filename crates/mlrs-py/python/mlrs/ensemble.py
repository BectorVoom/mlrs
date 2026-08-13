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
import warnings as _warnings
from contextlib import suppress as _suppress
from copy import deepcopy as _deepcopy

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
    is_regressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils import Bunch, get_tags
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
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


def _fit_one(estimator, X, y, fit_params):
    """sklearn ``_fit_single_estimator``: fit one composed estimator.

    With routing off, a ``sample_weight`` in ``fit_params`` is passed
    positionally-by-name and a ``TypeError`` from an estimator that does not
    accept it is re-raised as sklearn's clearer message.
    """
    if not _routing_enabled() and "sample_weight" in fit_params:
        try:
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
        estimator.fit(X, y, **fit_params)
    return estimator


def _effective_n_jobs(n_jobs, members):
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
        "StackingRegressor: n_jobs is ignored because at least one composed "
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


class StackingRegressor(
    RegressorMixin, TransformerMixin, MetaEstimatorMixin, BaseEstimator
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

    def _validate_estimators(self):
        """``(names, estimators, kept_indices)`` — the Rust-backed structural check.

        Raises sklearn's own ``ValueError`` texts for a malformed list, a
        duplicate/colliding/``__``-containing name, an all-``'drop'`` list, and a
        non-regressor entry.
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

        for i in kept:
            if not is_regressor(values[i]):
                raise ValueError(
                    "The estimator {} should be a regressor.".format(
                        type(values[i]).__name__
                    )
                )
        return list(names), list(values), list(kept)

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

    def _cv_is_prefit(self):
        """Is ``cv`` the ``"prefit"`` string? Classified in Rust.

        A non-string ``cv`` never reaches Rust — it stays here and goes to
        :func:`mlrs.model_selection.check_cv`. A string that is not ``"prefit"``
        raises sklearn's ``StrOptions`` message, re-raised as
        ``InvalidParameterError`` so that both ``except ValueError`` and
        ``except TypeError`` callers migrating from sklearn still catch it.
        """
        if not isinstance(self.cv, str):
            return False
        try:
            return _stack_ext().stacking_cv_is_prefit(self.cv)
        except ValueError as exc:
            raise InvalidParameterError(str(exc)) from None

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

        The column LAYOUT (widths, offsets, total width, and the
        ``_n_feature_outs`` that ``get_feature_names_out`` reads) is computed in
        Rust; the copy itself is ``np.hstack`` — see this module's
        ``StackingRegressor`` section.
        """
        blocks = []
        for pred in predictions:
            pred = np.asarray(pred)
            blocks.append(pred.reshape(-1, 1) if pred.ndim == 1 else pred)

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
            blocks.append(X)
            if _is_sparse(X):
                return _sys.modules["scipy.sparse"].hstack(blocks, format=X.format)

        # STACK-META-01: `np.hstack` is the default arm and the fallback for
        # everything the Rust arms cannot represent (see `_meta_via_rust`).
        engine = _stack_ext().stacking_meta_engine()
        if engine != "numpy":
            meta = _meta_via_rust(blocks, n_feature_outs, n_features, self.passthrough, engine)
            if meta is not None:
                return meta
        return np.hstack(blocks)

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
        """Meta-feature names: ``stackingregressor_<name>`` per kept estimator.

        Under ``passthrough=True`` the input feature names follow. Generated in
        Rust so the naming rule (including the separator-less index suffix on a
        multi-column block) has one definition.
        """
        check_is_fitted(self, "n_features_in_")
        kept_names = [
            name
            for name, est in self.estimators
            if not _is_drop(est, _stack_ext().stacking_drop_sentinel())
        ]
        if self.passthrough:
            inputs = [
                str(f)
                for f in _generate_input_feature_names(self, input_features)
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

    def __sklearn_tags__(self):
        """``allow_nan`` / ``sparse`` are the AND over the composed estimators.

        A stack is only as permissive as its least permissive member, so — as in
        sklearn — these are derived rather than declared. Stacking mlrs
        estimators therefore reports ``sparse=False`` (they ingest dense Arrow),
        while stacking sklearn estimators can report ``True``.
        """
        tags = super().__sklearn_tags__()
        try:
            drop = _stack_ext().stacking_drop_sentinel()
            tags.input_tags.allow_nan = all(
                True
                if _is_drop(est, drop)
                else get_tags(est).input_tags.allow_nan
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
