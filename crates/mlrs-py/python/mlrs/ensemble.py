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

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from . import _io
from .base import MlrsBase


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
        output_type="input",
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.l2_regularization = l2_regularization
        self.min_samples_leaf = min_samples_leaf
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
        output_type="input",
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.l2_regularization = l2_regularization
        self.min_samples_leaf = min_samples_leaf
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
