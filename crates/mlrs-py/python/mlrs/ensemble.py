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
