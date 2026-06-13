"""sklearn estimator_checks triage (PY-01/PY-02: sklearn-compatibility).

Wave-0 COLLECTING stub: enumerates the 12 estimator classes and
``importorskip("mlrs")`` so it collects green pre-wrapper. Plan 06 converts this
to a ``parametrize_with_checks([...])`` run that triages the *relevant* check
subset per family (cloneable / get_params invariance / set_params /
unfitted-raises / fit-returns-self / no-attributes-set-in-init /
parameters-default-constructible, plus the regressor/classifier/clusterer/
transformer/neighbors groups), marking the by-design-unsupported checks
(sparse / array-api / NaN) skipped-with-reason. See RESEARCH 06
§sklearn estimator_checks.
"""

import pytest

# Req: PY-01 (sklearn-compatible estimators) + PY-02 (get_params/set_params).
ESTIMATOR_NAMES = [
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "KMeans",
    "DBSCAN",
    "PCA",
    "TruncatedSVD",
    "NearestNeighbors",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
]


@pytest.mark.parametrize("estimator_name", ESTIMATOR_NAMES)
def test_estimator_checks(estimator_name):
    """PY-01/PY-02: each estimator passes the relevant sklearn checks."""
    mlrs = pytest.importorskip("mlrs")  # noqa: F841  (wired in Plan 06)
    pytest.xfail("estimator_checks triage lands in Plan 06")
