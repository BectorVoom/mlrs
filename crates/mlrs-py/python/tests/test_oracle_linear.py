"""Linear-model oracle harness (PY-01: full numpy->pyarrow->FFI->device path).

Wave-0 COLLECTING stub: parametrizes over the committed linear fixtures and
``importorskip("mlrs")`` so it collects green before the wrappers land
(Plans 04-06 convert the body to real 1e-5 assertions against the fixture
``coef`` / ``intercept`` / ``predict_proba`` keys written by gen_oracle.py).
"""

import pytest

from conftest import fixture_path, load_fixture  # noqa: F401  (Plan 04 uses)

# Req: PY-01 (1e-5 oracle parity through the Python binding path).
LINEAR_FIXTURES = [
    "linear_regression_f32_seed42",
    "linear_regression_f64_seed42",
    "ridge_f32_seed42",
    "ridge_f64_seed42",
    "lasso_f32_seed42",
    "lasso_f64_seed42",
    "elastic_net_f32_seed42",
    "elastic_net_f64_seed42",
    "logistic_binary_f32_seed42",
    "logistic_binary_f64_seed42",
    "logistic_multi_f32_seed42",
    "logistic_multi_f64_seed42",
]


@pytest.mark.parametrize("fixture", LINEAR_FIXTURES)
def test_linear_oracle(fixture):
    """PY-01: linear estimators match the sklearn oracle within 1e-5."""
    mlrs = pytest.importorskip("mlrs")  # noqa: F841  (wired in Plan 04)
    pytest.xfail("linear wrappers land in Plan 03/04")
