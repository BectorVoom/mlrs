# Testing Patterns

**Analysis Date:** 2026-06-11

---

## Python Test Framework

**Runner:** pytest (pinned as `pytest` in `python/cuml/pyproject.toml [project.optional-dependencies] test`)
**Config:** `python/cuml/pyproject.toml [tool.pytest.ini_options]`

**Key plugins:**
- `pytest-xdist` — parallel execution (`--numprocesses`, `--dist=worksteal`)
- `pytest-benchmark` — microbenchmarks
- `pytest-cases` — case-based parameterisation
- `pytest-cov` — coverage
- `hypothesis` — property-based testing

**Run commands:**
```bash
# All single-GPU tests (skip dask)
cd python/cuml/tests && python -m pytest --cache-clear --ignore=dask .

# CI single-GPU (8 parallel workers, JUnit output)
./ci/run_cuml_singlegpu_pytests.sh --numprocesses=8 --dist=worksteal \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml"

# Dask / multi-GPU tests
./ci/run_cuml_dask_pytests.sh

# Accelerator tests
./ci/run_cuml_singlegpu_accel_pytests.sh

# Integration tests
./ci/run_cuml_integration_pytests.sh

# Upstream scikit-learn compatibility tests
./ci/test_python_scikit_learn_tests.sh
```

Default pytest options from config:
```
addopts = "--tb=native --import-mode=append -r fExX"
```

---

## Python Test Locations

```
python/cuml/tests/                 # single-GPU tests (main suite)
  conftest.py                      # root conftest with fixtures + hooks
  test_<algorithm>.py              # per-algorithm test modules
  dask/                            # multi-GPU / Dask tests
  explainer/                       # SHAP explainer tests
  stemmer_tests/                   # NLP stemmer tests
  ts_datasets/                     # time-series dataset helpers
python/cuml/cuml/testing/          # shared testing utilities
  utils.py                         # array_equal, unit_param, etc.
  datasets.py                      # dataset generators
  strategies.py                    # hypothesis strategies
  plugins/quick_run_plugin.py      # pytest plugin auto-loaded by conftest
  dask/utils.py                    # Dask-specific test helpers
```

`testpaths` in `pyproject.toml`:
```
cuml/tests
cuml/tests/dask
cuml/tests/experimental
cuml/tests/explainer
cuml/tests/stemmer_tests
```

**File naming:** `test_<subject>.py` (all lowercase, underscore-separated).

---

## Custom pytest Markers

Defined in `python/cuml/pyproject.toml [tool.pytest.ini_options] markers`:

| Marker | Meaning | CLI Flag |
|---|---|---|
| `unit` | Quickest correctness tests | `--run_unit` (default if no flag set) |
| `quality` | Intermediate intensity | `--run_quality` |
| `stress` | Long-running hardware stress | `--run_stress` |
| `mg` | Multi-GPU / Dask tests | (separate CI job) |
| `memleak` | Memory leak detection | `--run_memleak` |
| `ucx` | UCXX Dask transport tests | (separate CI job) |

`pytest_collection_modifyitems` in `conftest.py` skips markers not matching the
active flag. If no `--run_*` flag is passed, `unit` tests run by default.

---

## Test Tier Parameterisation

Helper functions in `python/cuml/cuml/testing/utils.py`:

```python
def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)

def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)

def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)
```

Usage (from `python/cuml/tests/test_dbscan.py`):
```python
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "ncols", [unit_param(20), quality_param(100), stress_param(1000)]
)
def test_dbscan(datatype, nrows, ncols, ...):
    ...
```

This embeds the tier marker directly in the parameter value, so each
`(nrows, ncols)` combination is independently filterable by tier.

---

## Test Structure

### Typical Python Test Function

```python
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
def test_<algo>(datatype, nrows, ...):
    # 1. Generate data (sklearn or cuml generator)
    X, y = make_blobs(n_samples=nrows, n_features=ncols, random_state=0)
    X = X.astype(datatype)

    # 2. Fit cuML estimator
    cu_model = cuMLEstimator(...)
    cu_result = cu_model.fit_predict(X)

    # 3. Fit sklearn reference (for small data only)
    if nrows < 500000:
        sk_model = sklearnEstimator(...)
        sk_result = sk_model.fit_predict(X)

        # 4. Compare with tolerance-aware helpers
        assert array_equal(cu_result, sk_result)
```

### Comparison Utilities (`python/cuml/cuml/testing/utils.py`)

| Utility | Description |
|---|---|
| `array_equal(a, b, unit_tol=1e-4, total_tol=1e-4)` | Fuzzy array comparison — passes if `< total_tol` fraction of elements differ by `> unit_tol` |
| `array_difference(a, b, with_sign=True)` | Returns summed absolute difference |
| `assert_dbscan_equal(sk_labels, cu_labels, X, core_indices, eps)` | Cluster-label-order-independent equality check |

Both NumPy and GPU arrays (CuPy, CuDF, numba DeviceNDArray) are automatically
converted to NumPy before comparison via `to_nparray()`.

---

## Fixtures (`python/cuml/tests/conftest.py`)

| Fixture | Scope | Description |
|---|---|---|
| `random_seed` | `session` | Reads/generates `PYTEST_RANDOM_SEED` env var; prints seed on first use for reproducibility |
| `failure_logger` | `function` | Logs the random seed when a test fails (attach to tests that use random data) |
| `sparse_text_dataset` | `session` | Sparse text classification dataset (20-newsgroups style) |
| `supervised_learning_dataset` | `session` | Parameterised over `digits`, `diabetes`, `cancer` sklearn datasets |

CUDA JIT cache is per-xdist worker: `conftest.py` sets `CUDA_CACHE_PATH` to a
worker-specific subdirectory before any CUDA initialisation.

Plugin auto-loaded for every session:
```python
pytest_plugins = "cuml.testing.plugins.quick_run_plugin"
```
(file: `python/cuml/cuml/testing/plugins/quick_run_plugin.py`)

---

## Warning Filters

`filterwarnings` in `python/cuml/pyproject.toml` promotes `FutureWarning` and
`DeprecationWarning` to errors globally, with targeted `ignore:` exemptions for
known third-party noise (sklearn 1.6 tag transition, hdbscan, umap-learn,
numba, dask). Test code is exempted from the `no-deprecationwarning` pre-commit
hook, but must still pass the warning filter unless an explicit `ignore` entry
is added.

---

## Hypothesis (Property-Based Tests)

Config: `python/cuml/tests/conftest.py`

```python
hypothesis.settings.register_profile("unit",    max_examples=20,  ...)
hypothesis.settings.register_profile("quality", max_examples=100, ...)
hypothesis.settings.register_profile("stress",  max_examples=200, ...)
```

Hypothesis tests are **disabled by default** in CI. Enabled only on nightly
builds via:
```bash
export HYPOTHESIS_ENABLED="true"
```

**Enforcement:** `pytest_collection_modifyitems` raises `pytest.UsageError` if
any `@given`-decorated test lacks at least one `@example` case, ensuring every
property test has a deterministic baseline that runs even when hypothesis is
disabled.

Strategies are defined in `python/cuml/cuml/testing/strategies.py`.

---

## GPU / Memory Adaptation

Tests can skip or reduce data sizes based on GPU memory:

```python
max_gpu_memory = pytest.max_gpu_memory or 4  # GB
if nrows == 500000 and max_gpu_memory < 32:
    if pytest.adapt_stress_test:
        nrows = nrows * max_gpu_memory // 32
    else:
        pytest.skip("Insufficient GPU memory for this test. "
                    "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'")
```

Environment variables:
- `CUML_ADAPT_STRESS_TESTS=True` — scale stress data sizes to available GPU memory
- `PYTEST_RANDOM_SEED=<int>` — fix random seed for reproducibility
- `HYPOTHESIS_ENABLED=true` — enable full hypothesis generation in CI
- `CI=true` or `CI=1` — CI mode (suppresses all hypothesis HealthChecks)

---

## C++ Test Framework (GoogleTest + CTest)

**Framework:** GoogleTest (`<gtest/gtest.h>`)
**Runner:** CTest (`ctest --output-on-failure --no-tests=error`)

**Config:** `cpp/tests/CMakeLists.txt`

**Run commands:**
```bash
# From installed test location (CI/conda)
cd "${INSTALL_PREFIX:-${CONDA_PREFIX}}/bin/gtests/libcuml/"
ctest --output-on-failure --no-tests=error

# CI wrapper
./ci/run_ctests.sh

# Full C++ test job
./ci/test_cpp.sh
```

Installed location: `${INSTALL_PREFIX}/bin/gtests/libcuml/`
Build location (devcontainers): `cpp/build/latest/`

---

## C++ Test Locations

```
cpp/tests/
  sg/          # single-GPU algorithm tests (.cu files)
  mg/          # multi-GPU algorithm tests (.cu files)
  prims/       # primitive / utility tests (.cu files)
    test_utils.h    # shared test assertion helpers
```

**File naming:** `<algorithm>_test.cu` or `<algorithm>.cu`.
`.cu` extension used for all CUDA-containing tests; `.cpp` for CPU-only
(e.g., `cpp/tests/sg/genetic/node_test.cpp`).

---

## C++ Test Patterns

### Class Hierarchy

```cpp
// 1. Define input struct
template <typename T, typename IdxT>
struct DbscanInputs {
  IdxT n_row, n_col, n_centers;
  T cluster_std, eps;
  int min_pts;
  // ...
};

// 2. Test fixture inherits TestWithParam
template <typename T, typename IdxT>
class DbscanTest : public ::testing::TestWithParam<DbscanInputs<T, IdxT>> {
 protected:
  void basicTest() { /* test logic */ }
  void SetUp() override { basicTest(); }
};

// 3. Typed aliases
typedef DbscanTest<float, int> DbscanTestF_Int;

// 4. Declare the test
TEST_P(DbscanTestF_Int, Result) { ASSERT_TRUE(score == 1.0); }

// 5. Instantiate with value table
INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestF_Int, ::testing::ValuesIn(inputs_f_int));
```

Files: `cpp/tests/sg/dbscan_test.cu`, `cpp/tests/sg/pca_test.cu`

All tests use `raft::handle_t` for GPU resource management. Streams are
retrieved via `handle.get_stream()`.

### Device Assertions (`cpp/tests/prims/test_utils.h`)

| Helper | Description |
|---|---|
| `devArrMatch(expected, actual, len, compare_op)` | Element-wise device array comparison |
| `devArrMatchHost(expected_host_vec, actual_device, len, compare_op)` | Host reference vs device actual |

Standard GoogleTest macros used: `ASSERT_TRUE`, `EXPECT_TRUE`, `ASSERT_NEAR`,
`EXPECT_NEAR`, `ASSERT_EQ`, `EXPECT_EQ`.

---

## C++ Benchmarks (Google Benchmark)

**Framework:** Google Benchmark (`<benchmark/benchmark.h>`)
**Location:** `cpp/bench/sg/`

Base fixture: `ML::Bench::Fixture` in `cpp/bench/sg/benchmark.cuh`
Inherits from `MLCommon::Bench::Fixture` in `cpp/bench/common/ml_benchmark.hpp`

Virtual interface:
```cpp
virtual void runBenchmark(::benchmark::State& state) = 0;
virtual void allocateData(const ::benchmark::State& state) {}
virtual void deallocateData(const ::benchmark::State& state) {}
virtual void allocateTempBuffers(const ::benchmark::State& state) {}
virtual void deallocateTempBuffers(const ::benchmark::State& state) {}
virtual void generateMetrics(::benchmark::State& state) {}
```

`SetUp` creates a `raft::handle_t` with a `rmm::cuda_stream_pool`.
`TearDown` resets the handle.

Benchmark files: `cpp/bench/sg/dbscan.cu`, `cpp/bench/sg/kmeans.cu`,
`cpp/bench/sg/svc.cu`, `cpp/bench/sg/rf_classifier.cu`, etc.

---

## Python Benchmarks

Python-level benchmarks use `pytest-benchmark` and custom `BenchmarkRunner`
classes in `python/cuml/cuml/benchmark/`.

Test file: `python/cuml/tests/test_benchmark.py` (marked with `pytestmark =
pytest.mark.skip` — benchmarks not run in the standard test suite).

---

## Coverage

**Tool:** `pytest-cov` (in test dependencies)
**Codecov:** `codecov.yml` at repo root

```yaml
coverage:
  status:
    project: off
    patch: off
comment: false
codecov:
  allow_coverage_offsets: true
```

Coverage status checks are **disabled** (`project: off`, `patch: off`).
Coverage is collected but does not gate PRs.

To generate locally:
```bash
python -m pytest --cov=cuml --cov-report=html .
```

---

## CI Test Jobs

Scripts in `ci/`:

| Script | What it runs |
|---|---|
| `ci/test_cpp.sh` | C++ GoogleTest via CTest |
| `ci/test_python_singlegpu.sh` | Single-GPU pytest + accelerator pytest (8 workers, 1h timeout) |
| `ci/test_python_dask.sh` | Multi-GPU Dask pytest |
| `ci/test_python_integration.sh` | Integration tests |
| `ci/test_python_scikit_learn_tests.sh` | Upstream sklearn compatibility |
| `ci/test_python_cuml_accel_upstream.sh` | cuML acceleration upstream tests |
| `ci/test_notebooks.sh` | Notebook smoke tests |
| `ci/check_style.sh` | `pre-commit run --all-files` |

All Python test jobs share setup from `ci/test_python_common.sh`:
- Creates fresh conda environment from `dependencies.yaml`
- Enables `HYPOTHESIS_ENABLED=true` on nightly builds
- Sets `RAPIDS_TESTS_DIR` for JUnit XML output
- Calls `nvidia-smi` to verify GPU availability
- On aarch64: preloads `libgomp.so.1` to avoid static TLS allocation failures

---

*Testing analysis: 2026-06-11*
