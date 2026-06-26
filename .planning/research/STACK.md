# Stack Research

**Domain:** v4.0 feature additions to mlrs (cuML-in-Rust) — tree ensembles (RF→FIL→TreeSHAP), ARIMA/AutoARIMA, model-agnostic SHAP, `cuml.accel` drop-in, sklearn-utility surface, symbolic regression
**Researched:** 2026-06-26
**Confidence:** HIGH

## TL;DR (read this first)

**The headline finding: v4.0 needs ZERO new compute/algorithm Rust crates, and ZERO new Rust crates of any kind for the algorithm work.** Every new feature is assembled from the already-validated primitive stack (GEMM, reductions, distance, SVD/eig, Cholesky, top-k, L-BFGS, coordinate descent, RNG, KNN-graph, two-pass SGD) plus plain Rust data structures. This continues the v2/v3 track record (both added **zero** compute dependencies).

The only *additive* dependencies are:
1. **Python dev/test oracle packages** (test-only, never shipped in wheels): `shap`, `statsmodels`, `gplearn`, and additional `scikit-learn` sub-modules already covered by the existing `scikit-learn>=1.6` pin.
2. **Optionally** one Rust serialization crate (`serde` + `serde_json` or `bincode`) **only if** tree-model / ARIMA-model Python round-trip (pickle-equivalent persistence) is made an explicit requirement. Defer until a requirement demands it.

`cuml.accel` is **pure Python, no Rust** — it is an `importlib` meta-path import hook. mlrs already has the prerequisite (the v3 pure-Python sklearn shim with proxy estimator classes); the accel layer is the import-machinery wrapper around them.

---

## Recommended Stack

### Core Technologies (NEW for v4.0)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **(none — Rust compute)** | — | Tree construction, Kalman filter, evolutionary search, SHAP values | All build on the existing CubeCL primitive stack + plain Rust structs. Adding a compute crate would break the primitive-first / GATHER-idiom discipline that has held since v1. See "What NOT to Use". |
| `scikit-learn` (test oracle) | `>=1.6` (already pinned) | Oracle for RandomForest, ARIMA stationarity/seasonality tests, metrics, preprocessing, feature_extraction, model_selection | Already the project's master oracle. `sklearn.ensemble.RandomForest*`, `sklearn.metrics`, `sklearn.preprocessing`, `sklearn.feature_extraction.text`, `sklearn.model_selection` are all in the existing install — **no version bump needed**. |
| `shap` (test oracle) | `0.52.0` (latest, 2026-05-28; requires Py≥3.12) | Reference values for Kernel SHAP, Permutation SHAP, TreeSHAP | The canonical SHAP reference. cuML itself gates its explainers against `shap` (`explainer/base.pyx` imports `shap.Explanation`). Pin `shap>=0.46,<0.53` to stay current but avoid surprise breakage. |
| `statsmodels` (test oracle) | `0.14.4` stable (0.15 is dev) | Reference for ARIMA log-likelihood / forecasts and AutoARIMA stationarity (KPSS) + seasonality tests | cuML's own ARIMA tests use `statsmodels` (it is the listed test dep, **not** `pmdarima` — confirmed `pmdarima` appears nowhere in cuml-main). `statsmodels.tsa.arima.model.ARIMA` is the de-facto CPU reference. |
| `gplearn` (test oracle) | `0.4.3` (latest stable; 0.5 unreleased) | Reference for symbolic/genetic regression (`SymbolicRegressor`/`SymbolicTransformer`) | The PROJECT.md-specified oracle for the genetic subsystem. Built on scikit-learn's estimator API, so it slots into the existing oracle harness. Note: stochastic — gate structurally/property-wise (à la UMAP D-12), not element-wise 1e-5. |

### Supporting Libraries (Rust — EXISTING, reused; nothing new)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `cubecl` | `0.10.0` (workspace pin) | Device kernels for tree histogram/split, batched Kalman GEMM, batched L-BFGS | All v4.0 device kernels. **Stays at 0.10.0** — no bump. Histogram/split MUST use the single-owner GATHER idiom (no SharedMemory, no cross-unit atomics) per the cpu-MLIR constraint — this is the spike's make-or-break question, a kernel-authoring problem, not a dependency problem. |
| `mlrs-backend` L-BFGS prim | in-tree | ARIMA parameter optimization | cuML's `batched_lbfgs.py` is just `scipy._lbfgsb` in a per-series loop. mlrs already ships an L-BFGS prim (`mlrs-backend/src/prims/lbfgs.rs`); **batched ARIMA = extend it to a batch dimension**, not a new crate. |
| `mlrs-backend` GEMM / reductions / Cholesky | in-tree | Kalman filter recursions (state predict/update are matrix mul + rank updates), covariance | The Kalman filter is dense linear algebra mlrs already has. No filtering crate needed. |
| `bytemuck` | `1` | Tree-node array `Pod` bridging host↔device | Tree nodes stored as struct-of-arrays of `Pod` scalars (feature idx, threshold, left/right, value) — the existing host↔device bridge covers it. |
| `thiserror` / `anyhow` | `2` / `1` | Typed errors in new estimator modules / boundary errors | Same convention as all prior phases. |
| `pyo3` | `0.28` (**PINNED — do not bump**) | `#[pyclass]` wrappers for new estimators | v2/v3 added zero binding infrastructure across the 32-estimator surface; the `any_estimator!` machinery generalizes. RF/ARIMA/SHAP/symbolic-regression wrappers reuse it. Bumping to 0.29 double-links the PyInit ABI and crashes the wheel (D-09). |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `maturin` + `pyarrow` venv | Live FFI / accel import-hook validation | The `cuml.accel`-equivalent import hook is pure Python and can only be exercised through a real interpreter. Per memory ("Python wheel untestable in env" → now RUNNABLE when PyPI reachable), bootstrap a venv to validate the meta-path hook end-to-end. |
| Pinned-seed oracle fixtures (`.npz`) | Deterministic RF/ARIMA/symbolic references | RF and symbolic regression are stochastic; pin `random_state` in the sklearn/gplearn oracle generators (same pattern as the pinned deterministic SGD oracle in v2). Fixtures are committed blobs; regen needs the numpy venv (memory: "oracle fixture regen needs venv"). |

## Installation

```bash
# Rust side: NO new crate additions for v4.0 algorithm work.
# (Optional, ONLY if model-persistence becomes a requirement:)
#   cargo add serde --features derive
#   cargo add serde_json          # or: cargo add bincode

# Python oracle / test deps (test-only; NEVER in the shipped wheels):
pip install 'shap>=0.46,<0.53'        # 0.52.0 latest; Kernel/Permutation/Tree SHAP reference
pip install 'statsmodels>=0.14,<0.15' # 0.14.4 stable; ARIMA / KPSS / seasonality reference
pip install 'gplearn==0.4.3'          # symbolic-regression reference (pin exact; infrequent releases)
# scikit-learn>=1.6 already installed — covers ensemble/metrics/preprocessing/
# feature_extraction/model_selection oracles with NO version change.
```

## cuml.accel — mechanism analysis (pure Python, no Rust)

`cuml.accel` is a **module-proxying import hook**. An mlrs equivalent reproduces this machinery in pure Python on top of the existing v3 sklearn-shim estimators. Key components (read from `cuml-main/python/cuml/cuml/accel/`):

| cuML component | Mechanism | mlrs equivalent needs |
|----------------|-----------|------------------------|
| `AccelModule(types.ModuleType)` (`accelerator.py`) | A `ModuleType` subclass whose `__getattr__` returns the accelerated class for overridden names, else delegates to the wrapped real module | Same wrapper class. Trivial pure-Python. |
| `AccelFinder(importlib.abc.MetaPathFinder)` + `AccelLoader(importlib.abc.Loader)` | Inserted at `sys.meta_path[0]`; intercepts future imports of registered modules (`sklearn.*`, `umap`, `hdbscan`) and swaps in the accelerated module | Same `importlib.abc` machinery — standard library only. |
| `install()` | `sys.meta_path.insert(0, AccelFinder(self))` for future imports + a best-effort `sys.modules[name] = wrapped` pass over already-imported modules (rewriting the parent-module attribute too) | Same install routine. |
| Stack-walk `exclude` (`__getattr__`) | Walks `sys._getframe()` up past `importlib` frames to find the *calling* module, and returns the un-accelerated class if the caller is itself inside `sklearn`/`umap`/`hdbscan` (so their internals/tests still work) | Same caller-exclusion logic — important for letting upstream test suites run under accel. |
| `_OVERRIDES` registry | A set mapping `sklearn.linear_model`, `sklearn.cluster`, `umap`, `hdbscan`, … → override namespaces of GPU estimator classes | mlrs maps the same module names → the **mlrs estimator classes that already exist** (32-estimator surface + new v4 RF). The accel layer is glue, not new estimators. |
| `_PATCHES` (`sklearn.pipeline`, `sklearn.compose`, `sklearn.utils`) + `ProxyBase` with CPU fallback | Patches that mutate the real module; proxy raises `UnsupportedOnGPU` → falls back to the original sklearn estimator | mlrs proxies fall back to real sklearn when a config is unsupported. Reuse the v3 shim's parameter mapping. |

**mlrs accel requirements: pure-Python package only.** Standard-library `importlib.abc` + `sys.meta_path` + `sys.modules`. No new Rust, no new PyPI dependency. The hard part already exists (v3 sklearn shim provides `get_params`/`set_params`/`clone`-compatible proxy classes for all 32 estimators).

## Rust-side analysis: what each feature actually needs

| Feature | New compute crate? | Built from |
|---------|-------------------|------------|
| **RF tree construction** | **No** | GPU histogram/split kernel authored in CubeCL under the GATHER idiom (the spike). Tree = struct-of-arrays (`Vec<i32>` feature, `Vec<F>` threshold, `Vec<i32>` left/right, `Vec<F>` value). Plain Rust; no graph/tree crate. |
| **FIL (batched traversal)** | **No** | Read-only traversal kernel over the same node arrays; per-row GATHER walk. cuML uses `treelite` only as an interchange/serialization format for *importing external* models — **mlrs owns its tree format, so treelite is not needed**. |
| **TreeSHAP** | **No** | Path-dependent feature-attribution recursion over mlrs's own tree arrays (the Lundberg algorithm). Host or GATHER-kernel; arithmetic only. |
| **ARIMA / AutoARIMA** | **No** | Kalman filter = GEMM + Cholesky/rank-update recursions (have them). Parameter fit = **batched** extension of the existing L-BFGS prim. AutoARIMA order search + KPSS/seasonality tests = host-side loops over the ARIMA fit. |
| **Kernel SHAP** | **No** | Weighted linear regression — cuML's `kernel_shap.pyx` literally imports `cuml.linear_model.Lasso`/`LinearRegression`. mlrs already ships both. Coalition sampling is host-side. |
| **Permutation SHAP** | **No** | Permutation loop + repeated model `predict` calls + reductions. No solver. |
| **Symbolic / genetic regression** | **No** | Evolutionary search over expression trees: tournament selection, crossover, mutation, fitness eval. Host-side program objects + vectorized evaluation on existing element-wise kernels. No GP crate. |
| **metrics / preprocessing / model_selection / feature_extraction** | **No** | Reductions, scaling, CV splitting, TF-IDF counting — reductions + host bookkeeping. Mostly non-device or light-device. |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| In-house tree as struct-of-arrays | `treelite` (Rust/C++) model format | Only if mlrs needs to *import externally-trained* (XGBoost/LightGBM) forests for FIL inference. Not a v4.0 requirement; cuML uses it for cross-framework interop, which is out of scope. |
| Extend existing L-BFGS prim (batched) | `argmin` / `nlopt` Rust crate | Never for this project — would introduce a non-CubeCL solver and a host/device split that breaks the generic-runtime contract. The in-house prim is already validated. |
| `statsmodels` ARIMA oracle | `pmdarima` (AutoARIMA) | Only if AutoARIMA order-search needs an external reference. cuML deliberately uses `statsmodels` (not pmdarima); pmdarima lags on new numpy and adds a fragile dep. Implement order-search in-house, gate the ARIMA *fit* against statsmodels. |
| Optional `serde`+`bincode` for model persistence | `treelite` serialization | Only if/when Python `pickle`-style model round-trip is a stated requirement. Plain serde keeps it in-ecosystem; defer entirely until required. |
| `gplearn` oracle, property-gated | element-wise 1e-5 gate | Never — genetic programming is stochastic; an evolved expression won't match gplearn element-wise. Gate on R²/fitness band + structural reproducibility under a pinned seed. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Any new compute/algorithm Rust crate** (`linfa`, `smartcore`, `ndarray`, `nalgebra`) | Violates the CubeCL-only device-kernel constraint and the primitive-first discipline; v2 and v3 each added **zero** compute deps and shipped 18+14 estimators. ndarray specifically was deliberately rejected in v1 (npyz→`Vec` instead). | Existing mlrs primitive stack + plain `Vec`/struct-of-arrays. |
| **`treelite`** (and `treelite`-rs) | cuML uses it as a cross-framework model interchange/serialization format; mlrs owns its own tree node arrays and only needs to traverse models it built itself. Adds a heavy C++ build dependency for no in-scope benefit. | mlrs-native struct-of-arrays tree format. |
| **`pmdarima`** | Not used by cuML (statsmodels is its ARIMA oracle); frequently broken against current numpy/scipy; unmaintained-ish. | `statsmodels` for ARIMA fit reference; in-house order search. |
| **A genetic-programming framework** (`gpython`, GP crates) | Symbolic regression is the *thing being implemented*; pulling a GP engine would make it a wrapper, not a port. gplearn is the **oracle**, not a runtime dep. | In-house expression-tree GP on existing kernels. |
| **`scipy._lbfgsb` binding / any scipy FFI** | mlrs must not retain external solvers; the device-generic L-BFGS prim already exists and is validated. | Batched extension of `mlrs-backend/src/prims/lbfgs.rs`. |
| **Bumping `pyo3` past 0.28** | arrow-59's `pyarrow` feature pins pyo3 0.28.x; two PyInit ABIs in one cdylib crash the wheel at import (D-09). | Keep `pyo3 = 0.28`. |
| **Bumping `cubecl` past 0.10.0** | The whole generic-runtime + cpu-MLIR-safety story is validated against 0.10.0; F64-on-rocm behavior and MLIR lowering quirks are characterized at this pin. | Keep `cubecl = 0.10.0`. |
| **SharedMemory / cross-unit-atomic histogram kernels** | cpu-MLIR (cubecl-cpu) panics at launch on SharedMemory and has no cross-unit atomics; this is the #1 RF risk. | Single-owner GATHER idiom (spike-validated recipe; `Skill("spike-findings-mlrs")`). |

## Stack Patterns by Variant

**If the RF feasibility spike PASSES (GPU histogram/split works under GATHER):**
- Proceed RF → FIL → TreeSHAP on the in-house tree format.
- TreeSHAP depends on FIL's traversal over those arrays.

**If the RF feasibility spike FAILS (histogram/split infeasible under cpu-MLIR):**
- Per PROJECT.md, scope adjusts before committing the tree family.
- ARIMA, Kernel/Permutation SHAP, symbolic regression, sklearn-utility, and `cuml.accel` are **independent of the tree spike** and proceed regardless.
- TreeSHAP is the only explainer gated on trees; Kernel/Permutation SHAP are model-agnostic and unaffected.

**If model-persistence (Python pickle round-trip) becomes a requirement:**
- Add `serde` (derive) + `bincode` (compact) or `serde_json` (debuggable) — derive on the tree/ARIMA struct-of-arrays.
- Keep it behind a feature flag; it is not a compute dependency.

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `shap 0.52.0` | Python ≥3.12, numpy 2.x | Matches the project's Py≥3.12 floor; test-only. |
| `statsmodels 0.14.4` | numpy ≥1.26 (incl. 2.x), scipy ≥1.14 | 0.15 is dev-only; stay on 0.14.x. Test-only. |
| `gplearn 0.4.3` | scikit-learn (recent), numpy/scipy | Built on sklearn estimator API; slots into existing oracle harness. Test-only. |
| `scikit-learn >=1.6` | already pinned | No change; covers ensemble/metrics/preprocessing/feature_extraction/model_selection. |
| `pyo3 0.28` ↔ `arrow 59` | locked | arrow-59 `pyarrow` feature pins pyo3 0.28.x; do not bump either independently (D-09). |
| `cubecl 0.10.0` ↔ cpu-MLIR / rocm | locked | F64 unregistered on HIP (skips-with-log); MLIR has no SharedMemory/atomics. Gate = cpu(f64)+rocm(f32). |

## Sources

- `cuml-main/python/cuml/cuml/accel/{accelerator.py,core.py}` — read in full: meta-path finder/loader, AccelModule `__getattr__`, `install()`, `_OVERRIDES`/`_PATCHES`, stack-walk exclude — HIGH confidence (primary source).
- `cuml-main/python/cuml/cuml/explainer/{kernel_shap,permutation_shap,tree_shap,base}.pyx` — Kernel SHAP uses `cuml.linear_model.Lasso`/`LinearRegression`; base imports `shap.Explanation` — HIGH.
- `cuml-main/python/cuml/cuml/tsa/{arima.pyx,auto_arima.pyx,batched_lbfgs.py}` — batched L-BFGS is a `scipy._lbfgsb` per-series loop; AutoARIMA uses internal KPSS/seasonality tests — HIGH.
- `cuml-main/python/cuml/pyproject.toml` + `cuml-main/dependencies.yaml` — `shap`, `statsmodels` listed as test deps; `pmdarima` absent — HIGH.
- `crates/*/Cargo.toml` + workspace `Cargo.toml` — confirmed existing L-BFGS prim, no serde, pyo3 0.28 / cubecl 0.10.0 pins, no ndarray — HIGH.
- PyPI / web (2026-06-26): `gplearn 0.4.3` (latest stable), `shap 0.52.0` (2026-05-28, Py≥3.12), `statsmodels 0.14.4` stable / 0.15 dev — MEDIUM-HIGH (web, verified against project Python floor).
- `.planning/PROJECT.md`, `.planning/notes/v3-hard-algorithm-backlog.md`, `cuml-mlrs-gap-inventory.md` — milestone scope, oracle assignments, spike-gating — HIGH (project source of truth).

---
*Stack research for: mlrs v4.0 feature additions*
*Researched: 2026-06-26*
