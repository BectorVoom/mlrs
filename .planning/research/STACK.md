# Stack Research

**Domain:** Manifold/clustering pair (UMAP + HDBSCAN) + Rust-native builder API + pure-Python sklearn shim, on the existing mlrs Rust/CubeCL stack (v3.0)
**Researched:** 2026-06-22
**Confidence:** HIGH

## Bottom Line

**Zero new RUNTIME crate dependencies and zero new compute dependencies are required for v3.0.** UMAP, HDBSCAN, and the shared KNN-graph primitive are all host orchestration over the *already-validated* v1 `NearestNeighbors` (top-k) prim plus at most a handful of new in-repo feature-free CubeCL kernels (mutual-reachability map, fuzzy-set symmetrization, SGD layout step, MST/union-find on host). The four decisions the roadmapper needs:

1. **Builder pattern: KEEP HAND-WRITTEN builders + add fit/unfit typestate. Do NOT add `bon`/`typed-builder`/`derive_builder`.** The hand-rolled `T::builder() → TBuilder (Default + chained setters) → T` pattern *already ships* on 9 v2 estimators (`LinearSVR`, `LinearSVC`, `MBSGD*`, all 5 Naive-Bayes). The v3 work is to (a) retrofit that *same* pattern across the other ~21 estimators and (b) layer a zero-cost `PhantomData`-typestate (`Estimator<Unfit>` → `fit` → `Estimator<Fit>`) on top. A derive macro buys nothing the established convention doesn't already give, and adds a proc-macro dependency against the "no new heavy deps" discipline.

2. **Oracle libraries (dev/test only — regen in the /tmp venv, NOT a wheel dep):**
   - **UMAP → `umap-learn` 0.5.12** (the only reference; no sklearn equivalent). **Property/structural gate**, NOT element-wise 1e-5 (stochastic SGD layout — the RandomProjection D-12 precedent).
   - **HDBSCAN → `sklearn.cluster.HDBSCAN` (ships in scikit-learn ≥ 1.6, already pinned) as the PRIMARY gate oracle; `hdbscan` 0.8.44 as a cross-check.** **Exact labels up to permutation** is the hard gate.

3. **Test/dev deps:** Add `numba` (transitively, via `umap-learn`) to the /tmp oracle-regen venv *only*. `sklearn.cluster.HDBSCAN` and `hdbscan` need no new deps beyond the existing numpy/scipy/scikit-learn. Rust dev-deps unchanged (`npyz` 0.9 reads the new committed `.npz` fixtures).

4. **No new device/compute dependency.** Confirmed: `cubecl` 0.10, `cubek-matmul`/`cubek-reduce` 0.2.0 stay put; no `cubek-random`, no graph/MST crate, no host-linalg crate. KNN reuses the gated top-k prim; MST/union-find/condensed-tree are host-side integer algorithms (not device kernels).

## Recommended Stack

### Core Technologies (UNCHANGED — all already pinned)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| cubecl | 0.10.0 (`default-features=false`) | device-kernel layer, generic over float + runtime | The new KNN-graph / mutual-reach / UMAP-SGD-step kernels are feature-free CubeCL over the existing GATHER pattern. No version change. |
| cubek-matmul | 0.2.0 | GEMM prim | Not on the UMAP/HDBSCAN critical path (these are distance/graph, not dense GEMM), but stays wired. No change. |
| cubek-reduce | 0.2.0 | reductions | UMAP fuzzy-set normalization (per-point ρ/σ via row reductions) and HDBSCAN core-distance (k-th NN distance) are reductions over the KNN-distance matrix. Already wired. No change. |
| arrow | 59 (`pyarrow`) | zero-copy host↔device interchange | Dense Float32/Float64 ingress is sufficient; UMAP/HDBSCAN take a dense `X`. No change. |
| pyo3 | 0.28 (`abi3-py312`, `extension-module`) | Python bindings | **Stays PINNED at 0.28** (arrow-59's `pyarrow` feature transitively pins pyo3 0.28; mixing ABIs crashes the wheel at import — D-09/PY-05). v3 wraps UMAP/HDBSCAN as new `#[pyclass]`es only — no pyo3 pressure. |
| mimalloc | 0.1 (`local_dynamic_tls`) | global allocator in mlrs-py | dlopen-safe wheels. No change. |
| bytemuck | 1 (`derive`) | zero-copy `&[T]`↔`&[u8]` Arrow→CubeCL | No change. |
| thiserror / anyhow | 2 / 1 | typed errors / boundary errors | No change. The new typestate error surface uses the existing `AlgoError`. |
| log / env_logger | 0.4 / 0.11 | dtype/backend selection + f64-on-rocm skip-with-log | No change. |

### New Primitive / Kernels (mlrs-authored — NOT crates)

The "one new shared primitive per phase" discipline. The **KNN-graph primitive lands standalone first** (project rule), then UMAP and HDBSCAN consume it. All feature-free, generic over `F: Float`, GATHER idiom (single-owner outputs, u32/F accumulators, if-guards, **no SharedMemory / no cross-unit atomics / no `bool` / no `F::INFINITY` / no descending-shift loops**).

| Primitive / kernel | Composes from | New device work | cpu-MLIR / f64-rocm OK? |
|--------------------|---------------|-----------------|--------------------------|
| **KNN-graph prim** (shared) | v1 `NearestNeighbors` (top-k) | Materialize the (n × k) neighbor-index + neighbor-distance arrays from the existing top-k prim; this IS top-k's output, re-exposed as a graph adjacency. No new kernel if top-k already returns indices+distances (it does — `KNeighbors` trait). | Yes — top-k is already gated on cpu+rocm. f64-on-rocm skips-with-log. |
| **Mutual-reachability** (HDBSCAN) | KNN-graph distances + core-distance reduction | core-dist = k-th NN distance (row reduction over KNN distances); `mr(a,b) = max(core_a, core_b, d(a,b))` is an elementwise max over the sparse KNN graph. Single-owner per edge. | Yes — elementwise max + reduction, no atomics. |
| **Fuzzy simplicial set** (UMAP) | KNN-graph distances + per-row reductions | per-point ρ (nearest non-zero dist) + σ (binary search to fixed log2(k) target, host-driven or bounded device loop), membership `exp(-(d-ρ)/σ)`, then symmetrize `p+q-p·q`. Elementwise over the KNN graph. | Yes — elementwise + bounded reductions; the σ binary-search runs host-side over uploaded row stats (mirrors the spectral host-slice precedent). |
| **UMAP SGD layout step** (UMAP) | host SplitMix64 (edge/negative sampling) + elementwise gradient | per-epoch: sample positive edges + negative samples (host SplitMix64 → upload index lists), attractive/repulsive gradient is elementwise over sampled edge endpoints; embedding update is single-owner per embedding coordinate. Same shape as the v2 PRIM-10 two-pass SGD solver. | Yes — reductions+elementwise, no SharedMemory/atomics (the v2 SGD solver established this on cpu-MLIR). f64-on-rocm skips-with-log. |
| **MST + union-find + condensed tree + stability** (HDBSCAN) | mutual-reach edges (host-materialized) | **Host-side integer algorithms** (Prim's/Borůvka MST or Kruskal+DSU, condensed-tree walk, stability extraction). NOT device kernels — graph/tree construction is exactly the atomics-hostile pattern v3 deliberately avoids on-device. | N/A (host) — no device atomics needed; this is *why* HDBSCAN is feasible without the deferred tree-construction lift. |

### Development Tools / Oracle Libraries

| Tool | Version | Purpose | Notes |
|------|---------|---------|-------|
| scikit-learn (oracle) | **≥ 1.6 (current latest 1.9.0)** — ALREADY PINNED | HDBSCAN reference (`sklearn.cluster.HDBSCAN`) + the v1/v2 1e-5 gates | `sklearn.cluster.HDBSCAN` has shipped since sklearn 1.3 — **no new dependency**. This is the **primary HDBSCAN gate oracle**. |
| **umap-learn** | **0.5.12** (latest, 2026; `requires_python >=3.9`) | UMAP reference (the ONLY reference — no sklearn equivalent) | **NEW dev-venv dep.** Pulls `numba >=0.51.2` + `pynndescent` + `tqdm` transitively (the heavy bit — see Version Compatibility). Used ONLY to regenerate committed `.npz` UMAP fixtures in the /tmp venv. Stochastic → **property/structural gate**, never element-wise. |
| **hdbscan** | **0.8.44** (latest, 2026) | HDBSCAN cross-check oracle | **Optional NEW dev-venv dep.** Deps are only `numpy<3,>=1.20`, `scipy>=1.0`, `scikit-learn>=1.6` — **no numba**. Use as a secondary cross-check; `sklearn.cluster.HDBSCAN` is the gate. |
| numpy | **> 2.0.0** — ALREADY PINNED (latest 2.5.0) | seeded oracle RNG (`default_rng`) + fixture dtype | No change. Caveat: numba (via umap-learn) lags numpy — see Version Compatibility. |
| scipy | latest (1.18.0) — ALREADY in oracle venv | reference linalg/solve for existing fixtures | No change; not needed for UMAP/HDBSCAN oracles specifically. |
| npyz | 0.9 (`npz`) | Rust-side `.npz` fixture reader (dev/test) | The new UMAP/HDBSCAN fixtures are committed `.npz` blobs read by the same harness. No change. |
| maturin | (existing) | four per-backend abi3-py312 wheels | UMAP/HDBSCAN add `#[pyclass]` wrappers shipped in the existing wheels. No change. |

### Rust Builder-Pattern Decision (explicit)

**Decision: hand-written builders (extend the shipped convention) + a `PhantomData` fit/unfit typestate. Reject all derive-macro crates.**

| Option | Version | Verdict | Reason |
|--------|---------|---------|--------|
| **Hand-written builder + typestate (chosen)** | n/a | ✓ ADOPT | The pattern is **already shipped and gated** on 9 estimators: `LinearSVR::builder() → LinearSVRBuilder { Default + chained setters } → fit`. v3 = retrofit the same shape across the remaining ~21 + the 2 new ones, and add `Estimator<State>` where `State ∈ {Unfit, Fit}` via zero-sized `PhantomData` so `predict`/`transform` are only callable on `Estimator<Fit>` (compile-time NotFitted). Zero dependency, full control over the `AlgoError` surface, and it composes with the existing `any_estimator!` PyO3 macro which is written against the `Fit`/`Predict` traits — a derive macro would fight that hand-written trait-impl machinery. |
| `bon` | 3.9.3 (active, 2026-06) | ✗ REJECT | Best-maintained modern builder crate, but it is a **proc-macro dependency** (syn/quote/proc-macro2 in the build graph) added for a pattern already hand-written and proven. Its compile-time-required-field checks duplicate what `Default` + typestate already give here. No justification under "no new heavy deps". |
| `typed-builder` | 0.23.2 (2025-11) | ✗ REJECT | Same proc-macro-dependency objection. Its typestate is per-field (which fields are set), which is orthogonal to the fit/unfit lifecycle typestate this milestone actually wants — we'd still hand-write the lifecycle states. |
| `derive_builder` | 0.20.2 (2024-10, slowing) | ✗ REJECT | Older, less actively updated, and produces `Result`-returning builders (runtime "field unset" errors) rather than the compile-time guarantee. Worst fit for a "typed builder" goal. |

**Why typestate (not just builders):** the milestone explicitly wants "fit/unfit typestate". The current `Fit` trait returns `&mut Self` (sklearn-style in-place), so a fitted and unfitted estimator are the *same* type — calling `predict` before `fit` is a runtime error today. A `PhantomData<State>` parameter makes "predict requires a fitted estimator" a *compile-time* invariant on the Rust surface, while the PyO3 layer (which mirrors sklearn's mutable `fit→self`) keeps the runtime `NotFittedError` contract. This is a pure-Rust, zero-cost, zero-dependency addition.

### Pure-Python sklearn Shim Decision (explicit)

**Decision: EXTEND the already-existing pure-Python shim. No new Python runtime dep.**

A pure-Python shim **already exists** for the v1 12 estimators: `mlrs.base.MlrsBase` subclasses `sklearn.base.BaseEstimator` directly, so `get_params`/`set_params`/`clone`/`__repr__` come for free from a faithful `__init__`; `_check_fitted` delegates to `sklearn.utils.validation.check_is_fitted` (→ `NotFittedError`); `__sklearn_tags__` turns off unsupported (sparse/array-api/NaN) checks. `python/tests/test_shims.py` already gates `get_params`/`set_params`/`clone` round-trips for the 12. v3 work = (a) extend `MlrsBase` subclassing to the v2 18 + the new UMAP/HDBSCAN classes, (b) wire `check_estimator` coverage. **The runtime dep is the already-pinned `scikit-learn>=1.6`** (the wheel `dependencies` list in `cpu.pyproject.toml`: `numpy>2.0.0`, `pyarrow>=14`, `scikit-learn>=1.6`). No new Python dependency.

## Oracle Strategy Per Algorithm (explicit)

| Algorithm | Gate type | Reference lib + version | Rationale |
|-----------|-----------|-------------------------|-----------|
| **KNN-graph prim** | element-wise **≤ 1e-5** (indices exact, distances 1e-5) | `sklearn.neighbors.NearestNeighbors` (scikit-learn ≥ 1.6) | It is the v1 top-k prim re-exposed; deterministic; reuses the existing 1e-5 KNN gate. Land + gate standalone before UMAP/HDBSCAN consume it. |
| **UMAP** | **property/structural** (NO 1e-5; NO sklearn ref) | `umap-learn` 0.5.12 | Stochastic SGD layout (edge/negative sampling, SplitMix64 ≠ numba RNG) → embedding cannot match element-wise. Gate on *structure-preserving* properties: trustworthiness/continuity vs `umap-learn` within a band, k-NN-overlap of embedded vs original neighborhoods, and seed-reproducibility of mlrs itself. Mirrors the RandomProjection D-12 property-gate precedent (JL distortion + distribution stats + seed reproducibility). |
| **HDBSCAN** | **exact labels up to permutation** (hard gate) | PRIMARY: `sklearn.cluster.HDBSCAN` (scikit-learn ≥ 1.6). CROSS-CHECK: `hdbscan` 0.8.44 | MST→condensed-tree→stability is deterministic given the (deterministic) mutual-reachability graph → cluster *assignments* are integer-exact up to label permutation + the conventional `-1` noise label (the v2 classifier exact-label precedent). Use the in-tree sklearn estimator as the gate (no extra dep, version-locked to the pinned sklearn); `hdbscan` package as an independent corroboration of the labels. |

## Installation

```bash
# NO new crate installs. NO new wheel runtime deps.
# v3 work is additive modules in existing crates + extending the existing
# pure-Python shim:
#   crates/mlrs-kernels/src/{knn_graph,mutual_reach,fuzzy_set,umap_sgd}.rs  (feature-free kernels)
#   crates/mlrs-backend/src/prims/{knn_graph,...}.rs                        (KNN-graph prim lands FIRST)
#   crates/mlrs-algos/src/{manifold/umap.rs, cluster/hdbscan.rs}            (host MST/union-find/condensed-tree)
#   crates/mlrs-algos/src/**                                                (builder + PhantomData typestate retrofit)
#   crates/mlrs-py/src/estimators/...                                       (#[pyclass] UMAP/HDBSCAN)
#   crates/mlrs-py/python/mlrs/{manifold.py,cluster.py}                     (extend MlrsBase shim)
#
# Workspace Cargo.toml [workspace.dependencies] is UNCHANGED.

# Oracle fixture REGEN ONLY (committed .npz blobs; CI never runs this) —
# add umap-learn (pulls numba) + hdbscan to the existing /tmp PEP-668 venv:
python3 -m venv /tmp/oracle-venv
/tmp/oracle-venv/bin/pip install numpy scipy scikit-learn umap-learn==0.5.12 hdbscan==0.8.44
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Hand-written builder + typestate | `bon` 3.9.3 derive builder | Only if the estimator count exploded (100s) AND the hand-written boilerplate became a real maintenance burden AND the "no new deps" rule were relaxed — none true at 32 estimators with the pattern already shipped. |
| `sklearn.cluster.HDBSCAN` as gate | `hdbscan` 0.8.44 as gate | Use `hdbscan` as the gate ONLY if a behavior mlrs targets exists in the standalone package but not in sklearn's in-tree port (e.g. `approximate_predict`, soft clustering / membership vectors). For core fit-labels, prefer the in-tree (zero-dep, version-locked) oracle. |
| Property gate for UMAP | element-wise 1e-5 vs `umap-learn` | Never feasible — `umap-learn` uses numba-RNG SGD; element-wise match is impossible across RNG implementations (same reasoning that forced the RandomProjection property gate). |
| Host MST/union-find (HDBSCAN) | device-side MST kernel (atomics) | Only in a future milestone with a CUDA gate where host round-trips dominate; on-device MST needs atomics that cpu-MLIR forbids — host-side is correct AND the feasibility-enabling choice for v3. |
| Reuse v1 top-k for KNN graph | `cubec`/external ANN (pynndescent-style) | Only at very large n where exact k-NN is too slow; v3 sizes use exact k-NN from the gated top-k prim. ANN is a later optimization, not a v3 dependency. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `bon` / `typed-builder` / `derive_builder` (proc-macro builder crates) | New proc-macro dependency for a pattern already hand-written + gated on 9 estimators; fights the hand-written `Fit`/`Predict` trait machinery the `any_estimator!` PyO3 macro is built on | Hand-written `T::builder()` + `PhantomData<{Unfit,Fit}>` typestate |
| `cubek-random` 0.2.0 (device RNG) | No caller seed (breaks reproducibility) + shared-memory Tausworthe (breaks cpu-MLIR) — UMAP edge/negative sampling needs reproducible host RNG | Host SplitMix64 (already in `prims/rng.rs`) → upload sampled index lists |
| Any device-side MST / tree-construction kernel | GPU tree/graph construction needs atomics cpu-MLIR forbids — the exact lift v3 deliberately defers (RandomForest) | Host-side Prim's/Borůvka MST + DSU union-find + condensed-tree walk (integer host code) |
| `numba` as a wheel RUNTIME dependency | numba is a heavy JIT runtime; it belongs only in the oracle-regen venv (it arrives via `umap-learn`), never in the shipped mlrs wheel | Keep numba confined to `/tmp/oracle-venv`; committed `.npz` blobs carry the UMAP reference into CI |
| `umap-learn` as the HDBSCAN-or-anything-else oracle | It is the UMAP reference only, and a heavy (numba) dep | `sklearn.cluster.HDBSCAN` for HDBSCAN (zero new dep) |
| pyo3 0.29 | Links a second PyInit ABI alongside arrow-59's transitive pyo3 0.28 → wheel crashes at import (D-09/PY-05) | Stay on pyo3 0.28 |
| `ndarray` / `nalgebra` / a graph crate (`petgraph`) | Host MST/union-find is ~100 lines of integer code; a graph crate would split the codepath and add a dep | Hand-written host union-find + MST |
| Native sparse Arrow interchange | Still out of scope (v3 backlog item); UMAP/HDBSCAN take dense `X` | Dense Float32/Float64 ingress; densify at the Python wrapper if a sparse `X` is passed |

## Stack Patterns by Variant

**If building the KNN-graph primitive (lands first):**
- Re-expose the gated v1 top-k `KNeighbors` output (indices + distances) as graph adjacency; gate element-wise ≤ 1e-5 vs `sklearn.neighbors.NearestNeighbors` standalone.
- Because primitive-first discipline requires the shared prim validated before UMAP/HDBSCAN consume it.

**If implementing UMAP (stochastic):**
- KNN-graph → fuzzy simplicial set (elementwise + host σ binary-search) → host SplitMix64 edge/negative sampling → device elementwise SGD layout step. Property-gate (trustworthiness/k-NN-overlap/seed-reproducibility) vs `umap-learn` 0.5.12.
- Because the SGD layout is stochastic and has no sklearn reference — same contract class as RandomProjection.

**If implementing HDBSCAN (deterministic labels):**
- KNN-graph → mutual-reachability (device elementwise max) → **host** MST + union-find + condensed tree + stability extraction → labels. Exact-labels-up-to-permutation gate vs `sklearn.cluster.HDBSCAN` (cross-check `hdbscan` 0.8.44).
- Because tree/graph construction is host-side integer work (dodges the deferred GPU-tree-atomics lift) and the result is deterministic.

**If retrofitting the Rust builder API:**
- Copy the shipped `LinearSVRBuilder` shape (Default + chained setters + `T::builder()`); add `Estimator<State = Unfit>` with `fit(self) -> Estimator<Fit>` (or keep `&mut self` for PyO3 parity and gate `predict` on a `Fit` typestate wrapper).
- Because the convention is already proven on 9 estimators and the PyO3 `any_estimator!` macro is written against the existing traits.

**If extending the pure-Python sklearn shim:**
- Subclass `MlrsBase` (which subclasses sklearn `BaseEstimator`) for the v2 18 + UMAP/HDBSCAN; store ctor args verbatim in `__init__`; extend `test_shims.py` + `check_estimator` coverage.
- Because get_params/set_params/clone come free from a faithful `__init__`, and the base class already exists.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `umap-learn` 0.5.12 | scikit-learn ≥ 1.6, numpy ≥ 1.23, scipy ≥ 1.3.1 | Declared deps satisfy the existing pins (scikit-learn ≥ 1.6, numpy > 2.0, scipy current). |
| `umap-learn` 0.5.12 | **numba >= 0.51.2** (transitive) | ⚠️ THE compatibility watch-item. numba ships its own numpy upper-bound that historically lags new numpy releases; with numpy 2.5.0 the regen venv may need a recent numba (or a slightly older numpy) to resolve. **Mitigation:** numba lives ONLY in `/tmp/oracle-venv`; pin a numba that supports the installed numpy at regen time. The committed `.npz` UMAP fixtures decouple CI from this entirely. |
| `hdbscan` 0.8.44 | numpy <3 ≥ 1.20, scipy ≥ 1.0, scikit-learn ≥ 1.6 | **No numba** — clean against the existing pins; safe optional cross-check. |
| `sklearn.cluster.HDBSCAN` | scikit-learn ≥ 1.6 (already pinned; in-tree since 1.3) | **Zero new dependency** — the primary HDBSCAN gate. Version-locked to the wheel's own sklearn pin. |
| arrow 59 (`pyarrow`) | pyo3 0.28.x | HARD: only one pyo3 ABI may link the cdylib. Do not bump pyo3. Unchanged from v2. |
| cubecl 0.10 cpu (MLIR) | new v3 kernels | Must be SharedMemory-free / atomic-free (GATHER idiom) or the cpu backend panics at launch. New KNN/mutual-reach/fuzzy-set/SGD kernels follow this. |
| cubecl-cpp 0.10 (rocm/HIP) | f64 | F64 NOT registered for HIP → f64-on-rocm UMAP/HDBSCAN oracle cases skip-with-log; v3 kernels inherit the gate (cpu f64 + rocm f32). |
| Rust builders (hand-written) | n/a | No crate; nothing to version. Zero-cost `PhantomData` typestate is std-only. |

## Sources

- PyPI JSON API (fetched 2026-06-22): `umap-learn` 0.5.12 (requires_python ≥ 3.9; deps numpy ≥ 1.23, scipy ≥ 1.3.1, scikit-learn ≥ 1.6, **numba ≥ 0.51.2**); `hdbscan` 0.8.44 (deps numpy <3 ≥ 1.20, scipy ≥ 1.0, scikit-learn ≥ 1.6, **no numba**); `scikit-learn` 1.9.0 (≥ 1.6 pinned); numpy 2.5.0; scipy 1.18.0. HIGH (registry of record).
- crates.io API (fetched 2026-06-22, User-Agent): `bon` 3.9.3 (updated 2026-06-15), `typed-builder` 0.23.2 (2025-11), `derive_builder` 0.20.2 (2024-10). HIGH (registry of record).
- mlrs source `crates/mlrs-algos/src/linear/linear_svr.rs` — shipped hand-written `LinearSVR::builder()` / `LinearSVRBuilder` (Default + chained setters); 9 estimators total carry `pub fn builder()`. HIGH (direct read).
- mlrs source `crates/mlrs-algos/src/traits.rs` — `Fit` returns `&mut Self` (in-place, no typestate today). HIGH (direct read).
- mlrs source `crates/mlrs-py/python/mlrs/base.py` + `python/tests/test_shims.py` — pure-Python shim already exists: `MlrsBase(BaseEstimator)`, get/set_params/clone gated for the v1 12. HIGH (direct read).
- mlrs source `crates/mlrs-py/pyproject/cpu.pyproject.toml` — wheel runtime deps `numpy>2.0.0`, `pyarrow>=14`, `scikit-learn>=1.6`. HIGH (direct read).
- mlrs source `Cargo.toml [workspace.dependencies]` + `scripts/gen_oracle.py` header — pin rationale + /tmp PEP-668 oracle-venv regen workflow. HIGH (direct read).
- v2.0 STACK.md (`.planning/milestones/v2.0-research/STACK.md`) — host-SplitMix64 RNG decision, GATHER idiom, pyo3-0.28/arrow-59 ABI pin, property-gate precedent. HIGH (prior validated research).
- GitHub lmcinnes/umap "Compatibility with Numpy > 2.0" discussion — corroborates the numba/numpy-2 watch-item. MEDIUM (community thread).
- Project memory (`cubecl-cpu-no-shared-memory.md`, `rocm-is-runnable-gpu-gate.md`, `oracle-fixture-regen-needs-venv.md`, `cubecl algo crates moved to cubek`) — constraint corroboration. HIGH.

---
*Stack research for: mlrs v3.0 UMAP + HDBSCAN + Rust-native builder API + sklearn shim*
*Researched: 2026-06-22*
