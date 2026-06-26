# Project Research Summary

**Project:** mlrs v4.0 — Tree Ensembles, Time-Series & Full-Surface Completion
**Domain:** CubeCL/cpu-MLIR-gated Rust ML library; final cuML algorithm surface close-out
**Researched:** 2026-06-26
**Confidence:** HIGH (all four research areas grounded in primary in-tree sources and shipped project decisions)

---

## Executive Summary

v4.0 is the milestone that closes out the remaining cuML algorithm surface — tree ensembles (RandomForest → FIL → TreeSHAP), time-series (ARIMA / AutoARIMA), model-agnostic explainers (Kernel/Permutation SHAP), genetic/symbolic regression, the sklearn-utility surface, and the transparent `cuml.accel` drop-in. The core architectural verdict from all four research streams is identical: **every new feature is built from the existing primitive stack and plain Rust data structures; zero new compute/algorithm Rust crates are added**, continuing the v2/v3 record. The only additive dependencies are three Python test-oracle packages (`shap 0.52.0`, `statsmodels 0.14.4`, `gplearn 0.4.3`) and the already-pinned `scikit-learn >=1.6`. `pyo3` stays at `0.28` and `cubecl` stays at `0.10.0` — both hard-pinned; bumping either breaks the wheel or the cpu-MLIR correctness gate.

The defining structural tension of v4.0 is **mixed oracle/gate types across a single milestone**. Prior milestones had a single gate regime (v1/v2: 1e-5 value match or exact-label; v3: those plus property/structural for stochastic UMAP). v4.0 spans all four gate types simultaneously: exact value (FIL deterministic traversal; TreeSHAP ≤1e-5 plus additive-efficiency); exact-label (NOT applicable to RF — scoped to deterministic SGD/SVM/NB only); band-on-likelihood/forecast (ARIMA — sklearn has no ARIMA, `statsmodels` is the correct oracle, raw coefficients are multimodal and ungatable); and property/structural-plus-band (RF ensemble, Kernel/Permutation SHAP, genetic — SplitMix64 ≠ MT19937 makes element-wise match impossible). The recommended resolution is to establish the **two-tier stochastic gate** as a milestone-wide convention in the RF feasibility spike (Phase 17): a deterministic injected-fixed-index single-tree tier (tight, the real correctness witness) plus an ensemble/predictive-quality band tier (RNG-tolerant).

The single highest risk is **GPU tree construction under cpu-MLIR**: the textbook GPU histogram kernel (cuML's and every CUDA reference's) uses `SharedMemory` + `atomicAdd`, both of which the `cubecl-cpu` MLIR backend panics on at launch. The GATHER inversion is the hypothesis — one unit per `(node, feature, bin)` output cell loops over the node's sample range with no contended writes — and it must be proven in a dedicated feasibility spike before any tree-family estimator work begins. This spike (Phase 17) is gating: if the GATHER histogram is not both correct AND tractable under cpu-MLIR, the tree chain (RF → FIL → TreeSHAP) is re-scoped before commitment. Every other v4.0 track (ARIMA, model-agnostic SHAP, sklearn-utility surface, genetic, cuml.accel) is **independent of the tree spike** and proceeds regardless.

---

## Key Findings

### Recommended Stack

**Zero new Rust compute dependencies** — the headline finding from STACK.md, confirmed by ARCHITECTURE.md, FEATURES.md, and PITFALLS.md independently. All v4.0 features assemble from existing validated primitives: GEMM, reductions, distance, SVD/eig, Cholesky, top-k, L-BFGS, coordinate descent, RNG, KNN-graph, two-pass SGD solver. Tree histogram/split/partition kernels are **new CubeCL kernels** authored in-house (the spike's deliverable) but consume no new crates. The Kalman filter is GEMM + Cholesky recursion already in-tree; batched ARIMA extends the existing L-BFGS prim with a batched wrapper, not a new crate. `cuml.accel` is pure-Python `importlib` machinery — zero Rust, zero PyO3 changes.

**Core technologies (new for v4.0):**
- `shap 0.52.0` (test-only oracle): Kernel/Permutation/Tree SHAP reference values — pin `>=0.46,<0.53`
- `statsmodels 0.14.4` (test-only oracle): ARIMA log-likelihood / forecast / KPSS stationarity reference — stay on `0.14.x` (0.15 is dev)
- `gplearn 0.4.3` (test-only oracle): symbolic regression property gate — pin exact (infrequent releases)
- `scikit-learn >=1.6` (already pinned): covers RF/metrics/preprocessing/feature_extraction/model_selection oracles — no version bump

**Hard pins that must not change:**
- `pyo3 = 0.28` — arrow-59 `pyarrow` feature pins this; two PyInit ABIs in one cdylib crash the wheel at import (D-09)
- `cubecl = 0.10.0` — the entire cpu-MLIR safety story (GATHER idiom, F64 registration on HIP, MLIR lowering quirks) is characterized at this pin

### Expected Features

All four research files agree on the dependency topology. The tree chain is a hard dependency chain; everything else is spike-independent.

**Must have — table stakes (P1):**
- RandomForest GPU histogram/split **feasibility spike** — the gating make-or-break phase; runs first; delivers GO/ADJUST/ABORT verdict with A1–A5 evaluated
- RandomForestClassifier + RandomForestRegressor — spike-gated; property/predictive-quality two-tier gate
- FIL (Forest Inference Library) — exact deterministic inference over the mlrs tree format; the one exact tree-stack gate
- ARIMA(p,d,q) + forecast — batched Kalman filter + batched L-BFGS; `statsmodels` band gate on likelihood/forecasts
- metrics core (accuracy, confusion_matrix, r2, mse, mae) — reductions; exact/<=1e-5; degenerate fixtures mandatory
- preprocessing scalers (Standard/MinMax/MaxAbs/Robust/Normalizer) — column-stat fit + elementwise transform; <=1e-5
- model_selection splitters (train_test_split, KFold, StratifiedKFold) — structural gate; MT19937-host-match decision required

**Should have — differentiators (P2):**
- TreeSHAP — exact <=1e-5 vs `shap.TreeExplainer` on mlrs's own tree (NOT sklearn's forest); additive-efficiency invariant mandatory; after FIL
- Kernel SHAP + Permutation SHAP — model-agnostic; additive-efficiency invariant (exact) + convergence band vs `shap`; independent of tree stack
- AutoARIMA — order search via KPSS/seasonality + IC grid/stepwise; exact selected-order gate on clean synthetic series
- preprocessing encoders (OneHot/Ordinal/Label/LabelBinarizer/SimpleImputer)
- roc_auc_score / log_loss / precision_recall_curve
- symbolic/genetic regression — `gplearn` property gate (R^2 band + seed-reproducibility within mlrs)
- cuml.accel — pure-Python sys.modules import-hook; land LAST so it proxies the full v4.0 + v1-v3 surface

**Defer (out-of-v4.0 scope):**
- GridSearchCV/RandomizedSearchCV — delegate to sklearn passthrough (cuML itself does this via `__getattr__`)
- feature_extraction (TfidfVectorizer/CountVectorizer) — P3; text-heavy, host-side, lower demand
- Interventional/feature-perturbation TreeSHAP with background data — path-dependent TreeSHAP first
- Seasonal ARIMA (P,D,Q,s) + full exog in first cut — graded sub-requirements; non-seasonal first
- Treelite / XGBoost / LightGBM model ingestion into FIL — external-model import is a later milestone
- SymbolicClassifier / SymbolicTransformer (gplearn) — SymbolicRegressor first

### Architecture Approach

v4.0 attaches to the existing 5-crate spine (`mlrs-core` / `-kernels` / `-backend` / `-algos` / `-py`) without structural changes. The primitive-first discipline holds: every new compute primitive (GATHER histogram/split/partition, batched Kalman, program-eval) is validated standalone in `mlrs-kernels` + `mlrs-backend` before any `mlrs-algos` estimator consumes it. The tree node store — `SparseTreeNode { colid: u32, threshold: F, left_child: i32 (-1 = leaf), value: F }`, right child = `left_child + 1` (cuML `flatnode.h` convention) — is the single load-bearing contract binding RF (writer), FIL (reader), and TreeSHAP (reader); it is fixed in the Phase 17 spike. `cuml.accel` is a pure-Python subpackage (`python/mlrs/accel/`) layered over the shim surface; zero Rust changes.

**Major new components (v4.0 additions by crate):**
1. `mlrs-kernels/tree.rs` — GATHER histogram, best-split reduction, node-partition, batched FIL traversal (spike-gated; all cpu-MLIR-safe; no SharedMemory, no atomics, no `F::INFINITY`)
2. `mlrs-kernels/kalman.rs` — per-series sequential Kalman recursion (one unit/series; sequential time loop; cpu-MLIR-safe)
3. `mlrs-kernels/program.rs` — batch stack-program evaluator for genetic fitness
4. `mlrs-backend` prims: `quantiles`, `tree_hist`, `best_split`, `node_partition`, `tree_traverse`, `batched_kalman`, `program_eval`; `lbfgs.rs` (MODIFY: batched wrapper)
5. `mlrs-algos` new modules: `tree/` (DecisionTree core), `ensemble/` (RF clf/reg), `fil/` (batched inference), `tsa/` (ARIMA/AutoARIMA), `explainer/` (Kernel/Perm/Tree SHAP), `genetic/` (SymbolicRegressor)
6. `mlrs-py` estimator wrappers: `ensemble.rs`, `tsa.rs`, `explainer.rs`, `genetic.rs`; corresponding Python shims
7. `python/mlrs/accel/` — pure-Python MetaPathFinder + AccelModule + `_overrides` table (zero Rust)

### Critical Pitfalls

1. **GPU tree histogram assumes atomics/SharedMemory (P1 — CRITICAL)** — cuML's histogram kernel uses `__shared__` + `atomicAdd`; both cause `cubecl-cpu` MLIR to panic at launch. Avoid by: running the feasibility spike first (Phase 17) to prove the GATHER inversion — one unit per `(node, feature, bin)` loops over the node's sample range with local accumulation; relabel (not scan/compaction) for node partition; `seed-from-first` statement-form `if` for gain argmax (never `F::INFINITY` init). Five abort signals (A1–A5) must be evaluated; the spike delivers an explicit GO/ADJUST/ABORT verdict.

2. **Stochastic estimators gated element-wise (P2 — CRITICAL)** — mlrs SplitMix64 != NumPy MT19937; RF, Kernel/Permutation SHAP, and symbolic regression will never match sklearn/shap/gplearn element-wise. The exact-predicted-label gate (for deterministic SGD/SVM/NB) does NOT apply to RF. Avoid by: two-tier gate established in the RF spike: (a) deterministic-core tier — inject fixed bootstrap/feature-subset indices into both mlrs and sklearn; (b) ensemble score-band tier — accuracy/R^2 within a documented margin. SHAP: additive-efficiency invariant (exact) + brute-force enumeration on small `n` + convergence band on large `n`. Genetic: R^2-band + internal seed-reproducibility.

3. **ARIMA oracle is statsmodels, not sklearn (P3) — and a PROJECT.md slip** — sklearn has no ARIMA; gating ARIMA coefficients <=1e-5 against any library is wrong (multimodal optima). Avoid by: `statsmodels.tsa.arima.model.ARIMA` as the oracle; gate on log-likelihood / forecast-band / known-coefficient recovery via Jones/PACF transform; accumulate Kalman log-likelihood in f64 even on the f32 rocm path; per-series convergence flags so one non-converging series cannot NaN-poison the batch.

4. **cuml.accel silently returns wrong results instead of CPU fallback (P4)** — any unsupported param/config silently ignored rather than falling back to sklearn is a data-integrity failure worse than not accelerating. Avoid by: fail-closed capability gate per proxied estimator — unsupported config signals CPU sklearn fallback, never silent approximation. Install hook before any sklearn import; implement caller-module exclusion list; mirror sklearn's exact fitted-attribute names/shapes.

5. **Data-dependent memory blow-up on tree/ARIMA/SHAP structures (P6)** — naive full-histogram tensor `(nodes x features x bins x classes)`, full `batch x state^2` Kalman state, or `2^n` coalition enumeration fail the build-failing PoolStats gate. Avoid by: frontier-only histogramming; compact flat node arrays; tile over series batch for ARIMA; stream coalition blocks for Kernel SHAP; iterative `node_id` machine for FIL/TreeSHAP (no recursion — CubeCL kernels cannot recurse).

---

## Implications for Roadmap

Phase numbering continues from v3.0 (last = Phase 16). v4.0 starts at **Phase 17**.

### Phase 17: RandomForest GPU Histogram/Split Feasibility Spike (GATING)
**Rationale:** The make-or-break question — does the single-owner GATHER histogram lower under `cubecl-cpu` MLIR, and is it tractable? This gates the entire tree chain (RF -> FIL -> TreeSHAP). Must run first and deliver an explicit GO/ADJUST/ABORT verdict with abort signals A1-A5 evaluated. Models Phase 13 (KNN-graph keystone spike). Nothing in the tree family is committed until this answers GREEN.
**Delivers:** GATHER histogram + relabel partition + seed-from-first split-find kernels standalone-launching on cpu(f64) + rocm(f32); VALUE-asserting correctness test vs `sklearn.tree.DecisionTree*` on injected fixed bootstrap indices; per-tree cost benchmark; `SparseTreeNode` format contract finalized; two-tier stochastic gate convention established as a milestone-wide standard.
**Addresses:** RF table-stakes; node-format contract (hard dependency for FIL + TreeSHAP)
**Avoids:** P1 (atomic/SharedMemory histogram discovered mid-build); P2 (stochastic gate convention set here)
**Research flag:** NEEDS `/gsd-plan-phase --research-phase 17` — consult `Skill("spike-findings-mlrs")` for proven GATHER op-set and 002-A/002-B landmines.

### Phase 18: Tree Primitives + DecisionTree Core (standalone-validated)
**Rationale:** Primitive-first — hist/split/partition prims must be oracle-gated standalone vs a single sklearn `DecisionTree` BEFORE RF assembles many of them. Promotes spike's kernel probes to production prims with full prim contract.
**Delivers:** `quantiles`, `tree_hist`, `best_split`, `node_partition` prims in mlrs-backend; `tree.rs` in mlrs-kernels; `mlrs-algos/tree/` DecisionTree core (level-wise host loop); oracle-gated vs `sklearn.tree.DecisionTreeClassifier/Regressor` on injected fixed indices.
**Avoids:** Anti-pattern of building RF end-to-end before validating prims; P6 (frontier-only histogram memory gate established here)
**Research flag:** Standard primitive-first pattern after spike; skip research-phase.

### Phase 19: RandomForestClassifier + RandomForestRegressor
**Rationale:** Depends on Phase 18 prims. Full two-tier gate applies here for the first time against sklearn `RandomForest*`; OOB score, `feature_importances_`, and PyO3 shim land together.
**Delivers:** RF clf/reg estimators; PyO3 `ensemble.rs`; Python shim `mlrs/ensemble.py`; two-tier oracle gate; `feature_importances_` / `oob_score_` structural-gated.
**Avoids:** P2 (two-tier gate enforced, not exact-predicted-label against the ensemble)
**Research flag:** Skip research-phase — two-tier gate convention established in Phase 17.

### Phase 20: FIL — Batched Forest Inference
**Rationale:** Depends on node format (Phase 18) and a forest (Phase 19). The one tree-stack gate that is exact: device traversal must equal a CPU reference walk of the identical node arrays.
**Delivers:** `tree_traverse` prim; `mlrs-algos/fil/` batched inference (iterative `node_id` machine, no recursion); `predict`/`predict_proba`; exact gate vs host reference traversal (same arrays); PoolStats gate on row-streaming.
**Avoids:** P6 (recursive tree walk forbidden; stream output rows)
**Research flag:** Skip research-phase — deterministic traversal with clear oracle.

### Phase 21: TreeSHAP
**Rationale:** Depends on FIL/tree store (Phase 20). Deterministic — path-dependent Lundberg algorithm — so <=1e-5 gate applies, but ONLY against `shap.TreeExplainer` fed mlrs's own tree, NOT sklearn's forest. Additive-efficiency invariant is mandatory.
**Delivers:** `mlrs-algos/explainer/tree_shap`; `shap.TreeExplainer` oracle on mlrs trees; <=1e-5 + additive-efficiency gate; brute-force exact Shapley oracle on small hand-built trees.
**Avoids:** P2 (gated on mlrs's own tree, explicitly NOT sklearn's forest); P6 (iterative traversal, not recursive)
**Research flag:** Skip research-phase — deterministic algorithm with clear oracle.

### Phase 22: ARIMA / AutoARIMA
**Rationale:** Spike-independent — can run in parallel with the tree chain. New device kernels: batched Kalman sequential recursion and batched L-BFGS wrapper. Primitive-first: Kalman prim standalone-validated before ARIMA estimator consumes it.
**Delivers:** `kalman.rs` kernel; `batched_kalman.rs` prim; batched L-BFGS wrapper; `mlrs-algos/tsa/` ARIMA + AutoARIMA; `statsmodels` oracle — log-likelihood band + forecast band + known-coefficient recovery via Jones/PACF transform; order-selection gate for AutoARIMA.
**Avoids:** P3 (Jones/PACF transform mandatory — no raw-coefficient optimization; `statsmodels` oracle not `sklearn`; per-series convergence flags; f64 Kalman accumulation); P6 (tile over series batch)
**Research flag:** NEEDS `/gsd-plan-phase --research-phase 22` — Kalman ARIMA + Jones transform is novel for this codebase; batched L-BFGS convergence flag design needs care.

### Phase 23: Kernel SHAP + Permutation SHAP
**Rationale:** Model-agnostic — depends only on having a fitted estimator and reuses existing GEMM/lstsq prims. Spike-independent. No new device kernels beyond the existing predict path.
**Delivers:** `mlrs-algos/explainer/` kernel_shap + permutation_shap; `SHAPBase`; additive-efficiency invariant gate (exact); brute-force exact Shapley oracle for small `n`; convergence-band gate vs `shap` for large `n`; PyO3 `explainer.rs` + Python shim.
**Avoids:** P2 (sampling-based -> never element-wise vs `shap`; axiom gate is the correct contract); P6 (stream coalition blocks, never enumerate 2^n)
**Research flag:** Skip research-phase — SHAP axioms and weighted-lstsq are well-documented.

### Phase 24: sklearn-Utility Surface (metrics / preprocessing / model_selection)
**Rationale:** Spike-independent and foundational. Mostly host/reduction work; preprocessing scalers reuse the existing `reduce` prim. Degenerate fixtures (zero-variance column, empty class, constant target, single sample) are mandatory.
**Delivers:** `python/mlrs/metrics.py` (accuracy, confusion_matrix, r2, mse, mae, roc_auc, log_loss, precision_recall_curve); `python/mlrs/preprocessing.py` (Standard/MinMax/MaxAbs/Robust/Normalizer/Binarizer/OneHot/Ordinal/Label/LabelBinarizer/SimpleImputer); `python/mlrs/model_selection.py` (train_test_split, KFold, StratifiedKFold); MT19937-host-match decision for model_selection recorded.
**Avoids:** P5 (degenerate fixtures mandatory; MT19937-host match for split reproducibility; fit/transform statefulness enforced — stats learned only in `fit`, applied in `transform`)
**Research flag:** Skip research-phase — deterministic functions with well-defined sklearn contracts.

### Phase 25: Genetic / Symbolic Regression
**Rationale:** Spike-independent. Primitive-first: `program_eval` prim validated standalone before the evolutionary host loop. Property gate (R^2 band + valid program trees + internal seed-reproducibility) — never match gplearn's evolved expression.
**Delivers:** `program.rs` kernel; `program_eval.rs` prim; `mlrs-algos/genetic/` SymbolicRegressor (host evolutionary loop, device-evaluated fitness); PyO3 `genetic.rs` + Python shim; `gplearn` oracle (R^2 band + structural gate).
**Avoids:** P2 (stochastic — two-tier gate: internal seed-reproducibility + R^2-band vs gplearn; never element-wise expression match)
**Research flag:** Skip research-phase — gplearn API is well-documented; host-evolve/device-evaluate pattern is established.

### Phase 26: cuml.accel Drop-in (pure Python, last)
**Rationale:** Must land LAST because its value is proportional to the estimator surface it can proxy. After Phases 17-25 it can proxy the full v4.0 + v1-v3 surface (32 existing + new RF/ARIMA/SHAP/genetic estimators). Zero Rust changes. Fail-closed capability gate mandatory.
**Delivers:** `python/mlrs/accel/__init__.py` (install/uninstall), `_hook.py` (MetaPathFinder/AccelModule/caller-exclusion), `_overrides.py` (name->class override table); fallback-matrix test (every proxied estimator + unsupported-config -> CPU fallback); import-ordering test; fitted-attribute parity verification.
**Avoids:** P4 (fail-closed capability gate; caller-module exclusion list; detect-and-warn if sklearn already imported; exact fitted-attribute surface mirrored)
**Research flag:** Skip research-phase — sys.modules import-hook is standard-library Python; cuML's own `accel/` is the direct reference already read in-source.

### Phase Ordering Rationale

- Phase 17 gates Phases 18-21. The tree family is worthless if the histogram kernel cannot be made both correct AND tractable under cpu-MLIR. Spend one phase to answer cheaply (a handful of kernel probes) rather than discover the failure inside a half-built forest.
- Phase 18 before Phase 19 (primitive-first). Hist/split/partition prims validated standalone vs a single sklearn DecisionTree before RF assembles N of them. Failures are localizable; rework is cheap at the prim level.
- Phases 19 -> 20 -> 21 are a hard chain. FIL needs the node format (Phase 18) and a forest (Phase 19) to traverse. TreeSHAP needs FIL's tree store. These cannot be parallelized.
- Phases 22-25 are spike-independent and parallel-eligible. ARIMA, model-agnostic SHAP, sklearn-utility, and genetic have no dependency on the tree spike result. Each still obeys primitive-first internally.
- Phase 26 (cuml.accel) is last. Its override table is complete only when the entire estimator surface exists. Building it earlier means maintaining an incomplete proxy table.

### Research Flags

Phases needing deeper research during planning:
- **Phase 17 (RF feasibility spike):** Novel kernel authoring — consult `Skill("spike-findings-mlrs")` for the proven GATHER op-set, the 002-A (loud) / 002-B (silent) landmines, and the Phase 13 spike-as-model structure. GATHER histogram + relabel + seed-from-first argmax shapes need careful specification before any code is written.
- **Phase 22 (ARIMA/AutoARIMA):** Jones/PACF transform parameterization, Joseph-form stable Kalman, batched L-BFGS convergence flags, and `statsmodels` oracle matching strategy are domain-specific and not covered by existing project patterns.

Phases with standard patterns (skip research-phase):
- **Phase 18 (tree prims + DecisionTree):** Primitive-first pattern is established; spike delivers the kernel shapes.
- **Phase 19 (RF clf/reg):** Two-tier gate convention established in Phase 17.
- **Phase 20 (FIL):** Deterministic traversal; exact gate; no novel kernel shapes beyond what the spike proves.
- **Phase 21 (TreeSHAP):** Deterministic algorithm; oracle is `shap.TreeExplainer` on mlrs's own tree.
- **Phase 23 (Kernel/Permutation SHAP):** SHAP axioms and weighted-lstsq are well-documented.
- **Phase 24 (sklearn-utility):** Deterministic functions with exhaustively-documented sklearn contracts.
- **Phase 25 (symbolic regression):** `gplearn` API is stable; host-evolve/device-evaluate pattern is established.
- **Phase 26 (cuml.accel):** `importlib.abc` MetaPathFinder is standard-library Python; cuML's `accel/` is the direct reference.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new Rust compute deps confirmed from multiple independent angles; Python oracle versions verified against PyPI; hard pins (pyo3 0.28, cubecl 0.10.0) grounded in shipped decisions D-09 and D-07 |
| Features | HIGH | cuML reference source is in-tree; oracle/gate per feature grounded in prior milestone decisions (D-12, RandomProjection, exact-label vs property vs 1e-5); one confirmed PROJECT.md slip: ARIMA oracle is `statsmodels`, not `sklearn` |
| Architecture | HIGH (crate placement + build order); MEDIUM-HIGH (GPU-tree feasibility framing — analogous to Phase-13 pattern but tree-specific kernel shapes are unproven until the spike runs) |
| Pitfalls | HIGH | cpu-MLIR pitfalls grounded in shipped project memory and spike-findings-mlrs (002-A/002-B documented panics/miscompiles); stochastic-gate pitfall grounded in D-12 and UMAP/RandomProjection precedent; accel pitfall grounded in cuML's own UnsupportedOnGPU/ProxyBase design |

**Overall confidence: HIGH**

### Gaps to Address

- **RF GATHER histogram tractability (A3 abort signal):** Correctness of the GATHER inversion is well-grounded; the O(samples x bins) performance per (node, feature) is unknown until the spike benchmarks on a representative fixture. If A3 fires: fewer bins (64 default), shallower max-depth, frontier-only histogramming, or — last resort — drop RF/FIL/TreeSHAP from v4.0 and re-scope.
- **Batched L-BFGS vs host-batched for ARIMA:** ARCHITECTURE.md recommends host-batched (loop over the existing prim) as the starting point. The decision between host-batched and device-batched should be made early in Phase 22 planning based on representative series-count workloads.
- **ARIMA PROJECT.md slip — oracle assignment:** PROJECT.md lists "sklearn for ARIMA" (no ARIMA exists in sklearn). Must be corrected to `statsmodels.tsa` when REQUIREMENTS.md is written. `pmdarima` is NOT the oracle (absent from cuML test deps). AutoARIMA order-search reference = cuML's internal implementation tested against `statsmodels`; gate the selected (p,d,q) on synthetic series with known structure.
- **MT19937 host-match for model_selection:** Whether to implement an MT19937-compatible permutation in host Rust (so `train_test_split`/`KFold(shuffle=True)` indices match sklearn bit-for-bit) vs property-gate-only should be recorded as a decision in Phase 24 planning. Both are viable; the MT19937 match is higher-fidelity.
- **cuml.accel proxy coverage:** The `_overrides` table maps module-qualified names to mlrs classes. The exact set of proxied estimators and the supported-param space per capability gate should be drafted in Phase 26 planning against the then-complete estimator surface.

---

## Sources

### Primary (HIGH confidence)
- `.planning/research/STACK.md` — stack analysis, Python oracle versions, cuml.accel mechanism, hard dep pins
- `.planning/research/FEATURES.md` — oracle/gate per feature, API surface (constructor params/attrs), feature dependencies
- `.planning/research/ARCHITECTURE.md` — crate placement, build order, component responsibilities, data flow, anti-patterns
- `.planning/research/PITFALLS.md` — A1-A5 abort signals, two-tier stochastic gate, ARIMA numerical pitfalls, accel silent-wrong-results pitfall, memory blow-up patterns
- `.planning/PROJECT.md` — milestone scope, constraints, key decisions (D-07, D-09, D-12), Core Value
- `cuml-main/python/cuml/cuml/{accel,ensemble,fil,tsa,explainer,metrics,preprocessing,model_selection,feature_extraction,genetic}/` — RAPIDS cuML v26.08 reference behavior (read-only, in-tree)
- `cuml-main/cpp/src/decisiontree/batched-levelalgo/builder_kernels_impl.cuh` — atomic+SharedMemory histogram (confirms GATHER inversion need)
- `cuml-main/cpp/include/cuml/tree/flatnode.h` — SparseTreeNode format, right=left+1 convention
- `crates/*/Cargo.toml` + workspace `Cargo.toml` — confirmed zero new deps, pyo3 0.28, cubecl 0.10.0 pins
- `.planning/milestones/v3.0-phases/13-knn-graph-primitive-feasibility-keystone/` — Phase 13 spike model
- `Skill("spike-findings-mlrs")` + `references/cpu-mlir-kernel-authoring.md` — proven cpu-MLIR GATHER op-set; 002-A/002-B landmines

### Secondary (MEDIUM confidence)
- PyPI (2026-06-26): `gplearn 0.4.3`, `shap 0.52.0`, `statsmodels 0.14.4` — versions verified
- Project MEMORY.md — cpu-MLIR no-SharedMemory/atomics, rocm f64-unsupported, oracle-venv, disk/suite-slowness landmines
- Jones (1980) PACF stationarity transform; Lundberg et al. TreeSHAP algorithm — established domain knowledge
- `.planning/notes/v3-hard-algorithm-backlog.md`, `notes/cuml-mlrs-gap-inventory.md` — dependency ordering, tree feasibility flags

---
*Research completed: 2026-06-26*
*Ready for roadmap: yes*
