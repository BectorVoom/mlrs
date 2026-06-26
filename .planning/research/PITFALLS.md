# Pitfalls Research

**Domain:** Tree ensembles (RF→FIL→TreeSHAP), time-series (ARIMA), model-agnostic explainers (Kernel/Permutation SHAP), genetic/symbolic regression, sklearn-utility surface, and a `cuml.accel` import-hook — all added to an existing CubeCL/cpu-MLIR-gated, scikit-learn-oracled Rust system (mlrs v4.0)
**Researched:** 2026-06-26
**Confidence:** HIGH for the cpu-MLIR / RNG-mismatch / oracle-gate pitfalls (grounded in shipped project decisions D-07/D-12, spike findings 001/002, and project MEMORY landmines); MEDIUM for algorithm-specific numerical claims (ARIMA Kalman conditioning, SHAP axioms, gplearn behavior — established domain knowledge, not re-verified in-repo this session)

> **How to read this file.** These are not generic ML pitfalls. Every entry is the specific way one of the v4.0 features breaks *against mlrs's three hard constraints*: (1) cpu-MLIR kernels with **no SharedMemory / no cross-unit atomics / no `F::INFINITY`** (the f64 gate), (2) **SplitMix64 RNG ≠ NumPy MT19937**, so nothing RNG-dependent can match sklearn/shap/gplearn element-wise, and (3) the **scikit-learn ≤1e-5 oracle** which only applies to deterministic stages. The two highest-stakes entries — **Pitfall 1 (tree-construction feasibility)** and **Pitfall 2 (stochastic-oracle gate)** — are expanded in depth as the prompt requires.

---

## Critical Pitfalls

### Pitfall 1: GPU tree construction assumes atomics/SharedMemory the cpu-MLIR gate forbids — and the team discovers it mid-build instead of in the spike

**What goes wrong:**
The textbook GPU decision-tree builder (cuML's, XGBoost's, every CUDA reference) accumulates a per-node **histogram** of `(feature, bin) → {class counts | gradient sums}` by having every sample *scatter-add* into a shared-memory histogram with `atomicAdd`, then scans the histogram to find the best-gain split. Both halves of that — **SharedMemory histograms and cross-unit atomic accumulation** — are exactly the two things cubecl-cpu MLIR panics on at launch (project MEMORY: `cubecl-cpu-no-shared-memory`; D-CONTEXT cpu-MLIR landmines). A team that ports the standard kernel will get a launch panic on the f64 gate and then try to "fix" it under deadline pressure inside the RF estimator phase, where the feasibility question should already have been answered.

**Why it happens:**
Histogram-based tree construction is *the* canonical GPU ML kernel; it is what `cuml-main/cpp/src/` does. The scatter-add-with-atomics pattern is so standard that it is invisible — nobody flags it as a risk until the MLIR pass fails. Worse, the data-dependent control flow (tree depth, node count, sample→node partition) is unknown at compile time, so the kernel author reaches for the most flexible primitive (atomics) by reflex.

**How to avoid — the GATHER redesign, and why it is feasible:**
Invert scatter→gather, exactly as the KNN spike inverted the distance/top-k path. The redesign has four device steps, each of which must be proven to lower under `--features cpu` *in the spike, standalone, before any estimator work*:
1. **Histogram by single-owner GATHER, not scatter.** Each output cell `(node, feature, bin)` is owned by one unit; that unit loops over all samples, and for each sample checks "does this sample belong to my node AND fall in my bin?" via an `if`-guarded `u32` accumulator read **within the same loop** (the proven spike-001 feature-loop accumulator shape). No atomics: each cell has exactly one writer. The cost is `O(samples × bins)` per `(node, feature)` instead of `O(samples)` with atomic scatter — this is the price of the constraint and the thing the spike must *measure*, not just *compile*.
2. **Sample→node assignment by RELABEL, not compaction.** Partitioning samples to child nodes classically needs a stable partition = a prefix-sum/scan, which has no SharedMemory-free, atomic-free expression. **Avoid the scan entirely:** keep a `sample → node_id` label array; each level, each sample (owned by one unit) reads the split for its current node and overwrites its own `node_id` with the child id. Pure GATHER, no scan, no compaction. The histogram kernel in step 1 filters by `node_id`. This is the single insight that makes RF feasible under cpu-MLIR — *do not physically reorder samples*.
3. **Split-find argmax WITHOUT `F::INFINITY`.** Scanning bins for the best information gain is an argmax; the reflexive init is `best_gain = -INFINITY`, which is **banned** (panics at launch, project MEMORY). Seed from the first candidate (the Chebyshev seed-from-first / statement-form-`if` running-max idiom proven in spike 001), or a computed finite sentinel. Never an `if`-expression in value position.
4. **Per-level host orchestration is allowed; per-level device atomics are not.** The node frontier is data-dependent, so the host loop (one iteration per tree level) that launches the per-level histogram + split-find + relabel kernels and reads back the chosen splits is fine — that is host code, not a device kernel. Recursion is *not* available inside a kernel (CubeCL kernels cannot recurse); the tree walk must be an explicit iterative `node_id` machine.

**Abort signals — when the spike must say "infeasible, adjust scope" (this is the spike's whole job, à la Phase 13):**
The spike builds *one node's* histogram + split-find on a fixed fixture and benchmarks it. Abort / re-scope if ANY of these fire:
- **A1 — Lowering fails:** the GATHER histogram kernel (nested sample-loop with `node_id` filter and `bin` filter) trips an MLIR pass failure that no statement-form / loop-restructuring rewrite resolves. *(Low risk — feature-loop accumulators are spike-001-proven — but it is the gating compile check.)*
- **A2 — Quantile/binning needs an unlowering sort:** computing bin edges (feature quantiles) requires a device sort or percentile that won't lower cpu-MLIR-safe **and** cannot be acceptably moved to a host pre-pass. *(Mitigation first: compute quantile bin edges on the host once per fit — almost certainly acceptable. Abort only if binning must be on-device AND needs scan/sort.)*
- **A3 — Correct but superlinear-cost:** the GATHER histogram is correct on the fixture but, on a representative load (e.g. ~1000 samples × ~20 features × 128 bins × depth 8), wall-clock per tree exceeds a budget that makes a realistic forest (≥100 trees) impractical on the cpu/rocm gate. The `O(samples × bins)` factor is the suspect. *Re-scope options:* fewer bins, shallower max-depth default, histogram only over the active frontier, or — last resort — defer RF/FIL/TreeSHAP again and ship the non-tree v4.0 surface.
- **A4 — Split-find can't argmax safely:** the gain argmax cannot be expressed without `F::INFINITY` init or a cross-sibling-loop accumulator (spike-002-B silent-miscompile pattern). *(Low risk — seed-from-first is proven — but verify with a VALUE assertion, not non-panic.)*
- **A5 — Correctness fail:** end-to-end, a single tree built on a **fixed, injected bootstrap-index set** does NOT reproduce `sklearn.tree.DecisionTree*`'s split on the same data/criterion. This is the real correctness witness (see Pitfall 2) — if it fails, the histogram/gain math is wrong, not the RNG.

**Warning signs:**
Launch panic mentioning MLIR pass / "block successors must terminate"; output histograms read back as all-zeros (the 002-A symptom — kernel never ran); a histogram kernel that is correct on a single node but whose benchmark grows quadratically as bins increase; any urge to "just use a small SharedMemory scratch for the histogram."

**Phase to address:**
**The RandomForest feasibility spike — the FIRST v4.0 phase, gating, à la Phase 13.** It must deliver: the GATHER histogram + relabel + split-find kernels standalone-launching on `--features cpu` (f64) and `--features rocm` (f32, f64 skips-with-log), a VALUE-asserting correctness test against sklearn DecisionTree on injected fixed bootstrap indices, a cost benchmark, and an explicit GO / ADJUST / ABORT verdict with A1–A5 evaluated. RF/FIL/TreeSHAP phases are contingent on its GO.

---

### Pitfall 2: Gating stochastic estimators (RF / SHAP-sampling / genetic) element-wise against sklearn/shap/gplearn — guaranteed false failures, because SplitMix64 ≠ MT19937

**What goes wrong:**
RandomForest (bootstrap sampling + per-split feature subsampling), Kernel/Permutation SHAP (coalition/permutation sampling), and symbolic regression (random initial populations + random genetic operators) are all **RNG-driven**. mlrs uses **SplitMix64**; sklearn/shap/gplearn use **NumPy MT19937**. Even with the "same" seed the draws differ, so the bootstrap samples differ, so the trees differ, so the *predictions* differ — there is no element-wise (≤1e-5) match to be had against the reference library, and there is **no exact-label match** either. A team that applies mlrs's existing gates naively will either (a) wire RF classifiers to the **exact-predicted-label gate** (which was decided for *deterministic* solvers SGD/SVM/NB, D-12) and watch it fail spuriously, or (b) loosen to "accuracy within some band" and pick a band so wide it hides real split-math bugs.

**Why it happens:**
The project's two shipped correctness gates are seductive and both wrong here: ≤1e-5 value match (deterministic numeric paths) and exact-predicted-label (deterministic classifiers). RandomForest *is a classifier*, so the exact-label gate looks like it should apply — but RF is stochastic, and that gate was explicitly scoped to deterministic solvers. The fix is the same split the project already made for UMAP/RandomProjection (D-12): **decouple the deterministic core from the RNG-dependent wrapper and gate each differently.**

**How to avoid — the correct gate per feature:**

- **RandomForest — two-tier gate.**
  1. *Deterministic-core tier (tight, the real correctness witness):* feed an **identical, injected set of bootstrap-sample indices and feature-subset indices** into both mlrs's single-tree builder and `sklearn.tree.DecisionTreeClassifier/Regressor`, and require the tree structure / splits / predictions to match (exact labels for classification, ≤1e-5 for regression leaf values). This isolates the histogram/split math from the RNG and is where bugs actually surface.
  2. *Ensemble tier (banded, RNG-tolerant):* on held-out data, gate **prediction-agreement / score parity** — accuracy (classification) or R² (regression) within a documented band of sklearn's `RandomForest*` across several datasets — plus **feature-importance rank/distribution similarity** and an **OOB-error sanity check**. Never element-wise; never exact-label against the *ensemble*.
- **Kernel / Permutation SHAP — axiom + exact-enumeration gate.** SHAP sampling never matches `shap`'s sampled values. Gate instead on the **deterministic Shapley axioms** that hold regardless of sampling: **efficiency/local-accuracy** (Σ SHAP values + base value == model output) to ≤1e-5 *exactly*; symmetry; dummy. For **small feature counts (≲10, so 2^n coalitions enumerable)** compute **exact Shapley by brute-force enumeration** in-test and gate mlrs ≤1e-5 against that ground truth (NOT against `shap`). For large problems, gate convergence: mlrs's many-sample estimate within a variance band of `shap`'s many-sample estimate.
- **TreeSHAP — deterministic, but gate on mlrs's OWN tree.** TreeSHAP is *exact and sampling-free*, so it *can* be gated ≤1e-5 — but only against a reference fed the **same tree**. Gating `shap.TreeExplainer(sklearn_model)` is wrong because mlrs's forest ≠ sklearn's forest (RNG). Gate the **efficiency axiom** on mlrs's tree, and brute-force exact Shapley on **small mlrs trees**.
- **Symbolic regression (gplearn) — function-recovery + score band + internal reproducibility.** Never match gplearn's evolved expression. Gate: (a) on problems with a recoverable closed form (e.g. `x0*x1 + x2`), the discovered program achieves **R² ≥ threshold**; (b) final-program fitness within a band of gplearn across seeds/datasets; (c) **internal seed-reproducibility** — same mlrs seed ⇒ byte-identical population evolution (the RandomProjection D-12 reproducibility contract).

**Warning signs:**
A classifier oracle test that is green on `LogisticRegression` but red on `RandomForest` with "labels differ at N positions"; a SHAP test whose pass/fail flips when you change `n_samples`; any test comparing mlrs tree structure directly to a sklearn-fitted forest; a "band" gate that still passes when you deliberately corrupt the gain computation (band too loose — add the injected-index deterministic-core tier).

**Phase to address:**
Decided in the **RandomForest spike phase** (establish the two-tier gate as a milestone-wide convention) and applied in **every stochastic phase**: RF estimators, Kernel/Permutation SHAP, symbolic regression. Each phase's verification must explicitly state which tier(s) it uses and why element-wise/exact-label does not apply. TreeSHAP phase must state it gates on mlrs's own tree, not sklearn's.

---

### Pitfall 3: ARIMA's Kalman filter / L-BFGS wanders into non-stationary or ill-conditioned regions → NaN likelihood, and coefficients are gated against statsmodels they were never going to match

**What goes wrong:**
Three coupled failures: (1) the Kalman filter state covariance loses positive-definiteness under f32 roundoff (the rocm gate is f32), the predicted innovation variance goes ≤0, and `log(variance)` → NaN, poisoning the whole batched log-likelihood. (2) Batched L-BFGS optimizing raw AR/MA coefficients **escapes the stationary/invertible region** (AR roots inside the unit circle, MA invertible), where the likelihood is undefined → NaN gradients → the optimizer stalls or diverges, and in a *batch* one bad series can stall the shared iteration. (3) Even when numerically healthy, the team gates ARIMA coefficients ≤1e-5 against **statsmodels** — but the ARIMA likelihood is multimodal, statsmodels and mlrs find different local optima, and the coefficients legitimately differ.

**Why it happens:**
ARIMA is the first time-series algorithm in mlrs; the existing L-BFGS prim (`prims/lbfgs.rs`) is single-objective, not batched, and was validated on convex-ish losses (LogReg) where the parameter space is unconstrained. The AR/MA stationarity constraints are a domain subtlety with no analogue in the v1–v3 surface. And statsmodels is the natural oracle, so coefficient-matching is the reflexive gate.

**How to avoid:**
- **Optimize in unconstrained space via the Jones (1980) / partial-autocorrelation transform.** Parameterize AR/MA through reflection coefficients mapped through `tanh`-style transforms so that *every* point the optimizer visits maps to stationary/invertible parameters. The likelihood is then defined everywhere and L-BFGS cannot escape. This is what statsmodels/cuML do; do not optimize raw coefficients with hard constraints.
- **Use a numerically stable Kalman form and accumulate in f64.** Joseph-form or square-root covariance update; clamp innovation variance to `≥ ε`; accumulate the log-likelihood in f64 even on the f32 rocm path (the filter is sequential and roundoff-sensitive). Watch the cpu-MLIR landmines if any filter step becomes a kernel: no `F::INFINITY` sentinels in the diffuse-initialization, statement-form conditionals only.
- **Batch L-BFGS with per-series convergence, not a shared stop.** Each series gets its own convergence flag; converged series stop contributing gradient; the batch ends when all are done or max-iters hit. One non-converging series must not stall the batch or NaN-poison its neighbors (mask NaN series out).
- **Gate on likelihood / forecast / parameter-recovery, not coefficients.** Oracle = `statsmodels` (ARIMA is not in sklearn). Gate: log-likelihood within a band (mlrs ≥ statsmodels − tol), out-of-sample forecasts within a band, and on **simulated data with known coefficients**, parameter recovery within the sampling CI. For **AutoARIMA**, the order search is discrete: gate that the selected `(p,d,q)` matches statsmodels'/`pmdarima` OR that the chosen model's AIC/BIC is within tolerance of the reference's.

**Warning signs:**
Log-likelihood prints `NaN`/`-inf` after a few L-BFGS steps; gradients explode near the parameter-space boundary; one series in a batch derails the rest; coefficient comparison to statsmodels fails by >>1e-5 while forecasts are visually identical (the multimodal-optimum tell).

**Phase to address:**
**ARIMA / AutoARIMA phase.** Extend `prims/lbfgs.rs` to a batched form; add the Jones transform and stable Kalman in the algos layer; establish the likelihood/forecast/recovery gate (not coefficient match). Memory gate: tile over the series batch (see Pitfall 6).

---

### Pitfall 4: `cuml.accel` silently returns wrong results instead of falling back to CPU sklearn

**What goes wrong:**
The accel layer swaps `sklearn.X` for an mlrs proxy at import time. When the user constructs an estimator with a parameter/config mlrs **doesn't support** (sparse input, a callable metric, `sample_weight` mlrs ignores, multi-output, a non-default solver, a dtype outside f32/f64, a class-weight scheme), a naive proxy *silently ignores the unsupported option and returns a plausible-but-wrong result*. The user believes they ran their sklearn code "but faster"; they actually ran a different model. This is strictly worse than not accelerating — it is silent data corruption. A second failure mode: the import hook is installed **after** the user's code already imported sklearn, so the proxy never takes effect (silent no-op), or the proxy's own internal sklearn use recurses through itself.

**Why it happens:**
The whole value proposition of `cuml.accel` is "transparent — change nothing." That pressure pushes toward *never erroring*, which becomes *never falling back loudly*. cuML solved this with explicit `UnsupportedOnGPU` exceptions that trigger CPU fallback (CLAUDE.md: `UnsupportedOnGPU`/`UnsupportedOnCPU` + `ProxyBase`), but a fresh implementation tends to omit the capability gate. Import-hook ordering is a classic Python `sys.modules` trap.

**How to avoid:**
- **Capability gate is mandatory and fail-closed.** Per proxied estimator, explicitly enumerate the supported parameter space. ANY unsupported param/config/dtype/sparsity MUST raise an internal "unsupported" signal that triggers **CPU-sklearn fallback** — never silently ignore. Configs that MUST fall back: sparse input, custom/callable metrics, unsupported `sample_weight`, multi-output, params whose mlrs default/semantics differ, dtypes ≠ f32/f64, anything the matching mlrs estimator does not implement.
- **Install the hook before any sklearn import.** Prefer `python -m mlrs.accel script.py` (install at interpreter start); if an in-process `install()` is offered, document that it must run before `import sklearn` and detect/warn if sklearn is already imported. The proxy's own internal sklearn usage must reach the **real** sklearn (caller-module exclusion list — CLAUDE.md anti-pattern "Direct sklearn import inside cuml.accel scope").
- **Fitted-attribute parity.** The proxy must expose sklearn's exact fitted-attribute names/shapes (`coef_`, `feature_importances_`, `n_iter_`, …); missing attributes break downstream user code.
- **Document the approximation contract.** For stochastic estimators (RF), accel results differ from pure sklearn within a band (Pitfall 2) — accel is "same within band," not bit-identical. State it.

**Warning signs:**
An accel run that produces results *close to but not matching* a known sklearn baseline and never logs a fallback; `install()` having no effect because sklearn was imported first; infinite recursion / stack overflow on import (proxy calling itself); user code reading `est.some_attr_` and getting `AttributeError`.

**Phase to address:**
**cuml.accel phase** (last in the milestone, after the estimator surface exists to proxy to). Verification must include a **fallback matrix**: for each proxied estimator, a test that an unsupported config triggers CPU fallback (not a silent wrong answer), plus an import-ordering test.

---

### Pitfall 5: sklearn-utility surface looks trivial, then fails on the edge cases that ARE the spec (zero-division, multiclass averaging, fit/transform statefulness, split RNG)

**What goes wrong:**
metrics/preprocessing/model_selection feel like "easy non-device glue," so they get happy-path implementations that diverge from sklearn precisely on the degenerate inputs sklearn carefully defines:
- **Metrics:** precision/recall/F1 with no predicted positives (sklearn returns 0 *and* honors the `zero_division` param); multiclass `average` ∈ {micro, macro, weighted, samples} each computed differently; `r2_score` with constant `y`; `log_loss` probability clipping; `roc_auc` with a single present class.
- **Preprocessing:** `StandardScaler` on a **zero-variance** column (sklearn sets `scale_ = 1`, does not divide by 0); computing stats in `transform` instead of `fit` (data leakage); `partial_fit` running stats; `with_mean` on sparse.
- **model_selection:** `train_test_split` / `KFold(shuffle=True)` / `cross_val_score` shuffle with **NumPy MT19937**; mlrs SplitMix64 produces *different fold assignments*, so users lose split-reproducibility against their sklearn pipelines.

**Why it happens:**
These are deterministic and look like one-liners, so they skip the rigorous oracle treatment the device estimators got. But the edge cases are the whole point of sklearn's metric/preprocessing contracts.

**How to avoid:**
- **Oracle every metric/preprocessor against sklearn including degenerate fixtures** — all-same-label, empty class, single sample, zero-variance column, constant target. These are deterministic, so the **≤1e-5 (or exact for integer/label metrics) gate fully applies** — no banding. Port sklearn's exact edge handling (e.g. `scale_ = 1` where `var == 0`; `zero_division` semantics).
- **Enforce fit/transform statefulness.** Stats learned only in `fit`/`partial_fit`; `transform` applies them to (possibly different) data. Oracle = fit on train, transform on a *separate* test set, compare to sklearn doing the same.
- **For model_selection splitting, match NumPy exactly on the host.** Splitting/shuffling needs no device RNG — implement an **MT19937-compatible permutation in host Rust** so fold assignments match sklearn bit-for-bit and reproducibility holds. (This is the one place to deliberately *not* use SplitMix64.) Fallback if MT19937-matching is descoped: property-gate (folds disjoint, union complete, sizes correct, stratification preserves class ratios) **plus internal seed-reproducibility**, and document the divergence.

**Warning signs:**
A metric green on balanced multiclass but red/`NaN` on a class with no samples; a scaler emitting `inf`/`NaN` on a constant feature; `KFold` indices not matching a sklearn baseline; reviewers calling these "too simple to need oracle fixtures."

**Phase to address:**
**sklearn-utility phase** (metrics / preprocessing / feature_extraction / model_selection). Verification = per-function sklearn oracle with an explicit degenerate-fixture set; an MT19937-host-permutation decision recorded for model_selection.

---

### Pitfall 6: Memory blows the per-phase gate on data-dependent structures — tree node/histogram tensors, batched Kalman state, coalition enumeration

**What goes wrong:**
The v4.0 features all have **data-dependent or combinatorial intermediate sizes** that the v1–v3 fixed-shape memory discipline didn't have to handle, and the naive layout makes `peak_bytes`/`live_bytes` super-linear, failing the build-failing PoolStats gate:
- **RF:** materializing the full `(nodes × features × bins × classes)` histogram tensor resident at once, or pre-allocating `2^max_depth` node arrays.
- **ARIMA:** holding the full batch of Kalman state-covariance matrices (`batch × state²`) resident.
- **Kernel SHAP:** enumerating `2^n` coalitions for large `n` (the reason sampling exists) materialized as a matrix.
- **FIL:** per-row × per-tree intermediate scores held resident across the whole forest.

**Why it happens:**
The shapes are unknown at compile time, so the reflex is "allocate the max and fill it," and the combinatorial cases (`2^depth`, `2^n`) look fine on toy fixtures and explode on realistic sizes.

**How to avoid:**
Tile/stream over the data-dependent axis, exactly as Phase 13 tiled the query axis (D-CONTEXT criterion):
- **RF:** histogram only the **active frontier** of nodes per level; release each level's scratch via the `release_into(pool)` precedent before the next; never hold the whole tree's histograms. Store the tree as compact flat node arrays sized to *actual* node count, not `2^depth`.
- **ARIMA:** tile over the **series batch**; hold only a working block of Kalman states resident.
- **Kernel SHAP:** **never enumerate `2^n`** except for the small-`n` exact-Shapley oracle; the production path samples coalitions in streamed blocks.
- **FIL:** the per-output-row tree walk is GATHER-friendly (one unit per row, iterative `node_id` machine, runtime `while` bounded by depth — **no recursion**, which CubeCL kernels cannot do); stream output rows, don't hold per-tree intermediates.
- Add a **build-failing PoolStats gate per phase** asserting the scratch is bounded by the active working set (frontier nodes / series block / sample block), not the combinatorial total — and assert it stays sub-quadratic as the data-dependent dimension grows.

**Warning signs:**
`peak_bytes` growing as `2^depth`, `batch × state²`, or `2^n` in the memory test; OOM only on realistic (not toy) fixtures; a kernel author reaching for recursion in the FIL/TreeSHAP tree walk.

**Phase to address:**
Each owning phase: RF spike + RF estimators (histogram/node memory), ARIMA (batched state), Kernel SHAP (coalition streaming), FIL (row streaming). Every phase keeps the build-failing PoolStats gate it inherited from v1–v3 — never deferred.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Use a tiny `SharedMemory` histogram "just for the spike" | Kernel compiles like the CUDA reference | Panics on the f64 cpu-MLIR gate; invalidates the whole feasibility verdict | **Never** — the spike's entire purpose is to prove the SharedMemory-free path |
| Gate RF classifier with the existing exact-predicted-label gate | Reuses shipped harness | Spurious red on every run (RNG differs); masks/encourages loosening that hides real bugs | **Never** — use the two-tier injected-index + score-band gate |
| Gate TreeSHAP against `shap.TreeExplainer(sklearn_model)` | One-line oracle | Compares against a *different* forest; fails or falsely passes | **Never** — gate on mlrs's own tree (efficiency axiom + small-tree brute force) |
| Optimize raw ARIMA AR/MA coefficients with clamping | Skips the Jones transform | Optimizer escapes the stationary region → NaN likelihood | **Never** for the shipped path; ok only in a throwaway probe |
| `cuml.accel` ignores an unsupported param instead of falling back | "Transparent," never errors | Silent wrong results — the worst possible failure | **Never** — fail-closed to CPU sklearn |
| Host MT19937-match deferred; SplitMix64 for `train_test_split` | Reuses device RNG | Users can't reproduce their sklearn split indices | Only if documented + property-gated + internally seed-reproducible |
| Pre-allocate `2^max_depth` node arrays | Simple fixed shape | Fails the memory gate on deep trees | Only for tiny fixed-depth toy demos, never the estimator |
| Skip degenerate fixtures for "trivial" metrics | Faster phase | Diverges from sklearn on exactly the contract edge cases | **Never** — the edges are the spec |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| cubecl-cpu (MLIR, f64 gate) | Port the atomic/SharedMemory histogram kernel | Single-owner GATHER histogram; relabel (not scan) for partition; seed-from-first (not `F::INFINITY`) for argmax |
| scikit-learn oracle | Element-wise gate on RNG-dependent ensemble | Two-tier: injected-index deterministic core (tight) + score band (RNG-tolerant) |
| `shap` library | Match sampled SHAP values numerically | Efficiency axiom ≤1e-5 + brute-force exact Shapley on small `n`; convergence band for large `n` |
| `statsmodels` (ARIMA oracle) | Gate AR/MA coefficients ≤1e-5 | Gate log-likelihood / forecast / known-coefficient recovery (multimodal optima differ) |
| `gplearn` (symbolic-reg oracle) | Match the evolved expression | Function-recovery R² + fitness band + internal seed-reproducibility |
| NumPy MT19937 vs mlrs SplitMix64 | Assume same seed ⇒ same draws | Decouple deterministic core from RNG wrapper; for model_selection, MT19937-match on host |
| Python `sys.modules` (accel hook) | `install()` after sklearn already imported | Install at interpreter start (`-m`); caller-module exclusion; detect-and-warn if late |
| PyO3 fitted attributes (accel) | Omit sklearn attribute names/shapes | Mirror `coef_`/`feature_importances_`/`n_iter_` exactly; missing attrs break user code |
| Oracle fixture regen (RF/ARIMA/SHAP) | Regenerate in-repo Python | Needs a `/tmp` venv with numpy/sklearn/statsmodels/shap/gplearn (PEP 668); fixtures are committed `.npz` blobs |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| GATHER histogram `O(samples × bins)` per `(node,feature)` | Per-tree build time grows with bin count | Cap bins (e.g. 64–128); frontier-only histogramming; benchmark in the spike | Many bins × deep trees × many features → abort-signal A3 |
| `2^n` coalition enumeration (Kernel SHAP) | OOM / hang as feature count rises | Sample coalitions in streamed blocks; enumerate only for the small-`n` oracle | n ≳ 15–20 features |
| Full-batch Kalman state resident (ARIMA) | `peak_bytes` ~ `batch × state²` | Tile over series batch | Large series batches |
| Recursive tree walk (FIL/TreeSHAP) | Won't compile (CubeCL no recursion) / stack issues on host | Iterative `node_id` machine, runtime-`while` bounded by depth | Any non-trivial tree depth |
| Full `n×n`-style intermediate (carried from v1–v3) | Quadratic `peak_bytes` | Tile over the data-dependent axis; `release_into(pool)` per block | Realistic (non-toy) data sizes |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Unvalidated geometry/hyperparams before `unsafe` device launch (tree dims, `(p,d,q)`, n_bins, n_estimators) | Out-of-bounds device read (Tampering/DoS) | Host-side validation returning typed errors BEFORE any launch (the `distance.rs`/`topk.rs` precedent), incl. overflow-`u32` dim checks |
| `cuml.accel` executing untrusted estimator config without capability bounds | Silent wrong results = data-integrity (Repudiation) | Fail-closed capability gate → CPU fallback; never silently ignore params |
| Silent kernel miscompile in tree/histogram returns plausible-wrong data | Data integrity | VALUE-asserting oracle on an adversarial fixture (duplicate/degenerate rows), not non-panic — the spike-002-B lesson applies to histograms too |
| `ArrayArg::from_raw_parts` length mismatch on dynamically-sized node/histogram buffers | Tampering | Pass validated element counts only; kernels bounds-check every index |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| `cuml.accel` gives wrong-but-close results with no signal | User trusts a silently different model | Log every CPU fallback; document the within-band approximation for stochastic estimators |
| RF predictions differ from sklearn and users assume a bug | Lost confidence | Document up front: stochastic ⇒ score-parity within a band, not bit-identical (RNG differs) |
| `train_test_split` indices don't match the user's sklearn pipeline | Broken reproducibility across a mixed pipeline | MT19937-match on host so split indices are identical |
| ARIMA coefficients differ from statsmodels | "Which one is right?" | Document multimodal optima; surface log-likelihood/forecast parity as the contract |
| Missing fitted attribute on an accel-proxied estimator | `AttributeError` deep in user code | Mirror sklearn's exact fitted-attribute surface |

## "Looks Done But Isn't" Checklist

- [ ] **RF histogram kernel:** compiles AND *launches* on `--features cpu` (f64) — verify it isn't the 002-A "reads back zeros, never ran" symptom; assert VALUES on an adversarial fixture.
- [ ] **RF spike:** delivered an explicit GO/ADJUST/ABORT verdict with A1–A5 evaluated AND a cost benchmark — not just "the kernel compiled."
- [ ] **RF classifier gate:** uses the injected-fixed-bootstrap-index deterministic-core tier, not only an ensemble score band (the band alone can pass with corrupted gain math).
- [ ] **SHAP:** efficiency axiom asserted ≤1e-5 AND a brute-force exact-Shapley oracle on a small problem — sampling-vs-`shap` comparison alone proves nothing.
- [ ] **TreeSHAP:** gated on mlrs's own tree, not `shap.TreeExplainer(sklearn_model)`.
- [ ] **ARIMA:** Jones/PACF transform in place (optimizer can't leave the stationary region); likelihood accumulated in f64; gate is likelihood/forecast/recovery, not coefficients.
- [ ] **AutoARIMA:** order-selection gate (selected `(p,d,q)` or AIC within tol), not just a single fitted model.
- [ ] **cuml.accel:** a fallback-matrix test proves unsupported configs hit CPU sklearn (not silent wrong answers) + an import-ordering test.
- [ ] **Metrics/preprocessing:** degenerate fixtures (zero-variance, empty class, single sample, constant target) oracled — not just balanced happy-path data.
- [ ] **model_selection:** MT19937-host-match decision recorded; split reproducibility verified against a sklearn baseline.
- [ ] **FIL/TreeSHAP traversal:** iterative `node_id` machine (no recursion); memory gate bounded by frontier/rows, not `2^depth`.
- [ ] **Every kernel:** no `SharedMemory`/`Atomic`/`F::INFINITY`/mutable-bool/shift-loop; argmax via seed-from-first statement-form `if`; static `F::powf` (never instance `.powf()`).

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Tree histogram needs atomics/SharedMemory (discovered in estimator phase, not spike) | HIGH | Stop; redo the spike's GATHER redesign standalone; if A1–A5 fail, re-scope (fewer bins / shallower / defer RF→FIL→TreeSHAP) — exactly the abort path the spike exists to find cheaply |
| Stochastic estimator gated element-wise, perpetual red | LOW | Swap to two-tier gate (injected-index core + score band); re-derive fixtures from sklearn with fixed indices |
| ARIMA NaN likelihood | MEDIUM | Add Jones/PACF transform + stable Kalman + f64 accumulation; mask NaN series in the batch |
| accel silently wrong | MEDIUM | Add fail-closed capability gate + fallback matrix test; audit each proxied estimator's supported param space |
| Memory gate red on data-dependent structure | MEDIUM | Tile over frontier/batch/sample axis; `release_into(pool)` per block; size arrays to actual, not max |
| Metric diverges on edge case | LOW | Port sklearn's exact `zero_division`/`average`/zero-variance handling; add the degenerate fixture |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1 — Tree construction needs atomics/SharedMemory | **RF feasibility spike (FIRST, gating)** | GATHER histogram/relabel/split-find launch on cpu(f64)+rocm(f32); VALUE-assert vs sklearn DecisionTree on injected indices; cost benchmark; A1–A5 verdict |
| 2 — Stochastic element-wise gate | RF spike (set convention) → RF estimators, Kernel/Permutation SHAP, symbolic-reg | Injected-index deterministic-core tier + score/axiom band; explicit "why not element-wise" statement |
| 3 — ARIMA numerical stability / wrong gate | ARIMA / AutoARIMA phase | Jones transform + stable f64 Kalman + batched L-BFGS; likelihood/forecast/recovery gate; order-selection gate |
| 4 — accel silent wrong results | cuml.accel phase (last) | Fallback matrix per estimator; import-ordering test; fitted-attribute parity |
| 5 — sklearn-utility edge cases | sklearn-utility phase | Degenerate-fixture oracle (≤1e-5/exact); fit/transform statefulness; MT19937-host split match |
| 6 — Data-dependent memory blow-up | RF spike + RF + ARIMA + Kernel SHAP + FIL | Build-failing PoolStats gate bounded by frontier/batch/rows; sub-quadratic assertion |

## Sources

- **Project decisions (HIGH):** `.planning/PROJECT.md` — D-07 (cpu f64 + rocm f32 gate, f64-on-rocm skips-with-log), D-12 (property-gate vs 1e-5 for stochastic RandomProjection; exact-predicted-label gate scoped to *deterministic* SGD/SVM/NB), SplitMix64≠MT19937, per-phase build-failing memory gate.
- **Spike findings (HIGH):** `Skill("spike-findings-mlrs")` — `references/cpu-mlir-kernel-authoring.md` (proven op-set; 002-A loud launch failure; 002-B silent cross-loop miscompile; banned `SharedMemory`/`Atomic`/`F::INFINITY`/mutable-bool/shift-loops), `references/knn-graph-primitive.md` (single-owner GATHER idiom; VALUE-assert on adversarial fixture; query-axis tiling).
- **Feasibility-spike model (HIGH):** `.planning/milestones/v3.0-phases/13-knn-graph-primitive-feasibility-keystone/` (13-CONTEXT.md, 13-RESEARCH.md) — how a make-or-break primitive was de-risked standalone with explicit feasibility unknowns and VALUE-asserting oracles before consumers built on it.
- **Backlog risk flags (HIGH):** `.planning/notes/v3-hard-algorithm-backlog.md` — "histogram/split kernels need atomics or a GATHER redesign"; "spike GPU histogram/split under cpu-MLIR — this is the make-or-break feasibility question."
- **Project MEMORY landmines (MEDIUM):** `cubecl-cpu-no-shared-memory`, `rocm-is-runnable-gpu-gate` (f64 unsupported on rocm), `oracle-fixture-regen-needs-venv`, `full-cargo-test-exhausts-disk`, `backend-test-suite-slow`, `knn-oracle-tiebreak-needs-overfetch`.
- **cuML reference architecture (MEDIUM):** `cuml-main/` + CLAUDE.md — histogram-based GPU tree construction, FIL batched traversal, `UnsupportedOnGPU`/`ProxyBase` CPU-fallback pattern, `cuml.accel` `sys.modules` import-hook with caller-module exclusion.
- **Algorithm domain knowledge (MEDIUM):** Jones (1980) PACF stationarity transform and Kalman-filter ARIMA likelihood (statsmodels/cuML practice); Shapley efficiency/symmetry/dummy axioms and exact-by-enumeration for small feature counts; TreeSHAP exactness; gplearn evolutionary non-reproducibility.

---
*Pitfalls research for: mlrs v4.0 — tree ensembles, time-series, explainers, genetic regression, sklearn-utility, and cuml.accel under the CubeCL/cpu-MLIR + scikit-learn-oracle + SplitMix64-RNG constraints*
*Researched: 2026-06-26*
