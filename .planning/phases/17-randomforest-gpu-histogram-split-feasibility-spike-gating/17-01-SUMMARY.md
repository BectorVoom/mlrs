---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
plan: 01
subsystem: oracle-fixtures
tags: [tree, decision-tree, oracle, sklearn, fixtures, TREE-01, nyquist-wave-0]
requires:
  - scripts/gen_oracle.py (existing gen_knn / gen_* idiom, _FIXTURE_DIR, c() cast)
provides:
  - tree_dt_clf/reg sklearn reference fixtures (standard + adversarial, f32+f64)
  - gen_decision_tree_clf + gen_decision_tree_reg generators wired into main()
affects:
  - Plan 03 Tier-1 correctness witness (value-asserts against these blobs)
tech-stack:
  added: []
  patterns:
    - injected fixed bootstrap+feature indices (D-07) — RNG-free single-tree match
    - independent pure-numpy gain/variance tie verification (Phase-13 CR-01/CR-02)
key-files:
  created:
    - tests/fixtures/tree_dt_clf_f32_seed42.npz
    - tests/fixtures/tree_dt_clf_f64_seed42.npz
    - tests/fixtures/tree_dt_reg_f32_seed42.npz
    - tests/fixtures/tree_dt_reg_f64_seed42.npz
    - tests/fixtures/tree_dt_clf_adv_f32_seed42.npz
    - tests/fixtures/tree_dt_clf_adv_f64_seed42.npz
    - tests/fixtures/tree_dt_reg_adv_f32_seed42.npz
    - tests/fixtures/tree_dt_reg_adv_f64_seed42.npz
  modified:
    - scripts/gen_oracle.py
decisions:
  - "feature_idx is injected by fitting sklearn on X[bootstrap_idx][:, feature_idx]; tree_.feature indexes the SUBSET, not original X columns"
  - "adversarial fixtures use two IDENTICAL columns (forced gain tie) + perfectly-separable target (forced pure leaves); sklearn recorded feature 0 = lowest-index canonical tie-break"
  - "all arrays stored as fixture float dtype via c() (matches existing gen_knn idiom; small ints exact in f32)"
metrics:
  duration_min: 9
  completed: 2026-06-27
  tasks: 2
  files: 9
status: complete
---

# Phase 17 Plan 01: DecisionTree Oracle Foundation Summary

Committed sklearn DecisionTree reference fixtures (gini-classifier + squared-error-regressor, injected fixed bootstrap/feature indices, standard + adversarial, f32+f64) plus the two `gen_oracle.py` generators wired into `main()` — the Nyquist Wave-0 oracle the Plan-03 Tier-1 witness value-asserts against.

## What Was Built

**Task 1 — generators (`scripts/gen_oracle.py`, commit 735dcb3):**
- `gen_decision_tree_clf(seed, dtype, structure="standard")` — `DecisionTreeClassifier(criterion="gini")`.
- `gen_decision_tree_reg(seed, dtype, structure="standard")` — `DecisionTreeRegressor(criterion="squared_error")`.
- Both fit on `X[DT_BOOTSTRAP_IDX][:, DT_FEATURE_IDX]` — fixed integer index arrays (never RNG-drawn), so the single tree reproduces element-wise (D-07).
- Independent tie-verification helpers `_dt_gini_best_impurity` / `_dt_var_best_impurity` (pure numpy) prove the adversarial gain tie is genuine without ever consulting sklearn's pick (Phase-13 CR-01/CR-02).

**Task 2 — wiring + fixtures (commit 1019564):**
- `main()` unconditionally calls both generators for f32+f64 plus `structure="adversarial"`.
- 8 committed `.npz` blobs regenerated from a `/tmp` venv (numpy 2.5.0 + scikit-learn 1.9.0, PEP-668).

## Fixture Contract (for Plan 03 `load_npz` by-name binding)

Every fixture emits the SAME 9 array keys (all cast to the fixture float dtype via `c()`):

| Key | Meaning |
|-----|---------|
| `X` | full synthetic design matrix (standard: 60×8; adversarial: 16×2) |
| `y` | target (clf: int class labels; reg: continuous) |
| `bootstrap_idx` | injected fixed bootstrap row indices (standard: len 48 w/ repeats; adv: arange 16) |
| `feature_idx` | injected fixed feature-column subset (standard: [0,2,3,5,6]; adv: [0,1]) |
| `children_left` | sklearn `tree_.children_left` (leaf sentinel −1) |
| `children_right` | sklearn `tree_.children_right` |
| `feature` | sklearn `tree_.feature` — indexes the `feature_idx` SUBSET, not original X (leaf = −2) |
| `threshold` | sklearn `tree_.threshold` |
| `value` | sklearn `tree_.value` (clf: shape (n,1,2) class counts; reg: (n,1,1) means) |

**Fixture file names** (all under `tests/fixtures/`):
- Standard: `tree_dt_clf_{f32,f64}_seed42.npz`, `tree_dt_reg_{f32,f64}_seed42.npz`
- Adversarial: `tree_dt_clf_adv_{f32,f64}_seed42.npz`, `tree_dt_reg_adv_{f32,f64}_seed42.npz`

**Observed structures (f64):**
- `clf` standard: 9 nodes / 5 leaves. `reg` standard: 25 nodes / 13 leaves.
- `clf_adv`: 3 nodes — root splits on feature 0, leaves pure (counts [1,0] and [0,1]).
- `reg_adv`: 3 nodes — root splits on feature 0, leaves constant means 1.0 and 5.0 (zero variance).
- In both adversarial trees sklearn recorded **feature 0** (the lowest of the two identical tied columns) — matching the documented canonical tie-break (lowest feature index, then lowest threshold). Because the tied columns are identical, the partition/children/leaf-values are invariant to which tied index is recorded.

## Threat-Model Mitigation (T-17-01)

The adversarial fixtures are the VALUE-assert backstop for the cpu-MLIR SILENT histogram/argmax miscompile (FINDING 002-B), consumed by Plan 03. The forced-pure-leaf + exact-gain-tie is the histogram analogue of Phase 13's duplicate-point row. The tie-break is encoded as an INDEPENDENT generator rule (independently verified via pure-numpy impurity, not by reading sklearn's choice), keeping the Plan-03 gate non-circular.

## Deviations from Plan

None — plan executed exactly as written. The `feature_idx` injection was implemented by fitting on the explicit column subset `X[boot][:, feat]` (the deterministic, RNG-free interpretation the plan calls for, vs. `max_features` which would re-introduce RNG feature selection).

## Verification

- `python3 -c "import ast; ast.parse(open('scripts/gen_oracle.py').read())"` exits 0.
- `gen_decision_tree_clf` + `gen_decision_tree_reg` defined and called in `main()` for f32+f64 + adversarial (grep, comment lines excluded).
- 8 `tree_dt_*.npz` fixtures committed under `tests/fixtures/`; `git status` confirms NO other fixture added or modified.
- Adversarial blobs independently confirmed: pure leaves + genuine gain tie (assert in generator).

## Self-Check: PASSED
