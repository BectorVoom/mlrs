---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
verified: 2026-06-27T00:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 17: RandomForest GPU Histogram/Split Feasibility Spike Verification Report

**Phase Goal:** Prove (or refute) that GPU tree construction — single-owner GATHER histogram, relabel-partition, seed-from-first split-find — lowers and is tractable under cpu-MLIR, delivering an explicit GO/ADJUST/ABORT verdict that gates the entire tree chain (RF → FIL → TreeSHAP). Models the v3.0 Phase 13 KNN-graph keystone spike.
**Verified:** 2026-06-27
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | GATHER-histogram, relabel-partition, and seed-from-first split-find kernels standalone-launch on cpu(f64) [+ rocm(f32) where available] with no SharedMemory, no atomics, no F::INFINITY init | VERIFIED | `tree_spike/mod.rs` (666 lines) defines all three kernels with correct launch shapes. Grep confirms zero matches for SharedMemory, Atomic, F::INFINITY. `tree_spike_probes.rs` VALUE-asserts each kernel with 002-A all-zeros guards. Live: `cargo test tree_spike_probes` → 8 passed, 0 failed (0.25s). |
| 2 | A single decision tree built on injected fixed bootstrap/feature indices VALUE-matches sklearn.tree.DecisionTree* (split thresholds + leaf values) | VERIFIED | `tree_witness.rs` (839 lines, 8 tests) VALUE-asserts clf(gini) + reg(squared_error) + adversarial (f32+f64). Live: `cargo test tree_witness` → 8 passed, 0 failed (0.30s). Test output: clf 9 nodes/5 leaves exact structure + leaf values ≤1e-5; reg 25 nodes/13 leaves counts exact + regression-mean predictions ≤1e-5; adversarial 002-B backstop GREEN. |
| 3 | The SparseTreeNode { colid, threshold, left_child, value } format contract is finalized (right child = left_child + 1) | VERIFIED | Struct defined in `tree_spike/mod.rs` with exact fields `colid: i32, threshold: F, left_child: i32, value: i32`. D-02/D-03/D-04 semantics documented in code comments and in VERDICT.md as FINALIZED. `assert_contract` in `tree_witness.rs` validates D-02 (right=left_child+1), D-03 (leaf colid==-1), D-04 (value=offset into leaf buffer) across every internal node. cuML divergence noted: mlrs leaf = colid==-1 (NOT cuML's left_child==-1). |
| 4 | A per-tree cost benchmark is recorded and abort signals A1–A5 are each evaluated | VERIFIED | `tree_bench.rs` (265 lines, 1 test) records wall-clock at 64 and 128 bins on ≈1000×20×depth-8, a 250/500/1000 samples scaling sweep, and the frontier-memory observation. VERDICT.md frontmatter: `abort_signals_evaluated: [A1, A2, A3, A4, A5]`. Each signal has a concrete evidence row (A1 histogram kernel read-back; A2 host pre-pass, no device sort; A3 measured 64/128-bin wall-clock + scale shape; A4 split-find VALUE-assert incl. tie; A5 clf+reg+adversarial witness). All signals: PASS. |
| 5 | An explicit GO / ADJUST / ABORT verdict is delivered and the two-tier stochastic-gate convention is documented as the milestone-wide standard | VERIFIED | VERDICT.md exists with `verdict: GO` and `abort_signals_evaluated: [A1, A2, A3, A4, A5]` in frontmatter. §"Two-Tier Stochastic-Gate Convention" documents Tier-1 (deterministic injected-fixed-index core D-07) + Tier-2 (ensemble/predictive band ~0.02–0.05 D-08) as the "milestone-wide standard for every tree/ensemble phase (18–21)". Human-approved via Plan 17-05 blocking checkpoint (Task 3, orchestrator re-ran all three test targets before locking GO). |

**Score: 5/5 truths verified (0 present, behavior-unverified)**

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/gen_oracle.py` | gen_decision_tree_clf + gen_decision_tree_reg wired into main() | VERIFIED | Defines `gen_decision_tree_clf` (line 3063) + `gen_decision_tree_reg` (line 3142); both wired into main() for f32+f64 + adversarial (lines 3440-3446). Pattern `children_left` present. Commit 735dcb3 + 1019564. |
| `tests/fixtures/tree_dt_clf_f32_seed42.npz` | DecisionTreeClassifier(gini) reference blob | VERIFIED | File exists, committed in 1019564. |
| `tests/fixtures/tree_dt_clf_f64_seed42.npz` | DecisionTreeClassifier(gini) reference blob (f64) | VERIFIED | File exists. |
| `tests/fixtures/tree_dt_reg_f32_seed42.npz` | DecisionTreeRegressor(squared_error) reference blob | VERIFIED | File exists. |
| `tests/fixtures/tree_dt_reg_f64_seed42.npz` | DecisionTreeRegressor(squared_error) reference blob (f64) | VERIFIED | File exists. |
| `tests/fixtures/tree_dt_clf_adv_f32_seed42.npz` | Adversarial clf fixture (forced-pure-leaf + gain tie) | VERIFIED | File exists. |
| `tests/fixtures/tree_dt_clf_adv_f64_seed42.npz` | Adversarial clf fixture f64 | VERIFIED | File exists. |
| `tests/fixtures/tree_dt_reg_adv_f32_seed42.npz` | Adversarial reg fixture | VERIFIED | File exists. |
| `tests/fixtures/tree_dt_reg_adv_f64_seed42.npz` | Adversarial reg fixture f64 | VERIFIED | File exists. (8 total fixtures confirmed present.) |
| `crates/mlrs-backend/tests/tree_spike/mod.rs` | 3 kernels + launch wrappers + SparseTreeNode + build_tree loop (min 150 lines) | VERIFIED | 666 lines. All three kernels, launch wrappers, `SparseTreeNode<F>`, `build_tree<F>`. Commit da658f7. |
| `crates/mlrs-backend/tests/tree_spike_probes.rs` | Standalone VALUE-asserting probes per kernel + 002-A guard (min 90 lines) | VERIFIED | 339 lines, 8 tests. All three kernel probes with 002-A guards. Commit 04e4939. |
| `crates/mlrs-backend/tests/tree_witness.rs` | Tier-1 VALUE-assert witness vs sklearn clf+reg+adversarial (min 140 lines) | VERIFIED | 839 lines, 8 tests. Commit c69e21b. |
| `crates/mlrs-backend/tests/tree_bench.rs` | Wall-clock probe at 64 vs 128 bins + scaling sweep (min 80 lines) | VERIFIED | 265 lines, 1 test. Commit a95e40f. |
| `.planning/phases/17-.../VERDICT.md` | GO/ADJUST/ABORT verdict with A1–A5 + two-tier convention + SparseTreeNode contract | VERIFIED | 190 lines. Frontmatter: `verdict: GO`, `abort_signals_evaluated: [A1, A2, A3, A4, A5]`. Commit 739fb3b. |
| `.planning/spikes/MANIFEST.md` | Appended rows 003–006 | VERIFIED | Grep confirms 7 matches for `00[3-6]`; contains GO/VALIDATED. Commit 81a1672. |
| `.planning/spikes/003-gather-histogram-lower/` | Verbatim spike source (A1 evidence) | VERIFIED | Dir exists with README.md, kernels_and_harness.rs, probes.rs. |
| `.planning/spikes/004-seed-from-first-split-find/` | Verbatim spike source (A4 evidence) | VERIFIED | Dir exists with README.md, kernels_and_harness.rs, probes.rs. |
| `.planning/spikes/005-relabel-partition/` | Verbatim spike source (D-02 evidence) | VERIFIED | Dir exists with README.md, kernels_and_harness.rs, probes.rs. |
| `.planning/spikes/006-tier1-decisiontree-witness/` | Verbatim spike source (A5+A3 evidence) | VERIFIED | Dir exists with README.md, kernels_and_harness.rs, witness.rs, bench.rs. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tree_spike_probes.rs` | `tree_spike/mod.rs` | `mod tree_spike;` — calls launch wrappers | VERIFIED | Pattern `mod tree_spike` present; launch wrappers called for all three kernels. |
| `tree_spike_probes.rs` | `crates/mlrs-backend/src/capability.rs` | `capability::skip_f64_with_log()` early-return on every f64 probe | VERIFIED | Pattern `skip_f64_with_log` present in probes.rs. |
| `tree_witness.rs` | `tests/fixtures/tree_dt_clf_f64_seed42.npz` | `mlrs_core::oracle::load_npz` by-name X/y/bootstrap_idx/feature_idx/children_left/threshold/value | VERIFIED | Pattern `load_npz` present; fixture names in `fixture()` helper with clf/reg/adv variants. |
| `tree_witness.rs` | `tree_spike/mod.rs` | `mod tree_spike;` — calls `build_tree` composing the three kernels | VERIFIED | Pattern `build_tree` called in `run_witness`; `mod tree_spike;` declaration present. |
| `tree_bench.rs` | `tree_spike/mod.rs` | `mod tree_spike;` — drives `build_tree` at n_bins ∈ {64,128} | VERIFIED | Pattern `build_tree` present in bench; `n_bins` parameter drives 64 vs 128. |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Three tree kernels standalone-launch and VALUE-assert on cpu(f64) + f32 | `cargo test -p mlrs-backend --features cpu --test tree_spike_probes -- --nocapture` | `test result: ok. 8 passed; 0 failed; finished in 0.25s` | PASS |
| Single tree VALUE-matches sklearn clf+reg+adversarial on cpu(f64) + f32 | `cargo test -p mlrs-backend --features cpu --test tree_witness -- --nocapture` | `test result: ok. 8 passed; 0 failed; finished in 0.30s` — output: clf 9 nodes/5 leaves exact structure + leaf values ≤1e-5; reg 25 nodes/13 leaves counts exact + predictions ≤1e-5; adversarial GREEN | PASS |
| Build of all test modules compiles under cpu features | `cargo build -p mlrs-backend --features cpu --tests` | exit 0 (0.15s, cached) | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TREE-01 | Plans 01, 02, 03, 04, 05 | GPU tree-construction feasibility spike: kernels standalone-launch, VALUE-asserting sklearn witness, per-tree cost benchmark, finalized SparseTreeNode format contract, two-tier stochastic-gate convention, explicit GO/ADJUST/ABORT verdict with A1–A5 evaluated | SATISFIED | REQUIREMENTS.md marks `[x] TREE-01` complete; traceability table shows Phase 17 = Complete. All 5 success criteria verified via live test runs. |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tree_spike/mod.rs` | 521, 527, 619, 620 | `leaf_placeholder` variable name | INFO | False positive — this is a valid structural variable used to pre-allocate node slots in the flat array; it is filled in by the build loop. Not an implementation stub. Data flows correctly through the build loop. |
| `tree_witness.rs` | 234, 240, 336, 337 | `leaf_placeholder` variable name | INFO | Same pattern as above — `build_tree_variance` local builder; not a stub. |

No TBD, FIXME, or XXX markers found in any phase-17-modified file. No empty implementations. The code review (17-REVIEW.md, commit 159efa4) found 0 critical issues, 2 warnings, 3 info — no blockers.

---

### Human Verification Required

None. All 5 truths are VERIFIED with live test evidence. The blocking human-verify checkpoint (Plan 17-05, Task 3) was resolved with an APPROVED gate before this verification.

---

## Gaps Summary

No gaps. All 5 phase success criteria are VERIFIED against the actual codebase with behavioral evidence from live test runs:

- `tree_spike_probes.rs` → 8/8 tests GREEN (0.25s, cpu f64+f32)
- `tree_witness.rs` → 8/8 tests GREEN (0.30s, cpu f64+f32) — output confirms clf ≤1e-5, reg ≤1e-5, adversarial 002-B backstop GREEN
- `VERDICT.md` exists with GO verdict, A1–A5 all PASS, two-tier convention as milestone-wide standard, SparseTreeNode contract FINALIZED
- 8 committed .npz fixture blobs, 4 durable spike dirs, MANIFEST.md appended
- TREE-01 marked complete in REQUIREMENTS.md

The phase goal is achieved. The tree chain (Phase 18 → RF → FIL → TreeSHAP) is GO-gated.

---

_Verified: 2026-06-27_
_Verifier: Claude (gsd-verifier)_
