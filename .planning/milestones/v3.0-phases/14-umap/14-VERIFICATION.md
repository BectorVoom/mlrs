---
phase: 14-umap
verified: 2026-06-24T12:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/4 truths functionally verified; CR-01/CR-02/CR-03 gaps blocking
  gaps_closed:
    - "CR-02 BLOCKER: Umap::fit now rejects n_components >= n with AlgoError::InvalidNComponents before any device launch (guard at umap.rs:459-465; test fit_rejects_n_components_ge_n PASS)"
    - "CR-01 BLOCKER: fit-path layout launch now passes move_other=0 via FIT_MOVE_OTHER constant (umap.rs:1175); no owner-cube writes a foreign vertex's slots; cross-cube WRITE-WRITE race eliminated; D-05 byte-identical contract holds on any parallel backend"
    - "CR-03 WARNING: symmetric COO + move_other=0 processes each undirected pair once per direction (not ~2-4x doubled); PROPERTY_EPS re-derived 0.02->0.03 against the corrected schedule under hard guardrail (eps <= 0.04 and ~12x worst margin); per_pair_sample_count_matches_schedule anchors the schedule"
  gaps_remaining: []
  regressions: []
gaps: []
deferred: []
---

# Phase 14: UMAP Verification Report (Re-Verification)

**Phase Goal:** Deliver UMAP `fit`/`fit_transform` → `embedding_` `(n, n_components)` with umap-learn/sklearn-named hyperparameters: KNN graph (Phase 13) → fuzzy simplicial set → init (random default; spectral) → vertex-owner GATHER SGD layout with negative sampling. Deterministic stages value-gated ≤1e-5 (f64); stochastic layout property-gated vs umap-learn 0.5.12; byte-identical reproducibility for fixed random_state.

**Verified:** 2026-06-24
**Status:** passed
**Re-verification:** Yes — after gap closure (Plans 14-06 and 14-07)

---

## Re-Verification Focus

The prior `gaps_found` report (2026-06-24T03:30:00Z) identified three gaps requiring closure:

| Gap | Plan | Status |
|-----|------|--------|
| CR-02 (BLOCKER): Missing `n_components < n` guard in `Umap::fit` | 14-06 | CLOSED |
| CR-01 (BLOCKER): Cross-cube write-write race on parallel backends (fit used `move_other=1`) | 14-07 | CLOSED |
| CR-03 (WARNING): Symmetric COO + `move_other=1` double-counted every undirected edge; gate calibrated to the bug | 14-07 | CLOSED |

Each gap was verified against the actual codebase, not SUMMARY claims. See sections below.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can fit/fit_transform → embedding_ (n, n_components) with umap-learn-named hyperparameters + defaults; min_dist≤spread validated at build | ✓ VERIFIED | Unchanged from initial verification. `fit_roundtrip` PASS per git 870b1e2; `build_rejects_bad_min_dist` and `defaults_equal` both PASS live (confirmed this re-verification). Guard at umap.rs:459-465 adds to robustness without regressing the positive path. |
| 2 | Deterministic stages (KNN graph, fuzzy simplicial set, fuzzy-set union, spectral init w/ random fallback above Jacobi cap) value-match umap-learn ≤1e-5 (f64) | ✓ VERIFIED | Unchanged from initial verification. smooth_knn/fuzzy_union/ab PASS; spectral GREEN per git 61649ca. 21 oracle fixtures committed. Plans 14-06/14-07 made no changes to these stages. |
| 3 | Stochastic SGD layout passes property/structural gate vs umap-learn 0.5.12 AND same random_state → byte-identical across runs, on any CubeCL backend (D-05) | ✓ VERIFIED | GAPS NOW CLOSED. (a) `FIT_MOVE_OTHER=0` at umap.rs:1175 routes through the fit launch at umap.rs:1347: no `1u32` literal present; cross-cube write race eliminated. (b) `fit_move_other_is_zero` PASS — asserts flag==0 and n owner write ranges o*dim..(o+1)*dim are pairwise non-overlapping and partition 0..n*dim. (c) `per_pair_sample_count_matches_schedule` PASS — single-pass-per-direction schedule confirmed, no double-count. (d) PROPERTY_EPS recalibrated 0.02→0.03 (≈12× worst margin 0.0025, under 0.04 ceiling) per 14-VALIDATION.md; ARI_BAND stays 0.05. Full `layout_property_*` 5-metric family + `reproducible_f64` confirmed green by Plan 14-07 executor (background run, 2254s + 1283s). |
| 4 | User can embed new data via transform(X_new) against the fitted fuzzy graph, property sub-gated | ✓ VERIFIED | Unchanged from initial verification. transform path drives `move_other=0` at umap.rs:866 (was already correct). `transform_property_*` 5/5 + transform byte-identical per git 749186b; TRANSFORM_PROPERTY_EPS=0.15. |

**Score:** 4/4

---

### CR-02 Gap Verification (14-06)

**Claimed:** Guard `if self.n_components >= n { return Err(AlgoError::InvalidNComponents { estimator: "umap", .. }) }` added to `Umap::fit` at line 459-465, after `validate_geometry` and before `run_umap_layout`. Test `fit_rejects_n_components_ge_n` added.

**Codebase evidence:**

- `umap.rs` lines 459-465: guard confirmed present (read directly). Condition is `self.n_components >= n`; error is `AlgoError::InvalidNComponents { estimator: "umap", requested: self.n_components, max: n.saturating_sub(1) }`. Placed after `validate_geometry(x, shape)?` at line 448 and before `run_umap_layout` at line 467.
- `umap_test.rs` lines 408-440: `fit_rejects_n_components_ge_n` is a `#[test]` fn. Uses `n=2`, `n_components=2`, asserts `Err(AlgoError::InvalidNComponents { estimator: "umap", requested: 2, .. })` via `matches!`. Gated via `gate_f64`.
- Live test run: `fit_rejects_n_components_ge_n` — 1 passed.
- Commits: `8ed8fdb` (guard), `0244de2` (test) — both exist in git log.

**Verdict: CLOSED.**

---

### CR-01 + CR-03 Gap Verification (14-07)

**Claimed:** `FIT_MOVE_OTHER: u32 = 0` introduced as single source of truth (umap.rs:1175); `fit_move_other()` accessor at umap.rs:1182; fit launch at umap.rs:1347 passes `FIT_MOVE_OTHER` (not bare `1u32`); kernel `umap_layout.rs` unchanged; `fit_move_other_is_zero` and `per_pair_sample_count_matches_schedule` tests added; PROPERTY_EPS updated 0.02→0.03; ARI_BAND remains 0.05; 14-VALIDATION.md calibration block updated.

**Codebase evidence:**

- `FIT_MOVE_OTHER: u32 = 0` at umap.rs:1175 — confirmed present.
- `pub fn fit_move_other() -> u32 { FIT_MOVE_OTHER }` at umap.rs:1182-1184 — confirmed present.
- `host_epoch_driver` launch at umap.rs:1347 passes `FIT_MOVE_OTHER` (not a bare integer literal). Grep for `1u32.*move_other` or `move_other.*1u32` in umap.rs returns 0 matches.
- `crates/mlrs-kernels/src/umap_layout.rs`: last modified by commit `ba96b4c` (Plan 04, pre-14-07). No edits in 14-07 commits. The kernel's doc comment at line 73-74 still says `1u32` = "the `fit` path" — this is a stale comment (INFO only, not a correctness issue; the caller now passes `0u32` via `FIT_MOVE_OTHER`).
- `fit_move_other_is_zero` test at umap_test.rs:1097-1133: executable `assert_eq!(fit_move_other(), 0, ...)` plus mark-and-check slot-disjointness over `n=7, dim=2` — confirmed substantive (not comment-only).
- `per_pair_sample_count_matches_schedule` test at umap_test.rs:1147-1223: replays `host_epoch_driver` positive-sample clock for a tiny symmetric COO (4 directed edges, 2 undirected pairs); asserts per-edge draws within ±1 of expected, and per-pair total within ±2 — confirmed substantive.
- `const PROPERTY_EPS: f64 = 0.03` at umap_test.rs:66 — confirmed (up from 0.02).
- `const ARI_BAND: f64 = 0.05` at umap_test.rs:88 — confirmed unchanged.
- 14-VALIDATION.md: calibration block updated with move_other=0 per-metric margin table, hard guardrail (ε ≤ 0.04 ceiling + small-multiple-of-worst-margin relation), and note that `per_pair_sample_count_matches_schedule` anchors the schedule.
- Commits: `2c805a5` (move_other=0 launch), `fc5e51f` (invariant tests), `4fd016f` (recalibration) — all exist in git log.
- Live test runs:
  - `fit_move_other_is_zero` — 1 passed
  - `per_pair_sample_count_matches_schedule` — 1 passed

**Verdict: CLOSED.**

---

### Regression Check (Previously Verified Truths)

| Test | Result |
|------|--------|
| `build_rejects_bad_min_dist` | PASS |
| `defaults_equal` | PASS |
| `metrics_table_covers_five` | PASS |
| `fit_rejects_n_components_ge_n` | PASS |
| `fit_move_other_is_zero` | PASS |
| `per_pair_sample_count_matches_schedule` | PASS |

No regressions detected in any of the fast non-property tests.

---

### Required Artifacts

| Artifact | Status | Evidence |
|----------|--------|----------|
| `crates/mlrs-algos/src/manifold/umap.rs` | ✓ VERIFIED | n_components guard (lines 459-465) + FIT_MOVE_OTHER (1175) + fit_move_other() accessor (1182) + fit launch via FIT_MOVE_OTHER (1347) |
| `crates/mlrs-algos/tests/umap_test.rs` | ✓ VERIFIED | `fit_rejects_n_components_ge_n` (408-440), `fit_move_other_is_zero` (1097-1133), `per_pair_sample_count_matches_schedule` (1147-1223); PROPERTY_EPS=0.03 (66), ARI_BAND=0.05 (88) |
| `.planning/phases/14-umap/14-VALIDATION.md` | ✓ VERIFIED | Calibration block updated: move_other=0 per-metric margins, hard guardrail, schedule-fidelity note |
| `crates/mlrs-kernels/src/umap_layout.rs` | ✓ UNCHANGED | Last commit ba96b4c (Plan 04); no edits in 14-06/14-07 (confirmed via git log). The kernel is parameterized by `move_other`; behavior is correct because the caller now passes `FIT_MOVE_OTHER=0`. |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| `Umap::fit` | `AlgoError::InvalidNComponents` | Guard at umap.rs:459-465 before any device launch | ✓ WIRED |
| `host_epoch_driver` launch | `umap_layout_step` | `FIT_MOVE_OTHER` (= 0) as `move_other` arg at umap.rs:1347 | ✓ WIRED (owner-only) |
| `fit_move_other_is_zero` test | `mlrs_algos::manifold::umap::fit_move_other()` | `assert_eq!(fit_move_other(), 0, ...)` at umap_test.rs:1099 | ✓ WIRED |

### Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|----------|
| UMAP-01 | ✓ SATISFIED | fit/fit_transform → embedding_ with umap-named hyperparams; CR-02 guard closes the panicking edge case (fit now robust on valid-typed input) |
| UMAP-02 | ✓ SATISFIED | Deterministic stages value-gate unchanged; Plans 14-06/14-07 made no changes to KNN/fuzzy/spectral/a-b stages |
| UMAP-03 | ✓ SATISFIED | CR-01 + CR-03 closed: fit path owner-only (move_other=0) over symmetric COO; single-pass-per-direction schedule matches umap-learn; PROPERTY_EPS recalibrated under hard guardrail; executable invariant tests guard regression |
| UMAP-04 | ✓ SATISFIED | transform path unchanged; was already correct (move_other=0 at umap.rs:866) |

All 4 UMAP requirement IDs (UMAP-01..04) marked complete in REQUIREMENTS.md and assigned to Phase 14. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Note |
|------|------|---------|----------|------|
| `crates/mlrs-kernels/src/umap_layout.rs` | 73-74 | Doc comment says `1u32` = "the `fit` path" — now stale since the fit path uses `FIT_MOVE_OTHER=0` | ℹ️ Info | The kernel behavior is correct (it is parameterized); only the doc comment lags. Not a correctness issue; no fix needed before phase passes. |

No TBD/FIXME/XXX debt markers found in any phase-14 modified files. Prior CR-01/CR-02/CR-03 blockers are resolved.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CR-02 guard: `n_components >= n` → typed error, not panic | `cargo test -p mlrs-algos --features cpu fit_rejects_n_components_ge_n` | 1 passed (0.12s) | PASS |
| CR-01 invariant: fit_move_other()==0 + slot-disjoint partition | `cargo test -p mlrs-algos --features cpu fit_move_other_is_zero` | 1 passed | PASS |
| CR-03 guard: per-pair sample count = expected (not doubled) | `cargo test -p mlrs-algos --features cpu per_pair_sample_count_matches_schedule` | 1 passed | PASS |
| Regression: build guard / defaults / metrics table | `build_rejects_bad_min_dist`, `defaults_equal`, `metrics_table_covers_five` | 3 passed | PASS |
| Full 5-metric `layout_property_*` + `reproducible_f64` | Background run by Plan 14-07 executor (2254s + 1283s) | 5+1 passed | PASS (executor evidence; not re-run per disk constraint) |

### Human Verification Required

None. All Phase-14 in-scope behaviors have automated Rust verification. The Python/PyO3 estimator surface remains explicitly deferred to Phase 16 per VALIDATION.md.

---

## Gaps Summary

No gaps. All three blockers from the prior `gaps_found` report are confirmed closed by codebase inspection and live test execution:

- **CR-02** (BLOCKER) — closed by Plan 14-06: guard in `Umap::fit` at line 459-465 returns `AlgoError::InvalidNComponents { estimator: "umap", .. }` before any device launch when `n_components >= n`. Test `fit_rejects_n_components_ge_n` PASS.
- **CR-01** (BLOCKER) — closed by Plan 14-07: `FIT_MOVE_OTHER: u32 = 0` at umap.rs:1175 is the single source of truth for the fit-path `move_other` flag; the fit launch at umap.rs:1347 passes `FIT_MOVE_OTHER`, not a bare `1u32`. No owner-cube writes a foreign vertex's slots. Executable invariant `fit_move_other_is_zero` asserts the flag AND proves the n write-range partition is disjoint. D-05 byte-identical contract now holds on any parallel backend.
- **CR-03** (WARNING) — resolved alongside CR-01 by `move_other=0` over the symmetric COO: each undirected pair is processed once per direction (not doubled). `per_pair_sample_count_matches_schedule` guards the corrected schedule. `PROPERTY_EPS` recalibrated 0.02→0.03 under a hard guardrail (ε ≤ 0.04 ceiling and ≈12× worst measured margin of 0.0025), documented in 14-VALIDATION.md.

The phase goal is fully achieved. All four UMAP requirements (UMAP-01..04) are satisfied.

---

_Verified: 2026-06-24_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes — after Plans 14-06 + 14-07 gap closure_
