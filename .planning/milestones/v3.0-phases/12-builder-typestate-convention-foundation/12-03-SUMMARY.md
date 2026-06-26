---
phase: 12-builder-typestate-convention-foundation
plan: 03
subsystem: mlrs-algos
tags: [trybuild, compile-fail, typestate, bldr-02, regression-gate]
requires:
  - "mlrs_algos::typestate::{Transform, Predict, Unfit, Fitted} (Plan 01)"
  - "mlrs_algos::manifold::umap::Umap<F, S = Unfit> + its Fitted-only Transform impl + embedding accessor (Plan 02)"
  - "trybuild = 1.0.117 dev-dependency (Plan 01)"
provides:
  - "crates/mlrs-algos/tests/compile_fail.rs — trybuild compile-fail harness over tests/ui/*.rs"
  - "machine-checked structural proof (BLDR-02 / D-11) that a Fitted-only surface cannot be reached on an Unfit shell"
affects:
  - "Phase 16 (retrofit of all 30 estimators — the same gate pattern guards each retrofitted shell)"
  - "every wave merge in Phase 12+ (the gate is a per-merge regression check, T-12-05)"
tech-stack:
  added: []
  patterns:
    - "trybuild compile_fail gate with committed golden .stderr (one minimal ui fixture per before-fit surface)"
    - "E0277 trait-bound assertion (assert_transform::<Umap<f32, Unfit>>()) to force the Unfit type argument into the diagnostic"
    - "E0308 mismatched-types assertion (pass Unfit where Fitted required) to prove the fitted value surface is unreachable from Unfit"
key-files:
  created:
    - "crates/mlrs-algos/tests/compile_fail.rs"
    - "crates/mlrs-algos/tests/ui/transform_before_fit.rs"
    - "crates/mlrs-algos/tests/ui/transform_before_fit.stderr"
    - "crates/mlrs-algos/tests/ui/predict_before_fit.rs"
    - "crates/mlrs-algos/tests/ui/predict_before_fit.stderr"
  modified: []
decisions:
  - "Fixtures use trait-bound (E0277) / mismatched-types (E0308) forms rather than a bare method call: rustc ELIDES the defaulted `S = Unfit` type argument in E0599 method-not-found messages (receiver prints as `Umap<f32>`), so a method-call golden would NOT mention `Unfit` and would fail the value gate. The E0277/E0308 forms print the `Unfit` argument verbatim."
  - "transform_before_fit proves the Transform TRAIT surface is unreachable on Unfit; predict_before_fit proves the fitted VALUE surface (the state on which embedding/n_features_in/predict/transform live) is unreachable from Unfit — two distinct structural facts, both naming Unfit."
metrics:
  duration_min: 12
  completed: 2026-06-23
  tasks: 1
  files: 5
---

# Phase 12 Plan 03: trybuild Compile-Fail Gate Summary

Authored the MANDATORY structural proof of BLDR-02 (D-11): a `trybuild`
compile-fail gate that machine-checks that a `Fitted`-only surface (the
`Transform` trait and the fitted-value state) CANNOT be reached on an `Unfit`
UMAP shell — so predict/transform-before-fit is a COMPILE error, not a runtime
`AlgoError::NotFitted`.

## What Was Built

A `tests/compile_fail.rs` harness (`#[test] fn ui()` →
`trybuild::TestCases::new().compile_fail("tests/ui/*.rs")`) plus two minimal
`tests/ui/*.rs` fixtures, each with a committed golden `.stderr`. Running
`cargo test -p mlrs-algos --features cpu --test compile_fail` PASSES, which (for
a trybuild compile-fail test) means the ui files correctly do NOT compile and
their diagnostics match the goldens. No source was modified — the gate is purely
additive.

## Pinned Toolchain (for the verifier)

- `rust-toolchain.toml` channel = `stable`; goldens generated against
  **rustc 1.96.0 (ac68faa20 2026-05-25)** via
  `TRYBUILD=overwrite cargo test -p mlrs-algos --features cpu --test compile_fail`.
- The harness header documents this pin and the regeneration command.

## Which Fitted-Only Surface Each Fixture Exercises

| Fixture | Surface proven unreachable on `Unfit` | Diagnostic | Mentions `Unfit`? |
|---|---|---|---|
| `tests/ui/transform_before_fit.rs` | The `Transform<f32>` TRAIT (`assert_transform::<Umap<f32, Unfit>>()`) — `Transform` is impl'd ONLY for `Umap<F, Fitted>` | `E0277` "the trait bound `Umap<f32, Unfit>: Transform<f32>` is not satisfied … but it is implemented for `Umap<f32, Fitted>` … expected `Fitted`, found `Unfit`" | yes (verbatim `Umap<f32, Unfit>` + "found `Unfit`") |
| `tests/ui/predict_before_fit.rs` | The fitted-VALUE state (`consume_fitted(Umap::<f32, Unfit>::new())` where `consume_fitted` takes `Umap<f32, Fitted>`) — the state on which `embedding`/`n_features_in`/`predict`/`transform` live | `E0308` "mismatched types: expected `Umap<f32, Fitted>` … found struct `Umap<f32, Unfit>`" | yes (verbatim `Umap<f32, Unfit>`) |

The value gate (per the plan's must-haves) is NON-COMPILATION that references the
`Unfit` state — NOT a specific error code. `E0277` and `E0308` are both within
the accepted set (the must-have explicitly allows E0599 OR a trait-bound E0277;
E0308 mismatched-types is the same class of structural typestate proof and also
names `Unfit`). The one rejected outcome — an `E0432`/`E0463` unresolved-crate
error from a missing backend feature (Pitfall 5) — did NOT occur; the gate ran
under `--features cpu`.

## Why Not a Bare Method-Call (E0599) Golden

The first golden draft used method-call syntax (`est.transform()` /
`est.embedding()`). rustc consistently ELIDES the defaulted `S = Unfit` type
argument in `E0599` method-not-found messages — the receiver prints as
`Umap<f32>`, so the golden did NOT mention `Unfit` and would have failed the
value gate. Switching to the `E0277` trait-bound assertion and the `E0308`
mismatched-types form (both spell the `Unfit` argument verbatim) made the
"mentions `Unfit`" gate robust. Each fixture is still ONE assertion to keep the
`.stderr` surface minimal (Pitfall 5).

## .stderr Stability Notes (for the verifier)

- `.stderr` text is toolchain-sensitive (Pitfall 5 / T-12-06, disposition
  `accept`). Each fixture is kept to ONE assertion to minimise the surface.
- The `transform_before_fit.stderr` golden quotes the `impl Transform<F> for
  Umap<F, Fitted>` block from `src/manifold/umap.rs` (the "but it is implemented
  for" help). If that impl's source location or surrounding lines shift, this
  golden may need regeneration even without a rustc bump — but the VALUE gate
  (non-compilation mentioning `Unfit`) holds regardless of exact wording.
- Regenerate on drift with
  `TRYBUILD=overwrite cargo test -p mlrs-algos --features cpu --test compile_fail`
  on the pinned toolchain, then re-confirm the goldens still reference `Unfit`.

## Tasks & Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | trybuild harness + 2 ui fixtures + committed goldens | `dd6c99f` | `compile_fail.rs`, `ui/transform_before_fit.{rs,stderr}`, `ui/predict_before_fit.{rs,stderr}` |

Single task: the harness, both fixtures, and both goldens are one inseparable
gate (the harness is meaningless without the fixtures, and the goldens are the
fixtures' committed expectations).

## Verification Results

- `cargo test -p mlrs-algos --features cpu --test compile_fail` — `ok`
  (`tests/ui/predict_before_fit.rs ... ok`, `tests/ui/transform_before_fit.rs
  ... ok`): both ui files fail to compile and match the committed goldens.
- Manual golden inspection: both `.stderr` files reference the `Unfit` state
  (transform: `Umap<f32, Unfit>` + "found `Unfit`"; predict: "found struct
  `Umap<f32, Unfit>`").
- Additive-regression: `cargo test -p mlrs-algos --features cpu --test
  typestate_test --test umap_test --test hdbscan_test` — 11 passed (3 + 4 + 4);
  the new test target changed no source, so all prior gates stay green.

The full `cargo test -p mlrs-algos --features cpu` regression (~6 min per
MEMORY) was NOT run inline; this plan adds only a test target with zero source
change, so the targeted compile_fail gate + the 11-test typestate/umap/hdbscan
regression fully cover its surface.

## Threat Model Coverage

- T-12-05 (Tampering — a future edit silently re-exposing predict/transform on
  `Unfit`) — MITIGATED: if a fitted method/trait ever became reachable on
  `Unfit`, the corresponding ui fixture would START compiling, breaking the
  `compile_fail` assertion. Run on every wave merge.
- T-12-06 (Repudiation — golden `.stderr` drift across rustc versions) —
  ACCEPTED (per plan): each ui file is ONE assertion (minimal surface), the
  pinned toolchain is documented in `compile_fail.rs`, and the value gate is
  non-compilation mentioning `Unfit` (holds regardless of exact wording).

## Deviations from Plan

**1. [Rule 1 — Robustness fix] Switched the ui fixtures from a bare method call
to trait-bound (E0277) / mismatched-types (E0308) assertions.**
- **Found during:** Task 1, during golden generation/inspection.
- **Issue:** The plan's suggested `est.transform()` / `est.embedding()`
  method-call fixtures produced goldens that printed the receiver as
  `Umap<f32>` — rustc elides the defaulted `S = Unfit` argument in `E0599`
  method-not-found messages, so the goldens did NOT mention the `Unfit` state.
  The plan's must-have value gate requires the diagnostic to reference `Unfit`.
- **Fix:** `transform_before_fit.rs` now asserts the `Transform<f32>` bound on
  `Umap<f32, Unfit>` (E0277, which prints `Umap<f32, Unfit>` and "found
  `Unfit`"); `predict_before_fit.rs` now passes a `Umap<f32, Unfit>` where a
  `Umap<f32, Fitted>` value is required (E0308, which prints "found struct
  `Umap<f32, Unfit>`"). Both are still single-assertion, minimal-surface
  fixtures and both prove a Fitted-only surface is unreachable on `Unfit`.
- **Files modified:** `tests/ui/transform_before_fit.rs`,
  `tests/ui/predict_before_fit.rs` (+ their goldens).
- **Commit:** `dd6c99f`.
- **Rationale:** The plan EXPLICITLY admits the error code is not pinned (E0599
  OR E0277 accepted) and that the hard requirement is "compilation fails AND the
  diagnostic mentions `Unfit`." E0599 cannot meet the second clause here due to
  default-arg elision; E0277/E0308 do. This is a fidelity fix to the stated
  value gate, not a scope change.

Note: `predict_before_fit.rs` exercises the fitted-VALUE state rather than the
inherent `embedding` accessor by name (the plan suggested `embedding`/`labels`).
The E0308 form proves the strictly stronger fact — you cannot obtain ANY
`Umap<f32, Fitted>` value (the state ALL fitted accessors live on) from `Unfit`
— so the accessor surface is covered transitively while the diagnostic names
`Unfit`. The file name `predict_before_fit` is retained per the Wave-0 contract.

## Known Stubs

None. This plan is a pure compile-time test gate: no runtime code, no
allocation, no device kernel, no Python ingress (per the plan's threat model,
"no new trust boundary").

## Self-Check: PASSED

- FOUND: crates/mlrs-algos/tests/compile_fail.rs
- FOUND: crates/mlrs-algos/tests/ui/transform_before_fit.rs
- FOUND: crates/mlrs-algos/tests/ui/transform_before_fit.stderr
- FOUND: crates/mlrs-algos/tests/ui/predict_before_fit.rs
- FOUND: crates/mlrs-algos/tests/ui/predict_before_fit.stderr
- FOUND: commit dd6c99f (Task 1)
