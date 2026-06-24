# Phase 16: Builder Retrofit Sweep + Shim Coverage - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-24
**Phase:** 16-Builder Retrofit Sweep + Shim Coverage
**Areas discussed:** Retrofit depth & old-trait fate, new() reconciliation, Pilot selection & sweep order, Shim verification rigor

---

## Retrofit depth & old-trait fate

| Option | Description | Selected |
|--------|-------------|----------|
| Full convergence (delete old traits.rs) | Port all 9 old traits to typestate-aware versions, migrate every estimator to consuming-self fit, then DELETE crate::traits at phase end (Phase-12 D-07 end-state). One surface, zero permanent debt. | ✓ |
| Additive coexistence (keep old traits.rs) | Add builder()+typestate front-door but leave crate::traits live, skip consuming-self migration where it churns; defer deletion. Lowest risk, but standing two-surface debt. | |
| Convergence, delete deferred | Migrate all to typestate traits this phase but keep traits.rs as thin re-export shims; defer hard file removal + final call-site sweep to a follow-up. | |

**User's choice:** Full convergence (delete old traits.rs)
**Notes:** Resolves the BLDR-03 ("additive / fit path untouched") vs Phase-12 D-07 ("migrate all + delete") tension in favor of D-07's single-surface end-state. "Additive" reinterpreted as protecting config fields + fit numerics (byte-identical → gates preserved), NOT avoiding the typestate migration. Per-estimator green-suite gate is the safety mechanism.

---

## new() reconciliation

| Option | Description | Selected |
|--------|-------------|----------|
| Convert to convention, builder-only args | new() → zero-arg sklearn defaults everywhere; remove new(args)/with_*; migrate ~137 call sites to builder. True single-source defaults, one idiom. | ✓ |
| Keep new(args), add builder + zero-arg Default | Leave legacy constructors untouched (no churn); add builder()+Default alongside. Lowest churn, two idioms persist, invariant only partial. | |
| Convert simple, keep multi-constructor cases | Convert single-constructor estimators; keep KMeans-style multi-constructors and layer builder on top. Leaves inconsistency at the complex cases. | |

**User's choice:** Convert to convention, builder-only args
**Notes:** Consistent with full convergence — uniform `T::new() == T::builder().build()?` invariant across all estimators; ~137 `::new(` call sites in algos tests migrate to builder.

---

## Pilot selection & sweep order

| Option | Description | Selected |
|--------|-------------|----------|
| Ridge + MBSGDRegressor (cover both shapes) | Ridge = no-builder/arg-taking-new full build-out; MBSGDRegressor = already-has-builder typestate+trait-swap-only. Sweep rest module-by-module, KMeans late. | ✓ |
| Ridge only (simplest first) | Single simplest pilot, then sweep everything module-by-module. Less upfront coverage. | |
| KMeans + Ridge (hardest first) | Pilot nastiest case (KMeans 3 constructors) + clean baseline. Front-loads risk; pilot itself may churn. | |

**User's choice:** Ridge + MBSGDRegressor (cover both shapes)
**Notes:** Two pilots cover both structurally-distinct retrofit shapes before the bulk sweep. Sweep module-by-module, each estimator gated by its suite; KMeans handled late as the multi-constructor stress case.

---

## Shim verification rigor

| Option | Description | Selected |
|--------|-------------|----------|
| Full static subset (max verifiable) | Per class: import w/o ext + get_params/set_params/clone round-trip + AST __init__-purity assert + fit-free sklearn estimator_checks subset (check_no_attributes_set_in_init, check_parameters_default_constructible, check_get_params_invariance). | ✓ |
| Round-trip only (minimal) | get_params/set_params round-trip + clone in a Rust unit test + thin Python import test. Skips AST purity + estimator_checks subset. | |
| Round-trip + AST purity (no estimator_checks) | Round-trip/clone + AST __init__-purity, skip the estimator_checks harness plumbing. | |

**User's choice:** Full static subset (max verifiable)
**Notes:** Maximum coverage short of FFI; the live check_estimator run stays deferred (no maturin+pyarrow host) → UAT. Builds on existing MlrsBase + test_shims/test_params/test_estimator_checks infra.

---

## Claude's Discretion

- Boilerplate `derive`/declarative macro to emit the per-estimator retrofit — researcher to evaluate whether it pays off at ~21 estimators; hand-written retrofit acceptable. Per-estimator green-suite gate non-negotiable regardless.
- Module/file ordering within the sweep, ported-trait naming, commit granularity (per-estimator vs per-module) — planner's call.

## Deferred Ideas

- Builder/typestate boilerplate generator — promoted from Phase-12 deferred to a Phase-16 research candidate (not a scope expansion; same surface, same gates).
