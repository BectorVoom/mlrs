---
phase: 12-builder-typestate-convention-foundation
plan: 01
subsystem: mlrs-algos
tags: [typestate, traits, builder-foundation, sealed-trait, trybuild]
requires:
  - "crates/mlrs-algos/src/traits.rs (FROZEN — copied body shape, did not edit)"
  - "crates/mlrs-algos/src/error.rs (AlgoError return type)"
provides:
  - "mlrs_algos::typestate module — sealed State trait, Unfit/Fitted ZST markers"
  - "mlrs_algos::typestate::{Fit, Predict, Transform, PartialFit} consuming traits (Fit + PartialFit carry type Fitted)"
  - "trybuild = 1.0.117 dev-dependency on mlrs-algos (for Plan 03 compile-fail gate)"
affects:
  - "Plan 02 (UMAP/HDBSCAN shells import the new traits)"
  - "Plan 03 (trybuild compile-fail gate)"
  - "Plan 04 (PyO3 collapse)"
  - "Phase 16 (retrofit of all 30 estimators)"
tech-stack:
  added:
    - "trybuild 1.0.117 (dev-dependency)"
  patterns:
    - "Sealed marker trait (private Sealed supertrait) for a closed state set"
    - "Zero-sized type markers (Unfit/Fitted) carried as PhantomData<S>"
    - "Consuming-self Fit/PartialFit with associated type Fitted (typestate transition)"
key-files:
  created:
    - "crates/mlrs-algos/src/typestate.rs"
    - "crates/mlrs-algos/tests/typestate_test.rs"
  modified:
    - "crates/mlrs-algos/src/lib.rs"
    - "crates/mlrs-algos/Cargo.toml"
    - "Cargo.lock"
decisions:
  - "lib.rs registers `pub mod typestate;` but deliberately does NOT glob-re-export typestate::* alongside traits::* — the new trait names collide with the frozen surface, so consumers use the explicit `mlrs_algos::typestate::` path (Pitfall 1, D-07)"
  - "PartialFit is defined-but-unused in Phase 12 — declared as the multi-transition (Unfit→Fitted→Fitted) Phase-16 target for streaming estimators"
metrics:
  duration_min: 3
  completed: 2026-06-23T01:28:03Z
  tasks: 3
  files: 5
---

# Phase 12 Plan 01: Builder Typestate Convention Foundation Summary

Authored the canonical `mlrs_algos::typestate` surface — a sealed `State` marker
trait with `Unfit`/`Fitted` ZSTs and the four consuming lifecycle traits
(`Fit`/`Predict`/`Transform`/`PartialFit`) — coexisting with the frozen
`traits.rs` and adding the `trybuild` dev-dep for the Wave-3 compile-fail gate.

## What Was Built

A new file `crates/mlrs-algos/src/typestate.rs` (200 lines) providing the
type-level lifecycle machinery the v3 builder API is built on, wired into
`lib.rs` without colliding with the legacy `traits::*` re-export, plus a
runtime/compile smoke test and the `trybuild` dev-dependency.

## Exact Trait Signatures Authored (for Plans 02/03/04 to import)

Import path for all of these: `use mlrs_algos::typestate::{...};`

```rust
// Sealed marker set (closed; downstream cannot extend)
mod sealed { pub trait Sealed {} }       // private
pub trait State: sealed::Sealed {}
pub struct Unfit;                         // ZST — freshly built, not yet fit
pub struct Fitted;                        // ZST — predict/transform live here
impl State for Unfit {}                   // (+ Sealed)
impl State for Fitted {}                  // (+ Sealed)

pub trait Fit<F> where F: Float + CubeElement + Pod {
    type Fitted;
    fn fit(self,                          // CONSUMES self (D-05)
           pool: &mut BufferPool<ActiveRuntime>,
           x: &DeviceArray<ActiveRuntime, F>,
           y: Option<&DeviceArray<ActiveRuntime, F>>,
           shape: (usize, usize)) -> Result<Self::Fitted, AlgoError>;
}

pub trait Predict<F> where F: Float + CubeElement + Pod {
    fn predict(&self,                     // borrows &self
               pool: &mut BufferPool<ActiveRuntime>,
               x: &DeviceArray<ActiveRuntime, F>,
               shape: (usize, usize)) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

pub trait Transform<F> where F: Float + CubeElement + Pod {
    fn transform(&self,                   // borrows &self
                 pool: &mut BufferPool<ActiveRuntime>,
                 x: &DeviceArray<ActiveRuntime, F>,
                 shape: (usize, usize)) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError>;
}

pub trait PartialFit<F> where F: Float + CubeElement + Pod {
    type Fitted;
    fn partial_fit(self,                  // CONSUMES self; multi-transition (D-06)
                   pool: &mut BufferPool<ActiveRuntime>,
                   x: &DeviceArray<ActiveRuntime, F>,
                   y: Option<&DeviceArray<ActiveRuntime, F>>,
                   shape: (usize, usize)) -> Result<Self::Fitted, AlgoError>;
}
```

A `#[doc(hidden)] pub fn _state_phantom<S: State>() -> PhantomData<S>` zero-cost
helper is also exported for shell authors; it produces no runtime code.

Note: `Predict`/`Transform` differ from the legacy `traits.rs` only in module
path (signatures are identical — `&self` → `DeviceArray`). `Fit`/`PartialFit`
differ in BOTH the path AND the signature (`&mut self`/`&mut Self` → consuming
`self`/`Self::Fitted`).

## The lib.rs Non-Glob Decision

`lib.rs` now contains `pub mod typestate;` (with an explanatory comment) but
does **not** add `pub use typestate::{...}`. The legacy re-export
`pub use traits::{Fit, ..., Transform};` stays as-is. The two surfaces share
trait names (`Fit`/`Predict`/`Transform`/`PartialFit`), so globbing both into the
crate root would be an ambiguous-name collision (Pitfall 1). Consumers of the new
surface therefore write the explicit path: `use mlrs_algos::typestate::Fit;`.

## traits.rs Untouched (confirmed)

`git diff HEAD~3 -- crates/mlrs-algos/src/traits.rs` is empty — the frozen
trait file is byte-for-byte unchanged across all three commits (T-12-02). The
full `cargo build -p mlrs-algos --features cpu` succeeds, proving all 30 existing
estimators still compile against the untouched legacy surface.

## Tasks & Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Author sealed typestate module + wire lib.rs | `30a0220` | `typestate.rs`, `lib.rs` |
| 2 | Add trybuild dev-dependency | `139a3db` | `Cargo.toml`, `Cargo.lock` |
| 3 | Runtime smoke test markers compose | `fc9f914` | `tests/typestate_test.rs` |

Task 1 commits `typestate.rs` together with the `lib.rs` registration because the
new module only compiles once it is registered (the two are inseparable for a
green build); Task 2's `Cargo.toml` + `Cargo.lock` are a self-contained
dependency add.

## Verification Results

- `cargo build -p mlrs-algos --features cpu` — succeeds.
- `cargo test -p mlrs-algos --features cpu --test typestate_test` — 3 passed
  (`markers_are_zero_sized`, `markers_satisfy_sealed_state_bound`,
  `typestate_module_is_importable`).
- `cargo tree -p mlrs-algos -e dev | grep trybuild` — `trybuild v1.0.117`.
- `git diff -- crates/mlrs-algos/src/traits.rs` — empty (frozen).

## Deviations from Plan

None — plan executed exactly as written. All Rules 1–4 were no-ops (pure
type-level machinery, no input surface, no architectural change).

## Known Stubs

None. `PartialFit` is intentionally defined-but-unused this phase (declared as the
Phase-16 streaming-estimator target, D-06) — this is a documented forward
declaration, not a stub blocking the plan goal.

## Self-Check: PASSED

- FOUND: crates/mlrs-algos/src/typestate.rs
- FOUND: crates/mlrs-algos/tests/typestate_test.rs
- FOUND: commit 30a0220 (Task 1)
- FOUND: commit 139a3db (Task 2)
- FOUND: commit fc9f914 (Task 3)
