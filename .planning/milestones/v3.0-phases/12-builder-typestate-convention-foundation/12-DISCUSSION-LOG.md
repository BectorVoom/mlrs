# Phase 12: Builder + Typestate Convention Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-23
**Phase:** 12-Builder + Typestate Convention Foundation
**Areas discussed:** Typestate encoding, Additive-retrofit shape, Single-source defaults, UMAP/HDBSCAN shell scope

---

## Typestate encoding

| Option | Description | Selected |
|--------|-------------|----------|
| Generic state wrapper | One shared `Unfit<E>`/`Fitted<E>` newtype pair over the estimator `E`; retrofit becomes a pure wrap. | |
| Per-estimator state param | Marker param on each struct: `Umap<F, S = Unfit>` + `PhantomData<S>`; predict on `impl Umap<F, Fitted>`. Textbook typestate, signature-touching retrofit. | ✓ |
| Two distinct types | `Umap<F>` (config) → `FittedUmap<F>`; no marker types but ~2× type surface. | |

**User's choice:** Per-estimator state param.
**Notes:** Chosen knowingly over the shared wrapper despite the heavier signature-touching Phase-16 retrofit. `fit` consumes `self` to re-tag the marker. Surfaced consequence: `any_estimator!` fitted arms must spell `T<f32, Fitted>` explicitly since `S` defaults to `Unfit`.

---

## Additive-retrofit shape (trait reconciliation)

| Option | Description | Selected |
|--------|-------------|----------|
| Traits stay; typestate is inherent | Keep old `Fit`/`Predict`/`Transform` unchanged; add inherent typestate methods that delegate. Most additive Phase-16 retrofit. | |
| Redefine traits as typestate-aware | Canonical `Fit` consumes `self` with `type Fitted`; predict/transform on fitted-bound trait. One unified surface; Phase 16 rewrites each impl. | ✓ |
| Parallel typed traits | New `TypedFit`/`TypedPredict` coexist permanently beside old. | |

**User's choice:** Redefine traits as typestate-aware.
**Notes:** Wants a single unified surface as the end-state, accepting per-estimator migration risk in Phase 16 (suites are the safety net). Surfaced sequencing constraint (see Transition below) — redefining in place would break the 30 existing estimators in Phase 12, so the new surface is introduced by coexistence and converges to single only at end of Phase 16.

### PartialFit reconciliation

| Option | Description | Selected |
|--------|-------------|----------|
| Design it in now | `partial_fit(self) -> Result<Self::Fitted>` on both `T<Unfit>` and `T<Fitted>`; multi-transition convention with a defined Phase-16 target. | ✓ |
| Defer, document constraint | One-shot `Fit` only in Phase 12; PartialFit reconciliation noted as open Phase-16 sub-decision. | |
| PartialFit keeps &mut self | Incremental estimators a permanent documented exception outside typestate. | |

**User's choice:** Design it in now.
**Notes:** Incremental estimators (`IncrementalPCA`, `MBSGD*`) modeled as `Unfit → Fitted → Fitted`, predict allowed between batches.

### Transition (introducing the new surface without breaking the old)

| Option | Description | Selected |
|--------|-------------|----------|
| Coexist, same names, new module | New typestate-aware traits in `mlrs_algos::typestate`; old `traits.rs` untouched; UMAP/HDBSCAN impl only new; Phase 16 migrates + deletes old at end. | ✓ |
| Coexist, distinct names | New traits get interim distinct names (`FitInto`/`PredictOn`); Phase 16 ends with a rename pass. | |
| Redefine in place + pilot retrofit now | Redefine `traits.rs` and retrofit 1–2 pilots in Phase 12. Pulls BLDR-03 forward, breaches phase boundary. | |

**User's choice:** Coexist, same names, new module.
**Notes:** Honors Success Criterion 3 (all existing `any_estimator!` call sites keep compiling/passing). End-state is the single redefined surface; names collide only by path during transition.

---

## Single-source defaults

| Option | Description | Selected |
|--------|-------------|----------|
| Default on the builder | `impl Default for Builder` is the single source; `new()` = `builder().build().expect(..)`. Most idiomatic. | |
| new() is canonical | `new()` sets sklearn defaults as a struct literal; builder's `Default` re-derives via `new().into_builder()`. | ✓ |
| Const default table | Associated `const DEFAULTS` both new() and builder read. | |

**User's choice:** new() is canonical.
**Notes:** `new()` returns `T<F, Unfit>`, sets `_state: PhantomData`, constructs directly trusting defaults as valid, so `new() == builder().build()?` holds. `new()` retained as the zero-arg sklearn-default shortcut.

---

## UMAP/HDBSCAN shell scope

| Option | Description | Selected |
|--------|-------------|----------|
| Full shape, stub body | Real param + fitted-attr surface, `todo!()` fit. Compile-fail gate only. | |
| Minimal skeleton | A few representative params + generic fitted attr + stub fit. | |
| Full shape + trivial fit | Full param + accessor surface AND a non-algorithmic real fit (set `n_features_in_`, zeros embedding / all-noise labels) so a round-trip RUNS. | ✓ |

**User's choice:** Full shape + trivial fit.
**Notes:** Gives a runtime end-to-end round-trip test plus the compile-fail (predict-before-fit) gate. UMAP → new `manifold/` module; HDBSCAN → existing `cluster/`; PyO3 shells via `any_estimator!`.

---

## Claude's Discretion

- Exact naming/bounds of the sealed `State` trait and `Unfit`/`Fitted` marker types.
- New trait module filename (`typestate.rs` or other).
- Whether HDBSCAN's PyO3 shell extends `cluster.rs` or gets its own file.

## Deferred Ideas

- Builder/typestate boilerplate generator (derive/declarative macro) — considered, not decided; belongs with Phase-16 retrofit-sweep planning.
- Old-trait deletion / final single-surface convergence — end of Phase 16, not Phase 12.
