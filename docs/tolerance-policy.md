# f32 / f64 Tolerance Policy

This document defines the numerical-comparison policy used by the scikit-learn
oracle harness in `mlrs-core` (requirement **FOUND-08**, decisions **D-08** and
**D-09**). It is the contract every downstream phase's oracle tests rely on.

## Single global default (D-08)

The project starts with a **single global tolerance**, not a populated
per-estimator-family table:

| Constant   | `abs`  | `rel`  | Used for                          |
| ---------- | ------ | ------ | --------------------------------- |
| `F32_TOL`  | `1e-5` | `1e-5` | `f32`-precision oracle comparison |
| `F64_TOL`  | `1e-5` | `1e-5` | `f64`-precision oracle comparison |

Both are `Tolerance { abs: 1e-5, rel: 1e-5 }` (`crates/mlrs-core/src/tolerance.rs`),
matching the project's core value: results must match scikit-learn within `1e-5`.
`F64_TOL` is kept as a separate constant from `F32_TOL` so the f64 path can be
tightened independently later without touching call sites.

## Comparison rule: abs AND rel (D-09)

`compare::is_close(got, expected, tol)` requires **both** the absolute and the
relative error to pass — the stricter form, *not* numpy's `abs OR rel`:

```text
abs_err = |got - expected|
rel_err = abs_err / |expected|
pass    = abs_err <= tol.abs  AND  rel_err <= tol.rel
```

Exact-equal values (including matching infinities) pass immediately. Any `NaN`,
or a non-matching infinity, never compares close.

## Near-zero guard

When `|expected|` is very small the relative term `abs_err / |expected|`
explodes even for a genuinely-correct result, so the "both must pass" rule
would spuriously fail. To prevent that, `is_close` falls back to an
**absolute-only** check when:

```text
|expected| < NEAR_ZERO_FLOOR   (NEAR_ZERO_FLOOR = 1e-8)
```

### Why `1e-8`?

`1e-8` sits three orders of magnitude **below** the `1e-5` absolute tolerance.
Because the floor is below `tol.abs`, every value the guard admits is already
within the absolute bound — the guard therefore only ever *suppresses spurious
relative-error failures near zero*; it can never loosen the absolute check or
let a genuinely-wrong value pass. This is covered by an explicit test
(`tests/compare_test.rs::near_zero_guard_falls_back_to_abs_only`) per threat
T-02-02.

## Growth path: per-family tolerances

D-08 satisfies FOUND-08's "per-family policy" with a *structure that can grow
rows*, not a populated table. The growth point is:

```rust
Tolerance::for_family(family: &str) -> Tolerance
```

Today it returns the global default for **every** family. When a future
estimator family (e.g. `"pca"`, `"kmeans"`) demonstrates it needs looser bounds
than `1e-5`, add a `match family { ... }` arm inside `for_family`. Call sites
that already pass a family name pick up the new row automatically — no
restructuring required. Per-family tables are introduced in Phase 3/4/5, only
when a family proves it needs them (deferred per D-08).
