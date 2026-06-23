//! Compile-fail gate (D-11) — the MANDATORY structural proof of BLDR-02.
//!
//! This `trybuild` harness asserts that the `tests/ui/*.rs` fixtures DO NOT
//! compile. Each fixture exercises a `Fitted`-only surface on an `Unfit` UMAP
//! shell, and each golden `.stderr` explicitly names the `Unfit` state:
//!   - `transform_before_fit.rs` — requires the `Transform<f32>` bound on
//!     `Umap<f32, Unfit>`; `Transform` is implemented ONLY for
//!     `Umap<F, Fitted>`, so this is an `E0277` "trait bound ... not satisfied"
//!     naming `Umap<f32, Unfit>` (vs the `Umap<f32, Fitted>` impl).
//!   - `predict_before_fit.rs` — passes a `Umap<f32, Unfit>` where a
//!     `Umap<f32, Fitted>` value is required; this is an `E0308` "mismatched
//!     types" naming both `Fitted` and `Unfit`, proving the fitted accessor
//!     surface is unreachable from `Unfit`.
//! The trait-bound / mismatched-types forms are used in preference to a bare
//! method call because rustc ELIDES the defaulted `S = Unfit` type argument in
//! `E0599` method-not-found messages (the receiver prints as `Umap<f32>`),
//! whereas `E0277`/`E0308` print the `Unfit` argument verbatim — making the
//! value gate (a compile-fail diagnostic mentioning `Unfit`) robust.
//!
//! The value gate is NON-COMPILATION that references the `Unfit` state — NOT a
//! specific error code (`E0599`, `E0277`, and `E0308` are all acceptable). If a
//! future edit ever re-exposes a fitted method/trait on `Unfit`, the
//! corresponding ui file would start compiling and this test would FAIL — that
//! is the BLDR-02 regression guard (T-12-05).
//!
//! ## Toolchain sensitivity (Pitfall 5)
//! The golden `.stderr` files were generated with `TRYBUILD=overwrite` under
//! the repo's pinned toolchain (`rust-toolchain.toml` channel = `stable`;
//! generated against rustc 1.96.0). `.stderr` text is toolchain-sensitive:
//! rustc wording changes across versions can require regenerating the goldens
//! with `TRYBUILD=overwrite cargo test -p mlrs-algos --features cpu --test compile_fail`.
//! The VALUE gate is non-compilation that references the `Unfit` state — not a
//! specific error code. Each ui fixture is kept to ONE method call to minimise
//! the `.stderr` surface and keep the goldens stable.
//!
//! ## Feature requirement
//! Run under a backend feature (`--features cpu`); the ui fixtures import
//! `mlrs_backend`-backed types via `mlrs_algos`, so a missing backend feature
//! would surface as an unrelated `E0432`/`E0463` (Pitfall 5), not the intended
//! `Unfit` diagnostic.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
