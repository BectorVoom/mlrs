//! Compile-fail fixture (BLDR-02 / D-11): an `Unfit` UMAP shell does NOT satisfy
//! the `Fitted`-only lifecycle bound, so the fitted surface is unreachable
//! before `fit`.
//!
//! This fixture uses the `E0277` trait-bound assertion style (WR-05) rather than
//! the previous `E0308` mismatched-types form. The earlier form rendered the
//! receiver as `Umap<f32>` on its PRIMARY diagnostic line (rustc elides the
//! defaulted `S = Unfit`) and only named `Unfit` on a secondary "found struct"
//! note — a brittle golden whose primary line a future rustc could change
//! without naming `Unfit` at all. The `E0277` trait-bound form prints the
//! `Unfit` argument VERBATIM on the primary line, so the value gate
//! (non-compilation that references the `Unfit` state) stays robust across
//! stable rustc wording drift.
//!
//! `Transform<F>` is implemented ONLY for `Umap<F, Fitted>` (never for the
//! `Unfit` state), so requiring `Umap<f32, Unfit>: Transform<f32>` is an `E0277`
//! "the trait bound ... is not satisfied" that explicitly names `Umap<f32,
//! Unfit>` — proving the fitted-only surface (predict/transform/accessors) is
//! unreachable from `Unfit`. (File name `predict_before_fit` is retained per the
//! Wave-0 contract.) Kept to ONE assertion (minimal `.stderr` surface,
//! Pitfall 5).

use mlrs_algos::manifold::umap::Umap;
use mlrs_algos::typestate::{Transform, Unfit};

fn assert_fitted_surface<T: Transform<f32>>() {}

fn main() {
    assert_fitted_surface::<Umap<f32, Unfit>>();
}
