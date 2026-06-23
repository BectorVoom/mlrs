//! Compile-fail fixture (BLDR-02 / D-11): an `Unfit` UMAP shell does NOT satisfy
//! the [`Transform`] bound, so `transform`-before-`fit` cannot type-check.
//!
//! `Transform<F>` is implemented ONLY for `Umap<F, Fitted>` (never for the
//! `Unfit` state), so requiring `Umap<f32, Unfit>: Transform<f32>` is an `E0277`
//! "the trait bound `Umap<f32, Unfit>: Transform<f32>` is not satisfied" — a
//! diagnostic that explicitly names the `Unfit` state. The trait-bound form is
//! used (rather than a method call) because rustc elides the defaulted `S=Unfit`
//! type argument in `E0599` method-not-found messages, whereas the `E0277` bound
//! prints the `Unfit` argument verbatim — making the value gate (a compile-fail
//! diagnostic that mentions `Unfit`) robust. Kept to ONE assertion (minimal
//! `.stderr` surface, Pitfall 5).

use mlrs_algos::manifold::umap::Umap;
use mlrs_algos::typestate::{Transform, Unfit};

fn assert_transform<T: Transform<f32>>() {}

fn main() {
    assert_transform::<Umap<f32, Unfit>>();
}
