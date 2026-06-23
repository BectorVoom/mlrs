//! Compile-fail fixture (BLDR-02 / D-11): an `Unfit` UMAP shell cannot be used
//! where a `Fitted` estimator is required — the state that gates the fitted-only
//! accessors (`embedding` / `n_features_in`) is unreachable without `fit`.
//!
//! `consume_fitted` accepts only `Umap<f32, Fitted>` — the state on which the
//! fitted accessors and predict/transform live. Passing a freshly-built
//! `Umap<f32, Unfit>` is an `E0308` "mismatched types: expected `Umap<f32,
//! Fitted>`, found `Umap<f32, Unfit>`" — a diagnostic that explicitly names the
//! `Unfit` state. This proves the structural gate at the value level (you cannot
//! reach the `Fitted` accessor surface from `Unfit` without `fit`), distinct
//! from the trait-bound proof in `transform_before_fit.rs`. The mismatched-types
//! form is used because it prints BOTH state arguments verbatim (method-call
//! `E0599` elides the defaulted `S=Unfit`), keeping the "mentions `Unfit`" value
//! gate robust. (File name `predict_before_fit` is retained per the Wave-0
//! contract.) Kept to ONE call (minimal `.stderr` surface, Pitfall 5).

use mlrs_algos::manifold::umap::Umap;
use mlrs_algos::typestate::{Fitted, Unfit};

fn consume_fitted(_est: Umap<f32, Fitted>) {}

fn main() {
    consume_fitted(Umap::<f32, Unfit>::new());
}
