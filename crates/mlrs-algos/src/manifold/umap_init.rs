//! `umap_init` — UMAP embedding initialization (Plan 03's home).
//!
//! This module is an EMPTY stub created in Plan 14-01 to pre-declare file
//! ownership so Plans 02 and 03 fill their own sibling files WITHOUT both
//! editing `manifold/mod.rs` (file-disjoint, parallel-safe Wave 2).
//!
//! Plan 03 fills this with `fit_ab` (the host Levenberg–Marquardt a/b curve
//! fit), `spectral_init` (Laplacian → eig → recover via the existing prims),
//! `random_init` (SplitMix64 uniform), and `noisy_scale_coords`.
//!
//! Tests live in `crates/mlrs-algos/tests/umap_test.rs` (AGENTS.md §2).
