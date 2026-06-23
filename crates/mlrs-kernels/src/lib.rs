//! `mlrs-kernels` — generic CubeCL compute kernels, backend-feature-free.
//!
//! Every kernel here is generic over the float type (`<F: Float>`) and is
//! launched generic over the runtime (`::launch::<F, R>`). This crate MUST NOT
//! depend on any CubeCL backend runtime feature (Criterion 1); a concrete
//! runtime is chosen in `mlrs-backend`.

pub mod cholesky;
// Phase-13 KNN-graph direct distance + self-drop kernels (PRIM-11). Wave-1
// scaffold plan 13-01 owns this registration; plan 13-02 fills the file body
// (the direct pairwise feature-loop distance kernels + the per-row self-drop
// GATHER kernel) and adds its own `pub use distance::{…}` re-export INSIDE that
// plan's edit — file-disjoint, single-owner. Empty compiling module until then.
pub mod distance;
// Phase-5 kernel stubs (Wave-0 scaffold owns these registrations; plans
// 05-02..06 fill their own file body — file-disjoint, parallel-safe). Each is an
// empty compiling module until its plan adds the `#[cube]` kernel + a `pub use`
// of its symbol INSIDE that file.
pub mod coordinate;
pub mod dbscan;
pub mod elementwise;
pub mod jacobi_eig;
pub mod jacobi_svd;
pub mod kmeans;
pub mod lbfgs;
pub mod reduce;
// Phase-10 SGD kernels (Wave-0 scaffold plan 10-01 owns this registration; the
// Wave-1 plan drives them from `prims/sgd.rs` — file-disjoint, parallel-safe).
// `sgd_margin` (pass 1) + `sgd_weight_update` (pass 2) are the two-pass GATHER
// idiom (single-owner, cubecl-cpu MLIR-safe); `sgd.rs` adds its own `pub use`.
pub mod sgd;
pub mod smoke;
pub mod topk;
// Phase-14 UMAP layout (UMAP-03): the ONE new device kernel of the phase —
// `umap_layout_step` is a vertex-owner GATHER SGD step (cpu-MLIR-safe, frozen-
// subset-capable, host-drawn negative samples). This file owns its `pub mod` +
// `pub use` (file-disjoint, single-owner — the sgd/topk re-export precedent).
pub mod umap_layout;

pub use cholesky::cholesky_solve;
// Phase-13 KNN-graph (PRIM-11): direct pairwise distance kernels + per-row
// index-identity self-drop GATHER. Plan 13-02 owns this re-export (file-disjoint,
// single-owner) alongside the kernel bodies in `distance.rs`.
pub use distance::{chebyshev_dist, manhattan_dist, minkowski_dist, self_drop_gather};
pub use elementwise::{
    center_columns, clamp_nonneg, degree_guard, dist_combine_clamp, div_by_row, kde_cosine_map,
    kde_epanechnikov_map, kde_exponential_map, kde_gaussian_map, kde_linear_map, kde_tophat_map,
    laplacian_map, poly_map, rbf_map, scale, sigmoid_map, sqrt_elem, zero_diag_copy,
};
pub use jacobi_eig::{jacobi_eig_sweep, MAX_DIM};
pub use jacobi_svd::{jacobi_svd_sweep, MAX_COLS, MAX_ROWS};
pub use reduce::{
    argmax_shared, argmin_shared, reduce_max_plane, reduce_max_shared, reduce_min_plane,
    reduce_min_shared, reduce_sum_plane, reduce_sum_shared, reduce_sumsq_plane, reduce_sumsq_shared,
};
pub use smoke::saxpy_kernel;
// Phase-14 UMAP layout SGD step (UMAP-03): the per-owner GATHER kernel the host
// epoch driver in `manifold/umap.rs` launches each epoch (Plan 04) and the
// `transform` frozen-subset path reuses (Plan 05).
pub use umap_layout::umap_layout_step;
