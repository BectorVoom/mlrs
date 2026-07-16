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
// HistGradientBoosting kernels (GBT-01): sequential boosting over batched
// level-wise gradient/hessian histogram trees (row-blocked gather + reduce),
// driven by `prims/hist_gradient_boosting.rs`. Reuses `tree.rs` binning,
// cumulative-histogram and forest-traversal kernels.
pub mod gbt;
pub mod jacobi_eig;
pub mod jacobi_svd;
pub mod kmeans;
pub mod lbfgs;
// Phase-15 HDBSCAN mutual-reachability (HDBS-01, plan 15-05): the ONE new device
// kernel of the phase — a SharedMemory-free per-element 2D GATHER computing
// `out[i*n+j] = max(core_i, core_j, d_ij/alpha)` (the chebyshev_dist running-max
// shape). This file owns its `pub mod` + `pub use` (file-disjoint, single-owner,
// the distance/self-drop re-export precedent).
pub mod mutual_reachability;
pub mod reduce;
// Phase-10 SGD kernels (Wave-0 scaffold plan 10-01 owns this registration; the
// Wave-1 plan drives them from `prims/sgd.rs` — file-disjoint, parallel-safe).
// `sgd_margin` (pass 1) + `sgd_weight_update` (pass 2) are the two-pass GATHER
// idiom (single-owner, cubecl-cpu MLIR-safe); `sgd.rs` adds its own `pub use`.
pub mod sgd;
pub mod smoke;
pub mod topk;
// Random Forest level-wise tree-building + forest-inference kernels
// (ENSEMBLE-01): batched all-trees histogram builder (cuML-style row
// partitioning, gather-only, atomic-free) driven by `prims/random_forest.rs`.
pub mod tree;
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
// HistGradientBoosting kernels (GBT-01): loss gradients (squared error /
// binary log-loss / multiclass log-loss with staged softmax), row-blocked
// 3-slot histogram gather + block reduce, sklearn-gain split pipeline, row
// partition with a stage offset, and raw-prediction update/inference.
pub use gbt::{
    gbt_best_split, gbt_count_left, gbt_grad_binary, gbt_grad_multi, gbt_grad_reg, gbt_hist,
    gbt_hist_reduce, gbt_init_partition, gbt_init_raw, gbt_partition, gbt_proba_binary,
    gbt_proba_multi, gbt_row_max, gbt_row_sumexp, gbt_split_scores, gbt_sum_raw, gbt_update_raw,
};
pub use jacobi_eig::{jacobi_eig_sweep, MAX_DIM};
pub use jacobi_svd::{jacobi_svd_sweep, MAX_COLS, MAX_ROWS};
// Phase-15 HDBSCAN mutual-reachability GATHER (HDBS-01, plan 15-05): launched by
// the feature-metric/dense-cosine device front-end via the backend host wrapper
// in `prims/mutual_reachability.rs`. Re-exported under an explicit alias because
// the module and the kernel fn share the name `mutual_reachability` (a bare
// `pub use mutual_reachability::mutual_reachability` would collide the value with
// the module in this namespace); `mutual_reachability_kernel` is the launch
// symbol the backend wrapper calls.
pub use mutual_reachability::mutual_reachability as mutual_reachability_kernel;
pub use reduce::{
    argmax_shared, argmin_shared, reduce_max_plane, reduce_max_shared, reduce_min_plane,
    reduce_min_shared, reduce_sum_plane, reduce_sum_shared, reduce_sumsq_plane, reduce_sumsq_shared,
};
pub use smoke::saxpy_kernel;
// Random Forest kernels (ENSEMBLE-01): binning, level-wise histogram/split
// pipeline, row partition, and forest traversal/vote. Launched by the backend
// host orchestrator in `prims/random_forest.rs`.
pub use tree::{
    rf_best_split, rf_bin_features, rf_count_left, rf_hist_class, rf_hist_cum, rf_hist_reg,
    rf_mean_reg, rf_node_max, rf_node_total, rf_partition, rf_predict_leaf,
    rf_split_scores_class, rf_split_scores_reg, rf_vote_class, RF_NO_FEATURE,
};
// Phase-14 UMAP layout SGD step (UMAP-03): the per-owner GATHER kernel the host
// epoch driver in `manifold/umap.rs` launches each epoch (Plan 04) and the
// `transform` frozen-subset path reuses (Plan 05).
pub use umap_layout::umap_layout_step;
