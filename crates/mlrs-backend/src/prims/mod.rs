//! `prims` — host-side orchestration for the Phase-2 compute primitives.
//!
//! Each primitive's host API (shape validation, pool-routed scratch/out
//! buffers, kernel launch, device-resident result) lives in its own module
//! here. The device kernels themselves stay in the feature-free `mlrs-kernels`
//! crate (D-13); this layer owns the concrete `ActiveRuntime` and the launch
//! wrappers.
//!
//! Tests live in `crates/mlrs-backend/tests/` (never an in-source
//! `#[cfg(test)]` module — AGENTS.md §2).

// LinearRegression's large-`n_samples` Gram+eig path (LINEAR-01) needs JUST
// the device-resident centered matrix + mean (not the full Gram/scale that
// `covariance.rs` produces) — extracted as its own composition of
// `column_reduce` + `center_columns` so that path doesn't hand-roll the
// unsafe kernel-launch dance in the algos layer.
pub mod center;
pub mod cholesky;
pub mod covariance;
// LinearRegression Gram+eig path perf lever (LINEAR-01, D-02): row-blocked
// shared-memory XᵀX/Xᵀy formation replacing the skinny-output/huge-K `gemm`
// pair (the `kmeans.rs` "GEMM sums" pathology, same fix applied) — falls back
// to the original `gemm` formation on the cpu backend (SharedMemory-unsafe
// there, the `use_shared_sums` precedent).
pub mod gram;
// Phase-7 prim stubs (Wave-0 scaffold owns these registrations; plans 07-02
// (rng) / 07-03 (incremental_svd) fill their own file body — file-disjoint,
// parallel-safe). Each is an empty compiling module until its plan adds the
// launch/host-glue wrapper + a `pub use` of its symbol INSIDE that file.
pub mod incremental_svd;
pub mod rng;
// Phase-8 prim stub (Wave-0 scaffold 08-01 owns this registration; the Wave-1
// plan 08-02 fills the file body — file-disjoint, parallel-safe). The
// `Kernel<F>` enum + `kernel_matrix` host-fn signature compile today (geometry
// validation real; compute path `todo!()` until 08-02).
pub mod kernel_matrix;
// Phase-13 KNN-graph primitive (PRIM-11). Wave-1 scaffold plan 13-01 owns this
// registration; plan 13-03 fills the file body — the `Metric` enum +
// `knn_graph` host orchestrator composing `distance`/`topk` + the new
// `mlrs-kernels::distance` direct + self-drop kernels (file-disjoint,
// single-owner). Empty compiling shell until then; the oracle harness in
// `tests/knn_graph_test.rs` (plan 13-01) is RED until `Metric`/`knn_graph` land.
pub mod knn_graph;
// Phase-9 prim stub (Wave-0 scaffold 09-01 owns this registration; the Wave-1
// plan 09-02 fills the file body — file-disjoint, parallel-safe). The
// `laplacian` host-fn signature compiles today (geometry validation real;
// compute path `todo!()` until 09-02).
pub mod laplacian;
// Phase-15 HDBSCAN mutual-reachability device front-end (HDBS-01, plan 15-05): the
// host-launch wrapper for the `mlrs-kernels::mutual_reachability` GATHER kernel
// (dense `n×n` MR `out[i*n+j] = max(core_i, core_j, d_ij/alpha)`). Owns the
// concrete `ActiveRuntime` + the validate-before-launch guard. File-disjoint,
// single-owner (the prim re-export precedent).
pub mod mutual_reachability;
// Phase-5 prim stubs (Wave-0 scaffold owns these registrations; plans
// 05-02..06 fill their own file body — file-disjoint, parallel-safe). Each is an
// empty compiling module until its plan adds the launch wrapper + a `pub use` of
// its symbol INSIDE that file.
pub mod coordinate_descent;
pub mod dbscan;
pub mod distance;
// Random Forest prim (ENSEMBLE-01): the launch-only batched level-wise forest
// builder + forest inference over the `mlrs-kernels::tree` kernels. Owns the
// host quantile binning, seeded bootstrap/feature-subsample RNG (SplitMix64),
// and the validate-before-launch guards.
pub mod random_forest;
// HistGradientBoosting (GBT-01): launch-only sequential boosting over the
// batched level-wise histogram tree pipeline (`mlrs-kernels::gbt` +
// `tree.rs` binning/traversal reuse).
pub mod hist_gradient_boosting;
pub mod eig;
pub mod gemm;
pub mod kmeans;
// Dense linear-model inference perf lever (LINEAR-01/02): a single fused
// GATHER matvec+bias launch (`mlrs_kernels::linear_predict`) replacing the
// shared `gemm→to_host→host bias-loop→from_host` predict round-trips (the
// `center`/`gram` host-sync pathology, same fix). GATHER-only, so no cpu
// fallback branch. Consumed by Ridge/LinearRegression/ElasticNet/Lasso predict.
pub mod linear_predict;
pub mod lbfgs;
pub mod reduce;
// Phase-10 SGD solver prim (PRIM-10). `sgd_solve` is fully implemented: a
// validate-before-launch geometry guard fronts a host epoch loop that drives the
// two SharedMemory-free `sgd_margin` / `sgd_weight_update` kernels per minibatch,
// with host-side dloss / schedule / L2+L1 penalty arithmetic. It takes FLAT
// scalar params, NOT the algos `SgdConfig` (mlrs-backend does not depend on
// mlrs-algos).
pub mod sgd;
pub mod svd;
pub mod topk;
// TSNE-01: the exact-method t-SNE per-iteration gradient prim (Student-t
// affinity + KL-gradient GATHER over the Phase-2 distance prim).
pub mod tsne;
