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
// Host arm of the same normal-equations formation (RIDGE-POS-PERF-CPU): column
// means + centered Gram/Xᵀy computed straight from host memory, for the cpu
// backend where `center_columns` falls back to the per-column-round-trip
// `column_reduce` and every launch costs an OS thread spawn.
pub mod gram_host;
// The parallel host Gaussian-mixture EM engine (MIX-01). Unlike the other
// `*_host` prims this one is not a cpu-arm twin of a device kernel — it is the
// WHOLE algorithm on every backend, because the EM loop is launch-bound,
// `f64`-bound, and has an `O(k·d³)` serial factorization tail. See its module
// docs for the three structural wins over sklearn's own implementation.
pub mod gmm_host;
// DEVICE arm of the mixture EM loop (MIX-GPU), for large `n` on backends with
// genuine `f64` kernels AND `f64` transcendentals (cuda/rocm in practice — see
// `gmm_device_applicable`). Keeps `gmm_host`'s `O(k·d³)` Cholesky/triangular-
// inverse tail entirely host-side and moves only the two `n`-scaling passes
// (E-step, M-step covariance) onto device-resident `X`/`resp`, mirroring how
// `normal_eq` relates to `gram`: a device twin that is a DROP-IN replacement
// for the corresponding host engine's per-iteration calls, not a rewrite of
// the whole algorithm. See its own module docs for the full design.
pub mod gmm_device;
// DEVICE arm of the same normal-equations formation, pinned to `f64`
// accumulation whatever the estimator's own width (BAYES-GPU). `gram`'s Gram
// accumulates in the element type, which `BayesianRidge`'s residual identity
// cannot tolerate; this module widens on the device and keeps the whole
// reduction there, so the design never round-trips to the host.
pub mod normal_eq;
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
// Fused KNN neighbor-target gather (KNN-01 perf lever): keeps the brute-force
// KNN predict pipeline device-resident by forming the neighbor mean on-device
// instead of reading the indices back and looping on the host.
pub mod knn;
pub mod knn_graph;
// KNN-HOST: the plain-Rust worker-pool `distance -> top-k` scan that serves the
// cpu backend for EVERY metric, any `k` and any `n_features` — the rectangle the
// tuned `knn::cpu_rows_topk` kernel cannot cover, where the GPU-shaped
// composition it fell through to lost to sklearn by up to 39x.
pub mod knn_host;
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
// Non-negative ridge solve (`Ridge(positive=True)`): drives the single-cube
// `ridge_nnls_cd` kernel over the device-resident Gram, replacing the read-back
// /host-CD/re-upload round-trip. Owns its dispatch predicate so the algos-crate
// host twin stays the cpu + over-cap arm. File-disjoint, single-owner.
pub mod nnls;
// Phase-5 prim stubs (Wave-0 scaffold owns these registrations; plans
// 05-02..06 fill their own file body — file-disjoint, parallel-safe). Each is an
// empty compiling module until its plan adds the launch wrapper + a `pub use` of
// its symbol INSIDE that file.
pub mod coordinate_descent;
pub mod dbscan;
pub mod distance;
// Feature-selection column moments (FSEL-01): the three `f64` host sweeps every
// univariate score is assembled from (per-class column sums, X-vs-y cross
// moments, NaN-aware column moments), plus the device column gather/scatter that
// IS a selector's `transform` / `inverse_transform`. Host-accumulating for the
// same reason `gmm_host` and `special` are: the 1e-5 contract is RELATIVE and
// these scores' p-values reach 1e-27, cuda does not advertise `f64`, and the
// sweep is one-shot per `fit`. See the module docs.
pub mod feature_score;
// Random Forest prim (ENSEMBLE-01): the launch-only batched level-wise forest
// builder + forest inference over the `mlrs-kernels::tree` kernels. Owns the
// host quantile binning, seeded bootstrap/feature-subsample RNG (SplitMix64),
// and the validate-before-launch guards.
pub mod random_forest;
// HistGradientBoosting (GBT-01): launch-only sequential boosting over the
// batched level-wise histogram tree pipeline (`mlrs-kernels::gbt` +
// `tree.rs` binning/traversal reuse).
pub mod hist_gradient_boosting;
// GBT-PERF-CPU: the cpu HOST arm of that fit — the same algorithm replayed
// bit-identically on a persistent worker pool, because `cubecl-cpu` runs one
// OS thread per unit and the level pipeline is thousands of tiny launches.
pub(crate) mod hgb_host;
// The shared host worker-pool primitives (barrier + disjoint-slice handle +
// the reusable task-dispatch pool) every cpu arm synchronizes on; extracted
// from `hgb_host`, which was where they were first measured and tuned.
pub(crate) mod host_pool;
// Runs a host prim's hot region on the machine's REAL vector unit. The crate is
// compiled for the x86-64 BASELINE (no `target-cpu`), i.e. SSE2, while
// `cubecl-cpu` JITs for the host — which is why a `-O0` JIT kernel could
// outrun `-O3` Rust. First measured in `knn_host` (1.6-1.9x), then applied
// across the host arms.
pub mod host_simd;
// Huber-regression primal objective evaluator (HUBER-01): the margin matvec,
// the inlier/outlier classification against `ε·σ`, the three scalar reductions
// the `σ` derivative needs, and the `X̃ᵀg` gradient — all out of ONE fused pass
// where scikit-learn's NumPy form walks the design five times and fancy-index
// COPIES it twice per evaluation. Same cpu-host / device-GEMM split as
// `svm_objective`, and the same reason for it.
pub mod huber_objective;
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
// `radius_neighbors`' DEVICE arm (NEIGH-RADIUS-GPU): drives the
// `mlrs-kernels::radius` count + ordered-compaction pair over a device-resident
// distance tile, so only the matches cross the bus. File-disjoint,
// single-owner (the `dbscan` prim precedent).
pub mod radius;
// `radius_neighbors`' HOST arm (NEIGH-RADIUS-HOST): the fused, worker-pool
// distance→threshold scan the cpu backend runs, sharing `knn_host`'s
// vectorized per-metric lane loop.
pub mod radius_host;
// `RANSACRegressor`'s compute engine (RANSAC-01): the per-trial fused
// matvec→loss→threshold scan over a persistent worker pool, the consensus R²,
// and the `min_samples × d` one-sided-Jacobi least-squares solve each
// sub-sample poses. HOST on every backend — a hundred trials of a launch-bound
// `n × d` pass, each of which the NEXT draw's stopping rule must read back.
pub mod ransac_host;
pub mod reduce;
// Phase-10 SGD solver prim (PRIM-10). `sgd_solve` is fully implemented: a
// validate-before-launch geometry guard fronts a host epoch loop that drives the
// two SharedMemory-free `sgd_margin` / `sgd_weight_update` kernels per minibatch,
// with host-side dloss / schedule / L2+L1 penalty arithmetic. It takes FLAT
// scalar params, NOT the algos `SgdConfig` (mlrs-backend does not depend on
// mlrs-algos).
pub mod sgd;
pub(crate) mod sgd_host;
pub mod svd;
// Linear-SVM primal objective evaluator (SVM-FIT-CPU perf lever): the margin
// matvec + per-sample loss + `X̃ᵀg` gradient the `LinearSVC`/`LinearSVR` L-BFGS
// solve runs every iteration. Keeps the two-GEMM shape on the device backends
// and replaces it with ONE fused `-O3` host pass on cpu, where a cubecl launch
// costs three orders of magnitude more than the matvec it performs.
pub mod svm_objective;
// Scalar `lnΓ` / `ψ` / `lnB` (MIX-02). Host-only by construction: the
// variational mixture E-step calls them `O(k·d)` times per iteration against
// its own `O(n·k·d²)`, and cubecl's cuda backend advertises no `f64`
// transcendentals to evaluate them with anyway.
pub mod special;
pub mod topk;
// TSNE-01: the exact-method t-SNE per-iteration gradient prim (Student-t
// affinity + KL-gradient GATHER over the Phase-2 distance prim).
pub mod tsne;
// The parallel HOST t-SNE engine (TSNE-PARAMS): the Barnes-Hut quadtree, both
// gradient objectives, and the two-phase descent that drives them. Owns
// `method='barnes_hut'` on every backend — the tree walk is a per-point,
// per-iteration pointer chase, the shape a SIMT device cannot execute without
// full warp divergence — and serves `method='exact'` wherever the host arm is
// faster than the `tsne` device prim above.
pub mod tsne_host;

// ---------------------------------------------------------------------------
// Shared 1-D launch geometry
// ---------------------------------------------------------------------------

use cubecl::prelude::{CubeCount, CubeDim};
use mlrs_kernels::colmean::MAX_GRID_DIM;

/// Workgroup width for the prims whose launch geometry was tuned by a MEASURED
/// GPU campaign (`kmeans`, `random_forest`), rather than being an arbitrary
/// GPU-idiomatic default.
///
/// Those campaigns picked `256` against real hardware, and this repo's own rule
/// is not to re-gate a perf kernel from a different backend's measurement — so
/// they keep it explicitly instead of inheriting
/// [`crate::capability::gather_launch_width`]. On the cpu backend a `cube_dim`
/// IS a thread count, so `256` there spawns 256 OS threads per launch; moving
/// these two prims onto the gather width is a real cpu win waiting to be
/// measured on their own ladders, NOT a mechanical substitution.
pub(crate) const PERF_TUNED_BLOCK: u32 = 256;

/// Ceiling-division 1-D launch config: `ceil(n / block)` cubes of `block` units
/// along X only.
///
/// For kernels that address their element through `ABSOLUTE_POS_X`, which does
/// NOT linearize across a multi-axis grid — so this must stay single-axis. `n`
/// above `MAX_GRID_DIM * block` needs [`launch_dims_1d_folded`] and a kernel
/// that reads `ABSOLUTE_POS`.
///
/// `block` is passed explicitly at every call site rather than defaulted: it is
/// the cpu thread count (see [`PERF_TUNED_BLOCK`]), so which width a prim uses
/// is a decision worth reading at the launch, not inheriting silently. Prims
/// whose kernels are split-independent (no `SharedMemory`, no `sync_cube`, no
/// `CUBE_DIM`-dependent indexing) pass
/// [`crate::capability::gather_launch_width`].
///
/// This is the ONE definition. Nine byte-identical private copies of it used to
/// live across the prim modules, which is how three of them ended up fixed for
/// the cpu thread-storm and the rest did not.
pub(crate) fn launch_dims_1d(n: usize, block: u32) -> (CubeCount, CubeDim) {
    let cubes = (n as u32).div_ceil(block).max(1);
    (
        CubeCount::Static(cubes, 1, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}

/// [`launch_dims_1d`] with the cube count FOLDED across the X/Y grid axes, so it
/// never exceeds `MAX_GRID_DIM` in a single dimension.
///
/// Only for kernels that address their element through the flattened
/// `ABSOLUTE_POS` (which linearizes contiguously across the grid: cube `(x, y)`
/// covers `[(y·CUBE_COUNT_X + x)·block, +block)`) and bounds-check it, so the
/// second axis is transparent to them. A kernel reading `ABSOLUTE_POS_X` would
/// silently process only the first grid column — use [`launch_dims_1d`] there.
pub(crate) fn launch_dims_1d_folded(n: usize, block: u32) -> (CubeCount, CubeDim) {
    let cubes = (n as u32).div_ceil(block).max(1);
    let y = cubes.div_ceil(MAX_GRID_DIM).max(1);
    let x = cubes.div_ceil(y).max(1);
    (
        CubeCount::Static(x, y, 1),
        CubeDim { x: block, y: 1, z: 1 },
    )
}
