//! `mlrs-kernels` — generic CubeCL compute kernels, backend-feature-free.
//!
//! Every kernel here is generic over the float type (`<F: Float>`) and is
//! launched generic over the runtime (`::launch::<F, R>`). This crate MUST NOT
//! depend on any CubeCL backend runtime feature (Criterion 1); a concrete
//! runtime is chosen in `mlrs-backend`.

pub mod cholesky;
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
pub mod smoke;
pub mod topk;

pub use cholesky::cholesky_solve;
pub use elementwise::{
    center_columns, clamp_nonneg, dist_combine_clamp, div_by_row, kde_cosine_map,
    kde_epanechnikov_map, kde_exponential_map, kde_gaussian_map, kde_linear_map, kde_tophat_map,
    laplacian_map, poly_map, rbf_map, scale, sigmoid_map, sqrt_elem,
};
pub use jacobi_eig::{jacobi_eig_sweep, MAX_DIM};
pub use jacobi_svd::{jacobi_svd_sweep, MAX_COLS, MAX_ROWS};
pub use reduce::{
    argmax_shared, argmin_shared, reduce_max_plane, reduce_max_shared, reduce_min_plane,
    reduce_min_shared, reduce_sum_plane, reduce_sum_shared, reduce_sumsq_plane, reduce_sumsq_shared,
};
pub use smoke::saxpy_kernel;
