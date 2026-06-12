//! `mlrs-kernels` — generic CubeCL compute kernels, backend-feature-free.
//!
//! Every kernel here is generic over the float type (`<F: Float>`) and is
//! launched generic over the runtime (`::launch::<F, R>`). This crate MUST NOT
//! depend on any CubeCL backend runtime feature (Criterion 1); a concrete
//! runtime is chosen in `mlrs-backend`.

pub mod elementwise;
pub mod jacobi_eig;
pub mod jacobi_svd;
pub mod reduce;
pub mod smoke;

pub use elementwise::{center_columns, clamp_nonneg, dist_combine_clamp, scale, sqrt_elem};
pub use jacobi_eig::{jacobi_eig_sweep, MAX_DIM};
pub use jacobi_svd::{jacobi_svd_sweep, MAX_COLS, MAX_ROWS};
pub use reduce::{
    argmax_shared, argmin_shared, reduce_max_plane, reduce_max_shared, reduce_min_plane,
    reduce_min_shared, reduce_sum_plane, reduce_sum_shared, reduce_sumsq_plane, reduce_sumsq_shared,
};
pub use smoke::saxpy_kernel;
