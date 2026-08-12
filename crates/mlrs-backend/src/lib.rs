//! `mlrs-backend` — backend selection and the device-side data plane.
//!
//! This crate owns the `cpu` / `wgpu` / `cuda` / `rocm` Cargo features
//! (FOUND-03). It resolves the active CubeCL runtime (`runtime`), gates f64 and
//! other capabilities (`capability`), ingests Arrow buffers zero-copy
//! (`bridge`), wraps device buffers (`device_array`), and reuses them
//! (`pool`).
//!
//! Wave 0 (Plan 01) fills `runtime` and `capability` with spike-resolved
//! content; `bridge`, `device_array`, and `pool` are compiling stubs whose
//! real bodies land in Wave 1 (Plans 03/04).

pub mod runtime;
// A/B kernel-variant knobs (`MLRS_TOPK_SERIAL`, `MLRS_DIST_RB2`, ...) with a
// thread-local override, so the equivalence tests can drive each variant
// without `set_var`ing process-global state out from under the sibling tests
// libtest runs concurrently. Prim dispatchers read knobs through `abflag::var`.
pub mod abflag;
pub mod capability;
pub mod device;
pub mod bridge;
pub mod device_array;
pub mod pool;
pub mod prims;
