//! Generic `#[cube]` saxpy smoke kernel.
//!
//! This is the Wave-0 pipeline proof: a single kernel generic over the float
//! type `<F: Float>`, launched generic over the runtime
//! (`saxpy_kernel::launch::<F, R>`). It computes `y[i] = a * x[i] + y[i]` for
//! every in-bounds element.
//!
//! Reference: `cubecl_manual` `Cubecl_axpy.md`, `Cubecl_generics.md`,
//! `Cubecl_multi_threading.md`. Per AGENTS.md, source files carry no in-file
//! test modules — the live launch test is in `mlrs-backend/tests/spike_test.rs`
//! (which owns a concrete runtime feature; this crate is feature-free).

use cubecl::prelude::*;

/// SAXPY smoke kernel: `y = a * x + y`, generic over the float type `F`.
///
/// Each unit handles one element at `ABSOLUTE_POS`, bounds-checked against
/// `x.len()` so launching more threads than elements is safe (the standard
/// ceiling-division launch config over-provisions threads).
// `CubeElement` is required on `F` because the scalar argument `a: F` must
// implement `LaunchArg`/`ScalarArgSettings` for the generated `launch` fn
// (resolved against cubecl 0.10 — matches the `<F: Float + CubeElement>`
// pattern in the half-precision and axpy manuals).
#[cube(launch)]
pub fn saxpy_kernel<F: Float + CubeElement>(a: F, x: &Array<F>, y: &mut Array<F>) {
    let tid = ABSOLUTE_POS;
    if tid < x.len() {
        y[tid] = a * x[tid] + y[tid];
    }
}
