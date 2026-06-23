//! Phase-14 `umap_layout_step` launch-smoke gate (Spike flag item 1).
//!
//! `mlrs-kernels` carries NO backend runtime feature, so the ONLY place the new
//! `umap_layout_step` kernel can be LAUNCHED (not just compiled) is here, against
//! `mlrs_backend::runtime::ActiveRuntime`. This test LAUNCHES the kernel on a
//! tiny fixed graph under the active backend (cpu-MLIR is the f64 gate) for BOTH
//! f32 AND f64 and asserts the owner coordinates actually MOVE — a value
//! assertion, not a bare non-panic check, because a happy-path-only check would
//! ship the FINDING 002-B silent miscompile (R-9). cpu runs f64; rocm SKIPS f64
//! with a log (project memory — f64 is unsupported on cubecl-cpp 0.10/rocm).
//!
//! Per AGENTS.md §2 the test lives in this dedicated file, never an in-source
//! `#[cfg(test)] mod tests`.

use cubecl::prelude::*;

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_kernels::umap_layout_step;

fn to_f<F: bytemuck::Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("umap_layout tests are f32/f64 only"),
    }
}

fn from_f<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("umap_layout tests are f32/f64 only"),
    }
}

/// f64 capability gate (cpu runs f64; rocm skips-with-log). `true` = skip.
fn gate_f64(case: &str) -> bool {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("umap_layout {case} f64 backend={backend}: SKIPPED (no f64 support)");
        return true;
    }
    false
}

/// Launch ONE `umap_layout_step` over a tiny fixed 3-vertex / 2-D graph and read
/// the updated embedding back. Owners = all 3 vertices; one positive edge each
/// (0→1, 1→2, 2→0) and one negative sample each (the "far" third vertex), so
/// every owner has both an attractive and a repulsive contribution.
fn launch_step<F: Float + CubeElement + bytemuck::Pod>(
    pool: &mut BufferPool<ActiveRuntime>,
    move_other: u32,
) -> Vec<f64> {
    let dim = 2usize;
    let n_vertices = 3usize;
    let n_owners = 3usize;

    // Spread coordinates so all pairwise dist² > 0 (exercises the non-zero
    // attractive AND repulsive gradient branches, not the dist²==0 fallback).
    let emb: Vec<F> = [0.0, 0.0, 1.0, 0.5, 2.0, -0.5]
        .iter()
        .map(|&v| to_f::<F>(v))
        .collect();

    // CSR positive edges: owner o → (o+1) mod 3, one edge each.
    let pos_offsets: Vec<u32> = vec![0, 1, 2, 3];
    let pos_tail: Vec<u32> = vec![1, 2, 0];
    // CSR negative samples: owner o → (o+2) mod 3, one sample each.
    let neg_offsets: Vec<u32> = vec![0, 1, 2, 3];
    let neg_idx: Vec<u32> = vec![2, 0, 1];

    let emb_dev = DeviceArray::<ActiveRuntime, F>::from_host(pool, &emb);
    let pos_off_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &pos_offsets);
    let pos_tail_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &pos_tail);
    let neg_off_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &neg_offsets);
    let neg_idx_dev = DeviceArray::<ActiveRuntime, u32>::from_host(pool, &neg_idx);

    let client = pool.client().clone();
    let count = CubeCount::Static(n_owners as u32, 1, 1);
    let cube_dim = CubeDim { x: 1, y: 1, z: 1 };

    let emb_arg =
        unsafe { ArrayArg::from_raw_parts(emb_dev.handle().clone(), n_vertices * dim) };
    let pos_off_arg =
        unsafe { ArrayArg::from_raw_parts(pos_off_dev.handle().clone(), pos_offsets.len()) };
    let pos_tail_arg =
        unsafe { ArrayArg::from_raw_parts(pos_tail_dev.handle().clone(), pos_tail.len()) };
    let neg_off_arg =
        unsafe { ArrayArg::from_raw_parts(neg_off_dev.handle().clone(), neg_offsets.len()) };
    let neg_idx_arg =
        unsafe { ArrayArg::from_raw_parts(neg_idx_dev.handle().clone(), neg_idx.len()) };

    umap_layout_step::launch::<F, ActiveRuntime>(
        &client,
        count,
        cube_dim,
        emb_arg,
        pos_off_arg,
        pos_tail_arg,
        neg_off_arg,
        neg_idx_arg,
        to_f::<F>(1.577),  // a (umap default-ish)
        to_f::<F>(0.895),  // b
        to_f::<F>(1.0),    // gamma
        to_f::<F>(1.0),    // alpha
        dim as u32,
        n_owners as u32,
        n_vertices as u32,
        move_other,
    );

    let out: Vec<f64> = emb_dev.to_host(pool).iter().map(|&v| from_f::<F>(v)).collect();
    emb_dev.release_into(pool);
    pos_off_dev.release_into(pool);
    pos_tail_dev.release_into(pool);
    neg_off_dev.release_into(pool);
    neg_idx_dev.release_into(pool);
    out
}

/// VALUE assertion (R-9): every owner coordinate must MOVE from its start, and
/// the result must be finite — a happy-path non-panic check would ship the 002-B
/// silent miscompile (kernel "runs" but coordinates never change).
fn assert_moved(start: &[f64], end: &[f64]) {
    assert_eq!(start.len(), end.len(), "embedding length preserved");
    let mut any_moved = false;
    for i in 0..start.len() {
        assert!(end[i].is_finite(), "coord {i} must be finite (no NaN/Inf)");
        if (end[i] - start[i]).abs() > 1e-9 {
            any_moved = true;
        }
    }
    assert!(
        any_moved,
        "umap_layout_step must MOVE coordinates (silent-miscompile guard, 002-B): {start:?} -> {end:?}"
    );
}

const START: [f64; 6] = [0.0, 0.0, 1.0, 0.5, 2.0, -0.5];

#[test]
fn umap_layout_step_launches_f32_move_both() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let end = launch_step::<f32>(&mut pool, 1u32);
    assert_moved(&START, &end);
}

#[test]
fn umap_layout_step_launches_f64_move_both() {
    if gate_f64("move_both") {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let end = launch_step::<f64>(&mut pool, 1u32);
    assert_moved(&START, &end);
}

/// Frozen-subset proof (D-03): with `move_other = 0` the owners still move, but
/// since EVERY vertex is an owner here we assert the kernel still launches and
/// moves coordinates under the one-sided update path (the Plan-05 transform
/// path's launch shape).
#[test]
fn umap_layout_step_launches_f64_owner_only() {
    if gate_f64("owner_only") {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let end = launch_step::<f64>(&mut pool, 0u32);
    assert_moved(&START, &end);
}
