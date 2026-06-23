//! Launch smoke test for `mlrs_kernels::self_drop_gather` under cpu-MLIR.
//!
//! The `mlrs-kernels` crate is backend-feature-free (it cannot select a concrete
//! runtime), so — exactly like `spike_test.rs` — the live launch proof must live
//! here in `mlrs-backend`, where `ActiveRuntime` and `--features cpu` exist.
//!
//! This test transcribes the VALIDATED spike-002 self-drop scenario into a thin,
//! self-contained launch smoke proof: a hand-built `top_k(k+1)` result with a
//! known self-index column per row. It asserts the kernel
//!   1. actually LAUNCHED (non-zero correct read-back — the loud 002-A failure
//!      reads back all zeros because the kernel never runs), and
//!   2. dropped exactly the self column by INDEX IDENTITY (D-02 / R-3), keeping a
//!      genuine duplicate neighbour sitting at distance 0.
//!
//! Per AGENTS.md, tests live in `tests/`, never as `#[cfg(test)] mod tests`.

use cubecl::prelude::*;
use mlrs_backend::capability;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_kernels::self_drop_gather;

/// Byte-cast an `F` (f32/f64) value to host `f64` without calling any cube
/// function (`F::abs` etc. are `#[cube]` and panic on the host). Mirrors the
/// `topk_test.rs` convention.
fn host_to_f64<F: bytemuck::Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("self_drop tests are f32/f64 only"),
    }
}

/// Build an `F` (f32/f64) from a host `f64` literal without a cube function.
fn from_f64<F: bytemuck::Pod>(x: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(x as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&x)),
        _ => unreachable!("self_drop tests are f32/f64 only"),
    }
}

/// One cube per query row (`row = CUBE_POS_X`), one selecting unit per cube — the
/// VALIDATED spike-002 / `topk::launch_dims_rows` launch shape. NOT a bare 1D
/// `ABSOLUTE_POS` launch (that is the loud 002-A MLIR pass failure).
fn launch_dims_rows(n: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static(n.max(1) as u32, 1, 1),
        CubeDim { x: 1, y: 1, z: 1 },
    )
}

/// Generic launch + read-back of `self_drop_gather` on a hand-built
/// `(rows, k+1)` `(val, idx)` input. Returns the `(out_val, out_idx)` `(rows, k)`.
fn run_self_drop<F>(
    in_val: &[F],
    in_idx: &[u32],
    rows: usize,
    k: usize,
) -> (Vec<F>, Vec<u32>)
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let k1 = k + 1;
    let client = runtime::active_client();

    let iv_h = client.create(cubecl::bytes::Bytes::from_elems(in_val.to_vec()));
    let ii_h = client.create(cubecl::bytes::Bytes::from_elems(in_idx.to_vec()));
    let ov_h = client.empty(rows * k * std::mem::size_of::<F>());
    let oi_h = client.empty(rows * k * std::mem::size_of::<u32>());
    let ov_read = ov_h.clone();
    let oi_read = oi_h.clone();

    let (count, dim) = launch_dims_rows(rows);
    // cubecl 0.10: scalar args by value, `from_raw_parts(handle, len)` consumes the handle.
    let iv = unsafe { ArrayArg::from_raw_parts(iv_h, rows * k1) };
    let ii = unsafe { ArrayArg::from_raw_parts(ii_h, rows * k1) };
    let ov = unsafe { ArrayArg::from_raw_parts(ov_h, rows * k) };
    let oi = unsafe { ArrayArg::from_raw_parts(oi_h, rows * k) };
    self_drop_gather::launch::<F, ActiveRuntime>(
        &client,
        count,
        dim,
        iv,
        ii,
        ov,
        oi,
        rows as u32,
        k as u32,
        k1 as u32,
    );

    let got_val =
        bytemuck::cast_slice::<u8, F>(&client.read_one(ov_read).expect("read out_val")).to_vec();
    let got_idx =
        bytemuck::cast_slice::<u8, u32>(&client.read_one(oi_read).expect("read out_idx")).to_vec();
    (got_val, got_idx)
}

/// Build a hand-crafted `top_k(k+1)` result and assert the self column is dropped
/// by INDEX IDENTITY with correct non-zero read-back. Generic over `F`.
///
/// Scenario (rows=3, k=2, k1=3), ascending `(val, idx)` per row:
///   row 0: (0.0, 1) (0.0, 0) (1.0, 2)  — self idx 0 at col 1; idx 1 is a genuine
///          duplicate at distance 0. Index-identity must drop col 1 (self), keep
///          idx 1 (distance 0) then idx 2.  => out = [(0.0,1),(1.0,2)]
///   row 1: (0.0, 1) (0.5, 4) (0.9, 7)  — self idx 1 at col 0 (typical X-vs-X).
///          Drop col 0.                  => out = [(0.5,4),(0.9,7)]
///   row 2: (0.0, 2) (0.3, 5) (0.4, 8)  — self idx 2 at col 0.
///          Drop col 0.                  => out = [(0.3,5),(0.4,8)]
fn check_self_drop<F>()
where
    F: Float + CubeElement + bytemuck::Pod,
{
    let rows = 3usize;
    let k = 2usize;

    // (rows, k+1) ascending distances + their indices.
    let in_val: Vec<F> = [0.0f64, 0.0, 1.0, 0.0, 0.5, 0.9, 0.0, 0.3, 0.4]
        .iter()
        .map(|&v| from_f64::<F>(v))
        .collect();
    let in_idx: Vec<u32> = vec![1, 0, 2, 1, 4, 7, 2, 5, 8];

    let (got_val, got_idx) = run_self_drop::<F>(&in_val, &in_idx, rows, k);

    // Expected (rows, k) output after index-identity self-drop.
    let want_idx: Vec<u32> = vec![1, 2, 4, 7, 5, 8];
    let want_val: Vec<f64> = vec![0.0, 1.0, 0.5, 0.9, 0.3, 0.4];

    // 002-A loud-failure guard: a kernel that never launched reads back all zeros.
    // Several surviving slots expect non-zero values (1.0, 0.5, 0.9, 0.3, 0.4); a
    // non-trivial read-back proves the kernel actually ran.
    let any_nonzero = got_val.iter().any(|&v| host_to_f64(v) != 0.0);
    assert!(
        any_nonzero,
        "self_drop_gather read back all zeros — kernel did not launch (002-A loud failure)"
    );

    for slot in 0..(rows * k) {
        assert_eq!(
            got_idx[slot], want_idx[slot],
            "index-identity self-drop wrong at slot {slot}: got {} want {}",
            got_idx[slot], want_idx[slot]
        );
        // self index must never appear in its own output row.
        let row = slot / k;
        assert_ne!(
            got_idx[slot] as usize, row,
            "self idx leaked into output row {row} slot {slot}"
        );
        // value matches the surviving neighbour (proves the GATHER copied the right col).
        let diff = (host_to_f64(got_val[slot]) - want_val[slot]).abs();
        assert!(
            diff <= 1e-6,
            "value mismatch at slot {slot}: got {} want {}",
            host_to_f64(got_val[slot]),
            want_val[slot]
        );
    }

    // ── ADVERSARIAL dup-point (R-9): row 0's self (idx 0) sits at distance 0
    // alongside a GENUINE duplicate (idx 1) also at distance 0. Index-identity must
    // keep idx 1 as neighbour 0 (NOT drop it as "first zero-distance").
    assert_eq!(
        got_idx[0], 1u32,
        "DUP-POINT: row 0 neighbour 0 must be the genuine duplicate idx 1 \
         (index-identity), got idx {} — a first-zero-distance drop would fail here",
        got_idx[0]
    );
    assert_eq!(
        host_to_f64(got_val[0]),
        0.0,
        "dup neighbour distance must be 0"
    );

    println!(
        "self_drop_gather [{}]: launched (non-zero read-back), dropped self by index \
         identity, dup-point neighbour preserved ✓",
        std::any::type_name::<F>()
    );
}

#[test]
fn self_drop_gather_f64_launches_and_drops_self_by_index() {
    let _ = env_logger::builder().is_test(true).try_init();
    if capability::skip_f64_with_log() {
        println!(
            "self_drop_gather f64 backend={}: SKIPPED (no f64 support on this adapter)",
            capability::active_backend_name()
        );
        return;
    }
    check_self_drop::<f64>();
}

#[test]
fn self_drop_gather_f32_launches_and_drops_self_by_index() {
    let _ = env_logger::builder().is_test(true).try_init();
    check_self_drop::<f32>();
}
