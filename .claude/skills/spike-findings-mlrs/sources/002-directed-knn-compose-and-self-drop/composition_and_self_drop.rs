//! SPIKE 002 (Phase 13 keystone) — TEMPORARY run vehicle, not production code.
//!
//! Proves the directed KNN-graph composition + the open self-drop mechanism (D-02):
//!   distance(euclidean) → top_k(k+1) → `self_drop_gather` → directed (indices,distances) (n×k)
//! with `include_self=false` self-drop by INDEX IDENTITY (R-3) — robust against a
//! duplicate point sitting at distance 0, which a "drop first zero-distance" rule
//! would get WRONG. Validated end-to-end against a brute-force host KNN. Also shows
//! the `include_self=true` path is just a plain `top_k` of k (already proven).
//!
//! The self-drop is a NEW `#[cube(launch)]` kernel using only `u32`/`F` accumulators
//! and statement-form `if` (no mutable bool, no SharedMemory) — the cpu-MLIR-safe
//! GATHER mechanism the discussion left for the spike to confirm.
//!
//! Durable artifact: copied to `.planning/spikes/002-directed-knn-compose-and-self-drop/`.
//! Run: `cargo test -p mlrs-backend --features cpu --test knn_spike_002_test -- --nocapture`

use cubecl::prelude::*;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::topk::top_k;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Self-drop-by-index-identity GATHER kernel (one unit per query row).
/// Input is the `top_k(k+1)` result (ascending `(val, idx)` per row); output is the
/// `k` true neighbours with the self column removed. cpu-MLIR-safe: a `u32`
/// position accumulator `c_self` (NOT a mutable bool scan), statement-form `if`.
#[cube(launch)]
pub fn self_drop_gather<F: Float + CubeElement>(
    in_val: &Array<F>,
    in_idx: &Array<u32>,
    out_val: &mut Array<F>,
    out_idx: &mut Array<u32>,
    rows: u32,
    k: u32,
    k1: u32, // k + 1
) {
    // Mirror the lowering-proven top_k structure EXACTLY: row = CUBE_POS_X (native
    // u32, NOT `ABSOLUTE_POS` which is usize), select on unit 0 of the cube.
    let row = CUBE_POS_X;
    if row < rows {
        if UNIT_POS_X == 0u32 {
            let ibase = row * k1;
            let obase = row * k;

            // Self-drop by INDEX IDENTITY, with NO cross-sibling-loop carry
            // (FINDING 002-B): a mutable `c_self` written in one `while` and read in
            // a SEPARATE sibling `while` silently miscompiles under the cube macro —
            // the same "cross-loop flag" limitation top_k documents. Instead, for
            // each output slot s, recompute the shift LOCALLY as the count of
            // self-columns at input positions 0..=s (exactly 0 or 1), via a nested
            // accumulate read in the SAME outer iteration (the top_k-proven shape):
            //   src = s + (#self-cols at cols ≤ s)
            //   - s before self → 0 self-cols ≤ s → src = s
            //   - s at/after self → 1 self-col ≤ s → src = s+1 (skip self)
            //   - self absent (fallback) → 0 always → src = s (drops the last col k)
            // Col src is in-bounds: max is (k-1)+1 = col k. No conditional index, no
            // F::INFINITY, no SharedMemory, no cross-loop flag.
            let mut s = 0u32;
            while s < k {
                let mut bump = 0u32;
                let mut c = 0u32;
                while c < s + 1u32 {
                    if in_idx[(ibase + c) as usize] == row {
                        bump += 1u32;
                    }
                    c += 1u32;
                }
                let src = s + bump;
                out_val[(obase + s) as usize] = in_val[(ibase + src) as usize];
                out_idx[(obase + s) as usize] = in_idx[(ibase + src) as usize];
                s += 1u32;
            }
        }
    }
}

/// One cube per query row (row = CUBE_POS_X), one selecting unit per cube — the
/// top_k launch shape.
fn launch_1d(n: usize) -> (CubeCount, CubeDim) {
    (
        CubeCount::Static(n.max(1) as u32, 1, 1),
        CubeDim { x: 1, y: 1, z: 1 },
    )
}

/// Brute-force host KNN: for each query row i, order all rows by (distance, index)
/// (lowest-index tie-break). `exclude_self` drops the j==i entry. Returns the first
/// `k` (index, distance) pairs per row, flattened row-major.
fn host_knn(x: &[f64], n: usize, d: usize, k: usize, exclude_self: bool) -> (Vec<u32>, Vec<f64>) {
    let dist = |i: usize, j: usize| -> f64 {
        (0..d)
            .map(|c| {
                let diff = x[i * d + c] - x[j * d + c];
                diff * diff
            })
            .sum::<f64>()
            .sqrt()
    };
    let mut idx = vec![0u32; n * k];
    let mut val = vec![0.0f64; n * k];
    for i in 0..n {
        let mut pairs: Vec<(usize, f64)> = (0..n)
            .filter(|&j| !(exclude_self && j == i))
            .map(|j| (j, dist(i, j)))
            .collect();
        // sort by (distance, index) — the prim's documented lowest-index tie-break.
        pairs.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap()
                .then(a.0.cmp(&b.0))
        });
        for s in 0..k {
            idx[i * k + s] = pairs[s].0 as u32;
            val[i * k + s] = pairs[s].1;
        }
    }
    (idx, val)
}

#[test]
fn spike002_directed_knn_self_drop_by_index_identity() {
    let _ = env_logger::builder().is_test(true).try_init();
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client.clone());

    // n=6, d=2. ROWS 0 AND 1 ARE IDENTICAL (the adversarial duplicate): both sit
    // at distance 0 from each other, so for query row 1 the two nearest are indices
    // 0 (genuine neighbour) and 1 (self). "Drop first zero-distance" would drop
    // index 0 — the WRONG one. Index-identity must drop index 1 (self), keep 0.
    let n = 6usize;
    let d = 2usize;
    let k = 3usize;
    let k1 = k + 1;
    let x: Vec<f64> = vec![
        0.0, 0.0, // 0
        0.0, 0.0, // 1  (duplicate of 0)
        1.0, 0.0, // 2
        0.0, 1.0, // 3
        3.0, 3.0, // 4
        -1.0, -1.0, // 5
    ];

    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x);

    // distance (true euclidean) → top_k(k+1)
    let d_dev = distance::<f64>(&mut pool, &x_dev, (n, d), &x_dev, (n, d), false, None)
        .expect("pairwise distance");
    let (val_kp1, idx_kp1) =
        top_k::<f64>(&mut pool, &d_dev, n, n, k1, true, None, None).expect("top_k(k+1)");

    // self_drop_gather → directed (n×k)
    let out_val_h = client.empty(n * k * std::mem::size_of::<f64>());
    let out_idx_h = client.empty(n * k * std::mem::size_of::<u32>());
    let ov_read = out_val_h.clone();
    let oi_read = out_idx_h.clone();
    let (count, dim) = launch_1d(n);
    let iv = unsafe { ArrayArg::from_raw_parts(val_kp1.handle().clone(), n * k1) };
    let ii = unsafe { ArrayArg::from_raw_parts(idx_kp1.handle().clone(), n * k1) };
    let ov = unsafe { ArrayArg::from_raw_parts(out_val_h, n * k) };
    let oi = unsafe { ArrayArg::from_raw_parts(out_idx_h, n * k) };
    self_drop_gather::launch::<f64, ActiveRuntime>(
        &client, count, dim, iv, ii, ov, oi, n as u32, k as u32, k1 as u32,
    );

    let got_val: Vec<f64> =
        bytemuck::cast_slice::<u8, f64>(&client.read_one(ov_read).expect("read val")).to_vec();
    let got_idx: Vec<u32> =
        bytemuck::cast_slice::<u8, u32>(&client.read_one(oi_read).expect("read idx")).to_vec();

    // ───────────── include_self=false: match brute-force host KNN (excl. self) ─────
    let (ref_idx, ref_val) = host_knn(&x, n, d, k, true);
    for i in 0..n {
        for s in 0..k {
            let slot = i * k + s;
            assert_eq!(
                got_idx[slot], ref_idx[slot],
                "exclude-self index[{i}][{s}]: device={} host={} (index-identity self-drop)",
                got_idx[slot], ref_idx[slot]
            );
            assert!(
                (got_val[slot] - ref_val[slot]).abs() <= 1e-6,
                "exclude-self dist[{i}][{s}]: device={} host={}",
                got_val[slot], ref_val[slot]
            );
            // No self in any output row.
            assert_ne!(got_idx[slot], i as u32, "self leaked into output row {i} slot {s}");
        }
    }

    // ───────────── THE ADVERSARIAL ASSERTION ───────────────────────────────────
    // Query row 1 (self idx 1) has a distance-0 duplicate at idx 0. Neighbour 0
    // MUST be index 0 (the genuine neighbour), proving we dropped self by INDEX
    // IDENTITY, not by "first zero-distance" (which would have dropped index 0).
    assert_eq!(
        got_idx[1 * k + 0], 0u32,
        "DUP-POINT: query row 1 neighbour 0 must be the genuine duplicate (idx 0), \
         got idx {} — a first-zero-distance drop would fail here",
        got_idx[1 * k + 0]
    );
    assert_eq!(got_val[1 * k + 0] as f64, 0.0, "dup neighbour distance must be 0");

    // ───────────── include_self=true: plain top_k(k) (already-proven path) ─────────
    let (_val_self, idx_self) =
        top_k::<f64>(&mut pool, &d_dev, n, n, k, true, None, None).expect("top_k(k) self-incl");
    let inc_idx: Vec<u32> = idx_self.to_host(&pool);
    let (ref_idx_inc, _) = host_knn(&x, n, d, k, false);
    for slot in 0..(n * k) {
        assert_eq!(
            inc_idx[slot], ref_idx_inc[slot],
            "include-self index[{slot}] device={} host={}",
            inc_idx[slot], ref_idx_inc[slot]
        );
    }
    // Self must be PRESENT in each include_self row (HDBSCAN core-distance needs it).
    for i in 0..n {
        let present = (0..k).any(|s| inc_idx[i * k + s] == i as u32);
        assert!(present, "include_self row {i} is missing self");
    }

    println!(
        "SPIKE 002: directed (n×k) KNN composed distance→top_k(k+1)→self_drop_gather ✓\n  \
         include_self=false matches brute-force host KNN (excl self), no self leak ✓\n  \
         ADVERSARIAL dup-point: query-1 neighbour-0 == genuine idx 0 (index-identity, \
         NOT first-zero-distance) ✓\n  \
         include_self=true == plain top_k(k), self present in every row ✓\n  \
         self_drop_gather (CUBE_POS_X shape, per-slot nested-count shift — NO \
         cross-sibling-loop carry) launches AND computes correctly under cpu-MLIR ✓"
    );
}
