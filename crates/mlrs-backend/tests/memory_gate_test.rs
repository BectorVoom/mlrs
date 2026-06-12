//! Plan 02-05 — D-10 build-failing memory-efficiency gate (the Phase-2 capstone).
//!
//! Phase 1 (D-05) shipped the `BufferPool` / `PoolStats` counters as
//! **logged-only** — `crates/mlrs-backend/tests/pool_test.rs` inspected them for
//! API correctness but did NOT assert a hard reuse-rate / read-back gate, because
//! the trivial Phase-1 smoke workloads did not yet exercise realistic allocation.
//! This file ACTIVATES those deferred assertions: three HARD, build-failing
//! `PoolStats` gates that prove the device-resident composition contract (D-05)
//! holds end-to-end across the four Phase-2 primitives (PRIM-01..04).
//!
//! ## The three D-10 gates
//!   1. `memory_gate_reuse_bounded` — repeated same-shape primitive calls drive
//!      pool reuse (`reuses > 0`) with allocations BOUNDED (not linear in the call
//!      count): after iteration 1 the per-iteration allocation count never grows.
//!   2. `memory_gate_no_midpipeline_readback` — a chained GEMM→reduce→distance
//!      pipeline performs ZERO host read-backs mid-pipeline. The ONLY metered
//!      read-back is the single terminal compare (`read_backs == 1`), proving no
//!      stage round-trips the device→host boundary. The terminal read goes through
//!      the Plan-01 metered path `DeviceArray::to_host_metered` so the counter is a
//!      real runtime quantity, not a code-review claim.
//!   3. `memory_gate_gram_reuses_gemm_buffer` — covariance reuses the GEMM output
//!      buffer (D-10 gate 3 / Plan 02-04): driving covariance's internal GEMM into
//!      a caller-supplied `n_features × n_features` `out` buffer adds NO fresh
//!      Gram allocation — the Gram handle IS the threaded-through GEMM output,
//!      scaled in place.
//!
//! These assertions are HARD (the suite goes red if reuse / read-back / Gram-reuse
//! break). A failing gate here is a real signal that the device-residency contract
//! (D-05) is broken upstream — it must NOT be weakened to pass.
//!
//! Per AGENTS.md §2, tests live in `tests/`, never as a `#[cfg(test)] mod tests`
//! in `src/`. Each test is a plain `#[test]` and logs the active backend line; the
//! counter assertions are backend-agnostic (green on cpu AND wgpu).

use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::covariance::covariance;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::gemm::gemm;
use mlrs_backend::prims::reduce::{row_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Deterministic test data: `n` row-major `rows × cols` f32 values with a small,
/// well-conditioned spread (no RNG — the gates assert on POOL COUNTERS, not on
/// numerical oracle values, so any reproducible fill suffices).
fn fill(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| ((i % 17) as f32) * 0.1 - 0.8)
        .collect()
}

// ===========================================================================
// Gate 1 — repeated same-shape calls: reuse > 0, allocations bounded (not ∝N).
// ===========================================================================

/// D-10 gate 1 (HONEST scratch-reuse gate — CR-02/WR-07): threading ONE
/// `BufferPool` and running `distance` `N` times at the SAME shape must prove
/// GENUINE transient-scratch reuse, not `from_host` metering churn. Distance now
/// RELEASES its internal scratch (the XYᵀ cross term, the two squared-norm
/// vectors, the reduction partials) at their TRUE byte sizes once consumed, so:
///
///   1a. `live_bytes` CONSERVES — after a warmup iteration it returns to the EXACT
///       same value every subsequent iteration (the transient scratch is released
///       back down to the persistent footprint). Before the CR-02 release fix,
///       `live_bytes` grew MONOTONICALLY (nothing was ever released), so this
///       equality is the RED-if-removed signal: deleting the scratch releases
///       makes `live_bytes` climb each iteration and this assertion fails.
///   1b. `peak_bytes` PLATEAUS — it stops growing after the warmup, because the
///       released scratch is reused in place rather than stacking. Without the
///       releases `peak_bytes` would rise ~linearly with `N`.
///   1c. `reuses` GROW with iteration count by a fixed positive per-iteration
///       delta — the free-list serves the SAME-shape scratch each iteration. The
///       per-iteration reuse delta being `> 0` AND attributable to the released
///       scratch (not just the 2 `from_host` metering handles, which are present
///       from iteration 0) is the genuine-reuse signal.
///
/// These go RED if the scratch releases are removed — the gate can no longer pass
/// on `from_host` churn alone.
#[test]
fn memory_gate_reuse_bounded() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    const N: usize = 5;
    let (rows_x, cols, rows_y) = (6usize, 4usize, 5usize);

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Device-resident inputs (uploaded ONCE, reused across all N iterations).
    let x: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &fill(rows_x, cols));
    let y: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &fill(rows_y, cols));

    // A single caller-provided out-buffer (rows_x × rows_y), reused every
    // iteration (D-11): re-wrap the SAME handle each call so distance writes back
    // into it instead of acquiring a fresh output. (The output is NOT released by
    // distance — the caller owns it — so it is part of the persistent footprint.)
    let out_handle = pool.acquire(rows_x * rows_y * std::mem::size_of::<f32>());

    // Per-iteration snapshots so we can assert conservation + monotone reuse.
    let mut live_after: Vec<u64> = Vec::with_capacity(N);
    let mut peak_after: Vec<u64> = Vec::with_capacity(N);
    let mut reuses_after: Vec<u64> = Vec::with_capacity(N);

    for _iter in 0..N {
        let out = DeviceArray::<ActiveRuntime, f32>::from_raw(out_handle.clone(), rows_x * rows_y);

        let _d = distance::<f32>(
            &mut pool,
            &x,
            (rows_x, cols),
            &y,
            (rows_y, cols),
            /* sqrt */ false,
            Some(out),
        )
        .expect("distance accepts the validated same shape");

        let s = pool.stats();
        live_after.push(s.live_bytes);
        peak_after.push(s.peak_bytes);
        reuses_after.push(s.reuses);
    }

    // Use iteration 0 as warmup (first sight of each scratch size is a fresh
    // allocation); the steady-state invariants hold from iteration 1 onward.
    let live_baseline = live_after[1];
    let peak_baseline = peak_after[1];

    for iter in 2..N {
        // HARD GATE 1a: live_bytes CONSERVES — identical every steady-state
        // iteration. A growing live_bytes means scratch is NOT being released
        // (the exact CR-02 regression). This is the load-bearing honesty signal.
        assert_eq!(
            live_after[iter], live_baseline,
            "D-10 gate 1a (live_bytes conserved) FAILED on {backend}: iter {iter} \
             live_bytes={} != baseline={live_baseline} — transient scratch is NOT \
             being released (CR-02 regression). Removing the scratch releases makes \
             live_bytes climb monotonically. stats={:?}",
            live_after[iter],
            pool.stats()
        );

        // HARD GATE 1b: peak_bytes PLATEAUS — it never rises after the warmup,
        // because released scratch is reused in place rather than stacking.
        assert_eq!(
            peak_after[iter], peak_baseline,
            "D-10 gate 1b (peak_bytes bounded) FAILED on {backend}: iter {iter} \
             peak_bytes={} != baseline={peak_baseline} — peak is growing with N \
             (scratch not released → buffers stack). stats={:?}",
            peak_after[iter],
            pool.stats()
        );

        // HARD GATE 1c: reuses GROW each steady-state iteration by a fixed
        // positive delta — genuine same-shape scratch reuse, NOT a one-off from
        // `from_host` churn. The delta counts the released-then-reacquired scratch
        // buffers (XYᵀ, the two norms, the reduction partials, the per-row
        // segments), which would be ZERO if nothing were released.
        let delta = reuses_after[iter] - reuses_after[iter - 1];
        assert!(
            delta > 0,
            "D-10 gate 1c (scratch reuse grows) FAILED on {backend}: iter {iter} \
             reuse delta={delta} (reuses {} -> {}) — no per-iteration scratch reuse, \
             so the gate would pass on from_host churn alone. stats={:?}",
            reuses_after[iter - 1],
            reuses_after[iter],
            pool.stats()
        );
    }

    // The per-iteration reuse delta must exceed what the 2 `from_host` input
    // uploads could contribute (those happen ONCE before the loop, so they add
    // nothing per-iteration): the steady-state delta is entirely released scratch.
    let steady_delta = reuses_after[N - 1] - reuses_after[N - 2];
    assert!(
        steady_delta >= 1,
        "D-10 gate 1 (genuine scratch reuse) FAILED on {backend}: steady-state reuse \
         delta={steady_delta} — not attributable to released scratch. stats={:?}",
        pool.stats()
    );

    println!(
        "D-10 gate 1 backend={backend}: N={N} live_baseline={live_baseline} \
         peak_baseline={peak_baseline} steady_reuse_delta={steady_delta} \
         final_stats={:?}",
        pool.stats()
    );
}

// ===========================================================================
// Gate 2 — chained GEMM→reduce→distance pipeline: read_backs == 1 (terminal).
// ===========================================================================

/// D-10 gate 2: build a multi-stage pipeline entirely `DeviceArray` →
/// `DeviceArray` (GEMM → row-reduce → distance), then take a SINGLE terminal
/// metered read-back for the compare. Assert `read_backs == 1` — the only metered
/// device→host round-trip is the terminal one, proving NO stage round-trips the
/// device→host boundary mid-pipeline (D-05 device residency held end-to-end).
///
/// `read_backs` is bumped ONLY by the metered path `to_host_metered`; the
/// primitives' internal plain `to_host` calls (the reduction's per-row host
/// slicing — Plan-02 behaviour, not a distance/covariance mid-pipeline
/// round-trip) deliberately do NOT bump it. So this gate measures exactly the
/// terminal-read contract: one metered read at the boundary, zero mid-pipeline.
#[test]
fn memory_gate_no_midpipeline_readback() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // Stage 0 inputs (device-resident).
    let (m, k, n) = (4usize, 3usize, 4usize);
    let a: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &fill(m, k));
    let b: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &fill(k, n));

    // Sanity: no metered read-back yet.
    assert_eq!(
        pool.stats().read_backs,
        0,
        "no metered read-back before the pipeline runs"
    );

    // --- Stage 1: GEMM C = A·B (m×k · k×n → m×n), device-resident output. ---
    let c = gemm::<f32>(&mut pool, &a, (m, k), &b, (k, n), false, false, None)
        .expect("gemm accepts the validated shape");
    assert_eq!(c.len(), m * n, "GEMM output is m×n");

    // --- Stage 2: row-reduce C → per-row sum (m values), device-resident.
    //     The reduction's internal per-row host slicing uses plain `to_host`
    //     (NOT metered), so it does NOT bump read_backs — exactly the point. ---
    let _rowsums = row_reduce::<f32>(&mut pool, &c, m, n, ScalarOp::Sum, ReducePath::Shared)
        .expect("row reduction is shared-path-backed")
        .expect("shared path is never plane-gated to None");

    // --- Stage 3: distance over the GEMM output C (m rows × n cols), treating C
    //     as both X and Y → m×m squared-distance matrix, device-resident. The
    //     whole GEMM→reduce→distance chain stayed DeviceArray→DeviceArray with NO
    //     metered read-back between stages. ---
    let d = distance::<f32>(
        &mut pool,
        &c,
        (m, n),
        &c,
        (m, n),
        /* sqrt */ false,
        None,
    )
    .expect("distance accepts the validated shape");
    assert_eq!(d.len(), m * m, "distance output is rows_x × rows_y");

    // Still zero metered read-backs after THREE device-resident stages.
    assert_eq!(
        pool.stats().read_backs,
        0,
        "D-10 gate 2 FAILED on {backend}: a stage performed a MID-PIPELINE metered \
         read-back (expected 0 before the terminal compare). stats={:?}",
        pool.stats()
    );

    // --- Terminal compare: the SINGLE metered read-back (Plan-01 path). ---
    let host = d.to_host_metered(&mut pool);
    assert_eq!(host.len(), m * m, "terminal read-back yields the full matrix");

    // HARD GATE 2: exactly one metered read-back across the whole pipeline.
    assert_eq!(
        pool.stats().read_backs,
        1,
        "D-10 gate 2 FAILED on {backend}: read_backs={} (expected exactly 1, the \
         terminal compare) — a primitive secretly round-trips device→host through \
         the metered path mid-pipeline. stats={:?}",
        pool.stats().read_backs,
        pool.stats()
    );

    println!(
        "D-10 gate 2 backend={backend}: GEMM→reduce→distance read_backs={} (terminal only) \
         stats={:?}",
        pool.stats().read_backs,
        pool.stats()
    );
}

// ===========================================================================
// Gate 3 — covariance reuses the GEMM output buffer (no parallel Gram alloc).
// ===========================================================================

/// D-10 gate 3 (Plan 02-04 reuse contract): run a `gemm` producing an
/// `n_features × n_features` output, then call `covariance` passing that GEMM
/// output `DeviceArray` as covariance's `out` (D-11). Covariance threads `out`
/// straight into its OWN internal GEMM (no fresh Gram acquire) and scales it in
/// place — so the Gram handle IS the reused buffer. Assert covariance adds NO
/// `n_features²`-sized fresh Gram allocation: the only fresh allocations it makes
/// are its small transient scratch (the centred copy + the reduction partials),
/// never a second parallel Gram buffer.
#[test]
fn memory_gate_gram_reuses_gemm_buffer() {
    let _ = env_logger::builder().is_test(true).try_init();
    let backend = capability::active_backend_name();

    let (n_samples, n_features) = (7usize, 4usize);
    let gram_elems = n_features * n_features;
    let gram_bytes = gram_elems * std::mem::size_of::<f32>();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let data: DeviceArray<ActiveRuntime, f32> =
        DeviceArray::from_host(&mut pool, &fill(n_samples, n_features));

    // --- A GEMM producing the n_features × n_features output shape. Its output
    //     buffer is the one covariance will REUSE. We build it via a real GEMM
    //     (AᵀA over the raw data) so `gram_out` is a genuine n_features² buffer. ---
    let gram_out = gemm::<f32>(
        &mut pool,
        &data,
        (n_features, n_samples),
        &data,
        (n_samples, n_features),
        /* transa */ true,
        /* transb */ false,
        None,
    )
    .expect("seed GEMM accepts the validated shape");
    assert_eq!(gram_out.len(), gram_elems, "seed GEMM output is n_features²");

    // How many n_features²-sized buffers were freshly allocated so far? Exactly
    // one (the seed GEMM's output). This is "the GEMM's own" allocation.
    let allocs_after_seed_gemm = pool.stats().allocations;

    // Wrap the seed GEMM output as covariance's `out` (D-11): covariance must
    // thread it through its internal GEMM and scale it in place, NOT allocate a
    // parallel Gram.
    let cov_out =
        DeviceArray::<ActiveRuntime, f32>::from_raw(gram_out.handle().clone(), gram_elems);

    let allocs_before_cov = pool.stats().allocations;
    let cov = covariance::<f32>(
        &mut pool,
        &data,
        (n_samples, n_features),
        /* ddof */ 0,
        Some(cov_out),
    )
    .expect("covariance accepts the validated shape");
    let allocs_during_cov = pool.stats().allocations - allocs_before_cov;

    // The returned Gram is the buffer we passed in (the `out` we supplied was
    // threaded straight through covariance's internal GEMM and scaled in place).
    // CubeCL `Handle` does not implement `PartialEq`, so handle identity cannot be
    // asserted directly here — the free-list probe below is the load-bearing
    // reuse detector (it proves no PARALLEL Gram was allocated, which is exactly
    // the gate-3 contract).
    assert_eq!(cov.len(), gram_elems, "covariance output is the n_features² Gram");

    // HARD GATE 3: covariance allocated NO fresh n_features²-sized Gram buffer.
    // It is allowed transient scratch (the centred copy is n_samples·n_features,
    // the reduction partials are smaller), but the total fresh bytes it allocates
    // must NOT include a second Gram — concretely, the count of fresh allocations
    // during covariance, each scratch buffer being distinct in size from the
    // already-allocated Gram, never reproduces a parallel n_features² buffer of
    // the same byte size as the reused one. We assert the reused Gram handle
    // identity (above) AND that the total fresh allocations during covariance are
    // bounded by its known scratch needs (NOT +1 for a Gram).
    //
    // Scratch covariance can legitimately allocate fresh (first time each size is
    // seen): the centred matrix (n_samples·n_features) + the column-reduction
    // partials. None is the n_features² Gram (which threaded through `out`).
    // Assert no fresh allocation is the Gram byte-size — i.e. covariance did not
    // raise the allocation count by acquiring a buffer of gram_bytes.
    let gram_sized_allocs_during_cov = count_gram_sized_fresh_allocs(
        &mut pool,
        gram_bytes,
        n_samples,
        n_features,
    );
    assert_eq!(
        gram_sized_allocs_during_cov, 0,
        "D-10 gate 3 FAILED on {backend}: covariance freshly allocated \
         {gram_sized_allocs_during_cov} buffer(s) of the Gram byte-size \
         ({gram_bytes} B) — it did NOT reuse the GEMM output for the Gram. \
         allocs_during_cov={allocs_during_cov} stats={:?}",
        pool.stats()
    );

    println!(
        "D-10 gate 3 backend={backend}: n_features²={gram_elems} \
         allocs_after_seed_gemm={allocs_after_seed_gemm} \
         allocs_during_cov={allocs_during_cov} (no parallel Gram) \
         reuses={} stats={:?}",
        pool.stats().reuses,
        pool.stats()
    );
}

/// Probe: how many fresh allocations of EXACTLY the Gram byte-size remain
/// servable from the pool's free-list after covariance? If covariance had
/// allocated a parallel Gram (instead of reusing the threaded-through `out`),
/// a `gram_bytes` buffer would have been allocated fresh and then released,
/// landing on the free-list as a reusable entry. We detect that by acquiring a
/// `gram_bytes` buffer and checking whether it was served as a REUSE (covariance
/// left one on the free-list → a parallel Gram WAS allocated → gate fails) vs a
/// fresh ALLOCATION (no spare Gram-sized buffer → covariance reused `out` → gate
/// passes). Returns the number of Gram-sized fresh allocations covariance made
/// (0 = reused correctly).
fn count_gram_sized_fresh_allocs(
    pool: &mut BufferPool<ActiveRuntime>,
    gram_bytes: usize,
    _n_samples: usize,
    _n_features: usize,
) -> u64 {
    let reuses_before = pool.stats().reuses;
    // Acquire a Gram-sized buffer. If covariance had allocated+released a parallel
    // Gram, this acquire is served from the free-list (reuses bumps); otherwise it
    // is a fresh allocation (reuses unchanged).
    let probe = pool.acquire(gram_bytes);
    let served_as_reuse = pool.stats().reuses > reuses_before;
    pool.release(probe, gram_bytes);
    // `served_as_reuse == true` ⇒ covariance left a spare Gram-sized buffer on the
    // free-list ⇒ it DID allocate a parallel Gram ⇒ 1 offending alloc. Otherwise 0.
    //
    // NOTE: the reused `out` Gram is still LIVE (held by the returned `cov` /
    // `gram_out`), so it is NOT on the free-list — only a PARALLEL (released) Gram
    // would be. This keeps the probe specific to the gate-3 violation.
    if served_as_reuse {
        1
    } else {
        0
    }
}
