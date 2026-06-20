//! Plan 07-02 — RNG projection-matrix primitive (PRIM-06) oracle/property tests.
//!
//! The live property + distribution gate for `mlrs_backend::prims::rng` (the
//! `rng.rs` body landed in plan 07-02): the Gaussian matrix's per-element stats
//! (mean ≈ 0, var ≈ 1/n_components), seed-reproducibility (same `u64` seed →
//! BYTE-IDENTICAL matrix across runs/backends — the SplitMix64 host PRNG is
//! deterministic, NEVER OsRng, ASVS V6 / T-07-02), the Achlioptas sparse
//! density/value stats, the Fisher–Yates permutation bijection, and a PoolStats
//! memory gate (host-generate + single upload, bounded allocations — the D-10
//! precedent).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim from
//! `gemm_test.rs` (cpu runs f64; rocm skips-with-log per the CubeCL-HIP F64 gap,
//! D-07). Per AGENTS.md §2 tests live in `crates/mlrs-backend/tests/`, never an
//! in-source `#[cfg(test)] mod tests`.

use mlrs_backend::capability;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::rng::{gaussian_matrix, permutation, sparse_achlioptas_matrix};
use mlrs_backend::runtime::{self, ActiveRuntime};

/// Pinned number of averaging trials for the RNG distribution/property checks
/// (averaging over many seeds so a single unlucky draw never flips a strict band
/// — D-11 averaging precedent).
const RNG_TRIALS: usize = 50;

/// Build a fresh pool over the active-runtime client.
fn fresh_pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

/// Gaussian projection-matrix per-element distribution: mean ≈ 0,
/// var ≈ 1/n_components (PRIM-06). f32 path (cpu + rocm).
///
/// Generates a large `n_components × n_features` Gaussian matrix and asserts the
/// empirical per-element mean concentrates at 0 and the variance at
/// `1/n_components`, within a generous statistical band (this is a distribution
/// test, not a 1e-5 oracle).
#[test]
fn rng_gaussian_distribution() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let mut pool = fresh_pool();
    let n_components = 64usize;
    let n_features = 512usize;
    let count = n_components * n_features; // 32768 samples → tight concentration

    let mat = gaussian_matrix::<f32>(&mut pool, 0x5EED_C0FFEE, n_components, n_features)
        .expect("valid gaussian shape");
    let host = mat.to_host(&pool);
    mat.release_into(&mut pool);

    assert_eq!(host.len(), count, "gaussian matrix is n_components × n_features");

    let mean: f64 = host.iter().map(|&v| v as f64).sum::<f64>() / count as f64;
    let var: f64 =
        host.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / count as f64;

    let target_var = 1.0_f64 / n_components as f64;
    // Generous bands: with 32768 samples the sample mean SE ≈ sqrt(var/N) ≈ 7e-4
    // and the variance estimate is within a few percent.
    assert!(
        mean.abs() < 5.0e-3,
        "gaussian mean {mean} not ≈ 0 (backend={backend})"
    );
    assert!(
        (var - target_var).abs() < 0.20 * target_var,
        "gaussian var {var} not ≈ 1/n_components={target_var} (backend={backend})"
    );
}

/// Seed-reproducibility: the SAME `u64` seed yields a BYTE-IDENTICAL matrix across
/// independent generations (PRIM-06 / T-07-02 — host SplitMix64 is deterministic,
/// never OsRng — this is what guarantees cpu == rocm). A DIFFERENT seed differs.
/// f64 path gated by `skip_f64_with_log` (cpu runs; rocm skips).
#[test]
fn rng_seed_reproducible() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("rng f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }

    let mut pool = fresh_pool();
    let (nc, nf) = (32usize, 48usize);

    let a = gaussian_matrix::<f64>(&mut pool, 12345, nc, nf).expect("valid shape");
    let b = gaussian_matrix::<f64>(&mut pool, 12345, nc, nf).expect("valid shape");
    let a_host = a.to_host(&pool);
    let b_host = b.to_host(&pool);
    a.release_into(&mut pool);
    b.release_into(&mut pool);

    // BYTE-IDENTICAL: the PRIM-06 hard gate (same seed → same matrix, bit for bit).
    assert_eq!(
        a_host.to_vec(),
        b_host.to_vec(),
        "same-seed gaussian matrices must be byte-identical (T-07-02, backend={backend})"
    );

    // A different seed must produce a different matrix (the PRNG actually varies).
    let c = gaussian_matrix::<f64>(&mut pool, 67890, nc, nf).expect("valid shape");
    let c_host = c.to_host(&pool);
    c.release_into(&mut pool);
    assert_ne!(
        a_host.to_vec(),
        c_host.to_vec(),
        "different-seed gaussian matrices must differ (backend={backend})"
    );

    // Achlioptas + permutation are byte-reproducible too (same PRNG).
    let s1 = sparse_achlioptas_matrix::<f64>(&mut pool, 999, nc, nf, 0.3).expect("valid");
    let s2 = sparse_achlioptas_matrix::<f64>(&mut pool, 999, nc, nf, 0.3).expect("valid");
    let s1h = s1.to_host(&pool);
    let s2h = s2.to_host(&pool);
    s1.release_into(&mut pool);
    s2.release_into(&mut pool);
    assert_eq!(
        s1h.to_vec(),
        s2h.to_vec(),
        "same-seed Achlioptas matrices must be byte-identical (backend={backend})"
    );
    assert_eq!(
        permutation(7, 1000),
        permutation(7, 1000),
        "same-seed permutation must be identical (backend={backend})"
    );
}

/// Achlioptas sparse matrix density + value stats: the observed nonzero fraction
/// ≈ `density` and the nonzero entries are exactly ±v with
/// `v = sqrt((1/density)/n_components)` (PRIM-06). f32 path.
#[test]
fn rng_achlioptas_density() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    let mut pool = fresh_pool();
    let (nc, nf) = (128usize, 256usize);
    let count = nc * nf; // 32768 entries → tight density estimate
    let density = 0.25_f64;
    let v = ((1.0_f64 / density) / nc as f64).sqrt() as f32;

    let mat = sparse_achlioptas_matrix::<f32>(&mut pool, 4242, nc, nf, density).expect("valid");
    let host = mat.to_host(&pool);
    mat.release_into(&mut pool);
    assert_eq!(host.len(), count);

    let mut nonzero = 0usize;
    for &e in host.iter() {
        if e != 0.0 {
            nonzero += 1;
            // Every nonzero entry is exactly +v or −v (bit-exact cast value).
            assert!(
                (e - v).abs() < 1e-6 || (e + v).abs() < 1e-6,
                "Achlioptas nonzero {e} is not ±v={v} (backend={backend})"
            );
        }
    }
    let observed = nonzero as f64 / count as f64;
    // SE of a fraction ≈ sqrt(p(1-p)/N) ≈ 0.0024; a 0.03 band is generous.
    assert!(
        (observed - density).abs() < 0.03,
        "Achlioptas density {observed} not ≈ {density} (backend={backend})"
    );
}

/// Fisher–Yates permutation bijection: `permutation(seed, n)` is a true bijection
/// of `0..n` (sorted == 0..n — the UNBIASED `next_below` rejection-sampling draw,
/// NOT biased `% n`) and is seed-reproducible. PRIM-06.
#[test]
fn rng_permutation_bijection() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    // Sweep several seeds (RNG_TRIALS) and several n; every output is a bijection.
    for seed in 0..RNG_TRIALS as u64 {
        for &n in &[0usize, 1, 2, 7, 64, 257] {
            let perm = permutation(seed, n);
            assert_eq!(perm.len(), n, "permutation length (seed={seed}, n={n})");
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            let expected: Vec<usize> = (0..n).collect();
            assert_eq!(
                sorted, expected,
                "permutation(seed={seed}, n={n}) is not a bijection of 0..n (backend={backend})"
            );
        }
    }

    // Seed-reproducible; a different seed can differ (n large enough to vary).
    assert_eq!(permutation(11, 500), permutation(11, 500));
    assert_ne!(
        permutation(11, 500),
        permutation(22, 500),
        "different seeds should produce different permutations (backend={backend})"
    );
}

/// PoolStats memory gate for `rng.rs` (PRIM-06): the matrix is host-generated then
/// SINGLE-uploaded via `DeviceArray::from_host` — driving `gaussian_matrix` N
/// times at a fixed shape proves bounded allocations (no per-call growth after the
/// warmup) and conserved live/peak bytes after each generated matrix is released
/// (the D-10 `live_bytes`/`peak_bytes`/`allocations`/`reuses` PoolStats idiom from
/// `memory_gate_test.rs`). One `from_host` upload per call; no parallel device
/// scratch.
#[test]
fn rng_memory_gate() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");

    const N: usize = 6;
    let (nc, nf) = (32usize, 48usize);

    let mut pool = fresh_pool();

    let mut alloc_after: Vec<u64> = Vec::with_capacity(N);
    let mut live_after: Vec<u64> = Vec::with_capacity(N);
    let mut peak_after: Vec<u64> = Vec::with_capacity(N);

    for iter in 0..N {
        // Generate, read back, and RELEASE the matrix back to the pool each
        // iteration so the host-generate-then-single-upload footprint is bounded.
        let mat = gaussian_matrix::<f32>(&mut pool, iter as u64, nc, nf).expect("valid shape");
        let _host = mat.to_host(&pool);
        mat.release_into(&mut pool);

        let s = pool.stats();
        alloc_after.push(s.allocations);
        live_after.push(s.live_bytes);
        peak_after.push(s.peak_bytes);
    }

    // Use iteration 0 as warmup (first sight of the matrix size is a fresh
    // allocation); the steady-state invariants hold from iteration 1 onward.
    let alloc_baseline = alloc_after[1];
    let live_baseline = live_after[1];
    let peak_baseline = peak_after[1];

    for iter in 2..N {
        // HARD GATE: allocations BOUNDED — the released matrix buffer is served
        // from the free-list on every subsequent same-shape call, so fresh
        // allocations never grow past the first iteration.
        assert_eq!(
            alloc_after[iter], alloc_baseline,
            "rng memory gate (allocations bounded) FAILED on {backend}: iter {iter} \
             allocations={} != baseline={alloc_baseline} — the host-generated matrix is \
             NOT being reused from the free-list. stats={:?}",
            alloc_after[iter],
            pool.stats()
        );
        // HARD GATE: live_bytes CONSERVES — each generated matrix is released, so
        // live returns to the same value every steady-state iteration.
        assert_eq!(
            live_after[iter], live_baseline,
            "rng memory gate (live_bytes conserved) FAILED on {backend}: iter {iter} \
             live_bytes={} != baseline={live_baseline} — the matrix upload is not \
             released. stats={:?}",
            live_after[iter],
            pool.stats()
        );
        // HARD GATE: peak_bytes PLATEAUS — released buffer is reused in place, not
        // stacked, so the high-water mark stops growing after the warmup.
        assert_eq!(
            peak_after[iter], peak_baseline,
            "rng memory gate (peak_bytes bounded) FAILED on {backend}: iter {iter} \
             peak_bytes={} != baseline={peak_baseline} — peak grows with N (no reuse). \
             stats={:?}",
            peak_after[iter],
            pool.stats()
        );
    }

    println!(
        "rng memory gate backend={backend}: N={N} alloc_baseline={alloc_baseline} \
         live_baseline={live_baseline} peak_baseline={peak_baseline} final_stats={:?}",
        pool.stats()
    );
}
