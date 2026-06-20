//! Plan 07-02 — RNG projection-matrix primitive (PRIM-06) oracle/property tests.
//!
//! WAVE-0 SCAFFOLD (this file is created by plan 07-01). Every test function is
//! `#[ignore]` and asserts ONLY fixture-load + shape well-formedness — it makes
//! NO reference to the not-yet-existent `mlrs_backend::prims::rng` symbol (the
//! `rng.rs` body is an empty stub until plan 07-02). This is the 03-02 / 05-01
//! Wave-0 pattern: the test crate must COMPILE today; plan 07-02 removes the
//! `#[ignore]`, wires the real `rng::*` generator calls, and turns each stub into
//! the live property/distribution assertion.
//!
//! PRIM-06 gate (plan 07-02 wires): the Gaussian matrix's per-element stats
//! (mean ≈ 0, var ≈ 1/n_components), seed-reproducibility (same `u64` seed →
//! bit-identical matrix across runs/backends — the SplitMix64 host PRNG is
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

/// Pinned number of averaging trials for the RNG distribution/property checks
/// (plan 07-02 uses this when wiring the live stats so a single unlucky draw
/// never flips a strict band — D-11 averaging precedent).
const RNG_TRIALS: usize = 50;

/// Gaussian projection-matrix per-element distribution: mean ≈ 0,
/// var ≈ 1/n_components (PRIM-06). f32 path (cpu + rocm).
///
/// WAVE-0 STUB: asserts only that the trial constant is sane. Plan 07-02 removes
/// `#[ignore]` and wires `rng::gaussian(seed, n_features, n_components)` →
/// per-element mean/var over `RNG_TRIALS` draws.
#[test]
#[ignore = "wave-0 scaffold: prims::rng lands in plan 07-02"]
fn rng_gaussian_distribution() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(RNG_TRIALS >= 1, "RNG_TRIALS must be positive");
}

/// Seed-reproducibility: the SAME `u64` seed yields a bit-identical matrix across
/// independent generations (PRIM-06 / T-07-02 — host SplitMix64 is deterministic,
/// never OsRng). f64 path gated by `skip_f64_with_log` (cpu runs; rocm skips).
///
/// WAVE-0 STUB. Plan 07-02 wires two `rng::gaussian(seed, ..)` draws with the
/// same seed and asserts element-wise equality, then a different seed differs.
#[test]
#[ignore = "wave-0 scaffold: prims::rng lands in plan 07-02"]
fn rng_seed_reproducible() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("rng f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
        return;
    }
    assert!(RNG_TRIALS >= 1, "RNG_TRIALS must be positive");
}

/// Achlioptas sparse matrix density + value stats: the non-zero fraction ≈
/// `density` and the non-zero magnitude ≈ `sqrt((1/density)/n_components)`
/// (PRIM-06). f32 path.
///
/// WAVE-0 STUB. Plan 07-02 wires `rng::achlioptas(seed, .., density)` and checks
/// the empirical density + value over `RNG_TRIALS` draws.
#[test]
#[ignore = "wave-0 scaffold: prims::rng lands in plan 07-02"]
fn rng_achlioptas_density() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(RNG_TRIALS >= 1, "RNG_TRIALS must be positive");
}

/// Fisher–Yates permutation bijection: the generated permutation is a true
/// bijection of `0..n` (every index exactly once — the UNBIASED `next_below`
/// rejection-sampling draw, NOT biased `% n`). PRIM-06.
///
/// WAVE-0 STUB. Plan 07-02 wires `rng::permutation(seed, n)` and asserts the
/// output is a permutation (sorted == 0..n) + same-seed reproducible.
#[test]
#[ignore = "wave-0 scaffold: prims::rng lands in plan 07-02"]
fn rng_permutation_bijection() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(RNG_TRIALS >= 1, "RNG_TRIALS must be positive");
}

/// PoolStats memory gate for `rng.rs` (PRIM-06): the matrix is host-generated then
/// SINGLE-uploaded via `DeviceArray::from_host` — allocations bounded, the upload
/// metered exactly once (the D-10 precedent — `live_bytes`/`peak_bytes`/`reuses`
/// PoolStats idiom from `memory_gate_test.rs`).
///
/// WAVE-0 STUB. Plan 07-02 wires the N-iteration `rng::gaussian` generation and
/// asserts `pool.stats()` allocation/reuse bounds.
#[test]
#[ignore = "wave-0 scaffold: prims::rng + memory gate land in plan 07-02"]
fn rng_memory_gate() {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F32, backend, "default");
    assert!(RNG_TRIALS >= 1, "RNG_TRIALS must be positive");
}
