//! `rng` — host-side random projection-matrix generation (PRIM-06).
//!
//! Backend-reproducible, ASVS-V6-compliant (no `OsRng`, no `rand` crate) RNG
//! serving `GaussianRandomProjection` / `SparseRandomProjection` (PROJ-01/02)
//! and any future shuffle (Phase-10 MBSGD). Reproducibility across cpu/rocm is
//! the PRIM-06 hard gate and requires a HOST PRNG — a device RNG is
//! backend-divergent (RESEARCH Anti-Pattern "device-side RNG"). The whole prim
//! is HOST-side glue: generate on the host → ONE [`DeviceArray::from_host`]
//! upload, so there is NO device kernel and the cpu-MLIR SharedMemory/atomics
//! landmines are dodged entirely.
//!
//! - [`SplitMix64`] — the seeded host PRNG (Steele, Lea & Flood 2014), PROMOTED
//!   VERBATIM out of [`crate::prims::kmeans`] (RESEARCH Pitfall 7 — the mix
//!   constants are byte-frozen; altering them changes the stream and breaks
//!   `kmeanspp_test.rs`). It is NOT a CSPRNG and is NEVER seeded from `OsRng`.
//! - [`gaussian_matrix`] — a Box–Muller Gaussian matrix scaled `N(0, 1/n_components)`.
//! - [`sparse_achlioptas_matrix`] — the Achlioptas sparse matrix with value
//!   `v = sqrt((1/density)/n_components)`, stored DENSE (D-12).
//! - [`permutation`] — an UNBIASED Fisher–Yates shuffle via
//!   [`SplitMix64::next_below`] (rejection sampling, NEVER `next_u64() % n`).
//!
//! ## Validate before any allocation (ASVS V5)
//! `gaussian_matrix` / `sparse_achlioptas_matrix` reject `n_components == 0`,
//! `n_features == 0`, and (for Achlioptas) `density ∉ (0, 1]` as a typed
//! [`PrimError::ShapeMismatch`] BEFORE any host allocation or device upload.
//! The `PrimError` layer has no `InvalidDensity` variant (that lives in the
//! `AlgoError` estimator layer, validated estimator-side in Plan 06); a bad
//! `density` surfaces here as a synthetic `"density"` ShapeMismatch.
//!
//! Tests live in `crates/mlrs-backend/tests/rng_test.rs` (AGENTS.md §2 — no
//! in-source `#[cfg(test)] mod tests`).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::PrimError;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

// ===========================================================================
// Host-side documented seeded PRNG (ASVS V6 — NEVER OsRng; T-07-02)
//
// PROMOTED VERBATIM from `prims::kmeans` (was private there). The mix constants
// and method bodies are byte-frozen (RESEARCH Pitfall 7): `kmeans::kmeanspp_sample`
// now `use`s this exact struct, so any change to the stream would break
// `kmeanspp_test.rs` / `lloyd_test.rs`. Made `pub` on the move.
// ===========================================================================

/// SplitMix64 — a small, well-documented, fully-deterministic seeded PRNG
/// (Steele, Lea & Flood, 2014). Used HOST-SIDE only for the k-means++ draw and
/// the RNG-matrix generators so the sampler is seed-reproducible across runs and
/// backends. It is NOT a CSPRNG and is NEVER seeded from `OsRng` — the seed is
/// the caller's documented `u64` (T-07-02 / RESEARCH "host-side documented
/// seeded RNG").
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator with the caller's documented `u64`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit value (the canonical SplitMix64 mix).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Next `f64` in `[0, 1)` (53-bit mantissa precision).
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits → [0, 1) with full double mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// UNBIASED uniform integer in `[0, bound)` via rejection sampling (CR-02).
    /// A plain `next_u64() % bound` is biased whenever `bound` does not divide
    /// `2^64`; we reject the top non-uniform residue so every value is equally
    /// likely. `bound` must be `>= 1` (the caller guarantees `n >= k >= 1`).
    pub fn next_below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound >= 1, "next_below requires a positive bound");
        // WR-07: enforce the contract in release too. A `bound == 0` would skip
        // the (release no-op) debug_assert and the `== 1` branch and reach
        // `u64::MAX % bound` below — an opaque divide-by-zero arithmetic panic.
        // The only sensible value in `[0, 0)` is the empty-range degenerate `0`.
        if bound <= 1 {
            return 0;
        }
        // Largest multiple of `bound` that fits in u64; values at or above it are
        // the biased tail and get rejected. `zone = bound * (u64::MAX / bound)`
        // rounded down to the last full block; use the (MAX - MAX % bound) form to
        // avoid overflow.
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let v = self.next_u64();
            if v < zone {
                return v % bound;
            }
        }
    }
}

// ===========================================================================
// RNG-matrix generators (host generate → single upload).
// ===========================================================================

/// Generate the `n_components × n_features` Gaussian random-projection matrix,
/// each entry `~ N(0, 1/n_components)` (RESEARCH Pattern 4), row-major flat
/// layout. Box–Muller from pairs of [`SplitMix64::next_f64`] uniforms scaled by
/// `1/sqrt(n_components)` (i.e. a standard-normal sample divided by
/// `sqrt(n_components)`).
///
/// HOST-side glue: the matrix is built in an `f64` accumulator, cast to `F`, and
/// uploaded with ONE [`DeviceArray::from_host`] (the D-10 memory-gate contract —
/// no device kernel, no parallel scratch). Generic over `F` (`f32` / `f64`); the
/// f64 path is caller-gated by `skip_f64_with_log`.
///
/// `n_components == 0` / `n_features == 0` are rejected as a typed
/// [`PrimError::ShapeMismatch`] BEFORE any allocation (ASVS V5).
pub fn gaussian_matrix<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    seed: u64,
    n_components: usize,
    n_features: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_shape(n_components, n_features)?;

    let count = n_components * n_features;
    let scale = 1.0_f64 / (n_components as f64).sqrt();
    let mut host: Vec<F> = Vec::with_capacity(count);

    let mut rng = SplitMix64::new(seed);
    // Box–Muller produces standard-normal samples in pairs (z0, z1) from two
    // uniforms (u1, u2): r = sqrt(-2 ln u1), z0 = r·cos(2πu2), z1 = r·sin(2πu2).
    // We consume the cached second sample before drawing a fresh pair so the
    // stream is fully determined by `seed` (T-07-02). u1 is floored away from 0
    // so ln(u1) is finite.
    let mut cached: Option<f64> = None;
    for _ in 0..count {
        let z = match cached.take() {
            Some(z1) => z1,
            None => {
                let mut u1 = rng.next_f64();
                if u1 <= f64::MIN_POSITIVE {
                    // next_f64() ∈ [0,1); avoid ln(0) = -inf on the (vanishingly
                    // rare) exact-zero draw. Deterministic, seed-stable.
                    u1 = f64::MIN_POSITIVE;
                }
                let u2 = rng.next_f64();
                let r = (-2.0_f64 * u1.ln()).sqrt();
                let theta = 2.0_f64 * std::f64::consts::PI * u2;
                cached = Some(r * theta.sin());
                r * theta.cos()
            }
        };
        host.push(f64_to_host::<F>(z * scale));
    }

    Ok(DeviceArray::from_host(pool, &host))
}

/// Generate the `n_components × n_features` Achlioptas SPARSE random-projection
/// matrix (RESEARCH Pattern 4), stored DENSE (D-12), row-major flat layout. Each
/// entry is:
/// - `0` with probability `1 − density`,
/// - `+v` with probability `density/2`, `−v` with probability `density/2`,
///
/// where `v = sqrt((1/density)/n_components)`. The branch is drawn from a single
/// [`SplitMix64::next_f64`] uniform per entry.
///
/// HOST-side glue: built in an `f64` accumulator, cast to `F`, ONE upload. The
/// dense storage matches the v1 device GEMM `transform == X · componentsᵀ`
/// (no sparse device kernel — D-12). Generic over `F`.
///
/// `n_components == 0` / `n_features == 0` reject as a `"n_components"` /
/// `"n_features"` [`PrimError::ShapeMismatch`]; `density ∉ (0, 1]` rejects as a
/// synthetic `"density"` ShapeMismatch (the `PrimError` layer has no
/// `InvalidDensity` — that's the estimator-side `AlgoError`) — all BEFORE any
/// allocation (ASVS V5).
pub fn sparse_achlioptas_matrix<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    seed: u64,
    n_components: usize,
    n_features: usize,
    density: f64,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate_shape(n_components, n_features)?;
    // density ∈ (0, 1] (ASVS V5). The PrimError layer has no InvalidDensity
    // variant — surface a synthetic "density" ShapeMismatch (the estimator layer
    // raises AlgoError::InvalidDensity before calling down here).
    if !(density.is_finite() && density > 0.0 && density <= 1.0) {
        return Err(PrimError::ShapeMismatch {
            operand: "density",
            rows: 0,
            cols: 1,
            len: 0,
        });
    }

    let count = n_components * n_features;
    // v = sqrt((1/density)/n_components) = sqrt(1/density) / sqrt(n_components).
    let v = ((1.0_f64 / density) / n_components as f64).sqrt();
    let plus = f64_to_host::<F>(v);
    let minus = f64_to_host::<F>(-v);
    let zero = f64_to_host::<F>(0.0);

    let mut host: Vec<F> = Vec::with_capacity(count);
    let mut rng = SplitMix64::new(seed);
    let half_density = density / 2.0;
    for _ in 0..count {
        // u ∈ [0, 1): [0, density/2) → +v ; [density/2, density) → −v ;
        // [density, 1) → 0. Exactly density mass on the nonzero branches.
        let u = rng.next_f64();
        let e = if u < half_density {
            plus
        } else if u < density {
            minus
        } else {
            zero
        };
        host.push(e);
    }

    Ok(DeviceArray::from_host(pool, &host))
}

/// UNBIASED Fisher–Yates permutation of `0..n` seeded by `seed`, returned as a
/// host `Vec<usize>` (PRIM-06). Each swap index is drawn with
/// [`SplitMix64::next_below`] (rejection sampling) — NEVER `next_u64() % n`
/// (RESEARCH Anti-Pattern "biased modulo"). Same `seed` → same permutation.
///
/// `n == 0` yields the empty permutation (no draws).
pub fn permutation(seed: u64, n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    if n < 2 {
        return perm;
    }
    let mut rng = SplitMix64::new(seed);
    // Standard Fisher–Yates: for i from n-1 down to 1, swap perm[i] with
    // perm[j] where j is uniform in [0, i] (= next_below(i+1), unbiased).
    for i in (1..n).rev() {
        let j = rng.next_below((i + 1) as u64) as usize;
        perm.swap(i, j);
    }
    perm
}

// ===========================================================================
// Hyperparameter / shape guards (ASVS V5 — reject before any allocation).
// ===========================================================================

/// Reject a degenerate matrix shape (`n_components == 0` / `n_features == 0`) as
/// a typed [`PrimError::ShapeMismatch`] BEFORE any host allocation or device
/// upload (ASVS V5 / T-07-04).
fn validate_shape(n_components: usize, n_features: usize) -> Result<(), PrimError> {
    if n_components == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "n_components",
            rows: n_components,
            cols: n_features,
            len: 0,
        });
    }
    if n_features == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "n_features",
            rows: n_components,
            cols: n_features,
            len: 0,
        });
    }
    Ok(())
}

// ===========================================================================
// f32/f64 host bit-cast helper (mirror kmeans.rs / reduce.rs — F is f32/f64 only)
// ===========================================================================

/// Inverse of a host-to-f64 cast: build an `F` (f32 / f64) from an `f64`.
/// Copied VERBATIM from `kmeans.rs` (the `host_to_f64` companion is not needed
/// here — every generator accumulates in f64 and only writes back to F).
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("rng prims are f32/f64 only"),
    }
}
