//! `umap_init` — UMAP embedding initialization (Plan 03's home).
//!
//! Plan 03 fills this with the two remaining deterministic value-gated init
//! stages of UMAP:
//!
//! - [`fit_ab`] — the host Levenberg–Marquardt a/b curve fit (D-06), porting
//!   scipy `curve_fit`'s LM least-squares of `1/(1 + a·x^(2b))` to umap's smooth
//!   target curve. Value-gated ≤1e-5 (f64) vs umap-learn 0.5.12 `find_ab_params`.
//! - [`spectral_init`] — spectral embedding init reusing the EXISTING
//!   `laplacian` + `eig` + `recover` stack on the symmetric fuzzy graph, with the
//!   n≤64 Jacobi cap and umap's RANDOM FALLBACK above it (UMAP falls back, does
//!   NOT error like `SpectralEmbedding`). Value-gated ≤1e-5 up-to-sign per column.
//! - [`random_init`] — `uniform(-10, 10)` via `SplitMix64` (D-05 backbone, reused
//!   by Plan 04; the spectral cap fallback).
//! - [`noisy_scale_coords`] — umap's `max=10`/`noise=1e-4` post-spectral scaling.
//!
//! NO device kernel is added here (D-06): `fit_ab` is a self-contained host f64
//! numeric; `spectral_init` composes the already-validated `laplacian`/`eig`
//! device prims and the host `recover` math. Tests live in
//! `crates/mlrs-algos/tests/umap_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::eig::eig;
use mlrs_backend::prims::laplacian::laplacian;
use mlrs_backend::prims::rng::SplitMix64;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64};

use crate::cluster::spectral::recover;
use crate::error::AlgoError;

/// The v1 dense-eig MAX_DIM cap (`eig.rs` `MAX_DIM = 64`). The normalized
/// Laplacian is `n × n`, so `n ≤ 64` is the spectral problem-size ceiling. UNLIKE
/// `SpectralEmbedding` (which ERRORS above the cap, D-06), UMAP's `spectral_init`
/// takes the RANDOM-INIT FALLBACK above it — umap-learn falls back to random
/// init rather than failing (RESEARCH Pattern 4 / Q-cap).
const MAX_DIM: usize = 64;

/// LM iteration cap (the `sgd`/`eig` MAX_SWEEPS DoS-guard precedent, ASVS V5 /
/// T-14-06). scipy `curve_fit` converges on this smooth 2-parameter problem in a
/// handful of iterations; 200 is a generous ceiling whose only purpose is to make
/// non-convergence a typed error instead of an unbounded loop.
const AB_MAX_ITER: usize = 200;

/// Number of curve sample points (`linspace(0, spread*3, 300)`, umap-learn
/// `find_ab_params`, RESEARCH Code Examples line 298).
const AB_CURVE_POINTS: usize = 300;

/// Host Levenberg–Marquardt fit of UMAP's `a`/`b` curve parameters (D-06),
/// porting scipy `curve_fit` (which itself runs MINPACK's LM). Matches umap-learn
/// 0.5.12 `find_ab_params(spread, min_dist)` to ≤1e-5 (f64).
///
/// Target curve (umap `find_ab_params`, RESEARCH Code Examples 296–300):
/// ```text
/// curve(x, a, b) = 1 / (1 + a · x^(2b))
/// xv = linspace(0, spread*3, 300)
/// yv = if xv < min_dist { 1 } else { exp(-(xv − min_dist) / spread) }
/// ```
/// We minimize `Σ_i (curve(xv_i, a, b) − yv_i)²` with the analytic Jacobian:
/// ```text
/// let p = x^(2b);  let denom = (1 + a·p)²
/// ∂curve/∂a = −p / denom
/// ∂curve/∂b = −a · p · 2·ln(x) / denom     (0 at x = 0, since p = 0 there)
/// ```
///
/// The LM loop is bounded by [`AB_MAX_ITER`]; non-convergence within the cap
/// returns [`AlgoError::NotConverged`] rather than looping unboundedly (T-14-06,
/// ASVS V5).
pub fn fit_ab(min_dist: f64, spread: f64) -> Result<(f64, f64), AlgoError> {
    // --- Build the target curve sample points (umap find_ab_params). ---
    let xv = linspace(0.0, spread * 3.0, AB_CURVE_POINTS);
    let yv: Vec<f64> = xv
        .iter()
        .map(|&x| {
            if x < min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();

    // scipy curve_fit's default initial parameter guess is p0 = [1, 1].
    let mut a = 1.0_f64;
    let mut b = 1.0_f64;

    // Levenberg–Marquardt damping (MINPACK-style). lambda scales the diagonal of
    // the Gauss–Newton normal matrix; it grows when a step worsens the SSE and
    // shrinks when it improves, interpolating between gradient descent and
    // Gauss–Newton.
    let mut lambda = 1e-3_f64;
    let mut sse = sum_sq_residual(&xv, &yv, a, b);

    for _ in 0..AB_MAX_ITER {
        // Assemble the 2×2 Gauss–Newton normal equations JᵀJ and gradient Jᵀr
        // from the analytic Jacobian (J has columns ∂/∂a, ∂/∂b per residual).
        let (mut jtj00, mut jtj01, mut jtj11) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut jtr0, mut jtr1) = (0.0_f64, 0.0_f64);
        for i in 0..xv.len() {
            let x = xv[i];
            let (resid, da, db) = residual_and_grad(x, yv[i], a, b);
            jtj00 += da * da;
            jtj01 += da * db;
            jtj11 += db * db;
            jtr0 += da * resid;
            jtr1 += db * resid;
        }

        // Solve (JᵀJ + λ·diag(JᵀJ)) Δ = −Jᵀr for the LM step, trying successively
        // larger λ until the step reduces the SSE (or the cap is reached).
        let mut accepted = false;
        for _ in 0..30 {
            let h00 = jtj00 * (1.0 + lambda);
            let h11 = jtj11 * (1.0 + lambda);
            let h01 = jtj01;
            let det = h00 * h11 - h01 * h01;
            if det.abs() < f64::MIN_POSITIVE {
                // Singular augmented system — raise damping and retry.
                lambda *= 10.0;
                continue;
            }
            // Δ = −H⁻¹ · Jᵀr.
            let da_step = -(h11 * jtr0 - h01 * jtr1) / det;
            let db_step = -(-h01 * jtr0 + h00 * jtr1) / det;
            let a_new = a + da_step;
            let b_new = b + db_step;
            let sse_new = sum_sq_residual(&xv, &yv, a_new, b_new);

            if sse_new.is_finite() && sse_new < sse {
                // Accept: commit the step and relax the damping toward
                // Gauss–Newton.
                a = a_new;
                b = b_new;
                let improvement = sse - sse_new;
                sse = sse_new;
                lambda = (lambda * 0.1).max(1e-12);
                accepted = true;
                // Converged when the SSE barely moves (scipy's ftol regime).
                if improvement <= 1e-15 * sse.max(1.0) {
                    return Ok((a, b));
                }
                break;
            }
            // Reject: tighten the trust region (raise λ) and retry the step.
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }

        if !accepted {
            // No λ in the inner loop produced an improving step: we are at a
            // (local) minimum to numerical precision — the fit has converged.
            return Ok((a, b));
        }
    }

    Err(AlgoError::NotConverged {
        estimator: "umap_fit_ab",
        max_iter: AB_MAX_ITER,
    })
}

/// numpy `linspace(start, stop, num)` (inclusive endpoints, even spacing).
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num as f64 - 1.0);
    (0..num).map(|i| start + step * i as f64).collect()
}

/// `curve(x, a, b) = 1 / (1 + a·x^(2b))`. At `x = 0`, `x^(2b) = 0` so `curve = 1`.
fn curve(x: f64, a: f64, b: f64) -> f64 {
    1.0 / (1.0 + a * x.powf(2.0 * b))
}

/// SSE `Σ (curve − y)²` for a candidate `(a, b)`.
fn sum_sq_residual(xv: &[f64], yv: &[f64], a: f64, b: f64) -> f64 {
    xv.iter()
        .zip(yv.iter())
        .map(|(&x, &y)| {
            let r = curve(x, a, b) - y;
            r * r
        })
        .sum()
}

/// Residual `r = curve − y` and its analytic gradient `(∂r/∂a, ∂r/∂b)` at one
/// sample point. `x = 0` short-circuits to a constant `curve = 1` with zero
/// gradient (the `x^(2b)` factor — and hence both partials — vanish there, and
/// the `ln(x)` in `∂/∂b` is multiplied by that vanishing factor).
fn residual_and_grad(x: f64, y: f64, a: f64, b: f64) -> (f64, f64, f64) {
    if x <= 0.0 {
        // curve(0) = 1; both partials are 0.
        return (1.0 - y, 0.0, 0.0);
    }
    let p = x.powf(2.0 * b); // x^(2b)
    let denom = 1.0 + a * p;
    let denom2 = denom * denom;
    let resid = 1.0 / denom - y;
    let d_a = -p / denom2;
    let d_b = -(a * p * 2.0 * x.ln()) / denom2;
    (resid, d_a, d_b)
}

/// UMAP spectral embedding init (UMAP-02) over the symmetric fuzzy graph
/// `g_affinity` (`n × n`, row-major, device-resident). Reuses the EXISTING
/// `laplacian` + `eig` + `recover` stack — NO re-derived Laplacian or eig.
///
/// Pipeline (mirroring `spectral_embedding.rs:239-283`, but with the diffusion
/// recovery gated OFF — see the recovery note below):
/// `laplacian(g, n)` `(L, dd)` → `eig(L)` (DESCENDING; `recover` reverses to
/// ascending internally) → `recover(.., drop_first = true, diffusion_recover =
/// false)` (slice smallest → drop the trivial ≈0 eigenvector → transpose).
/// Returns the row-major `n × n_components` RAW spectral coords — exactly umap's
/// `spectral_layout` output, the value-gate target.
///
/// `noisy_scale_coords` (`max = 10`, `noise = 1e-4`) is umap's SEPARATE
/// post-spectral stage (applied by `simplicial_set_embedding`, NOT
/// `spectral_layout`), exposed as a standalone host helper for Plan 04's `fit`
/// pipeline — applying it here would break the ≤1e-5 value-gate against the raw
/// `spectral_layout` fixture.
///
/// Cap + fallback (T-14-07): when `n > MAX_DIM (== 64)`, this returns
/// [`random_init`] WITHOUT erroring or launching `eig` — umap's documented
/// random-init fallback above the Jacobi cap (RESEARCH Pattern 4). All RNG draws
/// are order-deterministic from `seed` (D-05 backbone; reused by Plan 04).
///
/// Recovery note (RESEARCH Q3/A3, dump-diff confirmed): umap's `spectral_layout`
/// decomposes the SAME `I − D^-1/2 A D^-1/2` Laplacian but returns the RAW
/// eigenvectors — NO `/dd` diffusion recovery and NO deterministic sign flip
/// (the dump-diff measured the `/dd` path mismatching umap by ~0.2 and the raw
/// path matching to ≤1e-6). So the shared `recover` is called with
/// `diffusion_recover = false`; the spectral-family callers keep
/// `diffusion_recover = true` and stay bit-identical. The value-gate still
/// compares up-to-sign per column (eigenvectors are sign-arbitrary).
pub fn spectral_init<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    g_affinity: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    n_components: usize,
    seed: u64,
) -> Result<Vec<F>, AlgoError>
where
    F: Float + CubeElement + Pod,
{
    // --- n > 64: take the random-init FALLBACK before any eig launch (T-14-07).
    //     UMAP falls back rather than erroring (unlike SpectralEmbedding). ---
    if n > MAX_DIM {
        return Ok(random_init::<F>(n, n_components, seed));
    }

    // --- Normalized Laplacian (L, dd) = laplacian(G, n) (reuse PRIM-09). ---
    let (l, dd) = laplacian::<F>(pool, g_affinity, n)?;
    let dd_host: Vec<f64> = dd.to_host(pool).iter().map(|&v| host_to_f64(v)).collect();
    dd.release_into(pool);

    // --- Full symmetric spectrum via v1 eig (DESCENDING, V col-major). Thread
    //     the Laplacian buffer through `out` so eig reuses it as its working input
    //     — saving one n² allocation (RESEARCH Anti-Pattern).
    //
    //     WR-05/WR-06: `&l` (the eig `a` input) and `l_out` (the eig `out`) wrap
    //     the SAME ref-counted cubecl handle (l.handle().clone()). This aliasing
    //     is SOUND only because of two load-bearing, eig-internal invariants
    //     (the same argument carried verbatim at spectral_embedding.rs:248-259
    //     and spectral_clustering.rs — this is the THIRD copy of the pattern, so
    //     the full soundness comment is repeated here rather than referenced):
    //       (1) eig READS `a_in` (= the `out` handle) and NEVER writes it — it
    //           writes its separate w/V/info outputs (eig.rs jacobi_eig_sweep);
    //       (2) eig ACQUIRES w/V/info from the pool BEFORE it releases the `out`
    //           working buffer (eig.rs: acquire happens before the
    //           `a_in_owned.release_into(pool)` post-launch), so the freed
    //           handle is never re-handed mid-call.
    //     If eig ever writes its working input in place, or reorders the
    //     acquire/release, this reuse becomes an aliased-write / use-after-free
    //     with NO compile-time signal — keep those invariants if eig changes.
    let l_out =
        DeviceArray::<ActiveRuntime, F>::from_raw(l.handle().clone(), n * n);
    let (w_desc, v_desc) = eig::<F>(pool, &l, n, Some(l_out))?;
    drop(l);
    w_desc.release_into(pool);
    let v_host: Vec<f64> = v_desc
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect();
    v_desc.release_into(pool);

    // --- Post-eig recovery (reuse the shared host `recover`, drop_first = true,
    //     diffusion_recover = FALSE): slice smallest → drop the trivial ≈0
    //     eigenvector → transpose → row-major n × n_components. umap's
    //     `spectral_layout` returns the RAW symmetric-Laplacian eigenvectors with
    //     NO /dd recovery and NO sign flip (dump-diff: /dd mismatches umap by
    //     ~0.2, raw matches ≤1e-6 — RESEARCH Q3/A3 confirmed empirically), so the
    //     diffusion-recovery family is gated OFF for the UMAP path. The shared
    //     `recover` keeps the spectral-family callers bit-identical (they pass
    //     `diffusion_recover = true`). ---
    // NOTE: noisy_scale_coords is NOT applied here (it is umap's separate
    // post-spectral stage; applying it would break the ≤1e-5 gate vs the raw
    // spectral_layout fixture). `seed` is retained in the signature for the
    // n > 64 random fallback above and for Plan 04's call-site symmetry.
    let _ = seed;
    Ok(recover::<F>(&v_host, &dd_host, n, n_components, true, false))
}

/// umap `noisy_scale_coords` (RESEARCH Pattern 4): expand the coords so the max
/// absolute value is `max_coord`, then add small Gaussian noise.
///
/// ```text
/// expansion = max_coord / max|coords|
/// coords *= expansion
/// coords += N(0, noise)         (via SplitMix64 Box–Muller, seed-deterministic)
/// ```
/// Operates in place on the row-major `n × n_components` `coords`. If all coords
/// are zero (degenerate), the expansion is skipped (no division by zero) and only
/// the noise is applied — matching umap's `max(1, max|coords|)`-style guard.
///
/// IN-04: `n` and `n_components` exist SOLELY for the debug-build shape assertion
/// (`debug_assert_eq!(coords.len(), n * n_components)`) — they are not used by the
/// scaling/noise computation and are dead parameters in release builds. They are
/// kept so callers carry the intended shape next to the buffer (a cheap
/// debug-only invariant check), not because the routine needs them.
pub fn noisy_scale_coords(
    coords: &mut [f64],
    n: usize,
    n_components: usize,
    max_coord: f64,
    noise: f64,
    seed: u64,
) {
    debug_assert_eq!(coords.len(), n * n_components);
    let max_abs = coords.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if max_abs > 0.0 {
        let expansion = max_coord / max_abs;
        for c in coords.iter_mut() {
            *c *= expansion;
        }
    }
    // Additive Gaussian noise, seed-deterministic Box–Muller draws (the rng.rs
    // gaussian_matrix precedent), consuming the cached second sample before
    // drawing a fresh pair so the stream is fully determined by `seed`.
    let mut rng = SplitMix64::new(seed);
    let mut cached: Option<f64> = None;
    for c in coords.iter_mut() {
        let z = next_standard_normal(&mut rng, &mut cached);
        *c += z * noise;
    }
}

/// UMAP random init (UMAP-02 fallback): `uniform(-10, 10)` row-major
/// `n × n_components`, seed-deterministic via [`SplitMix64::next_f64`]. This is
/// the n > 64 spectral fallback AND the explicit `init = "random"` path; the draw
/// order is fully determined by `seed` (D-05 backbone reused by Plan 04).
pub fn random_init<F>(n: usize, n_components: usize, seed: u64) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let mut rng = SplitMix64::new(seed);
    (0..n * n_components)
        .map(|_| {
            // next_f64 ∈ [0, 1) → uniform(-10, 10).
            let u = rng.next_f64();
            f64_to_host::<F>(u * 20.0 - 10.0)
        })
        .collect()
}

/// One standard-normal sample via Box–Muller (the `rng.rs::gaussian_matrix`
/// pattern): consume the cached second member of a pair before drawing a fresh
/// pair so the stream is fully `seed`-determined.
fn next_standard_normal(rng: &mut SplitMix64, cached: &mut Option<f64>) -> f64 {
    if let Some(z1) = cached.take() {
        return z1;
    }
    let mut u1 = rng.next_f64();
    let min_u1 = 2.0_f64.powi(-53);
    if u1 < min_u1 {
        u1 = min_u1;
    }
    let u2 = rng.next_f64();
    let r = (-2.0_f64 * u1.ln()).sqrt();
    let theta = 2.0_f64 * std::f64::consts::PI * u2;
    *cached = Some(r * theta.sin());
    r * theta.cos()
}
