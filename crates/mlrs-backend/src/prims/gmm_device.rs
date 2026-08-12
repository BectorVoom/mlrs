//! `gmm_device` — the DEVICE-RESIDENT EM engine behind `GaussianMixture`'s CUDA
//! fast path, and the twin of [`crate::prims::gmm_host`].
//!
//! ## What moved, and what deliberately did not
//! [`gmm_host`](crate::prims::gmm_host)'s module docs give three structural
//! reasons the mixture EM loop is host-resident on every backend: launch
//! overhead dominates a `max_iter · n_init` loop of tiny passes, the reduction
//! must be `f64` (and cuda's `supports_type(F64)` under-reports that
//! capability), and the per-iteration tail is `O(k·d³)`, serial and branchy —
//! exactly the shape a GPU is worst at. This module does not argue with any of
//! that. [`GmmDevice`] keeps the ENTIRE `O(k·d³)` tail — Cholesky, triangular
//! inverse, log-determinant — on the HOST, called from the estimator every
//! iteration exactly as it is today (`gaussian_mixture.rs`'s `fit_core` calls
//! `precisions_cholesky` on whatever `covariances` this module returns, same as
//! it calls it on [`gmm_host::GmmHost::covariances`]'s result). What moves is
//! ONLY the two passes whose cost scales with `n`:
//!
//! - the E-step's weighted-log-prob → responsibility-normalize → `nk`+`means`
//!   reduction ([`GmmDevice::e_step`]), and
//! - the M-step's covariance reduction ([`GmmDevice::covariances`]).
//!
//! `X` (`n × d`) and `resp` (`n × k`) are uploaded/allocated ONCE per fit and
//! stay device-resident for the WHOLE `max_iter` loop of every restart — never
//! downloaded mid-loop — exactly the shape `KMeans`'s Lloyd loop uses
//! (`crates/mlrs-algos/src/cluster/kmeans.rs::single_run` +
//! `crates/mlrs-backend/src/prims/kmeans.rs::centroid_sums_dev`). Only
//! `O(k·d)`-to-`O(k·d²)`-sized scalars round-trip to host each iteration:
//! `weights`/`means`/`precisions_cholesky` up (a few KB even at `k=32, d=256`),
//! `mean_lpn`/`nk`/`means`/`covariances` down. This is what answers reason #1
//! above without touching #2 or #3: the FIXED per-launch dispatch cost is now
//! paid a constant number of times per iteration rather than scaling with the
//! `O(n·k·d)` data crossing the bus, so it stops dominating once `n` is large
//! enough — [`gmm_device_applicable`]'s size floor is where "large enough"
//! starts.
//!
//! ## The `f64` gate is TWO checks, not one
//! [`gmm_device_applicable`] rejects a backend whose
//! [`crate::capability::f64_device_kernels_available`] is `false` — the correct
//! (not under-reporting) `f64`-ARITHMETIC probe, per that function's own docs.
//! But the E-step's `logsumexp` needs `f64` TRANSCENDENTALS (`.exp()`/`.ln()`
//! in [`mlrs_kernels::gmm::gmm_resp_normalize_rows`]), which is a STRICTLY
//! narrower capability
//! ([`crate::capability::f64_transcendental_supported`]'s own docs: a wgpu
//! adapter can advertise plain `f64` arithmetic and still SIGSEGV the driver's
//! shader compiler on an `f64 exp`). `prims::lbfgs::softmax_loss_grad` hit
//! exactly this evaluating a softmax at `f64` on wgpu; this module's E-step is
//! the same shape (a per-row `logsumexp`), so it is gated the same way. In
//! practice that makes the device arm reachable on cuda/rocm and never on wgpu
//! at `f64` — which is fine, because [`gmm_host`](crate::prims::gmm_host)
//! remains fully correct and is what every backend falls back to.
//!
//! ## What is NOT reduced with a `gemm`, and why
//! The per-iteration `nk`/`means`/covariance reductions are dedicated
//! ROW-BLOCKED GATHER kernels
//! ([`mlrs_kernels::gmm::gmm_soft_sumcount_blocked`],
//! [`mlrs_kernels::gmm::gmm_cov_diag_blocked`],
//! [`mlrs_kernels::gmm::gmm_cov_full_blocked`]), never `prims::gemm` and never
//! `prims::reduce::{row_reduce, column_reduce}`. The latter two do a FULL host
//! round-trip and re-upload ONE ROW/COLUMN AT A TIME
//! ([[mlrs-row-reduce-shared-landmine]]) — pathological inside a `max_iter`
//! loop. A `gemm`-shaped `respᵀ·X` formation was tried for exactly this
//! per-iteration weighted-sum-by-group shape in the `KMeans` campaign and
//! measured CATASTROPHICALLY slow ([[mlrs-kmeans-fit-optimization]]); this
//! module never repeats that experiment. The ONE place a `gemm`-STYLE dense
//! reduction is fine is [`GmmDevice::ensure_xtx`] — the `tied` covariance's
//! `XᵀX`, which (mirroring `gmm_host::GmmHost::ensure_xtx`, win #2 in that
//! module's docs) is computed EXACTLY ONCE per fit, not once per iteration, so
//! its cost is amortized over the whole `max_iter · n_init` loop rather than
//! paid every pass.
//!
//! ## Dense, not packed-triangular
//! `gmm_host` walks only the stored upper triangle of a `full`/`tied`
//! precision-Cholesky or covariance block — a CPU cache optimization
//! ([`gmm_host`](crate::prims::gmm_host) module docs, win #1). The device
//! kernels here read/write the DENSE `d × d` block instead (both symmetric
//! halves, redundantly) — see
//! [`mlrs_kernels::gmm::gmm_wlp_direct`]/[`mlrs_kernels::gmm::gmm_cov_full_blocked`]'s
//! own docs for why: a GPU thread doing `O(d²)` uniform work per output is a
//! better trade than a triangular loop nest that serializes unevenly across
//! threads, and the buffer layout `gmm_host` already produces (dense with
//! zeros below the diagonal) needs no translation to be read this way.
//!
//! Tests live in `crates/mlrs-algos/tests/gaussian_mixture_device_test.rs`
//! (AGENTS.md §2).

use cubecl::prelude::*;

use mlrs_core::PrimError;
use mlrs_kernels::gmm::{
    gmm_cov_diag_blocked, gmm_cov_full_blocked, gmm_entropy_rows, gmm_fold_partials,
    gmm_resp_normalize_rows, gmm_soft_sumcount_blocked, gmm_wlp_direct, gmm_xtx_blocked,
};

use cubecl::server::Handle;

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::gmm_host::{log_det_cholesky, CovarianceType, LOG_2PI, NK_EPS};
use crate::runtime::ActiveRuntime;

/// Is a device EM engine ([`GmmDevice`]) available AND worth taking for this
/// `(n, d, k)` shape?
///
/// Gates, in the order they are evaluated — correctness first, preference
/// last, mirroring [`crate::prims::normal_eq::device_gram_applicable`] (this
/// module's cited dispatch-predicate template):
///
/// 1. **`cpu` backend → never.** `cubecl-cpu` spawns one OS thread per unit and
///    JITs at `-O0`; every `n`-heavy reduction in this crate that tried a
///    device shape there lost to a host loop.
/// 2. **No `f64` ARITHMETIC kernels on the adapter → never.** The reduction is
///    `f64` throughout (module docs). This is
///    [`crate::capability::f64_device_kernels_available`], deliberately NOT
///    `supports_type(F64)` — see that function's docs for why the latter
///    under-reports on cuda specifically.
/// 3. **No `f64` TRANSCENDENTALS on the adapter → never.** A STRICTLY narrower
///    capability than #2 — see the module docs' "`f64` gate is TWO checks"
///    section. Gated separately because conflating them would either crash a
///    wgpu adapter's shader compiler (module docs) or (if folded into a single
///    weaker check) silently disable the arm on cuda, where it should run.
/// 4. **`MLRS_GMM_DEVICE` override.** `"0"` forces the host arm even where the
///    gates above hold; any other value forces the device arm (still subject
///    to gates 1–3, which are correctness, not preference — the
///    `RidgeClassifier::device_predict_applicable` override idiom, but applied
///    only to the PREFERENCE decision, never past a correctness gate). Read
///    through [`crate::abflag`] so a test can scope it without an environment
///    data race.
/// 5. **Size floor.** Below a conservative `n·k·d` product the fixed
///    per-launch dispatch cost still dominates a `max_iter · n_init` loop of
///    tiny passes (`gmm_host` reason #1, unmodified by this module) — see
///    [`GMM_DEVICE_MIN_WORK`].
/// Whether this backend can RUN the device EM engine at all — a capability,
/// not a preference.
///
/// The E-step evaluates a Gaussian log-density, so it needs `f64` device
/// kernels AND `f64` transcendentals. A backend missing them does not fail at
/// launch — the driver's shader compiler can SEGFAULT (see
/// `umap_host_knn.rs`). Split out of [`gmm_device_applicable`] so
/// `device="gpu"` overrides the SIZE half without overriding this one.
///
/// The cpu-backend check stays on the perf side deliberately: `cubecl-cpu`
/// compiles and runs these kernels correctly, just slowly, so forcing them
/// there is a legitimate (if unwise) request.
pub fn gmm_device_possible() -> bool {
    crate::capability::f64_device_kernels_available()
        && crate::capability::f64_transcendental_supported()
}

pub fn gmm_device_applicable(n: usize, d: usize, k: usize) -> bool {
    if crate::capability::active_backend_name() == "cpu" {
        return false;
    }
    if !gmm_device_possible() {
        return false;
    }
    if let Some(v) = crate::abflag::var("MLRS_GMM_DEVICE") {
        return v != "0";
    }
    let work = n.saturating_mul(k).saturating_mul(d.max(1));
    work >= GMM_DEVICE_MIN_WORK
}

/// `n·k·d` floor below which [`gmm_device_applicable`] keeps the fit on
/// [`crate::prims::gmm_host`] by default.
///
/// MEASURED, not guessed (this repo's own rule —
/// [[mlrs-feedback-verify-on-target-hardware]] — is never to gate a perf kernel
/// from a different backend's number, or from an un-A/B'd intuition either):
/// `gaussian_mixture_perf_test.rs`'s device-vs-host ladder run on a Tesla
/// P100 (2026-08-05) showed the device arm winning by 4.7–14.5× at EVERY rung
/// tested, including the smallest (`n=2 000, d=16, k=8` — `work = 256 000`,
/// 5.5× faster than host). The floor is set just above that smallest verified
/// win rather than at it, since nothing below `256 000` has been measured. A
/// re-A/B on a different card (or at a smaller `work`) is how this constant
/// should keep moving; `MLRS_GMM_DEVICE=1` bypasses it entirely for that
/// measurement.
const GMM_DEVICE_MIN_WORK: usize = 1 << 18;

/// `0/1/2/3` flag [`mlrs_kernels::gmm::gmm_wlp_direct`] branches its
/// Mahalanobis form on, matching [`CovarianceType`]'s variant order.
fn ct_flag(ct: CovarianceType) -> u32 {
    match ct {
        CovarianceType::Full => 0,
        CovarianceType::Tied => 1,
        CovarianceType::Diag => 2,
        CovarianceType::Spherical => 3,
    }
}

/// WR-03: reject a `usize` dimension that does not fit the kernel-launch `u32`
/// (an unguarded `as u32` truncation becomes an out-of-bounds device read).
fn guard_u32(operand: &'static str, dim: usize) -> Result<(), PrimError> {
    if dim > u32::MAX as usize {
        return Err(PrimError::ShapeMismatch {
            operand,
            rows: dim,
            cols: 0,
            len: u32::MAX as usize,
        });
    }
    Ok(())
}

/// Row-block layout for a stage-1 blocked-partial kernel whose per-block
/// partial is `stride` `f64` elements: `nb_cap = max(min_blocks, budget /
/// stride)` blocks, refined to the row count actually available.
///
/// `budget` bounds the INTERMEDIATE `nblocks · stride` partial buffer, not the
/// final reduced output — the concern [`GmmDevice::cov_full`] and
/// [`GmmDevice::ensure_xtx`] have that the `nk`/`means` reduction does not
/// (their `stride` is `O(k·d)`, small; `cov_full`'s dense `d²` stride can be
/// large enough that a blind `centroid_sums_dev`-style `.max(64)` floor would
/// blow the device-memory budget).
fn blocked_layout(n: usize, stride: usize, budget: usize, min_blocks: usize) -> (usize, usize) {
    let nb_cap = (budget / stride.max(1)).max(min_blocks);
    let nb = n.div_ceil(256).clamp(1, nb_cap);
    let rpb = n.div_ceil(nb);
    let nb = n.div_ceil(rpb);
    (nb, rpb)
}

/// Budget (in `f64` elements) for the `nk`/`means`/`cov_diag` blocked
/// partials — small per-block strides (`O(k·d)`), so a generous block count is
/// cheap and helps occupancy (the `centroid_sums_dev` precedent).
const REDUCE_BUDGET_SMALL: usize = 8 << 20;

/// Budget (in `f64` elements) for the `cov_full`/`xtx` blocked partials — the
/// per-block stride is `O(k·d²)`/`O(d²)`, which can be large enough that the
/// small-reduction budget above would allocate an unreasonable INTERMEDIATE
/// buffer; capped tighter and with NO forced minimum block count.
const REDUCE_BUDGET_DENSE: usize = 4 << 20;

/// The device-resident EM engine for one `mlrs_algos::mixture::gaussian_mixture::GaussianMixture`
/// fit: owns the uploaded `n × d` design and the `n × k` responsibility matrix
/// for the WHOLE `max_iter · n_init` loop, plus (lazily) the `tied`
/// covariance's loop-invariant `XᵀX`.
///
/// A drop-in per-iteration replacement for
/// [`gmm_host::GmmHost`](crate::prims::gmm_host::GmmHost): [`GmmDevice::e_step`]
/// and [`GmmDevice::covariances`] have the same INPUT/OUTPUT shape as
/// [`gmm_host::GmmHost::e_step`](crate::prims::gmm_host::GmmHost::e_step) /
/// [`gmm_host::GmmHost::covariances`](crate::prims::gmm_host::GmmHost::covariances),
/// modulo the `pool` + `Result` every device launch needs — the estimator's
/// iteration loop can call either engine without duplicating its structure
/// (`gaussian_mixture.rs::fit_core`).
pub struct GmmDevice {
    x: DeviceArray<ActiveRuntime, f64>,
    resp: DeviceArray<ActiveRuntime, f64>,
    n: usize,
    d: usize,
    k: usize,
    ct: CovarianceType,
    reg_covar: f64,
    /// The `tied` M-step's loop-invariant `XᵀX` (dense `d × d`), materialized
    /// on first use by [`GmmDevice::ensure_xtx`] and reused for every
    /// subsequent iteration of every restart. `None` for every other
    /// `covariance_type`, which never populates it.
    xtx: Option<Vec<f64>>,
}

impl GmmDevice {
    /// Upload the `n × d` HOST `f64` design ONCE and allocate the `n × k`
    /// responsibility matrix, both kept device-resident for the caller's whole
    /// fit (every restart, every iteration).
    pub fn new(
        pool: &mut BufferPool<ActiveRuntime>,
        x_host: &[f64],
        n: usize,
        d: usize,
        k: usize,
        ct: CovarianceType,
        reg_covar: f64,
    ) -> Result<Self, PrimError> {
        if n.checked_mul(d).map(|v| v != x_host.len()).unwrap_or(true) {
            return Err(PrimError::ShapeMismatch {
                operand: "x",
                rows: n,
                cols: d,
                len: x_host.len(),
            });
        }
        guard_u32("n", n)?;
        guard_u32("d", d)?;
        guard_u32("k", k)?;
        let x = DeviceArray::<ActiveRuntime, f64>::from_host(pool, x_host);
        let resp_len = n * k;
        let resp =
            DeviceArray::<ActiveRuntime, f64>::from_raw(pool.acquire(resp_len * 8), resp_len);
        Ok(Self {
            x,
            resp,
            n,
            d,
            k,
            ct,
            reg_covar,
            xtx: None,
        })
    }

    /// The current responsibilities, read back to host (`n × k`, row-major) —
    /// the device twin of
    /// [`gmm_host::GmmHost::resp`](crate::prims::gmm_host::GmmHost::resp),
    /// used ONLY for the fit's terminal label extraction (`argmax_rows`), never
    /// inside the iteration loop.
    pub fn resp_to_host(&self, pool: &BufferPool<ActiveRuntime>) -> Vec<f64> {
        self.resp.to_host(pool)
    }

    /// Release the device-resident `X`, `resp`, and (if populated) the cached
    /// `XᵀX` staging back to the pool. The caller must call this exactly once
    /// per [`GmmDevice`], after the fit is done with it (mirrors every other
    /// `DeviceArray`-owning prim's `release_into` convention).
    pub fn release_into(self, pool: &mut BufferPool<ActiveRuntime>) {
        self.x.release_into(pool);
        self.resp.release_into(pool);
    }

    /// One fused device E-step: overwrites `resp` in place and returns
    /// `(mean_log_prob_norm, nk, means)` — the same triple
    /// [`gmm_host::GmmHost::e_step`](crate::prims::gmm_host::GmmHost::e_step)
    /// returns.
    ///
    /// `weights`/`means`/`prec_chol` are the CURRENT (host, `f64`) iterate —
    /// tiny (`O(k·d²)` at most) and re-uploaded every call, which is the
    /// intended per-iteration host traffic (module docs).
    pub fn e_step(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        weights: &[f64],
        means: &[f64],
        prec_chol: &[f64],
    ) -> Result<(f64, Vec<f64>, Vec<f64>), PrimError> {
        let (n, d, k, ct) = (self.n, self.d, self.k, self.ct);
        // The O(k) bias fold (module docs — this stays host arithmetic
        // exactly like `gmm_host::GmmHost::e_step`'s): plain EM's bias is
        // `ln(weight_c)` plus the Cholesky log-det plus the `-0.5*d*ln(2pi)`
        // normalizer, all folded by `wlp_normalize`.
        let log_det = log_det_cholesky(prec_chol, k, d, ct);
        let bias: Vec<f64> = (0..k)
            .map(|c| weights[c].ln() + log_det[c] - 0.5 * d as f64 * LOG_2PI)
            .collect();

        let (wlp, wlp_len, lse) = self.wlp_normalize(pool, &bias, means, prec_chol)?;
        pool.release(wlp, wlp_len * 8);

        // --- mean_lpn: fold the n-length lse partials the SAME way KMeans
        //     folds its per-row shift/changed partials — a row-blocked
        //     partial-sum + a tiny readback, never an O(n) one. ---
        let lse_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(lse, n);
        let mean_lpn = super::kmeans::sum_device::<f64>(pool, &lse_dev, n)? / n as f64;
        lse_dev.release_into(pool);

        // --- nk / means: the soft-weight generalization of
        //     `centroid_sums_dev`'s two-stage blocked reduction. ---
        let (nk, means_out) = self.nk_means_reduce(pool)?;

        Ok((mean_lpn, nk, means_out))
    }

    /// The VARIATIONAL E-step: the device twin of
    /// [`gmm_host::GmmHost::e_step_biased`](crate::prims::gmm_host::GmmHost::e_step_biased).
    /// `log_weight_term[c]` REPLACES `ln(weight_c)` verbatim in the bias fold
    /// (see that method's docs for why this is the whole degree of freedom
    /// `BayesianGaussianMixture` needs) — every other pass (the Mahalanobis
    /// kernel, the `nk`/`means` reduction, [`GmmDevice::covariances`]) is
    /// byte-for-byte the same code [`GmmDevice::e_step`] uses, shared through
    /// [`GmmDevice::wlp_normalize`].
    ///
    /// The extra return is `Σ_i Σ_c r_ic·ln r_ic` (sklearn's
    /// `np.sum(np.exp(log_resp) * log_resp)`), computed by
    /// [`mlrs_kernels::gmm::gmm_entropy_rows`] from the SAME `wlp`/`lse`
    /// buffers `wlp_normalize` already produced, before they are released —
    /// one extra small kernel launch per iteration, never a second `O(n·k·d²)`
    /// pass.
    pub fn e_step_biased(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        log_weight_term: &[f64],
        means: &[f64],
        prec_chol: &[f64],
    ) -> Result<(f64, Vec<f64>, Vec<f64>, f64), PrimError> {
        let (n, d, k, ct) = (self.n, self.d, self.k, self.ct);
        let log_det = log_det_cholesky(prec_chol, k, d, ct);
        let bias: Vec<f64> = (0..k)
            .map(|c| log_weight_term[c] + log_det[c] - 0.5 * d as f64 * LOG_2PI)
            .collect();

        let (wlp, wlp_len, lse) = self.wlp_normalize(pool, &bias, means, prec_chol)?;

        // --- Entropy: one thread per row, reusing `wlp`/`lse` before they are
        //     released (module docs on `gmm_entropy_rows`). ---
        let client = pool.client().clone();
        let ent = pool.acquire(n * 8);
        {
            let (count, dim) =
                super::launch_dims_1d_folded(n, crate::capability::gather_launch_width());
            // SAFETY: `wlp`/`lse`/`ent` are sized to `n*k`/`n`/`n`; the kernel
            // bounds-checks `i < n` and reads `wlp` only at offsets `< n*k`.
            let wlp_arg = unsafe { ArrayArg::from_raw_parts(wlp.clone(), wlp_len) };
            let lse_arg = unsafe { ArrayArg::from_raw_parts(lse.clone(), n) };
            let ent_arg = unsafe { ArrayArg::from_raw_parts(ent.clone(), n) };
            gmm_entropy_rows::launch::<f64, ActiveRuntime>(
                &client, count, dim, wlp_arg, lse_arg, ent_arg, n as u32, k as u32,
            );
        }
        pool.release(wlp, wlp_len * 8);

        let lse_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(lse, n);
        let mean_lpn = super::kmeans::sum_device::<f64>(pool, &lse_dev, n)? / n as f64;
        lse_dev.release_into(pool);

        let ent_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(ent, n);
        let resp_log_resp = super::kmeans::sum_device::<f64>(pool, &ent_dev, n)?;
        ent_dev.release_into(pool);

        let (nk, means_out) = self.nk_means_reduce(pool)?;

        Ok((mean_lpn, nk, means_out, resp_log_resp))
    }

    /// Shared body of [`GmmDevice::e_step`]/[`GmmDevice::e_step_biased`]:
    /// given the per-component `bias` already folded (`ln(weight)` or
    /// `log_weight_term`, plus the Cholesky log-det and the `-0.5*d*ln(2pi)`
    /// normalizer), runs the `gmm_wlp_direct` → `gmm_resp_normalize_rows`
    /// pair, writing `self.resp` IN PLACE. Returns the STILL-ACQUIRED `wlp`
    /// handle (with its length) and the `lse` handle so a caller that also
    /// needs the entropy term can fold one more kernel over them before
    /// releasing `wlp` — the caller owns both releases.
    fn wlp_normalize(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        bias: &[f64],
        means: &[f64],
        prec_chol: &[f64],
    ) -> Result<(Handle, usize, Handle), PrimError> {
        let (n, d, k, ct) = (self.n, self.d, self.k, self.ct);
        guard_u32("n", n)?;
        guard_u32("d", d)?;
        guard_u32("k", k)?;

        let client = pool.client().clone();
        let means_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, means);
        let prec_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, prec_chol);
        let bias_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, bias);

        // --- Phase 1: wlp[i,c] (n x k GATHER, one thread per pair). ---
        let wlp_len = n * k;
        let wlp = pool.acquire(wlp_len * 8);
        {
            let (count, dim) =
                super::launch_dims_1d_folded(wlp_len, crate::capability::gather_launch_width());
            // SAFETY: every length below is the caller-validated element count
            // of a live buffer; the kernel bounds-checks `tid < n*k` and reads
            // `prec`/`means`/`bias` only at offsets `< k*d*d`/`< k*d`/`< k`,
            // which the uploads above are sized to.
            let x_arg = unsafe { ArrayArg::from_raw_parts(self.x.handle().clone(), self.x.len()) };
            let m_arg =
                unsafe { ArrayArg::from_raw_parts(means_dev.handle().clone(), means_dev.len()) };
            let p_arg =
                unsafe { ArrayArg::from_raw_parts(prec_dev.handle().clone(), prec_dev.len()) };
            let b_arg =
                unsafe { ArrayArg::from_raw_parts(bias_dev.handle().clone(), bias_dev.len()) };
            let w_arg = unsafe { ArrayArg::from_raw_parts(wlp.clone(), wlp_len) };
            gmm_wlp_direct::launch::<f64, ActiveRuntime>(
                &client,
                count,
                dim,
                x_arg,
                m_arg,
                p_arg,
                b_arg,
                w_arg,
                n as u32,
                d as u32,
                k as u32,
                ct_flag(ct),
            );
        }
        means_dev.release_into(pool);
        prec_dev.release_into(pool);
        bias_dev.release_into(pool);

        // --- Phase 2: row-wise logsumexp + resp normalize (n GATHER, one
        //     thread per row), writing resp IN PLACE and a per-row lse. ---
        let lse = pool.acquire(n * 8);
        {
            let (count, dim) =
                super::launch_dims_1d_folded(n, crate::capability::gather_launch_width());
            // SAFETY: `wlp`/`resp`/`lse` are sized to `n*k`/`n*k`/`n`; the
            // kernel bounds-checks `i < n`.
            let wlp_arg = unsafe { ArrayArg::from_raw_parts(wlp.clone(), wlp_len) };
            let r_arg =
                unsafe { ArrayArg::from_raw_parts(self.resp.handle().clone(), self.resp.len()) };
            let l_arg = unsafe { ArrayArg::from_raw_parts(lse.clone(), n) };
            gmm_resp_normalize_rows::launch::<f64, ActiveRuntime>(
                &client, count, dim, wlp_arg, r_arg, l_arg, n as u32, k as u32,
            );
        }

        Ok((wlp, wlp_len, lse))
    }

    /// Two-stage blocked `nk`/`means` reduction from the CURRENT `resp` —
    /// [`mlrs_kernels::gmm::gmm_soft_sumcount_blocked`] then
    /// [`mlrs_kernels::gmm::gmm_fold_partials`] (twice: once for the `k·d`
    /// `means` numerator, once for the `k` `nk`), finished on host exactly like
    /// [`gmm_host::GmmHost::e_step`](crate::prims::gmm_host::GmmHost::e_step)'s
    /// tail (`nk += NK_EPS`, `means /= nk`).
    fn nk_means_reduce(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
    ) -> Result<(Vec<f64>, Vec<f64>), PrimError> {
        let (n, d, k) = (self.n, self.d, self.k);
        let kd = k * d;
        let (nb, rpb) = blocked_layout(n, kd, REDUCE_BUDGET_SMALL, 64);
        let client = pool.client().clone();

        let psums_len = nb * kd;
        let pnk_len = nb * k;
        let psums = pool.acquire(psums_len * 8);
        let pnk = pool.acquire(pnk_len * 8);
        {
            // SAFETY: validated element counts; the kernel bounds-checks its
            // unit id against `nblocks*k*d` and clamps the block row range to
            // `n`.
            let x_arg = unsafe { ArrayArg::from_raw_parts(self.x.handle().clone(), self.x.len()) };
            let r_arg =
                unsafe { ArrayArg::from_raw_parts(self.resp.handle().clone(), self.resp.len()) };
            let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let pn_arg = unsafe { ArrayArg::from_raw_parts(pnk.clone(), pnk_len) };
            let (count, dim) = super::launch_dims_1d_folded(psums_len, super::PERF_TUNED_BLOCK);
            gmm_soft_sumcount_blocked::launch::<f64, ActiveRuntime>(
                &client, count, dim, x_arg, r_arg, ps_arg, pn_arg, n as u32, d as u32, k as u32,
                nb as u32, rpb as u32,
            );
        }

        let sums = pool.acquire(kd * 8);
        let nk_buf = pool.acquire(k * 8);
        {
            // SAFETY: `psums`/`pnk` are sized to `nb*kd`/`nb*k`; the fold
            // kernel bounds-checks `tid < len`.
            let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let s_arg = unsafe { ArrayArg::from_raw_parts(sums.clone(), kd) };
            let (c1, d1) = super::launch_dims_1d_folded(kd, super::PERF_TUNED_BLOCK);
            gmm_fold_partials::launch::<f64, ActiveRuntime>(
                &client, c1, d1, ps_arg, s_arg, kd as u32, nb as u32,
            );

            let pn_arg = unsafe { ArrayArg::from_raw_parts(pnk.clone(), pnk_len) };
            let nk_arg = unsafe { ArrayArg::from_raw_parts(nk_buf.clone(), k) };
            let (c2, d2) = super::launch_dims_1d_folded(k, super::PERF_TUNED_BLOCK);
            gmm_fold_partials::launch::<f64, ActiveRuntime>(
                &client, c2, d2, pn_arg, nk_arg, k as u32, nb as u32,
            );
        }

        let sums_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(sums, kd);
        let nk_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(nk_buf, k);
        let sums_host = sums_dev.to_host(pool);
        let mut nk = nk_dev.to_host(pool);
        sums_dev.release_into(pool);
        nk_dev.release_into(pool);
        pool.release(psums, psums_len * 8);
        pool.release(pnk, pnk_len * 8);

        let mut means_out = vec![0.0f64; kd];
        for c in 0..k {
            nk[c] += NK_EPS;
            let inv = 1.0 / nk[c];
            for j in 0..d {
                means_out[c * d + j] = sums_host[c * d + j] * inv;
            }
        }
        Ok((nk, means_out))
    }

    /// The M-step covariance for the CURRENT `resp`/`nk`/`means`, in the
    /// [`CovarianceType::param_len`] layout with `reg_covar` already on the
    /// diagonal — the drop-in twin of
    /// [`gmm_host::GmmHost::covariances`](crate::prims::gmm_host::GmmHost::covariances).
    pub fn covariances(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        nk: &[f64],
        means: &[f64],
    ) -> Result<Vec<f64>, PrimError> {
        match self.ct {
            CovarianceType::Full => self.cov_full(pool, nk, means),
            CovarianceType::Tied => self.cov_tied(pool, nk, means),
            CovarianceType::Diag => self.cov_diag(pool, nk, means),
            CovarianceType::Spherical => {
                let diag = self.cov_diag(pool, nk, means)?;
                let d = self.d;
                Ok((0..self.k)
                    .map(|c| diag[c * d..(c + 1) * d].iter().sum::<f64>() / d as f64)
                    .collect())
            }
        }
    }

    /// `diag`: blocked `Σ resp[i,c]·x[i,j]²` partial, finished on host as
    /// `s/nk_c − μ_cj² + reg_covar` (mirrors
    /// [`gmm_host::GmmHost::cov_diag`](crate::prims::gmm_host::GmmHost)'s
    /// tail).
    fn cov_diag(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        nk: &[f64],
        means: &[f64],
    ) -> Result<Vec<f64>, PrimError> {
        let (n, d, k) = (self.n, self.d, self.k);
        let kd = k * d;
        let (nb, rpb) = blocked_layout(n, kd, REDUCE_BUDGET_SMALL, 64);
        let client = pool.client().clone();

        let psums_len = nb * kd;
        let psums = pool.acquire(psums_len * 8);
        {
            // SAFETY: validated element counts; bounds-checked in-kernel.
            let x_arg = unsafe { ArrayArg::from_raw_parts(self.x.handle().clone(), self.x.len()) };
            let r_arg =
                unsafe { ArrayArg::from_raw_parts(self.resp.handle().clone(), self.resp.len()) };
            let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let (count, dim) = super::launch_dims_1d_folded(psums_len, super::PERF_TUNED_BLOCK);
            gmm_cov_diag_blocked::launch::<f64, ActiveRuntime>(
                &client, count, dim, x_arg, r_arg, ps_arg, n as u32, d as u32, k as u32, nb as u32,
                rpb as u32,
            );
        }

        let sums = pool.acquire(kd * 8);
        {
            let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let s_arg = unsafe { ArrayArg::from_raw_parts(sums.clone(), kd) };
            let (c1, d1) = super::launch_dims_1d_folded(kd, super::PERF_TUNED_BLOCK);
            gmm_fold_partials::launch::<f64, ActiveRuntime>(
                &client, c1, d1, ps_arg, s_arg, kd as u32, nb as u32,
            );
        }

        let sums_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(sums, kd);
        let sums_host = sums_dev.to_host(pool);
        sums_dev.release_into(pool);
        pool.release(psums, psums_len * 8);

        let mut out = vec![0.0f64; kd];
        for c in 0..k {
            let inv = 1.0 / nk[c];
            for j in 0..d {
                let m = means[c * d + j];
                out[c * d + j] = sums_host[c * d + j] * inv - m * m + self.reg_covar;
            }
        }
        Ok(out)
    }

    /// `full`: blocked DENSE weighted outer-product partial, finished on host
    /// as `s/nk_c` with `reg_covar` on the `a == b` diagonal (mirrors
    /// [`gmm_host::GmmHost::cov_full`](crate::prims::gmm_host::GmmHost)'s tail,
    /// minus its packed-triangle unpacking — this buffer is already dense).
    fn cov_full(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        nk: &[f64],
        means: &[f64],
    ) -> Result<Vec<f64>, PrimError> {
        let (n, d, k) = (self.n, self.d, self.k);
        let dd = d * d;
        let kdd = k * dd;
        let (nb, rpb) = blocked_layout(n, kdd, REDUCE_BUDGET_DENSE, 1);
        let client = pool.client().clone();
        let means_dev = DeviceArray::<ActiveRuntime, f64>::from_host(pool, means);

        let psums_len = nb * kdd;
        let psums = pool.acquire(psums_len * 8);
        {
            // SAFETY: validated element counts; bounds-checked in-kernel.
            let x_arg = unsafe { ArrayArg::from_raw_parts(self.x.handle().clone(), self.x.len()) };
            let r_arg =
                unsafe { ArrayArg::from_raw_parts(self.resp.handle().clone(), self.resp.len()) };
            let m_arg =
                unsafe { ArrayArg::from_raw_parts(means_dev.handle().clone(), means_dev.len()) };
            let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let (count, dim) = super::launch_dims_1d_folded(psums_len, super::PERF_TUNED_BLOCK);
            gmm_cov_full_blocked::launch::<f64, ActiveRuntime>(
                &client, count, dim, x_arg, r_arg, m_arg, ps_arg, n as u32, d as u32, k as u32,
                nb as u32, rpb as u32,
            );
        }
        means_dev.release_into(pool);

        let sums = pool.acquire(kdd * 8);
        {
            let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let s_arg = unsafe { ArrayArg::from_raw_parts(sums.clone(), kdd) };
            let (c1, d1) = super::launch_dims_1d_folded(kdd, super::PERF_TUNED_BLOCK);
            gmm_fold_partials::launch::<f64, ActiveRuntime>(
                &client, c1, d1, ps_arg, s_arg, kdd as u32, nb as u32,
            );
        }

        let sums_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(sums, kdd);
        let sums_host = sums_dev.to_host(pool);
        sums_dev.release_into(pool);
        pool.release(psums, psums_len * 8);

        let mut out = vec![0.0f64; kdd];
        for c in 0..k {
            let inv = 1.0 / nk[c];
            let base = c * dd;
            for a in 0..d {
                for b in 0..d {
                    out[base + a * d + b] = sums_host[base + a * d + b] * inv;
                }
            }
            for a in 0..d {
                out[base + a * d + a] += self.reg_covar;
            }
        }
        Ok(out)
    }

    /// `tied`: `(XᵀX − Σ_c nk_c·μ_c·μ_cᵀ) / Σnk`, with `XᵀX` cached by
    /// [`GmmDevice::ensure_xtx`] (mirrors
    /// [`gmm_host::GmmHost::cov_tied`](crate::prims::gmm_host::GmmHost)
    /// exactly, from a dense rather than packed-triangle `XᵀX`).
    fn cov_tied(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        nk: &[f64],
        means: &[f64],
    ) -> Result<Vec<f64>, PrimError> {
        self.ensure_xtx(pool)?;
        let d = self.d;
        let xtx = self
            .xtx
            .as_ref()
            .expect("ensure_xtx populates self.xtx before returning Ok");
        let nk_sum: f64 = nk.iter().sum();
        let mut out = vec![0.0f64; d * d];
        for a in 0..d {
            for b in 0..d {
                let mut s = xtx[a * d + b];
                for (c, &nkc) in nk.iter().enumerate() {
                    s -= nkc * means[c * d + a] * means[c * d + b];
                }
                out[a * d + b] = s / nk_sum;
            }
        }
        for a in 0..d {
            out[a * d + a] += self.reg_covar;
        }
        Ok(out)
    }

    /// Materialize the DENSE `d × d` `XᵀX` ONCE per fit (module docs) — a
    /// no-op on every call after the first.
    fn ensure_xtx(&mut self, pool: &mut BufferPool<ActiveRuntime>) -> Result<(), PrimError> {
        if self.xtx.is_some() {
            return Ok(());
        }
        let (n, d) = (self.n, self.d);
        let dd = d * d;
        let (nb, rpb) = blocked_layout(n, dd, REDUCE_BUDGET_DENSE, 1);
        let client = pool.client().clone();

        let psums_len = nb * dd;
        let psums = pool.acquire(psums_len * 8);
        {
            // SAFETY: validated element counts; bounds-checked in-kernel.
            let x_arg = unsafe { ArrayArg::from_raw_parts(self.x.handle().clone(), self.x.len()) };
            let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let (count, dim) = super::launch_dims_1d_folded(psums_len, super::PERF_TUNED_BLOCK);
            gmm_xtx_blocked::launch::<f64, ActiveRuntime>(
                &client, count, dim, x_arg, ps_arg, n as u32, d as u32, nb as u32, rpb as u32,
            );
        }

        let out = pool.acquire(dd * 8);
        {
            let ps_arg = unsafe { ArrayArg::from_raw_parts(psums.clone(), psums_len) };
            let o_arg = unsafe { ArrayArg::from_raw_parts(out.clone(), dd) };
            let (c1, d1) = super::launch_dims_1d_folded(dd, super::PERF_TUNED_BLOCK);
            gmm_fold_partials::launch::<f64, ActiveRuntime>(
                &client, c1, d1, ps_arg, o_arg, dd as u32, nb as u32,
            );
        }

        let out_dev = DeviceArray::<ActiveRuntime, f64>::from_raw(out, dd);
        let xtx_host = out_dev.to_host(pool);
        out_dev.release_into(pool);
        pool.release(psums, psums_len * 8);
        self.xtx = Some(xtx_host);
        Ok(())
    }
}
