//! `ransac_device` — the DEVICE-RESIDENT trial scan behind `RANSACRegressor`
//! (RANSAC-02), and the twin of [`ransac_host`](super::ransac_host).
//!
//! ## What changed since "host-resident on every backend"
//! [`ransac_host`](super::ransac_host)'s module docs argue — correctly — that a
//! device arm cannot win a loop that pays one launch, one `coef` upload and one
//! full host STALL per trial, because `max_trials` shrinks as a function of the
//! best consensus found so far and the loop therefore needs each trial's inlier
//! count before it draws the next sub-sample.
//!
//! This module does not contradict that; it removes the per-TRIAL part of it.
//! The trials inside a batch are mutually independent — a scan reads the design
//! and its own candidate model, nothing else — so `B` of them are drawn, solved
//! and scanned speculatively in one launch and the sequential bookkeeping is
//! replayed over the results in trial order afterwards. A stop rule that fires
//! at trial `k < B` discards the surplus scans and rewinds the draw stream, so
//! the fitted answer is the unbatched loop's, exactly; the cost is the wasted
//! work of `B − k − 1` trials, and the gain is that the launch and the stall are
//! now paid `⌈trials/B⌉` times instead of `trials` times. See
//! [`mlrs_kernels::ransac`] for the kernel-side statement of the same argument.
//!
//! ## The transfer budget, which is the whole design
//! Per FIT: the design goes up once (`n·d` + `n·t`) and never comes back.
//!
//! Per BATCH: `B·t·d` coefficients and `B·t` intercepts up (kilobytes), and
//! `B·nblocks·(1 + 2t)` partials down. **Nothing of size `n` is transferred.**
//! The inlier mask stays device-resident and is read back only when a trial
//! actually BECOMES the incumbent — under ten times in a typical fit — and the
//! R² denominator is formed on the device against that resident mask.
//!
//! ## What deliberately stays on the host
//! The sub-sample least-squares solve. With sklearn's default
//! `min_samples = d + 1` it is a `(d+1) × d` system, which is launch overhead on
//! a GPU ([[mlrs-gpu-perf-root-cause]]), and — more importantly — its Householder
//! QR rank verdict decides which trials produce a usable model at all, i.e. it
//! is CONTROL FLOW that reproduces `scipy.linalg.lstsq`. Moving it to a device
//! kernel would put a parity-critical decision on a second numerical
//! implementation for no measured gain. It does get batched on the host, across
//! the same pool ([`RansacHostEngine::subset_lstsq_batch`](super::ransac_host::RansacHostEngine::subset_lstsq_batch)),
//! which is a real win at `d ≥ 64` where that solve dominates.
//!
//! ## Arms agree, but not bit-for-bit
//! The device dot product accumulates serially in `F`; the host's accumulates in
//! [`DOT_LANES`](super::ransac_host) independent lanes and its reductions run in
//! `f64`. Both are legitimate summation orders and neither is sklearn's BLAS
//! `gemv` order, so a row sitting *exactly* on `residual_threshold` can be
//! classified differently by the two arms — the caveat
//! [`ransac_host`](super::ransac_host) already documents against numpy, now
//! applying between mlrs's own arms. `ransac_device_test.rs` gates the two
//! against each other on a fixture with a clear inlier margin, which is the
//! regime where the classification is well-posed.
//!
//! Tests live in `crates/mlrs-algos/tests/ransac_device_test.rs` (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::{ArrayArg, CubeElement, Float};
use cubecl::server::Handle;

use mlrs_core::PrimError;
use mlrs_kernels::ransac::{ransac_den_block, ransac_scan_batch, RANSAC_SCAN_BASE};

use crate::device::Device;
use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::prims::ransac_host::{RansacLoss, TrialScan};
use crate::runtime::ActiveRuntime;

/// Trials scanned in one launch, by default.
///
/// The batch exists to amortize the launch and the host stall, so it wants to be
/// large; it is bounded by the wasted work a stop rule discards, so it wants to
/// be small. Eight is the compromise, and it is not arbitrary: sklearn's default
/// `stop_probability = 0.99` typically leaves `max_trials` in the tens after the
/// first good consensus, so a batch of eight is still several batches of real
/// bookkeeping, while a batch of thirty-two would routinely overrun the whole
/// remaining budget.
///
/// The driver additionally clamps each batch to the trials that actually REMAIN
/// ([`ransac_batch_width`]), so the last batch of a fit never speculates past
/// the loop's own bound.
///
/// ## The measurement, which is also this engine's whole argument
/// Same rocm gfx1151 / `f32` / min-of-5 setup as
/// [`RANSAC_DEVICE_MIN_WORK`]'s table, sweeping `MLRS_RANSAC_BATCH` on the
/// device arm (host arm's wall in the last row for scale):
///
/// | B | 100 000 × 64 | 50 000 × 128 |
/// |---|---|---|
/// | 1 | 235.0 ms | 262.4 ms |
/// | 2 | 114.1 ms | 165.1 ms |
/// | 4 | 112.0 ms | 142.6 ms |
/// | **8** | **95.7 ms** | **125.9 ms** |
/// | 16 | 97.4 ms | 131.4 ms |
/// | *host* | *94.5 ms* | *153.0 ms* |
///
/// At `B = 1` — one launch and one host stall per trial, the shape
/// [`ransac_host`](super::ransac_host)'s docs said could never win — the device
/// arm LOSES to the host by 2.5× and 1.7×, exactly as predicted. Batching is
/// worth 2.5× / 2.1× on its own and is the entire difference between that and
/// parity-or-better. The knee is at eight and 16 is already slightly worse:
/// past the knee the launch is amortized and the only thing still growing is the
/// work a stop rule throws away.
pub const RANSAC_DEVICE_BATCH: usize = 8;

/// Units the block plan aims for across the whole batch — enough to fill a GPU
/// without shrinking a block to the point where the partial readback becomes the
/// cost it was meant to avoid.
const TARGET_UNITS: usize = 2048;

/// Fewest rows a device block may own. Below this the `B·nblocks·(1 + 2t)`
/// partial readback stops being small, and a block's inlier COUNT — accumulated
/// as an `F` and therefore exact only inside `f32`'s integer range — has no
/// reason to be split further.
const MIN_ROWS_PER_BLOCK: usize = 32;

/// `n·d` floor below which [`ransac_device_applicable`] keeps the fit on the
/// host engine.
///
/// **`usize::MAX` — the device arm is opt-in, via `device="gpu"` or
/// `MLRS_RANSAC_ENGINE=device`.** Not because it is slow: on the development
/// hardware it reaches PARITY with the host arm at the top of the ladder and
/// wins one rung of it. It stays opt-in because "parity, sometimes better" is
/// not a reason to move every fit onto a device by default, and the one rung it
/// loses badly (`200 000 × 32`) sits inside any simple `n·d` gate that would
/// admit the rung it wins.
///
/// rocm, `f32`, gfx1151 iGPU, 2026-08-12, idle box, min-of-5, each arm in its
/// own process (`scripts/bench_ransac_cpu.py --engine mlrs --device cpu|gpu`):
///
/// | n × d | n·d | host | device (B=8) | |
/// |---|---|---|---|---|
/// | 1 000 × 8 | 8 K | **0.8 ms** | 5.0 ms | 6.3× |
/// | 10 000 × 8 | 80 K | **2.7 ms** | 5.5 ms | 2.0× |
/// | 10 000 × 64 | 640 K | **18.0 ms** | 23.0 ms | 1.28× |
/// | 100 000 × 16 | 1.6 M | **40.1 ms** | 41.8 ms | 1.04× |
/// | 100 000 × 64 | 6.4 M | 94.5 ms | 95.7 ms | parity |
/// | 50 000 × 128 | 6.4 M | 153.0 ms | **125.9 ms** | **0.82×** |
/// | 200 000 × 32 | 6.4 M | **81.0 ms** | 110.2 ms | 1.36× |
///
/// Two things in that table are worth more than the ratios.
///
/// **The host CPU column collapses.** At `50 000 × 128` the host arm burns
/// 0.76 s of CPU across its worker pool for 0.15 s of wall; the device arm burns
/// 0.25 s. A fit that leaves three quarters of the machine free is a different
/// proposition from one that saturates it, and neither the wall column nor a
/// crossover constant expresses that — which is the other reason this is a
/// caller's choice rather than a heuristic's.
///
/// **Both arms are the SAME memory system.** At `100 000 × 64` a trial reads
/// 25.6 MB and takes ~0.95 ms on both, i.e. ~27 GB/s on both — this iGPU shares
/// DRAM with the host, and the host scan already runs at DRAM bandwidth
/// ([[mlrs-huber-gpu-device-engine]] hit the same wall). There is no separate
/// memory system to bring, so the device can only win where it uses that
/// bandwidth better (large `d`, where a row is a longer contiguous run). A
/// DISCRETE card holds the design in its own HBM after ONE upload and re-reads
/// it `max_trials` times for free, which is exactly the shape this engine was
/// built for — measure there and lower this constant rather than assuming it.
pub const RANSAC_DEVICE_MIN_WORK: usize = usize::MAX;

/// Whether this backend can RUN the device scan at all — a capability, not a
/// preference.
///
/// The kernels are cubecl-cpu MLIR-safe by construction
/// ([`mlrs_kernels::ransac`] house rules), so unlike
/// [`huber_device_possible`](super::huber_objective::huber_device_possible) this
/// does NOT exclude the cpu backend: `device="gpu"` on a cpu build is a legal
/// request that selects the `cubecl-cpu` kernel path, exactly as
/// [`Device`](crate::device) documents. It is normally much slower there, and a
/// caller who asks for it gets it.
///
/// The one real gate is `f64`: an `f64` design needs `f64` device kernels, and
/// this asks the ARITHMETIC probe rather than `supports_type(F64)` (which
/// under-reports on cuda — see
/// [`f64_device_kernels_available`](crate::capability::f64_device_kernels_available)).
/// No transcendental is involved — the scan is multiply-add, compare, add — so
/// the narrower transcendental probe `gmm_device` needs does not apply here.
///
/// Split out of [`ransac_device_applicable`] so `device="gpu"` can override the
/// SIZE half of that gate without overriding this one: a caller who asks for the
/// device arm at `f64` on an `f32`-only adapter gets the host arm and a
/// `device_` of `"cpu"`, which is visible rather than faked.
pub fn ransac_device_possible<F: Pod>() -> bool {
    size_of::<F>() != 8 || crate::capability::f64_device_kernels_available()
}

/// Whether an `n × d × t` RANSAC fit should run its scan on the DEVICE.
///
/// Gates, in order — correctness first, preference last (the
/// `gmm_device_applicable` template):
///
/// 1. **[`ransac_device_possible`]** — an `f64` design on an adapter without
///    `f64` kernels can never take this arm.
/// 2. **`MLRS_RANSAC_ENGINE` override.** `"host"` forces the host engine,
///    `"device"` forces this one past the size floor (but not past gate 1,
///    which is not a preference). Read through [`crate::abflag`] so a test can
///    scope it without an environment data race.
/// 3. **Size floor** — [`RANSAC_DEVICE_MIN_WORK`].
pub fn ransac_device_applicable<F: Pod>(n: usize, d: usize) -> bool {
    if !ransac_device_possible::<F>() {
        return false;
    }
    match crate::abflag::var("MLRS_RANSAC_ENGINE").as_deref() {
        Some("host") => return false,
        Some("device") => return true,
        _ => {}
    }
    n.saturating_mul(d.max(1)) >= RANSAC_DEVICE_MIN_WORK
}

/// Resolve the `device` preference against the shape heuristic — `true` when the
/// device scan should carry this fit.
///
/// The [`Device`] precedence contract: an EXPLICIT `device` never consults the
/// abflag, `Auto` does ([`Device::prefers_device`]).
pub fn ransac_device_chosen<F: Pod>(n: usize, d: usize, device: Device) -> bool {
    ransac_device_possible::<F>() && device.prefers_device(|| ransac_device_applicable::<F>(n, d))
}

/// Trials to scan in one launch: the configured width, clamped to the trials
/// that actually remain so the last batch of a fit cannot speculate past the
/// loop's own bound.
///
/// `MLRS_RANSAC_BATCH` overrides the width for on-target A/B, read through
/// [`abflag`](crate::abflag) so a test can scope the override to its own thread
/// ([[mlrs-abflag-test-knobs]]).
pub fn ransac_batch_width(remaining: usize) -> usize {
    let want = crate::abflag::var("MLRS_RANSAC_BATCH")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(RANSAC_DEVICE_BATCH)
        .max(1);
    want.min(remaining.max(1))
}

/// One trial's device scan: [`TrialScan`] plus the inlier target sums the R²
/// denominator's mean is formed from.
///
/// `y_sum` rides along because the scan pass is already reading `y`; forming it
/// later would cost a whole extra `O(n·t)` launch before
/// [`RansacDevice::r2_den`] could run.
#[derive(Debug, Clone)]
pub struct DeviceTrialScan {
    /// The consensus size — [`TrialScan::n_inliers`].
    pub n_inliers: usize,
    /// `Σ_inliers (y_k − ŷ_k)²` per target — [`TrialScan::sq_err`].
    pub sq_err: Vec<f64>,
    /// `Σ_inliers y_k` per target.
    pub y_sum: Vec<f64>,
}

impl DeviceTrialScan {
    /// The host-shaped view of this scan, for the driver's shared bookkeeping.
    pub fn as_trial_scan(&self) -> TrialScan {
        TrialScan {
            n_inliers: self.n_inliers,
            sq_err: self.sq_err.clone(),
        }
    }
}

/// The device engine: owns the uploaded design and the resident mask for the
/// length of one fit.
///
/// Built ONCE per fit and driven batch after batch, the
/// [`GmmDevice`](super::gmm_device) shape — the upload is the expensive part and
/// it must not be repeated per trial.
pub struct RansacDevice<F> {
    x: DeviceArray<ActiveRuntime, F>,
    y: DeviceArray<ActiveRuntime, F>,
    /// `batch_max · n` inlier flags, allocated once and OVERWRITTEN by every
    /// batch. Never transferred wholesale — see [`Self::mask_of`].
    mask: Handle,
    mask_len: usize,
    n: usize,
    d: usize,
    t: usize,
    batch_max: usize,
    nblocks: usize,
    rows_per_block: usize,
}

impl<F> RansacDevice<F>
where
    F: Float + CubeElement + Pod,
{
    /// Upload an `n × d` row-major design and its `n × t` targets, and reserve
    /// the resident mask for batches of up to `batch_max` trials.
    pub fn new(
        pool: &mut BufferPool<ActiveRuntime>,
        x: &[F],
        y: &[F],
        n: usize,
        d: usize,
        t: usize,
        batch_max: usize,
    ) -> Result<Self, PrimError> {
        if x.len() != n * d {
            return Err(PrimError::ShapeMismatch {
                operand: "ransac_x",
                rows: n,
                cols: d,
                len: x.len(),
            });
        }
        if y.len() != n * t {
            return Err(PrimError::ShapeMismatch {
                operand: "ransac_y",
                rows: n,
                cols: t,
                len: y.len(),
            });
        }
        let batch_max = batch_max.max(1);
        let (nblocks, rows_per_block) = plan_blocks(n, batch_max);
        let mask_len = batch_max * n;
        let mask = pool.acquire(mask_len * size_of::<u32>());
        Ok(Self {
            x: DeviceArray::from_host(pool, x),
            y: DeviceArray::from_host(pool, y),
            mask,
            mask_len,
            n,
            d,
            t,
            batch_max,
            nblocks,
            rows_per_block,
        })
    }

    /// Trials the resident mask has room for.
    pub fn batch_max(&self) -> usize {
        self.batch_max
    }

    /// Scan the design against `batch` candidate models in ONE launch.
    ///
    /// `coef` is `batch` consecutive `t × d` row-major blocks and `icept`
    /// `batch` consecutive length-`t` blocks, both `f64` (the host solve's
    /// width), narrowed here to the design's — sklearn's `estimator.predict`
    /// runs in the design's dtype and the residual comparison must too.
    ///
    /// Leaves this batch's masks resident; [`mask_of`](Self::mask_of) and
    /// [`r2_den`](Self::r2_den) read them until the NEXT call overwrites them.
    pub fn scan_batch(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        coef: &[f64],
        icept: &[f64],
        batch: usize,
        loss: RansacLoss,
        threshold: f64,
    ) -> Result<Vec<DeviceTrialScan>, PrimError> {
        if batch == 0 || batch > self.batch_max {
            return Err(PrimError::ShapeMismatch {
                operand: "ransac_batch",
                rows: self.batch_max,
                cols: 1,
                len: batch,
            });
        }
        let (n, d, t) = (self.n, self.d, self.t);
        let stride = RANSAC_SCAN_BASE as usize + 2 * t;
        let part_len = batch * self.nblocks * stride;

        let coef_host: Vec<F> = coef.iter().map(|&v| f64_to_dev::<F>(v)).collect();
        let icept_host: Vec<F> = icept.iter().map(|&v| f64_to_dev::<F>(v)).collect();
        let coef_dev = DeviceArray::from_host(pool, &coef_host);
        let icept_dev = DeviceArray::from_host(pool, &icept_host);
        let part = pool.acquire(part_len * size_of::<F>());
        let client = pool.client().clone();

        {
            // SAFETY: every element count below is the one the matching
            // allocation reserved, and the kernel bounds-checks its unit id
            // (`tid < batch·nblocks`) and every row index against `n`.
            let x_arg = unsafe { ArrayArg::from_raw_parts(self.x.handle().clone(), n * d) };
            let y_arg = unsafe { ArrayArg::from_raw_parts(self.y.handle().clone(), n * t) };
            let c_arg =
                unsafe { ArrayArg::from_raw_parts(coef_dev.handle().clone(), coef_host.len()) };
            let b_arg =
                unsafe { ArrayArg::from_raw_parts(icept_dev.handle().clone(), icept_host.len()) };
            let m_arg = unsafe { ArrayArg::from_raw_parts(self.mask.clone(), self.mask_len) };
            let p_arg = unsafe { ArrayArg::from_raw_parts(part.clone(), part_len) };
            let (count, dim) =
                super::launch_dims_1d_folded(batch * self.nblocks, super::PERF_TUNED_BLOCK);
            ransac_scan_batch::launch::<F, ActiveRuntime>(
                &client,
                count,
                dim,
                x_arg,
                y_arg,
                c_arg,
                b_arg,
                m_arg,
                p_arg,
                n as u32,
                d as u32,
                t as u32,
                batch as u32,
                self.nblocks as u32,
                self.rows_per_block as u32,
                f64_to_dev::<F>(threshold),
                u32::from(loss == RansacLoss::SquaredError),
            );
        }
        coef_dev.release_into(pool);
        icept_dev.release_into(pool);

        // THE one synchronization of the batch.
        let part_dev = DeviceArray::<ActiveRuntime, F>::from_raw(part, part_len);
        let host = part_dev.to_host_metered(pool);
        part_dev.release_into(pool);

        // Folded in (trial, block) order — the same order the host arm folds, so
        // neither arm's answer depends on how its units were scheduled.
        Ok((0..batch)
            .map(|b| {
                let mut n_inliers = 0.0f64;
                let mut sq_err = vec![0.0f64; t];
                let mut y_sum = vec![0.0f64; t];
                for blk in 0..self.nblocks {
                    let slot = (b * self.nblocks + blk) * stride;
                    n_inliers += dev_to_f64(host[slot]);
                    for k in 0..t {
                        sq_err[k] += dev_to_f64(host[slot + RANSAC_SCAN_BASE as usize + k]);
                        y_sum[k] += dev_to_f64(host[slot + RANSAC_SCAN_BASE as usize + t + k]);
                    }
                }
                DeviceTrialScan {
                    // ROUNDED, not truncated: the count is an integer summed in
                    // `F`, which is exact inside `f32`'s 2²⁴ integer range but
                    // would truncate to `k − 1` on the first ulp of error past
                    // it. A consensus size is not a quantity to be off by one in.
                    n_inliers: n_inliers.round() as usize,
                    sq_err,
                    y_sum,
                }
            })
            .collect())
    }

    /// `Σ_{inliers of trial `b`} (y_k − ymean_k)²` per target — the R²
    /// DENOMINATOR, formed against the resident mask.
    ///
    /// Two-pass by construction: `ymean` comes from the previous launch's
    /// [`DeviceTrialScan::y_sum`], because the one-pass `Σy² − n·ȳ²` identity is
    /// a different sum in floating point and this quantity decides the consensus
    /// tie-break ([`ransac_host::RansacHostEngine::r2_on_mask`](super::ransac_host::RansacHostEngine::r2_on_mask)).
    ///
    /// Launched per TRIAL rather than per batch because only a trial that has
    /// already matched the incumbent's consensus size needs a score at all.
    pub fn r2_den(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        b: usize,
        ymean: &[f64],
    ) -> Result<Vec<f64>, PrimError> {
        let t = self.t;
        debug_assert_eq!(ymean.len(), t);
        let mean_host: Vec<F> = ymean.iter().map(|&v| f64_to_dev::<F>(v)).collect();
        let mean_dev = DeviceArray::from_host(pool, &mean_host);
        let part_len = self.nblocks * t;
        let part = pool.acquire(part_len * size_of::<F>());
        let client = pool.client().clone();
        {
            // SAFETY: as `scan_batch` — reserved lengths, and the kernel bounds
            // its unit id by `nblocks` and its row index by `n`.
            let y_arg = unsafe { ArrayArg::from_raw_parts(self.y.handle().clone(), self.n * t) };
            let m_arg = unsafe { ArrayArg::from_raw_parts(self.mask.clone(), self.mask_len) };
            let mean_arg = unsafe { ArrayArg::from_raw_parts(mean_dev.handle().clone(), t) };
            let p_arg = unsafe { ArrayArg::from_raw_parts(part.clone(), part_len) };
            let (count, dim) = super::launch_dims_1d_folded(self.nblocks, super::PERF_TUNED_BLOCK);
            ransac_den_block::launch::<F, ActiveRuntime>(
                &client,
                count,
                dim,
                y_arg,
                m_arg,
                mean_arg,
                p_arg,
                self.n as u32,
                t as u32,
                self.nblocks as u32,
                self.rows_per_block as u32,
                (b * self.n) as u32,
            );
        }
        mean_dev.release_into(pool);
        let part_dev = DeviceArray::<ActiveRuntime, F>::from_raw(part, part_len);
        let host = part_dev.to_host_metered(pool);
        part_dev.release_into(pool);

        let mut den = vec![0.0f64; t];
        for blk in 0..self.nblocks {
            for (k, acc) in den.iter_mut().enumerate() {
                *acc += dev_to_f64(host[blk * t + k]);
            }
        }
        Ok(den)
    }

    /// Read trial `b`'s inlier mask back to the host.
    ///
    /// The ONE `n`-sized transfer this engine ever makes, and it is made only
    /// when a trial actually becomes the incumbent — a handful of times in a
    /// fit, against the `max_trials` scans that never transfer anything.
    pub fn mask_of(&self, pool: &mut BufferPool<ActiveRuntime>, b: usize, out: &mut [bool]) {
        debug_assert_eq!(out.len(), self.n);
        // A CLONED handle, read and dropped: the buffer itself is released once,
        // by `release`, at the end of the fit.
        let dev = DeviceArray::<ActiveRuntime, u32>::from_raw(self.mask.clone(), self.mask_len);
        let host = dev.to_host_metered(pool);
        for (i, o) in out.iter_mut().enumerate() {
            *o = host[b * self.n + i] == 1;
        }
    }

    /// Return every buffer this engine holds to the pool.
    pub fn release(self, pool: &mut BufferPool<ActiveRuntime>) {
        self.x.release_into(pool);
        self.y.release_into(pool);
        pool.release(self.mask, self.mask_len * size_of::<u32>());
    }
}

/// Row blocking for a batch: enough units to fill the device, never so many that
/// the partial readback becomes the cost the batching removed.
fn plan_blocks(n: usize, batch: usize) -> (usize, usize) {
    let want_blocks = TARGET_UNITS.div_ceil(batch.max(1)).max(1);
    let rows_per_block = n.div_ceil(want_blocks).max(MIN_ROWS_PER_BLOCK);
    let nblocks = n.div_ceil(rows_per_block).max(1);
    (nblocks, rows_per_block)
}

/// Narrow an `f64` host quantity to the device's width through its byte view
/// (`F`'s own ops are CubeCL *kernel* ops, not host ones — the
/// `linear_predict_host` idiom).
#[inline]
fn f64_to_dev<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        other => unreachable!("ransac_device is f32/f64 only, got a {other}-byte element"),
    }
}

/// The inverse of [`f64_to_dev`].
#[inline]
fn dev_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        other => unreachable!("ransac_device is f32/f64 only, got a {other}-byte element"),
    }
}
