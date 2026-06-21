//! `KernelDensity` (KERNEL-02) — kernel density estimation matching
//! `sklearn.neighbors.KernelDensity` forced-exact (`atol=0, rtol=0`).
//!
//! ## Composes the v1 `distance` + a density-value map + a device log-sum-exp (D-08)
//! KernelDensity is a DISTINCT kernel family from the kernel-matrix dot-product
//! kernels: its six kernels are functions of the RAW euclidean distance with a
//! dimension-dependent normalization (D-08). It therefore composes the v1
//! [`distance`](mlrs_backend::prims::distance) prim DIRECTLY (NOT the
//! kernel-matrix prim)
//! + a per-element density-value map (the `mlrs-kernels` `kde_*_map` kernels) + a
//! per-query (row) log-sum-exp over the v1 [`row_reduce`](mlrs_backend::prims::reduce)
//! prim. The final assembly is
//! `log_density(q) = logsumexp_i[log_kernel(dist_i, h)] + log_norm(h, d, kernel) − log(N)`
//! (RESEARCH §"Density assembly"; VERIFIED from sklearn 1.9.0 `_kde.py`).
//!
//! ## Linear-domain log-sum-exp, never `±∞` (D-11 / Pitfall 3)
//! The per-element map computes the kernel VALUE (`exp(log_kernel)`), so the
//! compact-support kernels (tophat/epanechnikov/linear/cosine) yield EXACT `0`
//! out of support — the sum stays a sum of non-negative finites, never poisoned by
//! `−∞`/the infinity constant. The single `log` is applied ONCE at the very end
//! (host-side), after the device row-sum. This is the cpu-MLIR-safe form
//! ([[cubecl-cpu-no-shared-memory]] — the map is shared-memory-free).
//!
//! ## Squared vs raw distance per kernel (Pitfall 4)
//! gaussian/epanechnikov consume `distance(sqrt=false)` (squared `‖q − x‖²`);
//! tophat/exponential/linear/cosine compare the RAW `dist < h`, so they consume
//! `distance(sqrt=true)`.
//!
//! ## Host-side `log_norm` in f64 (A1)
//! The per-kernel log-normalization constant `log_norm(h, d, kernel)` depends only
//! on the bandwidth `h`, the feature dimension `d`, and the kernel — NOT on the
//! data — so it is a per-query CONSTANT computed ONCE on the host in `f64`
//! (`logVn`/`logSn`/`lgamma`), then added to the device-computed `logsumexp`. The
//! `lgamma` is a self-contained Lanczos approximation (matching the C `lgamma`
//! sklearn's Cython uses within the documented KD tolerance, A1) — `lgamma` is
//! NEVER attempted on device.
//!
//! ## Bandwidth resolution (D-09)
//! `bandwidth` is numeric (`> 0`) OR the `'scott'` / `'silverman'` host closed
//! forms (`n^(−1/(d+4))` / `(n·(d+2)/4)^(−1/(d+4))` — the SKLEARN forms, not
//! scipy's). Resolved at `fit` from `n_samples`/`n_features`; `bandwidth_ > 0` is
//! validated (`InvalidBandwidth`) before any launch.
//!
//! ## ScoreSamples (D-12), NOT Predict
//! KernelDensity implements [`ScoreSamples`](crate::traits::ScoreSamples) — a
//! length-`n` per-query log-density vector — NOT a regression `Predict` / a
//! neighbor surface (it lives in its own `density/` home, RESEARCH Open Q2).
//!
//! Tests live in `crates/mlrs-algos/tests/kernel_density_test.rs` (AGENTS.md §2),
//! never an in-source `#[cfg(test)] mod tests`.

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::prims::distance::distance;
use mlrs_backend::prims::reduce::{row_reduce, ReducePath, ScalarOp};
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::PrimError;
use mlrs_kernels::{
    kde_cosine_map, kde_epanechnikov_map, kde_exponential_map, kde_gaussian_map, kde_linear_map,
    kde_tophat_map,
};

use crate::error::AlgoError;
use crate::traits::ScoreSamples;

/// The six sklearn KernelDensity kernels (D-07). Selected at construction; the
/// resolved numeric `bandwidth_` is computed at `fit` (D-09).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdKernel {
    /// Gaussian `exp(−0.5·‖q−x‖²/h²)` — squared distance, no compact support.
    Gaussian,
    /// Tophat `1` if `dist < h` else `0` — raw distance, compact.
    Tophat,
    /// Epanechnikov `1 − ‖q−x‖²/h²` inside, `0` outside — squared distance, compact.
    Epanechnikov,
    /// Exponential `exp(−dist/h)` — raw distance, no compact support.
    Exponential,
    /// Linear `1 − dist/h` inside, `0` outside — raw distance, compact.
    Linear,
    /// Cosine `cos(0.5·π·dist/h)` inside, `0` outside — raw distance, compact.
    Cosine,
}

impl KdKernel {
    /// The sklearn kernel name (for the [`AlgoError::InvalidKernel`] diagnostic).
    fn name(self) -> &'static str {
        match self {
            KdKernel::Gaussian => "gaussian",
            KdKernel::Tophat => "tophat",
            KdKernel::Epanechnikov => "epanechnikov",
            KdKernel::Exponential => "exponential",
            KdKernel::Linear => "linear",
            KdKernel::Cosine => "cosine",
        }
    }

    /// Whether this kernel's density map consumes the SQUARED distance
    /// (`distance(sqrt=false)`). gaussian/epanechnikov use squared; the four
    /// raw-distance kernels use `distance(sqrt=true)` (Pitfall 4).
    fn uses_squared_distance(self) -> bool {
        matches!(self, KdKernel::Gaussian | KdKernel::Epanechnikov)
    }
}

/// The bandwidth specification (D-09): a numeric value (`> 0`) used as-is, or one
/// of the two host closed-form auto-bandwidth rules resolved at `fit` from
/// `n_samples`/`n_features` (the SKLEARN forms, not scipy's).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandwidthSpec {
    /// A fixed numeric bandwidth (`> 0`, validated at `fit`).
    Numeric(f64),
    /// `'scott'`: `bandwidth_ = n^(−1/(d+4))`.
    Scott,
    /// `'silverman'`: `bandwidth_ = (n·(d+2)/4)^(−1/(d+4))`.
    Silverman,
}

/// Kernel density estimation (KERNEL-02) over the v1 `distance` prim + a
/// density-value map + a device log-sum-exp (D-08/D-11).
///
/// Construct with [`KernelDensity::new`] (`kernel`, `bandwidth`), then
/// [`fit`](Self::fit) and [`score_samples`](crate::traits::ScoreSamples::score_samples).
/// The fitted training matrix `X_fit_` is device-resident; the resolved
/// `bandwidth_` is a host `f64` accessor.
pub struct KernelDensity<F>
where
    F: Float + CubeElement + Pod,
{
    /// Which density kernel to evaluate (D-07).
    kernel: KdKernel,
    /// The bandwidth specification (numeric or scott/silverman, D-09).
    bandwidth_spec: BandwidthSpec,
    /// The fitted training matrix `X_fit_` (`n_samples × n_features`),
    /// device-resident, `None` until `fit`.
    x_fit_: Option<DeviceArray<ActiveRuntime, F>>,
    /// The RESOLVED numeric bandwidth (`> 0`), `None` until `fit` (D-09).
    bandwidth_: Option<f64>,
    /// Fitted `(n_samples, n_features)` geometry, `None` until `fit`.
    fit_shape_: Option<(usize, usize)>,
}

impl<F> KernelDensity<F>
where
    F: Float + CubeElement + Pod,
{
    /// Create an unfitted `KernelDensity`. `kernel` selects the family (D-07);
    /// `bandwidth` is a numeric value or a string rule (D-09). Validation
    /// (`bandwidth_ > 0`, kernel name) happens at `fit`, not construction.
    pub fn new(kernel: KdKernel, bandwidth: BandwidthSpec) -> Self {
        Self {
            kernel,
            bandwidth_spec: bandwidth,
            x_fit_: None,
            bandwidth_: None,
            fit_shape_: None,
        }
    }

    /// The resolved numeric `bandwidth_` (`> 0`) after `fit`. Errors with
    /// [`AlgoError::NotFitted`] before `fit` (D-09).
    pub fn bandwidth(&self) -> Result<f64, AlgoError> {
        self.bandwidth_.ok_or(AlgoError::NotFitted {
            estimator: "kernel_density",
            operation: "bandwidth_",
        })
    }

    /// Fit the density model: store `X_fit_` and resolve `bandwidth_` (D-09).
    ///
    /// `x` is `(n_samples × n_features)` row-major. Validates the kernel name and
    /// geometry, resolves the bandwidth (numeric or scott/silverman host closed
    /// form), and validates `bandwidth_ > 0` (`InvalidBandwidth`) — all BEFORE any
    /// device launch (T-08-04-01 / ASVS V5). Returns `&mut Self` (sklearn
    /// convention).
    pub fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError> {
        let (n_samples, n_features) = shape;

        // --- T-08-04-01 / ASVS V5: validate the kernel name + geometry BEFORE any
        //     launch. KdKernel is a closed set, but the guard documents the
        //     validate-before-launch contract and surfaces InvalidKernel rather
        //     than fall through (mirrors kernel_ridge.rs). ---
        if !matches!(
            self.kernel,
            KdKernel::Gaussian
                | KdKernel::Tophat
                | KdKernel::Epanechnikov
                | KdKernel::Exponential
                | KdKernel::Linear
                | KdKernel::Cosine
        ) {
            return Err(AlgoError::InvalidKernel {
                estimator: "kernel_density",
                kernel: self.kernel.name().to_string(),
            });
        }
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }

        // --- Bandwidth resolution (D-09, host f64). scott/silverman are the
        //     SKLEARN closed forms (no per-feature std factor — NOT scipy's). ---
        let n = n_samples as f64;
        let d = n_features as f64;
        let bandwidth = match self.bandwidth_spec {
            BandwidthSpec::Numeric(b) => b,
            BandwidthSpec::Scott => n.powf(-1.0 / (d + 4.0)),
            BandwidthSpec::Silverman => (n * (d + 2.0) / 4.0).powf(-1.0 / (d + 4.0)),
        };
        // Validate-before-launch: a non-positive bandwidth makes the −d·log(h)
        // normalization undefined (T-08-04-01). Require FINITE as well —
        // `inf > 0.0` passes the positivity check but drives `−d·h.ln()` → −inf
        // and `exp(−0.5·sqdist/inf²) = exp(0) = 1` on device, producing a
        // finite-but-meaningless log-density instead of a typed rejection (WR-03).
        if !(bandwidth > 0.0 && bandwidth.is_finite()) {
            return Err(AlgoError::InvalidBandwidth {
                estimator: "kernel_density",
                bandwidth,
            });
        }

        // Store a fresh device copy of X_fit_ (the caller's `x` is borrowed).
        let x_host = x.to_host(pool);
        let x_fit: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_host);

        // --- Re-fit buffer reuse (WR-07): on a re-`fit` the prior X_fit_ device
        //     allocation must return to the pool free-list, not be dropped to the
        //     allocator. Release the old buffer (if any) BEFORE reassigning. ---
        if let Some(old) = self.x_fit_.take() {
            old.release_into(pool);
        }

        self.x_fit_ = Some(x_fit);
        self.bandwidth_ = Some(bandwidth);
        self.fit_shape_ = Some((n_samples, n_features));
        Ok(self)
    }
}

impl<F> ScoreSamples<F> for KernelDensity<F>
where
    F: Float + CubeElement + Pod,
{
    /// Compute the length-`n_query` log-density for each row of `q` (D-12), via
    /// `distance(Q, X_fit_, sqrt=per-kernel)` → per-element density-value map →
    /// per-query (row) log-sum-exp over the v1 `reduce` prim → host assembly
    /// `lse_row + log_norm − log(N)` (D-08/D-11). Errors with
    /// [`AlgoError::NotFitted`] before `fit`, or a geometry / feature-count
    /// mismatch.
    fn score_samples(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        q: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, n_features) = shape;

        let x_fit = self.x_fit_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "kernel_density",
            operation: "score_samples",
        })?;
        let bandwidth = self.bandwidth_.ok_or(AlgoError::NotFitted {
            estimator: "kernel_density",
            operation: "score_samples",
        })?;
        let (n_samples, fit_features) = self.fit_shape_.ok_or(AlgoError::NotFitted {
            estimator: "kernel_density",
            operation: "score_samples",
        })?;

        // --- T-08-04-01 / ASVS V5: geometry + fitted-n_features consistency. ---
        if n_query == 0 || n_features == 0 || q.len() != n_query * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "q",
                rows: n_query,
                cols: n_features,
                len: q.len(),
            }));
        }
        if n_features != fit_features {
            return Err(AlgoError::Prim(PrimError::DimMismatch {
                dim: "n_features",
                lhs: n_features,
                rhs: fit_features,
            }));
        }

        // --- 1. D = distance(Q, X_fit_) (m×n). sqrt=false for gaussian/epanechnikov
        //        (squared distance), sqrt=true for the four raw-distance kernels
        //        (Pitfall 4). D-08 — the v1 distance prim DIRECTLY, NOT
        //        the kernel-matrix prim. ---
        let sqrt = !self.kernel.uses_squared_distance();
        let dmat = distance::<F>(
            pool,
            q,
            (n_query, n_features),
            x_fit,
            (n_samples, fit_features),
            sqrt,
            None,
        )?;

        // --- 2. Per-element KD density-value map IN PLACE over the distance buffer
        //        (linear domain — exact 0 out of support, never ±∞, D-11). The map
        //        kernel is shared-memory-free; the m×n operand stays in global
        //        memory (T-08-04-03). input handle == output handle (the
        //        the in-place scale-map idiom). ---
        let n_elems = n_query * n_samples;
        let h = f64_to_host::<F>(bandwidth);
        launch_kde_map_in_place(pool, &dmat, n_elems, self.kernel, h);

        // --- 3. Per-query (row) log-sum-exp via the v1 reduce prim (D-11). Plain
        //        reduce-SUM in the linear domain: row_sum = Σ_j kernel_value. The
        //        Shared path is forced (cpu-portable; the plane path returns None on
        //        non-subgroup adapters). The reduce-max rescale (div_by_row) is NOT
        //        needed — the kernel values are O(1) bounded (K(0,h)=1), so the
        //        linear sum has no overflow/underflow over the v2 problem sizes
        //        (RESEARCH Open Q1: rescale not needed; the f32 band passes). ---
        let row_sum = row_reduce::<F>(
            pool,
            &dmat,
            n_query,
            n_samples,
            ScalarOp::Sum,
            ReducePath::Shared,
        )?
        .expect("shared path is never plane-gated to None");
        dmat.release_into(pool);

        // --- 4. Host assembly (the single log applied ONCE at the end, D-11):
        //        log_density = log(row_sum) + log_norm(h, d, kernel) − log(N).
        //        log_norm is the per-kernel host-side f64 scalar (A1 — f64 lgamma,
        //        NEVER device). N = n_training_samples (no sample weights). ---
        let log_norm = kde_log_norm(self.kernel, bandwidth, n_features);
        let log_n = (n_samples as f64).ln();
        let row_sum_host = row_sum.to_host(pool);
        row_sum.release_into(pool);
        let mut out_host: Vec<F> = vec![F::from_int(0i64); n_query];
        for r in 0..n_query {
            let s = host_to_f64(row_sum_host[r]);
            // s is a sum of non-negative kernel values; log(0) → −∞ is the correct
            // log-density for a query with zero density in its support (matches
            // sklearn). It is produced ONLY at this terminal host step, never inside
            // a device map (Pitfall 3), so it cannot poison a device sum.
            let log_density = s.ln() + log_norm - log_n;
            out_host[r] = f64_to_host::<F>(log_density);
        }
        Ok(DeviceArray::from_host(pool, &out_host))
    }
}

/// Launch the per-element KD density-value map IN PLACE over the distance buffer
/// `dmat` (input handle == output handle), the backend prim's
/// scale-in-place idiom. `n` is the element count (`n_query · n_samples`); each
/// `kde_*_map` kernel bounds-checks `tid < input.len()` (T-08-04-01) and is
/// shared-memory-free (the m×n operand stays in global memory, T-08-04-03).
fn launch_kde_map_in_place<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    dmat: &DeviceArray<ActiveRuntime, F>,
    n: usize,
    kernel: KdKernel,
    h: F,
) where
    F: Float + CubeElement + Pod,
{
    let client = pool.client().clone();
    let (count, dim) = launch_dims_1d(n);
    // SAFETY: `n` is the carried distance-prim output element count (n_query ·
    // n_samples, itself derived from the validated geometry); each KD map kernel
    // bounds-checks `tid < input.len()`. input and output are the SAME handle so
    // the map is applied in place over the reused distance buffer (no parallel
    // allocation — T-08-04-03).
    let in_arg = unsafe { ArrayArg::from_raw_parts(dmat.handle().clone(), n) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(dmat.handle().clone(), n) };
    match kernel {
        KdKernel::Gaussian => {
            kde_gaussian_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, h)
        }
        KdKernel::Tophat => {
            kde_tophat_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, h)
        }
        KdKernel::Epanechnikov => kde_epanechnikov_map::launch::<F, ActiveRuntime>(
            &client, count, dim, in_arg, out_arg, h,
        ),
        KdKernel::Exponential => kde_exponential_map::launch::<F, ActiveRuntime>(
            &client, count, dim, in_arg, out_arg, h,
        ),
        KdKernel::Linear => {
            kde_linear_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, h)
        }
        KdKernel::Cosine => {
            kde_cosine_map::launch::<F, ActiveRuntime>(&client, count, dim, in_arg, out_arg, h)
        }
    }
}

/// Standard ceiling-division 1D launch config for the in-place map pass (the
/// elementwise per-element launch idiom shared with the backend prims).
fn launch_dims_1d(n: usize) -> (CubeCount, CubeDim) {
    let block = 256usize;
    // Compute the cube count in `usize` and check the `u32` launch-grid cast
    // (WR-02): an unchecked `n as u32` silently wraps for `n > u32::MAX`,
    // under-provisioning threads so trailing elements are never mapped — a silent
    // wrong-result. The KDE problem sizes are small today, but the guard turns the
    // overflow into a loud panic instead.
    let cubes = u32::try_from((n + block - 1) / block)
        .expect("element count exceeds u32 launch-grid limit");
    (
        CubeCount::Static(cubes.max(1), 1, 1),
        CubeDim {
            x: block as u32,
            y: 1,
            z: 1,
        },
    )
}

/// The per-kernel log-normalization constant `log_norm(h, d, kernel) = −factor −
/// d·log(h)` (RESEARCH §"Per-kernel log-normalization constant" TABLE; VERIFIED
/// from sklearn 1.9.0 `_binary_tree.pxi.tp` lines 438-476). Host-side f64; the
/// `lgamma` is the self-contained Lanczos approximation below (A1 — NEVER device).
fn kde_log_norm(kernel: KdKernel, h: f64, d_features: usize) -> f64 {
    let d = d_features as f64;
    let two_pi = 2.0 * std::f64::consts::PI;
    // logVn(n) = 0.5·n·log(π) − lgamma(0.5·n + 1)   (log volume of the unit n-ball)
    let log_vn = |n: f64| 0.5 * n * std::f64::consts::PI.ln() - lgamma(0.5 * n + 1.0);
    // logSn(n) = log(2π) + logVn(n − 1)              (log surface area)
    let log_sn = |n: f64| two_pi.ln() + log_vn(n - 1.0);

    let factor = match kernel {
        KdKernel::Gaussian => 0.5 * d * two_pi.ln(),
        KdKernel::Tophat => log_vn(d),
        KdKernel::Epanechnikov => log_vn(d) + (2.0 / (d + 2.0)).ln(),
        KdKernel::Exponential => log_sn(d - 1.0) + lgamma(d),
        KdKernel::Linear => log_vn(d) - (d + 1.0).ln(),
        KdKernel::Cosine => {
            // Cosine series (chain-rule integration, _binary_tree.pxi.tp 466-473):
            //   factor = 0; tmp = 2/π
            //   for k in 1, 3, 5, …, ≤ d:  factor += tmp;
            //                              tmp *= −(d−k)·(d−k−1)·(2/π)²
            //   factor = log(factor) + logSn(d−1)
            let two_over_pi = 2.0 / std::f64::consts::PI;
            let mut series = 0.0;
            let mut tmp = two_over_pi;
            let mut k = 1.0;
            while k <= d {
                series += tmp;
                tmp *= -(d - k) * (d - k - 1.0) * two_over_pi * two_over_pi;
                k += 2.0;
            }
            series.ln() + log_sn(d - 1.0)
        }
    };
    -factor - d * h.ln()
}

/// Natural log of the gamma function in `f64` via the Lanczos approximation
/// (g = 7, 9 coefficients), valid for `x > 0`. Matches the C `lgamma` sklearn's
/// Cython uses within the documented KD tolerance (A1) — used ONLY host-side for
/// the per-kernel `log_norm`, NEVER on device.
fn lgamma(x: f64) -> f64 {
    // Lanczos g=7 coefficients (Numerical Recipes / standard reference set).
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1−x) = π / sin(πx).
        let pi = std::f64::consts::PI;
        (pi / (pi * x).sin()).ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Reinterpret an `F` (f32 / f64) as `f64` for host-side combine (mirrors
/// `ridge.rs` / `kernel_ridge.rs`).
fn host_to_f64<F: Pod>(v: F) -> f64 {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kernel_density is f32/f64 only"),
    }
}

/// Inverse of [`host_to_f64`]: build an `F` (f32 / f64) from an `f64`.
fn f64_to_host<F: Pod>(v: f64) -> F {
    match size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("kernel_density is f32/f64 only"),
    }
}
