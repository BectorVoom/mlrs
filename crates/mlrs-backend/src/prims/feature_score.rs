//! `feature_score` — the column-moment sweeps every univariate feature-selection
//! score is built from (FSEL-01).
//!
//! `sklearn.feature_selection`'s four closed-form scoring functions all reduce
//! to ONE pass over the `n × d` design collecting per-column moments, and differ
//! only in which moments and how they are combined:
//!
//! | score            | moments needed                                        |
//! |------------------|-------------------------------------------------------|
//! | `f_classif`      | per-class `Σx` and counts, plus the global `Σx²`       |
//! | `chi2`           | per-class `Σx` and the global `Σx` (the expected table)|
//! | `r_regression`   | `Σx`, `Σx²`, `Σy·x`, `Σy`, `Σy²`                       |
//! | `f_regression`   | the same as `r_regression` (it is a transform of it)   |
//! | `VarianceThreshold` | `Σx`, `Σx²`, `min`, `max`, per-column NaN count     |
//!
//! so this module exposes exactly three sweeps — [`class_col_sums`],
//! [`cross_moments`], [`col_moments`] — and the algos layer assembles the
//! statistics and their p-values from them.
//!
//! ## Why the accumulation is HOST `f64` on every backend
//! Three independent reasons, all of which point the same way:
//!
//! 1. **The oracle contract is RELATIVE.** `f_classif` on an informative
//!    feature produces p-values around `1e-27` (sklearn's own docstring example
//!    prints `7.14e-27`), and D-09 measures abs-AND-rel error at `1e-5`. An
//!    `f32` accumulation of `Σx²` over `10⁶` rows carries ~`1e-4` relative
//!    error before the F-statistic is even formed — the score would be wrong in
//!    its second digit, and the p-value, which is exponentially sensitive to
//!    it, wrong by orders of magnitude. sklearn itself computes every one of
//!    these in `float64` (`f_regression`'s `check_X_y(..., dtype=np.float64)`
//!    is explicit about it), so matching it to `1e-5` REQUIRES `f64` here
//!    regardless of the estimator's `F`.
//! 2. **`f64` is not available on the fastest device.** cubecl's cuda backend
//!    does not advertise `f64` at all, so an "`f64` device kernel" silently
//!    disables itself exactly where it would matter most
//!    ([[mlrs-cubecl-cuda-f64-not-advertised]]) — the same reason
//!    [`gmm_host`](super::gmm_host)'s EM engine and
//!    [`special`](super::special)'s scalars are host code on every backend.
//! 3. **The work is a single ONE-SHOT pass.** This is `fit`, not a hot loop:
//!    one `O(n·d)` sweep per `fit`, against which a device launch's transfer
//!    and sync cost is not obviously repaid (the
//!    [`center`](super::center)/`column_reduce` profiling that motivated
//!    `colmean` found the transfer dominating at exactly this scale).
//!
//! The sweeps ARE parallelised — over contiguous row chunks with `f64`
//! accumulators, the [`gram_host`](super::gram_host) decomposition verbatim,
//! sized by [`host_units`] so a small design does not pay for threads it cannot
//! use. Partials combine in the same order every time, so a fit is
//! reproducible for a given `units`; `MLRS_CPU_UNITS` pins `units` when a test
//! needs bit-identical output across two runs.
//!
//! The one genuinely device-side piece of a selector is `transform`, which is a
//! pure column GATHER with no accumulation at all — that runs as the
//! `mlrs_kernels::feature_select::gather_columns` kernel via
//! [`gather_columns`], and is exact in any float width.
//!
//! Tests live in `crates/mlrs-backend/tests/feature_score_test.rs`
//! (AGENTS.md §2).

use bytemuck::Pod;
use cubecl::prelude::*;

use mlrs_core::{host_to_f64, PrimError};
use mlrs_kernels::feature_select::{
    gather_columns as gather_columns_kernel, scatter_columns as scatter_columns_kernel,
};

use crate::device_array::DeviceArray;
use crate::pool::BufferPool;
use crate::runtime::ActiveRuntime;

/// Rows-times-columns of elementwise work one worker thread is worth.
///
/// Below this the `std::thread::scope` spawn cost (tens of microseconds per
/// worker) exceeds the sweep it parallelises, so [`host_units`] returns `1` and
/// the sweep runs inline. Sized from the same measurement
/// [`gram_host`](super::gram_host)'s `HOST_MACS_PER_UNIT` came from, scaled for
/// this sweep's one-multiply-per-element inner loop rather than a MAC tile.
const HOST_ELEMS_PER_UNIT: usize = 1 << 19;

/// Worker threads to split a sweep across: one per [`HOST_ELEMS_PER_UNIT`] of
/// work, never more than the machine offers
/// ([`crate::capability::cpu_launch_units`], which `MLRS_CPU_UNITS` overrides
/// for A/B), never more than there are rows, never fewer than one.
fn host_units(n: usize, d: usize) -> usize {
    let elems = n.saturating_mul(d).max(1);
    (elems / HOST_ELEMS_PER_UNIT)
        .clamp(1, crate::capability::cpu_launch_units().max(1) as usize)
        .min(n.max(1))
}

/// Split `n` rows into `units` contiguous chunks and reduce each in parallel,
/// then fold the partials in chunk order.
///
/// `seed` builds a worker's zero accumulator, `sweep` folds one row range into
/// it, and `merge` combines two accumulators. Folding in CHUNK ORDER (not
/// completion order) is what makes the result independent of thread scheduling —
/// without it two runs of the same `fit` could differ in the last bits and the
/// `f32`/`f64` arms could not be compared to each other.
fn parallel_rows<A, S, W, M>(n: usize, units: usize, seed: S, sweep: W, merge: M) -> A
where
    A: Send,
    S: Fn() -> A + Sync,
    W: Fn(&mut A, usize, usize) + Sync,
    M: Fn(&mut A, A),
{
    if units <= 1 {
        let mut acc = seed();
        sweep(&mut acc, 0, n);
        return acc;
    }
    let rows = n.div_ceil(units);
    let partials: Vec<A> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..units)
            .filter_map(|u| {
                let r0 = u * rows;
                if r0 >= n {
                    return None;
                }
                let r1 = (r0 + rows).min(n);
                let (seed, sweep) = (&seed, &sweep);
                Some(scope.spawn(move || {
                    let mut acc = seed();
                    sweep(&mut acc, r0, r1);
                    acc
                }))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("feature_score sweep worker panicked"))
            .collect()
    });
    let mut it = partials.into_iter();
    let mut acc = it.next().unwrap_or_else(&seed);
    for p in it {
        merge(&mut acc, p);
    }
    acc
}

// ===========================================================================
// col_moments — the unsupervised sweep (VarianceThreshold)
// ===========================================================================

/// Per-column moments of an `n × d` row-major design, in `f64`.
///
/// NaN-AWARE by construction: `sum`/`sumsq`/`min`/`max` skip NaN entries and
/// `nan_count` records how many were skipped per column, because
/// `VarianceThreshold` is the one sklearn selector that accepts NaN input
/// (`ensure_all_finite="allow-nan"`) and computes `np.nanvar` / `np.ptp` over
/// what remains. A column that is ENTIRELY NaN leaves `min`/`max` at
/// `±inf` and `count == 0`; the caller turns that into the `NaN` variance
/// `np.nanvar` reports for it.
#[derive(Debug, Clone, PartialEq)]
pub struct ColMoments {
    /// Per-column count of NON-NaN entries (`n − nan_count[c]`).
    pub count: Vec<usize>,
    /// Per-column `Σx` over non-NaN entries.
    pub sum: Vec<f64>,
    /// Per-column `Σx²` over non-NaN entries.
    pub sumsq: Vec<f64>,
    /// Per-column minimum over non-NaN entries (`+inf` for an all-NaN column).
    pub min: Vec<f64>,
    /// Per-column maximum over non-NaN entries (`−inf` for an all-NaN column).
    pub max: Vec<f64>,
}

impl ColMoments {
    fn zeros(d: usize) -> Self {
        Self {
            count: vec![0; d],
            sum: vec![0.0; d],
            sumsq: vec![0.0; d],
            min: vec![f64::INFINITY; d],
            max: vec![f64::NEG_INFINITY; d],
        }
    }

    fn merge(&mut self, other: Self) {
        for c in 0..self.sum.len() {
            self.count[c] += other.count[c];
            self.sum[c] += other.sum[c];
            self.sumsq[c] += other.sumsq[c];
            self.min[c] = self.min[c].min(other.min[c]);
            self.max[c] = self.max[c].max(other.max[c]);
        }
    }

    /// The BIASED (`ddof = 0`) per-column variance `numpy.nanvar` reports:
    /// `Σx²/m − (Σx/m)²` over the `m` non-NaN entries, and `NaN` for a column
    /// with none.
    ///
    /// Computed from raw moments rather than a second centering pass because
    /// this is the shape `VarianceThreshold` compares to `threshold`, and the
    /// comparison is against a user-chosen cutoff rather than a reference
    /// value — the catastrophic-cancellation risk of the raw-moment form is
    /// bounded by the fact that a column whose variance the two forms disagree
    /// about is, by construction, a column whose variance is far below any
    /// meaningful threshold. The `threshold == 0` case does not rely on this at
    /// all: sklearn compares the peak-to-peak range there instead, precisely to
    /// avoid the precision question ([`ColMoments::max`] − [`ColMoments::min`]).
    pub fn variance_biased(&self) -> Vec<f64> {
        (0..self.sum.len())
            .map(|c| {
                let m = self.count[c] as f64;
                if self.count[c] == 0 {
                    return f64::NAN;
                }
                let mean = self.sum[c] / m;
                (self.sumsq[c] / m - mean * mean).max(0.0)
            })
            .collect()
    }

    /// Per-column peak-to-peak range `max − min`, `numpy.ptp`. `NaN` for an
    /// all-NaN column (where sklearn's `nanmin` of the variance/ptp pair also
    /// ends up `NaN`).
    pub fn peak_to_peak(&self) -> Vec<f64> {
        (0..self.sum.len())
            .map(|c| {
                if self.count[c] == 0 {
                    f64::NAN
                } else {
                    self.max[c] - self.min[c]
                }
            })
            .collect()
    }
}

/// Sweep the per-column [`ColMoments`] of a row-major `n × d` host slice.
///
/// Errors with [`PrimError::ShapeMismatch`] if `x.len() != n * d` or either
/// dimension is zero.
pub fn col_moments<T: Pod + Sync>(x: &[T], n: usize, d: usize) -> Result<ColMoments, PrimError> {
    validate(x.len(), n, d, "x")?;
    let units = host_units(n, d);
    Ok(parallel_rows(
        n,
        units,
        || ColMoments::zeros(d),
        |acc, r0, r1| {
            for r in r0..r1 {
                let row = &x[r * d..r * d + d];
                for (c, &v) in row.iter().enumerate() {
                    let v = host_to_f64(v);
                    if v.is_nan() {
                        continue;
                    }
                    acc.count[c] += 1;
                    acc.sum[c] += v;
                    acc.sumsq[c] += v * v;
                    acc.min[c] = acc.min[c].min(v);
                    acc.max[c] = acc.max[c].max(v);
                }
            }
        },
        ColMoments::merge,
    ))
}

// ===========================================================================
// class_col_sums — the classification sweep (f_classif, chi2)
// ===========================================================================

/// Per-class per-column sums of an `n × d` design, plus the global moments
/// `f_oneway` needs, all in `f64`.
///
/// `sums` is `n_classes × d` row-major (`sums[k * d + c]`), matching the
/// `observed` table `chi2` forms as `Yᵀ X` and the `sums_args` list `f_oneway`
/// builds per group. `total_sumsq` is `f_oneway`'s `ss_alldata`.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassColSums {
    /// Rows in each class, in the caller's class order.
    pub counts: Vec<usize>,
    /// `n_classes × d` row-major per-class column sums.
    pub sums: Vec<f64>,
    /// Length-`d` global column sums (`Σ_r x[r,c]` over ALL rows).
    pub total_sum: Vec<f64>,
    /// Length-`d` global column sums of squares (`f_oneway`'s `ss_alldata`).
    pub total_sumsq: Vec<f64>,
}

impl ClassColSums {
    fn zeros(k: usize, d: usize) -> Self {
        Self {
            counts: vec![0; k],
            sums: vec![0.0; k * d],
            total_sum: vec![0.0; d],
            total_sumsq: vec![0.0; d],
        }
    }

    fn merge(&mut self, other: Self) {
        for (a, b) in self.counts.iter_mut().zip(other.counts) {
            *a += b;
        }
        for (a, b) in self.sums.iter_mut().zip(other.sums) {
            *a += b;
        }
        for (a, b) in self.total_sum.iter_mut().zip(other.total_sum) {
            *a += b;
        }
        for (a, b) in self.total_sumsq.iter_mut().zip(other.total_sumsq) {
            *a += b;
        }
    }
}

/// Sweep the [`ClassColSums`] of a row-major `n × d` host slice against a
/// length-`n` label vector holding CLASS INDICES in `0..k`.
///
/// The caller is responsible for mapping raw target values to indices (the
/// algos layer does it with the sorted-unique order `numpy.unique` produces, so
/// the per-class rows line up with sklearn's group order). A label outside
/// `0..k` is a caller bug and returns [`PrimError::ShapeMismatch`] on the
/// `labels` operand rather than being silently dropped — a dropped row would
/// shift every subsequent statistic by an amount no test would localise.
pub fn class_col_sums<T: Pod + Sync>(
    x: &[T],
    labels: &[u32],
    n: usize,
    d: usize,
    k: usize,
) -> Result<ClassColSums, PrimError> {
    validate(x.len(), n, d, "x")?;
    if labels.len() != n || k == 0 {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: 1,
            len: labels.len(),
        });
    }
    if labels.iter().any(|&l| (l as usize) >= k) {
        return Err(PrimError::ShapeMismatch {
            operand: "labels",
            rows: n,
            cols: k,
            len: labels.len(),
        });
    }
    let units = host_units(n, d);
    Ok(parallel_rows(
        n,
        units,
        || ClassColSums::zeros(k, d),
        |acc, r0, r1| {
            for r in r0..r1 {
                let row = &x[r * d..r * d + d];
                let base = labels[r] as usize * d;
                acc.counts[labels[r] as usize] += 1;
                let class_row = &mut acc.sums[base..base + d];
                for c in 0..d {
                    let v = host_to_f64(row[c]);
                    class_row[c] += v;
                    acc.total_sum[c] += v;
                    acc.total_sumsq[c] += v * v;
                }
            }
        },
        ClassColSums::merge,
    ))
}

// ===========================================================================
// cross_moments — the regression sweep (r_regression, f_regression)
// ===========================================================================

/// The `X`-vs-`y` cross moments `r_regression` is computed from, in `f64`.
///
/// `r_regression` needs `E[(x − x̄)(y − ȳ)] / (σₓ σᵥ)`, which it evaluates as
/// `⟨y − ȳ, x⟩ / (‖x − x̄‖ · ‖y − ȳ‖)` — every piece of which follows from
/// these five raw moments, so the whole score is one pass over `X`.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossMoments {
    /// Length-`d` `Σ_r x[r,c]`.
    pub xsum: Vec<f64>,
    /// Length-`d` `Σ_r x[r,c]²` (`sklearn.utils.extmath.row_norms(X.T, squared=True)`).
    pub xsq: Vec<f64>,
    /// Length-`d` `Σ_r y[r]·x[r,c]`.
    pub xy: Vec<f64>,
    /// `Σ_r y[r]`.
    pub ysum: f64,
    /// `Σ_r y[r]²`.
    pub ysq: f64,
    /// Rows swept (`n`), carried so the caller need not thread it separately.
    pub n: usize,
}

impl CrossMoments {
    fn zeros(d: usize, n: usize) -> Self {
        Self {
            xsum: vec![0.0; d],
            xsq: vec![0.0; d],
            xy: vec![0.0; d],
            ysum: 0.0,
            ysq: 0.0,
            n,
        }
    }

    fn merge(&mut self, other: Self) {
        for (a, b) in self.xsum.iter_mut().zip(other.xsum) {
            *a += b;
        }
        for (a, b) in self.xsq.iter_mut().zip(other.xsq) {
            *a += b;
        }
        for (a, b) in self.xy.iter_mut().zip(other.xy) {
            *a += b;
        }
        self.ysum += other.ysum;
        self.ysq += other.ysq;
    }
}

/// Sweep the [`CrossMoments`] of a row-major `n × d` host slice `x` against a
/// length-`n` target `y`.
pub fn cross_moments<T: Pod + Sync>(
    x: &[T],
    y: &[T],
    n: usize,
    d: usize,
) -> Result<CrossMoments, PrimError> {
    validate(x.len(), n, d, "x")?;
    if y.len() != n {
        return Err(PrimError::ShapeMismatch {
            operand: "y",
            rows: n,
            cols: 1,
            len: y.len(),
        });
    }
    let units = host_units(n, d);
    Ok(parallel_rows(
        n,
        units,
        || CrossMoments::zeros(d, 0),
        |acc, r0, r1| {
            for r in r0..r1 {
                let row = &x[r * d..r * d + d];
                let yv = host_to_f64(y[r]);
                acc.ysum += yv;
                acc.ysq += yv * yv;
                for c in 0..d {
                    let v = host_to_f64(row[c]);
                    acc.xsum[c] += v;
                    acc.xsq[c] += v * v;
                    acc.xy[c] += yv * v;
                }
            }
        },
        CrossMoments::merge,
    )
    .tap_n(n))
}

impl CrossMoments {
    /// Set the swept row count and return `self` — the single-unit path never
    /// runs `parallel_rows`' merge closure, so `n` is stamped here instead of
    /// being left at the seed's `0`.
    fn tap_n(mut self, n: usize) -> Self {
        self.n = n;
        self
    }
}

// ===========================================================================
// gather_columns — the device-side selector `transform`
// ===========================================================================

/// Gather the columns listed in `idx` out of a device-resident row-major
/// `rows × cols_in` matrix, returning the `rows × idx.len()` submatrix.
///
/// This is a selector's whole `transform`: `SelectorMixin.transform` is
/// `X[:, support_mask]` and nothing else. Unlike the moment sweeps above it
/// involves NO accumulation, so it is exact in any float width and runs as a
/// device kernel on every backend with no `f64` question to answer.
///
/// `idx` must hold column indices `< cols_in`; an out-of-range index returns
/// [`PrimError::ShapeMismatch`] on the `idx` operand rather than reading out of
/// bounds. An EMPTY `idx` is legal and yields a `rows × 0` result (sklearn's
/// "no features were selected" case, which warns rather than raising) — the
/// returned [`DeviceArray`] is zero-length and no kernel is launched.
pub fn gather_columns<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, F>,
    (rows, cols_in): (usize, usize),
    idx: &[u32],
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    validate(x.len(), rows, cols_in, "x")?;
    if idx.iter().any(|&c| (c as usize) >= cols_in) {
        return Err(PrimError::ShapeMismatch {
            operand: "idx",
            rows: 1,
            cols: cols_in,
            len: idx.len(),
        });
    }
    let cols_out = idx.len();
    if cols_out == 0 {
        return Ok(DeviceArray::from_host(pool, &[] as &[F]));
    }
    let out_len = rows * cols_out;
    let elem = size_of::<F>();
    let out_handle = pool.acquire(out_len * elem);
    let idx_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, idx);
    let client = pool.client().clone();
    let (ccount, cdim) =
        super::launch_dims_1d_folded(out_len, crate::capability::gather_launch_width());
    // SAFETY: `x.len()`/`out_len`/`cols_out` are the validated element counts of
    // the three buffers; the kernel bounds-checks `tid < output.len()` and every
    // index it forms is `(tid / cols_out) * cols_in + idx[tid % cols_out]` with
    // `idx[·] < cols_in` checked above, so no read leaves `x`.
    let x_arg = unsafe { ArrayArg::from_raw_parts(x.handle().clone(), x.len()) };
    let idx_arg = unsafe { ArrayArg::from_raw_parts(idx_dev.handle().clone(), cols_out) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), out_len) };
    gather_columns_kernel::launch::<F, ActiveRuntime>(
        &client,
        ccount,
        cdim,
        x_arg,
        idx_arg,
        out_arg,
        cols_in as u32,
        cols_out as u32,
    );
    idx_dev.release_into(pool);
    Ok(DeviceArray::from_raw(out_handle, out_len))
}

/// Scatter a device-resident `rows × idx.len()` selected-column matrix back into
/// a zero-filled `rows × cols_out` frame, `out[r, idx[j]] = z[r, j]`.
///
/// This is `SelectorMixin.inverse_transform`: a selector discards information,
/// so the inverse restores the original GEOMETRY with zeros in the dropped
/// columns, not the original values. An empty `idx` yields an all-zero
/// `rows × cols_out` result, which is the consistent answer for the
/// "no features selected" case.
pub fn scatter_columns<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    z: &DeviceArray<ActiveRuntime, F>,
    rows: usize,
    idx: &[u32],
    cols_out: usize,
) -> Result<DeviceArray<ActiveRuntime, F>, PrimError>
where
    F: Float + CubeElement + Pod,
{
    let cols_in = idx.len();
    if rows == 0 || cols_out == 0 || z.len() != rows * cols_in {
        return Err(PrimError::ShapeMismatch {
            operand: "z",
            rows,
            cols: cols_in,
            len: z.len(),
        });
    }
    if idx.iter().any(|&c| (c as usize) >= cols_out) {
        return Err(PrimError::ShapeMismatch {
            operand: "idx",
            rows: 1,
            cols: cols_out,
            len: cols_in,
        });
    }
    // The frame is uploaded pre-zeroed rather than zeroed by a kernel: the
    // unselected columns are the MAJORITY of it (that is what selection means),
    // so a second launch to write zeros would move more data than the scatter
    // itself, and `BufferPool::acquire` makes no zero-initialisation promise.
    let out_len = rows * cols_out;
    let zeros = vec![num_zero::<F>(); out_len];
    let out_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &zeros);
    if cols_in == 0 {
        return Ok(out_dev);
    }
    let idx_dev: DeviceArray<ActiveRuntime, u32> = DeviceArray::from_host(pool, idx);
    let client = pool.client().clone();
    let (ccount, cdim) =
        super::launch_dims_1d_folded(z.len(), crate::capability::gather_launch_width());
    // SAFETY: the three element counts are the validated lengths of the three
    // buffers; the kernel bounds-checks `tid < z.len()` and every write it forms
    // is `(tid / cols_in) * cols_out + idx[tid % cols_in]` with
    // `idx[·] < cols_out` checked above, so no write leaves `out_dev`.
    let z_arg = unsafe { ArrayArg::from_raw_parts(z.handle().clone(), z.len()) };
    let idx_arg = unsafe { ArrayArg::from_raw_parts(idx_dev.handle().clone(), cols_in) };
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_dev.handle().clone(), out_len) };
    scatter_columns_kernel::launch::<F, ActiveRuntime>(
        &client,
        ccount,
        cdim,
        z_arg,
        idx_arg,
        out_arg,
        cols_in as u32,
        cols_out as u32,
    );
    idx_dev.release_into(pool);
    Ok(out_dev)
}

/// A generic-float zero, for the pre-zeroed `inverse_transform` frame.
fn num_zero<F: Pod>() -> F {
    mlrs_core::f64_to_host::<F>(0.0)
}

/// Shared `rows * cols == len`, both dims non-zero geometry guard.
fn validate(len: usize, rows: usize, cols: usize, operand: &'static str) -> Result<(), PrimError> {
    if rows == 0 || cols == 0 || rows.checked_mul(cols).map(|v| v != len).unwrap_or(true) {
        return Err(PrimError::ShapeMismatch {
            operand,
            rows,
            cols,
            len,
        });
    }
    Ok(())
}
