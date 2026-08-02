//! `CategoricalNB` (NB-05) — Categorical Naive Bayes,
//! ≈ `sklearn.naive_bayes.CategoricalNB`.
//!
//! Wave-0 SCAFFOLD: struct + the [`MinCategories`] enum (D-04) +
//! [`CategoricalNBBuilder`] (D-02 sklearn defaults) + `build()`
//! (data-INDEPENDENT validation incl. the D-06 `force_alpha` clip+warn and the
//! per-entry `min_categories >= 0` check) are SHIPPED; the `Fit` impl carries a
//! REAL geometry guard but a `todo!()` compute body filled in Wave 1. Analog:
//! `multinomial_nb.rs` (discrete builder shape) + the `BandwidthSpec` enum
//! precedent from `density/kernel_density.rs`. SEPARATE struct (D-03).
//!
//! `feature_log_prob_` is a RAGGED `Vec<Vec<f64>>` (one matrix per feature,
//! variable category count — Pitfall 7), NOT a single tensor; the non-negative-
//! integer input validation and the predict-time category-index guard live at
//! `fit` / `predict` (data-DEPENDENT — [`AlgoError::InvalidCategoricalInput`]),
//! wired in Wave 1.
//!
//! ## Fit shape (CATNB-FIT-CPU)
//!
//! The fit is ENTIRELY host-side — there is no `#[cube]` kernel here, only a
//! validate pass and a tabulation pass over the design matrix. Both are
//! ROW-MAJOR and chunked over rows across a scoped worker pool
//! ([`CategoricalNB::fit_host`] documents why, and what the column-strided shape
//! it replaced cost); [`CategoricalNB::fit_from_host_slice`] is the entry point
//! the PyO3 bridge uses so the operands are never round-tripped through a
//! `DeviceArray` just to be read straight back.
//!
//! Tests live in `crates/mlrs-algos/tests/categorical_nb_test.rs` (AGENTS.md §2).

use std::marker::PhantomData;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use mlrs_core::{f64_to_host, host_to_f64, PrimError};

use crate::error::{AlgoError, BuildError};
use crate::naive_bayes::multinomial_nb::{
    decode_classes_host, resolve_class_log_prior, validate_discrete_alpha,
};
use crate::naive_bayes::nb_common::{argmax_decode, log_sum_exp_normalize, NB_LABEL_INT_TOL};
// Phase 16 (D-02 shape-B trait-swap): builder UNTOUCHED; `<F, S = Unfit>` state
// param + migration to the consuming-self `typestate` surface. fit/predict math
// BYTE-IDENTICAL (D-03).
use crate::typestate::{
    validate_geometry, Fit, Fitted, PredictLabels, PredictLogProba, PredictProba, Unfit,
};

/// The minimum-categories-per-feature specification (D-04), modeled on the
/// `BandwidthSpec` value-shaped-knob precedent. Captures sklearn's
/// scalar-vs-per-feature-vs-None `min_categories` polymorphism at the type level.
///
/// `PerFeature` carries a `Vec`, so this enum is `Clone` (NOT `Copy`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MinCategories {
    /// `None`: infer each feature's category count from the data at `fit`
    /// (`max + 1` per feature).
    Infer,
    /// A single scalar applied to EVERY feature (sklearn's `int` form): each
    /// feature's category count is at least this value.
    Uniform(usize),
    /// A per-feature vector (sklearn's array-like form): entry `j` is feature
    /// `j`'s minimum category count. Length-`== n_features` is a data-DEPENDENT
    /// check at `fit`.
    PerFeature(Vec<usize>),
}

/// Categorical Naive Bayes (NB-05). Construct via [`CategoricalNB::builder`],
/// then [`Fit::fit`] + (Wave-1) the predict surface. Fitted `feature_log_prob_`
/// is a ragged host `Vec<Vec<f64>>` (one matrix per feature); `class_log_prior_`
/// is host f64 (D-03), `None` until `fit`.
pub struct CategoricalNB<F, S = Unfit> {
    /// Additive smoothing (D-02 default `1.0`).
    alpha: f64,
    /// Keep `alpha` as-is when `< 1e-10` (D-02 default `true`); else clip (D-06).
    /// Retained as fitted-config provenance (exposed via [`CategoricalNB::force_alpha`]);
    /// the clip already applied at `build()` (WR-08).
    force_alpha: bool,
    /// Learn class priors from the data (D-02 default `true`).
    fit_prior: bool,
    /// User-supplied class priors, or `None` → empirical (D-02 default `None`).
    class_prior: Option<Vec<f64>>,
    /// Minimum categories per feature (D-02 default `MinCategories::Infer`,
    /// i.e. sklearn `min_categories=None`).
    min_categories: MinCategories,
    /// DISTINCT sorted class labels inferred at `fit`.
    classes_: Vec<i64>,
    /// Feature count inferred at `fit`.
    n_features: usize,
    /// Per-feature category counts learned at `fit` (length `n_features`), `None`
    /// until `fit`.
    n_categories_: Option<Vec<usize>>,
    /// Ragged fitted `feature_log_prob_`: `feature_log_prob_[j]` is the
    /// `n_classes × n_categories_[j]` log-probability matrix for feature `j`
    /// (Pitfall 7). `None` until `fit`.
    feature_log_prob_: Option<Vec<Vec<f64>>>,
    /// Per-class log-prior (host f64), `None` until `fit`.
    class_log_prior_: Option<Vec<f64>>,
    /// Per-class sample counts (host f64, length `n_classes`), `None` until
    /// `fit`. The empirical-prior numerator AND the per-feature smoothing
    /// denominator `class_count[c] + alpha·n_categories_j` — kept so the
    /// predict-time unseen-category fallback computes the EXACT smoothed
    /// `log(alpha / denom_cj)` (T-11-04-02) without reconstructing it from the
    /// fitted table.
    class_count_: Option<Vec<f64>>,
    /// Marker to retain the `F` type parameter (the device buffers land in Wave-1).
    _marker: std::marker::PhantomData<F>,
    /// Compile-time lifecycle marker (zero-sized).
    _state: PhantomData<S>,
}

impl<F> CategoricalNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Start building a `CategoricalNB` with sklearn's defaults (D-02).
    pub fn builder() -> CategoricalNBBuilder {
        CategoricalNBBuilder::default()
    }
}

impl<F> CategoricalNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// The inferred class labels (empty until `fit`).
    pub fn classes(&self) -> &[i64] {
        &self.classes_
    }

    /// The stored `force_alpha` config provenance (WR-08). The D-06 alpha clip is
    /// already applied at `build()`; this exposes whether the clip was suppressed.
    pub fn force_alpha(&self) -> bool {
        self.force_alpha
    }

    /// The per-class log-prior (`None` until `fit`).
    pub fn class_log_prior(&self) -> Option<&[f64]> {
        self.class_log_prior_.as_deref()
    }

    /// The per-class sample counts (`None` until `fit`).
    pub fn class_count(&self) -> Option<&[f64]> {
        self.class_count_.as_deref()
    }

    /// The per-feature category counts `n_categories_` (length `n_features`,
    /// `None` until `fit`). Entry `j` is the padded `n_categories_j =
    /// max(observed_max+1, min_categories_j)`.
    pub fn n_categories(&self) -> Option<&[usize]> {
        self.n_categories_.as_deref()
    }

    /// The ragged fitted `feature_log_prob_` (`feature_log_prob_[j]` is the
    /// `n_classes × n_categories_[j]` row-major log-prob matrix for feature `j`),
    /// `None` until `fit`.
    pub fn feature_log_prob(&self) -> Option<&[Vec<f64>]> {
        self.feature_log_prob_.as_deref()
    }
}

/// Builder for [`CategoricalNB`] (D-01). Defaults (D-02): `alpha=1.0`,
/// `force_alpha=true`, `fit_prior=true`, `class_prior=None`,
/// `min_categories=Infer`. Setter names mirror sklearn (D-09).
#[derive(Debug, Clone)]
pub struct CategoricalNBBuilder {
    alpha: f64,
    force_alpha: bool,
    fit_prior: bool,
    class_prior: Option<Vec<f64>>,
    min_categories: MinCategories,
}

impl Default for CategoricalNBBuilder {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            force_alpha: true,
            fit_prior: true,
            class_prior: None,
            min_categories: MinCategories::Infer,
        }
    }
}

impl CategoricalNBBuilder {
    /// Set the additive smoothing `alpha`.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }
    /// Set whether to keep a tiny `alpha` as-is (else clip to `1e-10`, D-06).
    pub fn force_alpha(mut self, force_alpha: bool) -> Self {
        self.force_alpha = force_alpha;
        self
    }
    /// Set whether to learn class priors from the data.
    pub fn fit_prior(mut self, fit_prior: bool) -> Self {
        self.fit_prior = fit_prior;
        self
    }
    /// Set explicit class priors (`None` → empirical / uniform).
    pub fn class_prior(mut self, class_prior: Option<Vec<f64>>) -> Self {
        self.class_prior = class_prior;
        self
    }
    /// Set the minimum-categories-per-feature specification (D-04).
    pub fn min_categories(mut self, min_categories: MinCategories) -> Self {
        self.min_categories = min_categories;
        self
    }

    /// Build the estimator, validating the data-INDEPENDENT hyperparameters at
    /// `build()` (D-05): `alpha >= 0`, finite+non-negative `class_prior`, the D-06
    /// `force_alpha` clip+warn (shared [`validate_discrete_alpha`]). Since
    /// `MinCategories` carries `usize` entries they are non-negative by
    /// construction; the per-feature LENGTH-`== n_features` check is data-DEPENDENT
    /// and stays at `fit`. The [`BuildError::InvalidMinCategories`] variant exists
    /// for any future signed-input path (kept for the typed surface).
    pub fn build<F>(self) -> Result<CategoricalNB<F, Unfit>, BuildError>
    where
        F: Float + CubeElement + Pod,
    {
        let alpha = validate_discrete_alpha(
            "categorical_nb",
            self.alpha,
            self.force_alpha,
            self.class_prior.as_deref(),
        )?;
        Ok(CategoricalNB {
            alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            min_categories: self.min_categories,
            classes_: Vec::new(),
            n_features: 0,
            n_categories_: None,
            feature_log_prob_: None,
            class_log_prior_: None,
            class_count_: None,
            _marker: std::marker::PhantomData,
            _state: PhantomData,
        })
    }
}

/// The largest category index the fit can encode. The pass-1 scan narrows each
/// validated feature value to a `u32` (halving pass-2's index arithmetic and
/// keeping the flat count table cache-resident), so a value beyond this is
/// rejected with [`AlgoError::InvalidCategoricalInput`] rather than silently
/// saturating the cast. `u32::MAX` categories in ONE feature would need a
/// ≥ 16 GiB count table anyway.
const MAX_CATEGORY: f64 = u32::MAX as f64;

/// Below this many `n_samples · n_features` elements the fit stays
/// single-threaded: spawning a scoped worker costs ~30 µs, which dwarfs a scan
/// over a few tens of thousands of elements.
const PAR_MIN_ELEMS: usize = 1 << 15;

/// Per-worker flat count-table budget, in `u32` entries (4 MiB). The tabulation
/// replicates an `n_classes · Σ_j n_categories_j` table PER worker, so a fit
/// with a huge category cross-product drops back to one table (serial) rather
/// than allocating a copy per core.
///
/// Memory footprint, for the record: ONE flat table is `n_classes · Σ_j
/// n_categories_j` `u32`s — exactly half the size of the `f64`
/// `feature_log_prob_` the fit must return anyway, so the un-replicated table can
/// never dominate the estimator it builds. The previous body's per-feature tables
/// peaked lower (one `n_classes × n_categories_j` table at a time) but paid the
/// `O(n · d²)` traffic for it; this cap is what keeps the replicated case bounded
/// (at most `PAR_MAX_WORKERS · 4 MiB` of scratch) rather than scaling with cores.
const PAR_TABLE_MAX_ENTRIES: usize = 1 << 20;

/// Ceiling on the worker count. BOTH fit passes stream the design matrix, so
/// they are DRAM-bandwidth-bound, not core-bound: the wall clock stops improving
/// long before the cores run out, while CPU time keeps climbing linearly with
/// every worker added. Measured on a 16-core box, `100 000 × 128` fit
/// (wall / cpu, min of 5):
///
/// | workers | 1     | 2     | 4     | 8     | 16    |
/// |---------|-------|-------|-------|-------|-------|
/// | wall ms | 67.8  | 53.3  | 41.6  | 36.0  | 35.2  |
/// | cpu ms  | 67.4  | 77.5  | 74.5  | 88.0  | 132.4 |
///
/// 8 is the knee: it takes the last real wall-clock gain (13 % over 4), and
/// doubling again buys 2 % for half as much CPU again. Spending a whole machine
/// to shave 2 % off one fit is the wrong trade in a library, so the pool is
/// capped here and the rest of the box is left for the caller's other work.
/// Override with `MLRS_CATNB_WORKERS` to re-measure on new hardware.
const PAR_MAX_WORKERS: usize = 8;

/// Worker count for a row-chunked host pass over `n_elems` elements: `1` below
/// [`PAR_MIN_ELEMS`], else the machine's parallelism capped at
/// [`PAR_MAX_WORKERS`].
///
/// `MLRS_CATNB_WORKERS=<n>` forces the count (read through
/// [`mlrs_backend::abflag`], so a test can scope the override to its own thread
/// rather than racing `environ`). `MLRS_CATNB_WORKERS=1` pins the fully serial
/// arm, which is what makes a serial-vs-parallel agreement test possible.
fn host_workers(n_elems: usize) -> usize {
    if let Some(forced) = mlrs_backend::abflag::var("MLRS_CATNB_WORKERS")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v >= 1)
    {
        return forced;
    }
    if n_elems < PAR_MIN_ELEMS {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(1)
        .clamp(1, PAR_MAX_WORKERS)
}

/// Row-chunk size (in ROWS) for a `workers`-way split of `n_samples`, capped at
/// `u32::MAX` rows so a per-chunk `u32` count can never overflow (a count is
/// bounded by its chunk's row count).
fn chunk_rows(n_samples: usize, workers: usize) -> usize {
    n_samples
        .div_ceil(workers.max(1))
        .max(1)
        .min(u32::MAX as usize)
}

/// Pass 1 over ONE row-chunk: validate the categorical encoding and learn each
/// feature's observed max, in a single ROW-MAJOR sweep.
///
/// Returns `(per_feature_max, first_invalid)`, where `first_invalid` is the flat
/// index (in the WHOLE matrix — `flat_base` is the chunk's offset) and value of
/// the first element that is not a non-negative integer within
/// [`NB_LABEL_INT_TOL`] / is beyond [`MAX_CATEGORY`]. Reporting the flat index
/// lets the parallel driver pick the FIRST offender in row-major order, so the
/// error message does not depend on the worker count.
///
/// The integer test is written as `!(diff <= tol)` rather than `diff > tol` so a
/// `NaN` — for which EVERY comparison is false — is REJECTED rather than silently
/// rounding to category `0`. That is what lets the PyO3 fit arm relocate
/// `check_array`'s finite scan into this pass (`ensure_all_finite=False`):
/// `+inf`/`-inf` already fail the `> MAX_CATEGORY` / `< 0.0` arms.
fn scan_chunk<F>(chunk: &[F], n_features: usize, flat_base: usize) -> (Vec<u32>, Option<(usize, f64)>)
where
    F: Float + CubeElement + Pod,
{
    let mut fmax = vec![0u32; n_features];
    for (r, row) in chunk.chunks_exact(n_features).enumerate() {
        for (j, (&xv, m)) in row.iter().zip(fmax.iter_mut()).enumerate() {
            let xf = host_to_f64(xv);
            let xr = xf.round();
            if !((xr - xf).abs() <= NB_LABEL_INT_TOL) || xr < 0.0 || xr > MAX_CATEGORY {
                return (fmax, Some((flat_base + r * n_features + j, xf)));
            }
            let k = xr as u32;
            if k > *m {
                *m = k;
            }
        }
    }
    (fmax, None)
}

/// Pass 2 over ONE row-chunk: tabulate `(class, feature, category)` counts into
/// the FLAT table `table[c · total + off[j] + k]`, where `off[j]` is feature
/// `j`'s base in the ragged category axis and `total = Σ_j n_categories_j`.
///
/// One flat table replaces the per-feature tables the previous body built, so
/// the whole tabulation is ONE row-major sweep instead of `n_features` column-
/// strided ones. Every index is in range by construction (`k ≤ observed_max_j <
/// n_categories_j` from pass 1), and a `u32` count cannot overflow because
/// [`chunk_rows`] caps a chunk at `u32::MAX` rows.
fn count_chunk<F>(
    chunk: &[F],
    class_of_row: &[usize],
    n_features: usize,
    off: &[usize],
    total: usize,
    table: &mut [u32],
) where
    F: Float + CubeElement + Pod,
{
    for (row, &c) in chunk.chunks_exact(n_features).zip(class_of_row.iter()) {
        let base = c * total;
        for (&xv, &o) in row.iter().zip(off.iter()) {
            let k = host_to_f64(xv).round() as u32 as usize;
            table[base + o + k] += 1;
        }
    }
}

impl<F> CategoricalNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    /// Fit directly from HOST slices — the no-upload twin of [`Fit::fit`].
    ///
    /// `CategoricalNB` is a pure host estimator: every element of `x` and `y` is
    /// read on the host (the categorical encoding is validated, tabulated, and
    /// turned into ragged host-f64 tables — there is no device kernel on this
    /// path). Routing the operands through a `DeviceArray` therefore bought a
    /// round trip and nothing else: `from_host` copied `n·d` floats into a pool
    /// buffer and `to_host` copied them straight back out, ~2 × 8 `n·d` bytes of
    /// pure overhead. The PyO3 bridge hands the Arrow values here instead, so a
    /// fit touches the caller's buffer once.
    ///
    /// `shape` is `(n_samples, n_features)` and `x` is row-major, exactly as for
    /// [`Fit::fit`]; the geometry guard is the slice twin of `validate_geometry`.
    pub fn fit_from_host_slice(
        self,
        x: &[F],
        y: &[F],
        shape: (usize, usize),
    ) -> Result<CategoricalNB<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        if n_samples == 0 || n_features == 0 || x.len() != n_samples * n_features {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "x",
                rows: n_samples,
                cols: n_features,
                len: x.len(),
            }));
        }
        if y.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y.len(),
            }));
        }
        self.fit_host(x, y, shape)
    }

    /// The shared host fit body behind [`Fit::fit`] and
    /// [`CategoricalNB::fit_from_host_slice`] — both geometry guards run in the
    /// caller, so this is the math only.
    ///
    /// ## Why two row-major passes (PERF)
    ///
    /// The previous body materialized an `n·d` `Vec<usize>` category matrix, then
    /// ran `n_features` COLUMN-strided passes over it to find each feature's
    /// observed max, then `n_features` MORE column-strided passes to fill one
    /// count table per feature. Each of those passes touches a distinct cache
    /// line per row over a working set far larger than L3, so the fit moved
    /// `O(n · d²)` bytes: at `d = 512` a `20 000 × 512` fit cost as much as a
    /// `100 000 × 64` one with 8× fewer elements, and mlrs' margin over sklearn
    /// collapsed from ~2× to ~1× as `d` grew.
    ///
    /// Now: pass 1 ([`scan_chunk`]) validates and learns all `d` maxes in ONE
    /// row-major sweep; pass 2 ([`count_chunk`]) tabulates all `d` features into
    /// ONE flat offset-indexed table in a second row-major sweep. Traffic is
    /// `O(n · d)`, and the intermediate category matrix is gone entirely (pass 2
    /// re-derives each index from the already-validated float — an ALU op against
    /// what would be another `4 n d` bytes written and read back). Both passes
    /// are chunked over rows across a scoped worker pool; the reductions
    /// (elementwise max, table sum) are `O(d)` and `O(n_classes · Σ n_cat_j)`.
    fn fit_host(
        self,
        x_host: &[F],
        y_host: &[F],
        shape: (usize, usize),
    ) -> Result<CategoricalNB<F, Fitted>, AlgoError> {
        let (n_samples, n_features) = shape;
        // WR-08: `force_alpha` is fitted-config provenance (D-06 clip already
        // applied at build()); it is now exposed via the `force_alpha()` accessor
        // so the field is genuinely used (the prior `let _ = self.force_alpha;`
        // suppression is removed).

        // --- PASS 1 (T-11-04-01): validate X is a non-negative-INTEGER
        //     categorical encoding AND learn each feature's observed max, in one
        //     row-major sweep, BEFORE any table is sized (a negative /
        //     non-integer value would later index a ragged table out of bounds).
        //     Round-to-nearest within 1e-6 to tolerate the f32/f64 round-trip of
        //     integer-encoded categories. ---
        let n_elems = n_samples * n_features;
        let workers = host_workers(n_elems);
        let rows_per = chunk_rows(n_samples, workers);
        let elems_per = rows_per * n_features;

        let scans: Vec<(Vec<u32>, Option<(usize, f64)>)> = if workers == 1 {
            vec![scan_chunk::<F>(x_host, n_features, 0)]
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = x_host
                    .chunks(elems_per)
                    .enumerate()
                    .map(|(ci, chunk)| {
                        scope.spawn(move || scan_chunk::<F>(chunk, n_features, ci * elems_per))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("categorical_nb: pass-1 worker panicked"))
                    .collect()
            })
        };
        // The FIRST offender in row-major order — independent of worker count.
        if let Some((_, xf)) = scans
            .iter()
            .filter_map(|(_, e)| *e)
            .min_by_key(|(idx, _)| *idx)
        {
            return Err(AlgoError::InvalidCategoricalInput {
                estimator: "categorical_nb",
                reason: format!("feature values must be non-negative integers (got {xf})"),
            });
        }
        let mut observed_max = vec![0u32; n_features];
        for (fmax, _) in &scans {
            for (m, &v) in observed_max.iter_mut().zip(fmax.iter()) {
                if v > *m {
                    *m = v;
                }
            }
        }

        // --- classes_ / dense per-row class index / n_classes via the shared
        //     discrete decode (integer + i32-range label guard, WR-02). ---
        let (classes_, class_of_row, n_classes) =
            decode_classes_host::<F>("categorical_nb", y_host)?;

        // class_count_[c] = #rows of class c (every observed class has >= 1).
        let mut class_count_: Vec<f64> = vec![0.0; n_classes];
        for &c in &class_of_row {
            class_count_[c] += 1.0;
        }

        // --- Per-feature observed_max + the MinCategories padding (D-04, Pitfall 7):
        //     n_categories_j = max(observed_max+1, min_categories_j). The
        //     PerFeature length-`== n_features` check is data-DEPENDENT (D-05). ---
        if let MinCategories::PerFeature(v) = &self.min_categories {
            if v.len() != n_features {
                return Err(AlgoError::InvalidCategoricalInput {
                    estimator: "categorical_nb",
                    reason: format!(
                        "min_categories (per-feature) length {} != n_features {n_features}",
                        v.len()
                    ),
                });
            }
        }
        let mut n_categories_: Vec<usize> = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let base = observed_max[j] as usize + 1;
            let min_j = match &self.min_categories {
                MinCategories::Infer => 0,
                MinCategories::Uniform(u) => *u,
                MinCategories::PerFeature(v) => v[j],
            };
            n_categories_.push(base.max(min_j));
        }

        // --- PASS 2: host-tabulate category_count_[j][c, k] (one owner per
        //     (feature, class, category) — a host count, NEVER a device scatter)
        //     into ONE flat table indexed `c * total + off[j] + k`. ---
        let mut off: Vec<usize> = Vec::with_capacity(n_features);
        let mut total = 0usize;
        for &n_cat_j in &n_categories_ {
            off.push(total);
            total += n_cat_j;
        }
        let table_len = n_classes * total;
        // Replicating the table per worker is what makes the tabulation
        // lock-free; drop to one table (serial) when that replication would cost
        // more than the scan it accelerates.
        let count_workers = if table_len > PAR_TABLE_MAX_ENTRIES {
            1
        } else {
            workers
        };
        let count_rows_per = chunk_rows(n_samples, count_workers);
        let count_elems_per = count_rows_per * n_features;

        let mut counts: Vec<f64> = vec![0.0; table_len];
        {
            let mut accumulate = |table: &[u32]| {
                for (acc, &v) in counts.iter_mut().zip(table.iter()) {
                    *acc += v as f64;
                }
            };
            // The `n_samples <= u32::MAX` arm is not about parallelism: a single
            // table over MORE rows than that could overflow its `u32` counters,
            // so such a fit takes the chunked branch (whose chunks are capped at
            // `u32::MAX` rows by `chunk_rows`) even at one worker.
            if count_workers == 1 && n_samples <= u32::MAX as usize {
                let mut table = vec![0u32; table_len];
                count_chunk::<F>(x_host, &class_of_row, n_features, &off, total, &mut table);
                accumulate(&table);
            } else {
                let tables: Vec<Vec<u32>> = std::thread::scope(|scope| {
                    let handles: Vec<_> = x_host
                        .chunks(count_elems_per)
                        .zip(class_of_row.chunks(count_rows_per))
                        .map(|(chunk, cls)| {
                            let off = &off;
                            scope.spawn(move || {
                                let mut table = vec![0u32; table_len];
                                count_chunk::<F>(
                                    chunk, cls, n_features, off, total, &mut table,
                                );
                                table
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().expect("categorical_nb: pass-2 worker panicked"))
                        .collect()
                });
                for table in &tables {
                    accumulate(table);
                }
            }
        }

        // --- feature_log_prob_[j][c, k] = log((count + alpha) /
        //       (class_count[c] + alpha · n_categories_j))  (Pitfall 4 — the
        //     denominator smoothing is alpha · n_categories_j). The ragged
        //     per-feature matrices are sliced back out of the flat table. ---
        let alpha = self.alpha;
        let mut feature_log_prob_: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let n_cat_j = n_categories_[j];
            let o = off[j];
            let mut flp = vec![0.0f64; n_classes * n_cat_j];
            for c in 0..n_classes {
                let denom = class_count_[c] + alpha * n_cat_j as f64;
                let src = &counts[c * total + o..c * total + o + n_cat_j];
                let dst = &mut flp[c * n_cat_j..(c + 1) * n_cat_j];
                for (d, &count) in dst.iter_mut().zip(src.iter()) {
                    *d = ((count + alpha) / denom).ln();
                }
            }
            feature_log_prob_.push(flp);
        }

        // --- class_log_prior_: supplied class_prior (length == n_classes) takes
        //     precedence; else empirical log(count_c/n) when fit_prior=true; else
        //     uniform (the shared discrete resolver, sklearn semantics). ---
        let class_log_prior_ = resolve_class_log_prior(
            "categorical_nb",
            self.fit_prior,
            &self.class_prior,
            &class_count_,
            n_classes,
        )?;

        // The ragged fitted tables are host f64 (CategoricalNB holds NO device
        // buffer), so the consuming-self transition just moves the host state into
        // the Fitted value; buffer reuse across re-CONSTRUCT+fit is a non-issue.
        Ok(CategoricalNB {
            alpha: self.alpha,
            force_alpha: self.force_alpha,
            fit_prior: self.fit_prior,
            class_prior: self.class_prior,
            min_categories: self.min_categories,
            classes_,
            n_features,
            n_categories_: Some(n_categories_),
            feature_log_prob_: Some(feature_log_prob_),
            class_log_prior_: Some(class_log_prior_),
            class_count_: Some(class_count_),
            _marker: std::marker::PhantomData,
            _state: PhantomData,
        })
    }
}

impl<F> Fit<F> for CategoricalNB<F, Unfit>
where
    F: Float + CubeElement + Pod,
{
    type Fitted = CategoricalNB<F, Fitted>;

    fn fit(
        self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<CategoricalNB<F, Fitted>, AlgoError> {
        let (n_samples, _n_features) = shape;
        validate_geometry(x, shape)?;
        let y = y.ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "fit (requires y)",
        })?;
        if y.len() != n_samples {
            return Err(AlgoError::Prim(PrimError::ShapeMismatch {
                operand: "y",
                rows: n_samples,
                cols: 1,
                len: y.len(),
            }));
        }
        // The device buffers exist only because `Fit` is the shared typestate
        // surface — this estimator's math is entirely host-side, so read both
        // operands back once and run the SAME body as `fit_from_host_slice`.
        let x_host = x.to_host(pool);
        let y_host = y.to_host(pool);
        self.fit_host(&x_host, &y_host, shape)
    }
}

impl<F> CategoricalNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    /// Per-query-row joint log-likelihood matrix (`n_query × n_classes`, host
    /// f64, row-major). Shared by `predict_labels` / `predict_proba` /
    /// `predict_log_proba`. Runs the geometry guard, then evaluates
    /// `class_log_prior_[c] + Σ_j feature_log_prob_[j][c, x[i,j]]` in host f64
    /// with the per-feature lookup index GUARDED against `n_categories_[j]`
    /// (T-11-04-02): an unseen / out-of-range category index `k ≥ n_categories_j`
    /// maps to the smoothed `log(alpha / denom_cj)` rather than indexing the
    /// ragged table out of bounds.
    fn joint_log_likelihood(
        &self,
        pool: &BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<Vec<f64>, AlgoError> {
        let (n_query, n_features) = shape;
        let feature_log_prob = self.feature_log_prob_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "predict (call fit first)",
        })?;
        let n_categories = self.n_categories_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "predict (call fit first)",
        })?;
        let class_log_prior = self.class_log_prior_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "predict (call fit first)",
        })?;
        // WR-06: `class_count_` is still required-fitted (kept as a not-fitted
        // guard) but no longer consulted at predict — the unseen-category smoothed
        // fallback that used it is now a hard error. Bind with `_` to retain the
        // fitted-state check without an unused-variable warning.
        let _class_count = self.class_count_.as_ref().ok_or(AlgoError::NotFitted {
            estimator: "categorical_nb",
            operation: "predict (call fit first)",
        })?;
        // Geometry guard BEFORE any host work (T-11-04 / ASVS V5).
        if n_query == 0 || n_features != self.n_features || x.len() != n_query * n_features {
            return Err(AlgoError::InvalidCategoricalInput {
                estimator: "categorical_nb",
                reason: format!(
                    "predict geometry: got {n_query}x{n_features}, fitted n_features={}",
                    self.n_features
                ),
            });
        }
        let n_classes = self.classes_.len();
        let x_h = x.to_host(pool);

        let mut jll = vec![0.0f64; n_query * n_classes];
        for r in 0..n_query {
            for c in 0..n_classes {
                let mut acc = class_log_prior[c];
                for j in 0..n_features {
                    let n_cat_j = n_categories[j];
                    let flp_j = &feature_log_prob[j];
                    let xf = host_to_f64(x_h[r * n_features + j]);
                    let xr = xf.round();
                    // WR-06 / T-11-04-02: a predict-time category index that is
                    // negative, non-integer, or >= n_categories_j is REJECTED with
                    // `InvalidCategoricalInput` — matching sklearn (which raises
                    // IndexError/ValueError on an out-of-range category) and the
                    // documented purpose of the error variant. (Previously such a
                    // category was silently mapped to the smoothed log(alpha/denom)
                    // fallback, which diverged from sklearn and contradicted the
                    // variant's own doc.)
                    if (xr - xf).abs() > NB_LABEL_INT_TOL || xr < 0.0 {
                        return Err(AlgoError::InvalidCategoricalInput {
                            estimator: "categorical_nb",
                            reason: format!(
                                "feature values must be non-negative integers (got {xf} for feature {j})"
                            ),
                        });
                    }
                    let k = xr as usize;
                    if k >= n_cat_j {
                        return Err(AlgoError::InvalidCategoricalInput {
                            estimator: "categorical_nb",
                            reason: format!(
                                "category index {k} >= n_categories {n_cat_j} for feature {j}"
                            ),
                        });
                    }
                    acc += flp_j[c * n_cat_j + k];
                }
                jll[r * n_classes + c] = acc;
            }
        }
        Ok(jll)
    }
}

impl<F> PredictLabels<F> for CategoricalNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_labels(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, i32>, AlgoError> {
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let labels = argmax_decode(&jll, &self.classes_);
        Ok(DeviceArray::from_host(pool, &labels))
    }
}

impl<F> PredictProba<F> for CategoricalNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _n_features) = shape;
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let n_classes = self.classes_.len();
        let mut proba: Vec<F> = vec![f64_to_host::<F>(0.0); n_query * n_classes];
        for r in 0..n_query {
            let row = &jll[r * n_classes..(r + 1) * n_classes];
            let (p, _lp) = log_sum_exp_normalize(row, n_classes);
            for (c, &pv) in p.iter().enumerate() {
                proba[r * n_classes + c] = f64_to_host::<F>(pv);
            }
        }
        Ok(DeviceArray::from_host(pool, &proba))
    }
}

impl<F> PredictLogProba<F> for CategoricalNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    fn predict_log_proba(
        &self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        shape: (usize, usize),
    ) -> Result<DeviceArray<ActiveRuntime, F>, AlgoError> {
        let (n_query, _n_features) = shape;
        let jll = self.joint_log_likelihood(pool, x, shape)?;
        let n_classes = self.classes_.len();
        let mut log_proba: Vec<F> = vec![f64_to_host::<F>(0.0); n_query * n_classes];
        for r in 0..n_query {
            let row = &jll[r * n_classes..(r + 1) * n_classes];
            let (_p, lp) = log_sum_exp_normalize(row, n_classes);
            for (c, &lpv) in lp.iter().enumerate() {
                log_proba[r * n_classes + c] = f64_to_host::<F>(lpv);
            }
        }
        Ok(DeviceArray::from_host(pool, &log_proba))
    }
}
