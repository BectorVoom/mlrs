//! Phase-14 UMAP value-gate + property-gate + reproducibility harness.
//!
//! This file holds BOTH the Phase-12 convention tests (defaults / build /
//! roundtrip / no-leak — kept GREEN) AND the Phase-14 algorithm gates. The
//! Phase-14 gates are **RED-by-design**: they reference the real `Umap::fit`/
//! `transform` bodies and the committed umap-learn 0.5.12 oracle fixtures, but
//! the current shell `fit` emits an all-zeros embedding, so every value/property
//! assertion FAILS until Plans 02–05 land the real stages. They MUST compile and
//! run (the build is the contract Wave-0 establishes; runtime RED is expected).
//!
//! ## Test families (one per VALIDATION map row)
//!   - `smooth_knn_<metric>`  — per-row ρ/σ value ≤1e-5 vs umap `sigmas`/`rhos`
//!   - `fuzzy_union_<metric>` — t-conorm graph COO value ≤1e-5 vs `rows/cols/vals`
//!   - `spectral_init_<metric>` — spectral coords value ≤1e-5 up-to-sign per col
//!   - `ab_fit`                — a/b LM curve fit value ≤1e-5 vs umap `a`/`b`
//!   - `layout_property_<metric>` — trustworthiness/kNN-overlap ≥ umap−ε, ARI band
//!   - `reproducible_<dtype>` — byte-identical embedding across two `fit` runs
//!   - `transform_property_<metric>` — trustworthiness of new pts ≥ umap−ε
//!
//! ## Host property-gate helpers (no sklearn at test time)
//!   - `trustworthiness(high, low, n, d_high, d_low, k)`
//!   - `knn_overlap(high, low, n, d_high, d_low, k)`
//!   - `downstream_ari(labels_a, labels_b)`
//!
//! f64 cases carry the `skip_f64_with_log` capability gate VERBATIM (cpu runs
//! f64; rocm skips-with-log). Per AGENTS.md §2 tests live here, never an
//! in-source `#[cfg(test)] mod tests`.
//!
//! ## Calibration TODO (RESEARCH Q4 / Spike flag item 2)
//! The property-gate ε / ARI-band consts below are PLACEHOLDER values. Plan 04's
//! calibration run (mlrs vs umap on identical data) overwrites them with the
//! measured margin + safety buffer and records the calibrated numbers in
//! `14-VALIDATION.md`. Do NOT invent thresholds before that run.

use std::path::PathBuf;

use mlrs_algos::manifold::umap::{Metric, Umap};
use mlrs_algos::error::BuildError;
use mlrs_algos::typestate::{Fit, Transform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase, F64_TOL};

// ===========================================================================
// Placeholder calibration consts (Plan 04 overwrites — RESEARCH Q4)
// ===========================================================================

/// TODO(Plan 04 calibration): trustworthiness/kNN-overlap slack below the umap
/// reference score. Placeholder only — replace with the calibrated margin and
/// record in 14-VALIDATION.md (do NOT invent the real threshold here).
const PROPERTY_EPS: f64 = 0.05;
/// TODO(Plan 04 calibration): allowed downstream-ARI band below umap's ARI.
const ARI_BAND: f64 = 0.10;

// ===========================================================================
// Fixture path + load helpers
// ===========================================================================

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

/// The five metrics the per-stage fixtures cover, paired with the `Umap::Metric`
/// the estimator routes through (Minkowski carries the fixed oracle exponent).
const METRICS: &[(&str, Metric)] = &[
    ("euclidean", Metric::Euclidean),
    ("manhattan", Metric::Manhattan),
    ("cosine", Metric::Cosine),
    ("chebyshev", Metric::Chebyshev),
    ("minkowski", Metric::Minkowski { p: 3.0 }),
];

/// Standard f64 capability gate — COPIED VERBATIM (umap_test.rs convention):
/// cpu runs f64; rocm SKIPS f64-with-log. Returns `true` when the caller should
/// early-return (skip).
fn gate_f64(case: &str) -> bool {
    let backend = capability::active_backend_name();
    capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
    if capability::skip_f64_with_log() {
        println!("umap {case} f64 backend={backend}: SKIPPED (no f64 support)");
        return true;
    }
    false
}

/// Build a fresh f64 pool + upload `X` from a loaded fixture as a `DeviceArray`.
fn upload_x(
    pool: &mut BufferPool<ActiveRuntime>,
    case: &OracleCase,
    name: &str,
) -> (DeviceArray<ActiveRuntime, f64>, usize, usize) {
    let x = case.expect_f64(name);
    let shape = case.shape(name).expect("X has a shape");
    let (n, d) = (shape[0] as usize, shape[1] as usize);
    let dev = DeviceArray::from_host(pool, x);
    (dev, n, d)
}

// ===========================================================================
// Host property-gate metric helpers (no sklearn at test time)
// ===========================================================================

/// Pairwise Euclidean distance matrix (n×n, row-major) for a flat `(n, d)`
/// row-major buffer. Host f64 — used by the structural property gates.
fn pairwise_euclidean(data: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for c in 0..d {
                let diff = data[i * d + c] - data[j * d + c];
                acc += diff * diff;
            }
            out[i * n + j] = acc.sqrt();
        }
    }
    out
}

/// Ascending neighbour-index ranking of each row of a distance matrix, EXCLUDING
/// self (the diagonal). Returns `n` vectors of length `n-1` (nearest first).
fn rank_neighbors(dist: &[f64], n: usize) -> Vec<Vec<usize>> {
    let mut ranks = Vec::with_capacity(n);
    for i in 0..n {
        let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        idx.sort_by(|&a, &b| {
            dist[i * n + a]
                .partial_cmp(&dist[i * n + b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        ranks.push(idx);
    }
    ranks
}

/// sklearn `manifold.trustworthiness` host port (no sklearn at test time):
/// `T = 1 − (2 / (n·k·(2n − 3k − 1))) · Σ_i Σ_{j∈U_i^k} (r(i,j) − k)` where
/// `U_i^k` are the `k` low-D neighbours of `i` that are NOT among its `k` high-D
/// neighbours, and `r(i,j)` is `j`'s high-D rank (1-based) from `i`.
fn trustworthiness(
    high: &[f64],
    low: &[f64],
    n: usize,
    d_high: usize,
    d_low: usize,
    k: usize,
) -> f64 {
    assert!(n > 0 && k > 0 && 2 * n > 3 * k + 1, "trustworthiness shape guard");
    let dh = pairwise_euclidean(high, n, d_high);
    let dl = pairwise_euclidean(low, n, d_low);
    let rh = rank_neighbors(&dh, n);
    let rl = rank_neighbors(&dl, n);

    // high_rank[i][j] = 1-based rank of j among i's high-D neighbours.
    let mut high_rank = vec![vec![0usize; n]; n];
    for i in 0..n {
        for (pos, &j) in rh[i].iter().enumerate() {
            high_rank[i][j] = pos + 1;
        }
    }

    let mut sum = 0.0f64;
    for i in 0..n {
        let high_k: std::collections::HashSet<usize> =
            rh[i].iter().take(k).copied().collect();
        for &j in rl[i].iter().take(k) {
            if !high_k.contains(&j) {
                sum += high_rank[i][j] as f64 - k as f64;
            }
        }
    }
    let norm = 2.0 / (n as f64 * k as f64 * (2 * n - 3 * k - 1) as f64);
    1.0 - norm * sum
}

/// kNN-overlap: mean fraction of each point's `k` high-D neighbours retained
/// among its `k` low-D neighbours.
fn knn_overlap(
    high: &[f64],
    low: &[f64],
    n: usize,
    d_high: usize,
    d_low: usize,
    k: usize,
) -> f64 {
    let dh = pairwise_euclidean(high, n, d_high);
    let dl = pairwise_euclidean(low, n, d_low);
    let rh = rank_neighbors(&dh, n);
    let rl = rank_neighbors(&dl, n);
    let mut acc = 0.0f64;
    for i in 0..n {
        let hk: std::collections::HashSet<usize> = rh[i].iter().take(k).copied().collect();
        let retained = rl[i].iter().take(k).filter(|j| hk.contains(j)).count();
        acc += retained as f64 / k as f64;
    }
    acc / n as f64
}

/// Adjusted Rand Index between two integer label vectors (host, no sklearn).
/// `ARI = (Σ C(n_ij,2) − [Σ C(a_i,2)·Σ C(b_j,2)]/C(n,2)) /
///        (½[Σ C(a_i,2)+Σ C(b_j,2)] − [Σ C(a_i,2)·Σ C(b_j,2)]/C(n,2))`.
fn downstream_ari(labels_a: &[i64], labels_b: &[i64]) -> f64 {
    assert_eq!(labels_a.len(), labels_b.len(), "ARI label length mismatch");
    let n = labels_a.len();
    let c2 = |x: u64| -> f64 {
        if x < 2 {
            0.0
        } else {
            (x * (x - 1) / 2) as f64
        }
    };
    use std::collections::HashMap;
    let mut contingency: HashMap<(i64, i64), u64> = HashMap::new();
    let mut a_counts: HashMap<i64, u64> = HashMap::new();
    let mut b_counts: HashMap<i64, u64> = HashMap::new();
    for i in 0..n {
        *contingency.entry((labels_a[i], labels_b[i])).or_insert(0) += 1;
        *a_counts.entry(labels_a[i]).or_insert(0) += 1;
        *b_counts.entry(labels_b[i]).or_insert(0) += 1;
    }
    let sum_ij: f64 = contingency.values().map(|&v| c2(v)).sum();
    let sum_a: f64 = a_counts.values().map(|&v| c2(v)).sum();
    let sum_b: f64 = b_counts.values().map(|&v| c2(v)).sum();
    let total = c2(n as u64);
    let expected = if total == 0.0 { 0.0 } else { sum_a * sum_b / total };
    let max_index = 0.5 * (sum_a + sum_b);
    if (max_index - expected).abs() < f64::EPSILON {
        return 1.0;
    }
    (sum_ij - expected) / (max_index - expected)
}

/// Decode a float-encoded integer label array to `i64` (fixtures store labels as
/// floats — load_npz constraint).
fn decode_labels(case: &OracleCase, name: &str) -> Vec<i64> {
    case.expect_f64(name)
        .iter()
        .map(|&v| v.round() as i64)
        .collect()
}

/// Run the (currently trivial) `Umap::fit` and read back the host embedding.
fn fit_embedding(
    pool: &mut BufferPool<ActiveRuntime>,
    x: &DeviceArray<ActiveRuntime, f64>,
    n: usize,
    d: usize,
    metric: Metric,
    random_state: Option<u64>,
) -> Vec<f64> {
    let fitted = Umap::<f64>::builder()
        .n_neighbors(10)
        .n_components(2)
        .metric(metric)
        .random_state(random_state)
        .build::<f64>()
        .expect("umap builds")
        .fit(pool, x, None, (n, d))
        .expect("umap fit");
    fitted.embedding(pool)
}

// ===========================================================================
// Phase-12 convention tests (KEPT GREEN)
// ===========================================================================

/// BLDR-01: `Umap::new()` equals `Umap::builder().build()?` on the
/// hyperparameter subset. Pure host comparison — no device, so no f64 gate.
#[test]
fn defaults_equal() {
    let from_new = Umap::<f64>::new();
    let from_builder = Umap::<f64>::builder()
        .build::<f64>()
        .expect("default UmapBuilder builds");
    assert!(
        from_new.hyperparams_eq(&from_builder),
        "Umap::new() and builder().build()? must agree on hyperparameters (BLDR-01)"
    );
}

/// D-08 / T-12-02: `min_dist > spread` is rejected at `build()` with the typed
/// `BuildError::InvalidMinDist`, BEFORE any data is seen.
#[test]
fn build_rejects_bad_min_dist() {
    let bad = Umap::<f64>::builder()
        .min_dist(2.0)
        .spread(1.0)
        .build::<f64>()
        .err();
    assert!(
        matches!(
            bad,
            Some(BuildError::InvalidMinDist { min_dist, .. }) if min_dist == 2.0
        ),
        "min_dist > spread must be BuildError::InvalidMinDist, got {bad:?}"
    );
}

/// D-10 runtime proof: the fit round-trips — `embedding()` returns
/// `n * n_components` values and `n_features_in()` reports `p`.
#[test]
fn fit_roundtrip() {
    if gate_f64("fit_roundtrip") {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n = 6usize;
    let p = 3usize;
    let x_host: Vec<f64> = (0..n * p).map(|i| i as f64).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let fitted = Umap::<f64>::new()
        .fit(&mut pool, &x_dev, None, (n, p))
        .expect("fit succeeds");

    let n_components = 2usize;
    let embedding = fitted.embedding(&pool);
    assert_eq!(
        embedding.len(),
        n * n_components,
        "embedding length must be n * n_components"
    );
    assert_eq!(
        fitted.n_features_in(),
        p,
        "n_features_in() must report the fit-time feature count"
    );
}

/// Memory gate: re-CONSTRUCT + re-fit at the same shape does not grow
/// `live_bytes`.
#[test]
fn fit_no_leak() {
    if gate_f64("fit_no_leak") {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let n = 8usize;
    let p = 4usize;
    let x_host: Vec<f64> = (0..n * p).map(|i| i as f64).collect();
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let fitted = Umap::<f64>::new()
        .fit(&mut pool, &x_dev, None, (n, p))
        .expect("first fit");
    drop(fitted);
    let live_after_first = pool.stats().live_bytes;

    const REFITS: usize = 4;
    for k in 0..REFITS {
        let fitted = Umap::<f64>::new()
            .fit(&mut pool, &x_dev, None, (n, p))
            .expect("re-fit");
        drop(fitted);
        let live = pool.stats().live_bytes;
        assert!(
            live <= live_after_first,
            "live_bytes grew across re-construct+fit {k}: {live} > first {live_after_first}"
        );
    }
}

// ===========================================================================
// Phase-14 value-gate tests (RED-by-design — Plans 02–05 turn GREEN)
// ===========================================================================

/// Shared smooth-kNN ρ/σ value gate (UMAP-02). Plan 02: drive the real host
/// `smooth_knn_dist` on the SAME committed `knn_dist` umap consumed and assert
/// both `sigmas` and `rhos` match umap-learn 0.5.12 to ≤1e-5 per row, for every
/// metric. The host f64 path matches umap's own f32-cast internals well inside
/// the 1e-5 tolerance (no device-reduction-order drift). `metric` is unused
/// here — these stages are metric-agnostic (they consume precomputed distances).
fn run_smooth_knn(metric_tag: &str, _metric: Metric) {
    if gate_f64(&format!("smooth_knn_{metric_tag}")) {
        return;
    }

    let case = load_npz(fixture(&format!("umap_fuzzy_{metric_tag}_f64.npz")))
        .unwrap_or_else(|e| panic!("load umap_fuzzy_{metric_tag}: {e}"));
    let sigmas = case.expect_f64("sigmas");
    let rhos = case.expect_f64("rhos");
    let knn_dist = case.expect_f64("knn_dist");
    let kshape = case.shape("knn_dist").expect("knn_dist shape");
    let (n, k) = (kshape[0] as usize, kshape[1] as usize);
    let n_neighbors = case.expect_f64("n_neighbors")[0].round() as usize;
    let local_connectivity = case.expect_f64("local_connectivity")[0];

    assert_eq!(sigmas.len(), n, "one sigma per row");
    assert_eq!(rhos.len(), n, "one rho per row");

    let (produced_sigmas, produced_rhos) = mlrs_algos::manifold::umap_internals::smooth_knn_dist(
        knn_dist,
        n,
        k,
        n_neighbors,
        local_connectivity,
    );

    for i in 0..n {
        assert!(
            mlrs_core::is_close(produced_rhos[i], rhos[i], &F64_TOL),
            "smooth_knn {metric_tag} row {i}: rho {} != umap {}",
            produced_rhos[i],
            rhos[i]
        );
        assert!(
            mlrs_core::is_close(produced_sigmas[i], sigmas[i], &F64_TOL),
            "smooth_knn {metric_tag} row {i}: sigma {} != umap {}",
            produced_sigmas[i],
            sigmas[i]
        );
    }
}

#[test]
fn smooth_knn_euclidean() {
    run_smooth_knn("euclidean", Metric::Euclidean);
}
#[test]
fn smooth_knn_manhattan() {
    run_smooth_knn("manhattan", Metric::Manhattan);
}
#[test]
fn smooth_knn_cosine() {
    run_smooth_knn("cosine", Metric::Cosine);
}
#[test]
fn smooth_knn_chebyshev() {
    run_smooth_knn("chebyshev", Metric::Chebyshev);
}
#[test]
fn smooth_knn_minkowski() {
    run_smooth_knn("minkowski", Metric::Minkowski { p: 3.0 });
}

/// Shared fuzzy-union (t-conorm) value gate (UMAP-02, D-04). Plan 02: drive the
/// real `smooth_knn_dist` → `compute_membership_strengths` → `fuzzy_union`
/// pipeline on the committed KNN and assert the produced symmetric graph COO
/// (rows/cols/vals, scipy CSR-canonical order) matches umap-learn 0.5.12 to
/// ≤1e-5 for all 5 metrics. `metric` is unused (stages are metric-agnostic).
///
/// FLOAT32-INPUT FIDELITY (RESEARCH Pitfall 6): umap-learn feeds its stages the
/// pynndescent KNN distances, which are **float32**, whereas the fixture dumps
/// the f64 "true distance" array. Running the membership `exp` on the f64
/// distances reproduces umap's COO `vals` to only ~1.0e-5 *relative* on the few
/// edges where `(d−ρ)/σ` is largest (the `exp` amplifies the f32↔f64 distance
/// gap past the 1e-5 bound). Casting `knn_dist` to f32 precision first — exactly
/// the array umap's stages consumed — drives the whole pipeline to ≤1.6e-7 for
/// all 5 metrics. The stage fns stay pure f64; the f32 round here reconstructs
/// umap's actual numba input, which is the faithful per-stage gate.
fn run_fuzzy_union(metric_tag: &str, _metric: Metric) {
    if gate_f64(&format!("fuzzy_union_{metric_tag}")) {
        return;
    }

    let case = load_npz(fixture(&format!("umap_fuzzy_{metric_tag}_f64.npz")))
        .unwrap_or_else(|e| panic!("load umap_fuzzy_{metric_tag}: {e}"));
    let rows = case.expect_f64("rows");
    let cols = case.expect_f64("cols");
    let vals = case.expect_f64("vals");
    let knn_idx = case.expect_f64("knn_idx");
    // Reconstruct umap's actual stage input: the pynndescent KNN distances are
    // float32. Round-trip f64→f32→f64 so the host f64 numerics see the exact
    // values umap's numba kernels consumed (see fn doc — float32-input fidelity).
    let knn_dist: Vec<f64> = case
        .expect_f64("knn_dist")
        .iter()
        .map(|&d| d as f32 as f64)
        .collect();
    let kshape = case.shape("knn_dist").expect("knn_dist shape");
    let (n, k) = (kshape[0] as usize, kshape[1] as usize);
    let n_neighbors = case.expect_f64("n_neighbors")[0].round() as usize;
    let local_connectivity = case.expect_f64("local_connectivity")[0];
    let set_op_mix_ratio = case.expect_f64("set_op_mix_ratio")[0];

    assert_eq!(rows.len(), cols.len(), "COO rows/cols same length");
    assert_eq!(rows.len(), vals.len(), "COO rows/vals same length");

    use mlrs_algos::manifold::umap_internals;
    let (sigmas, rhos) =
        umap_internals::smooth_knn_dist(&knn_dist, n, k, n_neighbors, local_connectivity);
    let (m_rows, m_cols, m_vals) =
        umap_internals::compute_membership_strengths(knn_idx, &knn_dist, &rhos, &sigmas, n, k);
    let (g_rows, g_cols, g_vals) =
        umap_internals::fuzzy_union(&m_rows, &m_cols, &m_vals, n, set_op_mix_ratio);

    assert_eq!(
        g_vals.len(),
        vals.len(),
        "fuzzy_union {metric_tag}: produced {} edges, umap has {} (after eliminate_zeros)",
        g_vals.len(),
        vals.len()
    );
    for e in 0..vals.len() {
        assert_eq!(
            g_rows[e], rows[e].round() as usize,
            "fuzzy_union {metric_tag} edge {e}: row {} != umap {}",
            g_rows[e], rows[e]
        );
        assert_eq!(
            g_cols[e], cols[e].round() as usize,
            "fuzzy_union {metric_tag} edge {e}: col {} != umap {}",
            g_cols[e], cols[e]
        );
        assert!(
            mlrs_core::is_close(g_vals[e], vals[e], &F64_TOL),
            "fuzzy_union {metric_tag} edge {e} (r={},c={}): val {} != umap {}",
            g_rows[e],
            g_cols[e],
            g_vals[e],
            vals[e]
        );
    }
}

#[test]
fn fuzzy_union_euclidean() {
    run_fuzzy_union("euclidean", Metric::Euclidean);
}
#[test]
fn fuzzy_union_manhattan() {
    run_fuzzy_union("manhattan", Metric::Manhattan);
}
#[test]
fn fuzzy_union_cosine() {
    run_fuzzy_union("cosine", Metric::Cosine);
}
#[test]
fn fuzzy_union_chebyshev() {
    run_fuzzy_union("chebyshev", Metric::Chebyshev);
}
#[test]
fn fuzzy_union_minkowski() {
    run_fuzzy_union("minkowski", Metric::Minkowski { p: 3.0 });
}

/// Shared spectral-init value gate (UMAP-02). Compare ≤1e-5 UP-TO-SIGN per
/// column vs umap `spectral_layout` coords (RESEARCH Q3 — umap applies no
/// sign-flip; mlrs `recover` does). RED: no spectral-init stage yet (Plan 03).
fn run_spectral_init(metric_tag: &str, metric: Metric) {
    if gate_f64(&format!("spectral_init_{metric_tag}")) {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let case = load_npz(fixture(&format!("umap_spectral_{metric_tag}_f64.npz")))
        .unwrap_or_else(|e| panic!("load umap_spectral_{metric_tag}: {e}"));
    let coords = case.expect_f64("coords");
    let cshape = case.shape("coords").expect("coords shape");
    let (n, k) = (cshape[0] as usize, cshape[1] as usize);
    let (x, nx, d) = upload_x(&mut pool, &case, "X");
    assert_eq!(nx, n, "X and coords agree on n");
    let _embedding = fit_embedding(&mut pool, &x, n, d, metric, Some(42));

    // RED-by-design: no spectral-init stage yet. Plan 03 produces the spectral
    // coords; compare each column up-to-sign against `coords` ≤1e-5.
    let produced = vec![0.0f64; n * k]; // placeholder: Plan 03 spectral_init
    for c in 0..k {
        // Up-to-sign per column: pick the sign minimizing the abs error.
        let mut err_pos = 0.0f64;
        let mut err_neg = 0.0f64;
        for r in 0..n {
            let got = produced[r * k + c];
            let want = coords[r * k + c];
            err_pos = err_pos.max((got - want).abs());
            err_neg = err_neg.max((got + want).abs());
        }
        let col_err = err_pos.min(err_neg);
        assert!(
            col_err <= F64_TOL.abs,
            "spectral_init {metric_tag} col {c}: up-to-sign err {col_err} > {} (RED until Plan 03)",
            F64_TOL.abs
        );
    }
}

#[test]
fn spectral_init_euclidean() {
    run_spectral_init("euclidean", Metric::Euclidean);
}
#[test]
fn spectral_init_manhattan() {
    run_spectral_init("manhattan", Metric::Manhattan);
}
#[test]
fn spectral_init_cosine() {
    run_spectral_init("cosine", Metric::Cosine);
}
#[test]
fn spectral_init_chebyshev() {
    run_spectral_init("chebyshev", Metric::Chebyshev);
}
#[test]
fn spectral_init_minkowski() {
    run_spectral_init("minkowski", Metric::Minkowski { p: 3.0 });
}

/// a/b LM curve-fit value gate (UMAP-01/02, metric-independent). RED: no host LM
/// a/b fit yet (Plan 03). Asserts ≤1e-5 vs umap `find_ab_params` over the grid.
#[test]
fn ab_fit() {
    if gate_f64("ab_fit") {
        return;
    }
    let case = load_npz(fixture("umap_ab_f64.npz")).expect("load umap_ab");
    let min_dist = case.expect_f64("min_dist");
    let spread = case.expect_f64("spread");
    let a = case.expect_f64("a");
    let b = case.expect_f64("b");
    assert_eq!(min_dist.len(), spread.len(), "grid parallel arrays");
    assert_eq!(a.len(), b.len(), "a/b parallel arrays");

    // RED-by-design: no host LM a/b fit yet. Plan 03 lands `fit_ab(min_dist,
    // spread) -> (a, b)`; assert each grid point ≤1e-5.
    for g in 0..a.len() {
        let (produced_a, produced_b) = (0.0f64, 0.0f64); // placeholder: Plan 03 fit_ab
        assert!(
            mlrs_core::is_close(produced_a, a[g], &F64_TOL),
            "ab_fit grid {g} (min_dist={}, spread={}): a {} != umap {} (RED until Plan 03)",
            min_dist[g],
            spread[g],
            produced_a,
            a[g]
        );
        assert!(
            mlrs_core::is_close(produced_b, b[g], &F64_TOL),
            "ab_fit grid {g}: b {} != umap {} (RED until Plan 03)",
            produced_b,
            b[g]
        );
    }
}

// ===========================================================================
// Phase-14 property-gate + reproducibility + transform (RED-by-design)
// ===========================================================================

/// Shared SGD-layout structural property gate (UMAP-03). NOT element-wise:
/// trustworthiness / kNN-overlap ≥ umap−ε and downstream-ARI within band. RED:
/// the zeros embedding collapses all structure (trustworthiness ≈ low), so the
/// gate FAILS until Plan 04 lands the real layout.
fn run_layout_property(metric_tag: &str, metric: Metric) {
    if gate_f64(&format!("layout_property_{metric_tag}")) {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let case = load_npz(fixture(&format!("umap_layout_{metric_tag}_f64.npz")))
        .unwrap_or_else(|e| panic!("load umap_layout_{metric_tag}: {e}"));
    let (x, n, d) = upload_x(&mut pool, &case, "X");
    let umap_emb = case.expect_f64("embedding");
    let emb_shape = case.shape("embedding").expect("embedding shape");
    let n_components = emb_shape[1] as usize;
    let labels = decode_labels(&case, "labels");

    let mlrs_emb = fit_embedding(&mut pool, &x, n, d, metric, Some(42));

    let high = case.expect_f64("X");
    let k = 10usize.min(n - 1);
    // umap reference structural scores (computed in-repo on the dumped embedding).
    let umap_trust = trustworthiness(high, umap_emb, n, d, n_components, k);
    let mlrs_trust = trustworthiness(high, &mlrs_emb, n, d, n_components, k);
    let umap_overlap = knn_overlap(high, umap_emb, n, d, n_components, k);
    let mlrs_overlap = knn_overlap(high, &mlrs_emb, n, d, n_components, k);

    // Downstream ARI: a trivial host k=clusters labeling by nearest embedding
    // centroid is deferred to Plan 04; here assert the helper runs against the
    // true labels as a self-consistency witness, then the structural gate.
    let _self_ari = downstream_ari(&labels, &labels);
    assert_eq!(_self_ari, 1.0, "ARI of labels with themselves is 1.0");

    // RED-by-design structural gate: zeros embedding cannot match umap structure.
    assert!(
        mlrs_trust >= umap_trust - PROPERTY_EPS,
        "layout_property {metric_tag}: trustworthiness {mlrs_trust} < umap {umap_trust} − ε \
         (RED until Plan 04; PROPERTY_EPS is a calibration placeholder)"
    );
    assert!(
        mlrs_overlap >= umap_overlap - PROPERTY_EPS,
        "layout_property {metric_tag}: kNN-overlap {mlrs_overlap} < umap {umap_overlap} − ε \
         (RED until Plan 04)"
    );
    // ARI band is exercised once Plan 04 produces real cluster labels from both
    // embeddings; ARI_BAND is referenced so the calibration const is live.
    assert!(ARI_BAND > 0.0, "ARI_BAND placeholder must be positive");
}

#[test]
fn layout_property_euclidean() {
    run_layout_property("euclidean", Metric::Euclidean);
}
#[test]
fn layout_property_manhattan() {
    run_layout_property("manhattan", Metric::Manhattan);
}
#[test]
fn layout_property_cosine() {
    run_layout_property("cosine", Metric::Cosine);
}
#[test]
fn layout_property_chebyshev() {
    run_layout_property("chebyshev", Metric::Chebyshev);
}
#[test]
fn layout_property_minkowski() {
    run_layout_property("minkowski", Metric::Minkowski { p: 3.0 });
}

/// Same-`random_state` reproducibility (UMAP-03, D-05): two independent `fit`
/// runs with the same seed produce a BYTE-IDENTICAL embedding (per backend +
/// dtype). This is GREEN-trivially today (zeros == zeros) but is kept as the
/// reproducibility contract Plan 04's real stochastic layout MUST preserve.
fn run_reproducible(dtype_tag: &str) {
    if gate_f64(&format!("reproducible_{dtype_tag}")) {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let case = load_npz(fixture("umap_layout_euclidean_f64.npz"))
        .expect("load umap_layout_euclidean");
    let (x, n, d) = upload_x(&mut pool, &case, "X");

    let emb_a = fit_embedding(&mut pool, &x, n, d, Metric::Euclidean, Some(7));
    let emb_b = fit_embedding(&mut pool, &x, n, d, Metric::Euclidean, Some(7));
    assert_eq!(emb_a.len(), emb_b.len(), "reproducible embeddings same length");
    for i in 0..emb_a.len() {
        assert_eq!(
            emb_a[i].to_bits(),
            emb_b[i].to_bits(),
            "reproducible_{dtype_tag} elem {i}: same-seed fit must be byte-identical (D-05)"
        );
    }
}

#[test]
fn reproducible_f64() {
    run_reproducible("f64");
}

/// Transform new-points property sub-gate (UMAP-04). Trustworthiness of the
/// transformed new points ≥ umap−ε (NOT element-wise). RED: the trivial
/// transform emits zeros, collapsing new-point structure, so the gate FAILS
/// until Plan 05 lands the real frozen-subset transform.
fn run_transform_property(metric_tag: &str, metric: Metric) {
    if gate_f64(&format!("transform_property_{metric_tag}")) {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let case = load_npz(fixture(&format!("umap_transform_{metric_tag}_f64.npz")))
        .unwrap_or_else(|e| panic!("load umap_transform_{metric_tag}: {e}"));
    let (x_train, n_train, d) = upload_x(&mut pool, &case, "X_train");
    let x_new = case.expect_f64("X_new");
    let new_shape = case.shape("X_new").expect("X_new shape");
    let (n_new, d_new) = (new_shape[0] as usize, new_shape[1] as usize);
    assert_eq!(d, d_new, "train/new feature dims agree");
    let umap_new_emb = case.expect_f64("embedding_new");
    let emb_shape = case.shape("embedding_new").expect("embedding_new shape");
    let n_components = emb_shape[1] as usize;

    let fitted = Umap::<f64>::builder()
        .n_neighbors(10)
        .n_components(2)
        .metric(metric)
        .random_state(Some(42))
        .build::<f64>()
        .expect("umap builds")
        .fit(&mut pool, &x_train, None, (n_train, d))
        .expect("umap fit");

    let x_new_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, x_new);
    let mlrs_new = fitted
        .transform(&mut pool, &x_new_dev, (n_new, d))
        .expect("umap transform")
        .to_host(&pool);

    let k = 5usize.min(n_new - 1);
    let umap_trust = trustworthiness(x_new, umap_new_emb, n_new, d, n_components, k);
    let mlrs_trust = trustworthiness(x_new, &mlrs_new, n_new, d, n_components, k);
    assert!(
        mlrs_trust >= umap_trust - PROPERTY_EPS,
        "transform_property {metric_tag}: new-pt trustworthiness {mlrs_trust} < umap {umap_trust} − ε \
         (RED until Plan 05)"
    );
}

#[test]
fn transform_property_euclidean() {
    run_transform_property("euclidean", Metric::Euclidean);
}
#[test]
fn transform_property_manhattan() {
    run_transform_property("manhattan", Metric::Manhattan);
}
#[test]
fn transform_property_cosine() {
    run_transform_property("cosine", Metric::Cosine);
}
#[test]
fn transform_property_chebyshev() {
    run_transform_property("chebyshev", Metric::Chebyshev);
}
#[test]
fn transform_property_minkowski() {
    run_transform_property("minkowski", Metric::Minkowski { p: 3.0 });
}

/// Compile-time anchor so the `METRICS` table is exercised (and stays in sync
/// with the per-metric test fns above).
#[test]
fn metrics_table_covers_five() {
    assert_eq!(METRICS.len(), 5, "five-metric coverage");
    assert!(matches!(METRICS[4].1, Metric::Minkowski { p } if p == 3.0));
}
