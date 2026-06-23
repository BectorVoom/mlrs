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

/// Calibrated (Plan 04) trustworthiness/kNN-overlap slack below the umap-learn
/// 0.5.12 reference score (D-04 — RELATIVE-to-oracle, NEVER an absolute floor).
/// Set from the first fixture run's worst measured margin `(umap − mlrs)` across
/// all 5 metrics plus a tight safety buffer; recorded per metric in
/// `14-VALIDATION.md`. Measured worst positive margins: trust +0.0007 (euclidean),
/// overlap +0.0000 (mlrs ≥ umap on every metric). `ε = 0.02` keeps the gate tight
/// (≈28× the worst trust margin) while absorbing cpu/rocm structural jitter.
const PROPERTY_EPS: f64 = 0.02;
/// Calibrated (Plan 05) trustworthiness slack for the TRANSFORM new-points
/// sub-gate (UMAP-04, D-04 — RELATIVE-to-umap, NEVER an absolute floor). The
/// transform is a HARDER problem than the fit layout: it is a reduced-context
/// frozen-subset SGD (new points see only their training neighbours + random
/// negatives, never each other) driven by mlrs's SplitMix64 negatives vs
/// umap-learn's Tausworthe, so its relative margins are inherently wider than the
/// fit layout's `PROPERTY_EPS=0.02`. Calibrated from the first spectral-init
/// fixture sweep's worst measured margin `(umap − mlrs)` across all 5 metrics
/// (recorded per metric in 14-VALIDATION.md): euclidean −0.027, cosine −0.069
/// (mlrs BEATS umap), manhattan +0.0495, minkowski +0.0800, chebyshev +0.1448.
/// `ε = 0.15` covers the worst (chebyshev) margin with a small buffer while
/// keeping the gate a real relative-structure check (mlrs never collapses the
/// new-point structure — it stays within 0.15 trust of umap on every metric).
const TRANSFORM_PROPERTY_EPS: f64 = 0.15;
/// Calibrated downstream-ARI band: mlrs's clustering-vs-truth ARI must be within
/// this of umap's `(umap_ari − band)` (D-04). Measured ARI gap was 0.0000 on all
/// 5 metrics (both recover the 3 true clusters exactly); `band = 0.05` is a tight
/// relative gate, not an absolute floor.
const ARI_BAND: f64 = 0.05;

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

/// Deterministic host Lloyd k-means on a row-major `(n, dim)` embedding, returning
/// integer cluster labels. Seeded farthest-first-style init (first `k` distinct
/// rows by index) + fixed 50 Lloyd iterations — fully deterministic so the
/// downstream-ARI gate is reproducible (no sklearn / no device at test time).
fn host_kmeans_labels(emb: &[f64], n: usize, dim: usize, k: usize) -> Vec<i64> {
    // Init centroids from the first k rows (deterministic).
    let mut centroids: Vec<f64> = vec![0.0; k * dim];
    for c in 0..k {
        for d in 0..dim {
            centroids[c * dim + d] = emb[c * dim + d];
        }
    }
    let mut labels = vec![0i64; n];
    for _iter in 0..50 {
        // Assign.
        let mut changed = false;
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let mut acc = 0.0;
                for d in 0..dim {
                    let diff = emb[i * dim + d] - centroids[c * dim + d];
                    acc += diff * diff;
                }
                if acc < best_d {
                    best_d = acc;
                    best = c;
                }
            }
            if labels[i] != best as i64 {
                changed = true;
            }
            labels[i] = best as i64;
        }
        // Update.
        let mut sums = vec![0.0; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i] as usize;
            counts[c] += 1;
            for d in 0..dim {
                sums[c * dim + d] += emb[i * dim + d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] / counts[c] as f64;
                }
            }
        }
        if !changed {
            break;
        }
    }
    labels
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

/// D-10 runtime proof — REAL fit contract (Plan 05: the old trivial-zeros
/// assertion is gone now that `fit` runs the full pipeline). The fit produces a
/// FINITE, NON-zeros `(n, n_components)` embedding and `n_features_in()` reports
/// `p`. (Pre-Plan-04 this asserted an all-zeros embedding — that contract no
/// longer holds; the real layout moves coordinates off the origin.)
#[test]
fn fit_roundtrip() {
    if gate_f64("fit_roundtrip") {
        return;
    }
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // A small well-separated 2-cluster design (n=8, p=3) so the real SGD layout
    // produces non-degenerate, finite coordinates quickly (spectral init on n≤64
    // runs the dense Jacobi eig, but n=8 is fast).
    let n = 8usize;
    let p = 3usize;
    let x_host: Vec<f64> = vec![
        0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.0, 0.1, 0.0, 0.2, 0.0, // cluster A
        9.0, 9.0, 9.0, 9.1, 9.1, 9.1, 9.2, 9.0, 9.1, 9.0, 9.2, 9.0, // cluster B
    ];
    let x_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &x_host);

    let fitted = Umap::<f64>::builder()
        .n_neighbors(3)
        .n_components(2)
        .random_state(Some(42))
        .build::<f64>()
        .expect("umap builds")
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
    // Real-fit contract: every coordinate is finite (no NaN/Inf from the SGD).
    assert!(
        embedding.iter().all(|v| v.is_finite()),
        "real fit embedding must be all-finite, got {embedding:?}"
    );
    // Real-fit contract: the embedding is NOT the trivial all-zeros shell — the
    // SGD layout has moved coordinates off the origin (the old zeros contract is
    // replaced).
    assert!(
        embedding.iter().any(|&v| v != 0.0),
        "real fit embedding must be non-zeros (the trivial-zeros shell is gone)"
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
    let (_x, nx, _d) = upload_x(&mut pool, &case, "X");
    assert_eq!(nx, n, "X and coords agree on n");
    // `metric` is unused for spectral init — the fixture carries the symmetric
    // fuzzy graph COO directly (Plan 02 owns the X → graph metric path).
    let _ = metric;

    // Reconstruct umap's spectral_layout INPUT: the symmetric fuzzy graph
    // (graph.maximum(graph.T)) the fixture dumped as COO rows/cols/vals. Build
    // the dense n×n affinity and upload it, then drive the real spectral_init.
    let g_rows = case.expect_f64("rows");
    let g_cols = case.expect_f64("cols");
    let g_vals = case.expect_f64("vals");
    let mut affinity = vec![0.0f64; n * n];
    for e in 0..g_vals.len() {
        let r = g_rows[e].round() as usize;
        let c = g_cols[e].round() as usize;
        affinity[r * n + c] = g_vals[e];
    }
    let g_dev: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut pool, &affinity);

    use mlrs_algos::manifold::umap_init;
    let produced = umap_init::spectral_init::<f64>(&mut pool, &g_dev, n, k, 42)
        .unwrap_or_else(|e| panic!("spectral_init {metric_tag}: {e}"));
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
            "spectral_init {metric_tag} col {c}: up-to-sign err {col_err} > {}",
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

    // Plan 03: drive the real host LM a/b fit and assert each grid point ≤1e-5
    // vs umap-learn `find_ab_params`.
    use mlrs_algos::manifold::umap_init;
    for g in 0..a.len() {
        let (produced_a, produced_b) = umap_init::fit_ab(min_dist[g], spread[g])
            .unwrap_or_else(|e| panic!("fit_ab grid {g}: {e}"));
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

    // Downstream-ARI (D-04): cluster BOTH embeddings with the same deterministic
    // host k-means (k = number of true classes) and score each clustering against
    // the true labels via ARI. The gate is RELATIVE: mlrs's ARI must be within
    // ARI_BAND of umap's, never an absolute floor.
    let n_classes = {
        let mut s: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for &l in &labels {
            s.insert(l);
        }
        s.len().max(2)
    };
    let umap_km = host_kmeans_labels(umap_emb, n, n_components, n_classes);
    let mlrs_km = host_kmeans_labels(&mlrs_emb, n, n_components, n_classes);
    let umap_ari = downstream_ari(&labels, &umap_km);
    let mlrs_ari = downstream_ari(&labels, &mlrs_km);
    // Self-witness: the ARI helper is correct (ARI of labels with themselves = 1).
    assert_eq!(downstream_ari(&labels, &labels), 1.0, "ARI self-identity");

    // Calibration witness (printed under --nocapture so the recorded thresholds
    // in 14-VALIDATION.md are reproducible from the measured margins).
    println!(
        "CALIB layout_property {metric_tag}: trust mlrs={mlrs_trust:.4} umap={umap_trust:.4} \
         (margin {:.4}); overlap mlrs={mlrs_overlap:.4} umap={umap_overlap:.4} (margin {:.4}); \
         ARI mlrs={mlrs_ari:.4} umap={umap_ari:.4} (gap {:.4})",
        umap_trust - mlrs_trust,
        umap_overlap - mlrs_overlap,
        umap_ari - mlrs_ari,
    );

    // Relative-to-umap structural gate (D-04 — `≥ umap − ε`, NOT an absolute floor).
    assert!(
        mlrs_trust >= umap_trust - PROPERTY_EPS,
        "layout_property {metric_tag}: trustworthiness {mlrs_trust} < umap {umap_trust} − ε ({PROPERTY_EPS})"
    );
    assert!(
        mlrs_overlap >= umap_overlap - PROPERTY_EPS,
        "layout_property {metric_tag}: kNN-overlap {mlrs_overlap} < umap {umap_overlap} − ε ({PROPERTY_EPS})"
    );
    assert!(
        mlrs_ari >= umap_ari - ARI_BAND,
        "layout_property {metric_tag}: downstream-ARI {mlrs_ari} < umap {umap_ari} − band ({ARI_BAND})"
    );
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

/// Transform new-points property sub-gate (UMAP-04). The transformed new points'
/// trustworthiness ≥ umap−ε (NOT element-wise — mlrs uses SplitMix64 negatives so
/// coordinates ≠ umap by construction, the reason the gate is relative-structural,
/// D-04). ALSO asserts transform byte-identical reproducibility (D-05): two
/// `transform` runs with the same `random_state` produce a bit-identical result.
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
    // Match the oracle's `n_neighbors`; leave `n_epochs` at the default (None →
    // fit 500 / transform 100) — the TRANSFORM_PROPERTY_EPS below was calibrated
    // against exactly this config's measured margins (14-VALIDATION.md). (Raising
    // the transform epoch budget does NOT close the direct-metric margin — the gap
    // is structural, from the SplitMix64-vs-Tausworthe RNG divergence, not under-
    // convergence; the calibrated relative gate is the correct treatment, D-04.)
    let n_neighbors = case.expect_f64("n_neighbors")[0].round() as usize;

    let fitted = Umap::<f64>::builder()
        .n_neighbors(n_neighbors)
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

    // D-05: transform is byte-identical for a fixed random_state (per backend,
    // dtype). A second transform on the SAME fitted estimator must reproduce the
    // exact bits (host-drawn SplitMix64 negatives keyed by (seed, epoch, edge)).
    let mlrs_new_again = fitted
        .transform(&mut pool, &x_new_dev, (n_new, d))
        .expect("umap transform (repro)")
        .to_host(&pool);
    assert_eq!(
        mlrs_new.len(),
        mlrs_new_again.len(),
        "transform_property {metric_tag}: reproduced transform length mismatch"
    );
    for i in 0..mlrs_new.len() {
        assert_eq!(
            mlrs_new[i].to_bits(),
            mlrs_new_again[i].to_bits(),
            "transform_property {metric_tag} elem {i}: same-seed transform must be \
             byte-identical (D-05)"
        );
    }

    let k = 5usize.min(n_new - 1);
    let umap_trust = trustworthiness(x_new, umap_new_emb, n_new, d, n_components, k);
    let mlrs_trust = trustworthiness(x_new, &mlrs_new, n_new, d, n_components, k);
    println!(
        "CALIB transform_property {metric_tag}: new-pt trust mlrs={mlrs_trust:.4} \
         umap={umap_trust:.4} (margin {:.4})",
        umap_trust - mlrs_trust,
    );
    assert!(
        mlrs_trust >= umap_trust - TRANSFORM_PROPERTY_EPS,
        "transform_property {metric_tag}: new-pt trustworthiness {mlrs_trust} < umap {umap_trust} − ε \
         ({TRANSFORM_PROPERTY_EPS})"
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
