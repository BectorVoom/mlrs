//! TSNE-01 — t-SNE (exact method) oracle gates, two tiers (the UMAP
//! convention):
//!
//!   - `p_matrix_matches_sklearn` — DETERMINISTIC ≤1e-5 gate: the dense joint
//!     probability matrix from [`mlrs_algos::manifold::tsne::joint_probabilities`]
//!     vs sklearn 1.9.0's `_joint_probabilities` (fixture `P`). The port
//!     replicates sklearn's f32-rounded distances + f64 binary search exactly.
//!   - `fit_band_*` — BAND gate: the end-to-end embedding is chaotic (1000
//!     gradient-descent iterations; mlrs's full-SVD PCA init vs sklearn's
//!     randomized solver), so the gate is neighborhood preservation:
//!     `trustworthiness(X, emb, 5) >= fixture_trust − 0.05` AND
//!     `0 < kl_divergence_ <= fixture_kl + 0.25`.
//!   - `deterministic_refit` — same input → bit-identical embedding (PCA init
//!     is deterministic; no RNG on the descent path).
//!   - `build_validation` / `fit_validation` — typed hyperparameter errors
//!     (D-08 split: data-independent at build, `perplexity < n` at fit).
//!
//! f64 cases carry the `skip_f64_with_log` capability gate verbatim. Per
//! AGENTS.md §2 tests live in `crates/mlrs-algos/tests/`.

use std::path::PathBuf;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::error::{AlgoError, BuildError};
use mlrs_algos::manifold::tsne::{joint_probabilities, LearningRate, Tsne, TsneInit};
use mlrs_algos::typestate::Fit;
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::{load_npz, OracleCase};

/// Fixture geometry (gen_oracle.py `gen_tsne`: 3 blobs × 16 × 5).
const TSNE_N: usize = 48;
const TSNE_P: usize = 5;
const TSNE_D: usize = 2;
const TRUST_K: usize = 5;

fn fixture(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root is two levels above the crate manifest dir");
    workspace_root.join("tests").join("fixtures").join(name)
}

fn f64_to<F: Pod>(v: f64) -> F {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&(v as f32))),
        8 => *bytemuck::from_bytes::<F>(bytemuck::bytes_of(&v)),
        _ => unreachable!("tsne fixtures are f32/f64 only"),
    }
}

fn f_to_f64<F: Pod>(v: F) -> f64 {
    match std::mem::size_of::<F>() {
        4 => *bytemuck::from_bytes::<f32>(bytemuck::bytes_of(&v)) as f64,
        8 => *bytemuck::from_bytes::<f64>(bytemuck::bytes_of(&v)),
        _ => unreachable!(),
    }
}

/// Exact host squared-Euclidean pairwise distances (f64).
fn pairwise_sq(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut d = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..p {
                let diff = x[i * p + k] - x[j * p + k];
                acc += diff * diff;
            }
            d[i * n + j] = acc;
        }
    }
    d
}

/// sklearn `trustworthiness(X, emb, n_neighbors=k)` host port: rank each
/// point's k embedding-space neighbors by their original-space rank; penalize
/// ranks beyond k. Ties do not occur on the random fixture design.
fn trustworthiness(x: &[f64], emb: &[f64], n: usize, p: usize, d: usize, k: usize) -> f64 {
    // Original-space ranks: argsort each row of the distance matrix with the
    // diagonal at +inf; inverted_index[i][j] = 1-based rank of j from i.
    let dist_x = pairwise_sq(x, n, p);
    let mut inverted = vec![0usize; n * n];
    for i in 0..n {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            let da = if a == i { f64::INFINITY } else { dist_x[i * n + a] };
            let db = if b == i { f64::INFINITY } else { dist_x[i * n + b] };
            da.total_cmp(&db)
        });
        for (r, &j) in idx.iter().enumerate() {
            inverted[i * n + j] = r + 1;
        }
    }
    // Embedding-space k nearest (self excluded).
    let dist_e = pairwise_sq(emb, n, d);
    let mut t_sum = 0.0f64;
    for i in 0..n {
        let mut idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        idx.sort_by(|&a, &b| dist_e[i * n + a].total_cmp(&dist_e[i * n + b]));
        for &j in idx.iter().take(k) {
            let rank = inverted[i * n + j] as f64 - k as f64;
            if rank > 0.0 {
                t_sum += rank;
            }
        }
    }
    1.0 - t_sum * (2.0 / (n as f64 * k as f64 * (2.0 * n as f64 - 3.0 * k as f64 - 1.0)))
}

/// Fit TSNE on the fixture design; return `(embedding f64, kl, n_iter)`.
fn fit_tsne<F>(case: &OracleCase) -> (Vec<f64>, f64, usize)
where
    F: Float + CubeElement + Pod,
{
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let x_host: Vec<F> = case.expect_f64("X").iter().map(|&v| f64_to::<F>(v)).collect();
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(&mut pool, &x_host);
    let perplexity = case.expect_f64("perplexity")[0];

    let model = Tsne::<F>::builder()
        .perplexity(perplexity)
        .init(TsneInit::Pca)
        .build::<F>()
        .expect("TSNE build with valid hyperparameters")
        .fit(&mut pool, &x_dev, None, (TSNE_N, TSNE_P))
        .expect("TSNE::fit on a valid shape");

    let emb: Vec<f64> = model.embedding(&pool).iter().map(|&v| f_to_f64(v)).collect();
    (emb, model.kl_divergence(), model.n_iter())
}

// --- deterministic tier: the joint-P port -----------------------------------

#[test]
fn p_matrix_matches_sklearn() {
    let case = load_npz(&fixture("tsne_f64_seed42.npz")).expect("fixture loads");
    let x = case.expect_f64("X");
    let p_ref = case.expect_f64("P");
    let perplexity = case.expect_f64("perplexity")[0];

    let dsq = pairwise_sq(&x, TSNE_N, TSNE_P);
    let p_got = joint_probabilities(&dsq, TSNE_N, perplexity);

    assert_eq!(p_got.len(), p_ref.len());
    let mut max_abs = 0.0f64;
    for (g, r) in p_got.iter().zip(p_ref.iter()) {
        max_abs = max_abs.max((g - r).abs());
    }
    assert!(
        max_abs <= 1e-5,
        "joint P must match sklearn _joint_probabilities ≤1e-5 (max abs diff {max_abs:e})"
    );
    // The dense P must sum to ~1 and carry a zero diagonal.
    let total: f64 = p_got.iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "P sums to 1, got {total}");
    for i in 0..TSNE_N {
        assert_eq!(p_got[i * TSNE_N + i], 0.0, "P diagonal must be 0");
    }
}

// --- band tier: end-to-end embedding ----------------------------------------

fn run_band<F>(name: &str)
where
    F: Float + CubeElement + Pod,
{
    let case = load_npz(&fixture(name)).expect("fixture loads");
    let x = case.expect_f64("X");
    let kl_ref = case.expect_f64("kl")[0];
    let trust_ref = case.expect_f64("trust")[0];

    let (emb, kl, n_iter) = fit_tsne::<F>(&case);
    assert_eq!(emb.len(), TSNE_N * TSNE_D, "{name}: embedding shape");
    assert!(n_iter < 1000, "{name}: n_iter_ must be < max_iter (0-based last iter)");

    let trust = trustworthiness(&x, &emb, TSNE_N, TSNE_P, TSNE_D, TRUST_K);
    assert!(
        trust >= trust_ref - 0.05,
        "{name}: trustworthiness {trust} below band (sklearn {trust_ref} − 0.05)"
    );
    assert!(
        kl > 0.0 && kl <= kl_ref + 0.25,
        "{name}: kl_divergence_ {kl} outside band (sklearn {kl_ref} + 0.25)"
    );
}

#[test]
fn fit_band_f32() {
    run_band::<f32>("tsne_f32_seed42.npz");
}

#[test]
fn fit_band_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    run_band::<f64>("tsne_f64_seed42.npz");
}

#[test]
fn deterministic_refit() {
    // A SHORT budget (150 iters) is enough to prove bit-determinism — the full
    // 1000-iter convergence is the band tests' job (keeps suite wall-time down;
    // the cpu-runtime per-iteration cost is sync-bound).
    let case = load_npz(&fixture("tsne_f32_seed42.npz")).expect("fixture loads");
    let run = || {
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let x_host: Vec<f32> = case.expect_f64("X").iter().map(|&v| v as f32).collect();
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
        let model = Tsne::<f32>::builder()
            .perplexity(case.expect_f64("perplexity")[0])
            .init(TsneInit::Pca)
            .max_iter(150)
            .build::<f32>()
            .expect("valid build")
            .fit(&mut pool, &x_dev, None, (TSNE_N, TSNE_P))
            .expect("valid fit");
        (model.embedding(&pool), model.kl_divergence())
    };
    let (a, kl_a) = run();
    let (b, kl_b) = run();
    assert_eq!(a, b, "PCA-init TSNE must be bit-deterministic across refits");
    assert_eq!(kl_a, kl_b);
}

#[test]
fn kl_divergence_uses_unexaggerated_p_on_short_fit() {
    // Regression: when the whole fit fits inside the early-exaggeration phase
    // (`max_iter <= EXPLORATION_MAX_ITER = 250`, so phase 2 is skipped),
    // `kl_divergence_` must still report the KL against the UN-exaggerated P,
    // NOT phase 1's KL against `P·early_exaggeration` (which is inflated by
    // ~the exaggeration factor). Compare a short fit against a mid fit that
    // DOES run phase 2 on the SAME PCA-init: both report a true-P KL, so the
    // short fit's value must be sane (same order of magnitude), never ~12×.
    let case = load_npz(&fixture("tsne_f32_seed42.npz")).expect("fixture loads");
    let perplexity = case.expect_f64("perplexity")[0];
    let fit_kl = |max_iter: usize| -> f64 {
        let client = runtime::active_client();
        let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
        let x_host: Vec<f32> = case.expect_f64("X").iter().map(|&v| v as f32).collect();
        let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x_host);
        Tsne::<f32>::builder()
            .perplexity(perplexity)
            .init(TsneInit::Pca)
            .max_iter(max_iter)
            .build::<f32>()
            .expect("valid build")
            .fit(&mut pool, &x_dev, None, (TSNE_N, TSNE_P))
            .expect("valid fit")
            .kl_divergence()
    };
    // A short (phase-2-skipped) fit legitimately has a HIGHER true-P KL than a
    // longer fit — its embedding is simply less converged — so we do NOT
    // require it to match the mid fit's value. What we require is that it be
    // the TRUE-P KL, not phase 1's exaggerated-P KL. For early_exaggeration
    // `ee`, the buggy exaggerated form is `≈ ee·(ln ee + true_kl)`, which for
    // ee=12 is ALWAYS `> ee·ln(ee) ≈ 29.8` (true_kl > 0). Any real true-P KL
    // for this fixture is a few at most (measured: ~1.3 at 100 iters, ~0.1 at
    // 400). A ceiling of 12 cleanly separates the two regimes.
    let kl_short = fit_kl(100); // <= 250 → exploration only
    assert!(kl_short > 0.0 && kl_short.is_finite(), "short-fit KL must be a real value");
    assert!(
        kl_short < 12.0,
        "short-fit KL {kl_short} must be a true-P KL (a few at most), not phase 1's \
         exaggerated-P KL (~30+ for early_exaggeration=12)"
    );
}

// --- validation --------------------------------------------------------------

#[test]
fn build_validation() {
    fn expect_build_err(b: mlrs_algos::manifold::tsne::TsneBuilder) -> BuildError {
        match b.build::<f32>() {
            Err(e) => e,
            Ok(_) => panic!("expected a BuildError"),
        }
    }
    assert!(matches!(
        expect_build_err(Tsne::<f32>::builder().perplexity(0.0)),
        BuildError::InvalidPerplexity { .. }
    ));
    assert!(matches!(
        expect_build_err(Tsne::<f32>::builder().early_exaggeration(0.5)),
        BuildError::InvalidEarlyExaggeration { .. }
    ));
    assert!(matches!(
        expect_build_err(Tsne::<f32>::builder().learning_rate(LearningRate::Value(0.0))),
        BuildError::InvalidLearningRate { .. }
    ));
    assert!(matches!(
        expect_build_err(Tsne::<f32>::builder().max_iter(0)),
        BuildError::InvalidMaxIter { .. }
    ));
    assert!(matches!(
        expect_build_err(Tsne::<f32>::builder().n_components(0)),
        BuildError::InvalidNComponents { .. }
    ));
}

#[test]
fn fit_validation() {
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // perplexity >= n_samples → typed AlgoError BEFORE any launch.
    let x: Vec<f32> = (0..12).map(|v| v as f32).collect(); // 4 × 3
    let x_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut pool, &x);
    let err = Tsne::<f32>::builder()
        .perplexity(4.0)
        .build::<f32>()
        .expect("valid build")
        .fit(&mut pool, &x_dev, None, (4, 3))
        .err()
        .expect("perplexity >= n_samples must be rejected at fit");
    assert!(
        matches!(err, AlgoError::InvalidPerplexity { n_samples: 4, .. }),
        "expected InvalidPerplexity, got {err:?}"
    );
}
