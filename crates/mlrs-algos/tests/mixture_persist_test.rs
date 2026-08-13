//! MIX-PERSIST (prototype) — safetensors save/load round-trips for the two
//! mixture estimators: `GaussianMixture` and `BayesianGaussianMixture`.
//!
//! Three things about this family need gating beyond the container boilerplate.
//!
//! The FLAT LENGTH of `covariances_` and `precisions_cholesky_` depends on
//! `covariance_type` — `full` is `k·d·d`, `tied` is `d·d`, `diag` is `k·d`,
//! `spherical` is `k`. So `every_covariance_type_roundtrips` runs all four, and
//! `a_covariance_block_of_the_wrong_length_is_rejected` checks that a file
//! whose block does not match its declared type is refused rather than read past
//! its end on the first `score_samples`.
//!
//! `precisions_cholesky_` is DERIVABLE from `covariances_` and is stored anyway.
//! `the_precisions_are_stored_not_recomputed` is the gate: a hand-written file
//! pairing one model's covariances with another's Cholesky factors must score
//! using the STORED factors, because recomputing them on load would be an
//! `O(k·d³)` factorization that a near-singular covariance will not reproduce
//! bit for bit.
//!
//! Everything is `F64` regardless of the estimator's `F`, because the EM loop
//! is `f64` on every backend and a reloaded mixture is a valid `warm_start`
//! continuation point. `the_file_is_f64_at_both_model_widths` gates that so it
//! cannot be "optimized" into a silent precision loss.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::mixture::bayesian_gaussian_mixture::BayesianGaussianMixture;
use mlrs_algos::mixture::gaussian_mixture::GaussianMixture;
use mlrs_algos::mixture::mixture_persist::{
    AlignedBytes, LoadModel, MixtureFile, MixtureWriter, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::preprocessing::MaxAbsScaler;
use mlrs_algos::typestate::{Fit, Fitted};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

const N_SAMPLES: usize = 30;
const N_FEATURES: usize = 3;
const K: usize = 2;

/// Two well-separated Gaussian blobs. The separation keeps the EM loop away from
/// the multi-basin behavior that would make a round-trip comparison depend on
/// the fit rather than on the file.
fn fixture<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES)
        .flat_map(|i| {
            let base = if i % 2 == 0 { 0.0 } else { 10.0 };
            let j = (i / 2) as f64 * 0.11;
            [base + j, base - j * 0.5, base + j * 0.25]
        })
        .map(mlrs_core::f64_to_host::<F>)
        .collect()
}

fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

fn upload<F: Float + CubeElement + Pod>(
    p: &mut BufferPool<ActiveRuntime>,
) -> DeviceArray<ActiveRuntime, F> {
    DeviceArray::from_host(p, &fixture::<F>())
}

fn fit_gmm<F>(
    p: &mut BufferPool<ActiveRuntime>,
    covariance_type: &str,
) -> GaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    GaussianMixture::<F>::builder()
        .n_components(K)
        .covariance_type(covariance_type.to_string())
        .random_state(Some(3))
        .max_iter(30)
        .build::<F>()
        .expect("GaussianMixture builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("GaussianMixture fits the fixture")
}

fn fit_bgm<F>(
    p: &mut BufferPool<ActiveRuntime>,
    covariance_type: &str,
) -> BayesianGaussianMixture<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    BayesianGaussianMixture::<F>::builder()
        .n_components(K)
        .covariance_type(covariance_type.to_string())
        .random_state(Some(3))
        .max_iter(30)
        .build::<F>()
        .expect("BayesianGaussianMixture builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("BayesianGaussianMixture fits the fixture")
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn every_covariance_type_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // The flat length of both parameter blocks depends on this argument, so all
    // four parameterizations have to be exercised — a reader that assumed one
    // shape would pass three of these and fail the fourth, or worse, read past
    // the end of a shorter block.
    for ct in ["full", "tied", "diag", "spherical"] {
        let path = dir.path().join(format!("gmm-{ct}.safetensors"));
        let fitted = fit_gmm::<f32>(&mut p, ct);
        fitted.save(&p, &path).expect("save succeeds");
        let loaded: GaussianMixture<f32, Fitted> =
            GaussianMixture::load(&mut p, &path).expect("load succeeds");

        // `==` rather than a tolerance: the file stores the exact IEEE bits.
        assert_eq!(loaded.weights(), fitted.weights(), "{ct}: weights_");
        assert_eq!(loaded.means(), fitted.means(), "{ct}: means_");
        assert_eq!(
            loaded.covariances(),
            fitted.covariances(),
            "{ct}: covariances_"
        );
        assert_eq!(
            loaded.precisions_cholesky(),
            fitted.precisions_cholesky(),
            "{ct}: precisions_cholesky_"
        );
        assert_eq!(loaded.converged(), fitted.converged(), "{ct}: converged_");
        assert_eq!(loaded.n_iter(), fitted.n_iter(), "{ct}: n_iter_");
        assert_eq!(
            loaded.lower_bound(),
            fitted.lower_bound(),
            "{ct}: lower_bound_"
        );
        assert_eq!(
            loaded.lower_bounds(),
            fitted.lower_bounds(),
            "{ct}: the lower-bound TRACE is what shows whether EM plateaued or \
             was cut off by max_iter, and it is not recoverable from anything else"
        );
    }
}

#[test]
fn bayesian_mixture_roundtrips_its_posterior_and_priors() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bgm.safetensors");
    let mut p = pool();

    let fitted = fit_bgm::<f32>(&mut p, "full");
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: BayesianGaussianMixture<f32, Fitted> =
        BayesianGaussianMixture::load(&mut p, &path).expect("load succeeds");

    // The seven posterior blocks. Four of them are length-`k` `f64` vectors, so
    // comparing them INDIVIDUALLY is what would catch a positional swap in the
    // shared reader — a bulk comparison would not.
    assert_eq!(
        loaded.weight_concentration(),
        fitted.weight_concentration(),
        "weight_concentration_"
    );
    assert_eq!(
        loaded.mean_precision(),
        fitted.mean_precision(),
        "mean_precision_"
    );
    assert_eq!(loaded.means(), fitted.means(), "means_");
    assert_eq!(
        loaded.degrees_of_freedom(),
        fitted.degrees_of_freedom(),
        "degrees_of_freedom_"
    );
    assert_eq!(loaded.covariances(), fitted.covariances(), "covariances_");
    assert_eq!(
        loaded.precisions_cholesky(),
        fitted.precisions_cholesky(),
        "precisions_cholesky_"
    );
    assert_eq!(loaded.converged(), fitted.converged(), "converged_");
    assert_eq!(loaded.lower_bound(), fitted.lower_bound(), "lower_bound_");
}

#[test]
fn the_file_is_f64_at_both_model_widths() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let narrow = dir.path().join("f32.safetensors");
    let mut p = pool();

    // The DELIBERATE exception to the "stored dtype is the model's dtype" rule,
    // gated so it cannot be "optimized" into a silent precision loss. The EM
    // loop is `f64` on every backend and a reloaded mixture is a valid
    // `warm_start` continuation point, so a `precisions_cholesky_` rounded to
    // `f32` would restart the loop from a factorization the saved model never
    // computed.
    fit_gmm::<f32>(&mut p, "full")
        .save(&p, &narrow)
        .expect("save succeeds");

    let raw = AlignedBytes::read(&narrow).expect("read succeeds");
    let file = MixtureFile::parse(&raw, "gaussian_mixture").expect("parse succeeds");
    for name in ["weights_", "means_", "covariances_", "precisions_cholesky_"] {
        let view = file.tensor(name).expect("the tensor is present");
        assert_eq!(
            view.dtype(),
            safetensors::Dtype::F64,
            "'{name}' must be stored as F64 even for a GaussianMixture<f32>"
        );
        // And the `AlignedBytes` claim over them.
        assert!(
            bytemuck::try_cast_slice::<u8, f64>(view.data()).is_ok(),
            "'{name}' must be reinterpretable as &[f64] without a copy"
        );
    }
}

// ---------------------------------------------------------------------------
// The derivable-but-stored precisions
// ---------------------------------------------------------------------------

#[test]
fn the_precisions_are_stored_not_recomputed() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("mixed.safetensors");
    let mut p = pool();

    // `precisions_cholesky_` IS the Cholesky factor of the inverse covariances,
    // so a loader COULD recompute it. It does not, and this is the gate: a
    // hand-written file pairing one model's `covariances_` with a DIFFERENT
    // model's `precisions_cholesky_` must come back holding the stored factors,
    // not ones re-derived from the covariances.
    //
    // The failure mode it stands for is not this artificial pairing but the
    // ordinary one: an `O(k·d³)` factorization of a near-singular covariance is
    // not bit-reproducible, so a recomputing loader would score subtly
    // differently from the model that was saved.
    let a = fit_gmm::<f32>(&mut p, "full");
    let b = fit_gmm::<f32>(&mut p, "diag");
    let cov_a: Vec<f64> = a.covariances().iter().map(|&v| f64::from(v)).collect();
    let prec_a: Vec<f64> = a
        .precisions_cholesky()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let weights: Vec<f64> = a.weights().iter().map(|&v| f64::from(v)).collect();
    let means: Vec<f64> = a.means().iter().map(|&v| f64::from(v)).collect();
    // Scale the factors so they are recognisably NOT what the covariances imply.
    let prec_scaled: Vec<f64> = prec_a.iter().map(|v| v * 1.5).collect();
    assert_ne!(prec_a, prec_scaled, "the two blocks must differ");
    let _ = b;

    let mut w = MixtureWriter::new("gaussian_mixture");
    w.scalar_usize("param:n_components", K);
    w.scalar_str("param:covariance_type", "full");
    w.scalar_f64("param:tol", 1e-3);
    w.scalar_f64("param:reg_covar", 1e-6);
    w.scalar_usize("param:max_iter", 30);
    w.scalar_usize("param:n_init", 1);
    w.scalar_str("param:init_params", "kmeans");
    w.scalar_bool("param:warm_start", false);
    w.scalar_usize("param:verbose", 0);
    w.scalar_usize("param:verbose_interval", 10);
    w.scalar_str("param:device", "auto");
    w.scalar_bool("converged_", true);
    w.scalar_usize("n_iter_", 5);
    w.scalar_f64("lower_bound_", -1.0);
    w.scalar_usize("n_features_in_", N_FEATURES);
    w.scalar_usize("n_samples_", N_SAMPLES);
    w.tensor("weights_", TensorRef::f64s(&weights, vec![K]).expect("ok"));
    w.tensor(
        "means_",
        TensorRef::f64s(&means, vec![K, N_FEATURES]).expect("ok"),
    );
    w.tensor(
        "covariances_",
        TensorRef::f64s(&cov_a, vec![cov_a.len()]).expect("ok"),
    );
    w.tensor(
        "precisions_cholesky_",
        TensorRef::f64s(&prec_scaled, vec![prec_scaled.len()]).expect("ok"),
    );
    let trace = [-1.0f64];
    w.tensor(
        "lower_bounds_",
        TensorRef::f64s(&trace, vec![1]).expect("ok"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed");

    let loaded: GaussianMixture<f32, Fitted> =
        GaussianMixture::load(&mut p, &path).expect("load succeeds");
    let got: Vec<f64> = loaded
        .precisions_cholesky()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    for (i, (&want, &have)) in prec_scaled.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            want as f32, have as f32,
            "precisions_cholesky_[{i}] must come from the FILE, not be recomputed \
             from covariances_"
        );
    }
}

#[test]
fn a_covariance_block_of_the_wrong_length_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("wrong-len.safetensors");
    let mut p = pool();

    // A `tied` model's `covariances_` is `d·d`; this file declares `tied` and
    // supplies a `full`-length `k·d·d` block. Both halves are individually
    // well-formed — the string is a valid covariance type and the array is a
    // valid array — so only the cross-check against `param_len` catches it, and
    // without it the first `score_samples` would read past the end of the shared
    // matrix.
    let weights = [0.5f64; K];
    let means = [0.0f64; K * N_FEATURES];
    let too_long = vec![1.0f64; K * N_FEATURES * N_FEATURES];
    let mut w = MixtureWriter::new("gaussian_mixture");
    w.scalar_usize("param:n_components", K);
    w.scalar_str("param:covariance_type", "tied");
    w.scalar_f64("param:tol", 1e-3);
    w.scalar_f64("param:reg_covar", 1e-6);
    w.scalar_usize("param:max_iter", 30);
    w.scalar_usize("param:n_init", 1);
    w.scalar_str("param:init_params", "kmeans");
    w.scalar_bool("param:warm_start", false);
    w.scalar_usize("param:verbose", 0);
    w.scalar_usize("param:verbose_interval", 10);
    w.scalar_str("param:device", "auto");
    w.scalar_bool("converged_", true);
    w.scalar_usize("n_iter_", 5);
    w.scalar_f64("lower_bound_", -1.0);
    w.scalar_usize("n_features_in_", N_FEATURES);
    w.scalar_usize("n_samples_", N_SAMPLES);
    w.tensor("weights_", TensorRef::f64s(&weights, vec![K]).expect("ok"));
    w.tensor(
        "means_",
        TensorRef::f64s(&means, vec![K, N_FEATURES]).expect("ok"),
    );
    w.tensor(
        "covariances_",
        TensorRef::f64s(&too_long, vec![too_long.len()]).expect("ok"),
    );
    w.tensor(
        "precisions_cholesky_",
        TensorRef::f64s(&too_long, vec![too_long.len()]).expect("ok"),
    );
    let trace = [-1.0f64];
    w.tensor(
        "lower_bounds_",
        TensorRef::f64s(&trace, vec![1]).expect("ok"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match GaussianMixture::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a covariance block of the wrong length must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// The format claims and rejection
// ---------------------------------------------------------------------------

#[test]
fn saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    // RAW BYTES: a model file must be a deterministic function of the model.
    // This is the gate on the `third_party/safetensors` `BTreeMap` patch, and
    // `GaussianMixture` is a strong subject — it carries seventeen scalars, so a
    // randomly-seeded header map is all but certain to reorder one.
    let fitted = fit_gmm::<f32>(&mut p, "full");
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

#[test]
fn a_preprocessing_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("scaler.safetensors");
    let mut p = pool();

    let x: DeviceArray<ActiveRuntime, f32> = upload::<f32>(&mut p);
    MaxAbsScaler::<f32>::new()
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("MaxAbsScaler fits")
        .save(&p, &path)
        .expect("save succeeds");

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match GaussianMixture::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-prep file must not load as a mixture"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-mixture"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn sibling_estimators_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bgm.safetensors");
    let mut p = pool();

    // Both files hold `means_` and `covariances_` of the same shapes and dtypes
    // under the same names. The Bayesian model's density is parameterized by a
    // POSTERIOR the frequentist one has no notion of, so a cross-load would
    // score every sample differently with nothing structural to signal it.
    fit_bgm::<f32>(&mut p, "full")
        .save(&p, &path)
        .expect("save succeeds");

    let err = match GaussianMixture::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a bayesian file must not load as a gaussian_mixture"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "gaussian_mixture" && found == "bayesian_gaussian_mixture"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gmm.safetensors");
    let mut p = pool();
    fit_gmm::<f32>(&mut p, "full")
        .save(&p, &path)
        .expect("save succeeds");

    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .expect("the scratch directory is readable")
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .filter(|n| n.to_string_lossy().contains("mlrs-tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a successful save must leave no temporary file, found {leftovers:?}"
    );
    assert!(path.exists(), "the model file must exist");
}
