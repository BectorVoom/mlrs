//! MANIFOLD-PERSIST / TS-PERSIST (prototype) — safetensors save/load round-trips
//! for `Tsne`, `Umap` and `Arima`.
//!
//! Two small containers, and both of them turn on the same distinction: what a
//! saved model has to be able to DO afterwards.
//!
//! `Tsne` has no out-of-sample extension — sklearn's `TSNE` exposes
//! `fit_transform` and no `transform` — so its file is the embedding plus the
//! descent diagnostics, and a faithful round-trip means every attribute comes
//! back. `Umap` DOES generalize, and keeps the training matrix to do it, so its
//! file carries `_raw_data` too and `umap_keeps_its_training_matrix` gates that
//! the reloaded model can still transform.
//!
//! `Arima`'s file is a RESUMPTION point: `forecast` continues the Kalman
//! recursion from `final_state_`/`final_cov_`, so a file that stored only the
//! coefficients would load, report every information criterion correctly, and
//! forecast from a zero state. `arima_roundtrip_preserves_the_forecast` is the
//! only assertion that catches that — the attribute comparisons all pass on such
//! a file.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::manifold::manifold_persist::{
    AlignedBytes, LoadModel, ManifoldFile, ManifoldWriter, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::manifold::tsne::{Tsne, TsneInit, TsneMethod};
use mlrs_algos::manifold::umap::{Init, Metric as UmapMetric, Umap};
use mlrs_algos::preprocessing::MaxAbsScaler;
use mlrs_algos::timeseries::arima::Arima;
use mlrs_algos::timeseries::ts_persist::{
    AlignedBytes as TsBytes, LoadModel as TsLoad, PersistError as TsError, SaveModel as TsSave,
    TensorRef as TsTensorRef, TimeSeriesWriter,
};
use mlrs_algos::typestate::{Fit, Fitted};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

const N_SAMPLES: usize = 24;
const N_FEATURES: usize = 4;
const N_COMPONENTS: usize = 2;

fn fixture<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES * N_FEATURES)
        .map(|i| {
            let v = ((i * 53) % 89) as f64 / 22.0 - 2.0;
            mlrs_core::f64_to_host::<F>(v)
        })
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

/// A `Tsne` with a SHORT horizon. The descent's endpoint is irrelevant to the
/// container, and a long one would only make the fixture slow — but the
/// diagnostics (`kl_divergence_`, `n_iter_`) still have to be non-trivial for
/// the round-trip comparison to mean anything, which `max_iter = 260` gives.
fn fit_tsne<F>(p: &mut BufferPool<ActiveRuntime>) -> Tsne<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    Tsne::<F>::builder()
        .n_components(N_COMPONENTS)
        .perplexity(5.0)
        .max_iter(260)
        .method(TsneMethod::Exact)
        .init(TsneInit::Random)
        .seed(7)
        .build::<F>()
        .expect("Tsne builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("Tsne fits the fixture")
}

fn fit_umap<F>(p: &mut BufferPool<ActiveRuntime>) -> Umap<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    Umap::<F>::builder()
        .n_neighbors(5)
        .n_components(N_COMPONENTS)
        .metric(UmapMetric::Minkowski { p: 3.0 })
        .init(Init::Random)
        .random_state(Some(11))
        .n_epochs(Some(20))
        .build::<F>()
        .expect("Umap builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("Umap fits the fixture")
}

/// A deterministic series with enough structure for an ARIMA(2,1,1) to have
/// something to fit.
fn series<F: Pod>() -> Vec<F> {
    let mut out = Vec::with_capacity(60);
    let (mut a, mut b) = (0.0f64, 0.5f64);
    for i in 0..60 {
        let v = 0.6 * a - 0.2 * b + ((i * 37) % 13) as f64 * 0.05;
        out.push(mlrs_core::f64_to_host::<F>(v));
        b = a;
        a = v;
    }
    out
}

fn fit_arima<F>(p: &mut BufferPool<ActiveRuntime>) -> Arima<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let host = series::<F>();
    let y: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(p, &host);
    Arima::<F>::builder()
        .order(2, 1, 1)
        .build::<F>()
        .expect("Arima builds")
        .fit(p, &y, host.len())
        .expect("Arima fits the series")
}

// ---------------------------------------------------------------------------
// t-SNE
// ---------------------------------------------------------------------------

#[test]
fn tsne_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("tsne.safetensors");
    let mut p = pool();

    let fitted = fit_tsne::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: Tsne<f32, Fitted> = Tsne::load(&mut p, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits, so any
    // drift at all is a defect in the container, not rounding.
    assert_eq!(
        loaded.embedding(&p),
        fitted.embedding(&p),
        "embedding_ must round-trip exactly"
    );
    // The descent diagnostics. Neither is recoverable from the embedding without
    // the training data, which this file does not hold.
    assert_eq!(
        loaded.kl_divergence(),
        fitted.kl_divergence(),
        "kl_divergence_"
    );
    assert_eq!(loaded.n_iter(), fitted.n_iter(), "n_iter_");
    assert_eq!(
        loaded.n_features_in(),
        fitted.n_features_in(),
        "n_features_in_ is not implied by any tensor's shape here — the file \
         holds no training matrix"
    );
    assert!(
        fitted.kl_divergence() > 0.0,
        "the fixture must produce a real divergence, or the gate proves nothing"
    );
}

#[test]
fn tsne_non_default_params_roundtrip() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    // save → load → save is byte-stable, which is how the fifteen private
    // hyperparameters are gated at once. `method`, `init`, `learning_rate` and
    // `metric` are all enum-shaped with no public accessor, and none is
    // recoverable from the embedding — a `load` that dropped one would still
    // pass every comparison above.
    let fitted = fit_tsne::<f32>(&mut p);
    fitted.save(&p, &first).expect("save succeeds");
    let loaded: Tsne<f32, Fitted> = Tsne::load(&mut p, &first).expect("load succeeds");
    loaded.save(&p, &second).expect("re-save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "save→load→save must be byte-stable, or a hyperparameter was dropped"
    );
}

#[test]
fn an_unrecognised_tsne_method_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-method.safetensors");
    let mut p = pool();

    // A file from a hypothetical future build that grew a third gradient
    // objective. It must fail by NAME rather than fall back to the `barnes_hut`
    // default — the two methods compute different gradients, so a silent
    // fallback would be a different model with nothing to signal it.
    let embedding = vec![0.0f32; N_SAMPLES * N_COMPONENTS];
    let mut w = ManifoldWriter::new("tsne");
    w.scalar_usize("param:n_components", N_COMPONENTS);
    w.scalar_f64("param:perplexity", 5.0);
    w.scalar_f64("param:early_exaggeration", 12.0);
    w.scalar_str("param:learning_rate", "auto");
    w.scalar_usize("param:max_iter", 260);
    w.scalar_usize("param:n_iter_without_progress", 300);
    w.scalar_f64("param:min_grad_norm", 1e-7);
    w.scalar_str("param:metric", "euclidean");
    w.scalar_str("param:init", "random");
    w.scalar_usize("param:verbose", 0);
    w.scalar_u64("param:seed", 7);
    // The one bad key.
    w.scalar_str("param:method", "landmark");
    w.scalar_f64("param:angle", 0.5);
    w.scalar_str("param:device", "auto");
    w.scalar_usize("n_iter_", 260);
    w.scalar_usize("n_features_in_", N_FEATURES);
    w.tensor(
        "embedding_",
        TensorRef::floats(&embedding, vec![N_SAMPLES, N_COMPONENTS]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match Tsne::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an unrecognised method must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::BadMetadata { key } if *key == "param:method"),
        "expected BadMetadata naming param:method, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// UMAP
// ---------------------------------------------------------------------------

#[test]
fn umap_keeps_its_training_matrix() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("umap.safetensors");
    let mut p = pool();

    let fitted = fit_umap::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: Umap<f32, Fitted> = Umap::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.embedding(&p),
        fitted.embedding(&p),
        "embedding_ must round-trip exactly"
    );

    // `_raw_data` is what separates this estimator from `Tsne`: UMAP can embed a
    // row it never saw, and does so by scanning the retained training matrix.
    // The tensor has no public accessor, so its presence is checked in the file
    // and its CONTENT through the round-tripped transform below.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = ManifoldFile::parse(&raw, "umap").expect("parse succeeds");
    let view = file
        .tensor("_raw_data")
        .expect("the training matrix is stored");
    assert_eq!(
        view.shape(),
        &[N_SAMPLES, N_FEATURES],
        "_raw_data must hold the whole training matrix"
    );
    assert!(
        bytemuck::try_cast_slice::<u8, f32>(view.data()).is_ok(),
        "'_raw_data' must be reinterpretable without a copy"
    );
}

#[test]
fn umap_non_default_params_roundtrip() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    // The fixture uses `minkowski(p=3)`, which is the one metric carrying a
    // payload — so this also gates that the exponent survives, since a
    // `'minkowski'` without its `p` would be rejected on load rather than
    // silently becoming Euclidean.
    let fitted = fit_umap::<f32>(&mut p);
    fitted.save(&p, &first).expect("save succeeds");
    let loaded: Umap<f32, Fitted> = Umap::load(&mut p, &first).expect("load succeeds");
    loaded.save(&p, &second).expect("re-save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "save→load→save must be byte-stable, or a hyperparameter was dropped"
    );
}

#[test]
fn manifold_siblings_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("umap.safetensors");
    let mut p = pool();

    // Both files hold an `embedding_` of the same shape and dtype under the same
    // name; a `Umap` file differs only by carrying `_raw_data`. So a `Umap` file
    // loaded as a `Tsne` would pass every geometry check and quietly discard the
    // training matrix that makes it able to transform.
    fit_umap::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match Tsne::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a umap file must not load as a tsne"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "tsne" && found == "umap"
        ),
        "expected WrongEstimator, got {err:?}"
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

    let err = match Umap::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-prep file must not load as a manifold model"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-manifold"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// ARIMA — the resumption point
// ---------------------------------------------------------------------------

#[test]
fn arima_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("arima.safetensors");
    let mut p = pool();

    let fitted = fit_arima::<f64>(&mut p);
    TsSave::save(&fitted, &p, &path).expect("save succeeds");
    let loaded: Arima<f64, Fitted> = TsLoad::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.order(), fitted.order(), "(p, d, q)");
    assert_eq!(loaded.ar(), fitted.ar(), "arparams_");
    assert_eq!(loaded.ma(), fitted.ma(), "maparams_");
    assert_eq!(loaded.sigma2(), fitted.sigma2(), "sigma2_");
    assert_eq!(loaded.loglik(), fitted.loglik(), "loglik_");
    assert_eq!(loaded.aic(), fitted.aic(), "aic_");
    assert_eq!(loaded.aicc(), fitted.aicc(), "aicc_");
    assert_eq!(loaded.bic(), fitted.bic(), "bic_");
    assert_eq!(loaded.nobs(), fitted.nobs(), "nobs_");
    assert_eq!(loaded.converged(), fitted.converged(), "converged_");
    assert_eq!(loaded.final_cov(), fitted.final_cov(), "final_cov_");
}

#[test]
fn arima_roundtrip_preserves_the_forecast() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("arima.safetensors");
    let mut p = pool();

    // The gate none of the attribute comparisons can replace. `forecast`
    // continues the Kalman recursion from `final_state_`/`final_cov_` and
    // un-differences through `diff_last_`, so a file that stored only the
    // coefficients and the information criteria would pass every assertion in
    // `arima_roundtrip_is_bit_exact` and forecast from a zero state.
    let fitted = fit_arima::<f64>(&mut p);
    let before = fitted.forecast(8);
    TsSave::save(&fitted, &p, &path).expect("save succeeds");
    let loaded: Arima<f64, Fitted> = TsLoad::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.forecast(8),
        before,
        "the reloaded model must forecast identically — the file is a resumption \
         point, not just a description"
    );
    assert!(
        before.iter().all(|v| v.is_finite()),
        "the fixture must produce a real forecast, or the gate proves nothing"
    );
    assert!(
        before.iter().any(|&v| v != 0.0),
        "a forecast of all zeros would make the comparison vacuous — that is \
         exactly what a dropped final_state_ would produce"
    );
}

#[test]
fn an_arima_order_disagreeing_with_its_arrays_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-order.safetensors");
    let mut p = pool();

    // `param:p` says 3 but `arparams_` holds 2. Neither half is wrong on its own,
    // so only the cross-check catches it — and `forecast` indexes the
    // coefficient block by lag without a bound of its own, so a short array
    // would read past its end on the first horizon step.
    let ar = [0.5f64, -0.2];
    let ma: [f64; 0] = [];
    let diff = [1.0f64];
    let state = [0.0f64, 0.0];
    let cov = [1.0f64, 0.0, 0.0, 1.0];
    let mut w = TimeSeriesWriter::new("arima");
    w.scalar_usize("param:p", 3);
    w.scalar_usize("param:d", 1);
    w.scalar_usize("param:q", 0);
    for (k, v) in [
        ("sigma2_", 1.0),
        ("loglik_", -1.0),
        ("aic_", 1.0),
        ("aicc_", 1.0),
        ("bic_", 1.0),
    ] {
        w.scalar_f64(k, v);
    }
    w.scalar_usize("nobs_", 10);
    w.scalar_bool("converged_", true);
    w.tensor("arparams_", TsTensorRef::f64s(&ar, vec![2]).expect("ok"));
    w.tensor("maparams_", TsTensorRef::f64s(&ma, vec![0]).expect("ok"));
    w.tensor("diff_last_", TsTensorRef::f64s(&diff, vec![1]).expect("ok"));
    w.tensor(
        "final_state_",
        TsTensorRef::f64s(&state, vec![2]).expect("ok"),
    );
    w.tensor(
        "final_cov_",
        TsTensorRef::f64s(&cov, vec![2, 2]).expect("ok"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match <Arima<f64, Fitted> as TsLoad>::load(&mut p, &path) {
        Ok(_) => panic!("an order disagreeing with its arrays must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, TsError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_zero_order_arima_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("arima-000.safetensors");
    let mut p = pool();

    // `p = 0` and `q = 0` are ordinary ARIMA orders, so an EMPTY `arparams_` /
    // `maparams_` is a legitimate value rather than a malformed tensor. This is
    // the case a reader that treated empty-as-absent would get wrong.
    let host = series::<f64>();
    let y: DeviceArray<ActiveRuntime, f64> = DeviceArray::from_host(&mut p, &host);
    let fitted = Arima::<f64>::builder()
        .order(0, 1, 0)
        .build::<f64>()
        .expect("Arima builds")
        .fit(&mut p, &y, host.len())
        .expect("Arima(0,1,0) fits");
    assert!(
        fitted.ar().is_empty() && fitted.ma().is_empty(),
        "the fixture must have empty coefficient blocks"
    );

    TsSave::save(&fitted, &p, &path).expect("save succeeds");
    let loaded: Arima<f64, Fitted> = TsLoad::load(&mut p, &path).expect("load succeeds");
    assert_eq!(loaded.order(), (0, 1, 0), "the order must round-trip");
    assert!(
        loaded.ar().is_empty() && loaded.ma().is_empty(),
        "empty coefficient blocks must stay empty"
    );
    assert_eq!(
        loaded.forecast(4),
        fitted.forecast(4),
        "a zero-order model must still forecast identically"
    );
}

#[test]
fn arima_saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    // RAW BYTES: a model file must be a deterministic function of the model.
    // This is the gate on the `third_party/safetensors` `BTreeMap` patch —
    // `Arima` carries eleven scalars, so a randomly-seeded header map is very
    // likely to reorder one.
    let fitted = fit_arima::<f64>(&mut p);
    TsSave::save(&fitted, &p, &first).expect("save succeeds");
    TsSave::save(&fitted, &p, &second).expect("save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

#[test]
fn an_arima_file_is_rejected_by_the_manifold_reader() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("arima.safetensors");
    let mut p = pool();
    let fitted = fit_arima::<f64>(&mut p);
    TsSave::save(&fitted, &p, &path).expect("save succeeds");

    // The cross-FAMILY gate. Only the `format` discriminator separates the
    // containers, and it is checked before any tensor is fetched.
    let err = match Tsne::<f64, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-timeseries file must not load as a manifold model"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-manifold"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("umap.safetensors");
    let mut p = pool();
    fit_umap::<f32>(&mut p)
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
