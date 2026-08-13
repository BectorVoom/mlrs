//! KERNEL-PERSIST (prototype) — safetensors save/load round-trips for the two
//! kernel methods: `KernelRidge` and `KernelDensity`.
//!
//! These are the largest files mlrs writes for a given problem, and for a reason
//! that is not a defect: a kernel method evaluates against every training row at
//! predict time, so `X_fit_` is not a fitting artifact but the model itself.
//! `the_training_set_is_the_model` measures that directly.
//!
//! The family's sharp case is the RESOLUTION pair. `KernelRidge`'s
//! `gamma=None` resolves to `1/n_features` at fit, and `KernelDensity`'s
//! `bandwidth='scott'` resolves against `n_samples`/`n_features`. Both files
//! store the REQUEST and the RESOLVED value rather than re-running the rule at
//! load, so that a later change to either rule cannot silently give every
//! previously-saved model a different kernel.
//! `a_reloaded_model_uses_the_stored_resolution` is the gate that holds that
//! line, built the same way the projection family's RNG gate is: a hand-written
//! file whose recorded request would resolve to something OTHER than what it
//! stores.
//!
//! The two estimators also share the `param:kernel` KEY while speaking different
//! vocabularies that OVERLAP on `"linear"` — a `KernelRidge` linear kernel is
//! `X·Yᵀ`, a `KernelDensity` linear kernel is `1 − dist/h`. So the `estimator`
//! discriminator is not a formality here: it is what establishes which
//! vocabulary applies before either is parsed, and
//! `sibling_estimators_do_not_cross_load` proves it.
//!
//! The remaining gates are the standard container set: bit-exact round-trips,
//! prediction/score equivalence, the two dtype-tag claims, zero-copy loading,
//! byte-level determinism, and the rejection set over hand-built headers. The
//! file is untrusted input (T-04-01-01).
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::density::kernel_density::{BandwidthSpec, KdKernel, KernelDensity};
use mlrs_algos::kernel_persist::{
    AlignedBytes, KernelFile, KernelWriter, LoadModel, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::kernel_ridge::kernel_ridge::{KernelKind, KernelRidge};
use mlrs_algos::preprocessing::MaxAbsScaler;
use mlrs_algos::typestate::{Fit, Fitted, Predict, ScoreSamples};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

const N_SAMPLES: usize = 20;
const N_FEATURES: usize = 4;

/// A deterministic fixture. Hand-generated rather than seeded-random: a
/// persistence round-trip is exact or broken, and an RNG would only add a way
/// for the two arms to disagree for reasons unrelated to the file.
fn fixture<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES * N_FEATURES)
        .map(|i| {
            let v = ((i * 29) % 61) as f64 / 30.0 - 1.0;
            mlrs_core::f64_to_host::<F>(v)
        })
        .collect()
}

fn targets<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES)
        .map(|i| mlrs_core::f64_to_host::<F>(((i * 13) % 7) as f64 * 0.5 - 1.0))
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

/// A `KernelRidge` with every hyperparameter off its default, so a round-trip
/// that drops one is visible. `poly` is the one kernel that reads ALL FOUR of
/// gamma/degree/coef0/alpha, which makes it the right subject for the
/// resolution gates below.
fn fit_kr<F>(p: &mut BufferPool<ActiveRuntime>, gamma: Option<f64>) -> KernelRidge<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    let y: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(p, &targets::<F>());
    KernelRidge::<F>::builder()
        .kernel(KernelKind::Poly)
        .alpha(0.7)
        .gamma(gamma)
        .degree(3.0)
        .coef0(0.5)
        .build::<F>()
        .expect("KernelRidge builds")
        .fit(p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("KernelRidge fits the fixture")
}

fn fit_kd<F>(
    p: &mut BufferPool<ActiveRuntime>,
    kernel: KdKernel,
    bandwidth: BandwidthSpec,
) -> KernelDensity<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    KernelDensity::<F>::builder()
        .kernel(kernel)
        .bandwidth(bandwidth)
        .build::<F>()
        .expect("KernelDensity builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("KernelDensity fits the fixture")
}

fn predictions<F>(p: &mut BufferPool<ActiveRuntime>, m: &KernelRidge<F, Fitted>) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    m.predict(p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict succeeds on the training geometry")
        .to_host(p)
}

fn scores<F>(p: &mut BufferPool<ActiveRuntime>, m: &KernelDensity<F, Fitted>) -> Vec<F>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    m.score_samples(p, &x, (N_SAMPLES, N_FEATURES))
        .expect("score_samples succeeds on the training geometry")
        .to_host(p)
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn kernel_ridge_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kr.safetensors");
    let mut p = pool();

    let fitted = fit_kr::<f32>(&mut p, Some(0.3));
    let before = predictions(&mut p, &fitted);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: KernelRidge<f32, Fitted> = KernelRidge::load(&mut p, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits, so any
    // drift at all is a defect in the container, not rounding.
    assert_eq!(
        loaded.dual_coef(&p),
        fitted.dual_coef(&p),
        "dual_coef_ must round-trip exactly"
    );
    // And the observable a user actually depends on. This is the assertion that
    // covers `X_fit_` and the rebuilt typed kernel at once — neither has a public
    // accessor, and a predict that matched by coincidence would need both to be
    // wrong in exactly compensating ways.
    assert_eq!(
        predictions(&mut p, &loaded),
        before,
        "the reloaded regressor must predict identically"
    );
}

#[test]
fn kernel_density_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kd.safetensors");
    let mut p = pool();

    let fitted = fit_kd::<f32>(&mut p, KdKernel::Epanechnikov, BandwidthSpec::Numeric(0.8));
    let before = scores(&mut p, &fitted);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: KernelDensity<f32, Fitted> =
        KernelDensity::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.bandwidth(),
        fitted.bandwidth(),
        "bandwidth_ must round-trip exactly"
    );
    assert_eq!(
        scores(&mut p, &loaded),
        before,
        "the reloaded density estimator must score identically"
    );
}

#[test]
fn roundtrip_is_bit_exact_at_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kr64.safetensors");
    let mut p = pool();

    let fitted = fit_kr::<f64>(&mut p, Some(0.3));
    let before = predictions(&mut p, &fitted);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: KernelRidge<f64, Fitted> = KernelRidge::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.dual_coef(&p),
        fitted.dual_coef(&p),
        "dual_coef_ at f64"
    );
    assert_eq!(predictions(&mut p, &loaded), before, "predictions at f64");
}

#[test]
fn every_kernel_variant_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // Every one of the six density kernels produces a plausible score, so a
    // `load` that silently fell back to the `gaussian` default would be
    // invisible in any single-variant test. Each variant must round-trip AND
    // score differently from the others.
    //
    // Two fixture choices here are load-bearing, and both are about keeping NaN
    // out of the comparison — NaN never equals itself, so a perfectly
    // round-tripped model would fail the assertion for a reason that has nothing
    // to do with the file.
    //
    // The bandwidth is wider than the fixture's diameter. Four of the six
    // kernels have COMPACT support, and at a narrow bandwidth every training
    // point falls outside it, so the density goes to zero and its log is not
    // finite.
    //
    // The feature count is 3 rather than the module-wide 4, because the COSINE
    // kernel's log-normalization constant is NaN at d = 4 — the alternating
    // series sklearn uses (`_binary_tree.pxi.tp`, which mlrs reproduces exactly)
    // sums to a NEGATIVE number there, and its log is taken. This is upstream
    // sklearn behavior, verified directly: `KernelDensity(kernel='cosine')`
    // returns `nan` scores for 4-D input and finite ones for 2-, 3- and 5-D.
    // mlrs matches it, which is the contract; this test just needs a dimension
    // where the kernel is usable.
    const KERNEL_FEATURES: usize = 3;
    const BANDWIDTH: f64 = 5.0;
    let small: Vec<f32> = (0..N_SAMPLES * KERNEL_FEATURES)
        .map(|i| ((i * 29) % 61) as f32 / 30.0 - 1.0)
        .collect();

    let mut seen: Vec<Vec<f32>> = Vec::new();
    for (i, kernel) in [
        KdKernel::Gaussian,
        KdKernel::Tophat,
        KdKernel::Epanechnikov,
        KdKernel::Exponential,
        KdKernel::Linear,
        KdKernel::Cosine,
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("kd{i}.safetensors"));
        let x: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &small);
        let shape = (N_SAMPLES, KERNEL_FEATURES);
        let fitted = KernelDensity::<f32>::builder()
            .kernel(kernel)
            .bandwidth(BandwidthSpec::Numeric(BANDWIDTH))
            .build::<f32>()
            .expect("KernelDensity builds")
            .fit(&mut p, &x, None, shape)
            .expect("KernelDensity fits the fixture");

        let score_of = |p: &mut BufferPool<ActiveRuntime>, m: &KernelDensity<f32, Fitted>| {
            let x: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(p, &small);
            m.score_samples(p, &x, shape)
                .expect("score_samples succeeds")
                .to_host(p)
        };
        let before = score_of(&mut p, &fitted);
        // Finiteness first: a NaN score would make the equality below fail
        // spuriously on a model that round-tripped perfectly.
        assert!(
            before.iter().all(|v| v.is_finite()),
            "{kernel:?} must produce finite scores at bandwidth {BANDWIDTH} \
             over {KERNEL_FEATURES} features, got {before:?}"
        );
        fitted.save(&p, &path).expect("save succeeds");
        let loaded: KernelDensity<f32, Fitted> =
            KernelDensity::load(&mut p, &path).expect("load succeeds");
        assert_eq!(
            score_of(&mut p, &loaded),
            before,
            "{kernel:?} must round-trip its scores exactly"
        );
        assert!(
            !seen.contains(&before),
            "{kernel:?} must score differently from the earlier kernels, or a \
             silent fallback would be invisible"
        );
        seen.push(before);
    }
}

// ---------------------------------------------------------------------------
// The resolution pair — request and outcome are two different facts
// ---------------------------------------------------------------------------

#[test]
fn a_reloaded_model_uses_the_stored_resolution() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kr.safetensors");
    let mut p = pool();

    // The sharp gate. `gamma=None` resolves to `1/n_features` at fit; the file
    // stores BOTH the `None` request and the resolved number. A loader that
    // re-ran the rule instead would put the same formula in two places, and a
    // later change to it would silently give every saved model a different
    // kernel.
    //
    // This constructs exactly that situation: a file whose `param:gamma` is
    // absent (so a re-resolving loader would compute `1/n_features`) but whose
    // stored `gamma_` is something else. The reload must use the stored value.
    let explicit = fit_kr::<f32>(&mut p, Some(0.3));
    let expected = predictions(&mut p, &explicit);
    let defaulted = fit_kr::<f32>(&mut p, None);
    assert_ne!(
        predictions(&mut p, &defaulted),
        expected,
        "0.3 and 1/n_features must give different predictions, or this gate proves nothing"
    );

    let x_fit = fixture::<f32>();
    let dual = explicit.dual_coef(&p);
    let mut w = KernelWriter::new("kernel_ridge");
    w.scalar_str("param:kernel", "poly");
    w.scalar_f64("param:alpha", 0.7);
    w.scalar_f64("param:degree", 3.0);
    w.scalar_f64("param:coef0", 0.5);
    // No `param:gamma` — a re-resolving loader would take 1/n_features here.
    w.scalar_f64("gamma_", 0.3);
    w.tensor(
        "X_fit_",
        TensorRef::floats(&x_fit, vec![N_SAMPLES, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "dual_coef_",
        TensorRef::floats(&dual, vec![N_SAMPLES, 1]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed");

    let loaded: KernelRidge<f32, Fitted> = KernelRidge::load(&mut p, &path).expect("load succeeds");
    assert_eq!(
        predictions(&mut p, &loaded),
        expected,
        "a reloaded model must use the STORED resolution, never re-run the rule"
    );
}

#[test]
fn the_bandwidth_rule_and_its_resolution_both_roundtrip() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let rule = dir.path().join("scott.safetensors");
    let numeric = dir.path().join("numeric.safetensors");
    let mut p = pool();

    // `bandwidth='scott'` carries no numeric value, so the specification rides
    // as its sklearn STRING and the resolved number as a separate scalar —
    // exactly the split `n_components='auto'` makes in the projection family.
    let scott = fit_kd::<f32>(&mut p, KdKernel::Gaussian, BandwidthSpec::Scott);
    let resolved = scott.bandwidth();
    scott.save(&p, &rule).expect("save succeeds");

    let raw = AlignedBytes::read(&rule).expect("read succeeds");
    let file = KernelFile::parse(&raw, "kernel_density").expect("parse succeeds");
    assert_eq!(
        file.scalar_str("param:bandwidth")
            .expect("the key is present"),
        "scott",
        "the rule must be stored as its sklearn string"
    );
    assert_eq!(
        file.scalar_f64("bandwidth_").expect("the key is present"),
        resolved,
        "and the RESOLVED value separately"
    );

    let loaded: KernelDensity<f32, Fitted> =
        KernelDensity::load(&mut p, &rule).expect("load succeeds");
    assert_eq!(
        loaded.bandwidth(),
        resolved,
        "the reloaded model must report what the fit resolved to"
    );

    // The numeric arm stores its decimal in the same key, and must not be
    // confused for a rule.
    fit_kd::<f32>(&mut p, KdKernel::Gaussian, BandwidthSpec::Numeric(0.8))
        .save(&p, &numeric)
        .expect("save succeeds");
    let raw = AlignedBytes::read(&numeric).expect("read succeeds");
    let file = KernelFile::parse(&raw, "kernel_density").expect("parse succeeds");
    assert_eq!(
        file.scalar_str("param:bandwidth")
            .expect("the key is present"),
        "0.8",
        "a numeric bandwidth must be stored as its shortest round-tripping decimal"
    );
}

#[test]
fn an_unrecognised_kernel_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad.safetensors");
    let mut p = pool();

    // A file from a hypothetical future build that grew a seventh density
    // kernel. It must fail by NAME rather than fall back to `gaussian` — every
    // variant produces a plausible score, so a silent fallback would score every
    // sample differently with nothing to signal it.
    let x_fit = fixture::<f32>();
    let mut w = KernelWriter::new("kernel_density");
    w.scalar_str("param:kernel", "triweight");
    w.scalar_str("param:bandwidth", "1.0");
    w.scalar_f64("bandwidth_", 1.0);
    w.tensor(
        "X_fit_",
        TensorRef::floats(&x_fit, vec![N_SAMPLES, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match KernelDensity::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an unrecognised kernel must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::BadMetadata { key } if *key == "param:kernel"),
        "expected BadMetadata naming param:kernel, got {err:?}"
    );
}

#[test]
fn a_non_positive_bandwidth_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("zero-h.safetensors");
    let mut p = pool();

    // The fit rejects a non-positive bandwidth; a hand-edited header must be
    // rejected too. It is the one number `score_samples` divides by, so without
    // the guard the first query returns NaN for every sample rather than
    // reporting a bad file.
    let x_fit = fixture::<f32>();
    let mut w = KernelWriter::new("kernel_density");
    w.scalar_str("param:kernel", "gaussian");
    w.scalar_str("param:bandwidth", "0.0");
    w.scalar_f64("bandwidth_", 0.0);
    w.tensor(
        "X_fit_",
        TensorRef::floats(&x_fit, vec![N_SAMPLES, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match KernelDensity::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a zero bandwidth must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// The size story
// ---------------------------------------------------------------------------

#[test]
fn the_training_set_is_the_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let small = dir.path().join("small.safetensors");
    let large = dir.path().join("large.safetensors");
    let mut p = pool();

    // A kernel method has no compressed parameterization: `score_samples`
    // evaluates against every training row, so the file necessarily grows
    // linearly with `n_samples`. Measured here so the claim in the module docs
    // is a property of the code rather than a comment.
    fit_kd::<f32>(&mut p, KdKernel::Gaussian, BandwidthSpec::Numeric(1.0))
        .save(&p, &small)
        .expect("save succeeds");

    let big: Vec<f32> = (0..N_SAMPLES * 4 * N_FEATURES)
        .map(|i| (i as f32) * 0.013 - 1.0)
        .collect();
    let big_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &big);
    KernelDensity::<f32>::builder()
        .bandwidth(BandwidthSpec::Numeric(1.0))
        .build::<f32>()
        .expect("KernelDensity builds")
        .fit(&mut p, &big_dev, None, (N_SAMPLES * 4, N_FEATURES))
        .expect("KernelDensity fits the larger fixture")
        .save(&p, &large)
        .expect("save succeeds");

    let small_len = std::fs::metadata(&small).expect("stat").len();
    let large_len = std::fs::metadata(&large).expect("stat").len();
    let extra = (N_SAMPLES * 3 * N_FEATURES) as u64 * 4;
    assert!(
        large_len - small_len >= extra,
        "a 4x-larger training set must add at least {extra} bytes \
         (small {small_len}, large {large_len})"
    );
}

#[test]
fn f32_model_writes_a_half_size_file() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let narrow = dir.path().join("f32.safetensors");
    let wide = dir.path().join("f64.safetensors");
    let mut p = pool();

    fit_kd::<f32>(&mut p, KdKernel::Gaussian, BandwidthSpec::Numeric(1.0))
        .save(&p, &narrow)
        .expect("save succeeds");
    fit_kd::<f64>(&mut p, KdKernel::Gaussian, BandwidthSpec::Numeric(1.0))
        .save(&p, &wide)
        .expect("save succeeds");

    let narrow_len = std::fs::metadata(&narrow).expect("stat").len();
    let wide_len = std::fs::metadata(&wide).expect("stat").len();

    // The stored dtype is the MODEL's dtype. It carries more weight here than
    // anywhere else in the format, since `X_fit_` is essentially the whole file.
    let payload_saved = (N_SAMPLES * N_FEATURES) as u64 * 4;
    assert!(
        wide_len - narrow_len >= payload_saved,
        "an f32 file must be at least {payload_saved} bytes smaller \
         (f32 {narrow_len}, f64 {wide_len})"
    );
}

#[test]
fn f32_file_loads_into_an_f64_model() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kr.safetensors");
    let mut p = pool();

    let fitted = fit_kr::<f32>(&mut p, Some(0.3));
    fitted.save(&p, &path).expect("save succeeds");

    // The file is self-describing, so storing at the model's own width is a
    // STORAGE decision and not a commitment about how it is loaded back.
    let widened: KernelRidge<f64, Fitted> =
        KernelRidge::load(&mut p, &path).expect("an f32 file loads into an f64 model");

    let narrow = fitted.dual_coef(&p);
    let wide = widened.dual_coef(&p);
    assert_eq!(narrow.len(), wide.len(), "the geometry is unchanged");
    for (i, (&n, &w)) in narrow.iter().zip(wide.iter()).enumerate() {
        // f32 → f64 is exact, so `==` and not a tolerance.
        assert_eq!(f64::from(n), w, "dual_coef_[{i}] must widen exactly");
    }
}

#[test]
fn the_load_path_is_zero_copy() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kr.safetensors");
    let mut p = pool();
    fit_kr::<f64>(&mut p, Some(0.3))
        .save(&p, &path)
        .expect("save succeeds");

    // The claim `AlignedBytes` exists to make good, and the case it matters most
    // for: `X_fit_` is the largest tensor mlrs writes, and it reaches
    // `DeviceArray::from_host` as a borrow of the file's own bytes.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = KernelFile::parse(&raw, "kernel_ridge").expect("parse succeeds");
    for name in ["X_fit_", "dual_coef_"] {
        let view = file.tensor(name).expect("the tensor is present");
        assert!(
            bytemuck::try_cast_slice::<u8, f64>(view.data()).is_ok(),
            "'{name}' must be reinterpretable as &[f64] without a copy"
        );
    }
}

#[test]
fn saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    let fitted = fit_kr::<f32>(&mut p, Some(0.3));
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");

    // RAW BYTES: a model file must be a deterministic function of the model, so
    // it can be content-addressed and deduplicated. This is also the gate on the
    // `third_party/safetensors` `BTreeMap` patch — `KernelRidge` carries six
    // scalars, so a randomly-seeded header map is overwhelmingly likely to
    // reorder one.
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

// ---------------------------------------------------------------------------
// Rejection — the file is untrusted input (T-04-01-01)
// ---------------------------------------------------------------------------

#[test]
fn a_preprocessing_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("scaler.safetensors");
    let mut p = pool();

    // The cross-FAMILY gate: only the `format` discriminator separates the
    // containers, and it is checked before any tensor is fetched.
    let x: DeviceArray<ActiveRuntime, f32> = upload::<f32>(&mut p);
    MaxAbsScaler::<f32>::new()
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("MaxAbsScaler fits")
        .save(&p, &path)
        .expect("save succeeds");

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match KernelRidge::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-prep file must not load as a kernel method"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-kernel"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn sibling_estimators_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kd.safetensors");
    let mut p = pool();

    // The pointed case. Both files hold an `X_fit_` of the same shape and dtype
    // AND a `param:kernel` under the same key, and the two vocabularies overlap
    // on `"linear"` while meaning entirely different functions by it. The
    // `estimator` tag is what establishes which vocabulary applies before either
    // is parsed.
    fit_kd::<f32>(&mut p, KdKernel::Linear, BandwidthSpec::Numeric(1.0))
        .save(&p, &path)
        .expect("save succeeds");

    let err = match KernelRidge::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a kernel_density file must not load as a kernel_ridge"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "kernel_ridge" && found == "kernel_density"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_dual_coef_disagreeing_with_the_training_set_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ragged.safetensors");
    let mut p = pool();

    // A dual coefficient per training row is the definition; a `dual_coef_` with
    // fewer rows than `X_fit_` is neither individually malformed, so only the
    // CROSS-check catches it. Without it the kernel GEMM at predict time would
    // read the dual vector out of range on the first call.
    let x_fit = fixture::<f32>();
    let short = vec![0.0f32; N_SAMPLES - 1];
    let mut w = KernelWriter::new("kernel_ridge");
    w.scalar_str("param:kernel", "linear");
    w.scalar_f64("param:alpha", 1.0);
    w.scalar_f64("param:degree", 3.0);
    w.scalar_f64("param:coef0", 1.0);
    w.scalar_f64("gamma_", 0.25);
    w.tensor(
        "X_fit_",
        TensorRef::floats(&x_fit, vec![N_SAMPLES, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "dual_coef_",
        TensorRef::floats(&short, vec![N_SAMPLES - 1, 1]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match KernelRidge::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a dual_coef_ disagreeing with X_fit_ must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_zero_extent_training_set_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("empty.safetensors");
    let mut p = pool();

    // A kernel method with no training rows has nothing to evaluate against, and
    // an empty upload is a landmine on the device backends — so it is rejected
    // at parse time rather than becoming a model that fails later and elsewhere.
    let empty: [f32; 0] = [];
    let mut w = KernelWriter::new("kernel_density");
    w.scalar_str("param:kernel", "gaussian");
    w.scalar_str("param:bandwidth", "1.0");
    w.scalar_f64("bandwidth_", 1.0);
    w.tensor(
        "X_fit_",
        TensorRef::floats(&empty, vec![0, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match KernelDensity::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an empty training set must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kr.safetensors");
    let mut p = pool();
    fit_kr::<f32>(&mut p, Some(0.3))
        .save(&p, &path)
        .expect("save succeeds");

    // `save` writes to a sibling temporary and renames it into place so an
    // interrupted write cannot replace a good model with a truncated one.
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
