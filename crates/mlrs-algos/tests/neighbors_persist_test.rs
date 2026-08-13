//! NEIGH-PERSIST (prototype) — safetensors save/load round-trips for the three
//! neighbor estimators: `NearestNeighbors`, `KNeighborsClassifier` and
//! `KNeighborsRegressor`.
//!
//! All three store the training matrix and nothing else of substance — a k-NN
//! estimator has no parameters, so `_fit_X` is not a fitting artifact but the
//! entire fitted state. What distinguishes them on disk is only what rides
//! ALONGSIDE it, and that is where the gates concentrate.
//!
//! The sharp case is `KNeighborsClassifier`'s label pair. Its core is
//! integer-only: the training labels are `np.unique`-encoded to a dense `0..K`
//! because the gather indexes a per-class table by that value (CR-02). So the
//! file has to carry BOTH the encoding the kernel consumes and the `classes_`
//! table that decodes a prediction — and `non_contiguous_labels_roundtrip` is
//! the gate that proves it does: a file storing only the encoding would
//! round-trip its own state perfectly and predict `{0, 1, 2}` where training
//! said `{0, 2, 7}`. The two tensors also index each other, which makes
//! `an_out_of_range_class_id_is_rejected` a real bounds guard rather than a
//! formality.
//!
//! The second is the metric. `Metric::Minkowski` is the one variant carrying a
//! payload, and it rides as sklearn's own two-argument split (`metric` and `p`).
//! A `'minkowski'` with no `p` is REJECTED rather than defaulted to 2, because a
//! silent `p = 2` is Euclidean — a different metric, computing different
//! distances, with nothing to signal it.
//!
//! The remaining gates are the standard container set: bit-exact round-trips,
//! query/prediction equivalence, the two dtype-tag claims, zero-copy loading,
//! byte-level determinism, and the rejection set over hand-built headers. The
//! file is untrusted input (T-04-01-01).
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::neighbors::classifier::KNeighborsClassifier;
use mlrs_algos::neighbors::nearest::NearestNeighbors;
use mlrs_algos::neighbors::neighbors_persist::{
    AlignedBytes, LoadModel, NeighborsFile, NeighborsWriter, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::neighbors::regressor::KNeighborsRegressor;
use mlrs_algos::neighbors::{Metric, Weights};
use mlrs_algos::preprocessing::MaxAbsScaler;
use mlrs_algos::typestate::{Fit, Fitted, KNeighbors, Predict, PredictLabels};
use mlrs_backend::capability;
use mlrs_backend::device::Device;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

const N_SAMPLES: usize = 16;
const N_FEATURES: usize = 3;
const K: usize = 3;

/// A deterministic fixture whose rows are mutually distinct, so the `k` nearest
/// neighbors of every query are unambiguous — a tie would make the
/// query-equivalence gates depend on the scan's tie-break order rather than on
/// the file.
fn fixture<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES * N_FEATURES)
        .map(|i| {
            let v = ((i * 41) % 97) as f64 / 20.0 - 2.0;
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

/// Class labels that are NOT `0..K`. The gap is deliberate: the core encodes to
/// a dense range internally, so only a non-contiguous label set can tell a
/// faithful round-trip from one that stored the encoding alone.
fn labels<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES)
        .map(|i| mlrs_core::f64_to_host::<F>(f64::from([0i32, 2, 7][i % 3])))
        .collect()
}

fn targets<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES)
        .map(|i| mlrs_core::f64_to_host::<F>((i as f64) * 0.25 - 1.0))
        .collect()
}

/// A `NearestNeighbors` with every hyperparameter off its default —
/// `minkowski(p=3)` in particular, which is the one metric carrying a payload
/// and therefore the one a round-trip can lose half of.
fn fit_nn<F>(p: &mut BufferPool<ActiveRuntime>, metric: Metric) -> NearestNeighbors<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    NearestNeighbors::<F>::builder()
        .n_neighbors(K)
        .metric(metric)
        .device(Device::Cpu)
        .build::<F>()
        .expect("NearestNeighbors builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("NearestNeighbors fits the fixture")
}

fn fit_clf<F>(p: &mut BufferPool<ActiveRuntime>) -> KNeighborsClassifier<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    let y: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(p, &labels::<F>());
    KNeighborsClassifier::<F>::builder()
        .n_neighbors(K)
        .weights(Weights::Distance)
        .build::<F>()
        .expect("KNeighborsClassifier builds")
        .fit(p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("KNeighborsClassifier fits the fixture")
}

fn fit_reg<F>(p: &mut BufferPool<ActiveRuntime>) -> KNeighborsRegressor<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    let y: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(p, &targets::<F>());
    KNeighborsRegressor::<F>::builder()
        .n_neighbors(K)
        .weights(Weights::Distance)
        .build::<F>()
        .expect("KNeighborsRegressor builds")
        .fit(p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("KNeighborsRegressor fits the fixture")
}

/// The `(distances, indices)` of the fixture's own rows — the observable a
/// `NearestNeighbors` user depends on, and the one that would catch a training
/// matrix that came back reordered or truncated.
fn neighbors_of<F, T>(p: &mut BufferPool<ActiveRuntime>, m: &T) -> (Vec<F>, Vec<i32>)
where
    F: Float + CubeElement + Pod,
    T: KNeighbors<F>,
{
    let x = upload::<F>(p);
    let (d, i) = m
        .kneighbors(p, &x, (N_SAMPLES, N_FEATURES), K)
        .expect("kneighbors succeeds on the training geometry");
    (d.to_host(p), i.to_host(p))
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn nearest_neighbors_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("nn.safetensors");
    let mut p = pool();

    let fitted = fit_nn::<f32>(&mut p, Metric::Euclidean);
    let before = neighbors_of::<f32, _>(&mut p, &fitted);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: NearestNeighbors<f32, Fitted> =
        NearestNeighbors::load(&mut p, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits, so any
    // drift at all is a defect in the container, not rounding. `_fit_X` has no
    // public accessor, so the query result IS the comparison — and it covers the
    // whole model, since a k-NN estimator has nothing else.
    assert_eq!(
        neighbors_of::<f32, _>(&mut p, &loaded),
        before,
        "the reloaded estimator must return identical neighbors"
    );
    assert_eq!(loaded.train_shape(), fitted.train_shape(), "train_shape_");
}

#[test]
fn classifier_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("clf.safetensors");
    let mut p = pool();

    let fitted = fit_clf::<f32>(&mut p);
    let x = upload::<f32>(&mut p);
    let before = fitted
        .predict_labels(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict_labels succeeds")
        .to_host(&p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: KNeighborsClassifier<f32, Fitted> =
        KNeighborsClassifier::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.classes(),
        fitted.classes(),
        "classes_ must round-trip"
    );
    assert_eq!(loaded.n_classes(), fitted.n_classes(), "n_classes_");
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .predict_labels(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("predict_labels succeeds")
            .to_host(&p),
        before,
        "the reloaded classifier must predict identically"
    );
}

#[test]
fn regressor_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("reg.safetensors");
    let mut p = pool();

    let fitted = fit_reg::<f32>(&mut p);
    let x = upload::<f32>(&mut p);
    let before = fitted
        .predict(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict succeeds")
        .to_host(&p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: KNeighborsRegressor<f32, Fitted> =
        KNeighborsRegressor::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.n_outputs(), fitted.n_outputs(), "n_outputs_");
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .predict(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("predict succeeds")
            .to_host(&p),
        before,
        "the reloaded regressor must predict identically"
    );
}

#[test]
fn roundtrip_is_bit_exact_at_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("nn64.safetensors");
    let mut p = pool();

    let fitted = fit_nn::<f64>(&mut p, Metric::Euclidean);
    let before = neighbors_of::<f64, _>(&mut p, &fitted);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: NearestNeighbors<f64, Fitted> =
        NearestNeighbors::load(&mut p, &path).expect("load succeeds");
    assert_eq!(
        neighbors_of::<f64, _>(&mut p, &loaded),
        before,
        "the reloaded estimator must return identical neighbors at f64"
    );
}

// ---------------------------------------------------------------------------
// The label pair
// ---------------------------------------------------------------------------

#[test]
fn non_contiguous_labels_roundtrip() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("clf.safetensors");
    let mut p = pool();

    // The CR-02 contract. The kernel sees a dense `0..K`, so a file that stored
    // THOSE would round-trip its own state perfectly and still predict
    // `{0, 1, 2}` where training said `{0, 2, 7}`. Storing `classes_` alongside
    // the encoding is what makes the decode faithful, and asserting on the LABEL
    // VALUES rather than on the encoding is what checks it.
    let fitted = fit_clf::<f32>(&mut p);
    assert_eq!(
        fitted.classes(),
        &[0, 2, 7],
        "the fixture must use non-contiguous labels, or this gate proves nothing"
    );
    fitted.save(&p, &path).expect("save succeeds");

    let loaded: KNeighborsClassifier<f32, Fitted> =
        KNeighborsClassifier::load(&mut p, &path).expect("load succeeds");
    assert_eq!(
        loaded.classes(),
        &[0, 2, 7],
        "classes_ must survive verbatim"
    );

    let x = upload::<f32>(&mut p);
    let predicted = loaded
        .predict_labels(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict_labels succeeds")
        .to_host(&p);
    assert!(
        predicted.iter().all(|v| [0, 2, 7].contains(v)),
        "every prediction must be an original training label, got {predicted:?}"
    );
}

#[test]
fn an_out_of_range_class_id_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-id.safetensors");
    let mut p = pool();

    // `_y` and `classes_` INDEX each other: every entry of `_y` addresses the
    // class table. A file whose encoding reaches past that table is
    // individually well-formed in both halves — a `3` is a fine integer and a
    // 3-entry `classes_` is a fine table — so only the cross-check catches it,
    // and without it the first decoded prediction reads out of range.
    let x = fixture::<f32>();
    let y: Vec<i64> = (0..N_SAMPLES).map(|i| if i == 5 { 9 } else { 0 }).collect();
    let classes = [0i64, 2, 7];
    let mut w = NeighborsWriter::new("kneighbors_classifier");
    w.scalar_usize("param:n_neighbors", K);
    w.scalar_str("param:weights", "uniform");
    w.scalar_str("param:metric", "euclidean");
    w.scalar_str("param:device", "auto");
    w.tensor(
        "_fit_X",
        TensorRef::floats(&x, vec![N_SAMPLES, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "_y",
        TensorRef::i64s(&y, vec![N_SAMPLES]).expect("well-formed"),
    );
    w.tensor(
        "classes_",
        TensorRef::i64s(&classes, vec![3]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match KNeighborsClassifier::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an out-of-range class id must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// The metric
// ---------------------------------------------------------------------------

#[test]
fn every_metric_roundtrips_including_the_minkowski_exponent() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // Each metric must round-trip AND produce different distances from the
    // others — a `load` that silently fell back to Euclidean would be invisible
    // in any single-metric test.
    let mut seen: Vec<Vec<f32>> = Vec::new();
    for (i, metric) in [
        Metric::Euclidean,
        Metric::Manhattan,
        Metric::Chebyshev,
        Metric::Minkowski { p: 3.0 },
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("nn{i}.safetensors"));
        let fitted = fit_nn::<f32>(&mut p, metric);
        let (before, _) = neighbors_of::<f32, _>(&mut p, &fitted);
        fitted.save(&p, &path).expect("save succeeds");
        let loaded: NearestNeighbors<f32, Fitted> =
            NearestNeighbors::load(&mut p, &path).expect("load succeeds");
        let (after, _) = neighbors_of::<f32, _>(&mut p, &loaded);
        assert_eq!(
            after, before,
            "{metric:?} must round-trip its distances exactly"
        );
        assert!(
            !seen.contains(&before),
            "{metric:?} must give different distances from the earlier metrics, or a \
             silent fallback would be invisible"
        );
        seen.push(before);
    }
}

#[test]
fn a_minkowski_without_an_exponent_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("no-p.safetensors");
    let mut p = pool();

    // `p` is REQUIRED for `metric='minkowski'`, never defaulted to 2 — a silent
    // `p = 2` is Euclidean, so a file that lost its exponent would load as a
    // DIFFERENT metric and every distance it computed would be wrong with
    // nothing to signal it.
    let x = fixture::<f32>();
    let mut w = NeighborsWriter::new("nearest_neighbors");
    w.scalar_usize("param:n_neighbors", K);
    w.scalar_str("param:metric", "minkowski");
    w.scalar_str("param:device", "auto");
    w.tensor(
        "_fit_X",
        TensorRef::floats(&x, vec![N_SAMPLES, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match NearestNeighbors::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a minkowski metric without p must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::BadMetadata { key } if *key == "param:metric"),
        "expected BadMetadata naming param:metric, got {err:?}"
    );
}

#[test]
fn the_exponent_is_written_only_for_minkowski() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let euclid = dir.path().join("euclid.safetensors");
    let mink = dir.path().join("mink.safetensors");
    let mut p = pool();

    // Key-presence carries the `Option` exactly: the four payload-free metrics
    // write no `param:p` at all rather than a meaningless sentinel a reader
    // might act on.
    fit_nn::<f32>(&mut p, Metric::Euclidean)
        .save(&p, &euclid)
        .expect("save succeeds");
    fit_nn::<f32>(&mut p, Metric::Minkowski { p: 3.0 })
        .save(&p, &mink)
        .expect("save succeeds");

    let raw = AlignedBytes::read(&euclid).expect("read succeeds");
    let file = NeighborsFile::parse(&raw, "nearest_neighbors").expect("parse succeeds");
    assert_eq!(
        file.scalar_opt_f64("param:p").expect("parses"),
        None,
        "a payload-free metric must write no param:p"
    );

    let raw = AlignedBytes::read(&mink).expect("read succeeds");
    let file = NeighborsFile::parse(&raw, "nearest_neighbors").expect("parse succeeds");
    assert_eq!(
        file.scalar_opt_f64("param:p").expect("parses"),
        Some(3.0),
        "minkowski must record its exponent"
    );
}

// ---------------------------------------------------------------------------
// The dtype-tag and format claims
// ---------------------------------------------------------------------------

#[test]
fn f32_model_writes_a_half_size_file() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let narrow = dir.path().join("f32.safetensors");
    let wide = dir.path().join("f64.safetensors");
    let mut p = pool();

    fit_nn::<f32>(&mut p, Metric::Euclidean)
        .save(&p, &narrow)
        .expect("save succeeds");
    fit_nn::<f64>(&mut p, Metric::Euclidean)
        .save(&p, &wide)
        .expect("save succeeds");

    let narrow_len = std::fs::metadata(&narrow).expect("stat").len();
    let wide_len = std::fs::metadata(&wide).expect("stat").len();

    // The stored dtype is the MODEL's dtype. It is the only size lever this
    // family has, since `_fit_X` is the whole file.
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
    let path = dir.path().join("nn.safetensors");
    let mut p = pool();

    fit_nn::<f32>(&mut p, Metric::Euclidean)
        .save(&p, &path)
        .expect("save succeeds");

    // The file is self-describing, so storing at the model's own width is a
    // STORAGE decision and not a commitment about how it is loaded back.
    let widened: NearestNeighbors<f64, Fitted> =
        NearestNeighbors::load(&mut p, &path).expect("an f32 file loads into an f64 model");
    assert_eq!(
        widened.train_shape(),
        (N_SAMPLES, N_FEATURES),
        "the geometry is unchanged by the widening"
    );

    // And the widened model queries. The comparison is over each row's neighbor
    // SET, not its order.
    //
    // f32 → f64 widening is exact for every stored VALUE, but the distances are
    // recomputed at the wider precision, and this fixture contains rows with
    // near-ties — pairs whose distances agree to within f32's resolution and
    // separate at f64's. Those legitimately swap rank. Asserting the ordering
    // would be asserting that no such pair exists, which is a property of the
    // fixture rather than of the file; asserting the set still catches a
    // training matrix that came back truncated, reordered or partly converted.
    let narrow = fit_nn::<f32>(&mut p, Metric::Euclidean);
    let (_, narrow_idx) = neighbors_of::<f32, _>(&mut p, &narrow);
    let (_, wide_idx) = neighbors_of::<f64, _>(&mut p, &widened);
    for row in 0..N_SAMPLES {
        let mut a: Vec<i32> = narrow_idx[row * K..(row + 1) * K].to_vec();
        let mut b: Vec<i32> = wide_idx[row * K..(row + 1) * K].to_vec();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(
            a, b,
            "row {row}: the widened model must find the same neighbors"
        );
    }
}

#[test]
fn the_load_path_is_zero_copy() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("clf.safetensors");
    let mut p = pool();
    fit_clf::<f64>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    // The claim `AlignedBytes` exists to make good, over both dtype widths this
    // file mixes: `_fit_X` is `F64` and the two label tensors are `I64`, and
    // safetensors orders them by descending dtype width in an 8-aligned buffer,
    // so all three land on their natural alignment.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = NeighborsFile::parse(&raw, "kneighbors_classifier").expect("parse succeeds");
    let x = file.tensor("_fit_X").expect("the tensor is present");
    assert!(
        bytemuck::try_cast_slice::<u8, f64>(x.data()).is_ok(),
        "'_fit_X' must be reinterpretable as &[f64] without a copy"
    );
    for name in ["_y", "classes_"] {
        let view = file.tensor(name).expect("the tensor is present");
        assert!(
            bytemuck::try_cast_slice::<u8, i64>(view.data()).is_ok(),
            "'{name}' must be reinterpretable as &[i64] without a copy"
        );
    }
}

#[test]
fn saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    let fitted = fit_clf::<f32>(&mut p);
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");

    // RAW BYTES: a model file must be a deterministic function of the model, so
    // it can be content-addressed and deduplicated. This is also the gate on the
    // `third_party/safetensors` `BTreeMap` patch — stock safetensors serializes
    // `__metadata__` out of a randomly-seeded `HashMap`, which shuffles the
    // header between runs.
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
    let err = match NearestNeighbors::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-prep file must not load as a neighbor estimator"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-neighbors"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn sibling_estimators_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let reg_path = dir.path().join("reg.safetensors");
    let nn_path = dir.path().join("nn.safetensors");
    let mut p = pool();

    fit_reg::<f32>(&mut p)
        .save(&p, &reg_path)
        .expect("save succeeds");
    fit_nn::<f32>(&mut p, Metric::Euclidean)
        .save(&p, &nn_path)
        .expect("save succeeds");

    // A regressor file loaded as a `NearestNeighbors` would succeed on every
    // geometry check — the extra `_y` is simply never fetched — and quietly
    // discard the targets that make it a regressor. The tag is what reports it.
    let err = match NearestNeighbors::<f32, Fitted>::load(&mut p, &reg_path) {
        Ok(_) => panic!("a regressor file must not load as a nearest_neighbors"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "nearest_neighbors" && found == "kneighbors_regressor"
        ),
        "expected WrongEstimator, got {err:?}"
    );

    // And the reverse, where the missing tensor would otherwise read as
    // corruption rather than as "this model has no targets".
    let err = match KNeighborsRegressor::<f32, Fitted>::load(&mut p, &nn_path) {
        Ok(_) => panic!("a nearest_neighbors file must not load as a regressor"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::WrongEstimator { .. }),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_k_larger_than_the_training_set_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("big-k.safetensors");
    let mut p = pool();

    // `k > n_samples` is unanswerable, not merely odd: the scan cannot return
    // more neighbors than it holds. A hand-edited `param:n_neighbors` has to
    // fail here rather than produce an out-of-range gather on the first query.
    let x = fixture::<f32>();
    let mut w = NeighborsWriter::new("nearest_neighbors");
    w.scalar_usize("param:n_neighbors", N_SAMPLES + 1);
    w.scalar_str("param:metric", "euclidean");
    w.scalar_str("param:device", "auto");
    w.tensor(
        "_fit_X",
        TensorRef::floats(&x, vec![N_SAMPLES, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match NearestNeighbors::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a k larger than the training set must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_target_table_disagreeing_with_the_training_set_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ragged.safetensors");
    let mut p = pool();

    // One target per training row is the definition; neither extent is wrong on
    // its own, so only the CROSS-check catches it. Without it the regression
    // gather would read the target table out of range, since it addresses it by
    // neighbor index.
    let x = fixture::<f32>();
    let y = vec![0.0f32; N_SAMPLES - 2];
    let mut w = NeighborsWriter::new("kneighbors_regressor");
    w.scalar_usize("param:n_neighbors", K);
    w.scalar_str("param:weights", "uniform");
    w.scalar_str("param:metric", "euclidean");
    w.scalar_str("param:device", "auto");
    w.tensor(
        "_fit_X",
        TensorRef::floats(&x, vec![N_SAMPLES, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "_y",
        TensorRef::floats(&y, vec![N_SAMPLES - 2, 1]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match KNeighborsRegressor::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a target table disagreeing with _fit_X must not load"),
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

    // A k-NN estimator with no training rows has nothing to scan, and an empty
    // upload is a landmine on the device backends.
    let empty: [f32; 0] = [];
    let mut w = NeighborsWriter::new("nearest_neighbors");
    w.scalar_usize("param:n_neighbors", 1);
    w.scalar_str("param:metric", "euclidean");
    w.scalar_str("param:device", "auto");
    w.tensor(
        "_fit_X",
        TensorRef::floats(&empty, vec![0, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match NearestNeighbors::<f32, Fitted>::load(&mut p, &path) {
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
    let path = dir.path().join("nn.safetensors");
    let mut p = pool();
    fit_nn::<f32>(&mut p, Metric::Euclidean)
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
