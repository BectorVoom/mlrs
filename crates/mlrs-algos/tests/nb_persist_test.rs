//! NB-PERSIST (prototype) — safetensors save/load round-trips for the Naive
//! Bayes estimators.
//!
//! All five estimators are covered, across the three storage shapes the format
//! has to handle: `GaussianNB` keeps dense device-resident `theta_`/`var_`;
//! `MultinomialNB` / `ComplementNB` / `BernoulliNB` share one dense discrete
//! core; `CategoricalNB` keeps `feature_log_prob_` on the host and RAGGED (one
//! `n_classes × n_categories_[j]` matrix per feature).
//!
//! The gates, in the order they matter:
//!
//!   - `*_roundtrip_is_bit_exact` — every fitted table survives save→load with
//!     `==`, not a tolerance. Persistence has no numerical error budget: a
//!     round-trip that only matches to 1e-5 has a bug, and a band would hide it.
//!   - `*_roundtrip_preserves_predictions` — the reloaded model predicts
//!     identically, which is the property a user actually cares about.
//!   - `f32_model_writes_a_half_size_file` — the dtype-tag claim, measured on
//!     real files rather than asserted in a comment.
//!   - `f32_file_loads_into_an_f64_model` — the other half of that claim: the
//!     tag makes the file self-describing, so the width is a load-time choice.
//!   - `ragged_layout_beats_dense_padding` — the flat-CSR claim, measured
//!     against the padded-cube size the obvious alternative would have written.
//!   - `min_categories_*_roundtrips` — the three-variant enum knob, whose
//!     `PerFeature` arm is the only one that costs a tensor.
//!   - `bernoulli_names_its_matrix_for_what_it_holds` — the one deliberate
//!     break from sklearn's attribute naming, pinned so it cannot drift back.
//!   - the rejection gates — the full 5×5 estimator cross-product, a foreign
//!     safetensors file, and a header whose declared extents disagree. The file
//!     is untrusted input (T-04-01-01), so an inconsistent header must be a
//!     typed error, never an out-of-bounds read at predict time.
//!   - `save_leaves_no_temporary_behind` — the write-then-rename path.
//!
//! Fixtures are generated in-test rather than loaded from an oracle `.npz`:
//! these gates are about the CONTAINER, and comparing a model against itself
//! needs no sklearn reference. The sklearn-parity gates for the fits themselves
//! live in `gaussian_nb_test.rs` / `categorical_nb_test.rs`.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::naive_bayes::nb_persist::{
    ragged_block_offsets, AlignedBytes, LoadModel, NbFile, NbWriter, PersistError, SaveModel,
    TensorRef,
};
use mlrs_algos::naive_bayes::{
    BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MinCategories, MultinomialNB,
};
use mlrs_algos::typestate::{Fitted, PredictLabels, PredictProba};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};
use mlrs_core::host_to_f64;

/// A small, deterministic, well-separated 3-class continuous fixture.
///
/// Deterministic on purpose: a persistence round-trip is exact or broken, and
/// a seeded RNG would only add a way for the two arms to disagree for reasons
/// that have nothing to do with the file.
fn gaussian_fixture<F: Pod>() -> (Vec<F>, Vec<F>, (usize, usize)) {
    let rows: [[f64; 4]; 9] = [
        [0.0, 1.0, 2.0, 3.0],
        [0.2, 1.1, 2.2, 3.1],
        [-0.1, 0.9, 1.8, 2.9],
        [5.0, 6.0, 7.0, 8.0],
        [5.2, 6.1, 7.2, 8.1],
        [4.9, 5.9, 6.8, 7.9],
        [10.0, 11.0, 12.0, 13.0],
        [10.2, 11.1, 12.2, 13.1],
        [9.9, 10.9, 11.8, 12.9],
    ];
    let labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let x = rows
        .iter()
        .flatten()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect();
    let y = labels
        .iter()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect();
    (x, y, (9, 4))
}

/// A categorical fixture whose per-feature cardinalities differ WIDELY — 2, 3,
/// 8 and 2 categories.
///
/// The spread is the point: it is what makes `feature_log_prob_` genuinely
/// ragged, so the flat-CSR layout is exercised rather than a square special
/// case, and it is what `ragged_layout_beats_dense_padding` measures against.
fn categorical_fixture<F: Pod>() -> (Vec<F>, Vec<F>, (usize, usize)) {
    let rows: [[f64; 4]; 10] = [
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 7.0, 0.0],
        [0.0, 2.0, 3.0, 1.0],
        [1.0, 0.0, 5.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 6.0, 0.0],
        [0.0, 0.0, 2.0, 1.0],
        [1.0, 1.0, 4.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 0.0, 7.0, 0.0],
    ];
    let labels = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let x = rows
        .iter()
        .flatten()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect();
    let y = labels
        .iter()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect();
    (x, y, (10, 4))
}

fn fit_gaussian<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    priors: Option<Vec<f64>>,
) -> GaussianNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (x, y, shape) = gaussian_fixture::<F>();
    GaussianNB::<F>::builder()
        .priors(priors)
        .var_smoothing(1e-9)
        .build::<F>()
        .expect("GaussianNB builds with valid hyperparameters")
        .fit_from_host_slice(pool, &x, &y, shape, None)
        .expect("GaussianNB fits the fixture")
}

fn fit_categorical<F>(min_categories: MinCategories) -> CategoricalNB<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let (x, y, shape) = categorical_fixture::<F>();
    CategoricalNB::<F>::builder()
        .min_categories(min_categories)
        .build::<F>()
        .expect("CategoricalNB builds with valid hyperparameters")
        .fit_from_host_slice(&x, &y, shape, None)
        .expect("CategoricalNB fits the fixture")
}

/// Host `predict_labels` for a fitted estimator over the fixture's own rows.
fn labels_of<F, E>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &E,
    x: &[F],
    shape: (usize, usize),
) -> Vec<i32>
where
    F: Float + CubeElement + Pod,
    E: PredictLabels<F>,
{
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, x);
    model
        .predict_labels(pool, &x_dev, shape)
        .expect("predict_labels succeeds on the training geometry")
        .to_host(pool)
}

/// Host `predict_proba` for a fitted estimator, widened to `f64` for comparison.
fn proba_of<F, E>(
    pool: &mut BufferPool<ActiveRuntime>,
    model: &E,
    x: &[F],
    shape: (usize, usize),
) -> Vec<f64>
where
    F: Float + CubeElement + Pod,
    E: PredictProba<F>,
{
    let x_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, x);
    model
        .predict_proba(pool, &x_dev, shape)
        .expect("predict_proba succeeds on the training geometry")
        .to_host(pool)
        .iter()
        .map(|&v| host_to_f64(v))
        .collect()
}

// ---------------------------------------------------------------------------
// GaussianNB — the dense, device-resident shape
// ---------------------------------------------------------------------------

#[test]
fn gaussian_roundtrip_is_bit_exact() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gaussian.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_gaussian::<f64>(&mut pool, None);
    fitted.save(&pool, &path).expect("save succeeds");

    let loaded: GaussianNB<f64, Fitted> =
        GaussianNB::load(&mut pool, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits, so any
    // drift at all is a defect in the container, not rounding.
    assert_eq!(
        loaded.classes(),
        fitted.classes(),
        "classes_ must round-trip"
    );
    assert_eq!(
        loaded.class_count(),
        fitted.class_count(),
        "class_count_ must round-trip"
    );
    assert_eq!(
        loaded.class_log_prior(),
        fitted.class_log_prior(),
        "class_log_prior_ must round-trip"
    );
    assert_eq!(
        loaded.epsilon(),
        fitted.epsilon(),
        "epsilon_ must round-trip through its __metadata__ scalar"
    );
    assert_eq!(
        loaded.theta(&pool),
        fitted.theta(&pool),
        "theta_ must round-trip"
    );
    assert_eq!(loaded.var(&pool), fitted.var(&pool), "var_ must round-trip");
}

#[test]
fn gaussian_roundtrip_preserves_predictions() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gaussian.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let (x, _, shape) = gaussian_fixture::<f64>();
    let fitted = fit_gaussian::<f64>(&mut pool, None);
    let want_labels = labels_of(&mut pool, &fitted, &x, shape);
    let want_proba = proba_of(&mut pool, &fitted, &x, shape);

    fitted.save(&pool, &path).expect("save succeeds");
    let loaded: GaussianNB<f64, Fitted> =
        GaussianNB::load(&mut pool, &path).expect("load succeeds");

    assert_eq!(
        labels_of(&mut pool, &loaded, &x, shape),
        want_labels,
        "a reloaded model must predict the same labels"
    );
    assert_eq!(
        proba_of(&mut pool, &loaded, &x, shape),
        want_proba,
        "a reloaded model must predict bit-identical probabilities"
    );
}

#[test]
fn gaussian_priors_roundtrip() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let with = dir.path().join("with_priors.safetensors");
    let without = dir.path().join("without_priors.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // `priors` decides `class_log_prior_`, so it is part of the model's
    // identity and a round-trip that dropped it would silently change the fit.
    let supplied = vec![0.5, 0.25, 0.25];
    let fitted = fit_gaussian::<f64>(&mut pool, Some(supplied.clone()));
    fitted.save(&pool, &with).expect("save succeeds");
    let loaded: GaussianNB<f64, Fitted> =
        GaussianNB::load(&mut pool, &with).expect("load succeeds");
    let expected: Vec<f64> = supplied.iter().map(|&p| p.ln()).collect();
    assert_eq!(
        loaded.class_log_prior(),
        Some(expected.as_slice()),
        "a supplied `priors` must survive the round-trip"
    );

    // The `None` arm writes NO tensor at all rather than a sentinel, so the
    // absent case must also come back absent — and cost fewer bytes.
    let empirical = fit_gaussian::<f64>(&mut pool, None);
    empirical.save(&pool, &without).expect("save succeeds");
    assert!(
        std::fs::metadata(&without).unwrap().len() < std::fs::metadata(&with).unwrap().len(),
        "omitting `priors` must omit its tensor, not write a placeholder"
    );
}

#[test]
fn f32_model_writes_a_half_size_file() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let small = dir.path().join("f32.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    fit_gaussian::<f32>(&mut pool, None)
        .save(&pool, &small)
        .expect("save succeeds");
    let f32_len = std::fs::metadata(&small).unwrap().len();

    if capability::skip_f64_with_log() {
        return;
    }
    let big = dir.path().join("f64.safetensors");
    fit_gaussian::<f64>(&mut pool, None)
        .save(&pool, &big)
        .expect("save succeeds");
    let f64_len = std::fs::metadata(&big).unwrap().len();

    // `theta_` + `var_` are the only F-typed tensors, 2 × 3 × 4 values each, so
    // the f32 file must be exactly 2·3·4·2·(8−4) = 96 bytes smaller. An exact
    // difference rather than a ratio: everything else in the two files —
    // header, classes_, the f64 class vectors — is byte-identical, so this
    // pins the claim that ONLY the model's own float width changed.
    assert_eq!(
        f64_len - f32_len,
        96,
        "an f32 model must store f32 tensors: f32 file {f32_len} B, f64 file {f64_len} B"
    );
}

#[test]
fn f32_file_loads_into_an_f64_model() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("f32.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_gaussian::<f32>(&mut pool, None);
    let want_theta = fitted.theta(&pool).expect("a fitted theta_");
    fitted.save(&pool, &path).expect("save succeeds");

    // The dtype tag makes the file self-describing, so the element width is a
    // LOAD-time choice: train on a GPU in f32, evaluate in f64. The values are
    // the f32 ones exactly — widening f32→f64 is lossless.
    let widened: GaussianNB<f64, Fitted> =
        GaussianNB::load(&mut pool, &path).expect("an f32 file loads into an f64 model");
    assert_eq!(
        widened.theta(&pool),
        Some(want_theta),
        "widening an f32 payload to f64 must be exact"
    );
}

// ---------------------------------------------------------------------------
// MultinomialNB / BernoulliNB / ComplementNB — the shared discrete core
// ---------------------------------------------------------------------------

/// A small non-negative count fixture, valid input for all three discrete
/// variants (Multinomial and Complement reject negatives; Bernoulli binarizes).
fn count_fixture<F: Pod>() -> (Vec<F>, Vec<F>, (usize, usize)) {
    let rows: [[f64; 5]; 8] = [
        [3.0, 0.0, 1.0, 0.0, 2.0],
        [4.0, 1.0, 0.0, 0.0, 3.0],
        [2.0, 0.0, 2.0, 1.0, 1.0],
        [0.0, 3.0, 0.0, 4.0, 0.0],
        [1.0, 4.0, 0.0, 3.0, 0.0],
        [0.0, 2.0, 1.0, 5.0, 1.0],
        [5.0, 0.0, 3.0, 0.0, 4.0],
        [0.0, 5.0, 0.0, 2.0, 0.0],
    ];
    let labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
    let x = rows
        .iter()
        .flatten()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect();
    let y = labels
        .iter()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect();
    (x, y, (8, 5))
}

#[test]
fn multinomial_roundtrip_is_bit_exact() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("multinomial.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, y, shape) = count_fixture::<f64>();

    let fitted = MultinomialNB::<f64>::builder()
        .alpha(0.7)
        .class_prior(Some(vec![0.6, 0.4]))
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, shape, None)
        .expect("fits");
    let want_labels = labels_of(&mut pool, &fitted, &x, shape);
    let want_proba = proba_of(&mut pool, &fitted, &x, shape);

    fitted.save(&pool, &path).expect("save succeeds");
    let loaded: MultinomialNB<f64, Fitted> =
        MultinomialNB::load(&mut pool, &path).expect("load succeeds");

    assert_eq!(
        loaded.classes(),
        fitted.classes(),
        "classes_ must round-trip"
    );
    assert_eq!(
        loaded.class_log_prior(),
        fitted.class_log_prior(),
        "class_log_prior_ must round-trip"
    );
    assert_eq!(
        loaded.feature_log_prob(&pool),
        fitted.feature_log_prob(&pool),
        "feature_log_prob_ must round-trip"
    );
    assert_eq!(
        loaded.force_alpha(),
        fitted.force_alpha(),
        "force_alpha provenance must round-trip"
    );
    assert_eq!(
        labels_of(&mut pool, &loaded, &x, shape),
        want_labels,
        "a reloaded model must predict the same labels"
    );
    assert_eq!(
        proba_of(&mut pool, &loaded, &x, shape),
        want_proba,
        "a reloaded model must predict bit-identical probabilities"
    );
}

#[test]
fn complement_roundtrip_is_bit_exact() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, y, shape) = count_fixture::<f64>();

    // Both arms of `norm`: it is baked into the fitted weights, so a round-trip
    // that dropped it would produce a model whose `get_params` lied about how
    // its own weights were built.
    for norm in [false, true] {
        let path = dir.path().join(format!("complement_{norm}.safetensors"));
        let fitted = ComplementNB::<f64>::builder()
            .norm(norm)
            .alpha(0.5)
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&mut pool, &x, &y, shape, None)
            .expect("fits");
        let want_labels = labels_of(&mut pool, &fitted, &x, shape);
        let want_proba = proba_of(&mut pool, &fitted, &x, shape);

        fitted.save(&pool, &path).expect("save succeeds");
        let loaded: ComplementNB<f64, Fitted> =
            ComplementNB::load(&mut pool, &path).expect("load succeeds");

        assert_eq!(loaded.classes(), fitted.classes(), "classes_ (norm={norm})");
        assert_eq!(
            loaded.class_log_prior(),
            fitted.class_log_prior(),
            "class_log_prior_ (norm={norm})"
        );
        assert_eq!(
            loaded.feature_log_prob(&pool),
            fitted.feature_log_prob(&pool),
            "feature_log_prob_ (norm={norm})"
        );
        assert_eq!(
            labels_of(&mut pool, &loaded, &x, shape),
            want_labels,
            "labels (norm={norm})"
        );
        assert_eq!(
            proba_of(&mut pool, &loaded, &x, shape),
            want_proba,
            "probabilities (norm={norm})"
        );
    }
}

#[test]
fn bernoulli_roundtrip_is_bit_exact() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, y, shape) = count_fixture::<f64>();

    // `binarize = None` is a real configuration, not an unset default, and it
    // is the arm that would break if the absent-key encoding fell back to
    // sklearn's `Some(0.0)`. Both arms are exercised, and the `None` model is
    // fitted on ALREADY-BINARY input since that is what `None` promises.
    for binarize in [Some(1.5), None] {
        let tag = if binarize.is_some() { "thresh" } else { "none" };
        let path = dir.path().join(format!("bernoulli_{tag}.safetensors"));
        let (bx, by, bshape) = if binarize.is_some() {
            (x.clone(), y.clone(), shape)
        } else {
            let binary: Vec<f64> = x.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
            (binary, y.clone(), shape)
        };

        let fitted = BernoulliNB::<f64>::builder()
            .binarize(binarize)
            .alpha(0.3)
            .build::<f64>()
            .expect("builds")
            .fit_from_host_slice(&mut pool, &bx, &by, bshape, None)
            .expect("fits");
        let want_labels = labels_of(&mut pool, &fitted, &bx, bshape);
        let want_proba = proba_of(&mut pool, &fitted, &bx, bshape);

        fitted.save(&pool, &path).expect("save succeeds");
        let loaded: BernoulliNB<f64, Fitted> =
            BernoulliNB::load(&mut pool, &path).expect("load succeeds");

        assert_eq!(loaded.classes(), fitted.classes(), "classes_ ({tag})");
        assert_eq!(
            loaded.class_log_prior(),
            fitted.class_log_prior(),
            "class_log_prior_ ({tag})"
        );
        assert_eq!(
            loaded.feature_log_prob_delta(&pool),
            fitted.feature_log_prob_delta(&pool),
            "the folded GEMM operand must round-trip ({tag})"
        );
        // `neg_prob_sum_` has no accessor, so the predictions ARE its gate: it
        // enters the joint log-likelihood as a per-class bias, so dropping it
        // would shift every probability.
        assert_eq!(
            labels_of(&mut pool, &loaded, &bx, bshape),
            want_labels,
            "labels ({tag})"
        );
        assert_eq!(
            proba_of(&mut pool, &loaded, &bx, bshape),
            want_proba,
            "probabilities — neg_prob_sum_ and binarize must both round-trip ({tag})"
        );
    }
}

#[test]
fn bernoulli_names_its_matrix_for_what_it_holds() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bernoulli.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let (x, y, shape) = count_fixture::<f64>();
    BernoulliNB::<f64>::builder()
        .build::<f64>()
        .expect("builds")
        .fit_from_host_slice(&mut pool, &x, &y, shape, None)
        .expect("fits")
        .save(&pool, &path)
        .expect("save succeeds");

    // mlrs stores the folded operand `log p − log(1 − p)`, not sklearn's
    // `feature_log_prob_` (= `log p`). Writing it under sklearn's name would
    // hand a Python reader a matrix that is not what the name promises, so the
    // file must NOT claim to carry `feature_log_prob_`.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = NbFile::parse(&raw, "bernoulli_nb").expect("parse succeeds");
    assert!(
        file.tensor_opt("feature_log_prob_delta_").is_some(),
        "the folded operand must be stored under its true name"
    );
    assert!(
        file.tensor_opt("feature_log_prob_").is_none(),
        "the file must not claim to hold sklearn's feature_log_prob_"
    );
    assert!(
        file.tensor_opt("neg_prob_sum_").is_some(),
        "neg_prob_sum_ is fitted state and must be stored"
    );
}

#[test]
fn every_estimator_rejects_every_other_estimators_file() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // One file per estimator, then every cross pairing must be refused. The
    // three discrete variants share a tensor layout almost exactly, so without
    // the `estimator` discriminator a ComplementNB file would load happily as a
    // MultinomialNB and predict confident nonsense — this is the gate that
    // makes that impossible rather than merely unlikely.
    let (cx, cy, cshape) = count_fixture::<f64>();
    let paths = [
        "gaussian",
        "multinomial",
        "bernoulli",
        "complement",
        "categorical",
    ]
    .map(|n| dir.path().join(format!("{n}.safetensors")));

    fit_gaussian::<f64>(&mut pool, None)
        .save(&pool, &paths[0])
        .expect("save");
    MultinomialNB::<f64>::builder()
        .build::<f64>()
        .unwrap()
        .fit_from_host_slice(&mut pool, &cx, &cy, cshape, None)
        .unwrap()
        .save(&pool, &paths[1])
        .expect("save");
    BernoulliNB::<f64>::builder()
        .build::<f64>()
        .unwrap()
        .fit_from_host_slice(&mut pool, &cx, &cy, cshape, None)
        .unwrap()
        .save(&pool, &paths[2])
        .expect("save");
    ComplementNB::<f64>::builder()
        .build::<f64>()
        .unwrap()
        .fit_from_host_slice(&mut pool, &cx, &cy, cshape, None)
        .unwrap()
        .save(&pool, &paths[3])
        .expect("save");
    fit_categorical::<f64>(MinCategories::Infer)
        .save(&pool, &paths[4])
        .expect("save");

    for (i, path) in paths.iter().enumerate() {
        let outcomes = [
            GaussianNB::<f64, Fitted>::load(&mut pool, path).is_ok(),
            MultinomialNB::<f64, Fitted>::load(&mut pool, path).is_ok(),
            BernoulliNB::<f64, Fitted>::load(&mut pool, path).is_ok(),
            ComplementNB::<f64, Fitted>::load(&mut pool, path).is_ok(),
            CategoricalNB::<f64, Fitted>::load(&mut pool, path).is_ok(),
        ];
        for (j, ok) in outcomes.iter().enumerate() {
            assert_eq!(
                *ok,
                i == j,
                "file {i} loaded by estimator {j}: expected {} but got {ok}",
                i == j
            );
        }
    }
}

// ---------------------------------------------------------------------------
// CategoricalNB — the ragged, host-resident shape
// ---------------------------------------------------------------------------

#[test]
fn categorical_roundtrip_is_bit_exact() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("categorical.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_categorical::<f64>(MinCategories::Infer);
    fitted.save(&pool, &path).expect("save succeeds");
    let loaded: CategoricalNB<f64, Fitted> =
        CategoricalNB::load(&mut pool, &path).expect("load succeeds");

    assert_eq!(
        loaded.classes(),
        fitted.classes(),
        "classes_ must round-trip"
    );
    assert_eq!(
        loaded.class_count(),
        fitted.class_count(),
        "class_count_ must round-trip"
    );
    assert_eq!(
        loaded.class_log_prior(),
        fitted.class_log_prior(),
        "class_log_prior_ must round-trip"
    );
    // `n_categories_` is both a fitted attribute AND the ragged block-extent
    // table, so this assertion covers the descriptor and the payload at once.
    assert_eq!(
        loaded.n_categories(),
        fitted.n_categories(),
        "n_categories_ must round-trip"
    );
    assert_eq!(
        loaded.feature_log_prob(),
        fitted.feature_log_prob(),
        "the ragged feature_log_prob_ blocks must round-trip block-for-block"
    );

    // The fixture must actually be ragged, or the flat-CSR path is untested.
    let extents = fitted.n_categories().expect("fitted n_categories_");
    assert!(
        extents.iter().min() != extents.iter().max(),
        "the fixture must have differing per-feature cardinalities, got {extents:?}"
    );
}

#[test]
fn categorical_roundtrip_preserves_predictions() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("categorical.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let (x, _, shape) = categorical_fixture::<f64>();
    let fitted = fit_categorical::<f64>(MinCategories::Infer);
    let want_labels = labels_of(&mut pool, &fitted, &x, shape);
    let want_proba = proba_of(&mut pool, &fitted, &x, shape);

    fitted.save(&pool, &path).expect("save succeeds");
    let loaded: CategoricalNB<f64, Fitted> =
        CategoricalNB::load(&mut pool, &path).expect("load succeeds");

    assert_eq!(
        labels_of(&mut pool, &loaded, &x, shape),
        want_labels,
        "a reloaded model must predict the same labels"
    );
    assert_eq!(
        proba_of(&mut pool, &loaded, &x, shape),
        want_proba,
        "a reloaded model must predict bit-identical probabilities"
    );
}

#[test]
fn ragged_layout_beats_dense_padding() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("categorical.safetensors");

    let client = runtime::active_client();
    let pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_categorical::<f64>(MinCategories::Infer);
    fitted.save(&pool, &path).expect("save succeeds");

    let extents = fitted.n_categories().expect("fitted n_categories_");
    let n_classes = fitted.classes().len();
    let ragged_values: usize = extents.iter().map(|&k| n_classes * k).sum();
    let padded_values = extents.len() * n_classes * extents.iter().copied().max().unwrap();

    // The alternative layout — pad every per-feature block out to the widest
    // one and store a dense `[n_features, n_classes, max_cat]` cube — costs
    // `max_cat / mean_cat` of the payload. On this fixture (cardinalities
    // 2/3/8/2) that is a ~2.1× inflation.
    assert!(
        padded_values > ragged_values * 2,
        "the fixture must make padding expensive: ragged {ragged_values} values \
         vs padded {padded_values}"
    );

    // Now verify the file actually stores the flat form. The first 8 bytes of a
    // safetensors file are the JSON header's length, so the payload is
    // everything past it — and it must come to EXACTLY the ragged element
    // count, with no padding and no offsets tensor beyond `n_categories_`.
    //
    // Deliberately an exact byte count rather than an inequality against the
    // padded size: at this toy width (4 features) the JSON header is ~530 B and
    // dominates the 320 B payload, so any whole-file comparison would measure
    // the header, not the layout. The header is a fixed cost per tensor; the
    // payload is the term that scales with the model, and it is the one this
    // pins.
    let bytes = std::fs::read(&path).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let payload = bytes.len() - 8 - header_len;

    // feature_log_prob_ (ragged f64) + n_categories_ (u64) + the three
    // per-class vectors (i64/f64), every element 8 bytes wide.
    let expected = (ragged_values + extents.len() + 3 * n_classes) * size_of::<f64>();
    assert_eq!(
        payload,
        expected,
        "the on-disk payload must be exactly the ragged element count ({ragged_values} \
         feature_log_prob_ values + {} descriptor/class values), with no padding",
        extents.len() + 3 * n_classes
    );

    // And padding would have inflated that payload by this much:
    let padded_payload = expected + (padded_values - ragged_values) * size_of::<f64>();
    assert!(
        padded_payload > payload + payload / 2,
        "padding must cost real bytes: {payload} B ragged vs {padded_payload} B padded"
    );
}

#[test]
fn categorical_save_writes_the_fitted_buffer_verbatim() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("categorical.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    let fitted = fit_categorical::<f64>(MinCategories::Infer);
    fitted.save(&pool, &path).expect("save succeeds");

    // The estimator holds `feature_log_prob_` in the SAME flat block-major
    // layout the file uses, so saving is a byte-for-byte handoff — no flatten,
    // no repack. Comparing the on-disk payload against the in-memory buffer is
    // the observable form of that claim: if `save` ever reintroduced a
    // transform, these bytes would differ.
    let in_memory = fitted.feature_log_prob().expect("fitted");
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = NbFile::parse(&raw, "categorical_nb").expect("parse succeeds");
    let on_disk = file
        .tensor("feature_log_prob_")
        .expect("feature_log_prob_ is present");
    assert_eq!(
        on_disk.data(),
        bytemuck::cast_slice::<f64, u8>(in_memory),
        "the stored payload must be the fitted buffer verbatim"
    );

    // And a load reproduces that buffer exactly, in one piece.
    let loaded: CategoricalNB<f64, Fitted> =
        CategoricalNB::load(&mut pool, &path).expect("load succeeds");
    assert_eq!(
        loaded.feature_log_prob(),
        Some(in_memory),
        "the loaded buffer must equal the saved one"
    );
}

#[test]
fn feature_log_prob_block_agrees_with_the_flat_layout() {
    if capability::skip_f64_with_log() {
        return;
    }
    let fitted = fit_categorical::<f64>(MinCategories::Infer);
    let flat = fitted.feature_log_prob().expect("fitted");
    let extents = fitted.n_categories().expect("fitted");
    let n_classes = fitted.classes().len();

    // The per-feature accessor and the documented offset arithmetic must not be
    // able to disagree — `feature_log_prob_block(j)` is the only thing standing
    // between a caller and manual index math over the flat buffer, so it is
    // pinned against that math directly.
    let offsets = ragged_block_offsets(n_classes, extents);
    assert_eq!(
        offsets.len(),
        extents.len() + 1,
        "offsets must carry the end sentinel"
    );
    assert_eq!(
        *offsets.last().unwrap(),
        flat.len(),
        "the offsets must account for the whole buffer"
    );
    for (j, &n_cat_j) in extents.iter().enumerate() {
        let block = fitted
            .feature_log_prob_block(j)
            .expect("a fitted feature block");
        assert_eq!(
            block.len(),
            n_classes * n_cat_j,
            "block {j} must be n_classes x n_categories_[{j}]"
        );
        assert_eq!(
            block,
            &flat[offsets[j]..offsets[j + 1]],
            "block {j} must be the flat buffer's own slice"
        );
    }
    assert!(
        fitted.feature_log_prob_block(extents.len()).is_none(),
        "an out-of-range feature index must be None, not a panic"
    );
}

#[test]
fn min_categories_variants_roundtrip() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);

    // All three arms of the enum: the two scalar-shaped ones ride entirely in
    // `__metadata__`, and only `PerFeature` pays for a tensor.
    for (name, spec) in [
        ("infer", MinCategories::Infer),
        ("uniform", MinCategories::Uniform(9)),
        ("per_feature", MinCategories::PerFeature(vec![2, 4, 8, 3])),
    ] {
        let path = dir.path().join(format!("{name}.safetensors"));
        let fitted = fit_categorical::<f64>(spec.clone());
        fitted.save(&pool, &path).expect("save succeeds");
        let loaded: CategoricalNB<f64, Fitted> =
            CategoricalNB::load(&mut pool, &path).expect("load succeeds");
        assert_eq!(
            loaded.n_categories(),
            fitted.n_categories(),
            "min_categories = {spec:?} must reproduce the same padded cardinalities"
        );
        assert_eq!(
            loaded.feature_log_prob(),
            fitted.feature_log_prob(),
            "min_categories = {spec:?} must reproduce the same fitted tables"
        );
    }
}

#[test]
fn the_load_path_is_zero_copy() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gaussian.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_gaussian::<f64>(&mut pool, None)
        .save(&pool, &path)
        .expect("save succeeds");

    // The claim `AlignedBytes` exists to make good: every 8-byte tensor in a
    // file this crate wrote can be reinterpreted from the file buffer with NO
    // copy. safetensors pads its header to a multiple of 8 and emits tensors in
    // descending dtype width, so an 8-aligned base is all it takes — but a
    // `Vec<u8>` from `fs::read` is only guaranteed 1-aligned, which would push
    // every tensor onto the copying fallback in `cast_bytes`. Nothing about
    // that is visible in a round-trip assertion, so it is gated here directly.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = NbFile::parse(&raw, "gaussian_nb").expect("parse succeeds");

    for name in ["theta_", "var_", "class_count_", "class_log_prior_"] {
        let view = file.tensor(name).expect("the tensor is present");
        assert!(
            bytemuck::try_cast_slice::<u8, f64>(view.data()).is_ok(),
            "'{name}' must be reinterpretable as &[f64] without a copy"
        );
    }
    let classes = file.tensor("classes_").expect("classes_ is present");
    assert!(
        bytemuck::try_cast_slice::<u8, i64>(classes.data()).is_ok(),
        "'classes_' must be reinterpretable as &[i64] without a copy"
    );
}

// ---------------------------------------------------------------------------
// Rejection gates — the file is untrusted input
// ---------------------------------------------------------------------------

#[test]
fn a_sibling_estimators_file_is_rejected() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gaussian.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_gaussian::<f64>(&mut pool, None)
        .save(&pool, &path)
        .expect("save succeeds");

    // The `estimator` discriminator is checked BEFORE any tensor is fetched, so
    // this reports what the file actually is rather than a confusing
    // missing-`feature_log_prob_` error that reads like corruption.
    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match CategoricalNB::<f64, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("a GaussianNB file must not load as a CategoricalNB"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "categorical_nb" && found == "gaussian_nb"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_foreign_safetensors_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("foreign.safetensors");

    // A well-formed safetensors container that is simply not an mlrs model —
    // the case a user hits by pointing `load` at some other project's weights.
    let values = vec![1.0f64, 2.0, 3.0];
    let mut w = NbWriter::new("gaussian_nb");
    w.tensor("weights", TensorRef::f64s(&values, vec![3]).unwrap());
    w.write(&path).expect("write succeeds");
    // Strip the mlrs discriminators by rewriting the same tensor through a raw
    // header, i.e. simulate a file no mlrs writer produced.
    let bytes = std::fs::read(&path).unwrap();
    let stripped = bytes
        .windows(b"mlrs-nb".len())
        .position(|w| w == b"mlrs-nb")
        .map(|at| {
            let mut b = bytes.clone();
            b[at..at + b"mlrs-nb".len()].copy_from_slice(b"not-mlr");
            b
        })
        .expect("the format tag is present in the header");
    std::fs::write(&path, stripped).unwrap();

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let err = match GaussianNB::<f64, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("a foreign safetensors container must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::NotAnMlrsModel { .. }),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn an_inconsistent_header_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("tampered.safetensors");

    // A hand-built file whose `class_count_` claims 2 classes while `theta_`'s
    // shape declares 3. Every extent is cross-checked against `theta_` before a
    // single value is stored, so this is a typed error — NOT an out-of-bounds
    // read the first time the model is used (T-04-01-01).
    let theta = vec![0.0f64; 3 * 4];
    let var = vec![1.0f64; 3 * 4];
    let classes = vec![0i64, 1, 2];
    let short = vec![0.5f64; 2];

    let mut w = NbWriter::new("gaussian_nb");
    w.scalar_f64("param:var_smoothing", 1e-9);
    w.scalar_f64("epsilon_", 1e-9);
    w.tensor("theta_", TensorRef::f64s(&theta, vec![3, 4]).unwrap());
    w.tensor("var_", TensorRef::f64s(&var, vec![3, 4]).unwrap());
    w.tensor("classes_", TensorRef::i64s(&classes, vec![3]).unwrap());
    w.tensor("class_count_", TensorRef::f64s(&short, vec![2]).unwrap());
    w.tensor(
        "class_log_prior_",
        TensorRef::f64s(&short, vec![2]).unwrap(),
    );
    w.write(&path).expect("write succeeds");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let err = match GaussianNB::<f64, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("a header whose extents disagree must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_missing_file_reports_its_path() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("absent.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let err = match GaussianNB::<f64, Fitted>::load(&mut pool, &path) {
        Ok(_) => panic!("loading a nonexistent file must fail"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::Io { path: p, .. } if p == &path),
        "an I/O error must name the file it happened on, got {err:?}"
    );
}

#[test]
fn saving_twice_produces_an_identical_model() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    let fitted = fit_gaussian::<f64>(&mut pool, Some(vec![0.5, 0.25, 0.25]));
    fitted.save(&pool, &first).expect("save succeeds");
    fitted.save(&pool, &second).expect("save succeeds");

    // RAW BYTES, not just contents: a model file must be a deterministic
    // function of the model, so it can be content-addressed and deduplicated.
    //
    // This is the gate on the `third_party/safetensors` patch. Stock
    // safetensors serializes `__metadata__` out of a std `HashMap` whose
    // iteration order is randomly seeded, which makes two saves of one model
    // differ in header key order — semantically identical, byte-wise not. The
    // vendored fork retypes those maps to `BTreeMap`. If the `[patch.crates-io]`
    // entry is ever dropped, this assertion is what fails.
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );

    // And the parsed view agrees, which is what makes the byte check meaningful
    // rather than two identically-corrupt files.
    let (raw_a, raw_b) = (
        AlignedBytes::read(&first).expect("read"),
        AlignedBytes::read(&second).expect("read"),
    );
    let (a, b) = (
        NbFile::parse(&raw_a, "gaussian_nb").expect("parse"),
        NbFile::parse(&raw_b, "gaussian_nb").expect("parse"),
    );
    for name in [
        "theta_",
        "var_",
        "classes_",
        "class_count_",
        "class_log_prior_",
        "param:priors",
    ] {
        let (ta, tb) = (a.tensor(name).expect(name), b.tensor(name).expect(name));
        assert_eq!(ta.dtype(), tb.dtype(), "dtype of '{name}' must be stable");
        assert_eq!(ta.shape(), tb.shape(), "shape of '{name}' must be stable");
        assert_eq!(ta.data(), tb.data(), "payload of '{name}' must be stable");
    }
    assert_eq!(
        a.metadata(),
        b.metadata(),
        "the same model must declare the same scalars every save"
    );
}

#[test]
fn metadata_keys_are_written_in_sorted_order() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gaussian.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_gaussian::<f64>(&mut pool, None)
        .save(&pool, &path)
        .expect("save succeeds");

    // Determinism comes from the ORDER being a function of the keys, not from
    // luck — so check the header literally spells them out sorted. A byte
    // comparison of two saves would also pass if both happened to land on the
    // same random order, which for a five-key map is a 1-in-120 coincidence;
    // this cannot pass by chance.
    let raw = std::fs::read(&path).expect("read");
    let header_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&raw[8..8 + header_len]).expect("utf-8 header");

    let meta = header
        .split_once("\"__metadata__\":{")
        .expect("a __metadata__ object")
        .1;
    let meta = &meta[..meta.find('}').expect("the object closes")];
    let keys: Vec<&str> = meta
        .split(',')
        .map(|kv| kv.split(':').next().unwrap().trim_matches('"'))
        .collect();

    let mut sorted = keys.clone();
    sorted.sort_unstable();
    assert_eq!(
        keys, sorted,
        "__metadata__ keys must be serialized in sorted order, got {keys:?}"
    );
    assert!(
        keys.len() >= 5,
        "the fixture must exercise several keys, got {keys:?}"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gaussian.safetensors");

    let client = runtime::active_client();
    let mut pool: BufferPool<ActiveRuntime> = BufferPool::new(client);
    fit_gaussian::<f32>(&mut pool, None)
        .save(&pool, &path)
        .expect("save succeeds");

    // `save` writes to a sibling temporary and renames it into place so an
    // interrupted write cannot replace a good model with a truncated one; the
    // temporary must not survive a successful save.
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .filter(|n| n.to_string_lossy().contains("mlrs-tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "a successful save must leave no temporary file, found {leftovers:?}"
    );
    assert!(Path::new(&path).exists(), "the model file must exist");
}
