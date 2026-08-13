//! PREP-PERSIST (prototype) — safetensors save/load round-trips for the six
//! `preprocessing` transformers: `StandardScaler`, `MinMaxScaler`,
//! `MaxAbsScaler`, `RobustScaler`, `Binarizer` and `Normalizer`.
//!
//! The family splits cleanly in two, and both halves need gating.
//!
//! The four SCALERS are per-column statistic vectors — two, three or four of
//! them, all length `n_features` — so the interesting failure is not a lost
//! value but a SWAPPED one: `read_columns` returns its results positionally, and
//! `var_`/`scale_` (or `data_min_`/`min_`) are the same length and similar
//! magnitude, so a name reordered on one side only would produce a file that
//! round-trips its own geometry perfectly and transforms wrongly. Every scaler
//! therefore gets a round-trip gate that compares each vector INDIVIDUALLY, plus
//! a `transform`-equivalence gate that would catch a swap even if the vectors
//! themselves were somehow indistinguishable.
//!
//! `Binarizer` and `Normalizer` are the tensorless half: `fit` learns nothing
//! but `n_features_in_`, so their whole model is `__metadata__` and the file has
//! no tensors at all. That makes them the case where "the model round-tripped"
//! and "the file is well-formed" are almost the same statement, and where the
//! `n_features_in_` scalar is load-bearing rather than redundant.
//!
//! The gates, in the order they matter:
//!
//!   - `*_roundtrip_is_bit_exact` — every fitted vector survives save→load with
//!     `==`, not a tolerance. Persistence has no numerical error budget: a
//!     round-trip that only matches to 1e-5 has a bug, and a band would hide it.
//!   - `*_roundtrip_preserves_transform` — the reloaded transformer maps the
//!     same input to the same output, which is the property a user cares about
//!     and the one a positional swap breaks.
//!   - `*_non_default_params_roundtrip` — save→load→save is byte-stable, which
//!     covers the hyperparameters (`with_mean`, `clip`, `quantile_range`,
//!     `threshold`, `norm`) that have no public accessor to compare directly and
//!     are NOT recoverable from the fitted vectors.
//!   - `f32_model_writes_a_half_size_file` / `f32_file_loads_into_an_f64_model` —
//!     the dtype-tag claim, measured on real files rather than asserted in a
//!     comment, and its consequence: the width is a load-time choice.
//!   - `the_load_path_is_zero_copy` — the `AlignedBytes` claim, which nothing in
//!     a round-trip assertion would reveal.
//!   - `saving_twice_produces_an_identical_model` — byte-level determinism, and
//!     the gate on the `third_party/safetensors` `BTreeMap` patch.
//!   - `a_tensorless_model_file_is_tiny_and_constant_size` — the claim that
//!     `Binarizer`/`Normalizer` files do not grow with the data they were fitted
//!     on, since they store no statistic.
//!   - the rejection gates — a linear file (the OTHER container), a sibling
//!     scaler's file, a header whose vectors disagree in length, a zero-extent
//!     header, and a `n_features_in_` of zero. The file is untrusted input
//!     (T-04-01-01), so an inconsistent header must be a typed error, never an
//!     out-of-bounds read at transform time.
//!   - `save_leaves_no_temporary_behind` — the write-then-rename path.
//!
//! Fixtures are generated in-test rather than loaded from an oracle `.npz`:
//! these gates are about the CONTAINER, and comparing a model against itself
//! needs no sklearn reference. The sklearn-parity gates for the fits themselves
//! live in `preprocessing_test.rs`.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use std::path::Path;

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::linear::linear_regression::LinearRegression;
use mlrs_algos::preprocessing::prep_persist::{
    AlignedBytes, LoadModel, PersistError, PrepFile, PrepWriter, SaveModel, TensorRef,
};
use mlrs_algos::preprocessing::{
    Binarizer, MaxAbsScaler, MinMaxScaler, Norm, Normalizer, RobustScaler, StandardScaler,
};
use mlrs_algos::typestate::{Fit, Fitted, Transform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The fixture geometry. Small enough to reason about the file byte by byte in
/// the size gates, wide enough that a transposed or truncated vector cannot pass
/// by coincidence.
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;

/// A deterministic fixture whose four columns have DIFFERENT means, variances,
/// extrema and medians.
///
/// That is the point, not decoration: every gate below distinguishes vectors by
/// value, so two columns that happened to share a statistic would make a
/// positional swap invisible. Hand-written rather than seeded-random because a
/// persistence round-trip is exact or broken, and an RNG would only add a way
/// for the two arms to disagree for reasons that have nothing to do with the
/// file.
fn fixture<F: Pod>() -> Vec<F> {
    let rows: [[f64; N_FEATURES]; N_SAMPLES] = [
        [0.31, -12.4, 880.0, 2.10],
        [-0.75, 4.2, 1630.0, -0.19],
        [1.28, 0.7, -540.0, 0.93],
        [0.02, 18.5, 310.0, -1.47],
        [-1.11, -6.8, 2050.0, 0.24],
        [0.96, 13.2, -1180.0, 0.57],
        [2.14, -2.9, 460.0, -0.82],
        [-0.38, 7.1, 1090.0, 1.66],
        [1.53, -19.0, 120.0, 0.35],
        [-0.64, 1.5, -1370.0, 1.28],
        [0.87, 20.3, 790.0, -0.41],
        [-1.42, 5.6, 1940.0, 0.68],
    ];
    rows.iter()
        .flatten()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect()
}

/// A fresh pool on the active backend.
fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

fn upload<F: Float + CubeElement + Pod>(
    p: &mut BufferPool<ActiveRuntime>,
) -> DeviceArray<ActiveRuntime, F> {
    DeviceArray::from_host(p, &fixture::<F>())
}

/// `transform` over the fixture's own rows — the observable a user actually
/// depends on, and what every `*_roundtrip_preserves_transform` gate compares.
fn transformed<F, T>(p: &mut BufferPool<ActiveRuntime>, model: &T) -> Vec<F>
where
    F: Float + CubeElement + Pod,
    T: Transform<F>,
{
    let x = upload::<F>(p);
    model
        .transform(p, &x, (N_SAMPLES, N_FEATURES))
        .expect("transform succeeds on the training geometry")
        .to_host(p)
}

// ---------------------------------------------------------------------------
// Fitting helpers — every one moves at least one hyperparameter off its default
// ---------------------------------------------------------------------------

fn fit_standard<F>(p: &mut BufferPool<ActiveRuntime>, with_mean: bool) -> StandardScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    StandardScaler::<F>::builder()
        .with_mean(with_mean)
        .build::<F>()
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("StandardScaler fits the fixture")
}

fn fit_min_max<F>(p: &mut BufferPool<ActiveRuntime>) -> MinMaxScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    MinMaxScaler::<F>::builder()
        // Both off their defaults: `feature_range` is folded into `scale_`/`min_`
        // so a dropped copy is invisible in the vectors, and `clip` is not
        // reflected in any of them at all.
        .feature_range(-3.0, 7.0)
        .clip(true)
        .build::<F>()
        .expect("MinMaxScaler builds with these hyperparameters")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("MinMaxScaler fits the fixture")
}

fn fit_max_abs<F>(p: &mut BufferPool<ActiveRuntime>) -> MaxAbsScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    MaxAbsScaler::<F>::new()
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("MaxAbsScaler fits the fixture")
}

fn fit_robust<F>(p: &mut BufferPool<ActiveRuntime>) -> RobustScaler<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    RobustScaler::<F>::builder()
        // Every knob off its default, so a round-trip that drops one is visible.
        .quantile_range(10.0, 90.0)
        .unit_variance(true)
        .build::<F>()
        .expect("RobustScaler builds with these hyperparameters")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("RobustScaler fits the fixture")
}

fn fit_binarizer<F>(p: &mut BufferPool<ActiveRuntime>) -> Binarizer<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    Binarizer::<F>::with_threshold(0.5)
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("Binarizer fits the fixture")
}

fn fit_normalizer<F>(p: &mut BufferPool<ActiveRuntime>, norm: Norm) -> Normalizer<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    Normalizer::<F>::with_norm(norm)
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("Normalizer fits the fixture")
}

// ---------------------------------------------------------------------------
// Round-trip — the four scalers, vector by vector
// ---------------------------------------------------------------------------

#[test]
fn standard_scaler_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("standard.safetensors");
    let mut p = pool();

    let fitted = fit_standard::<f32>(&mut p, true);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: StandardScaler<f32, Fitted> =
        StandardScaler::load(&mut p, &path).expect("load succeeds");

    // Each vector compared INDIVIDUALLY, with `==` rather than a tolerance. The
    // file stores the exact IEEE bits, so any drift is a defect in the
    // container; and comparing them one at a time is what catches a positional
    // swap in `read_columns`, which a bulk comparison would not.
    assert_eq!(
        loaded.mean(&p),
        fitted.mean(&p),
        "mean_ must round-trip exactly"
    );
    assert_eq!(
        loaded.var(&p),
        fitted.var(&p),
        "var_ must round-trip exactly"
    );
    assert_eq!(
        loaded.scale(&p),
        fitted.scale(&p),
        "scale_ must round-trip exactly"
    );

    // `var_` and `scale_` are only distinguishable here because the fixture's
    // variances are not 1 — assert that, so a future fixture change cannot
    // quietly make the swap-detection above vacuous.
    assert_ne!(
        fitted.var(&p),
        fitted.scale(&p),
        "the fixture must keep var_ and scale_ distinct, or a swap would be invisible"
    );
}

#[test]
fn min_max_scaler_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("minmax.safetensors");
    let mut p = pool();

    let fitted = fit_min_max::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: MinMaxScaler<f32, Fitted> =
        MinMaxScaler::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.data_min(&p),
        fitted.data_min(&p),
        "data_min_ round-trips"
    );
    assert_eq!(
        loaded.data_max(&p),
        fitted.data_max(&p),
        "data_max_ round-trips"
    );
    assert_eq!(loaded.scale(&p), fitted.scale(&p), "scale_ round-trips");
    assert_eq!(loaded.min(&p), fitted.min(&p), "min_ round-trips");
}

#[test]
fn max_abs_scaler_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("maxabs.safetensors");
    let mut p = pool();

    let fitted = fit_max_abs::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: MaxAbsScaler<f32, Fitted> =
        MaxAbsScaler::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.max_abs(&p),
        fitted.max_abs(&p),
        "max_abs_ round-trips"
    );
    assert_eq!(loaded.scale(&p), fitted.scale(&p), "scale_ round-trips");
}

#[test]
fn robust_scaler_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("robust.safetensors");
    let mut p = pool();

    let fitted = fit_robust::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: RobustScaler<f32, Fitted> =
        RobustScaler::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.center(&p), fitted.center(&p), "center_ round-trips");
    assert_eq!(loaded.scale(&p), fitted.scale(&p), "scale_ round-trips");
}

#[test]
fn roundtrip_is_bit_exact_at_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("standard64.safetensors");
    let mut p = pool();

    let fitted = fit_standard::<f64>(&mut p, true);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: StandardScaler<f64, Fitted> =
        StandardScaler::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.mean(&p), fitted.mean(&p), "mean_ round-trips at f64");
    assert_eq!(
        loaded.scale(&p),
        fitted.scale(&p),
        "scale_ round-trips at f64"
    );
}

// ---------------------------------------------------------------------------
// Round-trip — the observable
// ---------------------------------------------------------------------------

#[test]
fn every_transformer_roundtrip_preserves_transform() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // The gate that would survive even if every fitted vector were somehow
    // indistinguishable: whatever the file stores, the reloaded transformer must
    // map the fixture to exactly the same numbers. Run over all six so no member
    // of the family is covered only by its own vector comparison.
    macro_rules! check {
        ($name:literal, $ty:ty, $fitted:expr) => {{
            let path = dir.path().join(concat!($name, ".safetensors"));
            let fitted = $fitted;
            let before = transformed::<f32, _>(&mut p, &fitted);
            fitted.save(&p, &path).expect("save succeeds");
            let loaded: $ty = <$ty>::load(&mut p, &path).expect("load succeeds");
            let after = transformed::<f32, _>(&mut p, &loaded);
            assert_eq!(
                before, after,
                concat!(
                    $name,
                    ": the reloaded transformer must map the fixture identically"
                )
            );
        }};
    }

    check!("standard", StandardScaler<f32, Fitted>, fit_standard::<f32>(&mut p, true));
    check!("minmax", MinMaxScaler<f32, Fitted>, fit_min_max::<f32>(&mut p));
    check!("maxabs", MaxAbsScaler<f32, Fitted>, fit_max_abs::<f32>(&mut p));
    check!("robust", RobustScaler<f32, Fitted>, fit_robust::<f32>(&mut p));
    check!("binarizer", Binarizer<f32, Fitted>, fit_binarizer::<f32>(&mut p));
    check!("normalizer", Normalizer<f32, Fitted>, fit_normalizer::<f32>(&mut p, Norm::L1));
}

// ---------------------------------------------------------------------------
// The hyperparameters with no accessor
// ---------------------------------------------------------------------------

/// save → load → save is byte-stable.
///
/// This is how the scalars WITHOUT a public accessor are gated. `with_mean`,
/// `clip`, `quantile_range`, `threshold` and `norm` are private and none is
/// recoverable from the fitted vectors — `clip` and `threshold` do not appear in
/// them at all, and `feature_range`/`quantile_range` are already folded in — so
/// a `load` that dropped one would still pass every comparison above. Re-saving
/// and comparing raw bytes closes that hole for every key at once: a dropped
/// scalar produces a shorter header, and a defaulted one a different value.
fn assert_resave_is_stable<T: SaveModel + LoadModel>(
    p: &mut BufferPool<ActiveRuntime>,
    dir: &Path,
    name: &str,
    fitted: &T,
) {
    let first = dir.join(format!("{name}-a.safetensors"));
    let second = dir.join(format!("{name}-b.safetensors"));
    fitted.save(p, &first).expect("save succeeds");
    let loaded = T::load(p, &first).expect("load succeeds");
    loaded.save(p, &second).expect("re-save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "{name}: save→load→save must be byte-stable, or a hyperparameter was dropped"
    );
}

#[test]
fn non_default_params_roundtrip() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // Every one of these was fitted with at least one hyperparameter off its
    // default (see the fitting helpers), which is what makes the comparison
    // meaningful — a file whose scalars are all defaults would re-save
    // identically even if `load` ignored them completely.
    let standard = fit_standard::<f32>(&mut p, false);
    assert_resave_is_stable(&mut p, dir.path(), "standard", &standard);
    let min_max = fit_min_max::<f32>(&mut p);
    assert_resave_is_stable(&mut p, dir.path(), "minmax", &min_max);
    let robust = fit_robust::<f32>(&mut p);
    assert_resave_is_stable(&mut p, dir.path(), "robust", &robust);
    let binarizer = fit_binarizer::<f32>(&mut p);
    assert_resave_is_stable(&mut p, dir.path(), "binarizer", &binarizer);
    let normalizer = fit_normalizer::<f32>(&mut p, Norm::Max);
    assert_resave_is_stable(&mut p, dir.path(), "normalizer", &normalizer);
}

#[test]
fn the_norm_is_not_defaulted_on_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // The sharpest case in the tensorless half. A `Normalizer` file is nothing
    // but `__metadata__`, so if `load` fell back to the `'l2'` default for an
    // unrecognised or missing norm, the model would still load, still transform,
    // and still be wrong. Each variant must survive its own round-trip and
    // produce a DIFFERENT transform from the others.
    let mut outputs = Vec::new();
    for (i, norm) in [Norm::L1, Norm::L2, Norm::Max].into_iter().enumerate() {
        let path = dir.path().join(format!("norm{i}.safetensors"));
        let fitted = fit_normalizer::<f32>(&mut p, norm);
        fitted.save(&p, &path).expect("save succeeds");
        let loaded: Normalizer<f32, Fitted> =
            Normalizer::load(&mut p, &path).expect("load succeeds");
        outputs.push(transformed::<f32, _>(&mut p, &loaded));
    }
    assert_ne!(
        outputs[0], outputs[1],
        "l1 and l2 must transform differently"
    );
    assert_ne!(
        outputs[1], outputs[2],
        "l2 and max must transform differently"
    );
}

#[test]
fn an_unrecognised_norm_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-norm.safetensors");
    let mut p = pool();

    // A file from a hypothetical future build that grew a fourth variant. It
    // must fail by NAME rather than silently fall back to `'l2'` and rescale
    // every row differently than the transformer that wrote it.
    let mut w = PrepWriter::new("normalizer");
    w.scalar_str("param:norm", "linf");
    w.scalar_usize("n_features_in_", N_FEATURES);
    w.write(&path)
        .expect("the hand-written file is well-formed");

    let err = match Normalizer::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an unrecognised norm must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::BadMetadata { key } if *key == "param:norm"),
        "expected BadMetadata naming param:norm, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// The dtype-tag claims
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

    fit_standard::<f32>(&mut p, true)
        .save(&p, &narrow)
        .expect("save succeeds");
    fit_standard::<f64>(&mut p, true)
        .save(&p, &wide)
        .expect("save succeeds");

    let narrow_len = std::fs::metadata(&narrow).expect("stat").len();
    let wide_len = std::fs::metadata(&wide).expect("stat").len();

    // The `TensorRef::floats` claim, measured rather than asserted: the stored
    // dtype is the MODEL's dtype, so an f32 scaler's payload is half its f64
    // twin's. The headers are the same size (identical names and shapes; only
    // the dtype tag and the offsets differ in length), so the difference is
    // exactly the 3 × n_features floats saved.
    let payload_saved = 3 * N_FEATURES as u64 * 4;
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
    let path = dir.path().join("f32.safetensors");
    let mut p = pool();

    let fitted = fit_standard::<f32>(&mut p, true);
    fitted.save(&p, &path).expect("save succeeds");

    // The other half of the dtype-tag claim: the file is self-describing, so
    // storing at the model's own width is a STORAGE decision and not a
    // commitment about how it is loaded back. Fit on a GPU in f32, evaluate in
    // f64.
    let widened: StandardScaler<f64, Fitted> =
        StandardScaler::load(&mut p, &path).expect("an f32 file loads into an f64 model");

    let narrow = fitted.mean(&p);
    let wide = widened.mean(&p);
    assert_eq!(
        narrow.len(),
        wide.len(),
        "the geometry is unchanged by the widening"
    );
    for (i, (&n, &w)) in narrow.iter().zip(wide.iter()).enumerate() {
        // f32 → f64 is exact (every f32 is representable), so this is `==` and
        // not a tolerance: the widening must not perturb a single value.
        assert_eq!(f64::from(n), w, "mean_[{i}] must widen exactly");
    }
}

// ---------------------------------------------------------------------------
// The format claims
// ---------------------------------------------------------------------------

#[test]
fn the_load_path_is_zero_copy() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("standard.safetensors");
    let mut p = pool();
    fit_standard::<f64>(&mut p, true)
        .save(&p, &path)
        .expect("save succeeds");

    // The claim `AlignedBytes` exists to make good: every 8-byte tensor in a
    // file this crate wrote can be reinterpreted from the file buffer with NO
    // copy, so `load` hands the file's own bytes to `DeviceArray::from_host`. A
    // `Vec<u8>` from `fs::read` is only guaranteed 1-aligned, which would push
    // every tensor onto the copying fallback in `cast_bytes`. Nothing about that
    // is visible in a round-trip assertion, so it is gated here directly.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = PrepFile::parse(&raw, "standard_scaler").expect("parse succeeds");
    for name in ["mean_", "var_", "scale_"] {
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

    let fitted = fit_robust::<f32>(&mut p);
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");

    // RAW BYTES, not just contents: a model file must be a deterministic
    // function of the model, so it can be content-addressed and deduplicated.
    //
    // This is the gate on the `third_party/safetensors` patch. Stock
    // safetensors serializes `__metadata__` out of a std `HashMap` whose
    // iteration order is randomly seeded, which makes two saves of one model
    // differ in header key order — semantically identical, byte-wise not. The
    // vendored fork retypes those maps to `BTreeMap`. If the `[patch.crates-io]`
    // entry is ever dropped, this assertion is what fails. `RobustScaler` is the
    // right subject: it carries five `param:` scalars, so a shuffled map is
    // overwhelmingly likely to reorder at least one.
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

#[test]
fn a_tensorless_model_file_is_tiny_and_constant_size() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let small = dir.path().join("small.safetensors");
    let large = dir.path().join("large.safetensors");
    let mut p = pool();

    // `Binarizer` learns no statistic, so its file must not grow with the data
    // it was shown. Fitting the same transformer against a matrix with 25× the
    // rows must produce a file of exactly the same size — which is the concrete
    // form of "the tensorless half stores `n_features_in_` and nothing else".
    fit_binarizer::<f32>(&mut p)
        .save(&p, &small)
        .expect("save succeeds");

    let big: Vec<f32> = (0..N_SAMPLES * 25 * N_FEATURES)
        .map(|i| (i as f32) * 0.031 - 4.0)
        .collect();
    let big_dev: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(&mut p, &big);
    Binarizer::<f32>::with_threshold(0.5)
        .fit(&mut p, &big_dev, None, (N_SAMPLES * 25, N_FEATURES))
        .expect("Binarizer fits the larger fixture")
        .save(&p, &large)
        .expect("save succeeds");

    let small_len = std::fs::metadata(&small).expect("stat").len();
    let large_len = std::fs::metadata(&large).expect("stat").len();
    assert_eq!(
        small_len, large_len,
        "a tensorless model's file must not grow with its training data"
    );
    // A few hundred bytes of header, and nothing else. The bound is loose on
    // purpose: it is a claim about the ORDER of magnitude, not a hash of the
    // exact JSON, so adding a scalar later does not break it.
    assert!(
        small_len < 512,
        "a tensorless model file must be a header and nothing else, got {small_len} bytes"
    );
}

// ---------------------------------------------------------------------------
// Rejection — the file is untrusted input (T-04-01-01)
// ---------------------------------------------------------------------------

#[test]
fn a_linear_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ols.safetensors");
    let mut p = pool();

    // The cross-FAMILY gate. Both containers are safetensors files written by
    // this crate with the same writer; only the `format` discriminator
    // (`mlrs-linear` vs `mlrs-prep`) separates them, and it is checked before
    // any tensor is fetched — so this reports what the file actually is rather
    // than a missing-`mean_` error that reads like corruption.
    let x: DeviceArray<ActiveRuntime, f32> = upload::<f32>(&mut p);
    let y: DeviceArray<ActiveRuntime, f32> = DeviceArray::from_host(
        &mut p,
        &(0..N_SAMPLES).map(|i| i as f32).collect::<Vec<_>>(),
    );
    LinearRegression::<f32>::builder()
        .build::<f32>()
        .expect("LinearRegression builds")
        .fit(&mut p, &x, Some(&y), (N_SAMPLES, N_FEATURES))
        .expect("LinearRegression fits the fixture")
        .save(&p, &path)
        .expect("save succeeds");

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match StandardScaler::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-linear file must not load as a preprocessing model"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-prep"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn sibling_scalers_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("maxabs.safetensors");
    let mut p = pool();

    // The cross-ESTIMATOR gate, inside one container. `MaxAbsScaler` and
    // `StandardScaler` files are both `mlrs-prep` and both hold a `scale_` of
    // the same shape and dtype, so the `estimator` discriminator is the ONLY
    // thing standing between them — without it, a `MaxAbsScaler` file would load
    // here and its `max_abs_` would be read as a mean.
    fit_max_abs::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    let err = match StandardScaler::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a max_abs_scaler file must not load as a standard_scaler"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "standard_scaler" && found == "max_abs_scaler"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_header_with_mismatched_vector_lengths_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ragged.safetensors");
    let mut p = pool();

    // A hand-built file whose `scale_` is one element short of its `mean_`.
    // Neither extent is wrong on its own — a 4-feature `mean_` and a 3-feature
    // `scale_` are each individually well-formed — so only the CROSS-check in
    // `read_columns` catches it. Without that check the affine map in
    // `transform` would index `scale_[3]` out of range on the first call.
    let mean = [1.0f32, 2.0, 3.0, 4.0];
    let var = [1.0f32, 1.0, 1.0, 1.0];
    let scale = [1.0f32, 1.0, 1.0];
    let mut w = PrepWriter::new("standard_scaler");
    w.scalar_bool("param:with_mean", true);
    w.scalar_bool("param:with_std", true);
    w.tensor(
        "mean_",
        TensorRef::floats(&mean, vec![4]).expect("well-formed"),
    );
    w.tensor(
        "var_",
        TensorRef::floats(&var, vec![4]).expect("well-formed"),
    );
    w.tensor(
        "scale_",
        TensorRef::floats(&scale, vec![3]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match StandardScaler::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a ragged column set must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_zero_extent_header_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("empty.safetensors");
    let mut p = pool();

    // A zero-feature scaler cannot transform anything, and an empty upload is a
    // landmine on the device backends — so it is rejected at parse time rather
    // than becoming a model that fails later and elsewhere.
    let empty: [f32; 0] = [];
    let mut w = PrepWriter::new("max_abs_scaler");
    w.tensor(
        "max_abs_",
        TensorRef::floats(&empty, vec![0]).expect("well-formed"),
    );
    w.tensor(
        "scale_",
        TensorRef::floats(&empty, vec![0]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match MaxAbsScaler::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a zero-feature scaler must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_tensorless_model_with_zero_features_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("zero.safetensors");
    let mut p = pool();

    // The tensorless half's version of the gate above. `Binarizer` has no tensor
    // whose shape could be checked, so `n_features_in_` is the ONLY place the
    // non-degeneracy rule can be applied — and it has to be, or a hand-written
    // header would produce a transformer that accepts only an empty matrix.
    let mut w = PrepWriter::new("binarizer");
    w.scalar_f64("param:threshold", 0.0);
    w.scalar_usize("n_features_in_", 0);
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match Binarizer::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a zero-feature Binarizer must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_missing_required_scalar_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("no-clip.safetensors");
    let mut p = pool();

    // `param:clip` is REQUIRED, never defaulted. It is invisible in every fitted
    // vector, so a `load` that substituted `false` for a missing key would hand
    // back a scaler that silently stops clamping out-of-range input — a model
    // that differs from the saved one with nothing to signal it.
    let v = [1.0f32; N_FEATURES];
    let mut w = PrepWriter::new("min_max_scaler");
    w.scalar_f64("param:feature_range_min", 0.0);
    w.scalar_f64("param:feature_range_max", 1.0);
    for name in ["data_min_", "data_max_", "scale_", "min_"] {
        w.tensor(
            name,
            TensorRef::floats(&v, vec![N_FEATURES]).expect("well-formed"),
        );
    }
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match MinMaxScaler::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a file missing param:clip must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::BadMetadata { key } if *key == "param:clip"),
        "expected BadMetadata naming param:clip, got {err:?}"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("standard.safetensors");
    let mut p = pool();
    fit_standard::<f32>(&mut p, true)
        .save(&p, &path)
        .expect("save succeeds");

    // `save` writes to a sibling temporary and renames it into place so an
    // interrupted write cannot replace a good model with a truncated one; the
    // temporary must not survive a successful save.
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
    assert!(Path::new(&path).exists(), "the model file must exist");
}
