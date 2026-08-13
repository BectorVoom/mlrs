//! DECOMP-PERSIST (prototype) — safetensors save/load round-trips for the three
//! `decomposition` estimators: `Pca`, `TruncatedSvd` and `IncrementalPCA`.
//!
//! The three exercise different corners of the shared spectral core.
//!
//! `TruncatedSvd` is the minimal case: its whole fitted state IS the core, four
//! tensors and one scalar. `Pca` is the same plus `mean_` — and that one extra
//! tensor is the ENTIRE difference between the two files, which makes the
//! `estimator` discriminator load bearing rather than decorative: a `Pca` file
//! that loaded as a `TruncatedSvd` would transform UNCENTERED and be silently
//! wrong rather than an error.
//!
//! `IncrementalPCA` is the case that is not merely a transform. Its file is a
//! CONTINUATION point — `n_samples_seen_`, `mean_` and `var_` are what the next
//! `partial_fit` merges against — so its round-trip gate continues both arms
//! with the same batch and compares the results, which is the only assertion
//! that would catch a dropped running statistic. It is also the one estimator
//! whose file is always `F64` regardless of the model's `F`, because
//! `IncrementalSvdState` is `f64` by construction.
//!
//! The gates, in the order they matter:
//!
//!   - `*_roundtrip_is_bit_exact` — every fitted tensor survives save→load with
//!     `==`, not a tolerance. Persistence has no numerical error budget.
//!   - `*_roundtrip_preserves_transform` — the reloaded model maps the fixture
//!     identically, which is the property a user actually cares about.
//!   - `incremental_pca_roundtrip_preserves_the_continuation` — the gate the
//!     other two do not need: a reloaded running model must merge the next batch
//!     exactly as the un-saved one would.
//!   - `pca_and_truncated_svd_files_do_not_cross_load` — the sharpest
//!     discriminator case in the family, one extra tensor apart.
//!   - `f32_model_writes_a_half_size_file` / `f32_file_loads_into_an_f64_model` —
//!     the dtype-tag claim and its consequence.
//!   - `incremental_pca_stores_f64_regardless_of_the_model_width` — the
//!     DELIBERATE exception to that claim, gated so it cannot be "fixed" into a
//!     silent precision loss on the continuation path.
//!   - `the_load_path_is_zero_copy` — the `AlignedBytes` claim.
//!   - `saving_twice_produces_an_identical_model` — byte-level determinism, and
//!     the gate on the `third_party/safetensors` `BTreeMap` patch.
//!   - `the_file_is_the_model_and_little_else` — the minimal-file claim,
//!     measured as payload vs total.
//!   - the rejection gates — a preprocessing file (the OTHER container), a
//!     sibling estimator's file, a header whose spectra disagree with
//!     `components_`, and a zero-extent header. The file is untrusted input
//!     (T-04-01-01).
//!
//! Fixtures are generated in-test: these gates are about the CONTAINER, and
//! comparing a model against itself needs no sklearn reference. The
//! sklearn-parity gates live in `pca_test.rs` / `truncated_svd_test.rs` /
//! `incremental_pca_test.rs`.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::decomposition::decomp_persist::{
    AlignedBytes, DecompFile, DecompWriter, LoadModel, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::decomposition::{IncrementalPCA, Pca, TruncatedSvd};
use mlrs_algos::preprocessing::MaxAbsScaler;
use mlrs_algos::typestate::{Fit, Fitted, PartialFit, Transform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The fixture geometry. `n_components = 2` over 4 features keeps the file small
/// enough to reason about byte by byte in the size gates, while leaving
/// `components_` genuinely rectangular so a transposed store cannot pass.
const N_SAMPLES: usize = 12;
const N_FEATURES: usize = 4;
const N_COMPONENTS: usize = 2;

/// A deterministic, full-rank fixture.
///
/// Hand-written rather than seeded-random: a persistence round-trip is exact or
/// broken, and an RNG would only add a way for the two arms to disagree for
/// reasons that have nothing to do with the file. The columns are mutually
/// non-collinear so the SVD returns `min(n, d)` non-zero singular values and the
/// fit retains the full `N_COMPONENTS` — a rank-deficient fixture would make
/// `param:n_components` and `components_`'s row extent differ, which is a real
/// case but not the one these gates are about.
fn fixture<F: Pod>() -> Vec<F> {
    let rows: [[f64; N_FEATURES]; N_SAMPLES] = [
        [0.31, -1.24, 0.88, 2.10],
        [-0.75, 0.42, 1.63, -0.19],
        [1.28, 0.07, -0.54, 0.93],
        [0.02, 1.85, 0.31, -1.47],
        [-1.11, -0.68, 2.05, 0.24],
        [0.96, 1.32, -1.18, 0.57],
        [2.14, -0.29, 0.46, -0.82],
        [-0.38, 0.71, 1.09, 1.66],
        [1.53, -1.90, 0.12, 0.35],
        [-0.64, 0.15, -1.37, 1.28],
        [0.87, 2.03, 0.79, -0.41],
        [-1.42, 0.56, 1.94, 0.68],
    ];
    rows.iter()
        .flatten()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect()
}

/// A second, DISJOINT batch — only `IncrementalPCA` uses it, to continue a
/// reloaded model past the point it was saved at.
fn second_batch<F: Pod>() -> Vec<F> {
    let rows: [[f64; N_FEATURES]; 6] = [
        [1.07, 0.63, -2.11, 0.44],
        [-0.29, -1.55, 0.72, 1.90],
        [0.81, 0.24, 1.46, -1.03],
        [-1.66, 1.12, 0.05, 0.37],
        [0.45, -0.88, -1.24, 2.31],
        [1.92, 0.36, 0.61, -0.75],
    ];
    rows.iter()
        .flatten()
        .map(|&v| mlrs_core::f64_to_host::<F>(v))
        .collect()
}

fn pool() -> BufferPool<ActiveRuntime> {
    BufferPool::new(runtime::active_client())
}

fn upload<F: Float + CubeElement + Pod>(
    p: &mut BufferPool<ActiveRuntime>,
    host: &[F],
) -> DeviceArray<ActiveRuntime, F> {
    DeviceArray::from_host(p, host)
}

fn fit_pca<F>(p: &mut BufferPool<ActiveRuntime>) -> Pca<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload(p, &fixture::<F>());
    Pca::<F>::builder()
        .n_components(N_COMPONENTS)
        .build::<F>()
        .expect("Pca builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("Pca fits the fixture")
}

fn fit_tsvd<F>(p: &mut BufferPool<ActiveRuntime>) -> TruncatedSvd<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload(p, &fixture::<F>());
    TruncatedSvd::<F>::builder()
        .n_components(N_COMPONENTS)
        .build::<F>()
        .expect("TruncatedSvd builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("TruncatedSvd fits the fixture")
}

/// An `IncrementalPCA` with every hyperparameter off its default, so a
/// round-trip that drops one is visible. `whiten` and `batch_size` are the two
/// that no fitted tensor reflects.
fn fit_ipca<F>(p: &mut BufferPool<ActiveRuntime>) -> IncrementalPCA<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload(p, &fixture::<F>());
    IncrementalPCA::<F>::builder()
        .n_components(N_COMPONENTS)
        .whiten(true)
        .batch_size(Some(6))
        .build::<F>()
        .expect("IncrementalPCA builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("IncrementalPCA fits the fixture")
}

fn transformed<F, T>(p: &mut BufferPool<ActiveRuntime>, model: &T) -> Vec<F>
where
    F: Float + CubeElement + Pod,
    T: Transform<F>,
{
    let x = upload(p, &fixture::<F>());
    model
        .transform(p, &x, (N_SAMPLES, N_FEATURES))
        .expect("transform succeeds on the training geometry")
        .to_host(p)
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn pca_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("pca.safetensors");
    let mut p = pool();

    let fitted = fit_pca::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: Pca<f32, Fitted> = Pca::load(&mut p, &path).expect("load succeeds");

    // Each attribute compared individually with `==` rather than a tolerance:
    // the file stores the exact IEEE bits, so any drift is a defect in the
    // container. Comparing one at a time is also what would catch a positional
    // mix-up between the three same-length spectra.
    assert_eq!(loaded.components(&p), fitted.components(&p), "components_");
    assert_eq!(
        loaded.explained_variance(&p),
        fitted.explained_variance(&p),
        "explained_variance_"
    );
    assert_eq!(
        loaded.explained_variance_ratio(&p),
        fitted.explained_variance_ratio(&p),
        "explained_variance_ratio_"
    );
    assert_eq!(
        loaded.singular_values(&p),
        fitted.singular_values(&p),
        "singular_values_"
    );
    assert_eq!(loaded.mean(&p), fitted.mean(&p), "mean_");

    // The three spectra must be mutually distinct on this fixture, or the
    // per-attribute comparison above could not detect a swap between them.
    assert_ne!(
        fitted.explained_variance(&p),
        fitted.singular_values(&p),
        "the fixture must keep the spectra distinct, or a swap would be invisible"
    );
}

#[test]
fn truncated_svd_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("tsvd.safetensors");
    let mut p = pool();

    let fitted = fit_tsvd::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: TruncatedSvd<f32, Fitted> =
        TruncatedSvd::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.components(&p), fitted.components(&p), "components_");
    assert_eq!(
        loaded.explained_variance(&p),
        fitted.explained_variance(&p),
        "explained_variance_"
    );
    assert_eq!(
        loaded.explained_variance_ratio(&p),
        fitted.explained_variance_ratio(&p),
        "explained_variance_ratio_"
    );
    assert_eq!(
        loaded.singular_values(&p),
        fitted.singular_values(&p),
        "singular_values_"
    );
}

#[test]
fn incremental_pca_roundtrip_is_bit_exact() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ipca.safetensors");
    let mut p = pool();

    let fitted = fit_ipca::<f64>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: IncrementalPCA<f64, Fitted> =
        IncrementalPCA::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.components(&p), fitted.components(&p), "components_");
    assert_eq!(loaded.mean(&p), fitted.mean(&p), "mean_");
    assert_eq!(loaded.var(&p), fitted.var(&p), "var_");
    assert_eq!(
        loaded.singular_values(&p),
        fitted.singular_values(&p),
        "singular_values_"
    );
    // The three hyperparameters and the running count, none of which any tensor
    // reflects.
    assert_eq!(loaded.n_components(), fitted.n_components(), "n_components");
    assert_eq!(loaded.whiten(), fitted.whiten(), "whiten");
    assert_eq!(loaded.batch_size(), fitted.batch_size(), "batch_size");
    assert_eq!(
        loaded.n_samples_seen(),
        fitted.n_samples_seen(),
        "n_samples_seen_"
    );
}

#[test]
fn roundtrip_preserves_transform() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // The gate that would survive even if every fitted tensor were somehow
    // indistinguishable: whatever the file stores, the reloaded model must map
    // the fixture to exactly the same numbers.
    let pca_path = dir.path().join("pca.safetensors");
    let pca = fit_pca::<f32>(&mut p);
    let before = transformed::<f32, _>(&mut p, &pca);
    pca.save(&p, &pca_path).expect("save succeeds");
    let loaded: Pca<f32, Fitted> = Pca::load(&mut p, &pca_path).expect("load succeeds");
    assert_eq!(
        before,
        transformed::<f32, _>(&mut p, &loaded),
        "Pca transform"
    );

    let tsvd_path = dir.path().join("tsvd.safetensors");
    let tsvd = fit_tsvd::<f32>(&mut p);
    let before = transformed::<f32, _>(&mut p, &tsvd);
    tsvd.save(&p, &tsvd_path).expect("save succeeds");
    let loaded: TruncatedSvd<f32, Fitted> =
        TruncatedSvd::load(&mut p, &tsvd_path).expect("load succeeds");
    assert_eq!(
        before,
        transformed::<f32, _>(&mut p, &loaded),
        "TruncatedSvd transform"
    );
}

#[test]
fn incremental_pca_roundtrip_preserves_the_continuation() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ipca.safetensors");
    let mut p = pool();

    // The gate the other two estimators do not need. An `IncrementalPCA` file is
    // a CONTINUATION point, not just a transform: `n_samples_seen_`, `mean_` and
    // `var_` are what the next `partial_fit` weights its merge by. A `load` that
    // dropped any of them would still produce a model that transforms correctly
    // — and would then diverge from the un-saved one on the very next batch,
    // with nothing to signal it. Continuing both arms with the same data is the
    // only assertion that catches that.
    let saved = fit_ipca::<f64>(&mut p);
    saved.save(&p, &path).expect("save succeeds");
    let loaded: IncrementalPCA<f64, Fitted> =
        IncrementalPCA::load(&mut p, &path).expect("load succeeds");

    let batch = upload(&mut p, &second_batch::<f64>());
    let continued_direct = saved
        .partial_fit(&mut p, &batch, None, (6, N_FEATURES))
        .expect("the un-saved model continues");
    let continued_loaded = loaded
        .partial_fit(&mut p, &batch, None, (6, N_FEATURES))
        .expect("the reloaded model continues");

    assert_eq!(
        continued_loaded.n_samples_seen(),
        continued_direct.n_samples_seen(),
        "the merge must count the same total samples"
    );
    assert_eq!(
        continued_loaded.components(&p),
        continued_direct.components(&p),
        "continuing a reloaded model must produce the same components_ as \
         continuing the un-saved one"
    );
    assert_eq!(
        continued_loaded.mean(&p),
        continued_direct.mean(&p),
        "the running mean must merge identically"
    );
    assert_eq!(
        continued_loaded.var(&p),
        continued_direct.var(&p),
        "the running variance must merge identically"
    );
}

// ---------------------------------------------------------------------------
// The dtype-tag claims, and the one deliberate exception
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

    fit_pca::<f32>(&mut p)
        .save(&p, &narrow)
        .expect("save succeeds");
    fit_pca::<f64>(&mut p)
        .save(&p, &wide)
        .expect("save succeeds");

    let narrow_len = std::fs::metadata(&narrow).expect("stat").len();
    let wide_len = std::fs::metadata(&wide).expect("stat").len();

    // The `TensorRef::floats` claim, measured rather than asserted: the stored
    // dtype is the MODEL's dtype. The payload is `components_` (k·d) + three
    // spectra (3k) + `mean_` (d) floats, and half of that is what an f32 file
    // saves.
    let payload_saved = (N_COMPONENTS * N_FEATURES + 3 * N_COMPONENTS + N_FEATURES) as u64 * 4;
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
    let path = dir.path().join("pca.safetensors");
    let mut p = pool();

    let fitted = fit_pca::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");

    // The other half of the dtype-tag claim: the file is self-describing, so
    // storing at the model's own width is a STORAGE decision, not a commitment
    // about how it is loaded back.
    let widened: Pca<f64, Fitted> =
        Pca::load(&mut p, &path).expect("an f32 file loads into an f64 model");

    let narrow = fitted.components(&p);
    let wide = widened.components(&p);
    assert_eq!(narrow.len(), wide.len(), "the geometry is unchanged");
    for (i, (&n, &w)) in narrow.iter().zip(wide.iter()).enumerate() {
        // f32 → f64 is exact (every f32 is representable), so `==` and not a
        // tolerance: the widening must not perturb a single value.
        assert_eq!(f64::from(n), w, "components_[{i}] must widen exactly");
    }
}

#[test]
fn incremental_pca_stores_f64_regardless_of_the_model_width() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let narrow = dir.path().join("f32.safetensors");
    let wide = dir.path().join("f64.safetensors");
    let mut p = pool();

    fit_ipca::<f32>(&mut p)
        .save(&p, &narrow)
        .expect("save succeeds");
    fit_ipca::<f64>(&mut p)
        .save(&p, &wide)
        .expect("save succeeds");

    // The DELIBERATE exception to the dtype-tag claim, gated so it cannot be
    // "optimized" into a silent precision loss. `IncrementalSvdState` is `f64`
    // by construction — the Chan-Golub-LeVeque running update loses its accuracy
    // guarantee at `f32` — so an `IncrementalPCA` file is a continuation point
    // at `f64` whatever the estimator's own width. Narrowing it would not shrink
    // a storage format; it would change the model, and the reloaded one would
    // drift from the un-saved one on every subsequent batch.
    assert_eq!(
        std::fs::metadata(&narrow).expect("stat").len(),
        std::fs::metadata(&wide).expect("stat").len(),
        "an IncrementalPCA file must be F64 at both model widths"
    );

    let raw = AlignedBytes::read(&narrow).expect("read succeeds");
    let file = DecompFile::parse(&raw, "incremental_pca").expect("parse succeeds");
    for name in ["components_", "mean_", "var_"] {
        let view = file.tensor(name).expect("the tensor is present");
        assert_eq!(
            view.dtype(),
            safetensors::Dtype::F64,
            "'{name}' must be stored as F64 even for an IncrementalPCA<f32>"
        );
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
    let path = dir.path().join("pca.safetensors");
    let mut p = pool();
    fit_pca::<f64>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    // The claim `AlignedBytes` exists to make good: every 8-byte tensor in a
    // file this crate wrote can be reinterpreted from the file buffer with NO
    // copy. A `Vec<u8>` from `fs::read` is only guaranteed 1-aligned, which
    // would push every tensor onto the copying fallback in `cast_bytes`.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = DecompFile::parse(&raw, "pca").expect("parse succeeds");
    for name in [
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
        "mean_",
    ] {
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

    let fitted = fit_pca::<f32>(&mut p);
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");

    // RAW BYTES, not just contents: a model file must be a deterministic
    // function of the model, so it can be content-addressed and deduplicated.
    // This is the gate on the `third_party/safetensors` `BTreeMap` patch — stock
    // safetensors serializes `__metadata__` out of a randomly-seeded `HashMap`,
    // which shuffles the header between runs. If the `[patch.crates-io]` entry
    // is ever dropped, this assertion is what fails.
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

#[test]
fn the_file_is_the_model_and_little_else() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("pca.safetensors");
    let mut p = pool();
    fit_pca::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    // The minimal-file claim, measured: nothing derivable is stored, so the only
    // non-payload bytes are the safetensors header itself. `n_components` and
    // `n_features` come off `components_`'s shape, and no fitted quantity is
    // written twice.
    let total = std::fs::metadata(&path).expect("stat").len();
    let payload = (N_COMPONENTS * N_FEATURES + 3 * N_COMPONENTS + N_FEATURES) as u64 * 4;
    assert!(
        total >= payload,
        "the file must hold the whole payload ({payload} bytes), got {total}"
    );
    // The header is five names, five shapes and four metadata entries — a few
    // hundred bytes. The bound is loose on purpose: it is a claim about the
    // ORDER of the overhead, not a hash of the exact JSON, so adding a scalar
    // later does not break it.
    assert!(
        total - payload < 1024,
        "the non-payload overhead must be header-sized, got {} bytes",
        total - payload
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

    // The cross-FAMILY gate. Both containers are safetensors files written by
    // this crate with the same writer; only the `format` discriminator separates
    // them, and it is checked before any tensor is fetched — so this reports what
    // the file actually is rather than a missing-`components_` error that reads
    // like corruption.
    let x: DeviceArray<ActiveRuntime, f32> = upload(&mut p, &fixture::<f32>());
    MaxAbsScaler::<f32>::new()
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("MaxAbsScaler fits")
        .save(&p, &path)
        .expect("save succeeds");

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match Pca::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-prep file must not load as a decomposition"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-decomp"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn pca_and_truncated_svd_files_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let pca_path = dir.path().join("pca.safetensors");
    let tsvd_path = dir.path().join("tsvd.safetensors");
    let mut p = pool();

    // The sharpest discriminator case in the family. The two files hold the SAME
    // four tensors at the same shapes and dtypes; a `Pca` file differs only by
    // carrying one EXTRA tensor. So a `Pca` file loaded as a `TruncatedSvd`
    // would succeed on every geometry check and transform UNCENTERED — silently
    // wrong output, not an error — while the reverse would fail with a
    // missing-`mean_` message that reads like corruption. The `estimator` tag is
    // the only thing that reports either honestly.
    fit_pca::<f32>(&mut p)
        .save(&p, &pca_path)
        .expect("save succeeds");
    fit_tsvd::<f32>(&mut p)
        .save(&p, &tsvd_path)
        .expect("save succeeds");

    let err = match TruncatedSvd::<f32, Fitted>::load(&mut p, &pca_path) {
        Ok(_) => panic!("a pca file must not load as a truncated_svd"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "truncated_svd" && found == "pca"
        ),
        "expected WrongEstimator, got {err:?}"
    );

    let err = match Pca::<f32, Fitted>::load(&mut p, &tsvd_path) {
        Ok(_) => panic!("a truncated_svd file must not load as a pca"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "pca" && found == "truncated_svd"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_spectrum_disagreeing_with_components_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ragged.safetensors");
    let mut p = pool();

    // A hand-built file whose `singular_values_` claims three components while
    // `components_` declares two. Neither extent is wrong on its own, so only the
    // CROSS-check in `read_spectral_core` catches it — and without that check the
    // reload would produce a model whose spectra and matrix disagree, which
    // `inverse_transform` would read out of range.
    let components = [1.0f32; N_COMPONENTS * N_FEATURES];
    let good = [1.0f32; N_COMPONENTS];
    let bad = [1.0f32; N_COMPONENTS + 1];
    let mut w = DecompWriter::new("truncated_svd");
    w.scalar_usize("param:n_components", N_COMPONENTS);
    w.tensor(
        "components_",
        TensorRef::floats(&components, vec![N_COMPONENTS, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "explained_variance_",
        TensorRef::floats(&good, vec![N_COMPONENTS]).expect("well-formed"),
    );
    w.tensor(
        "explained_variance_ratio_",
        TensorRef::floats(&good, vec![N_COMPONENTS]).expect("well-formed"),
    );
    w.tensor(
        "singular_values_",
        TensorRef::floats(&bad, vec![N_COMPONENTS + 1]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match TruncatedSvd::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a spectrum disagreeing with components_ must not load"),
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

    // A zero-component decomposition cannot transform anything, and an empty
    // upload is a landmine on the device backends — so it is rejected at parse
    // time rather than becoming a model that fails later and elsewhere.
    let empty: [f32; 0] = [];
    let mut w = DecompWriter::new("truncated_svd");
    w.scalar_usize("param:n_components", 0);
    w.tensor(
        "components_",
        TensorRef::floats(&empty, vec![0, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "explained_variance_",
        TensorRef::floats(&empty, vec![0]).expect("well-formed"),
    );
    w.tensor(
        "explained_variance_ratio_",
        TensorRef::floats(&empty, vec![0]).expect("well-formed"),
    );
    w.tensor(
        "singular_values_",
        TensorRef::floats(&empty, vec![0]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match TruncatedSvd::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a zero-component decomposition must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_pca_file_without_a_mean_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("no-mean.safetensors");
    let mut p = pool();

    // `mean_` is REQUIRED for a `Pca`, never defaulted to zeros. Substituting a
    // zero mean would produce a model that transforms UNCENTERED — a
    // `TruncatedSvd` wearing a `Pca`'s name, differing from the saved model on
    // every input with nothing to signal it.
    let components = [1.0f32; N_COMPONENTS * N_FEATURES];
    let spectrum = [1.0f32; N_COMPONENTS];
    let mut w = DecompWriter::new("pca");
    w.scalar_usize("param:n_components", N_COMPONENTS);
    w.tensor(
        "components_",
        TensorRef::floats(&components, vec![N_COMPONENTS, N_FEATURES]).expect("well-formed"),
    );
    for name in [
        "explained_variance_",
        "explained_variance_ratio_",
        "singular_values_",
    ] {
        w.tensor(
            name,
            TensorRef::floats(&spectrum, vec![N_COMPONENTS]).expect("well-formed"),
        );
    }
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match Pca::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a pca file without mean_ must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::MissingTensor { tensor } if *tensor == "mean_"),
        "expected MissingTensor naming mean_, got {err:?}"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("pca.safetensors");
    let mut p = pool();
    fit_pca::<f32>(&mut p)
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
    assert!(path.exists(), "the model file must exist");
}
