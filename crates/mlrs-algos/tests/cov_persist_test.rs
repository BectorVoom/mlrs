//! COV-PERSIST (prototype) — safetensors save/load round-trips for the two
//! `covariance` estimators: `EmpiricalCovariance` and `LedoitWolf`.
//!
//! This family's distinguishing property is that the two files are
//! INDISTINGUISHABLE by structure. Both hold a square `covariance_` and a
//! `location_` of the same shapes and dtypes; the only difference is that
//! `LedoitWolf`'s matrix has been shrunk toward a scaled identity, which no
//! geometry check can detect. The `estimator` discriminator is therefore the
//! whole of the separation, and `sibling_estimators_do_not_cross_load` is the
//! gate that proves it.
//!
//! The other case worth its own gate is `store_precision`. sklearn's flag means
//! "keep the inverse rather than recompute it on demand", and mlrs makes that a
//! real property of the FILE: `precision_` is written only when the model holds
//! it, so a `store_precision = false` model produces a file roughly half the
//! size. Two gates cover it — the size claim, and the cross-check that rejects a
//! file whose flag and tensor disagree, which neither half could catch alone.
//!
//! The remaining gates are the standard container set:
//!
//!   - `*_roundtrip_is_bit_exact` — every fitted matrix survives save→load with
//!     `==`, not a tolerance. Persistence has no numerical error budget.
//!   - `f32_model_writes_a_half_size_file` / `f32_file_loads_into_an_f64_model` —
//!     the dtype-tag claim and its consequence.
//!   - `the_load_path_is_zero_copy` — the `AlignedBytes` claim.
//!   - `saving_twice_produces_an_identical_model` — byte-level determinism, and
//!     the gate on the `third_party/safetensors` `BTreeMap` patch.
//!   - the rejection gates — a decomposition file (the OTHER container), a
//!     non-square `covariance_`, a `location_` disagreeing with it, and a
//!     `LedoitWolf` file missing its `shrinkage_`. The file is untrusted input
//!     (T-04-01-01).
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::covariance::cov_persist::{
    AlignedBytes, CovFile, CovWriter, LoadModel, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::covariance::{EmpiricalCovariance, LedoitWolf};
use mlrs_algos::decomposition::TruncatedSvd;
use mlrs_algos::typestate::{Fit, Fitted};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

const N_SAMPLES: usize = 16;
const N_FEATURES: usize = 5;

/// A deterministic fixture with genuinely correlated columns, so `covariance_`
/// is not near-diagonal and the Ledoit-Wolf shrinkage has something to bite on —
/// a fixture whose shrunk and unshrunk matrices coincided would make the
/// cross-load gate below vacuous.
fn fixture<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES)
        .flat_map(|i| {
            let t = i as f64 * 0.37;
            [
                t.sin(),
                t.sin() * 0.8 + t.cos() * 0.2,
                t.cos(),
                (t * 1.7).sin() * 2.0,
                t * 0.1 - 0.5,
            ]
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

fn fit_empirical<F>(
    p: &mut BufferPool<ActiveRuntime>,
    store_precision: bool,
) -> EmpiricalCovariance<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    EmpiricalCovariance::<F>::builder()
        .store_precision(store_precision)
        .build::<F>()
        .expect("EmpiricalCovariance builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("EmpiricalCovariance fits the fixture")
}

fn fit_ledoit<F>(p: &mut BufferPool<ActiveRuntime>) -> LedoitWolf<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    LedoitWolf::<F>::builder()
        .build::<F>()
        .expect("LedoitWolf builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("LedoitWolf fits the fixture")
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn empirical_covariance_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("emp.safetensors");
    let mut p = pool();

    let fitted = fit_empirical::<f32>(&mut p, true);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: EmpiricalCovariance<f32, Fitted> =
        EmpiricalCovariance::load(&mut p, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits, so any
    // drift at all is a defect in the container, not rounding.
    assert_eq!(
        loaded.covariance_(&p),
        fitted.covariance_(&p),
        "covariance_ must round-trip exactly"
    );
    assert_eq!(
        loaded.location_(&p),
        fitted.location_(&p),
        "location_ must round-trip exactly"
    );
    assert_eq!(
        loaded.precision_(&p).expect("precision_ was stored"),
        fitted.precision_(&p).expect("precision_ was stored"),
        "precision_ must round-trip exactly"
    );
}

#[test]
fn ledoit_wolf_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("lw.safetensors");
    let mut p = pool();

    let fitted = fit_ledoit::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: LedoitWolf<f32, Fitted> = LedoitWolf::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.covariance_(&p),
        fitted.covariance_(&p),
        "covariance_ must round-trip exactly"
    );
    assert_eq!(
        loaded.location_(&p),
        fitted.location_(&p),
        "location_ must round-trip exactly"
    );
    // The one scalar that is not recoverable from any tensor: the unshrunk
    // matrix `shrinkage_` was derived against is never stored.
    assert_eq!(
        loaded.shrinkage_(),
        fitted.shrinkage_(),
        "shrinkage_ must round-trip exactly"
    );
    assert!(
        fitted.shrinkage_() > 0.0,
        "the fixture must produce a non-zero shrinkage, or the gate proves nothing"
    );
}

#[test]
fn roundtrip_is_bit_exact_at_f64() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("emp64.safetensors");
    let mut p = pool();

    let fitted = fit_empirical::<f64>(&mut p, true);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: EmpiricalCovariance<f64, Fitted> =
        EmpiricalCovariance::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.covariance_(&p),
        fitted.covariance_(&p),
        "covariance_ at f64"
    );
    assert_eq!(
        loaded.location_(&p),
        fitted.location_(&p),
        "location_ at f64"
    );
}

// ---------------------------------------------------------------------------
// store_precision — a flag with a real file-size consequence
// ---------------------------------------------------------------------------

#[test]
fn store_precision_false_writes_a_smaller_file_and_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let with = dir.path().join("with.safetensors");
    let without = dir.path().join("without.safetensors");
    let mut p = pool();

    fit_empirical::<f32>(&mut p, true)
        .save(&p, &with)
        .expect("save succeeds");
    fit_empirical::<f32>(&mut p, false)
        .save(&p, &without)
        .expect("save succeeds");

    // sklearn's `store_precision=False` means "recompute on demand rather than
    // keep", and mlrs makes that a property of the FILE rather than only of the
    // in-memory model: the `precision_` tensor is simply absent, so the file is
    // a whole `d × d` matrix smaller. Recomputing the inverse at load instead
    // would be an O(d³) eigen-decomposition on a path that is otherwise one
    // sequential read, and would silently convert the model.
    let with_len = std::fs::metadata(&with).expect("stat").len();
    let without_len = std::fs::metadata(&without).expect("stat").len();
    let matrix = (N_FEATURES * N_FEATURES) as u64 * 4;
    assert!(
        with_len - without_len >= matrix,
        "omitting precision_ must save at least {matrix} bytes \
         (with {with_len}, without {without_len})"
    );

    // And the flag itself round-trips: the reloaded model still reports the
    // attribute as unavailable rather than silently having gained one.
    let loaded: EmpiricalCovariance<f32, Fitted> =
        EmpiricalCovariance::load(&mut p, &without).expect("load succeeds");
    assert!(
        loaded.precision_(&p).is_err(),
        "a store_precision=false model must not come back with a precision_"
    );
}

#[test]
fn a_store_precision_flag_disagreeing_with_the_tensor_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("mismatch.safetensors");
    let mut p = pool();

    // A hand-built file claiming `store_precision = true` with no `precision_`.
    // Neither half is wrong on its own — `true` is what the default builder
    // produces and an absent tensor is what a `false` model writes — so only the
    // CROSS-check catches it. Without that check the model would load with
    // `precision_` silently `None`, and every `precision_()` call would report
    // `NotFitted` on a model whose own header says it stored one.
    let cov = [1.0f32; N_FEATURES * N_FEATURES];
    let loc = [0.0f32; N_FEATURES];
    let mut w = CovWriter::new("empirical_covariance");
    w.scalar_bool("param:assume_centered", false);
    w.scalar_bool("param:store_precision", true);
    w.tensor(
        "covariance_",
        TensorRef::floats(&cov, vec![N_FEATURES, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "location_",
        TensorRef::floats(&loc, vec![N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match EmpiricalCovariance::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a flag disagreeing with the tensor must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
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

    fit_empirical::<f32>(&mut p, true)
        .save(&p, &narrow)
        .expect("save succeeds");
    fit_empirical::<f64>(&mut p, true)
        .save(&p, &wide)
        .expect("save succeeds");

    let narrow_len = std::fs::metadata(&narrow).expect("stat").len();
    let wide_len = std::fs::metadata(&wide).expect("stat").len();

    // The stored dtype is the MODEL's dtype. The payload is two `d × d` matrices
    // plus a `d` vector, and half of that is what an f32 file saves.
    let payload_saved = (2 * N_FEATURES * N_FEATURES + N_FEATURES) as u64 * 4;
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
    let path = dir.path().join("emp.safetensors");
    let mut p = pool();

    let fitted = fit_empirical::<f32>(&mut p, true);
    fitted.save(&p, &path).expect("save succeeds");

    // The file is self-describing, so storing at the model's own width is a
    // STORAGE decision and not a commitment about how it is loaded back.
    let widened: EmpiricalCovariance<f64, Fitted> =
        EmpiricalCovariance::load(&mut p, &path).expect("an f32 file loads into an f64 model");

    let narrow = fitted.covariance_(&p);
    let wide = widened.covariance_(&p);
    assert_eq!(narrow.len(), wide.len(), "the geometry is unchanged");
    for (i, (&n, &w)) in narrow.iter().zip(wide.iter()).enumerate() {
        // f32 → f64 is exact, so `==` and not a tolerance.
        assert_eq!(f64::from(n), w, "covariance_[{i}] must widen exactly");
    }
}

#[test]
fn the_load_path_is_zero_copy() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("emp.safetensors");
    let mut p = pool();
    fit_empirical::<f64>(&mut p, true)
        .save(&p, &path)
        .expect("save succeeds");

    // The claim `AlignedBytes` exists to make good: every 8-byte tensor can be
    // reinterpreted from the file buffer with NO copy. It matters most for the
    // two `d × d` matrices, which are essentially the whole file.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = CovFile::parse(&raw, "empirical_covariance").expect("parse succeeds");
    for name in ["covariance_", "location_", "precision_"] {
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

    let fitted = fit_empirical::<f32>(&mut p, true);
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
fn a_decomposition_file_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("tsvd.safetensors");
    let mut p = pool();

    // The cross-FAMILY gate: only the `format` discriminator separates the
    // containers, and it is checked before any tensor is fetched.
    let x: DeviceArray<ActiveRuntime, f32> = upload::<f32>(&mut p);
    TruncatedSvd::<f32>::builder()
        .n_components(2)
        .build::<f32>()
        .expect("TruncatedSvd builds")
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("TruncatedSvd fits")
        .save(&p, &path)
        .expect("save succeeds");

    // `expect_err` is unavailable: the estimators deliberately do not derive
    // `Debug` (they hold device handles), so the Ok arm is rejected by hand.
    let err = match EmpiricalCovariance::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-decomp file must not load as a covariance"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-cov"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn sibling_estimators_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("lw.safetensors");
    let mut p = pool();

    // The whole of this family's separation. The two files hold a `covariance_`
    // and a `location_` of identical shape and dtype; `LedoitWolf`'s matrix is
    // merely SHRUNK, which nothing structural can detect. Without the
    // `estimator` tag a `LedoitWolf` file would load as an `EmpiricalCovariance`
    // and report a sample covariance that was never computed.
    fit_ledoit::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    let err = match EmpiricalCovariance::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a ledoit_wolf file must not load as an empirical_covariance"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "empirical_covariance" && found == "ledoit_wolf"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_non_square_covariance_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("oblong.safetensors");
    let mut p = pool();

    // The check this family needs that no other does. A `[d, k]` covariance is
    // malformed on its face, and without the squareness guard a downstream
    // Mahalanobis distance or precision solve would index out of range rather
    // than report a bad file.
    let cov = [1.0f32; N_FEATURES * 2];
    let loc = [0.0f32; N_FEATURES];
    let mut w = CovWriter::new("ledoit_wolf");
    w.scalar_bool("param:assume_centered", false);
    w.scalar_f64("shrinkage_", 0.1);
    w.tensor(
        "covariance_",
        TensorRef::floats(&cov, vec![N_FEATURES, 2]).expect("well-formed"),
    );
    w.tensor(
        "location_",
        TensorRef::floats(&loc, vec![N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match LedoitWolf::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a non-square covariance must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_location_disagreeing_with_the_covariance_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ragged.safetensors");
    let mut p = pool();

    // Neither extent is wrong on its own, so only the CROSS-check catches it.
    let cov = [1.0f32; N_FEATURES * N_FEATURES];
    let loc = [0.0f32; N_FEATURES - 1];
    let mut w = CovWriter::new("ledoit_wolf");
    w.scalar_bool("param:assume_centered", false);
    w.scalar_f64("shrinkage_", 0.1);
    w.tensor(
        "covariance_",
        TensorRef::floats(&cov, vec![N_FEATURES, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "location_",
        TensorRef::floats(&loc, vec![N_FEATURES - 1]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match LedoitWolf::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a location_ disagreeing with covariance_ must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn a_ledoit_wolf_file_without_a_shrinkage_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("no-shrinkage.safetensors");
    let mut p = pool();

    // `shrinkage_` is REQUIRED, never defaulted. Substituting `0.0` would report
    // a model that did no shrinkage at all — the one claim a `LedoitWolf` must
    // never make falsely — and nothing in `covariance_` could contradict it,
    // since the unshrunk matrix it was derived from is not stored.
    let cov = [1.0f32; N_FEATURES * N_FEATURES];
    let loc = [0.0f32; N_FEATURES];
    let mut w = CovWriter::new("ledoit_wolf");
    w.scalar_bool("param:assume_centered", false);
    w.tensor(
        "covariance_",
        TensorRef::floats(&cov, vec![N_FEATURES, N_FEATURES]).expect("well-formed"),
    );
    w.tensor(
        "location_",
        TensorRef::floats(&loc, vec![N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match LedoitWolf::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a ledoit_wolf file without shrinkage_ must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::BadMetadata { key } if *key == "shrinkage_"),
        "expected BadMetadata naming shrinkage_, got {err:?}"
    );
}

#[test]
fn save_leaves_no_temporary_behind() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("emp.safetensors");
    let mut p = pool();
    fit_empirical::<f32>(&mut p, true)
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
