//! PROJ-PERSIST (prototype) — safetensors save/load round-trips for the two
//! random-projection estimators: `GaussianRandomProjection` and
//! `SparseRandomProjection`.
//!
//! This family is the one where the file's whole reason for existing is a claim
//! about REPRODUCIBILITY rather than about size. `components_` is a
//! deterministic function of `(seed, eps, geometry)`, so the tempting design is
//! to store those and regenerate — which would shrink the file to a few dozen
//! bytes and tie its meaning to whichever mlrs build happens to read it. mlrs
//! stores the matrix instead, and the gates below are what hold that line:
//!
//!   - `the_matrix_is_stored_not_regenerated` — the file grows with the model,
//!     which is the observable difference between storing and regenerating.
//!   - `a_reloaded_projection_does_not_depend_on_the_rng` — the sharp one. A
//!     model loaded from a file must transform identically even when the seed
//!     recorded in that file would generate a DIFFERENT matrix, which is exactly
//!     what an RNG change would look like from the file's side.
//!
//! Beyond that the family is the simplest in mlrs: one tensor, three or five
//! scalars. The remaining gates are the standard container set —
//!
//!   - `*_roundtrip_is_bit_exact` — `components_` survives save→load with `==`,
//!     not a tolerance. Persistence has no numerical error budget.
//!   - `*_roundtrip_preserves_transform` — the reloaded model maps the fixture
//!     identically, which is the property a user actually cares about.
//!   - `n_components_auto_roundtrips` and `n_components_fixed_roundtrips` — the
//!     two arms of the enum-shaped hyperparameter, which rides as its sklearn
//!     STRING and is NOT recoverable from `components_`'s row extent (under
//!     `auto` that extent is the JL bound, a different fact from the request).
//!   - `the_fitted_density_is_stored_separately_from_the_request` —
//!     `SparseRandomProjection`'s `density=None` means "1/sqrt(n_features)", so
//!     the request and the outcome are two facts and both round-trip.
//!   - `f32_model_writes_a_half_size_file` / `f32_file_loads_into_an_f64_model` —
//!     the dtype-tag claim and its consequence. This is the family where it
//!     matters most: `components_` is not merely the largest part of the file,
//!     it is essentially all of it.
//!   - `the_load_path_is_zero_copy`, `saving_twice_produces_an_identical_model` —
//!     the `AlignedBytes` and determinism claims.
//!   - the rejection gates — a decomposition file (the OTHER container, and one
//!     that also holds a `components_`), a sibling projection's file, a
//!     zero-extent header, and an unparsable `n_components`.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::decomposition::TruncatedSvd;
use mlrs_algos::projection::proj_persist::{
    AlignedBytes, LoadModel, PersistError, ProjFile, ProjWriter, SaveModel, TensorRef,
};
use mlrs_algos::projection::{GaussianRandomProjection, NComponents, SparseRandomProjection};
use mlrs_algos::typestate::{Fit, Fitted, Transform};
use mlrs_backend::capability;
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

/// The fixture geometry. Wide enough that `components_` dominates the file, so
/// the size gates measure the payload rather than the header.
const N_SAMPLES: usize = 24;
const N_FEATURES: usize = 16;
const N_COMPONENTS: usize = 6;

/// A deterministic fixture. The VALUES do not matter to a random projection —
/// the matrix comes from the RNG, not the data — but the geometry does, and a
/// fixed input is what makes the `transform`-equivalence gates comparable.
fn fixture<F: Pod>() -> Vec<F> {
    (0..N_SAMPLES * N_FEATURES)
        .map(|i| {
            let v = ((i * 37) % 101) as f64 / 50.0 - 1.0;
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

fn fit_gaussian<F>(
    p: &mut BufferPool<ActiveRuntime>,
    n_components: NComponents,
    seed: u64,
) -> GaussianRandomProjection<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    GaussianRandomProjection::<F>::builder()
        .n_components(n_components)
        .seed(seed)
        .eps(0.4)
        .build::<F>()
        .expect("GaussianRandomProjection builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("GaussianRandomProjection fits the fixture")
}

fn fit_sparse<F>(
    p: &mut BufferPool<ActiveRuntime>,
    density: Option<f64>,
) -> SparseRandomProjection<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    SparseRandomProjection::<F>::builder()
        .n_components(NComponents::Fixed(N_COMPONENTS))
        .seed(99)
        .eps(0.4)
        .density(density)
        .build::<F>()
        .expect("SparseRandomProjection builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("SparseRandomProjection fits the fixture")
}

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
// The reproducibility claim — why the matrix is stored rather than regenerated
// ---------------------------------------------------------------------------

#[test]
fn a_reloaded_projection_does_not_depend_on_the_rng() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gauss.safetensors");
    let mut p = pool();

    // The sharpest gate in this family. A model loaded from a file must
    // transform using the matrix IN the file, never one regenerated from the
    // seed it records.
    //
    // The test constructs the situation an RNG change would create: a file whose
    // recorded `seed` generates a DIFFERENT matrix from the one stored. Here that
    // is done by hand-writing a file that pairs one model's `components_` with
    // another model's seed — but the failure mode it stands for is real and
    // silent, a future edit to `prims::rng`'s stream turning every saved
    // projection into a different transform with nothing to signal it.
    let model = fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 7);
    let expected = transformed::<f32, _>(&mut p, &model);
    let components = model.components(&p);

    // Confirm the two seeds really do disagree, or the gate would be vacuous.
    let other = fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 12_345);
    assert_ne!(
        components,
        other.components(&p),
        "the two seeds must generate different matrices, or this gate proves nothing"
    );

    let mut w = ProjWriter::new("gaussian_random_projection");
    w.scalar_usize("param:n_components", N_COMPONENTS);
    // The MISMATCHED seed: a regenerating loader would use this and produce
    // `other`'s transform instead of `model`'s.
    w.scalar_u64("param:seed", 12_345);
    w.scalar_f64("param:eps", 0.4);
    w.tensor(
        "components_",
        TensorRef::floats(&components, vec![N_COMPONENTS, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed");

    let loaded: GaussianRandomProjection<f32, Fitted> =
        GaussianRandomProjection::load(&mut p, &path).expect("load succeeds");
    assert_eq!(
        transformed::<f32, _>(&mut p, &loaded),
        expected,
        "a loaded projection must use the stored matrix, never one regenerated \
         from the recorded seed"
    );
}

#[test]
fn the_matrix_is_stored_not_regenerated() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let small = dir.path().join("small.safetensors");
    let large = dir.path().join("large.safetensors");
    let mut p = pool();

    // The observable difference between storing and regenerating: a stored
    // matrix makes the file grow with the model. A regenerating format would
    // write the same few dozen bytes for both of these.
    fit_gaussian::<f32>(&mut p, NComponents::Fixed(2), 7)
        .save(&p, &small)
        .expect("save succeeds");
    fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 7)
        .save(&p, &large)
        .expect("save succeeds");

    let small_len = std::fs::metadata(&small).expect("stat").len();
    let large_len = std::fs::metadata(&large).expect("stat").len();
    let extra_rows = (N_COMPONENTS - 2) as u64 * N_FEATURES as u64 * 4;
    assert!(
        large_len - small_len >= extra_rows,
        "a file storing the matrix must grow by at least {extra_rows} bytes for \
         {} extra components (small {small_len}, large {large_len})",
        N_COMPONENTS - 2
    );
}

// ---------------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------------

#[test]
fn gaussian_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gauss.safetensors");
    let mut p = pool();

    let fitted = fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 7);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: GaussianRandomProjection<f32, Fitted> =
        GaussianRandomProjection::load(&mut p, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits, so any
    // drift at all is a defect in the container, not rounding.
    assert_eq!(
        loaded.components(&p),
        fitted.components(&p),
        "components_ must round-trip exactly"
    );
    assert_eq!(
        loaded.n_components_(),
        fitted.n_components_(),
        "n_components_ comes off the matrix shape"
    );
}

#[test]
fn sparse_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("sparse.safetensors");
    let mut p = pool();

    let fitted = fit_sparse::<f32>(&mut p, Some(0.6));
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: SparseRandomProjection<f32, Fitted> =
        SparseRandomProjection::load(&mut p, &path).expect("load succeeds");

    assert_eq!(
        loaded.components(&p),
        fitted.components(&p),
        "components_ must round-trip exactly"
    );
    assert_eq!(
        loaded.density_(),
        fitted.density_(),
        "density_ must round-trip"
    );
}

#[test]
fn roundtrip_preserves_transform() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    let gauss_path = dir.path().join("gauss.safetensors");
    let gauss = fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 7);
    let before = transformed::<f32, _>(&mut p, &gauss);
    gauss.save(&p, &gauss_path).expect("save succeeds");
    let loaded: GaussianRandomProjection<f32, Fitted> =
        GaussianRandomProjection::load(&mut p, &gauss_path).expect("load succeeds");
    assert_eq!(
        before,
        transformed::<f32, _>(&mut p, &loaded),
        "the reloaded Gaussian projection must map the fixture identically"
    );

    let sparse_path = dir.path().join("sparse.safetensors");
    let sparse = fit_sparse::<f32>(&mut p, None);
    let before = transformed::<f32, _>(&mut p, &sparse);
    sparse.save(&p, &sparse_path).expect("save succeeds");
    let loaded: SparseRandomProjection<f32, Fitted> =
        SparseRandomProjection::load(&mut p, &sparse_path).expect("load succeeds");
    assert_eq!(
        before,
        transformed::<f32, _>(&mut p, &loaded),
        "the reloaded sparse projection must map the fixture identically"
    );
}

// ---------------------------------------------------------------------------
// The enum-shaped hyperparameter
// ---------------------------------------------------------------------------

#[test]
fn n_components_auto_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("auto.safetensors");
    let mut p = pool();

    // Under `auto` the fitted row extent came out of
    // `johnson_lindenstrauss_min_dim(n_samples, eps)` — a DIFFERENT fact from
    // the request, and not one the request can be inferred back from. So the
    // string `"auto"` has to be stored, and re-saving is what proves it was.
    let fitted = fit_gaussian::<f32>(&mut p, NComponents::Auto, 7);
    fitted.save(&p, &path).expect("save succeeds");

    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = ProjFile::parse(&raw, "gaussian_random_projection").expect("parse succeeds");
    assert_eq!(
        file.scalar_str("param:n_components")
            .expect("the key is present"),
        "auto",
        "the 'auto' request must be stored as its sklearn string"
    );

    // And the outcome is separately recoverable, off the matrix shape.
    let loaded: GaussianRandomProjection<f32, Fitted> =
        GaussianRandomProjection::load(&mut p, &path).expect("load succeeds");
    assert_eq!(
        loaded.n_components_(),
        fitted.n_components_(),
        "the JL-derived component count must survive as the matrix's row extent"
    );
    assert_ne!(
        loaded.n_components_(),
        0,
        "the auto path must have produced a real embedding dimension"
    );
}

#[test]
fn n_components_fixed_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("fixed.safetensors");
    let mut p = pool();

    let fitted = fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 7);
    fitted.save(&p, &path).expect("save succeeds");

    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = ProjFile::parse(&raw, "gaussian_random_projection").expect("parse succeeds");
    assert_eq!(
        file.scalar_str("param:n_components")
            .expect("the key is present"),
        N_COMPONENTS.to_string(),
        "a fixed request must be stored as its decimal, not as 'auto'"
    );
    assert_eq!(
        fitted.n_components_(),
        N_COMPONENTS,
        "and the fit honored it"
    );
}

#[test]
fn an_unparsable_n_components_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad.safetensors");
    let mut p = pool();

    // Neither `"auto"` nor a decimal. It must fail by NAME rather than fall back
    // to `Auto`: the two arms size the embedding differently, so guessing would
    // hand back a model whose reported hyperparameter never produced its own
    // `components_`.
    let components = [1.0f32; N_COMPONENTS * N_FEATURES];
    let mut w = ProjWriter::new("gaussian_random_projection");
    w.scalar_str("param:n_components", "automatic");
    w.scalar_u64("param:seed", 7);
    w.scalar_f64("param:eps", 0.4);
    w.tensor(
        "components_",
        TensorRef::floats(&components, vec![N_COMPONENTS, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match GaussianRandomProjection::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an unparsable n_components must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::BadMetadata { key } if *key == "param:n_components"),
        "expected BadMetadata naming param:n_components, got {err:?}"
    );
}

#[test]
fn the_fitted_density_is_stored_separately_from_the_request() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let requested = dir.path().join("requested.safetensors");
    let defaulted = dir.path().join("defaulted.safetensors");
    let mut p = pool();

    // `density=None` MEANS "use 1/sqrt(n_features)", so the request and the
    // outcome are two different facts. A format that stored only one of them
    // would either lose the `None` (reporting a density the user never asked
    // for) or lose the resolved value (which `components_` does not carry — you
    // cannot read a probability off one sample of it).
    let with_request = fit_sparse::<f32>(&mut p, Some(0.6));
    with_request.save(&p, &requested).expect("save succeeds");
    let without = fit_sparse::<f32>(&mut p, None);
    without.save(&p, &defaulted).expect("save succeeds");

    let raw = AlignedBytes::read(&requested).expect("read succeeds");
    let file = ProjFile::parse(&raw, "sparse_random_projection").expect("parse succeeds");
    assert_eq!(
        file.scalar_opt_f64("param:density").expect("parses"),
        Some(0.6),
        "an explicit density must be recorded"
    );

    let raw = AlignedBytes::read(&defaulted).expect("read succeeds");
    let file = ProjFile::parse(&raw, "sparse_random_projection").expect("parse succeeds");
    assert_eq!(
        file.scalar_opt_f64("param:density").expect("parses"),
        None,
        "a None density must write no key at all, not a sentinel"
    );

    // Both round-trip, and the defaulted one still reports what it resolved to.
    let loaded: SparseRandomProjection<f32, Fitted> =
        SparseRandomProjection::load(&mut p, &defaulted).expect("load succeeds");
    assert_eq!(
        loaded.density_(),
        without.density_(),
        "the RESOLVED density must round-trip even when the request was None"
    );
    // The defaulted density must be the `1/sqrt(n_features)` rule and NOT the
    // explicit request the sibling fixture used, or the round-trip above could
    // pass with the two conflated.
    assert_eq!(
        loaded.density_(),
        1.0 / (N_FEATURES as f64).sqrt(),
        "a None request must resolve to 1/sqrt(n_features)"
    );
    assert_ne!(
        loaded.density_(),
        with_request.density_(),
        "the two fixtures must differ, or this gate proves nothing"
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

    fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 7)
        .save(&p, &narrow)
        .expect("save succeeds");
    fit_gaussian::<f64>(&mut p, NComponents::Fixed(N_COMPONENTS), 7)
        .save(&p, &wide)
        .expect("save succeeds");

    let narrow_len = std::fs::metadata(&narrow).expect("stat").len();
    let wide_len = std::fs::metadata(&wide).expect("stat").len();

    // The `TensorRef::floats` claim, measured rather than asserted. It matters
    // more here than anywhere else in mlrs: `components_` is essentially the
    // whole file, so the dtype choice halves the model outright.
    let payload_saved = (N_COMPONENTS * N_FEATURES) as u64 * 4;
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
    let path = dir.path().join("gauss.safetensors");
    let mut p = pool();

    let fitted = fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 7);
    fitted.save(&p, &path).expect("save succeeds");

    let widened: GaussianRandomProjection<f64, Fitted> =
        GaussianRandomProjection::load(&mut p, &path).expect("an f32 file loads into an f64 model");

    let narrow = fitted.components(&p);
    let wide = widened.components(&p);
    assert_eq!(narrow.len(), wide.len(), "the geometry is unchanged");
    for (i, (&n, &w)) in narrow.iter().zip(wide.iter()).enumerate() {
        // f32 → f64 is exact, so `==` and not a tolerance.
        assert_eq!(f64::from(n), w, "components_[{i}] must widen exactly");
    }
}

#[test]
fn the_load_path_is_zero_copy() {
    if capability::skip_f64_with_log() {
        return;
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("gauss.safetensors");
    let mut p = pool();
    fit_gaussian::<f64>(&mut p, NComponents::Fixed(N_COMPONENTS), 7)
        .save(&p, &path)
        .expect("save succeeds");

    // The claim `AlignedBytes` exists to make good: the matrix — which is the
    // whole model here — reaches `DeviceArray::from_host` as a borrow of the
    // file's own bytes, with no copy and no decode.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = ProjFile::parse(&raw, "gaussian_random_projection").expect("parse succeeds");
    let view = file.tensor("components_").expect("the tensor is present");
    assert!(
        bytemuck::try_cast_slice::<u8, f64>(view.data()).is_ok(),
        "'components_' must be reinterpretable as &[f64] without a copy"
    );
}

#[test]
fn saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    let fitted = fit_sparse::<f32>(&mut p, Some(0.6));
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");

    // RAW BYTES: a model file must be a deterministic function of the model, so
    // it can be content-addressed and deduplicated. This is also the gate on the
    // `third_party/safetensors` `BTreeMap` patch — stock safetensors serializes
    // `__metadata__` out of a randomly-seeded `HashMap`, which shuffles the
    // header between runs. `SparseRandomProjection` carries five scalars, so a
    // shuffled map is overwhelmingly likely to reorder one.
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

    // The cross-FAMILY gate, and the pointed one: `mlrs-decomp` also stores a
    // `components_` of the same rank and dtype. Only the `format` discriminator
    // separates a fitted decomposition's directions from a random projection's,
    // and it is checked before any tensor is fetched — so a `TruncatedSvd` file
    // cannot be silently reused as a projection matrix.
    let x: DeviceArray<ActiveRuntime, f32> = upload::<f32>(&mut p);
    TruncatedSvd::<f32>::builder()
        .n_components(N_COMPONENTS)
        .build::<f32>()
        .expect("TruncatedSvd builds")
        .fit(&mut p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("TruncatedSvd fits")
        .save(&p, &path)
        .expect("save succeeds");

    let err = match GaussianRandomProjection::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-decomp file must not load as a projection"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-proj"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn sibling_projections_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("sparse.safetensors");
    let mut p = pool();

    // The cross-ESTIMATOR gate, inside one container. Both files hold a
    // `components_` of the same shape and dtype; the matrices differ only in
    // their VALUE distribution — dense Gaussian against a mostly-zero Achlioptas
    // draw — which no geometry check could separate. The `estimator` tag is the
    // only thing that does.
    fit_sparse::<f32>(&mut p, Some(0.6))
        .save(&p, &path)
        .expect("save succeeds");

    let err = match GaussianRandomProjection::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a sparse projection file must not load as a Gaussian one"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "gaussian_random_projection"
                    && found == "sparse_random_projection"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_zero_extent_header_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("empty.safetensors");
    let mut p = pool();

    // A projection with no components cannot transform anything, and an empty
    // upload is a landmine on the device backends — so it is rejected at parse
    // time rather than becoming a model that fails later and elsewhere.
    let empty: [f32; 0] = [];
    let mut w = ProjWriter::new("gaussian_random_projection");
    w.scalar_str("param:n_components", "auto");
    w.scalar_u64("param:seed", 7);
    w.scalar_f64("param:eps", 0.4);
    w.tensor(
        "components_",
        TensorRef::floats(&empty, vec![0, N_FEATURES]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match GaussianRandomProjection::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a zero-component projection must not load"),
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
    let path = dir.path().join("gauss.safetensors");
    let mut p = pool();
    fit_gaussian::<f32>(&mut p, NComponents::Fixed(N_COMPONENTS), 7)
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
