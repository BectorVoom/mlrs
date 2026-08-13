//! CLUSTER-PERSIST (prototype) — safetensors save/load round-trips for the six
//! `cluster` estimators: `KMeans`, `DBSCAN`, `AgglomerativeClustering`,
//! `Hdbscan`, `SpectralClustering` and `SpectralEmbedding`.
//!
//! This is the family where "save the model" needs stating carefully, because
//! most of its members have no `predict`. `DBSCAN`, `AgglomerativeClustering`
//! and `SpectralClustering` are `fit_predict`-only: their entire output is the
//! labeling of the rows they were FITTED on. So for those, "the round-trip is
//! faithful" means "a reloaded estimator reports every attribute the saved one
//! did" — which is what the `*_roundtrip_is_bit_exact` gates check, attribute by
//! attribute, rather than by comparing a prediction that does not exist.
//!
//! `KMeans` is the exception that generalizes, so it gets the prediction gate
//! too; `Hdbscan` sits in between and carries the most conditional state in
//! mlrs, so it gets a gate per optional attribute.
//!
//! Three cases here are sharper than the container boilerplate:
//!
//!   - `the_affinity_layout_roundtrips` — a dense affinity means a KERNEL and a
//!     sparse one means a NEIGHBORHOOD GRAPH. Those are different models of the
//!     same data, not two encodings of one, so the layout is named explicitly in
//!     the header and round-trips with the values.
//!   - `a_malformed_csr_is_rejected` — the CSR invariants (`indptr` monotone,
//!     starting at 0, ending at `nnz`, columns in range) are individually
//!     invisible and collectively load-bearing: a violation is an out-of-bounds
//!     read inside the Lanczos matvec, not a wrong number.
//!   - `hdbscan_keeps_its_optional_attributes` — every optional tensor
//!     round-trips as key-presence, so a reloaded model reports exactly the
//!     attributes the saved one did, INCLUDING the `None`s. The GLOSH source in
//!     particular is stored rather than dropped, so `outlier_scores` still
//!     answers after a reload.
//!
//! Per AGENTS.md §2 tests live here, never in an in-source `#[cfg(test)] mod`.

use bytemuck::Pod;
use cubecl::prelude::{CubeElement, Float};

use mlrs_algos::cluster::agglomerative::{AgglomerativeClustering, Metric as AggMetric};
use mlrs_algos::cluster::cluster_persist::{
    AlignedBytes, ClusterFile, ClusterWriter, LoadModel, PersistError, SaveModel, TensorRef,
};
use mlrs_algos::cluster::dbscan::DBSCAN;
use mlrs_algos::cluster::hdbscan::{Hdbscan, Metric as HdbMetric, StoreCenters};
use mlrs_algos::cluster::kmeans::KMeans;
use mlrs_algos::cluster::spectral_clustering::SpectralClustering;
use mlrs_algos::cluster::spectral_embedding::SpectralEmbedding;
use mlrs_algos::preprocessing::MaxAbsScaler;
use mlrs_algos::typestate::{Fit, Fitted, PredictLabels};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::{self, ActiveRuntime};

const N_SAMPLES: usize = 18;
const N_FEATURES: usize = 3;

/// Three well-separated blobs. The separation is load-bearing: several gates
/// compare labelings, and a fixture whose clusters overlapped would make them
/// depend on tie-breaking rather than on the file.
fn fixture<F: Pod>() -> Vec<F> {
    let centers = [[0.0, 0.0, 0.0], [8.0, 8.0, 8.0], [-8.0, 6.0, -6.0]];
    (0..N_SAMPLES)
        .flat_map(|i| {
            let c = centers[i % 3];
            let jitter = (i / 3) as f64 * 0.15 - 0.4;
            [c[0] + jitter, c[1] - jitter, c[2] + jitter * 0.5]
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

fn fit_kmeans<F>(p: &mut BufferPool<ActiveRuntime>) -> KMeans<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    KMeans::<F>::builder()
        .n_clusters(3)
        .random_state(Some(7))
        .max_iter(50)
        .build::<F>()
        .expect("KMeans builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("KMeans fits the fixture")
}

fn fit_dbscan<F>(p: &mut BufferPool<ActiveRuntime>) -> DBSCAN<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    DBSCAN::<F>::builder()
        .eps(2.0)
        .min_samples(2)
        .build::<F>()
        .expect("DBSCAN builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("DBSCAN fits the fixture")
}

fn fit_agglomerative<F>(p: &mut BufferPool<ActiveRuntime>) -> AgglomerativeClustering<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    AgglomerativeClustering::<F>::builder()
        .n_clusters(3)
        .metric(AggMetric::Manhattan)
        .build::<F>()
        .expect("AgglomerativeClustering builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("AgglomerativeClustering fits the fixture")
}

fn fit_hdbscan<F>(
    p: &mut BufferPool<ActiveRuntime>,
    store_centers: Option<StoreCenters>,
) -> Hdbscan<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    Hdbscan::<F>::builder()
        .min_cluster_size(2)
        .metric(HdbMetric::Euclidean)
        .store_centers(store_centers)
        .build::<F>()
        .expect("Hdbscan builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("Hdbscan fits the fixture")
}

/// `affinity` picks the layout: `'rbf'` produces a DENSE kernel affinity,
/// `'nearest_neighbors'` a sparse CSR connectivity graph.
fn fit_spectral<F>(
    p: &mut BufferPool<ActiveRuntime>,
    affinity: &str,
) -> SpectralClustering<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    SpectralClustering::<F>::builder()
        .n_clusters(3)
        .affinity(affinity.to_string())
        .n_neighbors(4)
        .random_state(Some(11))
        .build::<F>()
        .expect("SpectralClustering builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("SpectralClustering fits the fixture")
}

fn fit_embedding<F>(
    p: &mut BufferPool<ActiveRuntime>,
    affinity: &str,
) -> SpectralEmbedding<F, Fitted>
where
    F: Float + CubeElement + Pod,
{
    let x = upload::<F>(p);
    SpectralEmbedding::<F>::builder()
        .n_components(2)
        .affinity(affinity.to_string())
        .n_neighbors(Some(4))
        .random_state(Some(11))
        .build::<F>()
        .expect("SpectralEmbedding builds")
        .fit(p, &x, None, (N_SAMPLES, N_FEATURES))
        .expect("SpectralEmbedding fits the fixture")
}

// ---------------------------------------------------------------------------
// Round-trip, estimator by estimator
// ---------------------------------------------------------------------------

#[test]
fn kmeans_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("kmeans.safetensors");
    let mut p = pool();

    let fitted = fit_kmeans::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: KMeans<f32, Fitted> = KMeans::load(&mut p, &path).expect("load succeeds");

    // `==` rather than a tolerance: the file stores the exact IEEE bits, so any
    // drift at all is a defect in the container, not rounding.
    assert_eq!(
        loaded.cluster_centers(&p),
        fitted.cluster_centers(&p),
        "cluster_centers_ must round-trip exactly"
    );
    assert_eq!(loaded.labels(&p), fitted.labels(&p), "labels_");
    assert_eq!(loaded.inertia(), fitted.inertia(), "inertia_");
    assert_eq!(loaded.n_iter(), fitted.n_iter(), "n_iter_");

    // KMeans is the one member of this family that GENERALIZES, so it gets the
    // prediction gate the others cannot have.
    let x = upload::<f32>(&mut p);
    let before = fitted
        .predict_labels(&mut p, &x, (N_SAMPLES, N_FEATURES))
        .expect("predict_labels succeeds")
        .to_host(&p);
    let x = upload::<f32>(&mut p);
    assert_eq!(
        loaded
            .predict_labels(&mut p, &x, (N_SAMPLES, N_FEATURES))
            .expect("predict_labels succeeds")
            .to_host(&p),
        before,
        "the reloaded KMeans must assign identically"
    );
}

#[test]
fn dbscan_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("dbscan.safetensors");
    let mut p = pool();

    let fitted = fit_dbscan::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: DBSCAN<f32, Fitted> = DBSCAN::load(&mut p, &path).expect("load succeeds");

    // DBSCAN has no `predict`, so the attributes ARE the model.
    assert_eq!(loaded.labels(&p), fitted.labels(&p), "labels_");
    assert_eq!(
        loaded.core_sample_indices(&p),
        fitted.core_sample_indices(&p),
        "core_sample_indices_ is not derivable from labels_ — a border point and \
         a core point of the same cluster share a label"
    );
}

#[test]
fn agglomerative_roundtrip_is_bit_exact() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("agglo.safetensors");
    let mut p = pool();

    let fitted = fit_agglomerative::<f32>(&mut p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: AgglomerativeClustering<f32, Fitted> =
        AgglomerativeClustering::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.labels(&p), fitted.labels(&p), "labels_");
    assert_eq!(
        loaded.children(),
        fitted.children(),
        "children_ is the dendrogram the labeling was CUT from — a cut at \
         n_clusters throws away every merge above it, so the labels cannot \
         reconstruct it"
    );
    assert_eq!(loaded.n_leaves(), fitted.n_leaves(), "n_leaves_");
    assert_eq!(
        loaded.n_features_in(),
        fitted.n_features_in(),
        "n_features_in_ is not implied by any tensor's shape here"
    );
}

#[test]
fn hdbscan_keeps_its_optional_attributes() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("hdbscan.safetensors");
    let mut p = pool();

    // `store_centers = Both` so BOTH optional center tables are present — the
    // case where a dropped tensor would be least visible.
    let fitted = fit_hdbscan::<f32>(&mut p, Some(StoreCenters::Both));
    let outliers_before = fitted.outlier_scores(&p);
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: Hdbscan<f32, Fitted> = Hdbscan::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.labels(&p), fitted.labels(&p), "labels_");
    assert_eq!(
        loaded.probabilities(&p),
        fitted.probabilities(&p),
        "probabilities_"
    );
    assert_eq!(loaded.centroids(&p), fitted.centroids(&p), "centroids_");
    assert_eq!(loaded.medoids(&p), fitted.medoids(&p), "medoids_");
    assert_eq!(
        loaded.single_linkage(),
        fitted.single_linkage(),
        "single_linkage_"
    );

    // The GLOSH source is the expensive part of this file, and the reason it is
    // stored: `outlier_scores` is derived LAZILY from it, so a reload that
    // dropped it would return `None` here — silently losing an attribute the
    // saved model had, with no error anywhere.
    assert_eq!(
        loaded.outlier_scores(&p),
        outliers_before,
        "outlier_scores must still answer after a reload"
    );
    assert!(
        outliers_before.is_some(),
        "the fixture must produce outlier scores, or the gate above proves nothing"
    );
}

#[test]
fn hdbscan_absent_attributes_stay_absent() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("hdbscan.safetensors");
    let mut p = pool();

    // The other half of key-presence: a model that stored NO centers must come
    // back reporting `None`, not an empty table. Optionality round-trips in both
    // directions or it does not round-trip.
    let fitted = fit_hdbscan::<f32>(&mut p, None);
    assert!(
        fitted.centroids(&p).is_none() && fitted.medoids(&p).is_none(),
        "store_centers=None must produce no center tables"
    );
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: Hdbscan<f32, Fitted> = Hdbscan::load(&mut p, &path).expect("load succeeds");

    assert!(
        loaded.centroids(&p).is_none(),
        "an absent centroids_ must stay absent"
    );
    assert!(
        loaded.medoids(&p).is_none(),
        "an absent medoids_ must stay absent"
    );
}

// ---------------------------------------------------------------------------
// The affinity graph
// ---------------------------------------------------------------------------

#[test]
fn the_affinity_layout_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // A dense affinity means a KERNEL and a sparse one means a NEIGHBORHOOD
    // GRAPH — different models of the same data, not two encodings of one. Both
    // must survive with their layout intact, and the layout must be readable
    // from the header rather than inferred.
    for (affinity, expect_sparse) in [("rbf", false), ("nearest_neighbors", true)] {
        let path = dir.path().join(format!("{affinity}.safetensors"));
        let fitted = fit_spectral::<f32>(&mut p, affinity);
        let dense_before = fitted.affinity_matrix_dense();
        let sparse_before = fitted.affinity_matrix_sparse().cloned();
        assert_eq!(
            sparse_before.is_some(),
            expect_sparse,
            "{affinity} must produce the {} layout",
            if expect_sparse { "sparse" } else { "dense" }
        );
        fitted.save(&p, &path).expect("save succeeds");

        let raw = AlignedBytes::read(&path).expect("read succeeds");
        let file = ClusterFile::parse(&raw, "spectral_clustering").expect("parse succeeds");
        assert_eq!(
            file.scalar_str("affinity_layout")
                .expect("the key is present"),
            if expect_sparse { "sparse" } else { "dense" },
            "{affinity}: the layout must be named explicitly in the header"
        );

        let loaded: SpectralClustering<f32, Fitted> =
            SpectralClustering::load(&mut p, &path).expect("load succeeds");
        assert_eq!(
            loaded.labels(&p),
            fitted.labels(&p),
            "{affinity}: labels_ must round-trip"
        );
        assert_eq!(
            loaded.affinity_matrix_dense(),
            dense_before,
            "{affinity}: the affinity values must round-trip"
        );
        assert_eq!(
            loaded.affinity_matrix_sparse().is_some(),
            expect_sparse,
            "{affinity}: the LAYOUT must round-trip, not just the values"
        );
        assert_eq!(
            loaded.n_graph_components(),
            fitted.n_graph_components(),
            "{affinity}: n_graph_components_"
        );
    }
}

#[test]
fn spectral_embedding_roundtrips() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("embed.safetensors");
    let mut p = pool();

    let fitted = fit_embedding::<f32>(&mut p, "nearest_neighbors");
    fitted.save(&p, &path).expect("save succeeds");
    let loaded: SpectralEmbedding<f32, Fitted> =
        SpectralEmbedding::load(&mut p, &path).expect("load succeeds");

    assert_eq!(loaded.embedding(&p), fitted.embedding(&p), "embedding_");
    assert_eq!(
        loaded.affinity_matrix_dense(),
        fitted.affinity_matrix_dense(),
        "the affinity values"
    );
    // The RESOLVED hyperparameters, distinct from the requests they came from.
    assert_eq!(loaded.n_neighbors_(), fitted.n_neighbors_(), "n_neighbors_");
    assert_eq!(loaded.gamma_(), fitted.gamma_(), "gamma_");
    assert_eq!(
        loaded.n_graph_components(),
        fitted.n_graph_components(),
        "n_graph_components_"
    );
}

#[test]
fn a_malformed_csr_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut p = pool();

    // Every CSR invariant is individually invisible and collectively
    // load-bearing: a violation is an out-of-bounds read inside the Lanczos
    // matvec, not a wrong number. Each case below is a header whose halves are
    // each well-formed in isolation.
    let n = 4usize;
    let labels = [0i64, 0, 1, 1];
    let cases: [(&str, Vec<u64>, Vec<u64>, Vec<f64>); 3] = [
        // `indptr` does not end at nnz.
        (
            "indptr-end",
            vec![0, 1, 2, 3, 3],
            vec![1, 0, 3, 2],
            vec![1.0; 4],
        ),
        // `indptr` is not monotone.
        (
            "indptr-order",
            vec![0, 2, 1, 3, 4],
            vec![1, 0, 3, 2],
            vec![1.0; 4],
        ),
        // A column index past the sample count.
        (
            "column-range",
            vec![0, 1, 2, 3, 4],
            vec![1, 0, 9, 2],
            vec![1.0; 4],
        ),
    ];

    for (name, indptr, indices, data) in cases {
        let path = dir.path().join(format!("{name}.safetensors"));
        let mut w = ClusterWriter::new("spectral_clustering");
        w.scalar_usize("param:n_clusters", 2);
        w.scalar_str("param:eigen_solver", "auto");
        w.scalar_usize("param:n_init", 10);
        w.scalar_f64("param:gamma", 1.0);
        w.scalar_str("param:affinity", "nearest_neighbors");
        w.scalar_usize("param:n_neighbors", 2);
        w.scalar_str("param:assign_labels", "kmeans");
        w.scalar_f64("param:degree", 3.0);
        w.scalar_f64("param:coef0", 1.0);
        w.scalar_bool("param:verbose", false);
        w.scalar_usize("n_graph_components_", 1);
        w.scalar_usize("n_features_in_", N_FEATURES);
        w.scalar_str("affinity_layout", "sparse");
        w.tensor(
            "labels_",
            TensorRef::i64s(&labels, vec![n]).expect("well-formed"),
        );
        w.tensor(
            "affinity_indptr",
            TensorRef::u64s(&indptr, vec![indptr.len()]).expect("well-formed"),
        );
        w.tensor(
            "affinity_indices",
            TensorRef::u64s(&indices, vec![indices.len()]).expect("well-formed"),
        );
        w.tensor(
            "affinity_data",
            TensorRef::f64s(&data, vec![data.len()]).expect("well-formed"),
        );
        w.write(&path)
            .expect("the hand-written file is well-formed as a container");

        let err = match SpectralClustering::<f32, Fitted>::load(&mut p, &path) {
            Ok(_) => panic!("{name}: a malformed CSR must not load"),
            Err(e) => e,
        };
        assert!(
            matches!(&err, PersistError::InconsistentGeometry { .. }),
            "{name}: expected InconsistentGeometry, got {err:?}"
        );
    }
}

#[test]
fn an_unrecognised_affinity_layout_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-layout.safetensors");
    let mut p = pool();

    // The layout is named explicitly rather than inferred from which tensors
    // are present, so a file naming a third layout must fail rather than fall
    // through to whichever arm happens to match.
    let labels = [0i64, 1];
    let mut w = ClusterWriter::new("spectral_embedding");
    w.scalar_usize("param:n_components", 1);
    w.scalar_str("param:affinity", "rbf");
    w.scalar_str("param:eigen_solver", "auto");
    w.scalar_usize("n_graph_components_", 1);
    w.scalar_usize("n_features_in_", N_FEATURES);
    w.scalar_str("affinity_layout", "coo");
    w.tensor(
        "embedding_",
        TensorRef::floats(&[0.0f32, 1.0], vec![2, 1]).expect("well-formed"),
    );
    let _ = labels;
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match SpectralEmbedding::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an unrecognised affinity layout must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// The format claims
// ---------------------------------------------------------------------------

#[test]
fn saving_twice_produces_an_identical_model() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let first = dir.path().join("a.safetensors");
    let second = dir.path().join("b.safetensors");
    let mut p = pool();

    // RAW BYTES: a model file must be a deterministic function of the model.
    // This is also the gate on the `third_party/safetensors` `BTreeMap` patch —
    // `KMeans` carries eleven scalars, so a randomly-seeded header map is
    // overwhelmingly likely to reorder one.
    let fitted = fit_kmeans::<f32>(&mut p);
    fitted.save(&p, &first).expect("save succeeds");
    fitted.save(&p, &second).expect("save succeeds");
    assert_eq!(
        std::fs::read(&first).expect("read"),
        std::fs::read(&second).expect("read"),
        "saving the same model twice must produce byte-identical files"
    );
}

#[test]
fn the_load_path_is_zero_copy() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("dbscan.safetensors");
    let mut p = pool();
    fit_dbscan::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    // The `AlignedBytes` claim over this family's `I64` label tensors — the
    // shape every member of it writes.
    let raw = AlignedBytes::read(&path).expect("read succeeds");
    let file = ClusterFile::parse(&raw, "dbscan").expect("parse succeeds");
    for name in ["labels_", "core_sample_indices_"] {
        let view = file.tensor(name).expect("the tensor is present");
        assert!(
            bytemuck::try_cast_slice::<u8, i64>(view.data()).is_ok(),
            "'{name}' must be reinterpretable as &[i64] without a copy"
        );
    }
}

// ---------------------------------------------------------------------------
// Rejection — the file is untrusted input (T-04-01-01)
// ---------------------------------------------------------------------------

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
    let err = match KMeans::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an mlrs-prep file must not load as a clustering"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::NotAnMlrsModel { expected, .. } if *expected == "mlrs-cluster"
        ),
        "expected NotAnMlrsModel, got {err:?}"
    );
}

#[test]
fn sibling_estimators_do_not_cross_load() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("agglo.safetensors");
    let mut p = pool();

    // Every member of this family writes a `labels_` of the same shape and
    // dtype, so the `estimator` tag is what keeps six different models apart.
    fit_agglomerative::<f32>(&mut p)
        .save(&p, &path)
        .expect("save succeeds");

    let err = match DBSCAN::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an agglomerative file must not load as a dbscan"),
        Err(e) => e,
    };
    assert!(
        matches!(
            &err,
            PersistError::WrongEstimator { expected, found }
                if *expected == "dbscan" && found == "agglomerative_clustering"
        ),
        "expected WrongEstimator, got {err:?}"
    );
}

#[test]
fn a_children_table_disagreeing_with_the_labels_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("ragged.safetensors");
    let mut p = pool();

    // A binary merge tree over `n` leaves has exactly `n - 1` internal nodes.
    // Neither extent is wrong on its own, so only the cross-check catches a
    // truncated tree — and a caller walking the dendrogram from the labels
    // would otherwise index past its end.
    let labels = [0i64, 0, 1, 1, 2, 2];
    let children = [0i64, 1, 2, 3];
    let mut w = ClusterWriter::new("agglomerative_clustering");
    w.scalar_usize("param:n_clusters", 3);
    w.scalar_str("param:metric", "euclidean");
    w.scalar_usize("n_features_in_", N_FEATURES);
    w.tensor(
        "labels_",
        TensorRef::i64s(&labels, vec![6]).expect("well-formed"),
    );
    w.tensor(
        "children_",
        TensorRef::i64s(&children, vec![2, 2]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match AgglomerativeClustering::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("a truncated merge tree must not load"),
        Err(e) => e,
    };
    assert!(
        matches!(&err, PersistError::InconsistentGeometry { .. }),
        "expected InconsistentGeometry, got {err:?}"
    );
}

#[test]
fn an_out_of_range_core_index_is_rejected() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = dir.path().join("bad-core.safetensors");
    let mut p = pool();

    // `core_sample_indices_` is by definition a set of ROW POSITIONS, so an
    // entry past the end would index the sample axis out of range for any
    // caller that used it to select rows.
    let labels = [0i64, 0, 1];
    let core = [0i64, 9];
    let mut w = ClusterWriter::new("dbscan");
    w.scalar_f64("param:eps", 1.0);
    w.scalar_usize("param:min_samples", 2);
    w.tensor(
        "labels_",
        TensorRef::i64s(&labels, vec![3]).expect("well-formed"),
    );
    w.tensor(
        "core_sample_indices_",
        TensorRef::i64s(&core, vec![2]).expect("well-formed"),
    );
    w.write(&path)
        .expect("the hand-written file is well-formed as a container");

    let err = match DBSCAN::<f32, Fitted>::load(&mut p, &path) {
        Ok(_) => panic!("an out-of-range core index must not load"),
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
    let path = dir.path().join("kmeans.safetensors");
    let mut p = pool();
    fit_kmeans::<f32>(&mut p)
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
