//! `cluster` — distance-based clustering estimators (CLUSTER-01 / CLUSTER-02).
//!
//! Module index for the two Phase-5 clustering estimators. They consume the new
//! Phase-5 distance/clustering primitives (`prims::kmeans`, `prims::dbscan`) and
//! return integer labels via the [`PredictLabels`](crate::traits::PredictLabels)
//! surface (D-05/D-06):
//!
//! - `KMeans` (CLUSTER-01) — k-means++ init (injected for the oracle, D-09) +
//!   Lloyd updates; stores `cluster_centers_` (F), `labels_`/`inertia_`. Up to a
//!   label permutation vs sklearn (D-09). Added by plan **05-07**.
//! - `DBSCAN` (CLUSTER-02) — eps-neighborhood core mask + host DFS expansion;
//!   stores `labels_` (noise = `-1`) and `core_sample_indices_` (i32). Added by
//!   plan **05-08**.
//!
//! Each estimator plan ADDS its own `pub mod <estimator>;` line here and creates
//! the matching file; the plans do NOT edit `lib.rs` (owned by the Wave-0
//! scaffold), keeping the estimator plans file-disjoint and parallel-safe.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

pub mod dbscan;
pub mod kmeans;
// Phase-9 spectral estimators (Wave-0 scaffold 09-01 owns these registrations;
// the Wave-2 plan 09-03 fills `spectral_embedding`, the Wave-3 plan 09-04 fills
// `spectral_clustering` — file-disjoint, parallel-safe). Both compile today as
// struct + `new()` stubs (fit / accessor bodies `todo!()`).
//
// - `SpectralEmbedding` (SPECTRAL-01) — affinity → normalized Laplacian →
//   smallest non-trivial eigenvectors → `D^-1/2` recovery → `embedding_`. Up to
//   sign alignment vs sklearn (subspace test for degenerate spectra, D-09).
//   Added by plan **09-03**.
// - `SpectralClustering` (SPECTRAL-02) — spectral embedding → v1 KMeans;
//   `labels_` matches sklearn up to label permutation (exact-labels gate, D-10).
//   Added by plan **09-04**.
// Shared spectral-family host recovery math (WR-06): the `recover` helper (with a
// `drop_first` param) + the `host_to_f64`/`f64_to_host` bytemuck pair, formerly
// duplicated verbatim across spectral_embedding / spectral_clustering.
pub(crate) mod spectral;
pub mod spectral_clustering;
pub mod spectral_embedding;

pub use spectral_clustering::SpectralClustering;
pub use spectral_embedding::SpectralEmbedding;
