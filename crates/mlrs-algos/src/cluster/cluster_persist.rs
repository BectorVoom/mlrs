//! `cluster_persist` (CLUSTER-PERSIST, prototype) — the `mlrs-cluster` half of
//! the mlrs model file format: the container discriminator, the aliases the six
//! clustering estimators write and read through, and the two pieces of state
//! more than one of them holds — the label vector and the affinity graph.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## What a clustering model even is
//!
//! This is the family where "save the model" needs stating carefully, because
//! most of its members have no `predict`. `DBSCAN`, `AgglomerativeClustering`
//! and `SpectralClustering` are `fit_predict`-only: their entire output is the
//! `labels_` assigned to the rows they were FITTED on, and there is no
//! parameterization that could label a new row. For those, the file IS the
//! labeling plus whatever fitted diagnostics sklearn exposes beside it
//! (`core_sample_indices_`, `children_`, `affinity_matrix_`), and a round-trip
//! is faithful when a reloaded estimator reports every attribute the saved one
//! did.
//!
//! `KMeans` is the exception that does generalize — `cluster_centers_` labels
//! any new row — and `Hdbscan` sits in between, carrying enough state
//! (`probabilities_`, the single-linkage tree, the source its GLOSH scores are
//! derived from) that a reload can answer questions the bare labels cannot.
//!
//! ## The on-disk shape
//!
//! | name | dtype | shape | held by |
//! |---|---|---|---|
//! | `labels_` | `I64` | `[n_samples]` | all but `SpectralEmbedding` |
//! | `cluster_centers_` | `F` | `[n_clusters, n_features]` | `KMeans` |
//! | `core_sample_indices_` | `I64` | `[n_core]` | `DBSCAN` |
//! | `children_` | `I64` | `[n_samples - 1, 2]` | `AgglomerativeClustering` |
//! | `probabilities_` | `F` | `[n_samples]` | `Hdbscan` |
//! | `embedding_` | `F` | `[n_samples, n_components]` | `SpectralEmbedding` |
//! | affinity graph | see [`write_affinity`] | — | the two spectral estimators |
//!
//! ## `labels_` is `I64`, not the model's float width
//!
//! Cluster ids are integers, and `-1` is MEANINGFUL in three of these
//! estimators — it is DBSCAN's and HDBSCAN's noise marker. Storing them as
//! floats would invite a reader to compare them with a tolerance and would make
//! a large id silently unrepresentable. mlrs holds them as `i32` in memory and
//! widens here, because `i32` is an internal choice while the file has to
//! survive a model whose ids do not fit one.
//!
//! Tests live in `crates/mlrs-algos/tests/cluster_persist_test.rs`
//! (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

use super::spectral_host::{Csr, HostAffinity};

// The container is shared with every other family; only the discriminator and
// the cluster-shaped helpers below are local. Re-exported (not just imported)
// so `cluster::cluster_persist::{AlignedBytes, SaveModel, …}` is the single
// import path for a clustering estimator's `save`/`load`.
pub use crate::persist::{
    as_f64, as_floats, as_i64, as_usizes, expect_len, shape_1d, shape_2d, AlignedBytes, Container,
    LoadModel, ModelFile, ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The clustering container discriminator (`format = "mlrs-cluster"`).
pub struct ClusterContainer;

impl Container for ClusterContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-cluster";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`ClusterFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the per-sample cluster assignment, `[n_samples]`.
pub const LABELS_NAME: &str = "labels_";

/// The clustering writer: [`ModelWriter`] pinned to the `mlrs-cluster`
/// container.
pub type ClusterWriter<'a> = ModelWriter<'a, ClusterContainer>;

/// The clustering reader: [`ModelFile`] pinned to the `mlrs-cluster` container.
pub type ClusterFile<'a> = ModelFile<'a, ClusterContainer>;

/// Widen an `i32` label vector to the `i64` the file stores.
///
/// A copy, and an unavoidable one — the widths differ — but a cheap one against
/// the `n_samples` it walks. Callers must bind the result before constructing
/// the writer, since the writer borrows it.
pub fn widen_labels(labels: &[i32]) -> Vec<i64> {
    labels.iter().map(|&v| i64::from(v)).collect()
}

/// Stage the label vector, rejecting an empty one.
///
/// A fitted clustering with no labels is not a degenerate model, it is an
/// absent one: every estimator here labels the rows it was fitted on, so a
/// zero-length `labels_` means the fit never happened.
pub fn write_labels<'a>(w: &mut ClusterWriter<'a>, labels: &'a [i64]) -> Result<(), PersistError> {
    if labels.is_empty() {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{LABELS_NAME}' is empty; a fitted clustering labels at least one sample"
            ),
        });
    }
    w.tensor(LABELS_NAME, TensorRef::i64s(labels, vec![labels.len()])?);
    Ok(())
}

/// Read the label vector back, narrowed to the `i32` the estimators hold.
///
/// The narrowing is CHECKED rather than truncating: the file is untrusted input
/// (T-04-01-01), and a label that does not fit an `i32` would silently become a
/// different cluster id — including, at the wrong bit pattern, the `-1` that
/// means "noise".
pub fn read_labels(file: &ClusterFile<'_>) -> Result<Vec<i32>, PersistError> {
    let view = file.tensor(LABELS_NAME)?;
    let n = shape_1d(&view, LABELS_NAME)?;
    if n == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{LABELS_NAME}' is empty; a fitted clustering labels at least \
                 one sample"
            ),
        });
    }
    as_i64(&view, LABELS_NAME)?
        .iter()
        .map(|&v| {
            i32::try_from(v).map_err(|_| PersistError::InconsistentGeometry {
                reason: format!(
                    "tensor '{LABELS_NAME}' holds the cluster id {v}, which does not fit \
                     the i32 the clustering kernels consume"
                ),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The affinity graph — the two spectral estimators' shared state
// ---------------------------------------------------------------------------

/// The `__metadata__` key naming which layout the affinity graph is stored in.
pub const AFFINITY_LAYOUT_KEY: &str = "affinity_layout";
/// The dense affinity tensor, `[n_samples, n_samples]`.
pub const AFFINITY_DENSE_NAME: &str = "affinity_matrix_";
/// CSR row starts, `[n_samples + 1]`.
pub const AFFINITY_INDPTR_NAME: &str = "affinity_indptr";
/// CSR column indices, one per stored entry.
pub const AFFINITY_INDICES_NAME: &str = "affinity_indices";
/// CSR values, one per stored entry.
pub const AFFINITY_DATA_NAME: &str = "affinity_data";

// ---------------------------------------------------------------------------
// The affinity graph, staged
// ---------------------------------------------------------------------------

/// The widened CSR index vectors, bound so they outlive the writer that borrows
/// them.
///
/// [`ModelWriter`] borrows every payload for its whole lifetime, which is what
/// keeps the save path to a single copy — but it means a caller cannot stage a
/// vector it builds inside the staging call. The CSR arms need exactly that:
/// mlrs holds `indptr`/`indices` as `u32` and the file stores `U64`, so the
/// widened copies have to live somewhere the writer can borrow from.
///
/// Prepare this BEFORE constructing the writer, then pass it to
/// [`AffinityStaging::write_into`].
pub struct AffinityStaging {
    /// CSR row starts, widened. Empty on the dense arm.
    pub indptr: Vec<u64>,
    /// CSR column indices, widened. Empty on the dense arm.
    pub indices: Vec<u64>,
}

impl AffinityStaging {
    /// Widen whatever index arrays `affinity` carries, ahead of the writer.
    pub fn prepare(affinity: &HostAffinity) -> Self {
        match affinity {
            HostAffinity::Dense(_) => AffinityStaging {
                indptr: Vec::new(),
                indices: Vec::new(),
            },
            HostAffinity::Sparse(csr) => AffinityStaging {
                indptr: csr.indptr.iter().map(|&v| u64::from(v)).collect(),
                indices: csr.indices.iter().map(|&v| u64::from(v)).collect(),
            },
        }
    }

    /// Stage the affinity graph in WHICHEVER layout the fit produced.
    ///
    /// The layout is named explicitly in `__metadata__` rather than inferred
    /// from which tensors happen to be present. Inferring would work today and
    /// would silently mis-read a file that grew a third layout later.
    ///
    /// ## Why the sparse arm stays sparse
    ///
    /// A kNN connectivity graph has `O(n · k)` stored entries against the `n²` a
    /// dense one would need, and at the sample counts spectral clustering is used
    /// at, that is the difference between a file of megabytes and one of
    /// gigabytes. mlrs keeps it CSR in memory precisely because the Lanczos
    /// matvec consumes it that way
    /// ([`spectral_host`](super::spectral_host)), so the file follows the compute
    /// path exactly as the rest of this format does — no expansion on either
    /// side.
    ///
    /// The two arms are not interchangeable and the layout is not a storage
    /// detail: a dense affinity means a kernel (`rbf`, `poly`, `precomputed`)
    /// and a sparse one means a neighborhood graph, which are different models
    /// of the same data. Round-tripping the layout is round-tripping the model.
    ///
    /// Everything is stored at `f64`, matching the host affinity's own width:
    /// the spectral path is `f64` throughout because a single-vector Lanczos on
    /// a disconnected graph is already delicate, and narrowing the operand to
    /// save bytes would change what the eigensolver converges to.
    ///
    /// The CSR index arrays widen to `U64`. `TensorRef` has no `u32`
    /// constructor, and the widening keeps the file readable on a host whose
    /// graphs outgrow a `u32` without a format change; the cost is `4 · nnz`
    /// bytes against a values payload that is already `8 · nnz`.
    pub fn write_into<'a>(
        &'a self,
        w: &mut ClusterWriter<'a>,
        affinity: &'a HostAffinity,
        n_samples: usize,
    ) -> Result<(), PersistError> {
        match affinity {
            HostAffinity::Dense(d) => {
                w.scalar_str(AFFINITY_LAYOUT_KEY, "dense");
                w.tensor(
                    AFFINITY_DENSE_NAME,
                    TensorRef::f64s(d, vec![n_samples, n_samples])?,
                );
            }
            HostAffinity::Sparse(csr) => {
                w.scalar_str(AFFINITY_LAYOUT_KEY, "sparse");
                w.tensor(
                    AFFINITY_INDPTR_NAME,
                    TensorRef::u64s(&self.indptr, vec![self.indptr.len()])?,
                );
                w.tensor(
                    AFFINITY_INDICES_NAME,
                    TensorRef::u64s(&self.indices, vec![self.indices.len()])?,
                );
                w.tensor(
                    AFFINITY_DATA_NAME,
                    TensorRef::f64s(&csr.data, vec![csr.data.len()])?,
                );
            }
        }
        Ok(())
    }
}

/// Read the affinity graph back in the layout the file names.
///
/// Every cross-array invariant CSR has is checked here, because none of them is
/// individually visible and all of them are load-bearing: `indptr` must have
/// `n + 1` entries, must be non-decreasing, must start at 0 and end at `nnz`,
/// and every column index must be in range. A file failing any of these would
/// otherwise index out of bounds inside the Lanczos matvec rather than report a
/// bad file (T-04-01-01).
pub fn read_affinity(
    file: &ClusterFile<'_>,
    n_samples: usize,
) -> Result<HostAffinity, PersistError> {
    match file.scalar_str(AFFINITY_LAYOUT_KEY)? {
        "dense" => {
            let view = file.tensor(AFFINITY_DENSE_NAME)?;
            let (rows, cols) = shape_2d(&view, AFFINITY_DENSE_NAME)?;
            if rows != n_samples || cols != n_samples {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!(
                        "tensor '{AFFINITY_DENSE_NAME}' declares shape [{rows}, {cols}], \
                         but the model has {n_samples} samples"
                    ),
                });
            }
            Ok(HostAffinity::Dense(
                as_f64(&view, AFFINITY_DENSE_NAME)?.into_owned(),
            ))
        }
        "sparse" => {
            let indptr_v = file.tensor(AFFINITY_INDPTR_NAME)?;
            expect_len(
                AFFINITY_INDPTR_NAME,
                shape_1d(&indptr_v, AFFINITY_INDPTR_NAME)?,
                n_samples + 1,
                "entries",
            )?;
            let indptr = as_usizes(&indptr_v, AFFINITY_INDPTR_NAME)?;

            let indices_v = file.tensor(AFFINITY_INDICES_NAME)?;
            let nnz = shape_1d(&indices_v, AFFINITY_INDICES_NAME)?;
            let indices = as_usizes(&indices_v, AFFINITY_INDICES_NAME)?;

            let data_v = file.tensor(AFFINITY_DATA_NAME)?;
            expect_len(
                AFFINITY_DATA_NAME,
                shape_1d(&data_v, AFFINITY_DATA_NAME)?,
                nnz,
                "entries",
            )?;
            let data = as_f64(&data_v, AFFINITY_DATA_NAME)?.into_owned();

            // The CSR invariants, all of them. `indptr[0] == 0`, monotone,
            // `indptr[n] == nnz`, and every column in `0..n`. A violation is an
            // out-of-bounds read inside the matvec, not a wrong number.
            if indptr[0] != 0 || indptr[n_samples] != nnz {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!(
                        "'{AFFINITY_INDPTR_NAME}' spans [{}, {}] but '{AFFINITY_INDICES_NAME}' \
                         holds {nnz} entries",
                        indptr[0], indptr[n_samples]
                    ),
                });
            }
            if indptr.windows(2).any(|w| w[0] > w[1]) {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!("'{AFFINITY_INDPTR_NAME}' is not non-decreasing"),
                });
            }
            if let Some(&bad) = indices.iter().find(|&&c| c >= n_samples) {
                return Err(PersistError::InconsistentGeometry {
                    reason: format!(
                        "'{AFFINITY_INDICES_NAME}' holds the column {bad}, out of range for \
                         {n_samples} samples"
                    ),
                });
            }

            Ok(HostAffinity::Sparse(Csr {
                indptr: indptr.iter().map(|&v| v as u32).collect(),
                indices: indices.iter().map(|&v| v as u32).collect(),
                data,
            }))
        }
        // `InconsistentGeometry` rather than `BadMetadata`: the latter names only
        // the key, and here the offending VALUE is the useful half of the
        // diagnostic.
        other => Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{AFFINITY_LAYOUT_KEY}' is '{other}'; the affinity graph is stored \
                 either 'dense' or 'sparse'"
            ),
        }),
    }
}

/// Read an OPTIONAL float tensor of a known length — the shape several of these
/// estimators' optional fitted attributes share (`probabilities_`, `centroids_`,
/// `medoids_`).
///
/// `Ok(None)` when the tensor is absent, which is exactly what a model that did
/// not compute it wrote. A present-but-misshapen tensor is an error rather than
/// a `None`: a truncated array is a corrupt file, not a model that chose not to
/// store one.
pub fn read_opt_floats<'a, F: Pod>(
    file: &ClusterFile<'a>,
    name: &'static str,
    expected: usize,
) -> Result<Option<Cow<'a, [F]>>, PersistError> {
    let Some(view) = file.tensor_opt(name) else {
        return Ok(None);
    };
    let len: usize = view.shape().iter().product();
    expect_len(name, len, expected, "entries")?;
    Ok(Some(as_floats::<F>(&view, name)?))
}
