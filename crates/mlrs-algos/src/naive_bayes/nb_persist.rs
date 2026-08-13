//! `nb_persist` (NB-PERSIST, prototype) — the Naive Bayes half of the mlrs model
//! file format: the `mlrs-nb` container discriminator, the aliases the five
//! estimators write and read through, and the shared discrete-NB core.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! aligned zero-copy read path, the typed tensor accessors, the error type and
//! the `SaveModel`/`LoadModel` surface — lives in [`crate::persist`] and is
//! estimator-agnostic. Read that module's docs first: they carry the four
//! decisions ("stored dtype is the model's dtype", "nothing derivable is
//! stored", "ragged state is never padded", "one aligned `read_exact`") that
//! make the files small and the loads fast. This module holds only what is
//! specific to Naive Bayes, and re-exports the rest so estimator code and the
//! test suite reach everything through one path.
//!
//! Each estimator implements its own `save` / `load` against
//! [`NbWriter`] / [`NbFile`] in its own file, which is what lets the fitted
//! fields stay private and keeps the D-03 "five mutually-independent structs,
//! shared math as free functions" shape intact (contrast a `Serialize` derive,
//! which would have to make every fitted field public or bolt a trait onto all
//! five).
//!
//! ## Naming, and the one place it is broken
//!
//! Bare tensor names are sklearn's fitted attribute names, so
//! `safetensors.numpy.load_file(path)` in Python returns a dict keyed the way
//! the sklearn estimator is; `param:`-prefixed entries are the constructor
//! inputs. The single exception is `BernoulliNB`, which does not hold sklearn's
//! `feature_log_prob_` at all — it holds the folded GEMM operand
//! `log p − log(1 − p)`, and stores it under `feature_log_prob_delta_` rather
//! than lie about the contents. `bernoulli_nb::MATRIX_NAME` documents the
//! trade in full.
//!
//! Tests live in `crates/mlrs-algos/tests/nb_persist_test.rs` (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

// The container is shared with the linear family; only the discriminator and
// the NB-shaped helpers below are local. Re-exported (not just imported) so
// `naive_bayes::nb_persist::{AlignedBytes, SaveModel, …}` keeps resolving for
// the five estimators and the test suite.
pub use crate::persist::{
    as_f64, as_floats, as_i64, as_usizes, expect_len, shape_1d, shape_2d, AlignedBytes, Container,
    LoadModel, ModelFile, ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The Naive Bayes container discriminator (`format = "mlrs-nb"`).
///
/// A zero-sized marker, never constructed — it exists only to carry the two
/// tags [`ModelFile::parse`] validates, so `NbWriter`/`NbFile` cannot be
/// instantiated with the linear family's tag by accident.
pub struct NbContainer;

impl Container for NbContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`. Present so a `.safetensors` file produced
/// by some other project — or by the `mlrs-linear` family — fails the check at
/// [`NbFile::parse`].
pub const FORMAT_ID: &str = "mlrs-nb";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`NbFile::parse`] rejects anything else outright (a prototype has
/// no back-compat obligations, so this is a hard equality, not a range).
pub const FORMAT_VERSION: &str = "1";

/// The NB writer: [`ModelWriter`] pinned to the `mlrs-nb` container, so an
/// estimator's `save` names only itself (`NbWriter::new("gaussian_nb")`) and
/// never a format id.
pub type NbWriter<'a> = ModelWriter<'a, NbContainer>;

/// The NB reader: [`ModelFile`] pinned to the `mlrs-nb` container. Loading a
/// `mlrs-linear` file through it fails with
/// [`PersistError::NotAnMlrsModel`] before any tensor is touched.
pub type NbFile<'a> = ModelFile<'a, NbContainer>;

// ---------------------------------------------------------------------------
// The shared discrete-NB core
// ---------------------------------------------------------------------------

/// The fitted state and hyperparameters that `MultinomialNB`, `BernoulliNB` and
/// `ComplementNB` all hold identically: one `n_classes × n_features` device
/// matrix, the per-class log-prior, the class labels, and the four
/// additive-smoothing knobs.
///
/// Staged BORROWING for the write side, so the estimators' host readbacks flow
/// into the serializer without a second copy. Named fields rather than a
/// positional function because the shape has three adjacent `bool`s and two
/// adjacent `&[f64]`s — exactly the signature a transposed argument slips
/// through unnoticed.
///
/// The three variants each keep their OWN `save`/`load` (D-03: independent
/// structs), and only this common middle is shared — the same split
/// [`nb_common`](crate::naive_bayes::nb_common) makes for the fit math.
pub struct DiscreteCoreRef<'a, F> {
    /// The name to store the `n_classes × n_features` matrix under.
    ///
    /// A parameter rather than a constant because `BernoulliNB` does NOT hold
    /// sklearn's `feature_log_prob_`: it stores the folded GEMM operand
    /// `log p − log(1 − p)`, so calling its tensor `feature_log_prob_` would
    /// mislabel the contents for anyone reading the file from Python.
    pub matrix_name: &'static str,
    /// Additive smoothing.
    pub alpha: f64,
    /// Whether the D-06 tiny-`alpha` clip was suppressed.
    pub force_alpha: bool,
    /// Whether the class prior was learned from the data.
    pub fit_prior: bool,
    /// The user-supplied class prior, if any.
    pub class_prior: Option<&'a [f64]>,
    /// The distinct sorted class labels.
    pub classes: &'a [i64],
    /// The per-class log-prior.
    pub class_log_prior: &'a [f64],
    /// The `n_classes × n_features` row-major matrix, at the model's own width.
    pub feature_log_prob: &'a [F],
    /// The fitted feature count (the matrix's column extent).
    pub n_features: usize,
}

impl<'a, F: Pod> DiscreteCoreRef<'a, F> {
    /// Stage every shared tensor and scalar into `w`.
    ///
    /// Variant-specific state is staged by the caller either side of this —
    /// `ComplementNB`'s `norm`, `BernoulliNB`'s `binarize` and `neg_prob_sum_`.
    pub fn write_into(self, w: &mut NbWriter<'a>) -> Result<(), PersistError> {
        let n_classes = self.classes.len();
        expect_len(
            "class_log_prior_",
            self.class_log_prior.len(),
            n_classes,
            "entries",
        )?;

        w.scalar_f64("param:alpha", self.alpha);
        w.scalar_bool("param:force_alpha", self.force_alpha);
        w.scalar_bool("param:fit_prior", self.fit_prior);

        w.tensor(
            self.matrix_name,
            TensorRef::floats(self.feature_log_prob, vec![n_classes, self.n_features])?,
        );
        w.tensor("classes_", TensorRef::i64s(self.classes, vec![n_classes])?);
        w.tensor(
            "class_log_prior_",
            TensorRef::f64s(self.class_log_prior, vec![n_classes])?,
        );
        if let Some(prior) = self.class_prior {
            w.tensor(
                "param:class_prior",
                TensorRef::f64s(prior, vec![prior.len()])?,
            );
        }
        Ok(())
    }
}

/// The owned counterpart of [`DiscreteCoreRef`], as recovered from a file.
///
/// `feature_log_prob` is a [`Cow`] rather than a `Vec` on purpose: when the
/// file's dtype matches `F` it BORROWS the mapped file bytes, so the matrix
/// goes to the device without an intervening allocation. Owning it here would
/// silently undo the whole zero-copy read path for the three estimators that
/// use it.
/// The `F: Clone` bound is what `Cow<'a, [F]>` needs (`[F]: ToOwned`). It is
/// implied by the `F: Pod` every caller already has, but a struct carries no
/// bound from its constructor, so it is spelled here.
pub struct DiscreteCore<'a, F: Clone> {
    /// Additive smoothing.
    pub alpha: f64,
    /// Whether the D-06 tiny-`alpha` clip was suppressed.
    pub force_alpha: bool,
    /// Whether the class prior was learned from the data.
    pub fit_prior: bool,
    /// The user-supplied class prior, if the file carried one.
    pub class_prior: Option<Vec<f64>>,
    /// The distinct sorted class labels.
    pub classes: Vec<i64>,
    /// The per-class log-prior.
    pub class_log_prior: Vec<f64>,
    /// The `n_classes × n_features` row-major matrix.
    pub feature_log_prob: Cow<'a, [F]>,
    /// Recovered from the matrix's row extent, not stored separately.
    pub n_classes: usize,
    /// Recovered from the matrix's column extent, not stored separately.
    pub n_features: usize,
}

/// Read back everything [`DiscreteCoreRef::write_into`] staged, validating the
/// geometry against the matrix's own shape.
///
/// `matrix_name` must match what the writer used. The file is untrusted, so
/// every 1-D tensor is measured against the matrix's row extent before any
/// value is handed back.
pub fn read_discrete_core<'a, F: Pod>(
    file: &NbFile<'a>,
    matrix_name: &'static str,
) -> Result<DiscreteCore<'a, F>, PersistError> {
    let matrix_v = file.tensor(matrix_name)?;
    let (n_classes, n_features) = shape_2d(&matrix_v, matrix_name)?;

    let classes_v = file.tensor("classes_")?;
    let prior_v = file.tensor("class_log_prior_")?;
    for (name, view) in [("classes_", &classes_v), ("class_log_prior_", &prior_v)] {
        expect_len(name, shape_1d(view, name)?, n_classes, "entries")?;
    }

    let class_prior = match file.tensor_opt("param:class_prior") {
        None => None,
        Some(view) => {
            let len = shape_1d(&view, "param:class_prior")?;
            expect_len("param:class_prior", len, n_classes, "entries")?;
            Some(as_f64(&view, "param:class_prior")?.into_owned())
        }
    };

    Ok(DiscreteCore {
        alpha: file.scalar_f64("param:alpha")?,
        force_alpha: file.scalar_bool("param:force_alpha")?,
        fit_prior: file.scalar_bool("param:fit_prior")?,
        class_prior,
        classes: as_i64(&classes_v, "classes_")?.into_owned(),
        class_log_prior: as_f64(&prior_v, "class_log_prior_")?.into_owned(),
        feature_log_prob: as_floats::<F>(&matrix_v, matrix_name)?,
        n_classes,
        n_features,
    })
}

// ---------------------------------------------------------------------------
// Ragged (CSR-style) payloads
// ---------------------------------------------------------------------------

/// The total element count of a ragged stack of `n_rows × extents[j]`
/// row-major blocks laid out back to back.
///
/// `CategoricalNB`'s `feature_log_prob_` is the live case: one
/// `n_classes × n_categories_[j]` matrix per feature. The two obvious storage
/// alternatives both lose:
///
/// * **one tensor per block** — correct, but costs an `n_features`-long JSON
///   header (~60 bytes each), which on a wide, low-cardinality dataset can
///   exceed the payload itself;
/// * **pad to a dense `[n_features, n_rows, max_extent]` cube** — costs
///   `max_extent / mean_extent` of the whole file, so one 100-category column
///   beside ninety-nine binary ones inflates it roughly 50×.
///
/// Flat concatenation costs neither, and needs no offsets tensor either: the
/// extents are `n_categories_`, a fitted attribute the file had to carry
/// anyway. The estimator holds the blocks in this SAME flat layout, so there is
/// no scatter/gather between memory and disk — a save writes the buffer
/// directly and a load fills one allocation. All this function does is compute
/// how long that buffer must be, so the reader can check the header's claim
/// against the payload it actually got.
///
/// Every extent comes from an untrusted header, so the walk is
/// overflow-checked at each step: a hostile `n_categories_` entry near
/// `usize::MAX` must produce an [`PersistError::InconsistentGeometry`], not a
/// wrapped multiply that then indexes out of range.
pub fn ragged_payload_len(
    n_rows: usize,
    extents: &[usize],
    tensor: &'static str,
) -> Result<usize, PersistError> {
    let bad = |reason: String| PersistError::InconsistentGeometry { reason };

    let mut total = 0usize;
    for (j, &extent) in extents.iter().enumerate() {
        let span = n_rows.checked_mul(extent).ok_or_else(|| {
            bad(format!(
                "tensor '{tensor}': block {j} of {n_rows} × {extent} overflows usize"
            ))
        })?;
        total = total.checked_add(span).ok_or_else(|| {
            bad(format!(
                "tensor '{tensor}': the extents overrun the address space at block {j}"
            ))
        })?;
    }
    Ok(total)
}

/// The start offset of each ragged block, plus a final end sentinel — so
/// `offsets[j]..offsets[j + 1]` is block `j` and `offsets.len() == extents.len()
/// + 1`.
///
/// The inverse of [`ragged_payload_len`]'s accounting, for callers that need to
/// address individual blocks inside the flat buffer. Unchecked arithmetic
/// because the only caller works from ALREADY-validated in-memory state; the
/// checked twin above is what guards the untrusted file.
pub fn ragged_block_offsets(n_rows: usize, extents: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(extents.len() + 1);
    let mut acc = 0usize;
    offsets.push(0);
    for &extent in extents {
        acc += n_rows * extent;
        offsets.push(acc);
    }
    offsets
}
