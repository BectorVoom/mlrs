//! `linear_persist` (LINEAR-PERSIST, prototype) — the `mlrs-linear` half of the
//! mlrs model file format: the container discriminator, the aliases the linear
//! estimators write and read through, and the shared dense-linear core.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic; the Naive Bayes family
//! ([`nb_persist`](crate::naive_bayes::nb_persist)) sits on the same machinery
//! under its own `mlrs-nb` tag. Read that module's docs for the four decisions
//! that make the files small and the loads fast. This module holds only what is
//! specific to a dense linear model, and re-exports the rest so estimator code
//! and the test suite reach everything through one path.
//!
//! ## The on-disk shape
//!
//! Every dense linear model in this family — `LinearRegression`, `Ridge`,
//! `Lasso` and `ElasticNet` today, and the shape `RidgeClassifier` would add its
//! own `classes_` to — is the same two fitted arrays:
//!
//! | name | dtype | shape | what |
//! |---|---|---|---|
//! | `coef_` | `F` (`F32`/`F64`) | `[n_targets, n_features]` | the weight matrix, row-major |
//! | `intercept_` | `F` | `[n_targets]` | the per-target bias |
//! | `param:*` | `__metadata__` | — | every constructor scalar (`fit_intercept`, `alpha`, …) |
//!
//! and NOTHING else: `n_targets` and `n_features` are recovered from `coef_`'s
//! shape at load, never written twice.
//!
//! ### Why `coef_` is one fused block
//!
//! The row-major `[n_targets, n_features]` layout is EXACTLY the layout the
//! fitted [`DeviceArray`](mlrs_backend::device_array::DeviceArray) already
//! holds, so neither direction reshuffles anything: a save is one device
//! readback streamed straight into the file, and a load hands the file's own
//! bytes to `DeviceArray::from_host` with no intermediate `Vec` and no
//! per-element decode. Splitting it into one tensor per target would instead
//! cost `n_targets` JSON header entries (~60 bytes each) and `n_targets`
//! separate uploads, for no benefit — the targets are never read independently.
//!
//! ### Why `intercept_` is a separate tensor and not folded into `coef_`
//!
//! The obvious "minimal file" move is the augmented-weight trick: store
//! `[n_targets, n_features + 1]` with the bias in each row's last column, and
//! save the ~60-byte header entry that a second tensor costs. It is rejected on
//! purpose. The bias would then be INTERLEAVED between the coefficient rows, so
//! `load` could no longer borrow `coef_` as one contiguous `&[F]` — it would
//! have to walk the buffer and copy `n_targets × n_features` elements into a
//! fresh allocation to strip the bias column out. That trades a fixed ~60 bytes
//! of header for an `O(n_targets · n_features)` copy on EVERY load, which is
//! backwards: the header cost is constant while the copy grows with the model.
//! Two tensors keep both fields zero-copy in both directions at any
//! `n_targets`.
//!
//! Storing `intercept_` as an array rather than a `__metadata__` scalar is the
//! same call one step further: it is a scalar only when `n_targets == 1`, and a
//! per-target-shape special case would be two code paths that can disagree, for
//! ~60 bytes on a file whose payload is `n_targets × n_features` floats.
//!
//! Tests live in `crates/mlrs-algos/tests/linear_persist_test.rs` (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

// The container is shared with the Naive Bayes family; only the discriminator
// and the linear-shaped helpers below are local. Re-exported (not just
// imported) so `linear::linear_persist::{AlignedBytes, SaveModel, …}` is the
// single import path for a linear estimator's `save`/`load`.
pub use crate::persist::{
    as_f64, as_floats, as_i64, as_usizes, expect_len, shape_1d, shape_2d, AlignedBytes, Container,
    LoadModel, ModelFile, ModelWriter, PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The dense-linear container discriminator (`format = "mlrs-linear"`).
///
/// A zero-sized marker, never constructed — it exists only to carry the two
/// tags [`ModelFile::parse`] validates, so `LinearWriter`/`LinearFile` cannot be
/// instantiated with the Naive Bayes family's tag by accident, and a
/// `GaussianNB` file cannot reach a linear estimator's geometry checks at all.
pub struct LinearContainer;

impl Container for LinearContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`. Present so a `.safetensors` file produced
/// by some other project — or by the `mlrs-nb` family — fails the check at
/// [`LinearFile::parse`].
pub const FORMAT_ID: &str = "mlrs-linear";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`LinearFile::parse`] rejects anything else outright (a prototype
/// has no back-compat obligations, so this is a hard equality, not a range).
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the weight matrix, row-major `[n_targets, n_features]`.
/// Named for sklearn's fitted attribute, so `safetensors.numpy.load_file(path)`
/// in Python hands back a dict keyed the way the sklearn estimator is.
pub const COEF_NAME: &str = "coef_";

/// The tensor holding the per-target bias, `[n_targets]`.
pub const INTERCEPT_NAME: &str = "intercept_";

/// The linear writer: [`ModelWriter`] pinned to the `mlrs-linear` container, so
/// an estimator's `save` names only itself
/// (`LinearWriter::new("linear_regression")`) and never a format id.
pub type LinearWriter<'a> = ModelWriter<'a, LinearContainer>;

/// The linear reader: [`ModelFile`] pinned to the `mlrs-linear` container.
/// Loading an `mlrs-nb` file through it fails with
/// [`PersistError::NotAnMlrsModel`] before any tensor is touched.
pub type LinearFile<'a> = ModelFile<'a, LinearContainer>;

/// The fitted state every dense linear model holds identically: one
/// `n_targets × n_features` weight matrix, the per-target bias, and whether the
/// bias was fitted at all.
///
/// Staged BORROWING for the write side, so the estimators' host readbacks flow
/// into the serializer without a second copy. Named fields rather than a
/// positional function because `coef` and `intercept` are both `&[F]` and
/// `n_targets`/`n_features` are both `usize` — exactly the signature a
/// transposed argument slips through unnoticed.
///
/// Each estimator keeps its OWN `save`/`load` and stages its own extra scalars
/// either side of this (`Ridge`'s `alpha`/`positive`, `ElasticNet`'s
/// `l1_ratio`, a classifier's `classes_`); only this common middle is shared,
/// the same split [`nb_persist`](crate::naive_bayes::nb_persist) makes for the
/// Naive Bayes core.
pub struct LinearCoreRef<'a, F> {
    /// The weight matrix, row-major `n_targets × n_features`.
    pub coef: &'a [F],
    /// The per-target bias, length `n_targets`. A model fitted with
    /// `fit_intercept = false` still writes it (as zeros) rather than omitting
    /// the tensor: an absent-vs-zero distinction the reader would have to
    /// reconstruct buys nothing, since `fit_intercept` is recorded separately.
    pub intercept: &'a [F],
    /// The output extent — 1 for a single-target regressor, `n_classes` for a
    /// one-vs-rest classifier.
    pub n_targets: usize,
    /// The fitted feature count (the matrix's column extent).
    pub n_features: usize,
    /// Whether the model centered `X`/`y` and recovered a bias term.
    pub fit_intercept: bool,
}

impl<'a, F: Pod> LinearCoreRef<'a, F> {
    /// Stage the two tensors and the `fit_intercept` scalar into `w`.
    ///
    /// Both tensors are written at `F`'s OWN width, so an `f32`-fitted model
    /// produces a file half the size of the `f64` one for the same geometry —
    /// and the dtype tag keeps the file self-describing, so that is a storage
    /// decision, not a commitment about how it is loaded back.
    pub fn write_into(self, w: &mut LinearWriter<'a>) -> Result<(), PersistError> {
        // Checked here rather than left to `TensorRef::new`'s shape check so the
        // error names the two fields that disagree, not just a shape and a
        // length. `coef`'s own length is validated by `TensorRef::floats`.
        expect_len(
            INTERCEPT_NAME,
            self.intercept.len(),
            self.n_targets,
            "entries",
        )?;

        w.scalar_bool("param:fit_intercept", self.fit_intercept);
        w.tensor(
            COEF_NAME,
            TensorRef::floats(self.coef, vec![self.n_targets, self.n_features])?,
        );
        w.tensor(
            INTERCEPT_NAME,
            TensorRef::floats(self.intercept, vec![self.n_targets])?,
        );
        Ok(())
    }
}

/// The owned counterpart of [`LinearCoreRef`], as recovered from a file.
///
/// Both arrays are [`Cow`]s rather than `Vec`s on purpose: when the file's
/// dtype matches `F` they BORROW the mapped file bytes, so they reach
/// [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray::from_host)
/// with no intervening allocation. Owning them here would silently undo the
/// whole zero-copy read path.
///
/// The `F: Clone` bound is what `Cow<'a, [F]>` needs (`[F]: ToOwned`). It is
/// implied by the `F: Pod` every caller already has, but a struct carries no
/// bound from its constructor, so it is spelled here.
pub struct LinearCore<'a, F: Clone> {
    /// The weight matrix, row-major `n_targets × n_features`.
    pub coef: Cow<'a, [F]>,
    /// The per-target bias, length `n_targets`.
    pub intercept: Cow<'a, [F]>,
    /// Recovered from `coef_`'s row extent, not stored separately.
    pub n_targets: usize,
    /// Recovered from `coef_`'s column extent, not stored separately.
    pub n_features: usize,
    /// Whether the model centered `X`/`y` and recovered a bias term.
    pub fit_intercept: bool,
}

// ---------------------------------------------------------------------------
// The one place two estimators disagree: the in-memory coefficient layout
// ---------------------------------------------------------------------------

/// Re-lay an `n_features × n_targets` (FEATURES-major) coefficient buffer as the
/// `n_targets × n_features` (TARGETS-major) one the file stores.
///
/// The file's layout is sklearn's — `Ridge.coef_` is `(n_targets, n_features)`
/// in Python — and that is not negotiable per estimator: the whole naming
/// contract is that `safetensors.numpy.load_file(path)["coef_"]` is the array
/// the sklearn attribute would be. But mlrs's [`Ridge`](crate::linear::ridge::Ridge)
/// holds the TRANSPOSE in memory, because its fused multi-target predict GEMM
/// consumes `n_features × n_targets`. So one of the two has to bend, and it is
/// the in-memory side, here, at the file boundary.
///
/// The cost is nil where it matters. A `1 × n` or `n × 1` matrix is the same
/// bytes in both layouts, so every single-target model — which is every model
/// [`Ridge::fit`](crate::typestate::Fit::fit) itself produces, and the only
/// shape `LinearRegression` has at all — takes the BORROWED arm and stays
/// zero-copy end to end. Only a genuine multi-target fit pays, and it pays one
/// `n_features · n_targets` permutation: the same order as the device readback
/// that produced the buffer, against an `O(n · d²)` fit.
///
/// The alternative — a layout flag in the header, letting each estimator store
/// whichever orientation it holds — was rejected. It would make `coef_` mean two
/// different shapes under one name, so a Python reader doing `X @ coef_.T` would
/// silently get garbage for half the family unless it thought to check a flag.
pub fn to_targets_major<F: Clone>(
    features_major: &[F],
    n_features: usize,
    n_targets: usize,
) -> Cow<'_, [F]> {
    transpose(features_major, n_features, n_targets)
}

/// The inverse of [`to_targets_major`]: re-lay the file's
/// `n_targets × n_features` buffer as the `n_features × n_targets` one a
/// features-major estimator uploads. Borrows on the single-target arm for the
/// same reason.
pub fn to_features_major<F: Clone>(
    targets_major: &[F],
    n_targets: usize,
    n_features: usize,
) -> Cow<'_, [F]> {
    transpose(targets_major, n_targets, n_features)
}

/// Row-major `rows × cols` → row-major `cols × rows`, borrowing when either
/// extent is 1 (a vector is its own transpose) or the buffer is degenerate.
///
/// The length guard is not defensive padding: `to_features_major`'s input comes
/// from an untrusted file, and while [`read_linear_core`] has already checked
/// `coef_`'s shape against its payload, this function is `pub` and must not
/// index out of range for a caller that has not.
fn transpose<F: Clone>(src: &[F], rows: usize, cols: usize) -> Cow<'_, [F]> {
    if rows <= 1 || cols <= 1 || src.len() != rows * cols {
        return Cow::Borrowed(src);
    }
    let mut out = Vec::with_capacity(src.len());
    for c in 0..cols {
        for r in 0..rows {
            out.push(src[r * cols + c].clone());
        }
    }
    Cow::Owned(out)
}

/// Read back everything [`LinearCoreRef::write_into`] staged, validating the
/// geometry against `coef_`'s own shape.
///
/// The file is UNTRUSTED input (T-04-01-01), so `coef_` defines the geometry and
/// every other extent is measured against it before a single value is handed
/// back — a header edited to claim more targets than `intercept_` holds fails
/// here, rather than reading out of bounds the first time the model predicts.
/// A zero extent is rejected for the same reason: a `[0, d]` or `[k, 0]` model
/// cannot predict anything, and an empty upload is a landmine on the device
/// backends.
pub fn read_linear_core<'a, F: Pod>(
    file: &LinearFile<'a>,
) -> Result<LinearCore<'a, F>, PersistError> {
    let coef_v = file.tensor(COEF_NAME)?;
    let (n_targets, n_features) = shape_2d(&coef_v, COEF_NAME)?;
    if n_targets == 0 || n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{COEF_NAME}' declares shape [{n_targets}, {n_features}]; \
                 a linear model needs at least one target and one feature"
            ),
        });
    }

    let intercept_v = file.tensor(INTERCEPT_NAME)?;
    expect_len(
        INTERCEPT_NAME,
        shape_1d(&intercept_v, INTERCEPT_NAME)?,
        n_targets,
        "entries",
    )?;

    // `as_floats` borrows the file buffer outright when the dtype matches `F`,
    // so both of these are views into the bytes `read_exact` landed — no
    // intermediate `Vec`, no per-element decode.
    Ok(LinearCore {
        coef: as_floats::<F>(&coef_v, COEF_NAME)?,
        intercept: as_floats::<F>(&intercept_v, INTERCEPT_NAME)?,
        n_targets,
        n_features,
        fit_intercept: file.scalar_bool("param:fit_intercept")?,
    })
}

// ---------------------------------------------------------------------------
// The coordinate-descent pair's shared knobs
// ---------------------------------------------------------------------------

/// The three hyperparameters `Lasso` and `ElasticNet` hold identically: the
/// penalty strength and the coordinate-descent stopping rule.
///
/// The two estimators are otherwise the same file — the shared core plus these,
/// and `ElasticNet` alone adds `param:l1_ratio` (`Lasso` IS `ElasticNet` with
/// `l1_ratio == 1`, per [`linear`](crate::linear)'s module docs, and their
/// `save`/`load` bodies are near-duplicates in exactly the way that invites one
/// to drift from the other).
///
/// The point is NOT line count — it is that each `param:` key is spelled ONCE.
/// Written out per estimator, `"param:alpha"` would appear in four places (two
/// estimators × save/load), and a typo in one of them is a file that saves fine
/// and loads a default. Here a typo cannot be one-sided.
///
/// `alpha` is `f64` even though the estimators hold it as `F`: it rides in
/// `__metadata__` as a shortest-round-trip decimal, so it converts at this
/// boundary through [`host_to_f64`](mlrs_core::host_to_f64) /
/// [`f64_to_host`](mlrs_core::f64_to_host) and survives `f32` and `f64` alike.
///
/// `Ridge` deliberately does NOT share this: its `max_iter` is an
/// `Option<usize>` (sklearn's `None` = "take the solver's own default"), which
/// is a different type carrying a different meaning, and collapsing the two
/// would mean inventing a sentinel for a value that has none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CdScalars {
    /// The penalty strength (`param:alpha`).
    pub alpha: f64,
    /// The coordinate-descent iteration cap (`param:max_iter`). A plain
    /// `usize` — unlike `Ridge`, both estimators default it to a concrete 1000.
    pub max_iter: usize,
    /// The coordinate-descent stopping tolerance (`param:tol`).
    pub tol: f64,
}

impl CdScalars {
    /// Stage all three into `w`.
    pub fn write_into(self, w: &mut LinearWriter<'_>) {
        w.scalar_f64("param:alpha", self.alpha);
        w.scalar_usize("param:max_iter", self.max_iter);
        w.scalar_f64("param:tol", self.tol);
    }

    /// Read all three back. Every one is REQUIRED: a missing key is a corrupt
    /// file, not a request for the default — silently substituting `alpha = 1.0`
    /// would hand back a model that predicts different numbers than the one that
    /// was saved.
    pub fn read(file: &LinearFile<'_>) -> Result<Self, PersistError> {
        Ok(CdScalars {
            alpha: file.scalar_f64("param:alpha")?,
            max_iter: file.scalar_usize("param:max_iter")?,
            tol: file.scalar_f64("param:tol")?,
        })
    }
}
