//! `kernel_persist` (KERNEL-PERSIST, prototype) — the `mlrs-kernel` half of the
//! mlrs model file format: the container discriminator, the aliases the two
//! kernel-method estimators write and read through, and the training-matrix core
//! they share.
//!
//! The container ITSELF — the safetensors layout, the writer/reader, the
//! 8-byte-aligned zero-copy read path, the typed tensor accessors, the error
//! type and the `SaveModel` / `LoadModel` surface — lives in [`crate::persist`]
//! and is estimator-agnostic. Read that module's docs for the four decisions
//! that make the files small and the loads fast.
//!
//! ## Why this container lives at the crate root
//!
//! Every other family's container sits inside its own module
//! ([`linear_persist`](crate::linear::linear_persist),
//! [`prep_persist`](crate::preprocessing::prep_persist), …), because every other
//! family is one module. This one is not: it serves
//! [`KernelRidge`](crate::kernel_ridge::KernelRidge) in `kernel_ridge/` and
//! [`KernelDensity`](crate::density::KernelDensity) in `density/`, which are two
//! modules for a reason unrelated to persistence (KD is not a neighbor estimator
//! in mlrs's trait sense, so it got its own home) but ONE family for this one:
//! both are kernel methods whose fitted model IS the training matrix. Putting
//! the container in either module would make the other import across a boundary
//! that means nothing here.
//!
//! ## The on-disk shape
//!
//! | name | dtype | shape | held by |
//! |---|---|---|---|
//! | `X_fit_` | `F` (`F32`/`F64`) | `[n_samples, n_features]` | both |
//! | `dual_coef_` | `F` | `[n_samples, n_targets]` | `KernelRidge` |
//! | `param:kernel` | `__metadata__` | — | both, different vocabularies |
//! | `param:alpha` / `param:gamma` / `param:degree` / `param:coef0` | `__metadata__` | — | `KernelRidge` |
//! | `gamma_` | `__metadata__` | — | `KernelRidge`, see [`write_resolved_gamma`] |
//! | `param:bandwidth` | `__metadata__` | — | `KernelDensity` |
//! | `bandwidth_` | `__metadata__` | — | `KernelDensity` |
//!
//! `n_samples` and `n_features` are recovered from `X_fit_`'s shape, and
//! `n_targets` from `dual_coef_`'s — none is stored again.
//!
//! ## The training set IS the model, and that is the whole size story
//!
//! A kernel method has no compressed parameterization: `predict` and
//! `score_samples` both evaluate the kernel against every training row, so the
//! matrix has to be there. These are therefore the LARGEST files mlrs writes for
//! a given problem — `n_samples × n_features` rather than the
//! `n_targets × n_features` a linear model gets away with — and the two levers
//! that remain are the ones the format already pulls everywhere: the stored
//! dtype is the model's own (an `f32` fit is half the size), and nothing
//! derivable is written beside it.
//!
//! Storing a decomposition of the kernel matrix instead — a Nyström or Cholesky
//! factor — would be smaller for `n_samples ≫ n_features`, and is not done: it
//! would change what the model computes, not merely how it is stored, and a
//! reloaded estimator would no longer predict what the saved one did.
//!
//! Tests live in `crates/mlrs-algos/tests/kernel_persist_test.rs` (AGENTS.md §2).

use std::borrow::Cow;

use bytemuck::Pod;

// The container is shared with every other family; only the discriminator and
// the kernel-shaped helpers below are local. Re-exported (not just imported) so
// `kernel_persist::{AlignedBytes, SaveModel, …}` is the single import path for a
// kernel method's `save`/`load`.
pub use crate::persist::{
    as_floats, expect_len, shape_2d, AlignedBytes, Container, LoadModel, ModelFile, ModelWriter,
    PersistError, SaveModel, TensorRef, PARAM_PREFIX,
};

/// The kernel-method container discriminator (`format = "mlrs-kernel"`).
pub struct KernelContainer;

impl Container for KernelContainer {
    const FORMAT: &'static str = FORMAT_ID;
    const VERSION: &'static str = FORMAT_VERSION;
}

/// The value written under `format`.
pub const FORMAT_ID: &str = "mlrs-kernel";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`KernelFile::parse`] rejects anything else outright.
pub const FORMAT_VERSION: &str = "1";

/// The tensor holding the training matrix, row-major `[n_samples, n_features]`.
///
/// Capitalized because sklearn's fitted attribute is `X_fit_` — the bare tensor
/// names in this format are sklearn's own, so a
/// `safetensors.numpy.load_file(path)` in Python hands back a dict keyed the way
/// the sklearn estimator is, capital and all.
pub const X_FIT_NAME: &str = "X_fit_";

/// The `__metadata__` key holding the kernel family's sklearn name.
///
/// The VOCABULARY differs between the two estimators — `KernelRidge` speaks
/// `linear`/`rbf`/`poly`/`sigmoid`, `KernelDensity` speaks
/// `gaussian`/`tophat`/`epanechnikov`/`exponential`/`linear`/`cosine` — and they
/// overlap on `linear` while meaning entirely different functions by it. Sharing
/// the KEY while each estimator owns its own parse is deliberate: the
/// `estimator` discriminator has already established which vocabulary applies
/// before this is read, so there is nothing for the two to disagree about.
pub const KERNEL_KEY: &str = "param:kernel";

/// The kernel-method writer: [`ModelWriter`] pinned to the `mlrs-kernel`
/// container.
pub type KernelWriter<'a> = ModelWriter<'a, KernelContainer>;

/// The kernel-method reader: [`ModelFile`] pinned to the `mlrs-kernel`
/// container.
pub type KernelFile<'a> = ModelFile<'a, KernelContainer>;

/// Stage the training matrix, rejecting a degenerate geometry.
///
/// Written at `F`'s OWN width. That decision carries more weight here than
/// anywhere else in the format: `X_fit_` is essentially the whole file for both
/// estimators, so an `f32` fit produces a model file half the size of its `f64`
/// twin outright.
pub fn write_x_fit<'a, F: Pod>(
    w: &mut KernelWriter<'a>,
    x_fit: &'a [F],
    n_samples: usize,
    n_features: usize,
) -> Result<(), PersistError> {
    if n_samples == 0 || n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "'{X_FIT_NAME}' would be [{n_samples}, {n_features}]; a fitted kernel \
                 method has at least one training sample and one feature"
            ),
        });
    }
    w.tensor(
        X_FIT_NAME,
        TensorRef::floats(x_fit, vec![n_samples, n_features])?,
    );
    Ok(())
}

/// Read the training matrix back with its `(n_samples, n_features)`.
///
/// The shape IS the schema — both extents come off it rather than being stored
/// separately (decision 2 in [`crate::persist`]'s docs). A zero extent is
/// rejected because a kernel method with no training rows has nothing to
/// evaluate against, and an empty upload is a landmine on the device backends.
///
/// The returned [`Cow`] BORROWS the mapped file bytes when the dtype matches
/// `F`, so the largest tensor mlrs writes reaches
/// [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray::from_host)
/// without a single copy — which is the case this whole read path was designed
/// for.
pub fn read_x_fit<'a, F: Pod>(
    file: &KernelFile<'a>,
) -> Result<(Cow<'a, [F]>, usize, usize), PersistError> {
    let view = file.tensor(X_FIT_NAME)?;
    let (n_samples, n_features) = shape_2d(&view, X_FIT_NAME)?;
    if n_samples == 0 || n_features == 0 {
        return Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{X_FIT_NAME}' declares shape [{n_samples}, {n_features}]; a \
                 fitted kernel method has at least one training sample and one feature"
            ),
        });
    }
    Ok((as_floats::<F>(&view, X_FIT_NAME)?, n_samples, n_features))
}

/// The `__metadata__` key holding `KernelRidge`'s RESOLVED kernel coefficient.
///
/// Distinct from `param:gamma`, which is the constructor argument and is
/// OPTIONAL — sklearn's `gamma=None` means "use `1/n_features`". Both are
/// stored, because they are two different facts and a reloaded model reports
/// each.
pub const GAMMA_FITTED_KEY: &str = "gamma_";

/// Stage the resolved kernel coefficient alongside the request.
///
/// The alternative was to store only `param:gamma` and re-run the
/// `gamma.unwrap_or(1 / n_features)` resolution at load, since `n_features` is
/// recoverable from `X_fit_`'s shape. It is rejected because it would put the
/// SAME rule in two places — the fit body and the loader — with nothing to keep
/// them in step. A later change to how `None` resolves (sklearn has changed this
/// kind of default before) would silently give every previously-saved model a
/// different kernel, and the two arms would disagree with no error anywhere.
/// Storing what the fit actually used costs one scalar and makes the file the
/// authority.
pub fn write_resolved_gamma(w: &mut KernelWriter<'_>, request: Option<f64>, resolved: f64) {
    w.scalar_opt_f64("param:gamma", request);
    w.scalar_f64(GAMMA_FITTED_KEY, resolved);
}

/// Read back the `(request, resolved)` pair [`write_resolved_gamma`] staged.
///
/// The resolved half is REQUIRED and the request is optional, which is exactly
/// the asymmetry the two keys exist to express: an absent request is a
/// meaningful `None`, while an absent resolution is a corrupt file — a kernel
/// whose coefficient is unknown cannot be evaluated at all.
pub fn read_resolved_gamma(file: &KernelFile<'_>) -> Result<(Option<f64>, f64), PersistError> {
    Ok((
        file.scalar_opt_f64("param:gamma")?,
        file.scalar_f64(GAMMA_FITTED_KEY)?,
    ))
}
