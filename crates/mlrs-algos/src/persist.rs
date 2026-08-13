//! `persist` — the estimator-agnostic safetensors container every mlrs model
//! file is written into and read back from.
//!
//! This module owns the *format* only. It knows nothing about any estimator:
//! each one implements its own `save` / `load` against the
//! [`ModelWriter`] / [`ModelFile`] pair inside its own module, which is what
//! lets the fitted fields stay private (contrast a `Serialize` derive, which
//! would have to make every fitted field public or bolt a trait onto every
//! estimator struct).
//!
//! Two families sit on top of it today, each pinning its own [`Container`]
//! discriminator so a file of one can never load as the other:
//!
//! | family | container | writer / reader |
//! |---|---|---|
//! | Naive Bayes | `mlrs-nb` | [`nb_persist`](crate::naive_bayes::nb_persist) |
//! | dense linear models | `mlrs-linear` | [`linear_persist`](crate::linear::linear_persist) |
//!
//! ## The layout, and why it is this shape
//!
//! A safetensors file is `u64 header_len | JSON header | raw little-endian
//! tensor bytes`. The JSON header maps a name to `{dtype, shape, data_offsets}`
//! and may carry one free-form `__metadata__` string→string map. mlrs uses the
//! two halves for two different kinds of state:
//!
//! | goes in | what | why |
//! |---|---|---|
//! | tensors | every fitted **array** (`coef_`, `theta_`, `feature_log_prob_`, …) | 8 bytes per `f64`, 4 per `f32`, no decode step |
//! | `__metadata__` | every **scalar** (`fit_intercept`, `alpha`, `epsilon_`, the bool knobs) and the format/estimator tags | a scalar in its own tensor costs ~60 bytes of JSON header for 8 bytes of payload |
//!
//! Four decisions do the actual work of "small file, fast load":
//!
//! 1. **The stored dtype is the model's dtype.** A `LinearRegression<f32>`
//!    writes `F32` tensors, not widened `f64`. That is a straight 2× on the
//!    dominant term, and it costs nothing at load because the dtype tag makes
//!    the file self-describing — [`as_floats`] reinterprets when the tag matches
//!    `F` and converts only when it does not, so an f32 file still loads into an
//!    `f64` model (train on a GPU in f32, evaluate in f64).
//! 2. **Nothing derivable is stored.** `n_features`, `n_classes` and `n_targets`
//!    are read back off the tensor *shapes*; there is no redundant copy in the
//!    header to disagree with them.
//! 3. **Ragged state is stored CSR-style, never padded** — see
//!    [`ragged_payload_len`](crate::naive_bayes::nb_persist::ragged_payload_len),
//!    which is where the one estimator that needs it documents the trade.
//! 4. **The read is one sequential `read_exact` into an 8-byte-aligned buffer**
//!    ([`AlignedBytes`]). Alignment is the whole trick: safetensors orders
//!    tensors by descending dtype width and pads the header to a multiple of 8,
//!    so in an 8-aligned buffer every `f64`/`i64`/`u64` tensor lands on its
//!    natural alignment and `bytemuck::try_cast_slice` succeeds — the file bytes
//!    become `&[f64]` with no copy and no parse, and go straight into
//!    [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray::from_host).
//!    A plain `Vec<u8>` from `fs::read` is only 1-byte aligned and would force a
//!    copy of every tensor. [`as_f64`] and friends still fall back to an
//!    unaligned element-wise read rather than failing, because a file written by
//!    some other tool carries no such guarantee.
//!
//! ## Naming
//!
//! Bare tensor names are sklearn's fitted attribute names, so
//! `safetensors.numpy.load_file(path)` in Python returns a dict keyed the way
//! the sklearn estimator is; [`PARAM_PREFIX`]-prefixed entries are the
//! constructor inputs.
//!
//! ## Reproducibility — a saved model is a deterministic function of itself
//!
//! Saving the SAME model twice produces byte-identical files, so a model file
//! can be content-addressed, deduplicated, and byte-diffed. [`ModelWriter`]
//! keeps its scalars in a [`BTreeMap`] and safetensors emits tensor entries
//! through an index map, so both halves of the header are ordered.
//!
//! That took an upstream patch. Stock safetensors 0.8.0 hands `__metadata__`
//! to `serde_json` as a `std::collections::HashMap`, whose iteration order
//! comes from a randomly seeded hasher — so the scalars shuffle between runs
//! even though the payload is stable. A caller cannot work around it, because
//! `serialize` / `serialize_to_file` take that `HashMap` by value in their
//! signatures. `third_party/safetensors` is 0.8.0 with the metadata maps
//! retyped to `BTreeMap`, wired in through the workspace's `[patch.crates-io]`;
//! see its README for the diff and upstream status.
//!
//! Both families' test suites carry a `saving_twice_produces_an_identical_model`
//! gate that compares raw file bytes — if the patch is ever dropped, those tests
//! fail rather than the property silently regressing.
//!
//! ## What this is not
//!
//! Reading the whole file into memory is right for the models mlrs fits (a
//! coefficient table or a class-conditional matrix — kilobytes to megabytes) and
//! wrong for a multi-gigabyte one. The swap is local: give [`AlignedBytes`] a
//! `memmap2` variant and [`ModelFile::parse`] is unchanged, because it already
//! takes a plain `&[u8]` and every tensor accessor borrows from it.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use bytemuck::Pod;
use mlrs_backend::pool::BufferPool;
use mlrs_backend::runtime::ActiveRuntime;
use safetensors::tensor::{Metadata, TensorView, View};
use safetensors::{serialize_to_file, Dtype, SafeTensorError, SafeTensors};
use thiserror::Error;

// safetensors is a little-endian format by specification (`TensorInfo`:
// "Endianness is assumed to be little endian"), and the whole zero-copy read
// path below reinterprets file bytes as native `f64`/`i64`. On a big-endian
// host that reinterpretation is silently wrong rather than an error, so refuse
// to build there instead of shipping a byte-swapped model loader.
#[cfg(target_endian = "big")]
compile_error!(
    "persist reinterprets little-endian safetensors payloads as native floats; \
     a big-endian target needs an explicit byte-swap path that does not exist yet"
);

/// `__metadata__` key holding the container discriminator, so a file that is
/// *not* an mlrs model of the expected family is rejected by name rather than
/// by a confusing missing-tensor error.
const KEY_FORMAT: &str = "format";
/// `__metadata__` key holding [`Container::VERSION`].
const KEY_VERSION: &str = "version";
/// `__metadata__` key holding the estimator discriminator (`"gaussian_nb"`,
/// `"linear_regression"`, …), which is what stops one estimator's file loading
/// as another's.
const KEY_ESTIMATOR: &str = "estimator";

/// The `param:` prefix marks a *constructor hyperparameter* as opposed to a
/// fitted attribute. Bare tensor/metadata names are exactly sklearn's fitted
/// attribute names (`coef_`, `theta_`, `class_count_`, …), so a
/// `safetensors.numpy.load_file(path)` in Python hands back a dict keyed the way
/// the sklearn estimator is, and the `param:`-prefixed entries are visibly the
/// inputs.
///
/// Estimators spell their keys as literals (`"param:alpha"`) rather than
/// building them through a helper, because [`ModelFile`]'s scalar readers take
/// `&'static str` so that [`PersistError::BadMetadata`] can name the key without
/// allocating.
pub const PARAM_PREFIX: &str = "param:";

/// The container discriminator one *family* of estimators shares.
///
/// A zero-sized marker carrying the two `__metadata__` tags every file of that
/// family declares and every load validates. Making it a type parameter rather
/// than a `ModelWriter::new` argument is what lets each family expose a plain
/// alias (`type NbWriter<'a> = ModelWriter<'a, NbContainer>`) whose
/// constructors cannot be called with the wrong tag — the format id is not a
/// value any estimator's `save` has to remember to pass.
pub trait Container {
    /// The value written under `format` (`"mlrs-nb"`, `"mlrs-linear"`). Present
    /// so a `.safetensors` file produced by some other project — or by the other
    /// family — fails the check at [`ModelFile::parse`].
    const FORMAT: &'static str;

    /// The container version. Bump on any layout change that an older reader
    /// would mis-read; [`ModelFile::parse`] rejects anything else outright (a
    /// prototype has no back-compat obligations, so this is a hard equality, not
    /// a range).
    const VERSION: &'static str;
}

/// Failures raised while saving or loading a model file.
///
/// Kept separate from [`AlgoError`](crate::error::AlgoError) because every
/// variant here is an I/O or file-contents fault, not a fit/predict fault, and
/// because a persistence error carries a `PathBuf` that no estimator error
/// wants. The file is UNTRUSTED input (T-04-01-01): every geometry implied by
/// the header is checked against every other before a single value is handed to
/// an estimator, so a hand-edited header produces an `InconsistentGeometry`
/// rather than an out-of-bounds read at predict time.
#[derive(Debug, Error)]
pub enum PersistError {
    /// The file could not be opened, read, written, or renamed into place.
    #[error("model I/O failed for '{path}': {source}")]
    Io {
        /// The file being read or written.
        path: PathBuf,
        /// The underlying `std::io` failure.
        source: std::io::Error,
    },

    /// The safetensors container itself is malformed (bad header length,
    /// non-UTF-8 or non-JSON header, offsets that do not cover the buffer).
    #[error("model file is not a valid safetensors container: {0}")]
    Container(#[from] SafeTensorError),

    /// The container parsed, but it is not an mlrs model of the expected family
    /// (or is a version this build cannot read).
    #[error("expected an '{expected}' v{version} container, found '{found}'")]
    NotAnMlrsModel {
        /// The [`Container::FORMAT`] of the family whose `load` was called.
        expected: &'static str,
        /// That family's [`Container::VERSION`].
        version: &'static str,
        /// What the file actually declared.
        found: String,
    },

    /// The file holds a different estimator than the one being loaded.
    #[error("this is a '{found}' model file; '{expected}' cannot load it")]
    WrongEstimator {
        /// The estimator whose `load` was called.
        expected: &'static str,
        /// The estimator that wrote the file.
        found: String,
    },

    /// A tensor the estimator requires is absent from the header.
    #[error("model file is missing the required tensor '{tensor}'")]
    MissingTensor {
        /// The absent tensor's name.
        tensor: &'static str,
    },

    /// A `__metadata__` entry the estimator requires is absent, or does not
    /// parse as the expected scalar type.
    #[error("model file has a missing or unparsable '{key}' metadata entry")]
    BadMetadata {
        /// The offending `__metadata__` key.
        key: &'static str,
    },

    /// A tensor is present but carries a dtype this reader cannot interpret as
    /// the requested element type.
    #[error("tensor '{tensor}': expected dtype {expected:?}, found {found:?}")]
    DtypeMismatch {
        /// The tensor's name.
        tensor: &'static str,
        /// The dtype the reader asked for.
        expected: Dtype,
        /// The dtype recorded in the header.
        found: Dtype,
    },

    /// Two or more tensors imply geometries that cannot both be true — the
    /// central guard against a tampered or truncated header.
    #[error("model file geometry is inconsistent: {reason}")]
    InconsistentGeometry {
        /// Which invariant failed, naming both tensors and both extents.
        reason: String,
    },

    /// `save` was called on an estimator whose fitted state is somehow absent.
    /// Unreachable through the typestate surface (only a `Fitted` value exposes
    /// `save`), and kept as a typed error rather than an `unwrap` so a future
    /// partial-fit path cannot turn it into a panic.
    #[error("estimator '{estimator}' cannot be saved: fitted '{field}' is absent")]
    MissingState {
        /// The estimator being saved.
        estimator: &'static str,
        /// The fitted field that was `None`.
        field: &'static str,
    },

    /// The element type is neither 4- nor 8-byte wide, so it has no
    /// safetensors float tag. Unreachable for the `f32`/`f64` instantiations
    /// mlrs actually builds.
    #[error("unsupported float width {width} bytes (mlrs floats are f32/f64)")]
    UnsupportedFloatWidth {
        /// `size_of::<F>()` for the offending `F`.
        width: usize,
    },
}

impl PersistError {
    /// Attach the path to an `io::Error` — every I/O failure in this module
    /// names the file it happened on.
    fn io(path: &Path, source: std::io::Error) -> Self {
        PersistError::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

// ---------------------------------------------------------------------------
// The uniform save/load surface
// ---------------------------------------------------------------------------

/// Write a fitted estimator to a safetensors file.
///
/// One signature for every estimator, so a caller can persist a model without
/// knowing which one it holds. That uniformity is the whole reason this is a
/// trait rather than an inherent method per estimator: the alternative was
/// `GaussianNB` taking a [`BufferPool`] and `CategoricalNB` not, which reads as
/// an arbitrary inconsistency at every generic call site.
///
/// `pool` is present because most estimators keep fitted tables on the device
/// and must read them back. The host-only ones ignore it — the same shape
/// [`Fit`](crate::typestate::Fit) already has, where every estimator takes a
/// pool because the surface is shared, not because each one launches a kernel.
///
/// This does NOT introduce a shared base struct: estimators stay independent
/// structs implementing a shared *behavior* trait, exactly as they already do
/// for `Fit` / `Predict` / `PredictProba`.
pub trait SaveModel {
    /// Serialize to `path`, replacing any existing file atomically.
    fn save(&self, pool: &BufferPool<ActiveRuntime>, path: &Path) -> Result<(), PersistError>;
}

/// Read a fitted estimator back from a safetensors file.
///
/// The counterpart of [`SaveModel`]. Implemented on the `Fitted`-tagged
/// estimator, because a file only ever holds a fitted model — which means the
/// state parameter is named at the call site, by turbofish or by annotating the
/// binding:
///
/// ```ignore
/// use mlrs_algos::persist::LoadModel;
/// let est: LinearRegression<f32, Fitted> = LinearRegression::load(&mut pool, path)?;
/// ```
pub trait LoadModel: Sized {
    /// Deserialize from `path`, re-uploading any device-resident tables to
    /// `pool`. Fails if the file is not this estimator's own.
    fn load(pool: &mut BufferPool<ActiveRuntime>, path: &Path) -> Result<Self, PersistError>;
}

// ---------------------------------------------------------------------------
// Write side
// ---------------------------------------------------------------------------

/// A single tensor staged for writing, BORROWING its payload.
///
/// The borrow is the point: `safetensors`' [`View::data`] returns a
/// `Cow<[u8]>`, so a `Cow::Borrowed` over the caller's host `Vec<F>` lets
/// [`serialize_to_file`] stream straight out of it. The only copy on the whole
/// save path is the one unavoidable device→host readback.
pub struct TensorRef<'a> {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: &'a [u8],
}

impl<'a> TensorRef<'a> {
    /// Stage an `f64` array (the host-side fitted vectors: `class_count_`,
    /// `class_log_prior_`, a ragged `feature_log_prob_` payload).
    pub fn f64s(values: &'a [f64], shape: Vec<usize>) -> Result<Self, PersistError> {
        Self::new(Dtype::F64, shape, values)
    }

    /// Stage an `i64` array (`classes_` — sklearn's class labels are integral
    /// and mlrs stores them as `i64`, so they need no float tag).
    pub fn i64s(values: &'a [i64], shape: Vec<usize>) -> Result<Self, PersistError> {
        Self::new(Dtype::I64, shape, values)
    }

    /// Stage a `u64` array — counts and extents. Callers hold these as
    /// `Vec<usize>` and must widen explicitly: `usize` is 4 bytes on a 32-bit
    /// host and 8 on a 64-bit one, so writing it raw would produce a file that
    /// only loads back on the architecture that wrote it.
    pub fn u64s(values: &'a [u64], shape: Vec<usize>) -> Result<Self, PersistError> {
        Self::new(Dtype::U64, shape, values)
    }

    /// Stage a generic-float array at ITS OWN width — `F32` for an `f32` model,
    /// `F64` for an `f64` one. This is the decision that halves the file for
    /// every f32-fitted model; going through an estimator's `f64` accessors
    /// instead would widen on save and narrow on load for no gain.
    pub fn floats<F: Pod>(values: &'a [F], shape: Vec<usize>) -> Result<Self, PersistError> {
        Self::new(float_dtype::<F>()?, shape, values)
    }

    /// The shared constructor, which is also the only place the declared shape
    /// is checked against the payload.
    ///
    /// Without this the writer could emit a header whose `shape` disagrees with
    /// its `data_offsets`; safetensors would reject that on *load*, meaning a
    /// caller finds out a model is corrupt only when trying to use it. Failing
    /// on the save side instead keeps a bad file from ever reaching disk.
    fn new<T: Pod>(dtype: Dtype, shape: Vec<usize>, values: &'a [T]) -> Result<Self, PersistError> {
        let declared: usize = shape.iter().product();
        if declared != values.len() {
            return Err(PersistError::InconsistentGeometry {
                reason: format!(
                    "declared shape {shape:?} holds {declared} elements but the payload has {}",
                    values.len()
                ),
            });
        }
        Ok(TensorRef {
            dtype,
            shape,
            bytes: bytemuck::cast_slice(values),
        })
    }
}

impl View for TensorRef<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

/// Accumulates the `__metadata__` scalars and the tensor set for one model,
/// then writes them as a single safetensors file.
///
/// Borrows every payload for `'a`, so the host arrays a caller reads back off
/// the device must outlive the writer — which the estimators' `save` bodies get
/// for free by binding them before constructing the writer.
///
/// `C` pins the container discriminator; families use it through an alias
/// (`NbWriter`, `LinearWriter`) so no `save` body ever names a format id.
pub struct ModelWriter<'a, C: Container> {
    meta: BTreeMap<String, String>,
    tensors: Vec<(String, TensorRef<'a>)>,
    _container: PhantomData<fn() -> C>,
}

impl<'a, C: Container> ModelWriter<'a, C> {
    /// Start a writer for `estimator`, seeding the three discriminator keys
    /// that [`ModelFile::parse`] validates.
    pub fn new(estimator: &'static str) -> Self {
        let mut meta = BTreeMap::new();
        meta.insert(KEY_FORMAT.to_string(), C::FORMAT.to_string());
        meta.insert(KEY_VERSION.to_string(), C::VERSION.to_string());
        meta.insert(KEY_ESTIMATOR.to_string(), estimator.to_string());
        ModelWriter {
            meta,
            tensors: Vec::new(),
            _container: PhantomData,
        }
    }

    /// Record an `f64` scalar.
    ///
    /// `{:?}` rather than `{}` deliberately: both of Rust's float formatters
    /// emit the shortest decimal that round-trips through `str::parse`, so the
    /// value is recovered EXACTLY either way, but `{:?}` picks the exponent
    /// form when it is shorter (`1e-9`, not `0.000000001`).
    pub fn scalar_f64(&mut self, key: &str, value: f64) {
        self.meta.insert(key.to_string(), format!("{value:?}"));
    }

    /// Record a `usize` scalar.
    pub fn scalar_usize(&mut self, key: &str, value: usize) {
        self.meta.insert(key.to_string(), value.to_string());
    }

    /// Record a `bool` scalar (`fit_intercept`, `fit_prior`, `norm`, …).
    pub fn scalar_bool(&mut self, key: &str, value: bool) {
        self.meta.insert(key.to_string(), value.to_string());
    }

    /// Record a string scalar — used for the enum-shaped knobs whose variant
    /// tag is text and whose payload, if any, rides in a companion tensor.
    pub fn scalar_str(&mut self, key: &str, value: &str) {
        self.meta.insert(key.to_string(), value.to_string());
    }

    /// Record a `u64` scalar (`random_state`). Distinct from
    /// [`ModelWriter::scalar_usize`] because a seed is 64-bit by definition and
    /// must survive a round-trip through a 32-bit host, where `usize` would
    /// truncate it.
    pub fn scalar_u64(&mut self, key: &str, value: u64) {
        self.meta.insert(key.to_string(), value.to_string());
    }

    /// Record an OPTIONAL `f64` scalar. `None` writes no key at all rather than
    /// a `"null"` sentinel, so `Option` round-trips as key-presence and costs
    /// zero bytes when absent (`binarize=None` is the live case).
    pub fn scalar_opt_f64(&mut self, key: &str, value: Option<f64>) {
        if let Some(v) = value {
            self.scalar_f64(key, v);
        }
    }

    /// Record an OPTIONAL `usize` scalar, mirroring
    /// [`ModelWriter::scalar_opt_f64`]. `Ridge`'s `max_iter` and `n_iter_` are
    /// the live cases, and for both `None` is a MEANINGFUL value — "take the
    /// solver's own default" and "this solver does not report an iteration
    /// count" — not merely an absent one, which is why key-presence has to carry
    /// it rather than a `0` sentinel that would be indistinguishable from a real
    /// zero.
    pub fn scalar_opt_usize(&mut self, key: &str, value: Option<usize>) {
        if let Some(v) = value {
            self.scalar_usize(key, v);
        }
    }

    /// Record an OPTIONAL `u64` scalar (`random_state`), mirroring
    /// [`ModelWriter::scalar_opt_f64`].
    pub fn scalar_opt_u64(&mut self, key: &str, value: Option<u64>) {
        if let Some(v) = value {
            self.scalar_u64(key, v);
        }
    }

    /// Stage a tensor under `name`.
    pub fn tensor(&mut self, name: &str, tensor: TensorRef<'a>) {
        self.tensors.push((name.to_string(), tensor));
    }

    /// Stage an OPTIONAL tensor — absent means no header entry, mirroring
    /// [`ModelWriter::scalar_opt_f64`].
    pub fn tensor_opt(&mut self, name: &str, tensor: Option<TensorRef<'a>>) {
        if let Some(t) = tensor {
            self.tensor(name, t);
        }
    }

    /// Serialize to `path`.
    ///
    /// Writes to a sibling temporary and `rename`s it into place, so an
    /// interrupted save cannot leave a half-written file where a valid model
    /// used to be (`rename` within a directory is atomic on both Linux and
    /// Windows-with-`ReplaceFile` semantics). The temporary is a sibling rather
    /// than a `/tmp` entry precisely so the rename stays within one filesystem.
    pub fn write(self, path: &Path) -> Result<(), PersistError> {
        let tmp = temp_sibling(path);
        // `serialize_to_file` streams the staged views out through a
        // BufWriter — the whole file is never materialized in memory, which is
        // what keeps the save's peak footprint at "one device readback".
        serialize_to_file(self.tensors, Some(self.meta), &tmp).map_err(|e| match e {
            SafeTensorError::IoError(io) => PersistError::io(&tmp, io),
            other => PersistError::Container(other),
        })?;
        fs::rename(&tmp, path).map_err(|e| {
            // Best-effort cleanup; the rename failure is the error worth
            // reporting, so a failed removal is deliberately ignored.
            let _ = fs::remove_file(&tmp);
            PersistError::io(path, e)
        })
    }
}

/// `<path>.mlrs-tmp` — a sibling so the subsequent `rename` is same-filesystem
/// and therefore atomic.
fn temp_sibling(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".mlrs-tmp");
    path.with_file_name(name)
}

// ---------------------------------------------------------------------------
// Read side
// ---------------------------------------------------------------------------

/// The file's bytes, held in a buffer guaranteed to be 8-byte aligned.
///
/// A `Vec<u8>` (what `fs::read` returns) is aligned to 1, so reinterpreting a
/// slice of it as `&[f64]` fails and every tensor has to be copied out
/// element-wise. Backing the buffer with a `Vec<u64>` instead makes the base
/// 8-aligned for free; safetensors pads its JSON header to a multiple of 8 and
/// emits tensors in descending dtype width, so every 8-byte tensor then lands
/// on its natural alignment and [`as_f64`] / [`as_i64`] / [`as_floats`] borrow
/// straight out of this buffer with no copy at all.
pub struct AlignedBytes {
    words: Vec<u64>,
    len: usize,
}

impl AlignedBytes {
    /// Read `path` in one `read_exact`.
    ///
    /// One sequential read beats `mmap` for a model of this size: there are no
    /// page faults to take, no `madvise` tuning, and — unlike a mapping — the
    /// bytes cannot change underfoot if the file is rewritten while a load is
    /// in flight.
    pub fn read(path: &Path) -> Result<Self, PersistError> {
        let mut file = fs::File::open(path).map_err(|e| PersistError::io(path, e))?;
        let len = file
            .metadata()
            .map_err(|e| PersistError::io(path, e))?
            .len();
        let len = usize::try_from(len).map_err(|_| {
            PersistError::io(
                path,
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "model file is larger than this platform's address space",
                ),
            )
        })?;

        let mut words = vec![0u64; len.div_ceil(size_of::<u64>())];
        {
            // u64 -> u8 is always a legal cast (alignment only ever relaxes),
            // and the trailing padding past `len` stays zero.
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut words);
            file.read_exact(&mut bytes[..len])
                .map_err(|e| PersistError::io(path, e))?;
        }
        Ok(AlignedBytes { words, len })
    }

    /// The file's bytes, without the alignment padding.
    pub fn as_slice(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.words)[..self.len]
    }
}

/// A parsed, validated model file: the tensor index plus the `__metadata__`
/// map, both borrowing the caller's [`AlignedBytes`].
///
/// Borrowing rather than owning is what makes the load zero-copy end to end —
/// a [`TensorView`] handed out here points into the file buffer, and the
/// estimators pass that slice directly to
/// [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray::from_host).
pub struct ModelFile<'a, C: Container> {
    tensors: SafeTensors<'a>,
    meta: BTreeMap<String, String>,
    _container: PhantomData<fn() -> C>,
}

impl<'a, C: Container> ModelFile<'a, C> {
    /// Parse `buffer` and check it is an `estimator` model this build can read.
    ///
    /// The three discriminators are validated BEFORE any tensor is touched, so
    /// loading a `MultinomialNB` file into a `GaussianNB` reports exactly that
    /// rather than a missing-`theta_` error that reads like corruption.
    pub fn parse(buffer: &'a AlignedBytes, estimator: &'static str) -> Result<Self, PersistError> {
        let bytes = buffer.as_slice();
        // `read_metadata` re-parses the header, which `deserialize` also does;
        // the header is a few hundred bytes and this is the only way to reach
        // `__metadata__` (SafeTensors exposes tensors but not the free-form map).
        let (_, header): (usize, Metadata) = SafeTensors::read_metadata(bytes)?;
        let meta = header.metadata().clone().unwrap_or_default();

        let format = meta.get(KEY_FORMAT).map(String::as_str).unwrap_or("");
        let version = meta.get(KEY_VERSION).map(String::as_str).unwrap_or("");
        if format != C::FORMAT || version != C::VERSION {
            return Err(PersistError::NotAnMlrsModel {
                expected: C::FORMAT,
                version: C::VERSION,
                found: format!("{format}' v'{version}"),
            });
        }
        let found = meta.get(KEY_ESTIMATOR).map(String::as_str).unwrap_or("");
        if found != estimator {
            return Err(PersistError::WrongEstimator {
                expected: estimator,
                found: found.to_string(),
            });
        }

        Ok(ModelFile {
            tensors: SafeTensors::deserialize(bytes)?,
            meta,
            _container: PhantomData,
        })
    }

    /// The whole `__metadata__` map, including the `format` / `version` /
    /// `estimator` discriminators.
    ///
    /// For inspection and tooling — listing what a file declares, or comparing
    /// two files' scalars — rather than for the load path, which reaches each
    /// key through the typed `scalar_*` accessors so a missing or malformed
    /// entry becomes a [`PersistError::BadMetadata`] naming the key.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.meta
    }

    /// Fetch a REQUIRED tensor.
    pub fn tensor(&self, name: &'static str) -> Result<TensorView<'a>, PersistError> {
        self.tensors
            .tensor(name)
            .map_err(|_| PersistError::MissingTensor { tensor: name })
    }

    /// Fetch an OPTIONAL tensor — `None` when the header has no such entry.
    pub fn tensor_opt(&self, name: &str) -> Option<TensorView<'a>> {
        self.tensors.tensor(name).ok()
    }

    /// Read a required `f64` scalar out of `__metadata__`.
    pub fn scalar_f64(&self, key: &'static str) -> Result<f64, PersistError> {
        self.meta
            .get(key)
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or(PersistError::BadMetadata { key })
    }

    /// Read a required `usize` scalar.
    pub fn scalar_usize(&self, key: &'static str) -> Result<usize, PersistError> {
        self.meta
            .get(key)
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or(PersistError::BadMetadata { key })
    }

    /// Read a required `bool` scalar.
    pub fn scalar_bool(&self, key: &'static str) -> Result<bool, PersistError> {
        self.meta
            .get(key)
            .and_then(|s| s.parse::<bool>().ok())
            .ok_or(PersistError::BadMetadata { key })
    }

    /// Read a required string scalar.
    pub fn scalar_str(&self, key: &'static str) -> Result<&str, PersistError> {
        self.meta
            .get(key)
            .map(String::as_str)
            .ok_or(PersistError::BadMetadata { key })
    }

    /// Read an OPTIONAL `f64` scalar: `Ok(None)` when the key is absent,
    /// `Err` when it is present but unparsable (a truncated value is a corrupt
    /// file, not a `None`).
    pub fn scalar_opt_f64(&self, key: &'static str) -> Result<Option<f64>, PersistError> {
        self.scalar_opt(key)
    }

    /// Read an OPTIONAL `usize` scalar, mirroring
    /// [`ModelFile::scalar_opt_f64`].
    pub fn scalar_opt_usize(&self, key: &'static str) -> Result<Option<usize>, PersistError> {
        self.scalar_opt(key)
    }

    /// Read an OPTIONAL `u64` scalar, mirroring [`ModelFile::scalar_opt_f64`].
    pub fn scalar_opt_u64(&self, key: &'static str) -> Result<Option<u64>, PersistError> {
        self.scalar_opt(key)
    }

    /// The shared body of the `scalar_opt_*` readers: absent is `None`, present
    /// but unparsable is an error. Kept as one generic function so the three
    /// cannot drift on that distinction — it is the whole contract.
    fn scalar_opt<T: std::str::FromStr>(
        &self,
        key: &'static str,
    ) -> Result<Option<T>, PersistError> {
        match self.meta.get(key) {
            None => Ok(None),
            Some(s) => s
                .parse::<T>()
                .map(Some)
                .map_err(|_| PersistError::BadMetadata { key }),
        }
    }
}

// ---------------------------------------------------------------------------
// Typed tensor readers
// ---------------------------------------------------------------------------

/// The safetensors float tag matching `F`'s width.
fn float_dtype<F: Pod>() -> Result<Dtype, PersistError> {
    match size_of::<F>() {
        4 => Ok(Dtype::F32),
        8 => Ok(Dtype::F64),
        width => Err(PersistError::UnsupportedFloatWidth { width }),
    }
}

/// Reinterpret raw bytes as `&[T]`, copying ONLY if forced to.
///
/// The [`AlignedBytes`] buffer makes the borrowed arm the one that fires in
/// practice. The owned arm exists for files this crate did not write: nothing
/// in the format *guarantees* a given tensor's offset is a multiple of its
/// element width, so a misaligned payload must still load — just slower —
/// rather than panic inside `bytemuck`. `pod_read_unaligned` is the
/// alignment-agnostic read, and unlike `pod_collect_to_vec` it needs no extra
/// bytemuck feature.
fn cast_bytes<T: Pod>(bytes: &[u8]) -> Cow<'_, [T]> {
    match bytemuck::try_cast_slice::<u8, T>(bytes) {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(
            bytes
                .chunks_exact(size_of::<T>())
                .map(bytemuck::pod_read_unaligned::<T>)
                .collect(),
        ),
    }
}

/// Read an `F64` tensor.
pub fn as_f64<'a>(
    view: &TensorView<'a>,
    tensor: &'static str,
) -> Result<Cow<'a, [f64]>, PersistError> {
    expect_dtype(view, tensor, Dtype::F64)?;
    Ok(cast_bytes(view.data()))
}

/// Read an `I64` tensor (`classes_`).
pub fn as_i64<'a>(
    view: &TensorView<'a>,
    tensor: &'static str,
) -> Result<Cow<'a, [i64]>, PersistError> {
    expect_dtype(view, tensor, Dtype::I64)?;
    Ok(cast_bytes(view.data()))
}

/// Read a `U64` tensor into a `Vec<usize>`.
///
/// Returns owned rather than borrowed because the on-disk element is `u64` by
/// construction (see [`TensorRef::u64s`]) while the estimators hold `usize`;
/// the two coincide on a 64-bit host but the conversion is where a 32-bit host
/// would legitimately fail, so it is explicit and checked.
pub fn as_usizes(view: &TensorView<'_>, tensor: &'static str) -> Result<Vec<usize>, PersistError> {
    expect_dtype(view, tensor, Dtype::U64)?;
    cast_bytes::<u64>(view.data())
        .iter()
        .map(|&v| {
            usize::try_from(v).map_err(|_| PersistError::InconsistentGeometry {
                reason: format!("tensor '{tensor}' holds {v}, too large for this platform's usize"),
            })
        })
        .collect()
}

/// Read a float tensor as `&[F]`, converting only across a width change.
///
/// The matching arm — an `f32` file loading into an `f32` model — borrows the
/// file buffer outright, so the values reach the device without ever being
/// touched element-wise. The crossing arms exist because the dtype tag makes
/// them safe and useful: a model fitted on a GPU in `f32` loads into an `f64`
/// estimator for a higher-precision evaluation, and vice versa. The conversion
/// goes through [`host_to_f64`](mlrs_core::host_to_f64) /
/// [`f64_to_host`](mlrs_core::f64_to_host) — the crate's only sanctioned way to
/// bridge a generic `F`, since `as` casts are unavailable under a bare `Pod`
/// bound.
pub fn as_floats<'a, F: Pod>(
    view: &TensorView<'a>,
    tensor: &'static str,
) -> Result<Cow<'a, [F]>, PersistError> {
    let want = float_dtype::<F>()?;
    let found = view.dtype();
    if found == want {
        return Ok(cast_bytes(view.data()));
    }
    let widened: Vec<F> = match found {
        Dtype::F32 => cast_bytes::<f32>(view.data())
            .iter()
            .map(|&v| mlrs_core::f64_to_host::<F>(v as f64))
            .collect(),
        Dtype::F64 => cast_bytes::<f64>(view.data())
            .iter()
            .map(|&v| mlrs_core::f64_to_host::<F>(v))
            .collect(),
        other => {
            return Err(PersistError::DtypeMismatch {
                tensor,
                expected: want,
                found: other,
            })
        }
    };
    Ok(Cow::Owned(widened))
}

fn expect_dtype(
    view: &TensorView<'_>,
    tensor: &'static str,
    expected: Dtype,
) -> Result<(), PersistError> {
    if view.dtype() == expected {
        Ok(())
    } else {
        Err(PersistError::DtypeMismatch {
            tensor,
            expected,
            found: view.dtype(),
        })
    }
}

// ---------------------------------------------------------------------------
// Geometry validation
// ---------------------------------------------------------------------------

/// Require a rank-2 tensor and return its `(rows, cols)`.
///
/// The shapes ARE the schema — `n_classes` / `n_targets` / `n_features` are read
/// off the fitted matrix rather than stored separately (decision 2 in the module
/// docs), so this is the function that recovers them.
pub fn shape_2d(
    view: &TensorView<'_>,
    tensor: &'static str,
) -> Result<(usize, usize), PersistError> {
    match view.shape() {
        [rows, cols] => Ok((*rows, *cols)),
        other => Err(PersistError::InconsistentGeometry {
            reason: format!("tensor '{tensor}' must be 2-D, header declares shape {other:?}"),
        }),
    }
}

/// Require a rank-1 tensor and return its length.
pub fn shape_1d(view: &TensorView<'_>, tensor: &'static str) -> Result<usize, PersistError> {
    match view.shape() {
        [len] => Ok(*len),
        other => Err(PersistError::InconsistentGeometry {
            reason: format!("tensor '{tensor}' must be 1-D, header declares shape {other:?}"),
        }),
    }
}

/// Assert an extent read off one tensor equals the value every other tensor
/// implies. Every cross-tensor invariant a `load` depends on funnels through
/// here, which is what turns a tampered header into a typed error instead of an
/// out-of-bounds index at predict time.
pub fn expect_len(
    tensor: &'static str,
    actual: usize,
    expected: usize,
    what: &'static str,
) -> Result<(), PersistError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PersistError::InconsistentGeometry {
            reason: format!(
                "tensor '{tensor}' has {actual} {what}, but the file implies {expected}"
            ),
        })
    }
}
