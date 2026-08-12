//! `nb_persist` (NB-PERSIST, prototype) — the on-disk container the Naive Bayes
//! estimators save to and load from, built on [`safetensors`].
//!
//! This module owns the *format* only. It is estimator-agnostic and holds no
//! knowledge of any particular NB variant — each estimator implements its own
//! `save` / `load` against the [`NbWriter`] / [`NbFile`] pair in its own file,
//! which is what lets the fitted fields stay private and keeps the D-03 "five
//! mutually-independent structs, shared math as free functions" shape intact
//! (contrast a `Serialize` derive, which would have to make every fitted field
//! public or bolt a trait onto all five).
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
//! | tensors | every fitted **array** (`theta_`, `var_`, `feature_log_prob_`, `classes_`, `class_count_`, `class_log_prior_`) | 8 bytes per `f64`, 4 per `f32`, no decode step |
//! | `__metadata__` | every **scalar** (`alpha`, `var_smoothing`, `epsilon_`, `binarize`, the bool knobs) and the format/estimator tags | a scalar in its own tensor costs ~60 bytes of JSON header for 8 bytes of payload |
//!
//! Four decisions do the actual work of "small file, fast load":
//!
//! 1. **The stored dtype is the model's dtype.** A `GaussianNB<f32>` writes
//!    `F32` tensors, not widened `f64`. That is a straight 2× on the dominant
//!    term, and it costs nothing at load because the dtype tag makes the file
//!    self-describing — [`as_floats`] reinterprets when the tag matches `F` and
//!    converts only when it does not, so an f32 file still loads into a
//!    `GaussianNB<f64>` (train on a GPU in f32, evaluate in f64).
//! 2. **Nothing derivable is stored.** `n_features` and `n_classes` are read
//!    back off the tensor *shapes*; there is no redundant copy in the header to
//!    disagree with them. Same reason the ragged descriptor below is free.
//! 3. **Ragged state is stored CSR-style, never padded.** `CategoricalNB`'s
//!    `feature_log_prob_` is one `n_classes × n_categories_[j]` matrix *per
//!    feature*. Padding those to a dense `[n_features, n_classes, max_cat]`
//!    cube wastes `max_cat / mean_cat` of the file — a single 100-category
//!    column next to ninety-nine binary ones inflates it ~50×. Writing one
//!    tensor per feature instead trades that for `n_features` header entries
//!    (~60 bytes of JSON each). So the payload is stored **concatenated flat**,
//!    and the per-feature extents come from `n_categories_`, which is a fitted
//!    sklearn attribute that had to be in the file regardless. The ragged
//!    descriptor is therefore *free* — zero bytes beyond the model itself.
//!
//!    The estimator holds that same flat layout in memory, so there is no
//!    scatter/gather in either direction: a save hands the fitted buffer to the
//!    writer by reference and a load fills exactly one allocation.
//!    [`ragged_payload_len`] recomputes the length the extents imply — with
//!    checked arithmetic, since they come from an untrusted header — and
//!    [`ragged_block_offsets`] addresses individual blocks inside the buffer.
//! 4. **The read is one sequential `read_exact` into an 8-byte-aligned
//!    buffer** ([`AlignedBytes`]). Alignment is the whole trick: safetensors
//!    orders tensors by descending dtype width and pads the header to a
//!    multiple of 8, so in an 8-aligned buffer every `f64`/`i64`/`u64` tensor
//!    lands on its natural alignment and `bytemuck::try_cast_slice` succeeds —
//!    the file bytes become `&[f64]` with no copy and no parse, and go straight
//!    into [`DeviceArray::from_host`](mlrs_backend::device_array::DeviceArray).
//!    A plain `Vec<u8>` from `fs::read` is only 1-byte aligned and would force
//!    a copy of every tensor. [`as_f64`] and friends still fall back to an
//!    unaligned element-wise read rather than failing, because a file written
//!    by some other tool carries no such guarantee.
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
//! ## Reproducibility — a saved model is a deterministic function of itself
//!
//! Saving the SAME model twice produces byte-identical files, so a model file
//! can be content-addressed, deduplicated, and byte-diffed. `NbWriter` keeps
//! its scalars in a [`BTreeMap`] and safetensors emits tensor entries through
//! an index map, so both halves of the header are ordered.
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
//! `saving_twice_produces_an_identical_model` in the test suite is the gate,
//! and it compares raw file bytes — if the patch is ever dropped, that test
//! fails rather than the property silently regressing.
//!
//! ## What this is not
//!
//! Reading the whole file into memory is right for an NB model (the payload is
//! `n_classes × n_features` doubles — kilobytes to megabytes) and wrong for a
//! multi-gigabyte one. The swap is local: give [`AlignedBytes`] a `memmap2`
//! variant and [`NbFile::parse`] is unchanged, because it already takes a plain
//! `&[u8]` and every tensor accessor borrows from it.
//!
//! Tests live in `crates/mlrs-algos/tests/nb_persist_test.rs` (AGENTS.md §2).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
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
    "naive_bayes::nb_persist reinterprets little-endian safetensors payloads as native floats; \
     a big-endian target needs an explicit byte-swap path that does not exist yet"
);

/// `__metadata__` key holding the container discriminator, so a file that is
/// *not* an mlrs NB model is rejected by name rather than by a confusing
/// missing-tensor error.
const KEY_FORMAT: &str = "format";
/// `__metadata__` key holding [`FORMAT_VERSION`].
const KEY_VERSION: &str = "version";
/// `__metadata__` key holding the estimator discriminator (`"gaussian_nb"`, …),
/// which is what stops a `MultinomialNB` file loading as a `GaussianNB`.
const KEY_ESTIMATOR: &str = "estimator";

/// The value written under `format`. Present so a `.safetensors` file produced
/// by some other project fails the check at [`NbFile::parse`].
pub const FORMAT_ID: &str = "mlrs-nb";

/// The container version. Bump on any layout change that an older reader would
/// mis-read; [`NbFile::parse`] rejects anything else outright (a prototype has
/// no back-compat obligations, so this is a hard equality, not a range).
pub const FORMAT_VERSION: &str = "1";

/// The `param:` prefix marks a *constructor hyperparameter* as opposed to a
/// fitted attribute. Bare tensor/metadata names are exactly sklearn's fitted
/// attribute names (`theta_`, `class_count_`, …), so a `safetensors.numpy
/// .load_file(path)` in Python hands back a dict keyed the way the sklearn
/// estimator is, and the `param:`-prefixed entries are visibly the inputs.
///
/// Estimators spell their keys as literals (`"param:alpha"`) rather than
/// building them through a helper, because [`NbFile`]'s scalar readers take
/// `&'static str` so that [`PersistError::BadMetadata`] can name the key
/// without allocating.
pub const PARAM_PREFIX: &str = "param:";

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
    #[error("naive-bayes model I/O failed for '{path}': {source}")]
    Io {
        /// The file being read or written.
        path: PathBuf,
        /// The underlying `std::io` failure.
        source: std::io::Error,
    },

    /// The safetensors container itself is malformed (bad header length,
    /// non-UTF-8 or non-JSON header, offsets that do not cover the buffer).
    #[error("naive-bayes model file is not a valid safetensors container: {0}")]
    Container(#[from] SafeTensorError),

    /// The container parsed, but it is not an mlrs NB model (or is a version
    /// this build cannot read).
    #[error("expected an '{expected}' v{version} container, found '{found}'")]
    NotAnNbModel {
        /// Always [`FORMAT_ID`].
        expected: &'static str,
        /// Always [`FORMAT_VERSION`].
        version: &'static str,
        /// What the file actually declared.
        found: String,
    },

    /// The file holds a different NB estimator than the one being loaded.
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
/// One signature for all five variants, so a caller can persist a model without
/// knowing which one it holds. That uniformity is the whole reason this is a
/// trait rather than five inherent methods: the alternative was `GaussianNB`
/// taking a [`BufferPool`] and `CategoricalNB` not, which reads as an arbitrary
/// inconsistency at every generic call site.
///
/// `pool` is present because four of the five keep fitted tables on the device
/// and must read them back. `CategoricalNB` holds none and ignores it — the
/// same shape [`Fit`](crate::typestate::Fit) already has, where every estimator
/// takes a pool because the surface is shared, not because each one launches a
/// kernel. Following the existing convention beats inventing a second one.
///
/// This does NOT conflict with the module's D-03 "no shared base struct" rule:
/// the five stay independent structs implementing a shared *behavior* trait,
/// exactly as they already do for `Fit` / `PredictLabels` / `PredictProba`.
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
/// use mlrs_algos::naive_bayes::nb_persist::LoadModel;
/// let clf: GaussianNB<f32, Fitted> = GaussianNB::load(&mut pool, path)?;
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
    /// `class_log_prior_`, the ragged `feature_log_prob_` payload).
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
    /// every f32-fitted model; going through the estimators' `f64` accessors
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
pub struct NbWriter<'a> {
    meta: BTreeMap<String, String>,
    tensors: Vec<(String, TensorRef<'a>)>,
}

impl<'a> NbWriter<'a> {
    /// Start a writer for `estimator`, seeding the three discriminator keys
    /// that [`NbFile::parse`] validates.
    pub fn new(estimator: &'static str) -> Self {
        let mut meta = BTreeMap::new();
        meta.insert(KEY_FORMAT.to_string(), FORMAT_ID.to_string());
        meta.insert(KEY_VERSION.to_string(), FORMAT_VERSION.to_string());
        meta.insert(KEY_ESTIMATOR.to_string(), estimator.to_string());
        NbWriter {
            meta,
            tensors: Vec::new(),
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

    /// Record a `bool` scalar (`fit_prior`, `force_alpha`, `norm`, …).
    pub fn scalar_bool(&mut self, key: &str, value: bool) {
        self.meta.insert(key.to_string(), value.to_string());
    }

    /// Record a string scalar — used for the enum-shaped knobs whose variant
    /// tag is text and whose payload, if any, rides in a companion tensor.
    pub fn scalar_str(&mut self, key: &str, value: &str) {
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

    /// Stage a tensor under `name`.
    pub fn tensor(&mut self, name: &str, tensor: TensorRef<'a>) {
        self.tensors.push((name.to_string(), tensor));
    }

    /// Stage an OPTIONAL tensor — absent means no header entry, mirroring
    /// [`NbWriter::scalar_opt_f64`].
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
pub struct NbFile<'a> {
    tensors: SafeTensors<'a>,
    meta: BTreeMap<String, String>,
}

impl<'a> NbFile<'a> {
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
        if format != FORMAT_ID || version != FORMAT_VERSION {
            return Err(PersistError::NotAnNbModel {
                expected: FORMAT_ID,
                version: FORMAT_VERSION,
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

        Ok(NbFile {
            tensors: SafeTensors::deserialize(bytes)?,
            meta,
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
        match self.meta.get(key) {
            None => Ok(None),
            Some(s) => s
                .parse::<f64>()
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
/// them safe and useful: a model fitted on a GPU in `f32` loads into a
/// `GaussianNB<f64>` for a higher-precision evaluation, and vice versa. The
/// conversion goes through [`host_to_f64`](mlrs_core::host_to_f64) /
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
/// The shapes ARE the schema — `n_classes` and `n_features` are read off
/// `theta_` rather than stored separately (decision 2 in the module docs), so
/// this is the function that recovers them.
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

// ---------------------------------------------------------------------------
// The shared discrete-NB core
// ---------------------------------------------------------------------------

/// The fitted state and hyperparameters that `MultinomialNB`, `BernoulliNB` and
/// `ComplementNB` all hold identically: one `n_classes × n_features` device
/// matrix, the per-class log-prior, the class labels, and the four
/// additive-smoothing knobs.
///
/// Staged BORROWING for the write side, so the estimators' host readbacks flow
/// into [`serialize_to_file`] without a second copy. Named fields rather than a
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
