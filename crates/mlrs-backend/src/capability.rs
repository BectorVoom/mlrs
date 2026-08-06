//! Runtime capability gating (FOUND-04).
//!
//! Resolves the f64 capability query against cubecl 0.10 and exposes a stable
//! facade so downstream call sites (Plan 03, Plan 05) never re-discover the
//! symbol.
//!
//! ## A1 RESOLVED (capability-query symbol)
//! cubecl 0.10 has NO `feature_enabled(Feature::Type(Elem::Float(..)))` form —
//! the `Feature` enum from older examples does not exist in this layout. The
//! real query is:
//!
//! ```ignore
//! client.properties().supports_type(FloatKind::F64)
//! ```
//!
//! `DeviceProperties::supports_type` takes `impl Into<Type>`, and
//! `FloatKind -> ElemType -> StorageType -> Type` conversions are provided, so
//! `FloatKind::F64` is accepted directly.
//!
//! ## A2 RESOLVED (wgpu SHADER_F64)
//! On the wgpu adapter in this environment (AMD Radeon RADV GFX1152, Vulkan)
//! the adapter feature set includes `SHADER_F64`, and `supports_type` returns
//! `true` for f64 on wgpu here. f64 oracle tests therefore RUN (not skip) on
//! this machine; the skip path still exists for adapters lacking it.

pub use cubecl::ir::FloatKind;

use cubecl::Runtime;
use cubecl::client::ComputeClient;

/// Query whether the given client's backend supports a given float type.
///
/// Generic over the runtime so it works for any backend (cpu / wgpu / cuda /
/// rocm) without naming a concrete client type.
pub fn supports_type<R: Runtime>(client: &ComputeClient<R>, kind: FloatKind) -> bool {
    client.properties().supports_type(kind)
}

/// Convenience: does the client's backend support f64?
pub fn supports_f64<R: Runtime>(client: &ComputeClient<R>) -> bool {
    supports_type(client, FloatKind::F64)
}

/// Stable facade over the active runtime's f64 capability (FOUND-04 wording).
///
/// Constructs a client for the active runtime's default device and reports
/// whether the requested float type is supported. Downstream code (Plan 03's
/// skip/xfail gate, Plan 05's oracle dtype logging) calls this and never spells
/// out the underlying `supports_type` query.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn feature_enabled(kind: FloatKind) -> bool {
    let client = crate::runtime::active_client();
    client.properties().supports_type(kind)
}

/// Units a **cpu** launch should spread over = the machine's usable core count.
///
/// `cubecl-cpu` maps ONE OS THREAD PER UNIT and runs the cube grid as a serial
/// loop inside each of them (`cubecl_cpu::compute::runner::execute_data`), so a
/// launch's `cube_dim` is literally its thread count. The GPU-idiomatic 256-unit
/// block therefore spawns 256 threads and pays a 256-way join per kernel, which
/// for a small GATHER pass can exceed the work itself. Barrier-free kernels
/// indexed by `ABSOLUTE_POS_X` alone are free to take this instead: the split
/// between cube and unit is a pure scheduling choice with no effect on results.
///
/// `std::thread::available_parallelism()` reads cgroup limits from `/proc` and
/// costs hundreds of microseconds, so it is resolved ONCE per process — a
/// per-launch call showed up on the KNN predict hot path. `MLRS_CPU_UNITS`
/// overrides the value for on-target A/B and is re-read per call so a test can
/// sweep it within one process.
pub fn cpu_launch_units() -> u32 {
    use std::sync::OnceLock;

    if let Some(v) = std::env::var("MLRS_CPU_UNITS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v >= 1)
    {
        return v;
    }
    static DETECTED: OnceLock<u32> = OnceLock::new();
    *DETECTED.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|v| v.get() as u32)
            .unwrap_or(8)
            .max(1)
    })
}

/// Unit width a 1D `ABSOLUTE_POS_X`-indexed launch should use on the active
/// backend: [`cpu_launch_units`] on cpu, the 256-unit warp multiple elsewhere.
///
/// Only for kernels whose result is independent of the cube/unit split — i.e.
/// no `SharedMemory`, no `sync_cube`, no `CUBE_DIM`-dependent indexing.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn gather_launch_width() -> u32 {
    if active_backend_name() == "cpu" {
        cpu_launch_units()
    } else {
        256
    }
}

/// Query whether the given client's backend supports plane (subgroup) ops.
///
/// Mirrors [`supports_type`] but for the plane/subgroup capability. cubecl 0.10
/// exposes this via `client.features().plane` — an `EnumSet<Plane>` of the
/// supported plane operations. We report support when the basic plane-op set
/// (`Plane::Ops`) is present, which is the prerequisite for the reduction
/// plane-path (Plan 02). The plane width is separately available via
/// `client.properties().hardware.plane_size_{min,max}`.
///
/// A3 RESOLVED (subgroup-query symbol): the stable query is
/// `client.features().plane.contains(Plane::Ops)` (NOT a `feature_enabled`
/// form). Downstream plane-path code calls this facade and never re-discovers
/// the symbol.
pub fn supports_plane<R: Runtime>(client: &ComputeClient<R>) -> bool {
    use cubecl::ir::features::Plane;
    client.features().plane.contains(Plane::Ops)
}

/// Stable facade over the active runtime's plane/subgroup capability.
///
/// Constructs a client for the active runtime's default device and reports
/// whether plane (subgroup) ops are supported. Plan 02's reduction plane-path
/// skip calls this and never spells out the underlying `features().plane`
/// query.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn plane_supported() -> bool {
    supports_plane(&crate::runtime::active_client())
}

/// Active runtime's plane (subgroup) width, used to size the plane-path
/// reduction's per-(cube, plane) partial output (Plan 02).
///
/// Reports `client.properties().hardware.plane_size_max` — the upper plane
/// size the adapter advertises (CUDA warp = 32; wgpu subgroups vary 4..128).
/// When the adapter reports no plane support the value may be `0`; callers
/// clamp to at least `1`. The min/max symbols were pinned in Plan 02-01
/// (`spike_subgroup_query_reports_support`).
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn active_plane_width() -> u32 {
    crate::runtime::active_client()
        .properties()
        .hardware
        .plane_size_max
}

/// Does the active backend support f64 TRANSCENDENTALS (`exp` / `ln` / `tanh` /
/// `powf`) in device code — not merely f64 arithmetic?
///
/// [`feature_enabled`]`(FloatKind::F64)` reports whether the adapter accepts the
/// f64 TYPE (`SHADER_F64`: add / mul / fma / sqrt). That is a strictly weaker
/// property than being able to EVALUATE a transcendental at f64, and the two
/// come apart on wgpu:
///
/// ```text
/// ACO ERROR: Unimplemented NIR instr bit size: con 64   %271 = fexp2 %270
/// ACO ERROR: Unimplemented NIR instr bit size: con 64   %280 = flog2 %276
/// → SIGSEGV inside the shader compiler
/// ```
///
/// (measured on this environment's adapter, AMD RADV GFX1152, compiling the
/// softmax loss/grad kernel at f64). The failure is a SEGFAULT in the driver's
/// compiler, not a clean error return, so a kernel that reaches it takes the
/// whole process down — `crates/mlrs-backend/tests/lbfgs_test.rs` died with
/// `signal: 11` and `LogisticRegression`'s f64 oracle either crashed or reported
/// a spurious `NotConverged`, depending on what else shared the process.
///
/// This is not adapter-specific bad luck: WGSL has no `f64` type at all, so a
/// wgpu backend's f64 support is an extension whose transcendental coverage is
/// entirely up to the driver. `false` on wgpu is therefore the principled
/// answer, not a workaround for one GPU. cpu / cuda / rocm evaluate f64
/// transcendentals natively.
///
/// Callers whose kernel uses `exp` / `ln` / `tanh` / `powf` consult this and
/// take their host arm at f64 (see
/// [`prims::lbfgs::softmax_loss_grad`](crate::prims::lbfgs::softmax_loss_grad)),
/// exactly as the shared-memory-budget callers consult
/// [`active_max_shared_memory`]. Kernels doing only f64 ARITHMETIC are
/// unaffected and keep running on device.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub const fn f64_transcendental_supported() -> bool {
    !matches!(active_backend_name().as_bytes(), b"wgpu")
}

/// Does the active backend FLUSH SUBNORMAL floats to zero in device code?
///
/// True on wgpu. WGSL explicitly permits flush-to-zero for subnormals, and this
/// environment's adapter (AMD RADV/ACO) does flush them, so a subnormal that is
/// perfectly representable on the host becomes exactly `0.0` once it reaches a
/// shader. cpu / cuda / rocm preserve subnormals.
///
/// This matters wherever a value's meaning depends on a ONE-ULP distinction near
/// zero. The known case is `ForestInference`'s threshold import: sklearn routes
/// `x <= t → left` while the mlrs tree kernel routes `x < t → left`, so import
/// bumps every threshold to `next_up(t)`. For `t = 0.0` that bump is
/// `1.4e-45` — the smallest positive SUBNORMAL — which a flushing backend turns
/// straight back into `0.0`, silently undoing the bump and routing
/// `x == 0.0` RIGHT instead of LEFT (measured: `fil_test` got `proba[0] = 0.375`
/// where sklearn gives `0.875`).
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub const fn flushes_subnormals() -> bool {
    matches!(active_backend_name().as_bytes(), b"wgpu")
}

/// Reject, BEFORE launch, a kernel that needs f64 transcendentals on a backend
/// that has none — for kernels with no host arm to fall back to.
///
/// The alternative is not a failed launch but a SEGFAULT inside the driver's
/// shader compiler (see [`f64_transcendental_supported`]), which takes the
/// caller's whole process with it. Returning
/// [`mlrs_core::PrimError::UnsupportedCapability`] converts that into a
/// recoverable typed error.
///
/// Call this at the NARROWEST site — the specific metric / loss / kernel that
/// evaluates the transcendental, never a prim's entry point. A whole-prim guard
/// also rejects callers that never reach the transcendental (euclidean kNN,
/// squared-loss SGD), which is a functional regression, not a fix. A path that
/// CAN compute the same thing on the host should do that instead: see
/// `prims::eig`, `prims::lbfgs::softmax_loss_grad` and `prims::kernel_matrix`,
/// which stay fully functional at f64 on every backend.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn guard_f64_transcendental<F>(operand: &'static str) -> Result<(), mlrs_core::PrimError> {
    if std::mem::size_of::<F>() == 8 && !f64_transcendental_supported() {
        return Err(mlrs_core::PrimError::UnsupportedCapability {
            operand,
            capability: "f64 transcendentals (exp/log/powf/tanh) in device code",
        });
    }
    Ok(())
}

/// Test-side skip gate for the f64-transcendental gap — the sibling of
/// [`skip_f64_with_log`] for a backend that HAS f64 arithmetic but cannot
/// evaluate f64 transcendentals.
///
/// An f64 oracle whose kernel uses `exp`/`log`/`tanh`/`powf` calls this and
/// returns early on `true`, so the run is skipped with a logged reason instead
/// of crashing the test binary. Use it ONLY for primitives that have no host
/// arm; one that does (eig, softmax loss/grad) must keep running and be
/// asserted.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn skip_f64_transcendental_with_log() -> bool {
    if f64_transcendental_supported() {
        return false;
    }
    let backend = active_backend_name();
    log::warn!(
        "skipping f64 oracle on {backend}: the adapter accepts f64 but its shader \
         compiler has no 64-bit exp/log (see capability::f64_transcendental_supported)"
    );
    true
}

/// Active runtime's maximum per-cube shared-memory budget, in bytes.
///
/// Reports `client.properties().hardware.max_shared_memory_size`. Unlike CUDA
/// (a fixed 48 KiB+ budget) a wgpu adapter can advertise as little as the
/// WebGPU downlevel default (`16384` = 16 KiB), so a `SharedMemory` kernel sized
/// against the CUDA budget can exceed a wgpu adapter's limit and fail pipeline
/// creation. Callers that dispatch a shared kernel ONLY on wgpu (e.g.
/// `prims::linear_predict::use_shared_predict`) query this and fall back to
/// their GATHER path when their tile would not fit — so `predict` never fails
/// on a small-SLM adapter where the GATHER kernel would have worked.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn active_max_shared_memory() -> usize {
    crate::runtime::active_client()
        .properties()
        .hardware
        .max_shared_memory_size
}

/// Static name of the active backend, derived from the compiled-in Cargo
/// feature (FOUND-03: exactly one backend feature is active).
///
/// Used in the dtype×backend oracle log line (Criterion 4) so CI output shows
/// which backend a given oracle run executed on.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub const fn active_backend_name() -> &'static str {
    #[cfg(feature = "cpu")]
    {
        "cpu"
    }
    #[cfg(feature = "wgpu")]
    {
        "wgpu"
    }
    #[cfg(feature = "cuda")]
    {
        "cuda"
    }
    #[cfg(feature = "rocm")]
    {
        "rocm"
    }
}

/// Emit the canonical oracle dtype×backend log line at the start of an oracle
/// test (Criterion 4: "CI log shows which dtype ran on which backend").
///
/// Logs at `info` level. `adapter` is a free-form adapter/device descriptor
/// (e.g. the wgpu adapter name, or `"default"` for cpu) so the line is
/// self-describing in CI output.
pub fn log_oracle_dtype(dtype: FloatKind, backend: &str, adapter: &str) {
    log::info!("oracle dtype={dtype:?} backend={backend} adapter={adapter}");
}

/// f64 skip-with-log gate (FOUND-04, T-03-04). Returns `true` when the f64 path
/// should be **skipped** because the active backend lacks f64 support, after
/// logging the reason at `warn` level. Returns `false` when f64 is supported and
/// the caller should proceed.
///
/// This is the chosen skip/xfail mechanism (logged early-return — Claude's
/// discretion per CONTEXT D-06/FOUND-04): an f64-gated oracle test calls this
/// and `return`s early when it reports `true`, so the run is **skipped, not
/// failed**, and CI shows the logged reason. On this environment's wgpu adapter
/// (AMD RADV GFX1152, `SHADER_F64` present) it returns `false` and f64 runs.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn skip_f64_with_log() -> bool {
    if feature_enabled(FloatKind::F64) {
        return false;
    }
    let backend = active_backend_name();
    log::warn!("skipping f64 oracle on {backend}: SHADER_F64 / f64 unsupported on this adapter");
    true
}

/// Can the active backend RUN an `f64` compute kernel — as distinct from
/// [`feature_enabled(FloatKind::F64)`](feature_enabled), which asks whether it
/// ADVERTISES the type?
///
/// The two questions have the same answer everywhere except CUDA, and there the
/// difference is load-bearing enough to be worth its own predicate.
///
/// ## Why `supports_type` under-reports on cuda
/// `cubecl-cpp` — the C++ source generator shared by the cuda and rocm backends
/// — omits `f64` from the type registry it hands the runtime, with this comment
/// at `cubecl-cpp-0.10.0/src/shared/base.rs:2114`:
///
/// ```text
/// // Causes CUDA_ERROR_INVALID_VALUE for matmul, disabling until that can be
/// // investigated
/// //gpu::Elem::Float(gpu::FloatKind::F64),
/// ```
///
/// So the flag is `false` on every CUDA device, including a GP100 whose `f64`
/// throughput is HALF its `f32` rate — the omission is an upstream workaround
/// for one operation (matmul), not a statement about the hardware or about
/// NVRTC, which compiles `double` natively like any other CUDA C++ type.
///
/// Reading that flag as "this backend cannot do `f64`" is therefore wrong in a
/// costly direction: it silently disables every `f64` device arm on the one
/// class of accelerator with fast double precision.
///
/// ## What this predicate promises, and what the caller still owes
/// `true` means *plain* `f64` kernels — element maps, reductions, register- and
/// shared-memory tiles — compile and run. It does NOT extend to the operation
/// upstream actually disabled: a caller whose `f64` path can reach `cubek-matmul`
/// must keep it off that path itself. `prims::normal_eq` does exactly that by
/// requiring `gram::fused_centering_available`, which excludes `GramPath::Gemm`
/// — so its Gram, column sums and widening pass are all hand-written kernels
/// and none of them emits a matmul.
///
/// - **cuda** → `true`. The registry omission is the matmul workaround above.
/// - **wgpu** → the advertised flag, which there is a GENUINE capability: WGSL
///   has no `f64` and an adapter without the `SHADER_F64` feature cannot run one
///   at all.
/// - **rocm** → `true`, same as cuda. `cubecl-hip` shares the SAME
///   `cubecl-cpp` type registry cuda does, with the SAME matmul-workaround
///   comment covering both backends — so the a priori guess was that `rocm`
///   under-reports for the identical reason. Measured 2026-08-06 on real
///   ROCm hardware (a gfx1151 Radeon 860M APU):
///   `crates/mlrs-algos/tests/gaussian_mixture_device_test.rs` and
///   `bayesian_mixture_device_test.rs` — genuine non-matmul `f64` device
///   kernels (`logsumexp`, blocked reductions) — ran and reproduced the host
///   arm exactly once `MLRS_F64_DEVICE=1` bypassed the advertised-`false`
///   default, confirming the under-report. The full `mlrs-backend`/
///   `mlrs-algos` suites also passed with this hardcoded to `true`, which is
///   the empirical version of the "wrong answer costs throughput, not
///   correctness" property below actually being exercised end to end.
/// - **cpu** → the advertised flag; `cubecl-cpu`'s MLIR registry does list
///   `f64`.
///
/// `MLRS_F64_DEVICE=0`/`1` overrides the verdict, read through
/// [`crate::abflag`] so a test can scope it without an environment data race.
/// `0` is the escape hatch if a future CUDA/driver combination does reject a
/// plain `f64` kernel: it puts every caller back on its host arm rather than
/// failing.
///
/// The cuda verdict being a HARD-CODED `true` rather than a probe means this
/// predicate can be wrong on a driver/CTK combination nobody has run, so it
/// must not be a caller's only fallback — a wrong answer here has to cost
/// throughput, not the result. `BayesianRidge::fit_with_sample_weight` is the
/// worked example: it catches a failed device Gram and reads the design back,
/// which is exactly what it did before the arm existed. `MLRS_F64_DEVICE=0`
/// then skips the failed launch rather than being the only way to get a fit at
/// all.
#[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn f64_device_kernels_available() -> bool {
    if let Some(v) = crate::abflag::var("MLRS_F64_DEVICE") {
        return v != "0";
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        true
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        feature_enabled(FloatKind::F64)
    }
}
