# Wave-0 Spike Findings — Resolved CubeCL 0.10 / Arrow / npz Symbols

**Resolved:** 2026-06-11 (Phase 01, Plan 01, Wave 0)
**Toolchain:** rustc 1.95.0 / cargo 1.95.0
**Pinned crate versions (Cargo.lock):** cubecl 0.10.0, arrow 59.0.0, bytemuck 1.25.0,
npyz 0.9.1, thiserror 2.0.18, anyhow 1.0.102, mimalloc 0.1.52, log 0.4, env_logger 0.11.

This document is the input contract for Plans 02/03/04/05. Every assumption A1–A7
from `01-RESEARCH.md` is resolved below against the **installed** crates (not training
data), with the exact symbol path and the live test that proves it.

> Live proof: `crates/mlrs-backend/tests/spike_test.rs` (5 tests). Run:
> `cargo test -p mlrs-backend --features cpu spike -- --nocapture`
> and `cargo test -p mlrs-backend --features wgpu spike -- --nocapture`.
> All 5 pass on **both** cpu and wgpu.

---

## Environment capability snapshot

| Backend | f32 | f64 | Notes |
|---------|-----|-----|-------|
| cpu (`CpuRuntime`) | ✅ | ✅ | `f64_supported=true` |
| wgpu (`WgpuRuntime`) | ✅ | ✅ | adapter **AMD Radeon RADV GFX1152** (Vulkan, Mesa 25.2.8); adapter feature set lists **`SHADER_F64`** |
| cuda | n/a | n/a | compile-only; not run (no CUDA toolkit; build succeeds host-side) |

**Consequence for Plan 05:** f64-on-wgpu oracle tests **RUN** in this environment (the
wgpu adapter reports `SHADER_F64`). The skip/xfail path is still required for adapters
that lack it, but it is not exercised here. Always log `dtype=… backend=…` so CI shows
which path ran (Criterion 4).

---

## A1 — f64 capability-query symbol  ✅ RESOLVED

**RESEARCH guess:** `client.properties().feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))`
**Reality (cubecl 0.10):** there is **NO** `feature_enabled` / `Feature` enum in this layout.
The query is `supports_type`:

```rust
use cubecl::ir::FloatKind;          // FloatKind lives in cubecl-ir, re-exported as cubecl::ir
use cubecl::Runtime;
use cubecl::client::ComputeClient;

pub fn supports_f64<R: Runtime>(client: &ComputeClient<R>) -> bool {
    client.properties().supports_type(FloatKind::F64)
}
```

- `DeviceProperties::supports_type(&self, ty: impl Into<Type>) -> bool`
  (`cubecl-ir-0.10.0/src/properties.rs:113`; delegates to `features.supports_type`).
- `FloatKind` → `ElemType` → `StorageType` → `Type` conversions exist
  (`storage_from_elem!` macro + blanket `impl<T: Into<StorageType>> From<T> for Type`,
  `cubecl-ir-0.10.0/src/type.rs:423,431`), so `FloatKind::F64` is accepted directly.
- `client.properties()` → `&DeviceProperties` (`cubecl-runtime-0.10.0/src/client.rs:819`).

**Facade exposed for downstream (Plan 03/05):** `mlrs_backend::capability`:
`supports_type<R>(client, FloatKind)`, `supports_f64<R>(client)`, and the active-runtime
`feature_enabled(FloatKind) -> bool`. `FloatKind` is re-exported from this module so call
sites never import `cubecl::ir` directly.

## A2 — wgpu f64 / `SHADER_F64` mapping  ✅ RESOLVED

`supports_type(FloatKind::F64)` returns `true` on the wgpu adapter here, and the adapter's
wgpu feature list includes `SHADER_F64`. So the capability query is the correct gate — no
need to query raw wgpu adapter features. **f64-on-wgpu works in this environment.**

## A3 — `cubecl::bytes::Bytes` constructor / zero-copy  ✅ RESOLVED (honest semantics)

Both 0.10 constructors consume an **owned** allocation:

```rust
cubecl::bytes::Bytes::from_elems(vec)          // owns a typed Vec<T>
cubecl::bytes::Bytes::from_bytes_vec(byte_vec) // owns a Vec<u8>
```

There is **no borrow/no-copy** constructor that takes an existing `&[u8]` without owning it.
The manuals' `slice.to_vec()` is therefore a genuine **host copy**.

**Decision for Plan 03 (D-06 honesty):** do **not** claim literal host zero-copy. The bridge
guarantees **validated single-upload** semantics: validate (offset/nulls/alignment) → one
upload copy into the device buffer, no extra host copies beyond that. Document this wording in
`bridge.rs`. Upload path: `client.create(Bytes::from_bytes_vec(bytemuck::cast_slice(slice).to_vec()))`.

## A4 — npz named-array reader (`by_name`)  ✅ RESOLVED — crate = **`npyz` 0.9.1**

`numpy` is **not installed** in this environment, so the fixture was generated **in pure Rust
via npyz's own writer** (no numpy needed) and read back — a full f32 + f64 round-trip.

Read API (the oracle loader in Plan 02 uses exactly this):
```rust
use npyz::npz::NpzArchive;
let mut npz = NpzArchive::new(reader)?;        // or NpzArchive::open(path)
let names = npz.array_names();                 // iterator of &str
let arr = npz.by_name("coef_")?.expect("present");
let v: Vec<f64> = arr.into_vec::<f64>()?;      // also Vec<f32>; arr.shape() for dims
```
Write API (for `scripts/gen_oracle.py` parity tests, optional): `npyz::npz::NpzWriter` +
`WriterBuilder` (`.array(name, opts).default_dtype().shape(&[n]).begin_nd().extend(iter)`).

**Crate decision:** `npyz` (chosen over `ndarray-npy` to avoid the heavy `ndarray` dependency;
reads into plain `Vec<T>`). Enabled via `features = ["npz"]`.

> ⚠ Plan 02 blocking checkpoint (per threat T-01-SC): confirm `npyz` legitimacy before its
> first real use in the oracle loader. It is first-party (ExpHP, fork of npy-rs) and is
> already exercised here, but the formal package-legitimacy gate still belongs in Plan 02.

## A5 — Rust seeded RNG  ✅ N/A in Wave 0

No Rust-side RNG is used. Oracle inputs are Python-`numpy.random.default_rng(seed)`-generated
and committed (`gen_oracle.py`, Plan 02). The saxpy smoke test uses deterministic
integer-valued inputs, no RNG. `rand` is not a dependency yet — add only if a non-oracle
fixture needs it (then pin the exact seeded API per RESEARCH Pitfall 7).

## A6 — `ComputeClient` generic signature  ✅ RESOLVED

cubecl 0.10 `ComputeClient` takes a **single** generic parameter:

```rust
pub struct ComputeClient<R: Runtime> { … }   // cubecl-runtime-0.10.0/src/client.rs:33
```

**NOT** the `<Server, Channel>` form from older examples (that fails: "expected 1 generic
argument but 2 were supplied"). Insulated once in `mlrs_backend::runtime`:

```rust
pub type Client = cubecl::client::ComputeClient<ActiveRuntime>;
pub fn active_client() -> Client { ActiveRuntime::client(&ActiveDevice::default()) }
```

Related launch API facts resolved while wiring the spike:
- Scalar kernel arg is passed **by value** in the generated `launch` fn (no `ScalarArg`
  wrapper from the prelude): `saxpy_kernel::launch::<f32, R>(&client, count, dim, a, x_arg, y_arg)`.
- `ArrayArg::from_raw_parts(handle: Handle, length: usize)` — **2 args, no turbofish**, takes
  the `Handle` **by value** (`cubecl-core-0.10.0/.../array/launch.rs:47`).
- Read-back: `client.read_one(handle: Handle) -> Result<Bytes, ServerError>`
  (`cubecl-runtime-0.10.0/src/client.rs:136`). Handles are cheap ref-counted clones — clone
  the output handle before the launch consumes one if you also need to read it.
- Kernel scalar generic needs `<F: Float + CubeElement>` (the `CubeElement` bound is required
  for the scalar arg to implement `LaunchArg`; matches the axpy / half-precision manuals).

## A7 — `bytemuck::try_cast_slice` Err-vs-panic  ✅ RESOLVED

`bytemuck::try_cast_slice::<T, U>(&[T]) -> Result<&[U], PodCastError>` returns a **recoverable
`Err`** on alignment/size violations (it does **not** panic). Proven by
`spike_try_cast_slice_is_recoverable`. So the Arrow bridge (Plan 03) can map the `Err` to a
typed `BridgeError::Misaligned` before any `unsafe` transmute (D-06/D-07), no manual
`ptr % align_of` check needed.

---

## Exactly-one-backend-feature contract (T-01-03)

`runtime.rs` resolves `ActiveRuntime`/`ActiveDevice` via mutually-exclusive `#[cfg(feature=…)]`
re-exports; `Client`/`active_client()` are `#[cfg(any(cpu,wgpu,cuda,rocm))]`. With **zero**
backend features the workspace fails to resolve `ActiveRuntime` (build error), enforcing
"select exactly one backend." `mlrs-kernels` stays backend-feature-free
(`cargo tree -p mlrs-kernels -e features` lists no `cubecl-{cpu,wgpu,cuda,rocm}`).

## Build / test matrix verified

| Command | Result |
|---------|--------|
| `cargo build --workspace --features cpu`  | ✅ exit 0 |
| `cargo build --workspace --features wgpu` | ✅ exit 0 |
| `cargo build --workspace --features cuda` | ✅ exit 0 (compile-only; host-side build succeeds without CUDA toolkit) |
| `cargo test -p mlrs-backend --features cpu spike` | ✅ 5/5 pass |
| `cargo test -p mlrs-backend --features wgpu spike` | ✅ 5/5 pass |
| `cargo tree -p mlrs-kernels -e features \| grep cubecl-{cpu,wgpu,cuda,rocm}` | ✅ empty (feature-free) |
| `grep -rn "mod tests" crates/*/src/` | ✅ empty (AGENTS.md separation) |
