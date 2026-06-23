# Spike Conventions

Patterns and stack choices established across mlrs spike sessions. New spikes follow these
unless the question requires otherwise.

## Stack

- **Language/runtime:** Rust + CubeCL 0.10, the project's own stack. Spikes that exercise device
  kernels run on `ActiveRuntime` with `--features cpu` (the cpu-MLIR gate; f64-capable).
- **No standalone spike crate.** A CubeCL `#[cube(launch)]` kernel can't launch from a
  `.planning/spikes/` cargo project without replicating the whole cubecl/runtime/feature setup.

## Structure (run vehicle)

- Each kernel-launching spike runs as a **self-contained temp integration test** in
  `crates/mlrs-backend/tests/knn_spike_NNN_test.rs` (precedent: the repo's own
  `tests/spike_test.rs`). Run targeted:
  `cargo test -p mlrs-backend --features cpu --test <file> -- --nocapture`.
- The kernel + harness source is **copied verbatim into `.planning/spikes/NNN-*/`** as the
  durable artifact; the temp test is deleted once findings are recorded (it is NOT the real prim).
- Keep datasets tiny (n≤6) and run **targeted** test names — the full `mlrs-backend` cpu suite is
  slow and a full `cargo test` can exhaust disk (project memory). Each spike test runs in <1s.

## Patterns (cpu-MLIR-safe kernel authoring — the hard-won core)

Stay inside this proven op-set and cpu-MLIR cooperates; step outside and it fails loudly OR
silently:

- **Launch/index builtins:** `ABSOLUTE_POS` is `usize` (use bare as an index); `ABSOLUTE_POS_X`
  / `CUBE_POS_X` / `UNIT_POS_X` are `u32`. For a per-row 1-unit kernel, use the `top_k` shape —
  `row = CUBE_POS_X` + `if row < rows { if UNIT_POS_X == 0u32 { … } }` — NOT a bare-`ABSOLUTE_POS`
  1D launch (the latter caused a `"operation with block successors must terminate its parent
  block"` MLIR pass failure; Spike 002-A).
- **NO cross-sibling-loop mutable accumulator.** A flag/counter written in one `while` and read
  in a SEPARATE sibling `while` silently miscompiles (the cube macro can't carry it; documented
  in `top_k`, confirmed by Spike 002-B). Compute per-row positional values with a self-contained
  nested accumulate read in the SAME outer iteration. Reading an accumulator within nested loops
  of one outer iteration IS fine (the `top_k` shape).
- **Banned (panic at launch, per project memory + spikes):** `SharedMemory`, `Atomic`,
  `F::INFINITY`, mutable `bool` scans, descending-shift loops.
- **Allowed (launch-proven):** runtime `while c < n { … c += 1u32 }` loops; `u32`/`F`
  accumulators with `if`-guarded updates; statement-form `if` for running max / conditional
  assignment (`let mut v = …; if cond { v = … }`); `.abs()` (instance form); static transcendentals
  `F::powf` / `F::exp` / `F::tanh` / `.sqrt()` (NEVER the `x.powf()` instance form); `F::from_int`,
  `F::new`. A bounded feature-dim accumulator loop + `F::powf` lowers fine (Spike 001).
- **Generics:** kernels generic over `F: Float + CubeElement` (validate f64 — the cpu gate — and
  f32). `as usize` cast at the array-index boundary; keep `u32` for loop counters / scalar params.

## Tools & Libraries (host-side launch idiom, cubecl 0.10)

- Upload: `client.create(cubecl::bytes::Bytes::from_elems(vec))`. Output buffer:
  `client.empty(n * size_of::<F>())`. Read back: `client.read_one(handle)` → `bytemuck::cast_slice`.
- Kernel args: `unsafe { ArrayArg::from_raw_parts(handle, len) }` — the **2-arg by-value** form
  (consumes the handle; clone first if you also read it back). Scalars passed **by value** (no
  `ScalarArg` wrapper). Launch: `kernel::launch::<F, ActiveRuntime>(&client, count, dim, …)`.
- Through the prim layer: `BufferPool::new(client)`, `DeviceArray::from_host(&mut pool, &vec)`,
  `dev.to_host(&pool)`, `dev.handle().clone()`; `distance::<F>(…)` and `top_k::<F>(…)` are pub and
  test-callable.

## Verification discipline

- Validate device output against an **in-test host oracle** (brute-force, same precision) with
  the prim's documented `(distance, index)` lowest-index tie-break — assert VALUES, not just
  non-panic.
- **Always include a duplicate-point row** (two samples at distance 0) in KNN/distance oracles.
  It is the case that distinguishes index-identity self-drop from first-zero-distance, and it
  caught a silent miscompile (002-B) that a happy-path check would have shipped.
