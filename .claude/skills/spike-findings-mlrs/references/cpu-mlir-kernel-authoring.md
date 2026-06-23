# cpu-MLIR-Safe CubeCL Kernel Authoring

Cross-cutting rules for writing `#[cube(launch)]` kernels that lower correctly under
`cubecl-cpu` (the MLIR backend, the project's f64 correctness gate). Distilled from spikes 001
+ 002, which hit one loud lowering failure and one **silent miscompile** before passing. These
apply to ANY new mlrs kernel, not just KNN.

## Requirements

- Every kernel generic over `F: Float + CubeElement`; validate f64 (the cpu gate) AND f32.
- Stay inside the proven op-set below. Outside it, cpu-MLIR fails loudly (pass failure) OR —
  worse — silently miscompiles (compiles, launches, returns plausible wrong data).

## How to Build It — the proven op-set

**Launch / index builtins (type matters):**
- `ABSOLUTE_POS` is `usize` (use it bare as an array index, like `rbf_map`).
- `ABSOLUTE_POS_X` / `CUBE_POS_X` / `UNIT_POS_X` are `u32`.
- For a **per-row, one-selecting-unit kernel**, use the `top_k` shape — NOT a bare-`ABSOLUTE_POS`
  1D launch:
  ```rust
  let row = CUBE_POS_X;                  // u32
  if row < rows {
      if UNIT_POS_X == 0u32 {
          // … per-row work …
      }
  }
  ```
  Launch: `CubeCount::Static(n, 1, 1)`, `CubeDim {x:1,y:1,z:1}`.
- For a **per-element 2D kernel** (e.g. pairwise distance), use `ABSOLUTE_POS_X` / `ABSOLUTE_POS_Y`
  with `CubeDim {x:16,y:16}` and ceiling-div counts; guard `if i < rows { if j < cols { … } }`.

**Loops & accumulation (allowed):**
- Runtime `while c < n { … c += 1u32 }` loops; nested loops are fine.
- `u32` / `F` accumulators with `if`-guarded updates, read **within the same outer iteration**
  (incl. via a nested inner loop) — the `top_k` shape.
- Statement-form `if` for running max / conditional assignment:
  `let mut v = …; if cond { v = … }` (NEVER an `if`-expression in value position).

**Math (allowed, launch-proven):**
- `.abs()` instance form (jacobi-proven). Static transcendentals `F::powf` / `F::exp` /
  `F::tanh` / `.sqrt()` — a bounded feature-loop accumulator + `F::powf` lowers fine (Spike 001).
  **Never** the instance `x.powf()` form (can mis-lower in the `#[cube]` IR).
- Seeds via `F::from_int(0i64)` / `F::new(1.0)`. `as usize` cast only at the array-index boundary;
  keep `u32` for counters / scalar params.

**Host launch idiom (cubecl 0.10):**
- Upload `client.create(cubecl::bytes::Bytes::from_elems(vec))`; output `client.empty(n*size_of::<F>())`;
  read `client.read_one(handle)` + `bytemuck::cast_slice`.
- Args `unsafe { ArrayArg::from_raw_parts(handle, len) }` — **2-arg by-value** (consumes handle;
  clone before if you also read it back). Scalars **by value** (no `ScalarArg`). Launch
  `kernel::launch::<F, ActiveRuntime>(&client, count, dim, …)`.

## What to Avoid (the landmines)

- **Bare-`ABSOLUTE_POS` 1D launch for a per-row loop kernel** → MLIR pass failure
  `"operation with block successors must terminate its parent block"`, kernel never runs
  (output reads back as zeros). **FINDING 002-A.** Use the `CUBE_POS_X` shape above.
- **Cross-sibling-loop mutable accumulator** — a flag/counter written in one `while` and read in
  a SEPARATE sibling `while` — **SILENTLY MISCOMPILES** (the value never updates). **FINDING
  002-B**, and documented in `top_k`'s own source. Recompute per-row positional values with a
  self-contained nested accumulate inside the consuming loop instead. Example (self-shift count):
  ```rust
  let mut s = 0u32;
  while s < k {
      let mut bump = 0u32;            // init INSIDE the consuming loop
      let mut c = 0u32;
      while c < s + 1u32 {            // nested, read in the SAME outer iteration
          if in_idx[(ibase + c) as usize] == row { bump += 1u32; }
          c += 1u32;
      }
      let src = s + bump;            // no carry across sibling loops
      // …
      s += 1u32;
  }
  ```
- **Banned entirely (panic at launch, project memory):** `SharedMemory`, `Atomic`, `F::INFINITY`,
  mutable-`bool` scans, descending-shift loops.

## Constraints

- These are cpu-MLIR (`cubecl-cpu` 0.10) constraints. wgpu/cuda/rocm lower more permissively, but
  cpu is the f64 gate, so author to cpu's rules.
- **Verification discipline:** validate against an in-test host oracle and assert VALUES, not just
  non-panic. Always include a **duplicate-point row** (two samples at distance 0) — it both
  distinguishes index-identity logic and catches silent miscompiles a happy-path check ships.

## Origin

Synthesized from spikes: 001 (powf/abs/feature-loop op-set), 002 (002-A launch-shape failure +
002-B silent cross-loop miscompile). Source: `sources/001-*/`, `sources/002-*/`.
