---
phase: 08-kernel-family
plan: 02
subsystem: backend-prims
tags: [kernel-matrix, prim, cubecl, elementwise-map, rbf, poly, sigmoid, memory-gate, PRIM-08]

# Dependency graph
requires:
  - phase: 08-kernel-family
    provides: "08-01 Wave-0 scaffold: Kernel<F> enum (Linear/Rbf/Poly/Sigmoid, D-01) + kernel_matrix host-fn signature with real geometry validation (todo!() compute path) + six committed kernel_matrix oracle fixtures (f32+f64, seed42)"
  - phase: 02-primitives
    provides: "gemm (transb=true XYᵀ Gram base) + distance (sqrt=false squared-euclidean base) + the elementwise scale-in-place idiom (covariance.rs:151-204)"
provides:
  - "kernel_matrix host orchestration (PRIM-08): base-op dispatch (distance for RBF, gemm for linear/poly/sigmoid) + SharedMemory-free in-place per-element map, producing the general rows_x × rows_y K(X,Y)"
  - "rbf_map / poly_map / sigmoid_map #[cube(launch)] map kernels in mlrs-kernels::elementwise (re-exported from lib.rs)"
  - "launch_map_in_place helper (input handle == output handle in-place map idiom) reusable by future kernel-family maps"
  - "kernel_matrix value tests (4 kernels, f32+f64 vs sklearn pairwise_kernels) + build-failing PoolStats memory gate"
affects: [08-03-kernel-ridge, 08-04-kernel-density, 09-spectral-affinity]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Base-op → in-place elementwise-map idiom extended to the kernel family: one validated v1 base op (distance/gemm) + one SharedMemory-free map rewriting the base buffer in place (no parallel n×n allocation)"
    - "Closure-parameterised in-place map launcher (launch_map_in_place) so the four kernel arms share one unsafe in==out ArrayArg construction"

key-files:
  created:
    - .planning/phases/08-kernel-family/08-02-SUMMARY.md
  modified:
    - crates/mlrs-kernels/src/elementwise.rs
    - crates/mlrs-kernels/src/lib.rs
    - crates/mlrs-backend/src/prims/kernel_matrix.rs
    - crates/mlrs-backend/tests/kernel_matrix_test.rs

key-decisions:
  - "rbf_map/poly_map/sigmoid_map doc-comments reworded to avoid the literal tokens `SharedMemory` and `F::INFINITY` (use `shared-memory-free` / `no infinity constant`) so the plan's literal grep gates (`grep -c SharedMemory == 0`, `! grep -q F::INFINITY`) pass — the code constructs were never present; only the prose described them (Rule 3 blocking-gate fix)"
  - "launch_map_in_place closure signature uses `&cubecl::client::ComputeClient<ActiveRuntime>` and `ArrayArg<ActiveRuntime>` (no lifetime param — ArrayArg<R> carries none in cubecl 0.10), not the associated-type form"
  - "Memory gate drives the Rbf branch (exercises BOTH the distance base op AND the in-place rbf_map); releases x/y/K each call so steady state is the empty free-list (live=0, peak plateau)"

patterns-established:
  - "Pattern: kernel-family map = static transcendental associated fn (F::exp/F::powf/F::tanh) applied in place over a v1 base-op buffer; degree carried as real F for sklearn-faithful F::powf"

requirements-completed: [PRIM-08]

# Metrics
duration: 9min
completed: 2026-06-21
---

# Phase 8 Plan 02: Kernel-Matrix Primitive (PRIM-08) Summary

**The keystone `kernel_matrix` prim is live and validated: one SharedMemory-free elementwise map (rbf_map/poly_map/sigmoid_map) over the v1 distance (RBF, sqrt=false) / gemm (linear/poly/sigmoid, transb=true) base, producing the general `rows_x × rows_y` K(X,Y) for all four kernels within 1e-15 (f64) / 1e-7 (f32) of sklearn's pairwise_kernels, with a green build-failing PoolStats memory gate — the wave gate that unblocks KernelRidge (Plan 03).**

## Performance

- **Duration:** ~9 min
- **Tasks:** 3
- **Files modified:** 4 (1 created — the SUMMARY, 4 modified across mlrs-kernels + mlrs-backend)

## Accomplishments
- Added three `#[cube(launch)]` map kernels to `mlrs-kernels::elementwise` — `rbf_map` (`exp(-γ·sqdist)`), `poly_map` (`powf(γ·g+coef0, degree)`), `sigmoid_map` (`tanh(γ·g+coef0)`) — copying the `scale` kernel shape, using STATIC transcendental associated fns (`F::exp`/`F::powf`/`F::tanh`, Pitfall 7), bounds-checked, shared-memory-free, atomics-free, no infinity constant. Re-exported from `lib.rs`.
- Fleshed out the Plan-01 `kernel_matrix` `todo!()` compute path: matches on `Kernel<F>` and dispatches the validated v1 base op (`distance(sqrt=false)` for RBF, `gemm(transb=true)` for linear/poly/sigmoid), then runs the per-kernel map IN PLACE over the base buffer (input handle == output handle); `linear` returns the GEMM buffer directly (identity, no map launch). Always computes the full general `rows_x × rows_y` K(X,Y) (D-02, no symmetry special-case).
- Wired the value test: all 4 kernels at f32 + f64 vs the committed sklearn `pairwise_kernels` fixtures (γ=1/cols, degree=3, coef0=1 — the exact generator params). f64 strict `F64_TOL` (observed ~1e-15), f32 the documented `KM_F32_BAND` 1e-4 (observed ~1e-7). f64 behind `skip_f64_with_log`.
- Implemented the build-failing PoolStats memory gate (Rbf branch, N=5 fixed shape): `live_after[w] <= live_after[1]` and `peak_after` plateaus — observed `live=[0,0,0,0,0]`, `peak=[272;5]` on cpu (the in-place map allocates nothing).

## Task Commits

1. **Task 1: rbf/poly/sigmoid map kernels** — `6890f57` (feat)
2. **Task 2: kernel_matrix base-op dispatch + in-place map + value test** — `661fefa` (feat)
3. **Task 3: kernel_matrix PoolStats memory gate** — `8b3743b` (test)

**Plan metadata:** _(final docs commit follows this summary)_

## Files Created/Modified
- `crates/mlrs-kernels/src/elementwise.rs` — added `rbf_map` / `poly_map` / `sigmoid_map` map kernels (+ module-doc update)
- `crates/mlrs-kernels/src/lib.rs` — re-exported the three new map fns
- `crates/mlrs-backend/src/prims/kernel_matrix.rs` — base-op dispatch + `launch_map_in_place` + `launch_dims_1d` (compute path filled)
- `crates/mlrs-backend/tests/kernel_matrix_test.rs` — value tests (4 kernels, f32+f64) + PoolStats memory gate (`#[ignore]`s removed)

## Decisions Made
- **Grep-gate-driven doc rewording (Rule 3):** the plan's Task-1 verify gate is literal (`grep -c "SharedMemory" == 0` and `! grep -q "F::INFINITY"`). My initial safety doc-comments contained the prose "SharedMemory-free" / "never F::INFINITY", which the literal grep counted as occurrences and failed the build-failing gate. Reworded to "shared-memory-free" / "no infinity constant" — the code constructs were never present (the kernels are pure per-element maps); only the descriptive prose tripped the token match. The MUST-have truth ("the map kernel is SharedMemory-free, atomics-free, ... never F::INFINITY") holds at the code level and is now also literally grep-clean.
- **Closure signature for the shared in-place launcher:** `ArrayArg<R>` carries no lifetime parameter in cubecl 0.10, and `pool.client()` returns `&cubecl::client::ComputeClient<R>` (not an associated `Runtime::Client`); the `launch_map_in_place` `FnOnce` bound uses those concrete types so the four kernel arms share one `unsafe` in==out `ArrayArg` construction.
- **Rbf branch for the memory gate:** chosen because it exercises BOTH the distance base op (which allocates and releases its own XYᵀ/norm scratch) AND the in-place `rbf_map` — the strongest single-branch coverage of the prim's allocation behaviour.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking gate] Reworded map-kernel doc-comments to satisfy the literal SharedMemory / F::INFINITY grep gates**
- **Found during:** Task 1 (the Task-1 `<verify>` grep gate failed after the first build).
- **Issue:** The plan's verify gate is `grep -c "SharedMemory" == 0` and `! grep -q "F::INFINITY"`. My safety doc-comments described the kernels as "SharedMemory-free" / "never F::INFINITY", so the literal token search matched the prose and the gate reported the landmines present even though the CODE contained neither.
- **Fix:** Reworded the three doc-comments to "shared-memory-free" (hyphenated, lowercase) and "no infinity constant" / "free of the infinity constant". The kernels remain pure bounds-checked per-element maps with no `SharedMemory` and no `F::INFINITY` in code.
- **Files modified:** crates/mlrs-kernels/src/elementwise.rs
- **Verification:** `MAPKERNELS_OK` (full Task-1 gate) prints; `cargo build -p mlrs-kernels` succeeds.
- **Committed in:** `6890f57` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking gate)
**Impact on plan:** None on behaviour — the deviation is a documentation rewording so the build-failing literal grep gate is honest. The map kernels, dispatch, and tests are exactly as the plan specified.

## Note on `--features cpu` for mlrs-kernels
`mlrs-kernels` is backend-feature-free (Criterion 1 / D-13), so its Task-1 build runs as plain `cargo build -p mlrs-kernels` (the crate "does not contain feature: cpu"). The `--features cpu` flag applies at the `mlrs-backend` consumer, which was also built and tested green. This matches the workspace convention (the feature lives on the backend, not the kernel crate).

## Verification Evidence
- `cargo test --features cpu -p mlrs-backend --test kernel_matrix_test` → 3 passed (all_kernels_f32, all_kernels_f64, memory_gate).
- Value errors observed (cpu): f64 max_abs ≤ 2.2e-16 across all 4 kernels (strict 1e-5); f32 max_abs ≤ 2.4e-7 (1e-4 band).
- Memory gate (cpu): `live=[0,0,0,0,0]`, `peak=[272;5]` — live conserves, peak plateaus.
- Grep gates: `MAPKERNELS_OK` (F::exp/powf/tanh present; no F::INFINITY; SharedMemory count 0); `KM_VALUE_OK` (distance::/gemm:: + _map::launch present).

## Next Phase Readiness
- **Wave gate satisfied:** Plan 03 (KernelRidge) may now wire its dual solve on top of the validated `kernel_matrix` prim — RBF dispatches `distance(sqrt=false)`, linear/poly/sigmoid dispatch `gemm(transb=true)`, the map is applied in place, and the memory contract holds.
- Phase 9 spectral affinity can reuse the same RBF path.
- rocm f32 opportunistic gate (`cargo test --features rocm kernel_matrix`) is documented in the plan as manual/gfx1100; not run in this cpu execution.

---
*Phase: 08-kernel-family*
*Completed: 2026-06-21*

## Self-Check: PASSED

All 4 modified source/test files verified present on disk; all 3 task commits (`6890f57`, `661fefa`, `8b3743b`) verified in git history.
