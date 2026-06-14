---
phase: 01-foundation-oracle-backend-abstraction-arrow-bridge
plan: 03
subsystem: backend
tags: [arrow, bridge, hard-reject, validation, bytemuck, cubecl, capability, f64, skip-with-log, security, ASVS-V5]

# Dependency graph
requires:
  - phase: 01-foundation-oracle-backend-abstraction-arrow-bridge (Plan 01)
    provides: "5-crate workspace; runtime::{Client, active_client}; capability spike facade; SPIKE-FINDINGS A1/A2/A3/A7"
  - phase: 01-foundation-oracle-backend-abstraction-arrow-bridge (Plan 02)
    provides: "mlrs_core::error::BridgeError (typed Arrow-violation enum, D-07)"
provides:
  - "mlrs_backend::bridge::{validate_f32, validate_f64} — hard-reject Arrow ingress returning Result<&[F], BridgeError>; rejects sliced/offset, nullable, misaligned BEFORE any transmute (D-06 / FOUND-06)"
  - "mlrs_backend::bridge::cast_validated<F: Pod> — alignment/size-checked &[u8] -> &[F] mapping bytemuck Err -> BridgeError::Misaligned (A7)"
  - "mlrs_backend::bridge::upload<F> — validated single-upload into a CubeCL device buffer (honest A3 semantics; one host copy, not literal zero-copy)"
  - "mlrs_backend::capability::{log_oracle_dtype, active_backend_name, skip_f64_with_log} — Criterion-4 dtype/backend logging + f64 skip-with-log gate (FOUND-04 / T-03-04)"
affects: [01-05-oracle-integration, 01-04-device-array, phase-03, phase-04, phase-05]

# Tech tracking
tech-stack:
  added: []  # all deps already declared in Plan 01 workspace; consumes arrow, bytemuck, cubecl, log, env_logger
  patterns:
    - "validate-before-unsafe: offset -> nulls -> alignment, every reject returns before any transmute (D-06, ASVS V5)"
    - "buffer-level slice detection: arrow 59 PrimitiveArray::offset() is always 0; detect a slice via ScalarBuffer::inner().ptr_offset() + inner.len() vs values.len()*size_of"
    - "bytemuck::try_cast_slice -> recoverable Err -> typed BridgeError::Misaligned (A7, no panic, no manual ptr%align)"
    - "honest 'validated single-upload' (one host copy) — NOT overclaimed as zero-copy (A3)"
    - "logged early-return skip as the f64 skip/xfail mechanism (skip, not fail)"
    - "in-memory log::Log capture in tests to assert log lines deterministically"

key-files:
  created:
    - crates/mlrs-backend/tests/bridge_test.rs
    - crates/mlrs-backend/tests/capability_test.rs
  modified:
    - crates/mlrs-backend/src/bridge.rs
    - crates/mlrs-backend/src/capability.rs

decisions:
  - "Detect a sliced/offset Arrow array at the BUFFER level (ScalarBuffer ptr_offset + inner length), because arrow 59 PrimitiveArray::offset() always returns 0 (slicing rebases into the ScalarBuffer). The logical-offset check alone would never fire."
  - "No `unsafe` block exists in bridge.rs: bytemuck::try_cast_slice performs the only reinterpretation and is safe. The '// SAFETY: on every unsafe' criterion is vacuously satisfied (stronger: zero unreviewed unsafe on the ingress path)."
  - "upload() documented as 'validated single-upload' (one host copy via Bytes::from_bytes_vec), honoring A3 — we do NOT claim literal host zero-copy."
  - "Skip mechanism = logged early-return (skip_f64_with_log returns true + log::warn!), chosen over xfail per CONTEXT discretion. On this env's wgpu adapter (RADV GFX1152, SHADER_F64) it returns false and f64 runs."

metrics:
  duration_min: 7
  completed: "2026-06-11"
  tasks: 2
  files_changed: 4
---

# Phase 1 Plan 03: Arrow Hard-Reject Bridge + f64 Capability Gate Summary

Hard-reject Arrow→CubeCL ingress (offset/nulls/misalignment → typed `BridgeError` before any transmute, ASVS V5) plus an f64 capability gate with dtype×backend logging and logged skip-with-reason — both passing on cpu and wgpu.

## What Was Built

### Task 1 — Arrow zero-copy bridge with hard-reject validation (TDD)
`mlrs_backend::bridge` is the single ingress for host data into device buffers and the phase's primary threat surface (D-06 / FOUND-06):

- `validate_f32(&Float32Array) -> Result<&[f32], BridgeError>` and `validate_f64(&Float64Array) -> Result<&[f64], BridgeError>`, both delegating to a generic `validate_primitive::<T>`.
- Validation order, every check BEFORE any reinterpretation:
  1. **offset/slice** → `BridgeError::Offset` (T-03-02). arrow 59's `PrimitiveArray::offset()` always returns 0 because slicing rebases the `ScalarBuffer`, so the slice is detected at the **buffer level**: a non-sliced array's `ScalarBuffer` covers its entire inner `Buffer` (`ptr_offset() == 0 && inner.len() == values.len() * size_of::<Native>()`). Any deviation = a view into a larger parent buffer → reject (no aliased-parent upload). Verified by a runtime probe: `slice(1,4)` reports `ptr_offset=8`.
  2. **nulls** → `BridgeError::HasNulls` (T-03-03), via `null_count()`/`nulls()`.
  3. **alignment/size** → `BridgeError::Misaligned` (T-03-01), via `cast_validated::<F>` which maps `bytemuck::try_cast_slice`'s recoverable `Err` (A7 — never panics) to the typed variant.
- `cast_validated<F: Pod>(&[u8]) -> Result<&[F], BridgeError>` is public so the misalignment class is testable directly against a deliberately misaligned `&[u8]` (a real arrow array is always element-aligned).
- `upload<F>(client, &[F]) -> cubecl::server::Handle` performs the **validated single-upload** (one host copy via `Bytes::from_bytes_vec`, honoring A3 — not literal zero-copy).

### Task 2 — f64 capability gate (FOUND-04)
Completed `mlrs_backend::capability` on top of the Plan 01 spike facade:

- `active_backend_name() -> &'static str` — compile-time backend name from the active Cargo feature.
- `log_oracle_dtype(dtype, backend, adapter)` — emits `oracle dtype=… backend=… adapter=…` at info level at oracle-test start (Criterion 4).
- `skip_f64_with_log() -> bool` — logged early-return gate: `true` + `log::warn!("skipping f64 oracle on {backend}: …")` when f64 is unsupported (skip, not fail — T-03-04); `false` (proceed) when supported. Returns `false` here (RADV GFX1152 reports `SHADER_F64`), so f64 runs.

## Tests

- `crates/mlrs-backend/tests/bridge_test.rs` (9 tests): positive f32/f64; sliced f32/f64 → `Offset`; nullable f32/f64 → `HasNulls`; misaligned f32/f64 via `cast_validated` → `Misaligned` (Err, not panic); aligned round-trip.
- `crates/mlrs-backend/tests/capability_test.rs` (3 tests): `feature_enabled(F64)` agrees with `supports_f64` without panic; `log_oracle_dtype` emits the dtype/backend/adapter info line (asserted via an in-memory `log::Log` capture); the f64-gated path skips-with-log or runs-with-dtype-line. The `oracle dtype=…` line is mirrored to stdout so it is visible under `--nocapture`.

All 12 pass on **both** `--features cpu` and `--features wgpu`; the Wave-0 spike suite (5/5) still passes (no regression). `bridge.rs`/`capability.rs` are clippy-clean.

## Deviations from Plan

### Auto-fixed / clarified

**1. [Rule 1 - Bug] Slice detection moved from logical offset to buffer offset**
- **Found during:** Task 1 (RED→GREEN). Initial impl used `arr.offset()` as the plan/`must_haves` literally suggested; the sliced-array tests failed because arrow 59's `PrimitiveArray::offset()` always returns 0 (slicing rebases the `ScalarBuffer`).
- **Fix:** detect via `values().inner().ptr_offset()` + inner-buffer length vs logical length; report the element offset. This is the only correct way to honor the "reject sliced/offset array" requirement in arrow 59. Confirmed with a live probe.
- **Files:** crates/mlrs-backend/src/bridge.rs
- **Commit:** 82142bb

**2. [Clarification - no unsafe needed] `// SAFETY:` criterion vacuously satisfied**
- `bytemuck::try_cast_slice` is the only reinterpretation and is itself safe, so `bridge.rs` contains **no `unsafe` block**. The acceptance criterion "every `unsafe` carries a `// SAFETY:` comment" holds trivially, and the stronger property (zero unreviewed unsafe on the ingress path) is achieved.

**3. [Clarification - upload semantics] honest single-upload, not zero-copy**
- Per A3, cubecl 0.10's `Bytes` constructors own their allocation; `upload()` is documented as "validated single-upload" (one host copy), not literal zero-copy. No overclaim in code or docs.

No architectural changes; no Rule 4 checkpoints; no new package installs (T-03-SC accept holds).

## Authentication Gates

None.

## Verification Results

| Check | Result |
|-------|--------|
| `cargo test -p mlrs-backend --features cpu --test bridge_test --test capability_test` | 12/12 pass |
| `cargo test -p mlrs-backend --features wgpu --test bridge_test --test capability_test` | 12/12 pass |
| `--features wgpu capability -- --nocapture` shows `oracle dtype=… backend=wgpu` | yes; f64 runs (SHADER_F64) |
| Every reject path precedes any transmute | yes (source order: offset → nulls → cast) |
| `grep -rn "mod tests" crates/mlrs-backend/src/` | empty |
| Wave-0 spike suite regression | 5/5 still pass |
| clippy (bridge.rs / capability.rs) | clean |

## Success Criteria

- [x] Bridge rejects sliced/nullable/misaligned input with typed `BridgeError` before any unsafe [Criterion 3 / FOUND-06 / ASVS V5]
- [x] Capability layer reports f64 via `feature_enabled(FloatKind::F64)`; f64 tests skip-with-log on no-SHADER_F64; dtype×backend logged [Criterion 4 / FOUND-04]

## Self-Check: PASSED

All 4 source/test files and the SUMMARY exist on disk; all three task commits (44c05c4 test, 82142bb bridge, d435652 capability) are present in git history.
