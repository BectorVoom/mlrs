# Phase 1: Foundation — Oracle, Backend Abstraction, Arrow Bridge - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-11
**Phase:** 1-Foundation — Oracle, Backend Abstraction, Arrow Bridge
**Areas discussed:** Oracle mechanism, Device-array scope, Bridge reject behavior, f32 tolerance policy

---

## Oracle mechanism

### How Rust tests obtain scikit-learn reference values

| Option | Description | Selected |
|--------|-------------|----------|
| Pre-generated fixtures | Python script generates references offline into committed files; Rust loads them. No Python at test time, fully reproducible. | ✓ |
| Live Python at test time | Rust invokes Python/sklearn at runtime (subprocess/PyO3). Always in sync, but CI needs Python env, slower/less hermetic. | |
| You decide | Defer to researcher. | |

**User's choice:** Pre-generated fixtures.

### Fixture format

| Option | Description | Selected |
|--------|-------------|----------|
| NumPy .npz | Bundled named arrays per case; read in Rust via npy/npz crate. | ✓ |
| Arrow IPC/Parquet | Reuse the Arrow path; dogfoods interchange format; heavier reader. | |
| You decide | Defer to researcher. | |

**User's choice:** NumPy .npz.

### Regeneration / when it runs

| Option | Description | Selected |
|--------|-------------|----------|
| Committed blobs + checked-in script | Fixtures committed; `scripts/gen_oracle.py` regenerates on demand; CI runs Rust against committed files. | ✓ |
| Generated in CI step | Python generator runs in CI before Rust job; fixtures are build artifacts. | |
| You decide | Defer to researcher. | |

**User's choice:** Committed blobs + checked-in script.

**Notes:** Extends the project's "oracle runs in CI without a GPU" goal to "without a Python env at test time" — the test job is hermetic.

---

## Device-array scope

### How much of FOUND-05 lands in Phase 1

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal wrapper now | Thin device-array over CubeCL buffers; pooling deferred to Phase 2. | |
| Reuse/pool built in now | Buffer-reuse/pool layer (free-list/arena) from the start, per memory-first mandate. | ✓ |
| You decide | Defer to researcher. | |

**User's choice:** Reuse/pool built in now.

### How to verify reuse this early

| Option | Description | Selected |
|--------|-------------|----------|
| Pool-instrumentation test | Counters + hard asserts that allocs stay flat / reuses occur after warm-up, now. | |
| Counters now, asserts later | Build stats/counters API now, log only; hard reuse asserts deferred to Phase 2. | ✓ |
| You decide | Defer to researcher. | |

**User's choice:** Counters now, asserts later.

**Notes:** Phase 1 only has the trivial smoke kernel, so a hard reuse gate now would test an artifact; real allocation patterns arrive with Phase 2 primitives.

---

## Bridge reject behavior

### Boundary policy for non-conforming Arrow input

| Option | Description | Selected |
|--------|-------------|----------|
| Hard reject only | Zero-copy is the only path; non-conforming input returns typed `BridgeError`; caller compacts. | ✓ |
| Reject + explicit copy path | Default rejects, but a separately named method offers a compacting copy. | |
| You decide | Defer to researcher. | |

**User's choice:** Hard reject only.

**Notes:** Question was first clarified by the user to mandate `thiserror` for the typed error enum (and `anyhow` at boundaries; all crates latest) — a project-wide convention captured as D-10. No copy escape hatch in Phase 1.

---

## f32 tolerance policy

### Policy shape

| Option | Description | Selected |
|--------|-------------|----------|
| Per-family table seeded now | Tolerance table keyed by estimator family, seeded with f32 defaults. | |
| Single global tolerance | One global f32/f64 tolerance now; split per-family later if needed. | ✓ |
| You decide | Defer to researcher. | |

**User's choice:** Single global tolerance (`F32_TOL`/`F64_TOL`, abs 1e-5, rel 1e-5).

### Comparison metric

| Option | Description | Selected |
|--------|-------------|----------|
| Combined abs OR rel (numpy-style) | Pass if `|got-exp| <= abs + rel*|exp|`. | |
| Both abs AND rel must pass | Stricter; requires both; can be brittle near zero. | ✓ |
| You decide | Defer to researcher. | |

**User's choice:** Both abs AND rel must pass.

**Notes:** Flagged to researcher/planner that "both must pass" is brittle for near-zero reference values; `assert_close` should include a near-zero abs-only guard. Captured as an implementation consideration on D-09, not a re-opened decision.

---

## Claude's Discretion

- Trivial smoke-test kernel choice (SAXPY / elementwise add, etc.).
- Specific npy/npz reader crate and Arrow crate (latest).
- `BridgeError` variant names; pool internal structure (free-list vs arena).
- f64 capability-gate skip-vs-xfail mechanics on wgpu lacking `SHADER_F64`.
- Near-zero guard floor value in `assert_close`.

## Deferred Ideas

- Per-estimator-family tolerance tables → Phase 3/4/5 when a family needs looser bounds.
- Compacting-copy Arrow ingest path → revisit Phase 6 (Python surface / PyCapsule).
- Hard buffer-reuse assertions → Phase 2, gated on realistic allocation patterns.
