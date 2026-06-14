---
phase: 02
slug: core-compute-primitives
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-12
---

# Phase 02 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Domain: numerical compute-kernel phase — no auth/session/network/PII surface. The
> dominant attack surface is the unsafe `ArrayArg::from_raw_parts` device-launch path
> (gated by validated `DeviceArray.len` + caller-geometry shape asserts) plus one
> third-party dependency (`cubek-matmul`, the GEMM substrate).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| host slice → device buffer | host `&[F]` uploaded via `DeviceArray::from_host`; `len` is the validated read-back source of truth (T-04-01, carried from Phase 1) | `f32`/`f64` numeric arrays (no PII) |
| caller `(rows, cols)` geometry → kernel index math | caller-supplied shapes must satisfy `rows*cols == len` before any launch (D-04 input validation, ASVS V5) | shape integers |
| cargo/crates.io dependency → build | `cubek-matmul 0.2.0` (+ transitive `cubek-std 0.2.0`) wraps the GEMM substrate | third-party kernel code |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-0201-01 | Tampering | `from_raw_parts` in gemm launch | mitigate | `len` from validated `DeviceArray.len` (`device_array.rs:93-100`); cubek-matmul wrap, no caller-supplied raw len | closed |
| T-0201-02 | Tampering | caller `(m,k)/(k,n)` vs operand len (D-04) | mitigate | `gemm.rs:68` → `validate_geometry` (`gemm.rs:132-174`) asserts via `checked_mul` → `PrimError::ShapeMismatch`/`DimMismatch` before launch | closed |
| T-0201-SC | Tampering | cargo dependency (matmul wrap path) | mitigate | `cubek-matmul`+`cubek-std` `=0.2.0` crates.io w/ checksums (`Cargo.toml:20,24`, `Cargo.lock:1141-1158`); human-verify checkpoint-approved; no typosquat, no other new dep | closed |
| T-0202-01 | Tampering | `from_raw_parts` in reduction launches | mitigate | `reduce.rs:472-473,571-573` lens from validated `DeviceArray.len`; kernels bounds-check `ABSOLUTE_POS < input.len()` | closed |
| T-0202-02 | Tampering | caller axis geometry vs len | mitigate | `reduce.rs:191,235,347,368` → `validate_matrix` (`reduce.rs:613-635`) `rows*cols==len` via `checked_mul` before launch | closed |
| T-0202-03 | Denial of Service | plane path on no-subgroup adapter | accept | `reduce.rs:192-195,236-239,398-401` logs `warn!` and returns `Ok(None)` (skip-with-log); shared path is portable fallback | closed |
| T-0202-SC | Tampering | cargo dependency | mitigate | Zero new deps — pure `#[cube(launch)]` kernels | closed |
| T-0203-01 | Tampering | `from_raw_parts` in distance launches | mitigate | `distance.rs:137-140` lens from `DeviceArray.len`; `elementwise.rs:118` bounds-checks `i<rows && j<cols` | closed |
| T-0203-02 | Tampering | caller `(rows,cols)` vs operand len | mitigate | `distance.rs:92` → `validate_geometry` (`distance.rs:186-228`) before launch | closed |
| T-0203-03 | Information Disclosure | sqrt-of-negative → NaN to KNN | mitigate | unconditional `max(d²,0)` statement clamp (`elementwise.rs:121-124`) before optional `sqrt_elem`; pinned by `distance_min_nonnegative` (`distance_test.rs:206-257`) | closed |
| T-0203-SC | Tampering | cargo dependency | mitigate | Zero new deps — in-tree gemm+reduce+feature-free elementwise | closed |
| T-0204-01 | Tampering | `from_raw_parts` in AᵀA launch | mitigate | `covariance.rs:127-129,198-199` lens from validated counts; `center_columns`/`scale` bounds-check `tid<input.len()` | closed |
| T-0204-02 | Tampering | caller `(n_samples,n_features)` vs len | mitigate | `covariance.rs:101` → `validate_geometry` (`covariance.rs:212-262`) `checked_mul` before launch | closed |
| T-0204-SC | Tampering | cargo dependency | mitigate | Zero new deps — in-tree GEMM+column-reduce+scale+center_columns | closed |
| T-0205-01 | Tampering | test reads `PoolStats` via `stats()` | accept | `memory_gate_test.rs` reads `pool.stats()` snapshot only; test-only, no external input | closed |
| T-0205-SC | Tampering | cargo dependency | mitigate | Zero new deps — single test file over existing primitives + pool API | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-0202-03 | T-0202-03 | Plane/subgroup reduction path on adapters lacking subgroup support degrades to the portable shared-memory fallback (skip-with-log), never a crash. A missing adapter feature is a capability gap, not a vulnerability. | appservice27 (phase owner) | 2026-06-12 |
| AR-0205-01 | T-0205-01 | The D-10 memory gate reads the public `PoolStats` snapshot from test code only; no external input, no production read path. | appservice27 (phase owner) | 2026-06-12 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-12 | 16 | 16 | 0 | gsd-security-auditor (opus) |

**Notes:**
- Supply-chain verified: `cubek-matmul 0.2.0` + transitive `cubek-std 0.2.0` from crates.io with valid checksums; matches the Plan 02-01 human-verify checkpoint decision. No typosquats; only commit `9c7a84a` touched dependencies.
- Code-review gap-closure (CR-01 plane-skip panic, CR-02 buffer-release, CR-03 empty-input rejection, WR-01 n==ddof guard, WR-07/IN-02) verified resolved in code — strengthens the D-04 shape-validation mitigations (T-*-02) via `validate_nonempty`/`checked_mul`.
- Test-hardening recommendation (non-blocking, not a mitigation gap): add dedicated `(rows,cols)`-mismatch negative tests for GEMM and reduce (covariance and reduce-empty already have them). The mitigation *code* (`validate_geometry`/`validate_matrix`) is present and reached at every public entry point.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-12
