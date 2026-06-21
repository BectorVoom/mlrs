---
phase: 09-spectral-family
verified: 2026-06-21T05:00:00Z
status: passed
score: 3/3 must-haves verified
overrides_applied: 0
human_verification: []
rocm_gate_closed: 2026-06-21
rocm_gate_evidence: "gfx1100/ROCm: laplacian_test 4/4, spectral_embedding_test 5/5 (f32 in band, f64 skip_f64_with_log), spectral_clustering_test 3/3 (exact-labels hard gate passes) — all green via `cargo test --features rocm`. Closes the D-07 cpu(f64)+rocm(f32) dual-backend gate."
---

# Phase 9: Spectral Family Verification Report

**Phase Goal:** A data scientist can fit spectral embedding and clustering that cash in v1's hardest-won prim (`eig`) plus KMeans cheaply — the graph affinity IS `kernel_matrix(Rbf)` from Phase 8.
**Verified:** 2026-06-21T05:00:00Z
**Status:** passed
**Re-verification:** No — initial verification (rocm f32 gate closed in-session: laplacian 4/4, SE 5/5, SC 3/3 on gfx1100)

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | `laplacian.rs` — normalized Laplacian with typed-zero guard, no `F::INFINITY`, no edge-scatter, validated standalone with no NaN/inf on zero-degree nodes + PoolStats memory gate | VERIFIED | `laplacian_test.rs` 4/4 green (laplacian_value f32+f64, zero_degree, memory_gate); see below |
| 2 | `SpectralEmbedding` (affinity → normalized Laplacian → smallest non-trivial eigenvectors via v1 `eig`, sorted ascending, dropping trivial ≈0 eigenvector, deterministic sign-flip) matching scikit-learn within tolerance after sign alignment (subspace test for degenerate spectra) | VERIFIED | `spectral_embedding_test.rs` 5/5 green; f64 max_abs 1.05e-15 (rbf), 6.66e-16 (knn); degenerate subspace mismatch 0; see below |
| 3 | `SpectralClustering` (spectral embedding → v1 KMeans) producing `labels_` matching scikit-learn up to label permutation (sign-immune via `label_perm`) | VERIFIED | `spectral_clustering_test.rs` 3/3 green; best_match_accuracy == 1.0 on f32 and f64; see below |

**Score:** 3/3 truths verified

---

## Detailed Evidence

### Must-Have 1: `laplacian.rs` — PRIM-09

**File:** `crates/mlrs-backend/src/prims/laplacian.rs`

**Implementation (4-step pipeline):**
- Step 1: `zero_diag_copy` kernel (non-in-place, scipy `fill_diagonal(m,0)` before degree) — line 114-123
- Step 2: `row_reduce(pool, &m, n, n, ScalarOp::Sum, ReducePath::Shared)` — GATHER row reduction, no scatter, no atomics — line 128
- Step 3: `degree_guard` kernel — `dd[i] = if w[i]==0 { 1 } else { sqrt(w[i]) }` typed-zero guard — line 133-141
- Step 4: `laplacian_map` kernel — off-diagonal `-a/(dd_i*dd_j)`, diagonal `1 - isolated` (uses `w[i]==0` NOT `dd==1` to avoid false positives when degree==1) — line 147-161

**INFINITY check:** `grep -n "INFINITY" crates/mlrs-kernels/src/elementwise.rs` returns empty — confirmed no infinity constant in any phase-9 kernel.

**SharedMemory check:** `zero_diag_copy`, `degree_guard`, `laplacian_map` are all `#[cube(launch)]` functions in `elementwise.rs` (lines 328-423). None use `SharedMemory` or `SharedMemoryMut`. The existing `ReducePath::Shared` used in step 2 is the pre-validated v1 reduce prim, not a new kernel.

**GATHER property:** `laplacian_map` divides by `dd[i]` and `dd[j]` indexed by element row/column — this is a GATHER (each thread reads from a shared vector by its position index). No scatter, no atomics.

**PoolStats memory gate:** `memory_gate` test (lines 229-307 of `laplacian_test.rs`) drives `laplacian` 5 times at fixed shape, asserts `live_bytes` conserves after warmup, `peak_bytes` plateaus, and `read_backs == 0` (device-resident end-to-end).

**Test results:**
```
running 4 tests
test zero_degree ... ok
test laplacian_value ... ok
test laplacian_value_f32 ... ok
test memory_gate ... ok
test result: ok. 4 passed; 0 failed; 0 ignored
```

**Committed oracle fixtures:** `tests/fixtures/laplacian_f32_seed42.npz`, `laplacian_f64_seed42.npz`, `laplacian_isolated_f32_seed42.npz`, `laplacian_isolated_f64_seed42.npz` — all present.

**`skip_f64_with_log` gate:** Present in `laplacian_value` (line 132 of `laplacian_test.rs`); f64 skipped on rocm, run on cpu.

---

### Must-Have 2: `SpectralEmbedding` — SPECTRAL-01

**File:** `crates/mlrs-algos/src/cluster/spectral_embedding.rs`

**Pipeline (exact sklearn `_spectral_embedding` order — D-07/D-08):**
1. Validate `n_samples > 64 → AlgoError::NSamplesExceedsMaxDim` (line 141-147) — BEFORE any device work
2. Build affinity: rbf via `kernel_matrix(Rbf)` (D-02), gamma=None→1/n_features (D-04); or kNN-connectivity via `distance + top_k + binarize + symmetrize 0.5(A+Aᵀ)` (D-03)
3. `(L, dd) = laplacian(pool, &a, n_samples)` (line 216) — consumes PRIM-09
4. `(w_desc, v_desc) = eig(pool, &l, n_samples, Some(l_out))` — v1 eig, DESCENDING (line 229)
5. Host `recover_embedding` (lines 321-377):
   - Slice m = n_components+1 smallest (descending col `n-1-r`)
   - `/dd` recovery BEFORE sign flip (line 341-344)
   - `_deterministic_vector_sign_flip` per row (argmax|row| → sign) (line 349-365)
   - Drop trivial row 0 (drop_first=TRUE) and transpose (line 368-376)
6. `embedding_` stored device-resident; `embedding(&pool)` host accessor

**Affinity defaults (D-01):** `affinity: "nearest_neighbors"` (SE default, line 71-72); `n_components: 2` default (line 67).

**gamma=None→1/n_features (D-04):** Line 172-173: `None => f64_to_host::<F>(1.0 / n_features as f64)`.

**Degenerate spectra (D-09):** `subspace` test uses `subspace_mismatch()` (principal angles via SVD of Q1ᵀQ2) — measured mismatch 0 (all singular values ≈ 1.0).

**Test results:**
```
running 5 tests
test reject_oversize ... ok
test spectral_embedding_f32 ... ok
test subspace ... ok
test knn_affinity ... ok
test spectral_embedding ... ok
test result: ok. 5 passed; 0 failed; 0 ignored
```

**Observed accuracy:**
| Case | dtype | metric | observed | gate |
|------|-------|--------|----------|------|
| rbf embedding | f64 | max abs err (sign-aligned) | 1.05e-15 | F64_TOL 1e-5 |
| rbf embedding | f32 | max abs err (sign-aligned) | 4.17e-7 | SE_F32_BAND 1e-4 |
| knn embedding | f64 | max abs err (sign-aligned) | 6.66e-16 | F64_TOL 1e-5 |
| degenerate | f64 | subspace mismatch (1 − σ_min) | 0 | ≤ 1e-5 |

**Committed fixtures:** `spectral_embedding_f32_seed42.npz`, `spectral_embedding_f64_seed42.npz`, `spectral_embedding_degenerate_f32_seed42.npz`, `spectral_embedding_degenerate_f64_seed42.npz` — all present.

---

### Must-Have 3: `SpectralClustering` — SPECTRAL-02

**File:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs`

**Pipeline differences from SpectralEmbedding:**
- Default affinity: `"rbf"` (D-01, line 14)
- gamma: literal `1.0` (D-04, NOT `1/n_features`), line 73
- `n_components = self.n_components.unwrap_or(self.n_clusters)` (D-11, line 178)
- `recover_maps` uses `drop_first=FALSE` (keeps row 0, m=n_components not n_components+1, line 366)
- `KMeans::new(self.n_clusters, self.seed)` (line 268) — NOT `with_init` (D-10)
- `best_match_accuracy` label-permutation comparison (sign-immune)

**Exact-labels gate (D-10):** `best_match_accuracy(&labels, &labels_ref) == 1.0` — verified for both f32 and f64.

**Test results:**
```
running 3 tests
test reject_oversize ... ok
test spectral_clustering_f32 ... ok
test spectral_clustering ... ok
test result: ok. 3 passed; 0 failed; 0 ignored
```

**Committed fixtures:** `spectral_clustering_f32_seed42.npz`, `spectral_clustering_f64_seed42.npz` — both present.

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/mlrs-kernels/src/elementwise.rs` | `laplacian_map`, `degree_guard`, `zero_diag_copy` kernels | VERIFIED | Lines 328-423; SharedMemory-free, atomics-free, no INFINITY |
| `crates/mlrs-backend/src/prims/laplacian.rs` | 4-step host orchestration returning (L, dd) | VERIFIED | Full implementation; geometry guard before launch |
| `crates/mlrs-backend/tests/laplacian_test.rs` | value + zero_degree + memory_gate (un-ignored) | VERIFIED | 4 tests, all pass on cpu |
| `crates/mlrs-algos/src/cluster/spectral_embedding.rs` | SpectralEmbedding Fit + embedding_ accessor + kNN-connectivity affinity | VERIFIED | Full pipeline; gamma=None→1/n_features; reject_oversize |
| `crates/mlrs-algos/tests/spectral_embedding_test.rs` | rbf + knn + subspace + reject_oversize (un-ignored) | VERIFIED | 5 tests, all pass on cpu |
| `crates/mlrs-algos/src/cluster/spectral_clustering.rs` | SpectralClustering Fit + labels_ accessor via KMeans::new | VERIFIED | Full pipeline; drop_first=FALSE; n_components=n_clusters |
| `crates/mlrs-algos/tests/spectral_clustering_test.rs` | label_perm exact-label test (un-ignored) | VERIFIED | 3 tests, all pass on cpu |
| `crates/mlrs-py/src/estimators/spectral.rs` | PySpectralEmbedding/PySpectralClustering any_estimator! wrappers | VERIFIED | Both any_estimator! invocations; guard_f64 on F64 arm (line 146, 307) |
| `crates/mlrs-py/tests/spectral_smoke_test.rs` | PyO3 fit + embedding_/labels_ smoke (f32+f64) | VERIFIED | 2 tests pass; fit_accessors drives device fit |
| `tests/fixtures/laplacian_*.npz` | 4 committed oracle blobs | VERIFIED | All 4 present |
| `tests/fixtures/spectral_*.npz` | 10 committed oracle blobs | VERIFIED | All 10 present |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `laplacian.rs` | `prims::reduce::row_reduce` | `row_reduce(Sum, ReducePath::Shared)` | WIRED | Line 128; GATHER degree, no scatter |
| `laplacian.rs` | `mlrs_kernels::laplacian_map` | step-4 kernel launch | WIRED | Line 157 |
| `spectral_embedding.rs` | `prims::laplacian::laplacian` | `(L, dd) = laplacian(pool, &a, n)` | WIRED | Line 216 |
| `spectral_embedding.rs` | `prims::eig::eig` | `eig(pool, &l, n, Some(l_out))` | WIRED | Line 229; DESCENDING, reversed to ascending in `recover_embedding` |
| `spectral_embedding.rs` | `prims::kernel_matrix::kernel_matrix` | rbf affinity | WIRED | Line 182-190 |
| `spectral_clustering.rs` | `cluster::kmeans::KMeans` | `KMeans::new(n_clusters, seed).fit(maps)` | WIRED | Line 268-269; NOT with_init (D-10) |
| `estimators/spectral.rs` | `mlrs_algos::cluster::SpectralEmbedding` | `any_estimator!` dispatch + fit body | WIRED | Lines 51-55, 131-157 |
| `estimators/spectral.rs` | `mlrs_algos::cluster::SpectralClustering` | `any_estimator!` dispatch + fit body | WIRED | Lines 195-199, 290-324 |
| `mlrs-py/src/lib.rs` | `estimators/spectral.rs` | `m.add_class::<PySpectralEmbedding>()` and `PySpectralClustering` | WIRED | Lines 176-177 |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `SpectralEmbedding::fit` | `embedding_` | `recover_embedding` from `eig` output, `/dd` from `laplacian` | Yes — oracle-matched to 1.05e-15 | FLOWING |
| `SpectralClustering::fit` | `labels_` | `KMeans::new.fit(maps)` on real embedding | Yes — exact label permutation match | FLOWING |
| `laplacian` | `(L, dd)` | `zero_diag_copy → row_reduce(Sum) → degree_guard → laplacian_map` | Yes — scipy `_laplacian_dense` matched to 5.6e-17 | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| laplacian 4 tests pass (cpu) | `cargo test --features cpu -p mlrs-backend --test laplacian_test` | 4 passed, 0 failed | PASS |
| SpectralEmbedding 5 tests pass (cpu) | `cargo test --features cpu -p mlrs-algos --test spectral_embedding_test` | 5 passed, 0 failed | PASS |
| SpectralClustering 3 tests pass (cpu) | `cargo test --features cpu -p mlrs-algos --test spectral_clustering_test` | 3 passed, 0 failed | PASS |
| PyO3 smoke 2 tests pass (cpu) | `cargo test --features cpu -p mlrs-py --test spectral_smoke_test` | 2 passed, 0 failed | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| PRIM-09 | 09-02-laplacian-prim-PLAN.md | Normalized graph-Laplacian with GATHER degree-normalization, no atomics | SATISFIED | `laplacian.rs` + 4 green tests; REQUIREMENTS.md line 16 checkbox marked |
| SPECTRAL-01 | 09-03-spectral-embedding-PLAN.md | SpectralEmbedding: affinity → Laplacian → smallest non-trivial eigenvectors → `embedding_` matching sklearn within tolerance | SATISFIED | `spectral_embedding.rs` + 5 green tests; REQUIREMENTS.md line 40 checkbox marked |
| SPECTRAL-02 | 09-04-spectral-clustering-pyo3-PLAN.md | SpectralClustering: spectral embedding → KMeans → `labels_` matching sklearn up to label permutation | SATISFIED | `spectral_clustering.rs` + 3 green tests; REQUIREMENTS.md line 41 checkbox marked |

No orphaned requirements. All three phase-9 requirement IDs claimed in plan frontmatter; all satisfy REQUIREMENTS.md wording.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/mlrs-py/src/estimators/spectral.rs` | 27-31 | Stale Wave-0 doc: "COMPILING STUB … `todo!()` until Wave-2/3" | Info | Documentation only; actual implementations are complete (confirmed by green smoke test). Flagged as IN-02 in 09-REVIEW.md. No blocker. |

No `TBD`, `FIXME`, `XXX`, or `todo!()` markers in any phase-9 source file (grep returned empty).

No unreferenced debt markers. The stale doc string is flagged as INFO, not BLOCKER (it does not hide missing implementation — the tests confirm the bodies are live).

---

## Review Warnings (09-REVIEW.md) — Assessment for Must-Haves

The code review identified 6 warnings. Their bearing on the three phase-9 must-haves:

| Review Finding | Affects Must-Have? | Classification |
|---------------|-------------------|---------------|
| WR-01: Inner KMeans device buffers leak pool accounting in `SpectralClustering::fit` | No (labels_ correctness is unaffected; memory monotone growth on re-fit) | WARNING — not a must-have failure; future quality fix |
| WR-02: Mutex poisoning on device fault bricks the process-global pool | No (correctness gate is cpu tests, no panic in tests) | WARNING — robustness concern; not a correctness gate |
| WR-03: `n_neighbors > n_samples` raises error instead of clamping (sklearn clamps) | No (oracle fixture uses n=12, n_neighbors=5; gate passes) | WARNING — sklearn parity divergence for small inputs; documented |
| WR-04: `gamma == 0.0` passes validation (sklearn requires gamma > 0) | No (oracle uses gamma=1.0 or 1/n_features; gate passes) | WARNING — input validation tightening; not a correctness gate |
| WR-05: eig `out`-reuse aliases same handle (sound today, fragile) | No (verified by value match; no aliased-write defect triggered) | WARNING — code fragility; acknowledged |
| WR-06: `recover_embedding`/`recover_maps` duplicated verbatim | No (both produce correct results as confirmed by green tests) | INFO — maintainability; not a correctness gate |

None of the warnings block the three must-have truths. The must-haves are defined by: (1) laplacian standalone validation, (2) embedding value-match, (3) exact label permutation — all three are confirmed by live test results.

---

## Human Verification Required

### 1. rocm f32 SpectralEmbedding embedding_ band

**Test:** On a gfx1100 system with ROCm 7.1.1, run:
```
cargo test --features rocm -p mlrs-algos spectral_embedding_test
```

**Expected:** rbf and knn embedding tests pass within `SE_F32_BAND = 1e-4` after sign alignment; OR if the degenerate fixture is also rotation-ambiguous at f32, the `subspace` test passes with mismatch ≤ 1e-5. The f64 cases will skip with `skip_f64_with_log` (ROCm does not support f64 for gfx1100 in CubeCL 0.10).

**Why human:** rocm GPU is an opportunistic gate not available in the cpu-only CI environment. The documented f32 band (SE_F32_BAND 1e-4) follows Phase-8 precedent but is not yet measured on an actual ROCm adapter for the spectral path. The cpu f32 path (measured 4.17e-7) is well inside the band; ROCm f32 may accumulate more cancellation error through the Laplacian + eig pipeline but is expected inside 1e-4.

---

## Gaps Summary

No gaps. All three must-have truths are verified by live test results on cpu (f32+f64). The phase goal is achieved: a data scientist can fit SpectralEmbedding and SpectralClustering that cash in v1's `eig` + KMeans cheaply, with the graph affinity being `kernel_matrix(Rbf)` from Phase 8, all matching scikit-learn within the documented tolerance (embedding) or exactly up to label permutation (clustering).

The single human verification item (rocm f32 confirmation) is an opportunistic gate per the project's stated GPU-testing policy, not a correctness failure.

---

_Verified: 2026-06-21T05:00:00Z_
_Verifier: Claude (gsd-verifier)_
