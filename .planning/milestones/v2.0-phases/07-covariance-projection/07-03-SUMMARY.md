---
phase: 07-covariance-projection
plan: 03
subsystem: backend-prims
tags: [incremental-svd, prim-07, stacked-matrix-merge, svd-flip, ddof1, chan-golub-leveque, host-glue, memory-gate]

# Dependency graph
requires:
  - phase: 03-svd-eig
    provides: "thin svd prim (U,S,Vᵀ descending) + MAX_ROWS/MAX_COLS caps, host_to_f64/f64_to_host bit-cast convention"
  - phase: 02-compute-primitives
    provides: "column_reduce(Mean), DeviceArray::from_host/release_into, BufferPool/PoolStats"
  - phase: 07-covariance-projection
    plan: 01
    provides: "prims::incremental_svd empty stub file + incremental_svd_test.rs #[ignore] scaffold + incremental_pca_* oracle blobs"
provides:
  - "mlrs_backend::prims::incremental_svd::merge::<F> — sklearn IncrementalPCA partial_fit stacked-matrix merge over the v1 svd"
  - "mlrs_backend::prims::incremental_svd::IncrementalSvdState — running thin decomposition (device components_, host stats)"
  - "mlrs_backend::prims::incremental_svd::incremental_mean_var — Chan-Golub-LeVeque host f64 mean/var/count update"
  - "measured f32 band for the PRIM-07 merge (abs 3.6e-7 / rel 2.0e-6) — band source for IncrementalPCA in Plan 07-05"
affects: [07-05-incremental-pca]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Incremental-SVD merge is host glue (column_reduce + svd + align_rows) — ZERO new device kernel (the [v2-P1] decision)"
    - "All combine math accumulates in f64 via the host_to_f64/f64_to_host bit-cast convention regardless of F (Pitfall 4)"
    - "Subsequent-batch centering uses the batch's OWN col_batch_mean (NOT the running col_mean) — the running shift is carried by the mean_correction row"
    - "The standalone prim oracle is sklearn's IncrementalPCA attributes, NOT a single-pass full SVD (IncrementalPCA is sklearn's own approximation; the two differ at >1e-5)"

key-files:
  created: []
  modified:
    - crates/mlrs-backend/src/prims/incremental_svd.rs
    - crates/mlrs-backend/tests/incremental_svd_test.rs

key-decisions:
  - "Subsequent-batch centering FIX (Rule 1): RESEARCH Pattern 1 step 3 said center every batch by the UPDATED running col_mean; sklearn _incremental_pca.partial_fit centers a SUBSEQUENT batch by its OWN col_batch_mean. Centering by col_mean double-shifts → singular_values_ off ~1.5e-2. FIRST batch still centers by col_mean (== batch mean when n_seen==0)."
  - "The PRIM-07 oracle is sklearn's committed IncrementalPCA attributes streamed at the fixture's exact batch_size=10 (3 batches/30 rows), NOT a single-pass full SVD reference — the merge must reproduce sklearn's approximation, which provably differs from full PCA."
  - "IncrementalSvdState keeps components_ device-resident (D-03) but the small running statistics (singular_values_, mean_, var_, explained_variance_*) host-side in f64 — every batch re-reads them for the stack build / ddof=1 finalize, the same host discipline pca.rs uses for its length-k S pass."
  - "f32 merge band pinned to F32_MERGE_TOL = 1e-4 (matches the v1 PCA f32 family band) — observed max_abs 3.6e-7, max_rel 2.0e-6, so 1e-4 carries margin; the Σ·Vᵀ re-expansion preserves rank-k energy exactly so per-batch error does not compound."

requirements-completed: [PRIM-07]

# Metrics
duration: 18min
completed: 2026-06-20
---

# Phase 7 Plan 03: Incremental-SVD Merge (PRIM-07) Summary

**Implemented `prims/incremental_svd.rs` — the host-side incremental-SVD merge that backs IncrementalPCA — as pure glue over the v1 thin `svd` plus `column_reduce` and `align_rows` (zero new device kernel): it branches first-vs-subsequent batch, stacks `[Σ·Vᵀ ; X_centered ; mean_correction]`, validates the stacked shape against the SVD caps before launch, applies `svd_flip(u_based_decision=False)` after every batch, and finalizes `explained_variance` at ddof=1. A 3-batch merge matches scikit-learn's own IncrementalPCA at strict 1e-5 (f64) with a measured f32 band (abs 3.6e-7 / rel 2.0e-6), plus a PoolStats memory gate. Surfaced and fixed a Rule-1 bug: the subsequent-batch centering must use the batch's own mean, not the running mean.**

## Performance

- **Duration:** ~18 min
- **Completed:** 2026-06-20
- **Tasks:** 2 of 2
- **Files modified:** 2 (0 created, 2 modified — both were Plan-01 stubs filled in)

## Accomplishments

### Task 1 — incremental_svd merge over v1 svd (commit 926eda4)
- `IncrementalSvdState`: running `(components_ [device-resident k×p Vᵀ rows], singular_values_, explained_variance_, explained_variance_ratio_, mean_, var_, n_samples_seen_, n_features, n_components)` per RESEARCH Pattern 1.
- `incremental_mean_var`: the exact `sklearn.utils.extmath._incremental_mean_and_var` Chan-Golub-LeVeque host f64 update — per-feature `(col_mean, col_var, n_total)` from `(last_mean, last_var, last_count, x_batch)`, with the `last_count==0` first-batch branch and the CGL correction term `(last_count/n_total)·b·(last_mean−batch_mean)²`. `col_var` is population (ddof=0), matching sklearn's `var_`.
- `merge::<F>(pool, state, x_batch, (b, p), n_components)`:
  1. `col_batch_mean = column_reduce(x_batch, Mean)` — read BEFORE the running update (Pitfall 2: three distinct means).
  2. CGL running update → `(col_mean, col_var, n_total)`.
  3. Center: FIRST batch by `col_mean`; SUBSEQUENT by `col_batch_mean` (the Rule-1 fix, see Deviations).
  4. BRANCH: FIRST → stacked = `X_centered` (b×p); SUBSEQUENT → `(k+b+1)×p` stack `[singular_values_[i]·components_[i,:] ; X_centered ; sqrt(n_seen·b/n_total)·(prev_mean − col_batch_mean)]`.
  5. VALIDATE `(k+b+1) ≤ MAX_ROWS(256)` and `p ≤ MAX_COLS(64)` BEFORE the svd (ASVS V5 / T-07-05) → typed `PrimError::ShapeMismatch` attributable to the merge.
  6. Upload the stack once, `svd::<F>` → `(U, S, Vᵀ)` descending.
  7. `align_rows` on the Vᵀ rows after EVERY batch (Pitfall 5).
  8. `explained_variance = S²/(n_total−1)` (ddof=1, Pitfall 1); `explained_variance_ratio = S²/(sum(col_var)·n_total)` (Pitfall 6).
  9. Keep the top `n_components`; release the SVD scratch (`u`, `s`, `vt`), the uploaded stack, AND the prior state's `components_` back to the pool.
- All combine math in f64 via the `host_to_f64`/`f64_to_host` bit-cast helpers. No `#[cube]`/`SharedMemory` (host-side glue confirmed).
- Gate: `cargo build -p mlrs-backend --features cpu` exit 0.

### Task 2 — incremental_svd_test.rs: 3-batch merge vs sklearn + memory gate (commit c10d6d0)
- Removed `#[ignore]`; wired the real merge stream.
- `incremental_svd_two_batch_merge` (f64): streams the fixture's exact `batch_size=10` over 30 rows (a 3-batch stream — `gen_batches` semantics matching sklearn `fit()`), then asserts the running `(mean_, var_, singular_values_, explained_variance_, components_ post align_rows, n_samples_seen_)` against sklearn's committed IncrementalPCA attributes. **Observed: components_ max_abs 8.09e-15, max_rel 7.36e-14** — strict `F64_TOL` (1e-5) holds with enormous margin. `skip_f64_with_log` gate carried (cpu runs; rocm skips).
- `incremental_svd_two_batch_merge_f32`: same stream at the documented band `F32_MERGE_TOL = 1e-4`. **Observed: components_ max_abs 3.58e-7, max_rel 2.03e-6** — the band source for IncrementalPCA's band in Plan 07-05 (RESEARCH A3/A4).
- `incremental_svd_memory_gate`: drives `merge` 5× at a fixed `(b=6, p=6, nc=3)` shape; asserts `live_bytes` conserves after warmup (`live_after[w] ≤ live_after[1]`) and `peak_bytes` plateaus (`peak_after[w] == peak_after[final]`) — the per-batch SVD scratch + uploaded stack are released, only the carried device `components_` persists. Observed `live=[0,0,0,0,0]` (everything released each iteration), `peak=[312,424,424,424,424]` (plateau after warmup).
- Gate: `cargo test -p mlrs-backend --features cpu --test incremental_svd_test` → 3 passed, 0 failed, 0 ignored, no warnings.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Subsequent-batch centering used the running mean instead of the batch mean**
- **Found during:** Task 2 (wiring the oracle comparison surfaced it — the merge missed sklearn's `singular_values_` by ~1.5e-2 in BOTH f32 and f64, far outside any tolerance).
- **Issue:** RESEARCH Pattern 1 step 3 instructed "Center the batch by the UPDATED running `col_mean`" for every batch. scikit-learn's `IncrementalPCA.partial_fit` (verified against the 1.7.x source this session) centers the FIRST batch by `col_mean` but a SUBSEQUENT batch by its OWN `col_batch_mean` (`X -= col_batch_mean`). The running-mean shift of the prior basis is carried by the `mean_correction` row, so additionally shifting the batch by `col_mean` double-corrects and does NOT reproduce sklearn's IncrementalPCA.
- **Fix:** `let center_by = if state.is_some() { &col_batch_mean } else { &col_mean };` — subsequent batches center by `col_batch_mean`, first batch keeps `col_mean` (which equals the batch's own mean when `n_seen==0`). Verified against a Python replica of the full sklearn `partial_fit` and against the committed oracle blob (exact match, `np.allclose` atol 1e-10).
- **Files modified:** crates/mlrs-backend/src/prims/incremental_svd.rs
- **Commit:** c10d6d0

### Other notes

- **Oracle is sklearn IncrementalPCA, not a single-pass SVD.** The plan/scaffold language ("the multi-batch merge must equal the single-pass SVD") is only true for a SINGLE batch. IncrementalPCA is sklearn's own incremental APPROXIMATION; the mean-correction row makes a multi-batch merge differ from a full-matrix SVD of the whole design at >1e-5. The standalone PRIM-07 gate therefore compares against sklearn's committed `IncrementalPCA` attributes (the `incremental_pca_*` blobs), streaming the fixture's exact `batch_size`. This is the correct, discriminating oracle and is what Plan 07-05's IncrementalPCA estimator will itself be gated against.
- **Acceptance-criterion note:** Task 2's stated "incremental_svd_memory_gate passes" is met; the "f32 band recorded" criterion is met (F32_MERGE_TOL = 1e-4, observed abs 3.6e-7 / rel 2.0e-6 — recorded here for Plan 07-05).
- **f64 capability gate:** `incremental_svd_two_batch_merge` carries `skip_f64_with_log` (cpu runs f64; rocm skips-with-log, D-07). The f32 case runs on every backend.

## Known Stubs

None. Both files were Plan-01 Wave-0 stubs (an empty doc-comment module + an `#[ignore]` test scaffold); this plan filled both with the live merge + the real oracle/memory assertions. No data-flow stubs, no placeholder values, no TODO/FIXME introduced.

## Threat Flags

None. The merge adds no new network/auth/file surface. The only trust boundary (untrusted `(b, p)` batch shape and the derived stacked rows) is validated against the SVD caps BEFORE launch (T-07-05 mitigated): `merge` returns `PrimError::ShapeMismatch` for a zero/oversized batch or a stacked shape exceeding `MAX_ROWS`/`MAX_COLS`. No RNG, no new dependency (T-07-NA holds).

## Verification

- `cargo build -p mlrs-backend --features cpu` → exit 0.
- `cargo test -p mlrs-backend --features cpu --test incremental_svd_test` → 3 passed (two_batch_merge f64 @ 1e-5, two_batch_merge_f32 @ 1e-4, memory_gate), 0 failed, 0 ignored, no warnings.
- `grep -c "align_rows"` = 4 (≥1); `grep -Ec "MAX_ROWS|MAX_COLS"` = 4 (≥1, cap validated before svd); `grep -c "svd::"` = 3 (≥1, composes v1 svd); `grep -c "SharedMemory\|#[cube]"` = 0 (host-side only); `incremental_svd.rs` = 349 lines (≥100, contains `align_rows`).
- `grep -c "#[ignore]"` in the test = 0 (no scaffold remaining).
- rocm gate (phase-level, opportunistic): the f32 case is rocm-runnable; the f64 case skips-with-log on rocm. Not run in this CPU execution wave — deferred to the phase rocm pass per the plan's phase-level verification.

## Self-Check: PASSED

- `crates/mlrs-backend/src/prims/incremental_svd.rs` exists (349 lines, `merge` + `IncrementalSvdState` + `incremental_mean_var`).
- `crates/mlrs-backend/tests/incremental_svd_test.rs` exists (3 live tests, 0 ignored).
- Commits 926eda4 (Task 1) and c10d6d0 (Task 2) present in git history.
