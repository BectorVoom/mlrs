---
phase: 09-spectral-family
fixed_at: 2026-06-21T05:10:00Z
review_path: .planning/phases/09-spectral-family/09-REVIEW.md
iteration: 2
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 9: Code Review Fix Report

**Fixed at:** 2026-06-21T05:10:00Z
**Source review:** .planning/phases/09-spectral-family/09-REVIEW.md
**Iteration:** 2

> Note: a prior fix report (iteration 1, for an earlier review pass covering
> WR-01-old/WR-05/WR-06/WR-07) is preserved in git history at commit `cecd1de`.
> This iteration-2 report covers the fresh `09-REVIEW.md` pass (reviewed
> 2026-06-21T04:30, 4 Warnings + 3 Info).

**Summary:**
- Findings in scope: 7 (4 Warning + 3 Info; fix_scope = "all")
- Fixed: 7
- Skipped: 0

All in-scope findings were fixed and committed atomically. Spectral algo tests
(`spectral_clustering_test`, `spectral_embedding_test` — 8 tests incl.
`knn_affinity` and the clustering `fit_predict` path) pass post-fix, and both
`mlrs-algos` and `mlrs-py` type-check clean under `--features cpu --tests`.

## Human-verification flags

Three fixes change runtime behaviour or data flow (not just docs) and are flagged
`fixed: requires human verification` per the logic-bug limitation — syntax/parse
checks and the targeted test run pass, but a human should confirm the semantics:

- **WR-02** — re-fit now reads persisted params (behavioural change to fit path).
- **WR-03** — `fit_predict` now sources labels from a retained host buffer.
- **IN-02** — affinity builder moved to a shared function (behaviour-preserving
  move; validated by the passing `knn_affinity` + clustering tests).

## Fixed Issues

### WR-01: Mutex-poison recovery silently corrupts pool accounting

**Files modified:** `crates/mlrs-py/src/lib.rs`
**Commit:** 2b34701
**Status:** fixed
**Applied fix:** Qualified the `lock_pool` doc (the reviewer's first option).
Documented that "not left torn" is a memory-safety statement only; after a
recovered poison, `live_bytes`/`peak_bytes` may be permanently inflated, and the
FOUND-05 conservation property is VOID. The reviewer's alternative
(`reconcile_live_bytes`) is not implementable: the pool's free-list only tracks
released handles, while live handles are owned by `DeviceArray`s outside the pool,
so there is no in-pool truth source to recompute `live_bytes` from. The honest doc
qualification is the correct fix.

### WR-02: Re-`fit` discards user hyperparameters and reverts to defaults

**Files modified:** `crates/mlrs-py/src/estimators/spectral.rs`
**Commit:** 5b59088
**Status:** fixed: requires human verification
**Applied fix:** Added a persisted `params` field (`SpectralEmbeddingParams` /
`SpectralClusteringParams`) to each `#[pyclass]` struct, populated in
`new`/`unfit_default`, and read in `fit` via a destructuring of `self.params`
instead of the `Unfit`-arm match with a hardcoded-defaults catch-all. A second
`fit` of the same object now honours the user's constructor params (sklearn
semantics). The shared `any_estimator!` macro was deliberately NOT modified (used
by 12 estimators); this leaves the `Unfit` enum fields unread, producing 2 benign
`dead_code` warnings on those macro-generated fields (no `deny(warnings)` in the
project; build passes).

### WR-03: `fit_predict` round-trips labels host→device→host→device needlessly

**Files modified:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs`
**Commit:** 60b334b
**Status:** fixed: requires human verification
**Applied fix:** Added a private `labels_host_: Option<Vec<i32>>` field that `fit`
populates from the `labels_host` it already materializes. `fit_predict` now builds
its returned device buffer directly from that retained host vector, eliminating the
extra device→host read-back that `self.labels(pool)` incurred. The returned buffer
is an independent allocation (no aliasing of `self.labels_`). Verified by the
passing clustering tests.

### WR-04: Inconsistent pool-lock helper undermines the poison recovery

**Files modified:** `crates/mlrs-py/src/estimators/kernel.rs`,
`crates/mlrs-py/src/dispatch.rs`, `crates/mlrs-py/src/lib.rs`
**Commit:** adc0979
**Status:** fixed
**Applied fix:** Converted all 8 `kernel.rs` lock sites (the named sibling file the
spectral wrappers copy structure from) from the panicking
`global_pool().lock().expect(...)` form to the poison-recovering
`crate::lock_pool()`. Updated the `dispatch.rs` doc skeleton + extension comment to
show `lock_pool` as the sanctioned path, and documented in `lib.rs` that `lock_pool`
is the single authoritative lock path — explicitly noting the remaining
`linear`/`cluster`/`decomposition`/`covariance`/`neighbors`/`projection` wrappers
as a tracked legacy migration (the reviewer's "or explicitly document" alternative).
Those ~110 sites across unrelated, out-of-phase estimators were intentionally NOT
bulk-converted in this spectral-phase fix to keep the change scoped; the policy is
now documented so the migration is an explicit decision.

### IN-01: Unused `rng` binding on the degenerate SpectralEmbedding fixture path

**Files modified:** `scripts/gen_oracle.py`
**Commit:** 655ce21
**Status:** fixed
**Applied fix:** Moved `rng = np.random.default_rng(seed)` into the `else`
(non-degenerate) branch that actually consumes it, with a comment noting the
degenerate path is deterministic. Python AST parse check passes.

### IN-02: `knn_connectivity_affinity` duplicated verbatim across the two estimators

**Files modified:** `crates/mlrs-algos/src/cluster/spectral.rs`,
`crates/mlrs-algos/src/cluster/spectral_embedding.rs`,
`crates/mlrs-algos/src/cluster/spectral_clustering.rs`
**Commit:** 4e0c451
**Status:** fixed: requires human verification
**Applied fix:** Moved the byte-identical builder into
`crate::cluster::spectral::knn_connectivity_affinity::<F>` (alongside `recover`),
called from both estimators, and dropped the now-unused `distance`/`top_k`/
`f64_to_host` imports. Behaviour-preserving; validated by the passing
`knn_affinity` and clustering tests.

### IN-03: SC gamma underflow accepted (sklearn-parity question)

**Files modified:** `crates/mlrs-algos/src/cluster/spectral_clustering.rs`,
`crates/mlrs-algos/src/cluster/spectral_embedding.rs`
**Commit:** 6536974
**Status:** fixed
**Applied fix:** The reviewer asked for an explicit decision, not a code change.
Added comments at both rbf gamma guards recording that accepting a finite-positive
gamma which underflows to an effective-constant affinity (rejecting only
`gamma <= 0` / non-finite) is intentional sklearn parity, not an oversight.

## Skipped Issues

None — all in-scope findings were fixed.

---

_Fixed: 2026-06-21T05:10:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
