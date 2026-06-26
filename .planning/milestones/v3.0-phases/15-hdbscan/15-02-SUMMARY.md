---
phase: 15-hdbscan
plan: 02
subsystem: testing
tags: [hdbscan, oracle-fixtures, sklearn, npz, nyquist-gate, label-perm]

# Dependency graph
requires:
  - phase: 15-01
    provides: "mlrs_core::best_match_accuracy_pinned_noise (-1-pinned exact-label matcher)"
  - phase: 13-knn-graph
    provides: "load_npz oracle loader + per-metric fixture conventions (gen_knn_metric)"
  - phase: 05-kmeans
    provides: "kmeans_test.rs exact-label + same-permutation value-gate precedent"
provides:
  - "gen_hdbscan(seed, dtype, metric, structure) generator in scripts/gen_oracle.py"
  - "22 committed hdbscan_*.npz oracle fixtures (6 metrics x {f32,f64} + tieheavy/nested/allnoise/single/tiny)"
  - "hdbscan_test.rs oracle gate suite — 9 gate families wired to fixtures + pinned-noise matcher"
affects: [15-03, 15-04, 15-05, 15-06]

# Tech tracking
tech-stack:
  added: [hdbscan==0.8.44 (fixture-gen only, /tmp venv), scikit-learn==1.9.0 (oracle)]
  patterns:
    - "Per-structure oracle fixture dispatch (blobs/tieheavy/nested/edge) with in-script knob-divergence assertions"
    - "Wave-0 #[ignore]-pending gate functions whose bodies compile against the trivial shell (oracle self-consistency placeholders)"

key-files:
  created:
    - ".planning/phases/15-hdbscan/15-02-SUMMARY.md"
  modified:
    - "scripts/gen_oracle.py"
    - "crates/mlrs-algos/tests/hdbscan_test.rs"
    - "tests/fixtures/hdbscan_*.npz (22 new committed blobs)"

key-decisions:
  - "hdbscan 0.8.44 oracle forced to algorithm='generic' — default 'best' BallTree rejects cosine and approximates the others (Rule 3 blocking fix)"
  - "epsilon knob cross-oracled against hdbscan 0.8.44 labels_epsilon — sklearn 1.9.0 epsilon_search crashes (traverse_upwards TypeError) on every merging-epsilon tree (Rule 3 blocking fix, D-07-sanctioned)"
  - "tie-heavy fixture redesigned as two integer-lattice clusters (not a single grid) so a real 2-cluster partition forms — a single grid yielded all-noise, making the D-04 gate trivial"
  - "single edge case uses allow_single_cluster=True + min_samples=2 — a homogeneous blob has no density split so default eom yields all-noise"
  - "precomputed fixtures store empty centers — sklearn refuses store_centers with a precomputed matrix"
  - "Wave-0 gates that need not-yet-built symbols are #[ignore]d but their bodies compile (oracle self-consistency placeholders) per the plan's compile-green requirement"

patterns-established:
  - "In-generator divergence assertions: each non-default-knob fixture asserts its labels differ from the eom default BEFORE writing (Pitfall 5)"
  - "Gate-fn macro family (labels_match_metric!) generating per-metric x {f32,f64} #[ignore]-pending oracle gates with un-ignore markers"

requirements-completed: []

# Metrics
duration: ~75min
completed: 2026-06-24
status: complete
---

# Phase 15 Plan 02: HDBSCAN Wave-0 Oracle Scaffolding Summary

**gen_hdbscan generators + 22 committed sklearn/hdbscan-0.8.44 .npz oracle fixtures (6 metrics x f32/f64 distinct-MST-weight gates, a tie-heavy/duplicate D-04 gate, nested-density knob fixtures, and edge cases) plus the 9-family hdbscan_test.rs oracle gate suite wired to the -1-pinned label matcher.**

## Performance

- **Duration:** ~75 min
- **Completed:** 2026-06-24
- **Tasks:** 2
- **Files modified:** 3 (gen_oracle.py, hdbscan_test.rs, +22 fixture blobs)

## Accomplishments
- `gen_hdbscan(seed, dtype, metric, structure)` in `scripts/gen_oracle.py`: fits `sklearn.cluster.HDBSCAN` (primary oracle, `copy=True`) + `hdbscan` 0.8.44 (GLOSH `outlier_scores_` + cross-check, `algorithm='generic'`), stores `X/labels/probabilities/centroids/medoids` + `hdb_labels/outlier_scores`, with per-knob label vectors for the nested fixture.
- 22 committed `.npz` fixtures: 6 metrics × {f32,f64} distinct-MST-edge-weight gate blobs (Pitfall 1 option 2), a tie-heavy + duplicate-point D-04 TRUE GATE fixture (R-9), a nested-density knob fixture (eom/leaf/epsilon/max_cluster_size/alpha all demonstrably diverge — asserted in-script), and all-noise/single-cluster/tiny edge cases.
- `hdbscan_test.rs` replaced with the full oracle gate suite: 9 gate families (`labels_match_sklearn`, `tie_break_exact`, `probabilities_match`, `outlier_scores_match`, `centers_match`, `selection_knobs`, `build_validation`, `memory_gate`, `edge_cases`) wired to the fixtures and the `best_match_accuracy_pinned_noise` matcher. Compiles green; 3 active gates pass, 24 `#[ignore]`-pending with `un-ignore in 15-NN` markers.

## Task Commits

1. **Task 1: gen_hdbscan generators + committed fixtures** - `5d0415f` (test)
2. **Task 2: replace shell tests with oracle gate suite** - `20df96a` (test)

## Files Created/Modified
- `scripts/gen_oracle.py` - Added `gen_hdbscan` + `_hdbscan_{blob,tieheavy,nested}_design` helpers + per-metric/structure dispatch in `main()`.
- `crates/mlrs-algos/tests/hdbscan_test.rs` - Replaced 4 shell convention tests with the 9-family oracle gate suite (kept `defaults_equal`; removed the all-`-1` `fit_roundtrip`).
- `tests/fixtures/hdbscan_*.npz` - 22 committed oracle blobs (load_npz-safe: all arrays 4/8-byte float).

## Decisions Made
- **hdbscan `algorithm='generic'`** — uniform exact brute path supporting cosine + every metric, matching sklearn's dense computation (the default `'best'` BallTree rejects cosine).
- **epsilon knob cross-oracled against hdbscan 0.8.44** — sklearn 1.9.0's `epsilon_search`/`traverse_upwards` crashes on any cluster-merging epsilon tree; the fixture stores `labels_epsilon` from hdbscan (a sanctioned D-07 cross-oracle).
- **Tie-heavy = two integer lattices** (not one grid) so a real 2-cluster partition forms and the duplicate-row (R-9) label-share assert is meaningful.
- **single edge case** = `allow_single_cluster=True` + `min_samples=2` (a single homogeneous Gaussian has no density split → default eom yields all-noise).
- **precomputed centers empty** — sklearn rejects `store_centers` on a precomputed matrix; the `centers_match` gate naturally skips precomputed.
- **Wave-0 compile strategy** — `#[ignore]`-pending gate bodies avoid not-yet-built symbols by asserting oracle self-consistency (the matcher/`assert_close` plumbing the later wave reuses), so the suite compiles green today.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] hdbscan default algorithm rejects cosine**
- **Found during:** Task 1 (fixture generation)
- **Issue:** `hdbscan.HDBSCAN` default `algorithm='best'` routes to a BallTree that raises `ValueError: Unrecognized metric 'cosine'` (and only approximates the others).
- **Fix:** Forced `algorithm='generic'` (the exact brute path) in every hdbscan call, matching sklearn's dense `algorithm='auto'`/'brute' computation.
- **Files modified:** scripts/gen_oracle.py
- **Verification:** All 22 fixtures regenerate crash-free; cosine/precomputed fixtures carry valid hdb_labels/outlier_scores.
- **Committed in:** `5d0415f`

**2. [Rule 3 - Blocking] sklearn 1.9.0 epsilon_search crash**
- **Found during:** Task 1 (nested-density knob design)
- **Issue:** `sklearn.cluster.HDBSCAN` with `cluster_selection_epsilon>0` on any cluster-merging tree crashes inside `epsilon_search`/`traverse_upwards` (`TypeError: only 0-dimensional arrays can be converted to Python scalars`). A genuine sklearn-1.9.0 bug, not a design flaw — no crash-free merging-epsilon design exists in this env.
- **Fix:** Oracle the epsilon knob against `hdbscan` 0.8.44 instead (a sanctioned D-07 cross-oracle): the fixture stores `labels_epsilon` (hdbscan leaf+eps=1.0, which merges 4→2 crash-free) and `labels_leaf_default` (the matching hdbscan leaf base). The in-script Pitfall-5 divergence assertion gates `labels_leaf_default != labels_epsilon`.
- **Files modified:** scripts/gen_oracle.py
- **Verification:** Nested fixture writes with epsilon divergence asserted; `selection_knobs` reads `labels_epsilon`.
- **Committed in:** `5d0415f`

**3. [Rule 1 - Bug] Tie-heavy fixture was all-noise (trivial gate)**
- **Found during:** Task 2 (inventorying fixture geometries)
- **Issue:** The initial single 4×4 integer grid had no density variation → HDBSCAN labelled all 16 points noise, so the D-04 TRUE GATE would assert against an all-`-1` partition (trivially true, no real tie-break exercised).
- **Fix:** Redesigned the tie-heavy fixture as TWO well-separated 3×3 integer-lattice clusters (tie-heavy internal MST distances) with `min_cluster_size=3`; the duplicate row lives inside cluster A. Now yields 2 clusters with the dup pair sharing label 0.
- **Files modified:** scripts/gen_oracle.py, tests/fixtures/hdbscan_tieheavy_*.npz
- **Verification:** `tieheavy` fixture has 2 clusters, centroids (2,2), dup rows (0,7) → label (0,0).
- **Committed in:** `5d0415f` (amended)

**4. [Rule 1 - Bug] single edge case produced all-noise**
- **Found during:** Task 1 (edge-case generation)
- **Issue:** A single homogeneous Gaussian blob under default eom is rejected as noise (eom needs a density split to select a child), so the `single` fixture had no cluster.
- **Fix:** `allow_single_cluster=True` + `min_samples=2` for the single structure (both sklearn + hdbscan calls).
- **Files modified:** scripts/gen_oracle.py
- **Verification:** `single` fixture yields exactly 1 cluster (12/40 noise).
- **Committed in:** `5d0415f`

**5. [Rule 3 - Blocking] tiny edge case (n < min_cluster_size) errored**
- **Found during:** Task 1 (edge-case generation)
- **Issue:** The `tiny` design (3 samples) with default `min_samples=min_cluster_size=5` raises `ValueError: min_samples (5) must be at most the number of samples`.
- **Fix:** `min_samples=1` for the tiny structure so both oracles run and yield the all-noise labelling.
- **Files modified:** scripts/gen_oracle.py
- **Verification:** `tiny` fixture writes; labels all-`-1`.
- **Committed in:** `5d0415f`

---

**Total deviations:** 5 auto-fixed (3 blocking, 2 bug). All in the fixture generator — driven by sklearn-1.9.0 / hdbscan-0.8.44 environment behaviour and degenerate-design discovery, not scope creep. The test suite (Task 2) executed exactly as planned.
**Impact on plan:** All fixes necessary to produce non-trivial, crash-free oracle fixtures. The epsilon cross-oracle is the only oracle-source change and is explicitly D-07-sanctioned.

## Issues Encountered
- `cargo fmt -p mlrs-algos` reformats the ENTIRE crate, not just the target file. After formatting `hdbscan_test.rs` I reverted the ~55 out-of-scope reformatted files (kmeans.rs, spectral.rs, all other *_test.rs, etc.) via `git checkout --`, keeping only my authored test file, then formatted it standalone with `rustfmt --edition 2021`. Confirmed only `hdbscan_test.rs` is staged.

## Known Stubs
The `#[ignore]`-pending gate functions (24 of 27 tests) contain intentional Wave-0 placeholders: their bodies clone the oracle array into `got` and assert it against itself (exercising the matcher/`assert_close`/`best_mapping` plumbing) instead of comparing a fitted estimator that does not exist yet. Each carries a `// 15-NN: replace got with the fitted mlrs labels/...` comment and an `un-ignore in 15-NN` ignore reason. This is the planned Nyquist scaffold — the later HDBSCAN waves replace each placeholder with the real fit-vs-oracle comparison and drop `#[ignore]`. The 3 currently-runnable gates (`defaults_equal`, `build_validation`, `memory_gate`) are real and pass.

## Self-Check: PASSED
- Files exist: scripts/gen_oracle.py, hdbscan_test.rs, all 22 hdbscan_*.npz fixtures (euclidean/precomputed/tieheavy/nested spot-checked) — FOUND.
- Commits exist: 5d0415f, 20df96a — FOUND.
- `cargo test --features cpu --test hdbscan_test -- --list` lists all 9 gate families; `fit_roundtrip` count = 0; 7 pinned-noise matcher calls, 0 unpinned.

## Next Phase Readiness
- Wave-0 Nyquist gate is in place: every later HDBSCAN wave (15-03..15-06) has an automated `cargo test --features cpu --test hdbscan_test <fn>` gate to drive it green.
- The back-end port (15-NN) must: extend `Metric` (Manhattan/Cosine/Chebyshev/Minkowski/Precomputed), add `probabilities_`/`outlier_scores_`/`centroids_`/`medoids_` accessors + `store_centers`, implement the real fit, then for each gate swap the placeholder `got` for the fitted value and drop `#[ignore]`.
- Note for the back-end: the epsilon-knob oracle is `labels_epsilon` (hdbscan 0.8.44), NOT sklearn — sklearn 1.9.0 epsilon_search is broken in this env.

---
*Phase: 15-hdbscan*
*Completed: 2026-06-24*
