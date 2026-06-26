---
phase: 15-hdbscan
plan: 04
subsystem: cluster
tags: [hdbscan, condensed-tree, excess-of-mass, eom, leaf, stability, labelling, probabilities, union-find, sklearn-oracle]

# Dependency graph
requires:
  - phase: 15-03
    provides: "single-linkage hierarchy (make_single_linkage + SingleLinkageEdge), dense Variant-A precomputed MST path, Metric::Precomputed fit wiring, BuildError variants"
  - phase: 15-01
    provides: "mlrs_core::best_match_accuracy_pinned_noise (-1-pinned exact-label gate)"
  - phase: 15-02
    provides: "committed hdbscan_*.npz oracle fixtures + the #[ignore]-pending gate suite"
provides:
  - "condense.rs: bfs_from_hierarchy + condense_tree (runt-prune by min_cluster_size, lambda=1/distance with the INFTY branch)"
  - "stability.rs: compute_stability + max_lambdas"
  - "select.rs: get_clusters (EoM/leaf/epsilon/max_cluster_size) + do_labelling (TreeUnionFind) + get_probabilities"
  - "Hdbscan probabilities_ fitted field + Fitted::probabilities() accessor"
  - "allow_single_cluster builder field"
  - "precomputed fit end-to-end: exact labels (HDBS-02) + ≤1e-5 probabilities (HDBS-01)"
affects: [15-05, 15-06, hdbscan, glosh, centers]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Verbatim line-for-line host port of sklearn _hdbscan/_tree.pyx (condense/stability/select), the No-Analog-Found path"
    - "Two distinct union-finds in one estimator: single_linkage::UnionFind (fresh-label per merge) vs select::TreeUnionFind (union-by-rank + path compression)"
    - "Precomputed-path test routing: feature fixtures fitted via host-built euclidean distance matrix + Metric::Precomputed (precomputed-euclidean == euclidean for labels/probs)"

key-files:
  created:
    - crates/mlrs-algos/src/cluster/hdbscan/condense.rs
    - crates/mlrs-algos/src/cluster/hdbscan/stability.rs
    - crates/mlrs-algos/src/cluster/hdbscan/select.rs
  modified:
    - crates/mlrs-algos/src/cluster/hdbscan.rs
    - crates/mlrs-algos/tests/hdbscan_test.rs

key-decisions:
  - "Added allow_single_cluster builder field (Rule 2) — the single edge fixture needs it and it is a faithful sklearn _get_clusters/_do_labelling input"
  - "labels_alpha knob deferred to 15-05: it is intrinsically a feature-path (Variant-B) alpha gate (Pitfall 2); sklearn itself partitions precomputed+alpha differently from feature+alpha"
  - "probabilities() accessor returns Option (mirrors single_linkage()) — None on the not-yet-wired feature-metric path, Some on precomputed"

patterns-established:
  - "Pattern: host tree back-end is metric-agnostic — it consumes a single-linkage hierarchy, so any path that produces one (precomputed now, feature in 15-05) reuses condense->stability->select->labelling->probabilities unchanged"
  - "Pattern: degenerate guard — an empty condensed tree (no internal cluster) short-circuits to all-noise labels + all-0 probabilities, matching sklearn without entering the condensed-tree assumptions"

requirements-completed: [HDBS-01, HDBS-02]

# Metrics
duration: 70min
completed: 2026-06-24
status: complete
---

# Phase 15 Plan 04: HDBSCAN condensed-tree back-end (condense/stability/select/labelling/probabilities) Summary

**Verbatim host port of sklearn `_tree.pyx` — condense by `min_cluster_size`, compute EoM stabilities, run eom/leaf/epsilon/max_cluster_size selection, label points (`-1`=noise) and compute `probabilities_` — making the precomputed `fit` produce exact labels (HDBS-02) and ≤1e-5 probabilities (HDBS-01).**

## Performance

- **Duration:** ~70 min
- **Started:** 2026-06-24T04:20Z (approx)
- **Completed:** 2026-06-24T05:29Z
- **Tasks:** 2
- **Files modified:** 6 (3 created, 2 modified, 1 deferred-items log)

## Accomplishments
- `condense.rs` — `bfs_from_hierarchy` + `condense_tree`: runt-prunes the 15-03 single-linkage hierarchy by `min_cluster_size` (NOT `min_samples`, Pitfall 4), `lambda = 1/distance` with the `INFTY` branch on `distance==0`, the exact four-way left/right keep/fall-out branches. Matches sklearn `_condense_tree` bit-for-bit on the n=8 ground-truth.
- `stability.rs` — `compute_stability` (`(lambda - births[parent]) * cluster_size`, `births[root]=0`) + `max_lambdas` (per-parent death-lambda sweep), matching sklearn's exact accumulation (e.g. `{8:0.8, 9:2.766667, 10:2.766667}` at mcs=2).
- `select.rs` — `get_clusters` (Excess-of-Mass + leaf traversals, the `epsilon_search`/`traverse_upwards` merge for `cluster_selection_epsilon>0`, the `max_cluster_size` deselect bound), `do_labelling` (a `TreeUnionFind` over the non-cluster edges → cluster label, else `-1`), and `get_probabilities` (`min(lambda,max_lambda)/max_lambda`; `1.0` on `max_lambda==0` or non-finite lambda).
- Wired the `Metric::Precomputed` `fit` branch end-to-end: condense → stability → select → labelling → probabilities; added the `probabilities_` device-resident field + `Fitted::probabilities()` accessor and the `allow_single_cluster` builder knob.
- Un-ignored + wired the real precomputed-path fit for `labels_match_sklearn` (precomputed), `probabilities_match`, `selection_knobs` (eom/leaf/maxcluster/epsilon), `edge_cases` (allnoise/single/tiny), and `tie_break_exact` — all green on cpu(f64) + f32.

## Task Commits

1. **Task 1: Port _condense_tree + _compute_stability** - `7ce3fed` (feat) — condense.rs + stability.rs + module decls + ground-truth tests
2. **Task 2: Port selection/labelling/probabilities; wire precomputed labels** - `8b0ceab` (feat) — select.rs + fit wiring + probabilities_ field/accessor + allow_single_cluster + test gates

_TDD: both tasks carried `tdd="true"`; the condense/stability ground-truth tests and the un-ignored oracle gates are the behavior assertions written alongside the ports._

## Files Created/Modified
- `crates/mlrs-algos/src/cluster/hdbscan/condense.rs` (created) — `bfs_from_hierarchy` + `condense_tree` + `CondensedNode`
- `crates/mlrs-algos/src/cluster/hdbscan/stability.rs` (created) — `compute_stability` + `max_lambdas`
- `crates/mlrs-algos/src/cluster/hdbscan/select.rs` (created) — `get_clusters` (eom/leaf/epsilon/max) + `do_labelling` + `get_probabilities` + `TreeUnionFind` + `SelectionMethod`
- `crates/mlrs-algos/src/cluster/hdbscan.rs` (modified) — `condense`/`select`/`stability` module decls; `probabilities_` field + `probabilities()` accessor; `allow_single_cluster` builder field; `tree_to_labels` host helper; precomputed `fit` branch now produces labels + probabilities
- `crates/mlrs-algos/tests/hdbscan_test.rs` (modified) — condense/stability/max_lambdas ground-truth tests; precomputed-path fit helpers; un-ignored + wired labels/probabilities/selection_knobs/edge_cases/tie_break gates; `labels_alpha` deferral marker
- `.planning/phases/15-hdbscan/deferred-items.md` (created) — out-of-scope log

## Decisions Made
- **Added `allow_single_cluster` builder field (Rule 2 — missing critical functionality).** The `single` edge fixture (a homogeneous blob with no density split) only matches sklearn with `allow_single_cluster=True`, and the field is a genuine input to sklearn's `_get_clusters`/`_do_labelling`. Threaded through `new()`/`build()`/`into_builder()`/`hyperparams_eq()`.
- **`probabilities()` accessor returns `Option<Vec<F>>`** (mirrors `single_linkage()`), since the feature-metric path (15-05) produces no probabilities yet — `None` there, `Some` after a precomputed fit.
- **`labels_alpha` selection knob deferred to 15-05** (see Deviations).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added `allow_single_cluster` builder field**
- **Found during:** Task 2 (edge_cases / single fixture)
- **Issue:** The plan's edge-case gate includes the `single` fixture, generated with sklearn `allow_single_cluster=True`. Without the field the homogeneous-blob case yields all-noise and the gate cannot pass; the field is also a required input to the faithful `_get_clusters`/`_do_labelling` port.
- **Fix:** Added `allow_single_cluster` to the struct + builder (setter, `build`, `into_builder`, `new`, `hyperparams_eq`) and threaded it into `get_clusters`/`do_labelling`.
- **Files modified:** crates/mlrs-algos/src/cluster/hdbscan.rs, crates/mlrs-algos/src/cluster/hdbscan/select.rs
- **Verification:** `edge_cases_f32`/`edge_cases_f64` (single sub-case) pass.
- **Committed in:** 8b0ceab (Task 2 commit)

**2. [Rule 3 - Blocking] `labels_alpha` selection knob deferred to 15-05 (feature-path Variant-B alpha)**
- **Found during:** Task 2 (selection_knobs)
- **Issue:** The nested fixture's `labels_alpha` oracle (alpha=0.5) was generated on the FEATURE path, whose alpha placement is Variant B (`pair_distance /= alpha`, RAW core). The precomputed path wired in 15-04 uses Variant A (divide the whole matrix BEFORE core). sklearn ITSELF resolves these to different partitions (precomputed+α=0.5 → 2 clusters; feature+α=0.5 → 3), so the knob is intrinsically the feature-metric (15-05) gate and cannot pass via the precomputed path. mlrs's precomputed output matches sklearn's precomputed alpha result exactly.
- **Fix:** `selection_knobs` exercises the four precomputed-reproducible knobs (eom/leaf/maxcluster/epsilon); added the `#[ignore]`d `selection_knob_alpha_feature_path` marker with an `un-ignore in 15-05` note so the D-09 alpha knob is not silently dropped.
- **Files modified:** crates/mlrs-algos/tests/hdbscan_test.rs
- **Verification:** `selection_knobs_f32`/`selection_knobs_f64` pass on the four reproducible knobs; alpha confirmed as a 15-05 responsibility against the 15-05 plan (Variant-B front-end, Pitfall 2).
- **Committed in:** 8b0ceab (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 missing critical, 1 blocking/scope-sequencing)
**Impact on plan:** Both are correctness/sequencing-driven, no scope creep. The alpha deferral is an honest path-ownership boundary (15-05 owns the feature/Variant-B path) explicitly tracked, not a silent drop.

## Deferred Issues
- **`cargo clippy --features cpu` fails in `mlrs-kernels`** — pre-existing, unrelated crate (it has no `cpu` feature; 27 warnings + 1 error independent of HDBSCAN). `cargo build` / `cargo test --features cpu -p mlrs-algos` are clean with zero warnings on the new code. Logged in deferred-items.md.

## Known Stubs
None — `condense.rs`/`stability.rs`/`select.rs` are complete ports (no `todo!`/`unimplemented!`/placeholder). The feature-metric `fit` branch remaining all-`-1` is the documented 15-05 boundary, not a stub.

## Issues Encountered
- The `selection_knobs` alpha sub-case initially failed (acc=0.75, 2 vs 3 clusters). Root-caused to the Variant-A/Variant-B alpha-placement divergence (Pitfall 2) by reproducing both paths in sklearn; resolved by scoping the precomputed gate to the alpha-independent knobs and deferring `labels_alpha` to 15-05.

## Threat Flags
None — no new trust boundary (pure host scalar tree over the 15-03 hierarchy; no device launch, no untrusted input). T-15-04-MIS (min_samples/min_cluster_size swap) is satisfied: `condense.rs`/`select.rs` use `min_cluster_size` only in code (no `min_samples` usage). T-15-04-INF (NaN/inf probabilities) is satisfied: `get_probabilities` returns `1.0` on `max_lambda==0`/non-finite lambda, verified by the ≤1e-5 probabilities gate. T-15-04-SC: no new crate dependency.

## Next Phase Readiness
- The full host tree back-end is proven on the precomputed path (exact labels + ≤1e-5 probabilities + all reproducible selection knobs + edge cases). 15-05 can now feed it feature-metric single-linkage hierarchies (Variant A for cosine, Variant B for the other four) and reuse `tree_to_labels` unchanged.
- 15-05 must un-ignore: the 5 feature-metric `labels_match_sklearn_*` gates and the `selection_knob_alpha_feature_path` marker (Variant-B alpha).
- 15-06 (centers) and the GLOSH plan (outlier_scores) own the remaining ignored gates.

## Self-Check: PASSED

- condense.rs / stability.rs / select.rs — FOUND on disk
- 15-04-SUMMARY.md — FOUND on disk
- Commits 7ce3fed, 8b0ceab — FOUND in git history
- `cargo test --features cpu --test hdbscan_test` — 23 passed, 0 failed, 15 ignored (feature-metric/centers/glosh/alpha owned by later plans)

---
*Phase: 15-hdbscan*
*Completed: 2026-06-24*
