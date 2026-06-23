---
phase: 14-umap
plan: 01
subsystem: testing
tags: [umap, manifold, oracle-fixtures, umap-learn, cubecl, property-gate, knn-graph]

# Dependency graph
requires:
  - phase: 13-knn-graph-primitive-feasibility-keystone
    provides: knn_graph::Metric (5-variant canonical shape mirrored by Umap::Metric)
  - phase: 12 (manifold convention shell)
    provides: Umap<F, S> builder + typestate shell (fit/transform/embedding surface)
provides:
  - "Umap::Metric extended to 5 variants (Euclidean/Manhattan/Cosine/Chebyshev/Minkowski{p}) mirroring knn_graph::Metric"
  - "Empty umap_internals.rs (Plan 02) + umap_init.rs (Plan 03) module stubs wired pub(crate) in manifold/mod.rs (file-disjoint Wave 2)"
  - "gen_umap_{fuzzy,spectral,ab,layout,transform} generators in scripts/gen_oracle.py"
  - "21 committed umap-learn 0.5.12 oracle fixtures: 5 metrics × {fuzzy,spectral,layout,transform} + 1 metric-independent ab fixture (all f64)"
  - "RED-by-design value/property/reproducibility/transform harness + host helpers (trustworthiness/knn_overlap/downstream_ari) in umap_test.rs"
affects: [14-02 (smooth-kNN/membership/union), 14-03 (a/b LM + spectral/random init), 14-04 (layout kernel + real fit + threshold calibration), 14-05 (transform)]

# Tech tracking
tech-stack:
  added: [umap-learn==0.5.12 (build-time-only /tmp venv oracle, never a runtime/manifest dep)]
  patterns:
    - "Per-stage × per-metric oracle fixture matrix dumping umap-learn's OWN internals (never recomputed)"
    - "Centralized gate_f64() capability skip helper (skip_f64_with_log verbatim) for every device test"
    - "Property-gate host metrics (trustworthiness/kNN-overlap/ARI) computed in-repo, no sklearn at test time"
    - "Placeholder calibration consts (PROPERTY_EPS/ARI_BAND) with TODO marker for Plan 04 calibration run"

key-files:
  created:
    - crates/mlrs-algos/src/manifold/umap_internals.rs
    - crates/mlrs-algos/src/manifold/umap_init.rs
    - tests/fixtures/umap_{fuzzy,spectral,layout,transform}_{euclidean,manhattan,cosine,chebyshev,minkowski}_f64.npz
    - tests/fixtures/umap_ab_f64.npz
  modified:
    - crates/mlrs-algos/src/manifold/umap.rs
    - crates/mlrs-algos/src/manifold/mod.rs
    - scripts/gen_oracle.py
    - crates/mlrs-algos/tests/umap_test.rs

key-decisions:
  - "Umap::Metric drops the Eq derive (Minkowski{p:f64} carries a non-Eq f64); hyperparams_eq compares via PartialEq"
  - "Centralized the verbatim skip_f64_with_log gate into a gate_f64() helper rather than copy-paste per test (no un-gated f64 device path)"
  - "Fixtures are f64-only (the cpu value gate); deterministic stages value-gate to <=1e-5 in host f64 without device-reduction drift"
  - "spectral_layout fixture uses the symmetric (graph.max(graph.T)) affinity on an n=60 connected design (single-component path matches the laplacian+eig prim)"
  - "Property/threshold consts left as TODO placeholders — no invented thresholds before Plan 04's calibration run (RESEARCH Q4)"

patterns-established:
  - "RED-by-design harness: tests reference the real fit/transform + committed fixtures, compile + run, FAIL at runtime against the zeros shell until Plans 02-05 land real stages"
  - "Float-encoded indices + metric-tag-in-filename fixture convention (gen_knn_metric precedent) so load_npz only sees 4/8-byte float arrays"

requirements-completed: [UMAP-01, UMAP-02, UMAP-03, UMAP-04]

# Metrics
duration: ~25min
completed: 2026-06-23
---

# Phase 14 Plan 01: UMAP Verification Substrate (Nyquist Wave 0) Summary

**Stood up the committed umap-learn 0.5.12 per-stage × per-metric oracle fixtures, the 5-variant Umap::Metric mirroring the Phase-13 KNN prim, the two file-disjoint Plan-02/03 module stubs, and a RED-by-design value/property/reproducibility/transform test harness that compiles and runs against the zeros shell.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 3
- **Files modified:** 4 modified + 23 created (2 stub modules + 21 fixtures)

## Accomplishments
- Extended `Umap::Metric` to the full 5-variant set (`Euclidean`/`Manhattan`/`Cosine`/`Chebyshev`/`Minkowski{p:f64}`) mirroring `knn_graph::Metric` exactly — no lossy conversion at the future KNN call site; dropped `Eq` (non-`Eq` `f64` payload), kept `PartialEq` so `hyperparams_eq` still compiles.
- Created the two EMPTY `umap_internals.rs` (Plan 02) and `umap_init.rs` (Plan 03) stubs, wired `pub(crate)` in `manifold/mod.rs`, so the two Wave-2 plans fill their own file WITHOUT both editing `mod.rs` (parallel-safe).
- Added 5 `gen_umap_*` generator families to `scripts/gen_oracle.py` dumping umap-learn 0.5.12's OWN internals (`smooth_knn_dist`, `fuzzy_simplicial_set`, `find_ab_params`, `spectral_layout`, `UMAP.fit_transform`/`transform`) — never recomputed in numpy.
- Generated + committed 21 f64 `.npz` oracle blobs (5 metrics × fuzzy/spectral/layout/transform + 1 metric-independent ab) from a `/tmp` venv with `umap-learn==0.5.12`.
- Built the RED-by-design harness in `umap_test.rs`: 7 test families (`smooth_knn`/`fuzzy_union`/`spectral_init`/`ab_fit`/`layout_property`/`reproducible`/`transform_property`), 3 in-repo host property-gate helpers (`trustworthiness`/`knn_overlap`/`downstream_ari`), every device test gated via `gate_f64()`. Harness compiles + runs; value/property tests FAIL with explicit "RED until Plan NN" messages; the 4 Phase-12 convention tests + reproducibility stay GREEN.

## Task Commits

1. **Task 1: Extend Metric enum + create umap_internals/umap_init stubs** - `5edfbaf` (feat)
2. **Task 2: Add gen_umap_* generators + commit 5-metric per-stage fixtures** - `a29b1e5` (feat)
3. **Task 3: RED value/property/reproducibility/transform harness** - `f437443` (test)

**Plan metadata:** (final docs commit below)

## Files Created/Modified
- `crates/mlrs-algos/src/manifold/umap.rs` - `Metric` enum extended to 5 variants, `Eq` dropped
- `crates/mlrs-algos/src/manifold/mod.rs` - declared `pub(crate) mod umap_init/umap_internals`
- `crates/mlrs-algos/src/manifold/umap_internals.rs` - EMPTY stub (Plan 02's home)
- `crates/mlrs-algos/src/manifold/umap_init.rs` - EMPTY stub (Plan 03's home)
- `scripts/gen_oracle.py` - `gen_umap_fuzzy/spectral/ab/layout/transform` + `main()` dispatch
- `tests/fixtures/umap_*.npz` - 21 committed umap-learn 0.5.12 oracle blobs
- `crates/mlrs-algos/tests/umap_test.rs` - RED harness + host property-gate helpers

## Decisions Made
- `Umap::Metric` drops `Eq` (Minkowski `f64` payload is non-`Eq`); `PartialEq` retained for `hyperparams_eq`.
- Centralized `skip_f64_with_log` into a `gate_f64()` helper (every device test calls it) rather than verbatim copy-paste — same no-un-gated-f64-path guarantee, less duplication.
- Fixtures are f64-only (the cpu value gate); host-f64 deterministic stages match umap-learn without device-reduction drift.
- Spectral fixture symmetrizes via `graph.max(graph.T)` on an n=60 connected design (single-component laplacian+eig path the mlrs prim reproduces).
- Calibration consts (`PROPERTY_EPS`/`ARI_BAND`) left as TODO placeholders — no invented thresholds before Plan 04's calibration run (RESEARCH Q4 / Spike flag item 2).

## Deviations from Plan

None - plan executed exactly as written. The `/tmp` venv with `umap-learn==0.5.12` (the only build-time pip pin, OK-verdict in the RESEARCH Package Legitimacy Audit) installed and imported cleanly, so no package-legitimacy checkpoint was triggered.

## Issues Encountered
- `fuzzy_simplicial_set(..., return_dists=True)` returns a 4-tuple `(graph, sigmas, rhos, dists)` in umap-learn 0.5.12, not the 3-tuple the first draft assumed — corrected the unpack before generating fixtures (caught during fixture generation, not committed wrong).

## Known Stubs

The RED-by-design harness intentionally uses placeholder produced-values (`vec![0.0; ...]` for sigmas/vals, `(0.0, 0.0)` for a/b, the zeros embedding from the shell fit) inside the value-gate tests. These are NOT goal-blocking stubs — they are the deliberate RED state Wave 0 establishes, each tagged with the resolving plan in the assertion message:
- `smooth_knn_*` / `fuzzy_union_*` placeholders → resolved by Plan 02 (`umap_internals`).
- `spectral_init_*` / `ab_fit` placeholders → resolved by Plan 03 (`umap_init`).
- `layout_property_*` / `reproducible_*` → resolved by Plan 04 (real layout + threshold calibration).
- `transform_property_*` → resolved by Plan 05 (real transform).
- `umap_internals.rs` / `umap_init.rs` are intentionally empty module stubs (file ownership pre-creation for parallel-safe Wave 2).
- `PROPERTY_EPS` / `ARI_BAND` are TODO calibration placeholders (Plan 04 overwrites + records in 14-VALIDATION.md).

## User Setup Required
None - no external service configuration required. (Fixture regeneration needs a `/tmp` venv with `numpy scipy scikit-learn umap-learn==0.5.12`, but the blobs are committed; CI never runs the generator.)

## Next Phase Readiness
- Plans 02 and 03 are file-disjoint (own `umap_internals.rs` / `umap_init.rs`; neither edits `mod.rs`) → safe to run in parallel in Wave 2.
- Every deterministic stage has its committed value-gate fixture + RED test ready to turn GREEN.
- Plan 04 must run the threshold calibration and replace `PROPERTY_EPS`/`ARI_BAND`, recording the calibrated numbers in `14-VALIDATION.md`.
- `Umap::Metric → knn_graph::Metric` mapping fn is intentionally NOT added here (Plan 04 adds it at the KNN call site).

## Self-Check: PASSED

- All created files exist on disk (2 module stubs, 21 fixtures, modified umap.rs/mod.rs/gen_oracle.py/umap_test.rs).
- All 3 task commits present in git history (`5edfbaf`, `a29b1e5`, `f437443`).
- `cargo build -p mlrs-algos --features cpu` exits 0; `cargo test ... --test umap_test --no-run` exits 0 (harness compiles); value/property tests RED-by-design at runtime as intended.

---
*Phase: 14-umap*
*Completed: 2026-06-23*
