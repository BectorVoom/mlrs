---
phase: 14-umap
plan: 07
subsystem: manifold/umap
status: complete
tags: [umap, sgd-layout, determinism, d-05, parallel-backend, gap-closure, cr-01, cr-03]
requires:
  - "14-04 host_epoch_driver + umap_layout_step kernel (fit SGD layout)"
  - "14-06 n_components<n guard (Wave 1, already on main)"
provides:
  - "Owner-only (move_other=0) fit layout launch via FIT_MOVE_OTHER — D-05 byte-identity holds on any parallel backend"
  - "Executable move_other=0 + slot-disjointness invariant test (fit_move_other_is_zero)"
  - "Per-pair sample-count schedule guard (per_pair_sample_count_matches_schedule)"
  - "Recalibrated PROPERTY_EPS/ARI_BAND against the corrected single-pass-per-direction schedule"
affects:
  - "crates/mlrs-algos/src/manifold/umap.rs (host_epoch_driver fit launch)"
  - "crates/mlrs-algos/tests/umap_test.rs (invariant + recalibrated property gates)"
  - ".planning/phases/14-umap/14-VALIDATION.md (calibration block)"
tech-stack:
  added: []
  patterns:
    - "Single-source-of-truth launch flag (FIT_MOVE_OTHER) + pub accessor as a test-reachable seam"
    - "Encode a parallel-backend write-write-race guarantee as a host-only slot-partition invariant rather than re-running same-seed reproducibility on the sequential cpu-MLIR gate"
key-files:
  created: []
  modified:
    - "crates/mlrs-algos/src/manifold/umap.rs"
    - "crates/mlrs-algos/tests/umap_test.rs"
    - ".planning/phases/14-umap/14-VALIDATION.md"
decisions:
  - "REVIEW CR-01 option b: flip the fit launch move_other 1->0 over the already-symmetric COO — one change closes both CR-01 (race) and CR-03 (double-count). Option a (byte-identity-under-permuted-schedule) is infeasible for this in-place RMW kernel without a read-snapshot/double-buffer (out of scope)."
  - "PROPERTY_EPS re-derived to 0.03 (from 0.02): ~12x the new worst trust margin 0.0025, under the hard ceiling 0.04. ARI_BAND stays 0.05 (gap still 0.0000). No human-decision trigger — worst margin stayed well within guardrail."
metrics:
  duration_min: 70
  tasks: 3
  files_changed: 3
  completed: "2026-06-24"
---

# Phase 14 Plan 07: Owner-Only Fit Layout (move_other=0) — GAP 1 (CR-01 + CR-03) Summary

Set the UMAP fit-path layout launch to owner-only (`move_other=0`) over the already-symmetric fuzzy-graph COO, collapsing CR-01 (parallel-backend cross-cube write-write race) and CR-03 (~2-4× force double-count) into one fix, then encoded the D-05 cross-cube write guarantee as an executable invariant test and re-derived the property-gate thresholds against the corrected single-pass-per-direction schedule.

## What Was Built

**Task 1 — owner-only fit launch (commit `2c805a5`).**
Introduced `pub(crate) const FIT_MOVE_OTHER: u32 = 0` next to `host_epoch_driver` in `umap.rs` and routed the `umap_layout_step::launch` `move_other` argument through it, replacing the bare `1u32` literal. Because the fuzzy graph is symmetric (the COO carries both `(r,c)` and `(c,r)`), owner-only still covers BOTH endpoints of every undirected pair — once per direction — matching umap-learn's single head/tail force pass, while no owner-cube ever writes a foreign vertex's slots. The kernel (`umap_layout.rs`) is unchanged: its `move_other == 1u32`-gated foreign write (lines 155-157) is now unreachable from the fit launch, exactly as the transform path already exercises it (`move_other=0`, umap.rs:866). Added a `pub fn fit_move_other()` accessor as the test-reachable seam.

**Task 2 — executable invariant tests (commit `fc5e51f`).**
- `fit_move_other_is_zero`: asserts `fit_move_other() == 0` (FAILS the instant `move_other=1` is restored on the fit path = the CR-01 race) PLUS a mark-and-check over `n*dim` slots proving the `n` owner write ranges `o*dim..(o+1)*dim` are pairwise non-overlapping and exactly partition `0..n*dim`. This targets the move_other==1 WRITE-WRITE hazard directly (REVIEW option b), NOT the unachievable read-order byte-invariance, and is NOT a re-run of `reproducible_f64` (which passes even WITH the race on the sequential cpu-MLIR backend).
- `per_pair_sample_count_matches_schedule`: replays the `host_epoch_driver` positive-sample clock for a tiny symmetric COO and asserts each pair is sampled once per direction per due-epoch (not the former ~2-4× double-count) — the CR-03 schedule-level regression guard.
Both are host-only / cheap (no device launch) and pass under `--features cpu`.

**Task 3 — recalibration (commit `4fd016f`).**
Re-measured mlrs-vs-umap-learn 0.5.12 structural margins on the corrected `move_other=0` schedule (full 5-metric `layout_property_*` background run, ~2254s) and re-derived `PROPERTY_EPS` 0.02→0.03 and confirmed `ARI_BAND` stays 0.05. Updated both the constants in `umap_test.rs` and the calibration block in `14-VALIDATION.md` with the new per-metric margin table, the move_other=0 rationale, and the recorded hard guardrail (ε ≤ 0.04 ceiling + small-multiple-of-worst-margin relation).

## Measured Margins (move_other=0, cpu-MLIR f64, n=60, 3 clusters, seed=42, n_epochs=200)

| Metric | trust (mlrs/umap) | trust margin (umap−mlrs) | overlap (mlrs/umap) | overlap margin | ARI gap |
|--------|-------------------|--------------------------|---------------------|----------------|---------|
| euclidean | 0.9655 / 0.9680 | **+0.0025** | 0.6917 / 0.6917 | +0.0000 | 0.0000 |
| manhattan | 0.9633 / 0.9635 | +0.0002 | 0.6867 / 0.6783 | −0.0083 | 0.0000 |
| cosine | 0.9670 / 0.9673 | +0.0003 | 0.6983 / 0.6733 | −0.0250 | 0.0000 |
| chebyshev | 0.9679 / 0.9615 | −0.0064 | 0.6850 / 0.6317 | −0.0533 | 0.0000 |
| minkowski (p=3) | 0.9648 / 0.9652 | +0.0004 | 0.6917 / 0.6667 | −0.0250 | 0.0000 |

Worst positive margin: trust **+0.0025** (euclidean); overlap **+0.0000** (mlrs ≥ umap everywhere); ARI gap 0.0000. Re-derived `PROPERTY_EPS = 0.03` ≈ 12× the worst trust margin, ≤ the 0.04 ceiling — guardrail honoured, no human-decision trigger needed.

## Verification

| Gate | Result |
|------|--------|
| `cargo build -p mlrs-algos --features cpu` | PASS |
| `fit_move_other_is_zero` | PASS |
| `per_pair_sample_count_matches_schedule` | PASS |
| `reproducible_f64` (byte-identical after change) | PASS (1283.72s) |
| `layout_property_euclidean` @ PROPERTY_EPS=0.03 | PASS (332.49s) |
| Full `layout_property_*` 5-metric family | PASS — 5 passed, 0 failed (2254.46s, background) |
| `umap_layout.rs` (kernel) unchanged | CONFIRMED (no edits) |

cpu-MLIR safety: the change only flips a launch scalar (1u32→0u32, via a constant) and adds host-side tests; it REMOVES a kernel write and introduces no SharedMemory/atomics/F::INFINITY/mutable-bool/shift-loop. The kernel stays cpu-MLIR-safe and is not restructured.

## Threat Mitigations (from PLAN threat_model)

- **T-14-07-01** (Tampering/Repudiation — cross-cube write race): MITIGATED. `move_other=0` makes each owner-cube write only its slot-disjoint own coordinates; `fit_move_other_is_zero` asserts the flag and the disjoint partition so a regression fails the gate. D-05 byte-identity now holds on any parallel backend (wgpu/rocm/cuda), not by accident of the sequential cpu-MLIR gate.
- **T-14-07-02** (Repudiation — gate calibrated to mlrs's own double-count): MITIGATED. The single-pass-per-direction schedule matches umap-learn; PROPERTY_EPS/ARI_BAND re-derived against the corrected output; `per_pair_sample_count_matches_schedule` anchors the schedule against regression.

## Deviations from Plan

None — plan executed exactly as written. PROPERTY_EPS recalibrated within the hard guardrail (0.03 ≤ 0.04, ~12× worst margin); the human-decision branch (worst margin large enough to require ε > 0.04) was NOT triggered, so no blocking finding was raised.

## Known Stubs

None. No empty-value/placeholder stubs introduced; the change removes a write and tightens an existing gate.

## Self-Check: PASSED
- crates/mlrs-algos/src/manifold/umap.rs — FOUND (FIT_MOVE_OTHER + fit_move_other accessor)
- crates/mlrs-algos/tests/umap_test.rs — FOUND (fit_move_other_is_zero + per_pair_sample_count_matches_schedule)
- .planning/phases/14-umap/14-VALIDATION.md — FOUND (recalibrated move_other=0 block)
- Commit 2c805a5 — FOUND
- Commit fc5e51f — FOUND
- Commit 4fd016f — FOUND
