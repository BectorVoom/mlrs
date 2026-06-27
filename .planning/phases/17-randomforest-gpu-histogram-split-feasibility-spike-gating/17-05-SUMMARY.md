---
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
plan: 05
subsystem: testing
tags: [randomforest, decision-tree, cpu-mlir, cubecl, feasibility-spike, verdict, sklearn-witness, sparsetreenode]

# Dependency graph
requires:
  - phase: 17-02
    provides: A1/A4 cpu-MLIR kernel evidence (GATHER histogram lowering, split-find argmax + tie VALUE assert) + indexing-shape findings
  - phase: 17-03
    provides: A5 Tier-1 sklearn witness (clf gini + reg squared_error + adversarial, exact structure + ≤1e-5 leaf values)
  - phase: 17-04
    provides: A3 per-tree cost benchmark (64-vs-128 bins wall-clock + samples scaling sweep + frontier-memory observation)
provides:
  - "VERDICT.md — the GATING artifact: A1–A5 evaluated on hard evidence, explicit GO decision, two-tier stochastic-gate convention, finalized SparseTreeNode contract + cuML divergence note"
  - "Durable spike evidence dirs .planning/spikes/003–006-*/ (verbatim proven source) + appended MANIFEST.md Phase-17 block"
  - "Human-approved GO verdict gating Phases 18–21"
affects: [phase-18-production-prims, phase-19-randomforest-ensemble, phase-20-fil, phase-21-treeshap]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-tier stochastic-gate convention (Tier-1 deterministic injected-fixed-index core D-07 + Tier-2 ensemble predictive band D-08) as milestone-wide tree/ensemble correctness standard"
    - "SparseTreeNode flat node contract finalized (leaf = colid==-1, right = left_child+1, value = offset into shared leaf buffer)"
    - "Durable spike evidence artifact pattern: verbatim copy of proven test source into .planning/spikes/NNN-*/ (mirrors Phase 13, D-01)"

key-files:
  created:
    - .planning/phases/17-randomforest-gpu-histogram-split-feasibility-spike-gating/VERDICT.md
    - .planning/spikes/003-gather-histogram-lower/
    - .planning/spikes/004-seed-from-first-split-find/
    - .planning/spikes/005-relabel-partition/
    - .planning/spikes/006-tier1-decisiontree-witness/
  modified:
    - .planning/spikes/MANIFEST.md

key-decisions:
  - "GO verdict (human-approved): A1/A2/A4/A5 PASS on hard evidence, A3 tractable (sub-second/tree); serial tree chain Phase 18→19→20→21 proceeds"
  - "No D-06 ADJUST rung applied — D-06 levers (128→64 bins, frontier-only histogramming, shallower max_depth) recorded as data-backed headroom for Phase 18, not needed to clear the gate"
  - "SparseTreeNode contract FINALIZED with deliberate cuML divergence: mlrs leaf = colid==-1 (NOT cuML left_child==-1); Phase 20 FIL binds to mlrs convention"
  - "Two-tier stochastic-gate convention adopted as milestone-wide standard for every tree/ensemble phase"

patterns-established:
  - "Feasibility verdict structure mirrors Phase 13 VERIFICATION.md: A1–A5 evidence table + explicit GO/ADJUST/ABORT + downstream contract"
  - "Tier-1 = element-wise sklearn match on injected fixed indices; Tier-2 = predictive band (~0.02–0.05) for RNG-divergent forests"

requirements-completed: [TREE-01]

# Metrics
duration: 8min
completed: 2026-06-27
status: complete
---

# Phase 17 Plan 05: Feasibility Verdict & Spike Wrap-Up Summary

**Human-approved GO verdict for the mlrs tree chain — A1–A5 abort signals all cleared on hard evidence, SparseTreeNode contract finalized (leaf = colid==-1, diverging from cuML), two-tier stochastic-gate convention adopted, and the proven spike source preserved verbatim as durable evidence.**

## Performance

- **Duration:** 8 min (continuation finalization; authoring + wrap-up completed in the pre-checkpoint autonomous pass)
- **Completed:** 2026-06-27
- **Tasks:** 3 (2 autonomous pre-committed + 1 human-verify gate, now approved)
- **Files modified:** 6 (VERDICT.md, 4 spike dirs, MANIFEST.md)

## Accomplishments

- **VERDICT.md authored and human-locked: GO.** All five abort signals evaluated against the concrete evidence Plans 02/03/04 produced (not "it compiled"):
  - **A1 (lowering)** PASS — `tree_gather_histogram` lowered cleanly on the first attempt under the 2D guarded `ABSOLUTE_POS_X/Y` shape; non-zero correct read-back on f64 + f32.
  - **A2 (binning sort)** PASS — host quantile bin-edge pre-pass (D-10) consumed as binned u32; no device sort/scan kernel exists anywhere in the spike.
  - **A3 (cost)** PASS (tractable) — sub-second per tree at ≈1000×20×depth-8; 64-vs-128 delta 6.68× (f32) / 2.57× (f64); top-end 500→1000 super-quadratic step traced to cumulative-node scratch (the frontier-only headroom lever), not a pathological blow-up.
  - **A4 (split-find argmax)** PASS — seed-from-candidate-0, u32 admit/better flags, no `F::INFINITY`/mutable bool; VALUE-asserts the deliberate gain tie resolves to lowest feature then lowest bin.
  - **A5 (sklearn correctness)** PASS — Tier-1 witness: clf (gini), reg (squared_error), and adversarial (identical-columns tie + separable target) all GREEN on cpu f64 + f32, exact structure + ≤1e-5 leaf values.
- **SparseTreeNode contract FINALIZED** with the binding cuML divergence note: `{ colid: i32, threshold: F, left_child: i32, value: i32 }`, leaf sentinel `colid == -1`, `right = left_child + 1`, `value` = offset into a shared leaf-value buffer (multiclass-uniform). Phase 20 FIL must bind to the mlrs `colid == -1` convention, NOT cuML's `left_child == -1`.
- **Two-tier stochastic-gate convention** documented as the milestone-wide standard (Tier-1 deterministic injected-fixed-index core D-07; Tier-2 ensemble predictive band ~0.02–0.05 D-08, governs Phase 19).
- **Durable spike evidence preserved** — `.planning/spikes/003–006-*/` hold verbatim copies of the proven probe/witness source; MANIFEST.md appended with the Phase-17 block (rows 003–006 with verdicts/tags). Live `crates/mlrs-backend/tests/tree_*` files remain the runnable gate.
- **Human gate (Task 3) resolved: APPROVED.** The orchestrator independently re-ran the gates after the executor returned — `tree_spike_probes` 8/8, `tree_witness` 8/8 (cpu f64+f32), `tree_bench` A3 sub-second/tree — confirming GO. The verdict is locked; Phase 18 may proceed.

## Task Commits

1. **Task 1: Author VERDICT.md (A1–A5 + GO decision + two-tier convention + SparseTreeNode contract)** — `739fb3b` (docs)
2. **Task 2: Spike wrap-up — verbatim source into .planning/spikes/003–006-*/ + MANIFEST.md** — `81a1672` (docs)
3. **Task 3: Lock the GO/ADJUST/ABORT verdict — human-verify gate** — APPROVED (no code commit; gate resolution)

**Plan metadata:** committed with this summary + STATE.md + ROADMAP.md.

## Files Created/Modified

- `.planning/phases/17-.../VERDICT.md` - The gating artifact: A1–A5 evaluation, GO decision, two-tier convention, finalized SparseTreeNode contract + cuML divergence note, Phase-18 caveats, evidence index
- `.planning/spikes/003-gather-histogram-lower/` - Verbatim GATHER histogram kernel + probe source (A1 evidence)
- `.planning/spikes/004-seed-from-first-split-find/` - Verbatim split-find kernel + probe source (A4 evidence)
- `.planning/spikes/005-relabel-partition/` - Verbatim relabel/partition kernel + probe source (D-02 evidence)
- `.planning/spikes/006-tier1-decisiontree-witness/` - Verbatim Tier-1 sklearn witness + bench source (A5/A3 evidence)
- `.planning/spikes/MANIFEST.md` - Appended Phase-17 block (idea + design decisions + spikes table rows 003–006)

## Decisions Made

- **GO, no ADJUST rung** — the correctness/feasibility quartet (A1/A2/A4/A5) is fully cleared and A3 is sub-second per tree, so no D-06 lever is needed to pass. The levers are recorded as data-backed headroom for Phase 18 in priority order (fewer bins → frontier-only histogramming → shallower max_depth → defer as last resort).
- **mlrs SparseTreeNode deliberately diverges from cuML** — leaf = `colid == -1` (treelite/FIL iterative-traversal convention) rather than cuML's `left_child == -1`; the `right = left_child + 1` rule is shared. This is binding for Phase 20.
- **Forests gated on predictive quality only** — SplitMix64 ≠ MT19937 makes element-wise forest equality impossible by construction, so Tier-2 uses an absolute accuracy/R² band (~0.02–0.05).

## Deviations from Plan

None - plan executed exactly as written. The two autonomous tasks were committed in the pre-checkpoint pass; this continuation agent resolved the approved human gate and finalized the plan metadata.

## Issues Encountered

None. The blocking human-verify checkpoint paused execution as designed; the user approved the GO verdict and this continuation finalized the plan.

## Human Gate Resolution

Task 3 was a blocking `checkpoint:human-verify`. The user response was **APPROVED** — the GO verdict is locked and Phase 18 may proceed. The orchestrator independently re-verified the gates (probes 8/8, witness 8/8 cpu f64+f32, bench A3 sub-second/tree) before this finalization.

## Next Phase Readiness

- **Phase 18 (production prims) is GO-gated and ready.** Re-author `quantiles` / `tree_hist` / `best_split` / `node_partition` from the spike findings + the spike-findings skill. Carry forward the VERDICT.md caveats: threshold = decision-equivalence (not raw float); regressor split-feature ties are splitter-RNG (gate on function-equivalence); size `tree_hist` scratch by the active frontier; regression variance = a second histogram on `y²`; cpu-MLIR op-set locked (single-owner GATHER, seed-from-candidate-0 argmax, no SharedMemory/atomics/`F::INFINITY`).
- SparseTreeNode contract + two-tier convention are finalized and binding for Phases 19/20/21.

## Self-Check: PASSED

- VERDICT.md exists, `verdict: GO` confirmed
- Prior commits present: 739fb3b (Task 1), 81a1672 (Task 2)
- Spike evidence dirs .planning/spikes/003–006-*/ present; MANIFEST.md appended
- 17-05-SUMMARY.md created

---
*Phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating*
*Completed: 2026-06-27*
