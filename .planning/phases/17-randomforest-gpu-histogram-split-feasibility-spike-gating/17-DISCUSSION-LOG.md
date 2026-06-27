# Phase 17: RandomForest GPU Histogram/Split Feasibility Spike (GATING) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-27
**Phase:** 17-randomforest-gpu-histogram-split-feasibility-spike-gating
**Areas discussed:** Spike disposition, SparseTreeNode encoding, Verdict thresholds (A3 + ADJUST ladder), Two-tier gate + witness scope, Binning

---

## Spike disposition

| Option | Description | Selected |
|--------|-------------|----------|
| Keepable probes in-place | Draft tree kernels into mlrs-backend + spike harness, Phase 18 promotes in-place | |
| Mirror Phase 13 exactly | Throwaway `.planning/spikes/NNN-*` raw probes + `spike_test.rs`, wrap into findings skill; Phase 18 re-authors prims | ✓ |
| Hybrid | Throwaway raw exploration + commit proven kernels as draft prims + findings skill | |

**User's choice:** Mirror Phase 13 exactly
**Notes:** Maximal separation of throwaway exploration from production code; produces a spike-findings skill + VERDICT.md as the gating artifacts.

---

## SparseTreeNode encoding — leaf marker

| Option | Description | Selected |
|--------|-------------|----------|
| `colid = -1` sentinel | Leaf sets colid=-1 (treelite/FIL convention); traversal stops on colid<0; colid is signed (i32) | ✓ |
| `left_child = -1` sentinel | Leaf sets left_child=-1; traversal stops on left_child<0 | |
| Both sentinels on leaves | Set both colid=-1 and left_child=-1 for redundancy | |

**User's choice:** `colid = -1` sentinel
**Notes:** Matches treelite/cuML-FIL convention; implies `colid` is i32.

---

## SparseTreeNode encoding — `value` payload

| Option | Description | Selected |
|--------|-------------|----------|
| Scalar + deferred multiclass matrix | value = regression mean / binary P(class=1); multiclass in a separate side array (noted, not built) | |
| Class label only | value = argmax class label; loses predict_proba + TreeSHAP margins | |
| value-offset into shared buffer | value = index/offset into a shared leaf-value buffer; multiclass-uniform from day one | ✓ |

**User's choice:** value-offset into shared buffer
**Notes:** Redefines the locked `value` field from a scalar prediction to an index. Multiclass-uniform from the start; FIL/TreeSHAP must treat `value` as an offset, not a prediction.

---

## Verdict thresholds — A3 cost-tractability

| Option | Description | Selected |
|--------|-------------|----------|
| Tractability + scaling shape | Record benchmark; confirm sub-quadratic scaling; A3 fires only on clearly-impractical cost; no absolute cpu ms ceiling | ✓ |
| Relative-to-sklearn budget | GO if within a small multiple (≤10×) of sklearn single-tree fit | |
| Absolute ms ceiling | Hard per-tree ms target (e.g. ≤250ms) | |

**User's choice:** Tractability + scaling shape
**Notes:** cpu-MLIR is the correctness gate, not the runtime target (rocm/cuda are) — absolute cpu wall-clock would mislead.

---

## Verdict thresholds — ADJUST ladder

| Option | Description | Selected |
|--------|-------------|----------|
| bins → frontier-only → depth → defer | Fewer bins first (cheapest), then frontier-only histogram, then shallower depth, then defer | ✓ |
| frontier-only → bins → depth → defer | Frontier-only first (biggest structural win), then bins, then depth, then defer | |
| Let benchmark data pick | Choose lever from profile data | |

**User's choice:** bins → frontier-only → depth → defer
**Notes:** Cheapest change first; bins lever data-backed by the 64-vs-128 benchmark.

---

## Two-tier gate — Tier-2 ensemble band

| Option | Description | Selected |
|--------|-------------|----------|
| Absolute accuracy/R² band | Within ~0.02–0.05 of sklearn RF; fixed synthetic + seed; never element-wise | ✓ |
| Relative band | Within X% of sklearn's metric | |
| Statistical (overlapping CIs) | Not-significantly-worse across seeds | |

**User's choice:** Absolute accuracy/R² band
**Notes:** Documented as the milestone-wide standard this phase (not run — no forest yet; governs Phase 19). SplitMix64 ≠ MT19937, so only predictive quality is gated.

---

## Two-tier gate — correctness witness scope

| Option | Description | Selected |
|--------|-------------|----------|
| Classifier-gini + regressor-MSE | DecisionTreeClassifier(gini) + DecisionTreeRegressor(squared_error) on injected fixed indices | ✓ |
| Classifier-gini only | Only DecisionTreeClassifier(gini); regressor deferred | |
| gini + entropy + regressor | Broadest coverage, largest spike surface | |

**User's choice:** Classifier-gini + regressor-MSE
**Notes:** Exercises both leaf-value paths (class probabilities + regression mean) through the shared leaf-value buffer. entropy/log_loss/absolute_error deferred to Phase 18/19.

---

## Binning

| Option | Description | Selected |
|--------|-------------|----------|
| Host pre-pass, 128 default | Host quantile bin edges once per fit; default 128 bins; benchmark 64 vs 128; no on-device sort | ✓ |
| Host pre-pass, 64 default | Same host pre-pass, default 64 bins (A3-safer, coarser) | |
| Probe on-device binning | Attempt on-device bin edges (risks A2 unlowering sort) | |

**User's choice:** Host pre-pass, 128 default
**Notes:** A2 mitigation (host pre-pass "almost certainly acceptable" per PITFALLS); 128 sklearn-like fidelity; 64-vs-128 benchmark feeds the ADJUST ladder's bins lever.

---

## Claude's Discretion

- Exact relabel-partition kernel mechanics, benchmark-harness plumbing, fixture seeding, and rocm f32 skip-with-log wiring — follow Phase 13 / `cpu-mlir-kernel-authoring` conventions within the locked op-set.

## Deferred Ideas

- Multiclass leaf-value-buffer population path (defined this phase, populated Phase 19+)
- `entropy`/`log_loss`/`absolute_error` criteria (Phase 18/19)
- RNG-driven bootstrap/feature sampling (Phase 19 RF; spike injects fixed indices)
- Production prims `quantiles`/`tree_hist`/`best_split`/`node_partition` (Phase 18)
- On-device bin-edge computation (only if host pre-pass proves insufficient)
