---
spike: 005
name: relabel-partition
type: standard
phase: 17-randomforest-gpu-histogram-split-feasibility-spike-gating
requirement: TREE-01
validates: "Given samples labelled by node + per-node frontier split arrays, when a per-sample relabel-partition GATHER kernel (one cube per sample, CUBE_POS_X/UNIT_POS_X==0, self-overwrite, no scan/compaction) is launched under --features cpu, then each sample moves to its left child (go-left) or left_child+1 (go-right, D-02) value-correctly"
verdict: VALIDATED
related: [003, 004, 006]
tags: [tree, randomforest, cpu-mlir, relabel, partition, gather, D-02, TREE-01]
---

# Spike 005: Per-Sample Relabel-Partition GATHER (D-02 child layout)

## What This Validates

**Given** samples labelled by current node and the per-node frontier split arrays
(`split_active` / `split_col` / `split_bin` / `left_child`), **when** the relabel kernel — one cube
per sample (`CUBE_POS_X` / `UNIT_POS_X == 0`), the sample reads its node's split and **overwrites its
own label** with the left child (go-left) or `left_child + 1` (go-right) — is launched under
`--features cpu`, **then** each sample moves to the correct child value-exactly, validating the **D-02
adjacent-child layout** (`right = left_child + 1`). Modelled on `mlrs_kernels::self_drop_gather`.

The cpu-MLIR risk is the partition step: the textbook GPU partition is a prefix-scan / stream
compaction (scan loops + a write cursor — a 002-A/002-B class hazard). This spike proves the partition
is expressible as a **pure per-sample self-overwrite GATHER** — no scan, no compaction, no cross-sibling
accumulator — which both sidesteps the landmine and locks the D-02 layout the whole `SparseTreeNode`
contract depends on.

## Result

**VERDICT: VALIDATED ✓.**

- The relabel lowered as the proven per-row `CUBE_POS_X` / `UNIT_POS_X == 0` shape (NOT a bare 1D
  `ABSOLUTE_POS` launch — the 002-A failure). Each sample reads its node's split from the frontier
  arrays and self-overwrites: `bv > split_bin → left_child + 1` (right), else `left_child` (left).
- The kernel is **non-generic, all-`u32`** — a `split_active` `u32` flag array replaces a signed `-1`
  leaf sentinel so there are **no in-kernel signed value casts** (`as usize` only at the index
  boundary). A `split_active[nid] == 0` (leaf/inactive) node leaves its samples untouched.
- The probe VALUE-asserts exact left/right labels per sample (left = 1, right = left + 1 = 2) with a
  002-A guard (a non-launch leaves every label at the root 0). f64 + f32 green.

## cpu-MLIR Findings (for the Phase 18 prim re-author)

- **Partition = per-sample self-overwrite GATHER, never a scan/compaction.** This is the key landmine
  avoidance for `node_partition`: keep one cube per sample writing its own new label; do NOT build a
  prefix-sum write-cursor.
- **`split_active` u32 flag instead of a signed sentinel** keeps the relabel kernel in the all-`u32`
  op-set (no in-kernel signed casts). The signed `colid == -1` sentinel lives in the host-side
  `SparseTreeNode`, not the device relabel.
- **D-02 `right = left_child + 1` is enforced at relabel time** (the go-right branch literally writes
  `lc + 1`) — the layout the FIL/TreeSHAP phases bind to is produced here, not just asserted later.

## Source (verbatim — durable evidence, D-01)

- `kernels_and_harness.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_spike/mod.rs`. The
  relabel kernel is `tree_relabel_partition` + `launch_relabel`.
- `probes.rs` — byte-identical copy of `crates/mlrs-backend/tests/tree_spike_probes.rs`. The evidence is
  `check_relabel<F>` (+ the `tree_relabel_{f32,f64}_child_labels` entry points).

## How to Run (live gate)

```bash
cargo test -p mlrs-backend --features cpu --test tree_spike_probes -- --nocapture
```

The live test files remain the runnable gate; these copies are the durable artifact (D-01). Phase 18
re-authors the production `node_partition` prim from these findings.
