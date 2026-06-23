---
spike: 002
name: directed-knn-compose-and-self-drop
type: standard
validates: "Given distance→top_k, when composed into directed (indices,distances) with include_self true/false and index-identity self-drop, then it matches a brute-force host KNN incl. duplicate-point/tie cases, self-drop GATHER-safe under cpu-MLIR"
verdict: VALIDATED
related: [001]
tags: [knn, composition, self-drop, topk, cpu-mlir, index-identity]
---

# Spike 002: Directed KNN Composition + Self-Drop by Index Identity

## What This Validates

**Given** the launch-proven `distance → top_k` path, **when** composed into the directed
`(indices, distances)` graph of shape `(n, k)` with:
- `include_self=true` → plain `top_k(k)` (self present, HDBSCAN core-distance path), and
- `include_self=false` → `top_k(k+1)` then a NEW `self_drop_gather` kernel dropping the
  column whose **index == query row** (R-3 / D-02), returning `k` true neighbours (UMAP path),

**then** the result matches a brute-force host KNN — including the **adversarial
duplicate-point case** that a "drop first zero-distance" rule gets wrong — and the self-drop
kernel is GATHER-safe under cpu-MLIR (no SharedMemory / mutable-bool scan / cross-loop flag).

Decision D-02 explicitly deferred *"the exact GATHER-safe mechanism under cpu-MLIR"* to this
spike. This is that confirmation — and it was **not** free.

## Research

No new external deps. Prior art: `top_k`/`distance` prims (launch-proven, oracle-tested);
`top_k`'s own source comments are the key reference — they document two cube-macro/cpu-MLIR
limitations (`F::INFINITY` rejection; *"the cube macro fails to infer a cross-loop flag"*)
that turned out to BOTH bear directly on the self-drop kernel.

**Approach:** true end-to-end through the real `distance` + `top_k` prims (n=6, d=2, k=3),
then the new `self_drop_gather` kernel, validated against an in-test brute-force host KNN with
the prim's documented `(distance, index)` lowest-index tie-break. Dataset has **rows 0 and 1
identical** (the duplicate at distance 0) to force the index-identity-vs-first-zero distinction.

## How to Run

```bash
cargo test -p mlrs-backend --features cpu --test knn_spike_002_test -- --nocapture
```

(`composition_and_self_drop.rs` here is the verbatim copy of
`crates/mlrs-backend/tests/knn_spike_002_test.rs` — the temporary run vehicle, deleted once
findings are recorded. The real prim lands in Phase 13 execution.)

## What to Expect

1 test passes, printing four ✓ lines: directed composition; exclude-self matches host with no
self leak; the adversarial dup-point assertion (query-1 neighbour-0 == genuine idx 0); and
include-self == plain `top_k(k)` with self present in every row.

## Investigation Trail

The composition was trivial; the self-drop kernel took **three iterations**, each surfacing a
distinct cpu-MLIR landmine:

1. **Compile error — `ABSOLUTE_POS` is `usize`, not `u32`.** Mixing it with `u32` kernel
   params failed type-check. (Note: `ABSOLUTE_POS_X` used in Spike 001 *is* `u32` — the bare
   linear `ABSOLUTE_POS` is `usize`, used bare as an array index in `rbf_map`.) First fix:
   `let row = ABSOLUTE_POS as u32`.

2. **FINDING 002-A — cpu-MLIR lowering FAILURE.** With the `ABSOLUTE_POS as u32` + 1D launch,
   the kernel triggered `error: operation with block successors must terminate its parent
   block` / `failed to run pass` (cubecl-cpu MLIR), so it never ran (output read back as
   zeros). **Fix:** adopt `top_k`'s exact lowering-proven launch shape — `row = CUBE_POS_X`
   (native `u32`) with a `UNIT_POS_X == 0` guard and one cube per row. The MLIR error vanished.
   *The launch/index-builtin shape matters to lowering, independent of the body.*

3. **FINDING 002-B — silent MISCOMPILE of a cross-sibling-loop accumulator.** The natural
   self-drop (scan all `k+1` columns into a mutable `c_self`, then a second sibling loop reads
   `c_self` to compact) compiled and ran but produced WRONG results: `c_self` never updated, so
   every row dropped its last column instead of self. This is precisely the limitation
   `top_k`'s comments call out — *the cube macro fails to carry a mutable flag written in one
   loop and read in a separate sibling loop.* **Fix:** eliminate the cross-loop carry — for
   each output slot `s`, recompute the shift LOCALLY as `bump = (#self-columns at input cols
   ≤ s)` via a nested accumulate read in the SAME outer iteration (the `top_k`-proven
   nested-reduce shape), then `src = s + bump`. Correct AND lowers.

(Intermediate hypothesis — that a *conditionally-mutated array index* was the MLIR trigger —
was tested and DISPROVEN: rewriting to unconditional dual-index + value-select did not fix
002-A; the launch-builtin shape (002-A) did.)

## Results

**VERDICT: VALIDATED ✓** — the directed multi-`include_self` KNN graph is feasible and correct
under cpu-MLIR, *with two specific kernel-authoring constraints the planner must honour.*

- Directed `(indices, distances)` `(n, k)` composes cleanly from the real `distance` + `top_k`
  prims; `include_self=true` is just `top_k(k)` (no new kernel needed).
- **Index-identity self-drop is correct on the duplicate-point case** — query row 1 (with a
  distance-0 duplicate at index 0) returns the genuine neighbour (idx 0) at slot 0, not self.
  A first-zero-distance drop would have failed here. R-3 is validated as necessary AND
  implementable.
- The self-drop kernel lowers and computes correctly under cpu-MLIR f64.

**Signal for the build (carry these into the real `knn_graph.rs` / kernel):**
1. **Self-drop kernel MUST use the `top_k` launch shape** — `row = CUBE_POS_X` + `UNIT_POS_X
   == 0` guard, NOT a bare-`ABSOLUTE_POS` 1D launch (FINDING 002-A → MLIR pass failure).
2. **NO cross-sibling-loop mutable accumulator** — compute any per-row positional value with a
   self-contained nested accumulate inside the output loop (FINDING 002-B → silent
   miscompile). Add a regression test that includes a duplicate point AND asserts values, not
   just "it launched" — 002-B passed compilation and only a value assertion caught it.
3. `include_self=false` path: query `k+1`, drop by index identity; fallback (self absent from
   top-`k+1`, which shouldn't happen for X-vs-X) drops the last column — covered by the `bump`
   formulation for free.
4. Memory gate (R-6) unaffected: self-drop runs on the small `(n, k+1)` top_k output, not the
   `n×n` distance block.

## Surprises

- **The cheap-looking step was the dangerous one.** The "trivial" self-drop — not the new
  distance kernels — produced both cpu-MLIR failures this milestone. One was a hard launch-time
  error (loud); the other was a SILENT miscompile that compiled, launched, and returned plausible
  wrong data. Only an end-to-end value assertion against an oracle (with a duplicate point)
  caught the silent one. Strong argument for the primitive-first + oracle-gate discipline:
  a happy-path "does it launch" check would have shipped 002-B.
