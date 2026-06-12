---
phase: 03-svd-eigendecomposition-primitive-hard-gate
reviewed: 2026-06-12T00:00:00Z
depth: standard
files_reviewed: 13
files_reviewed_list:
  - crates/mlrs-kernels/src/jacobi_svd.rs
  - crates/mlrs-kernels/src/jacobi_eig.rs
  - crates/mlrs-kernels/src/lib.rs
  - crates/mlrs-backend/src/prims/svd.rs
  - crates/mlrs-backend/src/prims/eig.rs
  - crates/mlrs-backend/src/prims/mod.rs
  - crates/mlrs-backend/src/runtime.rs
  - crates/mlrs-core/src/error.rs
  - crates/mlrs-backend/Cargo.toml
  - scripts/gen_oracle.py
  - crates/mlrs-backend/tests/svd_test.rs
  - crates/mlrs-backend/tests/eig_test.rs
  - crates/mlrs-backend/tests/memory_gate_test.rs
findings:
  critical: 1
  warning: 5
  info: 4
  total: 10
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-06-12
**Depth:** standard
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Reviewed the Phase-03 SVD/eigendecomposition primitive: two `#[cube(launch)]` Jacobi
kernels (one-sided SVD, two-sided symmetric eig), their host orchestration in
`mlrs-backend/src/prims/{svd,eig}.rs`, the runtime/error/Cargo wiring, the oracle
generator, and the three test files.

The error surface, validate-before-unsafe ordering (D-06), buffer-reuse plumbing, and
the eig two-sided sequential-pair race fix all look correct. The numerical-extraction
algebra in the host (thin-U via `A·V`, `Vᵀ` transpose read, wide-path relabel, descending
sort) traces correctly.

The headline defect is a **convergence-scheduling bug in the one-sided SVD kernel**: the
round-robin "circle method" only enumerates the full pair set for an **even** thin
dimension `cols`. For any **odd** `cols` it silently omits roughly half the column pairs
every sweep, so those off-diagonals are never zeroed — the SVD will either fail to
converge or return a wrong factorization. Every committed and in-test case uses an even
thin dimension (4, 64), so the test suite does not exercise this path and the bug is
fully masked. Because PCA/TruncatedSVD/OLS in Phase 4 consume this primitive with
caller-chosen ranks (frequently odd), this must be fixed before the gate is considered
passed.

Secondary findings concern the SVD convergence-norm being a within-sweep mixture (not a
true post-sweep measurement, unlike the eig kernel which explicitly avoids this), an eig
buffer released to the pool free-list immediately after an enqueued launch, and several
documentation/coverage gaps.

## Critical Issues

### CR-01: One-sided SVD circle-method schedule omits ~half the column pairs for odd `cols`

**File:** `crates/mlrs-kernels/src/jacobi_svd.rs:167-247` (schedule), `:296-304` (`circle_player`)

**Issue:** The sweep uses `n_steps = cols - 1` steps with `half = cols / 2` pairs per
step, pairing circle position `p` with position `cols-1-p` while holding player 0 fixed
(`circle_player`). This standard round-robin construction only enumerates all
`n(n-1)/2` pairs when the number of players is **even**. For an **odd** `cols` the
correct circle method needs `cols` rounds with a per-round "bye" over a rotation of
`cols` positions — not `cols-1` rounds over `cols-1` rotating positions. Empirically the
implemented schedule covers:

- `cols=4`: 6/6 pairs (OK), `cols=6`: 15/15 (OK)
- `cols=3`: 2/3 — misses `(1,2)`
- `cols=5`: 6/10 — misses `(1,2),(1,4),(2,3),(3,4)`
- `cols=7`: 12/21 — misses 9 pairs

A column pair that is never visited has its off-diagonal `γ_ij` never driven toward zero.
Consequences for any odd thin dimension:
1. The orthogonalization is incomplete → `U`/`Vᵀ` are not orthonormal and
   `‖U·diag(S)·Vᵀ − A‖` does not reach tolerance (silent wrong answer), and/or
2. The off-diagonal norm plateaus above `conv_thr` → `NotConverged` on inputs that are
   perfectly well-conditioned.

The bug is **completely masked** by the test matrix: `svd_tall`/`svd_wide` fixtures use
`k=4`, the moderate case is `256×64`, the memory-gate uses `cols=4`, and the wide path
transposes to an even tall thin-dim — every exercised `cols` is even. Phase-4 consumers
(PCA components, TruncatedSVD `n_components`, OLS pseudo-inverse) routinely request odd
ranks/feature counts and will hit this directly.

**Fix:** Use the correct odd/even-aware round-robin. The simplest correct form pads to an
even player count `P = cols` if even else `cols + 1` (the extra "ghost" player gives the
bye), runs `P - 1` rounds over `P/2` positions, and skips any pair touching the ghost
column `>= cols`:

```rust
// even player count: real cols if even, else cols+1 (ghost gives the bye).
let players = if cols % 2u32 == 0u32 { cols } else { cols + 1u32};
let n_steps = players - 1u32;
let half = players / 2u32;
// ... inside the step loop, p in [0, half):
let col_a = circle_player_even(p, step, players);
let col_b = circle_player_even(players - 1u32 - p, step, players);
let lo = if col_a < col_b { col_a } else { col_b };
let hi = if col_a < col_b { col_b } else { col_a };
// skip ghost pairs and self-pairs:
if c == lo && lo != hi && hi < cols { /* ...rotate... */ }
```

where `circle_player_even` fixes position 0 and rotates positions `1..players` modulo
`players - 1`. Add explicit odd-`cols` tests (e.g. `7×3`, `9×5`, a `256×63` moderate
case) to the reconstruction + orthonormality invariants so this can never regress
undetected. The kernel header comment (`jacobi_svd.rs:33-37`) and the inline comment
(`:161-166`) both assert the schedule "covers all n(n-1)/2 pairs over n-1 steps" — that
claim is only true for even `n` and must be corrected alongside the fix.

## Warnings

### WR-01: SVD convergence norm is a within-sweep mixture, not a true post-sweep measurement

**File:** `crates/mlrs-kernels/src/jacobi_svd.rs:208-210, 253-268`

**Issue:** `off_sh[lo] += gamma*gamma` records each pair's `γ²` at the moment that pair
is processed *during* the sweep. Because an earlier pair's rotation in the same sweep
mutates columns that a later pair then reads, the accumulated `Σγ²` is not the
off-diagonal norm of any single consistent matrix state — it mixes pre- and mid-sweep
values. The eig kernel explicitly avoids exactly this (`jacobi_eig.rs:250-258`:
"a real measurement of where the matrix stands after this sweep's rotations — NOT an
in-sweep estimate that a later rotation could refill"). The SVD kernel does the thing the
eig kernel warns against. This can declare convergence one sweep early (the reported
`info[1]` residual then does not describe the returned matrix) or, conversely, keep
sweeping when already converged. The reconstruction invariant test is only asserted at
`1e-4` (`svd_test.rs:290`), looser than the project's 1e-5 contract, which would not
catch a one-sweep-early exit that leaves a residual between 1e-5 and 1e-4.

**Fix:** Measure the off-diagonal norm the same way the eig kernel does — after the full
sweep's rotations, recompute the column-pair Gram off-diagonals from the *current*
`a_out` state in a dedicated post-sweep pass before the tree reduction, rather than
accumulating during rotation. Then tighten the SVD reconstruction-invariant assertion to
the 1e-5 contract (or document why 1e-4 is the forced per-family bound per D-10, naming
the case).

### WR-02: Eig releases the reused `out` buffer to the pool free-list immediately after an enqueued launch

**File:** `crates/mlrs-backend/src/prims/eig.rs:146-151`

**Issue:** `jacobi_eig_sweep::launch(...)` enqueues the kernel; immediately afterward, if
`out` was supplied, `buf.release_into(pool)` returns that buffer to the free-list. If the
CubeCL launch is asynchronous (queued) and the kernel still reads `a_in` (= this buffer)
when a *subsequent* `pool.acquire` of the same byte-size hands the buffer to another
consumer, the in-flight kernel read aliases a reused buffer — a data hazard. In the
current control flow the next pool interaction is the `info_dev.to_host` read-back
(line 157), which forces a sync, so today it is safe; but the safety is incidental
(depends on a read-back happening before any same-size `acquire`) rather than enforced.
The SVD path has the analogous pattern at `svd.rs:200-202` (releasing the rotated-A
scratch right after launch).

**Fix:** Either (a) hold the released-buffer return until after the first synchronizing
read-back, or (b) document the invariant explicitly ("released input is safe because the
immediately-following metered/plain `to_host` syncs the queue before any reuse") and add
a test that a same-size `acquire` between launch and read-back cannot occur. Prefer (a)
for robustness against future reordering.

### WR-03: Eig kernel/host disagree on the `sqrt(2)` off-diagonal-norm scale; comment claims a fold the host does not perform

**File:** `crates/mlrs-kernels/src/jacobi_eig.rs:60-70, 256-258` vs `crates/mlrs-backend/src/prims/eig.rs:243-275`

**Issue:** The eig kernel measures `sqrt(2·Σ_{i<j} a_ij²)` (per-row double-count). Its
comment (`jacobi_eig.rs:258`) states "the scalar factor is folded consistently into the
conv_thr comparison (the host scales conv_thr the same way)." The host `compute_thresholds`
(`eig.rs:243-275`) uses the identical `8·ε·‖A‖·sqrt(pairs)` formula as SVD with **no**
`sqrt(2)` factor — it does not "scale conv_thr the same way." The net effect is a
convergence break that is `sqrt(2)`× stricter than nominal (benign — converges slightly
later/tighter), so this is not a correctness defect, but the comment is false and will
mislead the next maintainer into thinking the factors cancel.

**Fix:** Either divide the kernel's measured norm by `sqrt(2)` before the `conv_thr`
compare (so the reported `info[1]` is the true off-diagonal Frobenius norm and matches the
host's `residual > conv_thr` check semantics), or correct the comments in both files to
state that the kernel norm carries an extra `sqrt(2)` the host does NOT compensate, making
the break intentionally stricter.

### WR-04: `info[1]` residual reported by the SVD kernel may not reflect the returned matrix on convergence

**File:** `crates/mlrs-kernels/src/jacobi_svd.rs:264-288`; consumed at `crates/mlrs-backend/src/prims/svd.rs:210-223`

**Issue:** On the sweep that sets `converged = true`, `off_sh[0]` holds the within-sweep
mixed accumulation (see WR-01), and that value is written to `info_out[1]`. The host's
`NotConverged` decision (`svd.rs:212-216`) compares this `residual` against `conv_thr`.
If the within-sweep estimate under-reports the true post-sweep norm, a genuinely
non-converged result (cap hit) could pass the `residual > conv_thr` guard and be returned
as "converged," producing a silently wrong factorization without the `NotConverged`
signal. This is the convergence-detection consequence of WR-01 reaching the host's
correctness gate.

**Fix:** Resolve WR-01 (true post-sweep measurement); then `info[1]` describes the
returned matrix and the host guard is sound. Until then, the NotConverged guard cannot be
trusted to fire on every non-converged input.

### WR-05: No test exercises the `NotConverged` path, the odd-`cols` path, or a `n=1`/`cols=1` degenerate

**File:** `crates/mlrs-backend/tests/svd_test.rs` (whole), `crates/mlrs-backend/tests/eig_test.rs` (whole)

**Issue:** `PrimError::NotConverged` is never asserted by any test — there is no input
that forces a cap hit, so the entire convergence-failure surface (kernel `info` write →
host threshold compare → error construction → output-handle release on the eig error
path, `eig.rs:162-164`) is unverified. Likewise no test uses an odd thin dimension (which
would have caught CR-01) and no test uses a single-column SVD or `1×1` eig. The
reconstruction/orthonormality invariants are the project's strongest hermetic check
(D-09) but only run on even, well-conditioned shapes.

**Fix:** Add (a) an odd-`cols` SVD invariant test and odd-`n` eig residual test (gates
CR-01), (b) a deliberately pathological/over-cap-sweep input asserting `Err(NotConverged
{ .. })` (gates the failure surface), and (c) `cols=1` / `n=1` degenerate cases.

## Info

### IN-01: Doc comment in `error.rs` mislabels `PrimError`/`BridgeError` provenance

**File:** `crates/mlrs-core/src/error.rs:1-15, 59`

**Issue:** The module/struct docs reference "Plan 03's bridge," "Plan 03 bridge,
FOUND-06 / D-07," and "Phase 2 primitives, D-04" — stale plan/decision tags carried from
earlier phases that no longer match this phase's D-06/D-12 decisions for the new
`NotSquare`/`NotConverged` variants. Misleading for traceability but harmless at runtime.

**Fix:** Update the doc tags to reference the actual Phase-03 decisions (D-06 squareness,
D-12 convergence) for the variants added here.

### IN-02: `memory_gate_test.rs` header still claims "green on cpu AND wgpu"

**File:** `crates/mlrs-backend/tests/memory_gate_test.rs:33`

**Issue:** Comment says "the counter assertions are backend-agnostic (green on cpu AND
wgpu)" while the Phase-3 gate (D-07) moved the GPU gate to rocm; the Phase-3 section
lower in the same file correctly says cpu+rocm (`:476`). Inconsistent within one file.

**Fix:** Update the top-of-file comment to cpu+rocm to match D-07 and the Phase-3 section.

### IN-03: `circle_player`/schedule comments describe a "disjoint pairs run concurrently" property that the kernel does not actually exploit

**File:** `crates/mlrs-kernels/src/jacobi_svd.rs:161-166, 173-183`

**Issue:** The comments describe partitioning so "disjoint pairs in a step run
concurrently," but the actual code has every unit scan all `half` pair positions and only
the matching `lo` unit acts — there is no per-unit direct partner computation; it is an
all-units-scan-all-positions loop. The behavior is correct (for even cols) but the
comment overstates the parallel structure, which contributed to the CR-01 schedule error
going unnoticed.

**Fix:** Simplify the comment to match the actual "every unit scans every pair position,
the `lo` unit acts" implementation, and (after CR-01's fix) state the exact parity
contract.

### IN-04: `gen_oracle.py` ships no odd-shape or rank-deficient SVD/eig fixture, and no wide-f64 case

**File:** `scripts/gen_oracle.py:61-71, 345-354`

**Issue:** Fixture shapes are `SVD_TALL=(8,4)`, `SVD_WIDE=(4,8)`, `EIG_N=4` — all even
thin dimensions, all well-conditioned, and the wide path is f32-only (no wide-f64 cpu
gate). The degenerate D-08 cases (rank-deficient, clustered) are generated in-test rather
than committed, so the numpy oracle never pins them. Combined with WR-05 this is why
CR-01 had zero detection coverage.

**Fix:** Add an odd-thin-dim SVD fixture (e.g. `(7,3)`) and an odd-`n` eigh fixture, plus
a wide-f64 case, so the committed numpy oracle covers the parity and dtype matrix the
primitive must hold.

---

_Reviewed: 2026-06-12_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
