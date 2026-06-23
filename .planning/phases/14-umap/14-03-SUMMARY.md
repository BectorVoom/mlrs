---
phase: 14-umap
plan: 03
subsystem: manifold
tags: [umap, manifold, levenberg-marquardt, spectral-init, laplacian, eig, recover, splitmix64, oracle-fixtures]

# Dependency graph
requires:
  - phase: 14-umap (Plan 01)
    provides: empty umap_init.rs stub wired pub(crate) in mod.rs; committed umap_ab + umap_spectral oracle fixtures; RED ab_fit/spectral_init harness
  - phase: prior spectral phase
    provides: laplacian + eig prims, shared recover host math (cluster/spectral.rs)
provides:
  - "fit_ab host Levenberg–Marquardt a/b curve fit (D-06) matching umap-learn find_ab_params <=1e-5 (f64) with analytic Jacobian + AB_MAX_ITER cap + typed NotConverged"
  - "spectral_init reusing laplacian+eig+recover (diffusion_recover=false: raw symmetric-Laplacian eigenvectors, the umap spectral_layout convention) <=1e-5 up-to-sign per column × 5 metrics"
  - "random_init uniform(-10,10) SplitMix64 — the n>64 Jacobi-cap fallback AND init=random path (D-05 backbone for Plan 04)"
  - "noisy_scale_coords host helper (max=10, noise=1e-4 SplitMix64 Box–Muller) — umap's separate post-spectral stage for Plan 04"
  - "recover gains a diffusion_recover flag: true = sklearn /dd+sign-flip (SE/SC), false = raw eigenvectors (UMAP)"
affects: [14-04 (real fit drives fit_ab + spectral/random init through the SGD layout)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Host Levenberg–Marquardt (analytic Jacobian, λ trust-region) for a 2-param curve fit — no device kernel (D-06)"
    - "Flag-gated shared recover: diffusion_recover selects sklearn /dd+sign-flip vs umap raw eigenvectors, keeping all callers bit-identical by construction"
    - "Dump-diff verification of the value-gate convention BEFORE wiring (proved umap returns raw eigenvectors, not /dd recovery)"
    - "Oracle fixture precision repair: tighten umap's own ARPACK tol so its spectral_layout converges to the exact eigenvectors the mlrs Jacobi solver computes"

key-files:
  created: []
  modified:
    - crates/mlrs-algos/src/manifold/umap_init.rs
    - crates/mlrs-algos/src/manifold/mod.rs
    - crates/mlrs-algos/src/cluster/spectral.rs
    - crates/mlrs-algos/src/cluster/spectral_embedding.rs
    - crates/mlrs-algos/src/cluster/spectral_clustering.rs
    - crates/mlrs-algos/tests/umap_test.rs
    - scripts/gen_oracle.py
    - tests/fixtures/umap_spectral_{euclidean,manhattan,cosine,chebyshev,minkowski}_f64.npz

key-decisions:
  - "umap spectral_layout returns RAW symmetric-Laplacian eigenvectors (no /dd diffusion recovery, no sign flip) — verified by dump-diff (/dd path mismatched umap by ~0.2; raw path matched <=1e-6). Threaded a diffusion_recover flag into the shared recover (false for UMAP, true for SE/SC) rather than duplicating the slice/drop-first/transpose math."
  - "Regenerated the 5 committed umap_spectral fixtures with a tight ARPACK tol=1e-12: umap's default tol=1e-4 left ~4e-5 iterative error vs the exact eigenvectors, making the <=1e-5 gate unachievable against an exact Jacobi solver. The fixture still dumps umap's OWN spectral_layout — just at full precision."
  - "noisy_scale_coords is NOT applied inside spectral_init: umap applies it in simplicial_set_embedding (a separate stage), and the 1e-4 noise would break the <=1e-5 gate against the raw spectral_layout fixture. Exposed as a standalone pub helper for Plan 04."
  - "Made umap_init pub (was pub(crate)) so the integration value-gate in umap_test.rs can reach fit_ab/spectral_init (an integration test is an external crate and cannot see pub(crate))."
  - "fit_ab uses scipy's p0=[1,1] initial guess + LM λ trust-region with an analytic Jacobian; converges to umap find_ab_params <=1e-5 well within the AB_MAX_ITER=200 cap."

requirements-completed: [UMAP-01, UMAP-02]

# Metrics
duration: ~120min (dominated by the slow cpu Jacobi eig test runtime, ~32min/run)
completed: 2026-06-23
---

# Phase 14 Plan 03: UMAP a/b LM Fit + Spectral/Random Init Summary

**Landed UMAP's two remaining deterministic value-gated stages in `umap_init.rs`: a host Levenberg–Marquardt a/b curve fit (D-06) matching umap-learn `find_ab_params` to ≤1e-5 (f64), and spectral init reusing the existing laplacian+eig+recover stack — matching umap `spectral_layout` to ≤1e-5 up-to-sign across all 5 metrics — with the n≤64 Jacobi cap and the umap random-init fallback above it.**

## Performance

- **Duration:** ~120 min (the cpu Jacobi `eig` makes each 5-metric spectral_init run ~32 min single-threaded)
- **Tasks:** 2
- **Files modified:** 7 source/test/script + 5 regenerated fixtures

## Accomplishments

- **Task 1 — `fit_ab` (D-06):** self-contained host f64 Levenberg–Marquardt fitting `1/(1 + a·x^(2b))` to umap's smooth target curve over `linspace(0, spread*3, 300)`, with an analytic Jacobian (∂/∂a, ∂/∂b), a λ trust-region accept/reject step, a finite `AB_MAX_ITER=200` cap, and a typed `AlgoError::NotConverged` branch (T-14-06 / ASVS V5). Derived (a,b) match umap-learn 0.5.12 `find_ab_params` to ≤1e-5 across the whole `(min_dist,spread)` grid. NO device kernel.
- **Task 2 — `spectral_init` + `random_init` + `noisy_scale_coords`:** spectral init reuses `laplacian` → `eig` (descending) → `recover` (drop_first=true) on the symmetric fuzzy graph, with the n>64 random-init fallback (umap falls back, does NOT error). Value-gated ≤1e-5 up-to-sign per column vs umap `spectral_layout` for all 5 metrics (euclidean/manhattan/cosine/chebyshev/minkowski). `random_init` is `uniform(-10,10)` via SplitMix64; `noisy_scale_coords` is umap's separate post-spectral stage exposed for Plan 04.

## Task Commits

1. **Task 1: host Levenberg–Marquardt a/b curve fit (D-06)** - `3e723ca` (feat)
2. **Task 2: spectral init + random fallback + noisy scale, 5 metrics** - `61649ca` (feat)

## Files Created/Modified

- `crates/mlrs-algos/src/manifold/umap_init.rs` - `fit_ab` (LM), `spectral_init`, `random_init`, `noisy_scale_coords` + private `linspace`/`curve`/`residual_and_grad`/`next_standard_normal` helpers
- `crates/mlrs-algos/src/manifold/mod.rs` - `umap_init` made `pub` (was `pub(crate)`) so the integration value-gate can reach it
- `crates/mlrs-algos/src/cluster/spectral.rs` - `recover` gains `diffusion_recover: bool`; made `pub` for the umap path
- `crates/mlrs-algos/src/cluster/spectral_embedding.rs` / `spectral_clustering.rs` - pass `diffusion_recover=true` (sklearn /dd+sign-flip preserved)
- `crates/mlrs-algos/tests/umap_test.rs` - wired `fit_ab` into `ab_fit` and `spectral_init` (built from the fixture COO graph) into `run_spectral_init`
- `scripts/gen_oracle.py` - `gen_umap_spectral` now passes `tol=1e-12, maxiter=20000` to umap's `spectral_layout`
- `tests/fixtures/umap_spectral_*_f64.npz` - 5 fixtures regenerated at full ARPACK precision

## Decisions Made

- umap `spectral_layout` returns RAW eigenvectors of `I − D^-1/2 A D^-1/2` (no `/dd`, no sign flip) — proved by dump-diff (`/dd` mismatched umap by ~0.2; raw matched ≤1e-6). Threaded a `diffusion_recover` flag into the shared `recover` (false for UMAP, true for SE/SC) instead of forking the recovery math.
- Regenerated the spectral fixtures with `tol=1e-12`: umap's default ARPACK `tol=1e-4` carried ~4e-5 iterative error vs the exact eigenvectors, making the ≤1e-5 gate unachievable against the exact mlrs Jacobi solver. Still umap's own `spectral_layout`, just full precision.
- `noisy_scale_coords` is NOT applied inside `spectral_init` (umap applies it later in `simplicial_set_embedding`; the 1e-4 noise would break the ≤1e-5 gate). Exposed as a standalone helper for Plan 04.
- Made `umap_init` `pub` so the external integration test can call `fit_ab`/`spectral_init`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `recover`'s `/dd` diffusion recovery is WRONG for the UMAP spectral path**
- **Found during:** Task 2 (dump-diff before wiring)
- **Issue:** The plan/RESEARCH (A3, ASSUMED-from-source) said to reuse `recover(.., drop_first=true)` verbatim and compare up-to-sign. But `recover` applies sklearn's `/dd` diffusion-map scaling + a deterministic sign flip, whereas umap's `spectral_layout` returns the RAW eigenvectors. The `/dd` is a per-sample scaling — NOT a global per-column sign — so up-to-sign comparison could never pass. Dump-diff measured the `/dd` path mismatching umap by ~0.2 vs ~1e-6 for the raw path.
- **Fix:** Threaded a `diffusion_recover: bool` into the shared `recover` (the plan's sanctioned "thread a flag into recover" option, generalized from sign-only to the full diffusion family). UMAP passes `false` (raw eigenvectors); SpectralEmbedding/SpectralClustering pass `true` (unchanged behavior). Documented the empirical resolution of RESEARCH Q3/A3 in both the `recover` and `spectral_init` doc-comments.
- **Files modified:** `cluster/spectral.rs`, `cluster/spectral_embedding.rs`, `cluster/spectral_clustering.rs`, `manifold/umap_init.rs`
- **Commit:** `61649ca`

**2. [Rule 1 - Bug] Committed spectral fixtures carried ARPACK's loose-tolerance error**
- **Found during:** Task 2 (first spectral_init run — chebyshev 1.37e-5, cosine 4.03e-5 > 1e-5)
- **Issue:** umap's `spectral_layout` defaults its `eigsh` ARPACK solver to `tol=1e-4`, so the Plan-01 fixtures' `coords` carried up to ~4e-5 iterative error vs the exact eigenvectors. The exact mlrs Jacobi `eig` cannot match ARPACK's loosely-converged output to 1e-5 — confirmed with `np.linalg.eigh` showing the SAME ~4e-5 deviation (not an mlrs eig bug).
- **Fix:** Updated `gen_umap_spectral` to pass `tol=1e-12, maxiter=20000` so umap's OWN `spectral_layout` converges to the exact eigenvectors, and regenerated the 5 fixtures. Re-verified all 5 match `np.linalg.eigh` to ≤1.5e-7. All 5 spectral_init tests then GREEN.
- **Files modified:** `scripts/gen_oracle.py`, `tests/fixtures/umap_spectral_*_f64.npz`
- **Commit:** `61649ca`

**3. [Rule 3 - Blocking] `umap_init` was `pub(crate)`, unreachable from the integration value-gate**
- **Found during:** Task 1
- **Issue:** The `ab_fit`/`spectral_init` value-gate tests live in `crates/mlrs-algos/tests/umap_test.rs` (an external integration crate) and must call `umap_init::fit_ab`/`spectral_init`, but Plan 01 declared `umap_init` `pub(crate)` (per its own mod.rs note). An external crate cannot see `pub(crate)` items.
- **Fix:** Changed `pub(crate) mod umap_init` to `pub mod umap_init` in `manifold/mod.rs` (mirrors the sibling `pub mod umap_internals` that Plan 02 needed for the same reason). One-line, non-conflicting (Plan 02 is already complete).
- **Files modified:** `manifold/mod.rs`
- **Commit:** `3e723ca`

## Issues Encountered

- The cpu MLIR Jacobi `eig` is extremely slow at runtime: each full 5-metric `spectral_init` run takes ~32 min single-threaded, so iterating on the spectral fixtures/code was the long pole. The build itself is ~2 min incremental.

## Known Stubs

None for this plan's two stages. `noisy_scale_coords` is a fully-implemented host helper that `spectral_init` deliberately does NOT call (it is Plan 04's responsibility in the `fit` pipeline); the `seed` parameter of `spectral_init` is retained for the n>64 random fallback and Plan-04 call-site symmetry. These are intentional, not goal-blocking stubs.

## User Setup Required

None. (Fixture regeneration used the existing `/tmp/umap-oracle-venv` with `umap-learn==0.5.12`; the blobs are committed and CI never runs the generator.)

## Next Phase Readiness

- Plan 04 drives `fit_ab` (a/b) and `spectral_init`/`random_init` through the real `fit` + SGD layout, then applies `noisy_scale_coords` (max=10, noise=1e-4) and runs the property-gate threshold calibration (overwriting `PROPERTY_EPS`/`ARI_BAND`).
- The `diffusion_recover` flag is the single switch between the sklearn spectral family and the UMAP convention — Plan 04 needs no further recover changes.
- All RNG draws (`random_init`, `noisy_scale_coords`) are order-deterministic from `seed` (D-05 backbone).

## Self-Check: PASSED

- `crates/mlrs-algos/src/manifold/umap_init.rs` exists with `fn fit_ab`, `fn spectral_init`, `fn random_init`, `fn noisy_scale_coords` (grep-confirmed).
- Both task commits present in git history (`3e723ca`, `61649ca`).
- `cargo build -p mlrs-algos --features cpu --tests` exits 0.
- `cargo test -p mlrs-algos --features cpu --test umap_test ab_fit` GREEN (1 passed).
- `cargo test -p mlrs-algos --features cpu --test umap_test spectral_init` GREEN (5 passed, 0 failed — all metrics ≤1e-5 up-to-sign).

---
*Phase: 14-umap*
*Completed: 2026-06-23*
