# Phase 14: UMAP - Research

**Researched:** 2026-06-23
**Domain:** Manifold learning (UMAP) — fuzzy-simplicial-set construction, spectral/random init, stochastic SGD layout, new-point transform; CubeCL kernels generic over `F`/runtime under the cpu-MLIR f64 gate.
**Confidence:** HIGH (algorithm fidelity verified against umap-learn 0.5.12 source; every reuse prim located and read in-repo)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01: Expose ALL 5 Phase-13 metrics** on UMAP's `metric=` param — euclidean, manhattan (L1), cosine, chebyshev (L∞), minkowski-p. The shell's `Metric` enum (currently `Euclidean`-only) is extended to the full set this phase.
- **D-02: Full deterministic value-gate × all 5 metrics.** Run the ≤1e-5 (f64) deterministic-stage value-gate (fuzzy set, union, spectral init, a/b) AND a property-gated layout run for **every** metric — not Euclidean-only. Oracle-fixture regen covers all 5 metrics (needs the `/tmp` numpy venv; fixtures are committed blobs).
- **D-03: Full umap-learn transform path.** `transform(X_new)` = KNN(new→train) → fuzzy membership against the fitted graph → neighbor-weighted-average init → reduced-epoch SGD optimizing **only the new points** with the **training embedding frozen** (read-only GATHER targets). Reuse the SAME vertex-owner layout kernel — new points are the sole "owners"; trained coords are read-only. **The vertex-owner SGD kernel must support a "frozen-subset" mode from day one** (a contiguous owner set whose non-owner neighbors are read-only), since both `fit` and `transform` drive the same kernel.
- **D-04: Track umap-learn 0.5.12 TIGHTLY.** Property gate requires mlrs to score within a small margin of umap-learn on the SAME data: trustworthiness ≥ umap-learn − ε, kNN-overlap ≥ umap-learn − ε, downstream-ARI within a tight band — NOT just absolute floors. Margins (`ε`, band) calibrated empirically on the first oracle-fixture run, kept tight.
- **D-05: `fit` AND `transform` byte-identical for a fixed `random_state`** — covering init RNG + negative-sampling RNG + new-point SGD RNG. Both `fit` and `transform` reproduce byte-identical mlrs embeddings across runs. Every PRNG draw must be **order-deterministic** in the kernel. Byte-identity is per `(backend, dtype)` (f32-vs-f64 alone precludes cross-dtype bit-identity; float reduction order differs across runtimes). SplitMix64 (≠ NumPy MT) is why the layout is property-gated, never coordinate value-matched.
- **D-06: Port a host-side Levenberg–Marquardt least-squares fit.** When `a`/`b` not overridden, derive them by least-squares fitting `1/(1 + a·d^(2b))` to the smooth target curve from `min_dist`/`spread`, replicating scipy `curve_fit`. Value-gate derived `a`/`b` to ≤1e-5 vs umap-learn — a FIFTH deterministic value-gated stage. Self-contained host numeric routine (NO device kernel).

### Claude's Discretion
- **Spectral-init Jacobi size cap & disconnected-graph handling** — NOT discussed; follow the existing v2 graph-Laplacian + v1 Jacobi-eig convention (cap value, above-cap random fallback, disconnected-component handling). Planner may finalize using the established convention; surface to the user only if the v2 convention doesn't transfer cleanly.
- `n_epochs=None` auto heuristic (umap-learn: 500 small / 200 large) — match umap-learn; exact threshold planner's to confirm against the oracle.
- Negative-sampling index draw mechanics under cpu-MLIR (order-deterministic per D-05 and GATHER/SharedMemory-free per the spike landmines) — planner/spike detail.
- Exact `Metric` enum extension shape and whether minkowski-p `p` is `F` or `f64` — follow the Phase-13 prim's `Metric` shape for consistency.
- LM solver internals (Gauss-Newton vs full LM, damping schedule, convergence tol) — any choice that hits the ≤1e-5 a/b value-gate.

### Deferred Ideas (OUT OF SCOPE)
- Spectral-init Jacobi cap / disconnected-graph handling — defaults to the existing v2 convention (Claude's discretion); raise to user only if it doesn't transfer.
- PyO3 wrap of `Umap` and the builder-retrofit sweep — Phase 16.
- Supervised/target-metric UMAP, `inverse_transform`, approximate/NN-Descent KNN build, native sparse path, custom/callable metrics — already out of scope in REQUIREMENTS.md.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| UMAP-01 | `fit`/`fit_transform` → `embedding_` `(n, n_components)` with umap-learn-named hyperparameters & defaults (`n_neighbors=15`, `n_components=2`, `metric='euclidean'`, `min_dist=0.1`, `spread=1.0`, `n_epochs=None`, `init='spectral'`, `random_state`, `learning_rate=1.0`, `set_op_mix_ratio=1.0`, `local_connectivity=1.0`, `repulsion_strength=1.0`, `negative_sample_rate=5`, `a`/`b`), `min_dist ≤ spread` validated. | Shell already ships the full surface + build-time `min_dist≤spread` (umap.rs:325-348). §Standard Stack maps each param to its umap-learn 0.5.12 formula; §Architecture pins the pipeline order; §Code Examples give the exact membership / a-b / gradient formulas. |
| UMAP-02 | Deterministic stages (KNN graph, fuzzy simplicial set, fuzzy-set union, spectral init) value-match umap-learn intermediates to ≤1e-5 (f64). | §Architecture Patterns 1–4 give exact formulas + the load-bearing operation ORDER; §Validation Architecture specifies the per-stage × per-metric committed `.npz` fixtures and what umap-learn internals to dump; reuse prims (`knn_graph`, `laplacian`, `eig`) located + read. |
| UMAP-03 | Stochastic SGD layout passes a property/structural gate vs umap-learn 0.5.12 (trustworthiness / kNN-overlap ≥ umap-learn − margin, downstream-ARI in band) + same-`random_state` byte-identical reproducibility. | §Architecture Pattern 5 (vertex-owner GATHER SGD); §Pattern 6 (order-deterministic SplitMix64 plumbing); §Validation Architecture (property-gate metric definitions + calibration protocol); §Pitfalls (cross-sibling miscompile, F::INFINITY ban). |
| UMAP-04 | `transform(X_new)` against fitted fuzzy graph, gated by a property sub-gate on the new points. | §Architecture Pattern 7 (frozen-subset transform path) + the exact umap-learn `transform` recipe (init_graph_transform, n_epochs=100, move_other=False); the SAME layout kernel drives it (D-03). |
</phase_requirements>

## Summary

UMAP-01..04 fills the Phase-12 `Umap<F,S>` shell with the real algorithm. The good news for planning: **every numerically-deterministic stage is fully specified by umap-learn 0.5.12 source (verified this session) and every reuse primitive already exists, validated, in-repo.** The KNN graph is the Phase-13 prim (`knn_graph`, `include_self=false`). The spectral-init stack — symmetric-normalized Laplacian `L = I − D^-1/2 A D^-1/2` plus the Jacobi eigensolver — is exactly the existing `laplacian` prim + `eig` prim + the shared `recover` host math, and umap-learn's spectral Laplacian is byte-for-byte the same normalization as `laplacian.rs`. The `a`/`b` curve fit is a self-contained host Levenberg–Marquardt routine (no device kernel, no new dependency). The fuzzy-simplicial-set and t-conorm union are host array math over the directed `(n,k)` KNN result.

The single genuine new device kernel is the **vertex-owner GATHER SGD layout step** (`umap_layout_step`). Its design is fully constrained by the cpu-MLIR landmines documented in `spike-findings-mlrs`: no `SharedMemory`/atomics/`F::INFINITY`/mutable-bool/shift-loop, no bare-`ABSOLUTE_POS` 1D launch (use the `CUBE_POS_X`/`UNIT_POS_X==0` per-owner shape), and **no cross-sibling-loop accumulator** (the silent-miscompile landmine — recompute per-owner positional values inside the consuming loop). The precedent the spike flag cites — the v2 two-pass `sgd_margin`/`sgd_weight_update` GATHER solver — is the proven shape to mirror. The frozen-subset mode (D-03) falls out naturally: the kernel updates only "owner" rows and reads non-owner (training) coords read-only, so `fit` (all rows are owners, `move_other` emulated by a two-sided update) and `transform` (only new points are owners) drive the same kernel.

The hard part is **not** the math; it is the gate philosophy. Because UMAP's negative sampling uses umap-learn's `tau_rand_int` Tausworthe PRNG (3×uint32 xorshift) and mlrs uses SplitMix64, coordinates can never match — so the SGD layout is property-gated (trustworthiness/kNN-overlap/ARI relative to umap-learn, D-04), while mlrs's own draws are made order-deterministic so `fit`/`transform` are byte-identical run-to-run per (backend,dtype) (D-05). The property-gate thresholds are calibrated empirically on the first fixture run (Spike flag item 2). The deterministic stages 1–4 + a/b ARE value-gated ≤1e-5 across all 5 metrics (D-02).

**Primary recommendation:** Build the deterministic pipeline (KNN → smooth-kNN ρ/σ → membership → t-conorm union → spectral/random init → a/b LM fit) entirely as host orchestration over existing prims + host array math, value-gated ≤1e-5 per stage per metric against dumped umap-learn 0.5.12 internals. Add ONE new device kernel `umap_layout_step` (mirroring the v2 two-pass SGD GATHER shape, cpu-MLIR-safe, frozen-subset-capable from day one), property-gated. Spike the layout kernel launch + threshold calibration BEFORE planning, exactly as the Spike flag directs.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| KNN graph (5 metrics, `include_self=false`) | Backend prim (`knn_graph`) | — | Phase-13 prim; UMAP only calls it. Directed `(n,k)`; UMAP owns symmetrization (D-04). |
| smooth-kNN ρ/σ binary search | Host (estimator) | — | Per-row scalar binary search over the `(n,k)` distances; no device parallelism win at UMAP n; host f64 keeps it value-gatable. |
| membership strengths + t-conorm union | Host (estimator) | — | Sparse `(n,k)` array math (COO rows/cols/vals); umap-learn does it host-side in scipy.sparse. |
| spectral init Laplacian | Backend prim (`laplacian`) | — | Existing PRIM-09; umap's `I−D^-1/2 A D^-1/2` matches it exactly. |
| spectral init eigensolve | Backend prim (`eig`, Jacobi) | Host (`recover`) | Existing PRIM-05 + shared `recover` host math (slice-smallest → /dd → sign-flip). `n ≤ MAX_DIM=64` cap → random fallback above it. |
| random init | Host (estimator) | — | `uniform(-10,10)` via SplitMix64; one upload. |
| a/b curve fit (LM) | Host (estimator) | — | Self-contained Gauss-Newton/LM; D-06 NO device kernel. |
| SGD layout step (attract/repel/neg-sample) | **Backend kernel (NEW `umap_layout_step`)** | Host (epoch driver) | The one new device kernel; per-owner GATHER, cpu-MLIR-safe, frozen-subset-capable. Host drives the epoch loop + RNG state (mirrors `sgd_solve`). |
| negative-sample index draw + shuffle | Host (PRNG) | Kernel (consumes drawn indices) | Order-deterministic SplitMix64 host draws threaded into the kernel as a buffer (NO device RNG — backend-divergent, breaks D-05). |
| transform new-point init + frozen SGD | Host (init_graph_transform) + Backend kernel | — | Same `umap_layout_step` kernel, owner-set = new points only, training coords read-only (D-03). |

## Standard Stack

### Core (all already in-repo — NO new crates)
| Component | Path | Purpose | Why Standard |
|-----------|------|---------|--------------|
| `knn_graph<F>` prim | `crates/mlrs-backend/src/prims/knn_graph.rs` | Directed `(indices,distances)` `(n,k)`, 5 metrics, `include_self=false` | Phase-13 keystone, per-metric oracle-validated [VERIFIED: read in-repo] |
| `laplacian<F>` prim | `crates/mlrs-backend/src/prims/laplacian.rs` | `(L, dd)` symmetric-normalized Laplacian `I−D^-1/2 A D^-1/2` | Matches umap-learn spectral_layout EXACTLY [VERIFIED: read both sources] |
| `eig<F>` prim | `crates/mlrs-backend/src/prims/eig.rs` | Jacobi symmetric eigendecomp, DESCENDING, `MAX_DIM=64` cap | The v1 eig stack the ROADMAP/CONTEXT names; cap = the spectral fallback boundary [VERIFIED: read in-repo] |
| `recover<F>` host math | `crates/mlrs-algos/src/cluster/spectral.rs` | slice-smallest → /dd → sign-flip → (drop_first) → transpose | Shared spectral-family recovery; reuse for UMAP spectral init [VERIFIED: read in-repo] |
| `SplitMix64` / `permutation` | `crates/mlrs-backend/src/prims/rng.rs` | Seeded host PRNG + unbiased Fisher–Yates; `next_below` rejection sampling | The project's reproducible host PRNG (D-05 backbone); `pub` and reusable [VERIFIED: read in-repo] |
| two-pass SGD GATHER precedent | `crates/mlrs-backend/src/prims/sgd.rs` + `mlrs-kernels::sgd` | host epoch loop → per-batch GATHER kernel launch → readback | The exact shape the spike flag names as the `umap_layout_step` precedent [VERIFIED: read in-repo] |
| `Umap<F,S>` shell | `crates/mlrs-algos/src/manifold/umap.rs` | Full hyperparameter surface + builder + typestate; `Umap::new` = single-source defaults | Phase-12 shell; fill `fit`/`transform`, extend `Metric` [VERIFIED: read in-repo] |

### Supporting (in-repo)
| Component | Path | Purpose | When to Use |
|-----------|------|---------|-------------|
| `mlrs-kernels` direct kernels | `crates/mlrs-kernels/` | The one NEW `umap_layout_step` kernel lands here (feature-free, generic-over-`F`) | The single new device kernel |
| `capability::skip_f64_with_log` | `crates/mlrs-backend/src/capability.rs` | f64-on-rocm SKIP-with-log gate | Every f64 test path (cpu runs f64; rocm skips) |
| `load_npz` / `OracleCase` | `mlrs-core` | `.npz` fixture loader (4/8-byte float arrays only) | Every value-gate test reads committed umap-learn fixtures |
| `scripts/gen_oracle.py` | repo root | Seeded fixture generator (run in `/tmp` venv) | Add `gen_umap_*` generators; commit blobs |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Host smooth-kNN binary search | A device kernel | No parallelism win at UMAP-scale n; host f64 keeps it value-gatable to ≤1e-5 and dodges a cpu-MLIR kernel. Reject device. |
| Reusing `eig` (Jacobi, `MAX_DIM=64`) for spectral init | Lanczos/ARPACK-style partial eig | umap-learn uses `eigsh`/`lobpcg` (partial), but the repo has no Lanczos prim and the v2 convention is dense-Jacobi-under-cap + random fallback. Reuse the existing stack (Claude's discretion confirmed it transfers). |
| Host LM for a/b | scipy-style precomputed lookup | D-06 rejected lookup (fixed offset the tight property gate must absorb). Host LM hits ≤1e-5. |
| Host SplitMix64 draws fed into kernel | In-kernel device RNG | Device RNG is backend-divergent → breaks D-05 byte-identity. `rng.rs` doc explicitly bans device RNG. Host draws only. |

**Installation:** No new crates. The `a`/`b` LM fit and smooth-kNN binary search are hand-written host numerics (the project already hand-rolls Box–Muller, Fisher–Yates, Jacobi sweeps — consistent with "zero new compute dependencies", REQUIREMENTS oracle note).

## Package Legitimacy Audit

No external packages are installed by this phase. All compute reuses existing in-repo prims/kernels and hand-written host numerics (per the v3.0 "zero new compute dependencies" constraint, REQUIREMENTS.md oracle note). The only Python touched is `scripts/gen_oracle.py`, which already depends on `numpy`/`scipy`/`scikit-learn`; this phase adds **`umap-learn==0.5.12`** (pinned, per CONTEXT landmine) to the **build-time-only `/tmp` oracle venv** — never a runtime/test dependency, never committed to any manifest (fixtures are committed blobs; CI never runs the generator).

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| umap-learn (pin 0.5.12) | PyPI (oracle venv only) | mature (8+ yrs) | ~1M/mo | github.com/lmcinnes/umap | OK | Approved — build-time fixture-gen only, not a code dependency |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────── fit(X) ───────────┐
X (n×d) ──► knn_graph(include_self=false, metric)         │  [Phase-13 prim, BACKEND]
            └─► (knn_idx (n,k), knn_dist (n,k)) directed   │
                         │                                  │
                         ▼                                  │
   smooth_knn_dist: per-row binary search ──► rhos[n], sigmas[n]   [HOST f64]
                         │   target = log2(n_neighbors)*bandwidth
                         ▼
   compute_membership_strengths: val = exp(-(d-rho)/sigma)  ──► COO (rows,cols,vals)  [HOST]
                         │
                         ▼
   t-conorm union: G = mix*(A + Aᵀ − A∘Aᵀ) + (1-mix)*(A∘Aᵀ)   ──► symmetric fuzzy graph G  [HOST]
                         │
          ┌──────────────┼───────────────────────┐
          ▼ (init)        ▼ (a/b)                  ▼ (epochs_per_sample)
   spectral OR random   LM curve fit            make_epochs_per_sample(G, n_epochs)  [HOST]
   ├ if n ≤ 64: laplacian(symm-from-G) →         1/(1+a·d^(2b)) fit   [HOST LM]
   │   eig(Jacobi) → recover(slice-smallest/dd)
   │   → noisy_scale_coords(max=10, noise=1e-4)  [BACKEND prims + HOST]
   └ else: uniform(-10,10) via SplitMix64        [HOST]
          │
          ▼
   ┌─────────────────────────────────────────────────────────┐
   │  SGD LAYOUT (host epoch driver + NEW umap_layout_step kernel)  │
   │  for n in 0..n_epochs:                                          │
   │    alpha = initial_alpha*(1 - n/n_epochs)                       │
   │    host draws neg-sample indices (SplitMix64, order-determ.)    │
   │    umap_layout_step<F>(embedding, head, tail, eps_per_sample,   │
   │       neg_idx_buffer, a, b, gamma, alpha, owner_set)  [GATHER]  │
   └─────────────────────────────────────────────────────────┘
          │
          ▼
   embedding_ (n × n_components)  ──► property-gate vs umap-learn 0.5.12

   ┌────────── transform(X_new) (m×d) — SAME kernel, frozen subset ──────────┐
   X_new ─► knn_graph(X_new vs X_train, metric) ─► membership vs new dists    │
        ─► init_graph_transform: row-normalized weighted avg of trained coords │
        ─► umap_layout_step(owner_set = new points only, train coords RO,      │
             n_epochs=100, move_other=False)  ─► new_embedding (m×n_components) │
```
File-to-implementation mapping is in the Component Responsibilities table below — the diagram is data flow only.

### Component Responsibilities
| Component | New/Reuse | File |
|-----------|-----------|------|
| `Umap::fit` / `fit_transform` / `transform` bodies | NEW (replace trivial zeros) | `crates/mlrs-algos/src/manifold/umap.rs` |
| `Metric` enum extension (5 variants, mirror Phase-13 shape) | NEW | same file |
| smooth-kNN ρ/σ, membership, t-conorm, init_graph_transform, LM a/b | NEW host fns | same file (or a private sibling module `umap_internals.rs` if it grows) |
| `umap_layout_step<F>` SGD kernel | NEW device kernel | `crates/mlrs-kernels/src/` |
| KNN graph | REUSE | `prims/knn_graph.rs` |
| Laplacian + eig + recover (spectral init) | REUSE | `prims/laplacian.rs`, `prims/eig.rs`, `cluster/spectral.rs` |
| SplitMix64 / permutation | REUSE | `prims/rng.rs` |
| oracle generators `gen_umap_*` | NEW | `scripts/gen_oracle.py` |
| value-gate + property-gate tests | NEW | `crates/mlrs-algos/tests/umap_test.rs` |

### Recommended Project Structure
```
crates/mlrs-algos/src/manifold/
├── umap.rs            # estimator: fit/transform bodies + host stage fns + Metric(5)
└── (umap_internals.rs # OPTIONAL private split if umap.rs grows past ~700 lines)
crates/mlrs-kernels/src/
└── umap_layout.rs     # NEW umap_layout_step<F> GATHER kernel (+ re-export in lib.rs)
crates/mlrs-algos/tests/
└── umap_test.rs       # value-gate (5 metrics × stages) + property-gate + reproducibility
scripts/gen_oracle.py  # + gen_umap_<stage>_<metric>_<dtype> generators
tests/fixtures/        # committed umap_*.npz blobs
```

### Pattern 1: smooth-kNN ρ/σ binary search (deterministic, value-gated) — VERIFIED umap-learn 0.5.12
**What:** Per row `i`, find `sigma_i` so `Σ_j exp(-(max(0, d_ij − rho_i))/sigma_i) = target`.
**Constants (verified):** `target = log2(n_neighbors) * bandwidth` (bandwidth=1.0 default); `SMOOTH_K_TOLERANCE = 1e-5`; `MIN_K_DIST_SCALE = 1e-3`; binary-search `n_iter = 64`.
**rho (local_connectivity):** with `local_connectivity = 1.0`, `index = floor(1.0) = 1`, `interpolation = 1.0 − 0` → `rho_i = non_zero_dists[0]` (the nearest non-zero-distance neighbor). For non-integer `local_connectivity`: `rho_i = non_zero_dists[index−1] + interpolation*(non_zero_dists[index] − non_zero_dists[index−1])` (and `interpolation*non_zero_dists[0]` when `index==0`).
**sigma floor:** after the search, `sigma_i = max(sigma_i, MIN_K_DIST_SCALE * mean(d_i.))` (per-row mean) and the global fallback `MIN_K_DIST_SCALE * mean(all d)` when `rho_i <= 0`.
**When to use:** every fit/transform; host f64. **Order is load-bearing**: rho first, then binary search using `d − rho`.

### Pattern 2: membership strengths — VERIFIED
**Formula:** `val_ij = exp(-(max(0, d_ij − rho_i)) / sigma_i)`; `val = 1.0` when `d_ij − rho_i <= 0` or `sigma_i == 0`. Self edges excluded (already dropped by `include_self=false`). Emit as COO `(rows[i*k+j]=i, cols=knn_idx[i,j], vals=val)`.

### Pattern 3: t-conorm fuzzy union — VERIFIED
**Formula:** Let `A` be the directed sparse membership matrix (`(n,n)` from the COO). `prod = A ∘ Aᵀ` (elementwise). `G = set_op_mix_ratio*(A + Aᵀ − prod) + (1 − set_op_mix_ratio)*prod`. At `set_op_mix_ratio=1.0` → pure union `A + Aᵀ − A∘Aᵀ`. `G` is symmetric — this is UMAP's symmetrization (D-04). Build it as a host hashmap/COO merge over the `(n,k)` entries (n is small for the value-gate fixtures).

### Pattern 4: spectral init — VERIFIED, matches the existing `laplacian` prim EXACTLY
**umap-learn:** `sqrt_deg = sqrt(graph.sum(axis=0)); D = diag(1/sqrt_deg); L = I − D·graph·D`. This is the symmetric-normalized Laplacian — **identical** to `laplacian.rs` (`L = I − D^-1/2 A D^-1/2`, with the same zero-degree guard). Compute `k = n_components + 1` smallest eigenvectors, `order = argsort(eigenvalues)[1:k]` (drop the trivial ≈0), eigenvectors as embedding. Then `noisy_scale_coords(max_coord=10, noise=1e-4)`: `expansion = 10 / max|coords|; coords *= expansion; coords += normal(scale=1e-4)`.
**mlrs mapping:** feed the symmetric fuzzy graph `G` (as the affinity) → `laplacian(G, n)` → `eig` (DESCENDING; reverse to ascending) → `recover(..., drop_first=true)` → noisy-scale via SplitMix64 normal draws.
**Cap + fallback (Claude's discretion, v2 convention):** `eig` caps `n ≤ MAX_DIM = 64`. For `n > 64`, fall back to random init (`uniform(-10,10)`) — this is the "random fallback above the Jacobi size cap" the ROADMAP/CONTEXT names. The v2 `SpectralEmbedding` rejects `n > 64`; UMAP instead **falls back** (does not error). **Open: disconnected-component handling** — see §Open Questions Q1.

### Pattern 5: vertex-owner GATHER SGD layout kernel (the ONE new kernel) — cpu-MLIR-safe
**What:** One owner row per cube (`row = CUBE_POS_X`, work under `if row < n_owners { if UNIT_POS_X == 0 { … } }` — the `top_k`/`self_drop_gather` proven shape, NEVER bare `ABSOLUTE_POS` 1D). Per owner, loop its positive edges (attractive) and its negative samples (repulsive), accumulating the coordinate delta in an `F` accumulator read **within the same outer iteration** (no cross-sibling-loop accumulator — FINDING 002-B silent miscompile).
**Gradient formulas (VERIFIED, work in SQUARED distance `dist_squared`):**
- attractive: `if dist_squared > 0: grad = (-2·a·b·pow(dist_squared, b−1)) / (a·pow(dist_squared,b)+1) else grad = 0`
- repulsive: `grad = (2·gamma·b) / ((0.001 + dist_squared)·(a·pow(dist_squared,b)+1))` (and `grad=0` when `dist_squared==0`, skip when neg-index `k == j`)
- per-dim: `grad_d = clip(grad·(cur_d − other_d), −4.0, 4.0)`; when grad is the repulsive-zero/`dist²==0` case umap uses `grad_d = 4.0`.
- update: `cur_d += grad_d · alpha`; if `move_other`: `other_d += −grad_d · alpha`.
**`pow` lowers under cpu-MLIR:** the static `F::powf` form is launch-proven (Spike 001, Minkowski-p). Use `F::powf(dist_squared, b−1)` etc., NEVER the instance `x.powf()`.
**Clip without F::INFINITY:** implement `clip(v,-4,4)` with statement-form `if`: `let mut g=v; if g>hi {g=hi;} if g<lo {g=lo;}` — no `F::INFINITY`, no `max`/`min` intrinsic that might pull infinity.
**Frozen-subset mode (D-03):** the kernel takes `n_owners` (the contiguous owner-row count) and updates only `embedding[owner]`; non-owner neighbor coords are read-only GATHER targets. `fit`: owners = all n, `move_other = true` (two-sided). `transform`: owners = m new points (placed contiguously after the n frozen training rows in the buffer), `move_other = false`.
**Precedent:** mirror `sgd.rs`'s host epoch driver + two-pass GATHER kernel shape (the spike-flag-named precedent).

### Pattern 6: order-deterministic SplitMix64 PRNG plumbing (D-05) — HOST draws, NO device RNG
**What:** Reproducibility requires every random draw to be a fixed function of `random_state` and a deterministic counter — NEVER a device RNG (backend-divergent, banned in `rng.rs`).
- **init RNG:** spectral noise / random uniform drawn host-side via `SplitMix64::new(seed)` in a fixed traversal order; one upload.
- **edge-shuffle / epoch order:** if any shuffle is used, `permutation(seed, n)` (existing, unbiased Fisher–Yates).
- **negative-sampling RNG:** per epoch, per owner edge, draw `negative_sample_rate` indices host-side with a deterministic per-edge `SplitMix64` substream (e.g. seed mixed with `epoch*E + edge_id`), `next_below(n)` (unbiased). Pack into a per-epoch `neg_idx` device buffer the kernel GATHERs. This keeps the kernel RNG-free (cpu-MLIR-safe) AND order-deterministic.
**Note:** mlrs uses SplitMix64, umap-learn uses `tau_rand_int` Tausworthe — so mlrs coordinates ≠ umap coordinates by construction (the reason UMAP-03 is property-gated, REQUIREMENTS landmine). mlrs's OWN runs are byte-identical because the substream seeding is a pure function of `(random_state, epoch, edge)`.

### Pattern 7: transform new-point frozen path (UMAP-04, D-03) — VERIFIED
1. `knn_graph(X_new vs X_train)` — NOTE: this is X_new-against-X_train (m×n), NOT self-graph. The Phase-13 prim is X-vs-X; **transform needs a query-vs-train variant** — see §Open Questions Q2.
2. membership of new points: `smooth_knn_dist` + `compute_membership_strengths` on the new points' OWN knn distances (same constants).
3. `init_graph_transform`: `init[new_i] = Σ_j (graph_ij / rowsum_i) · embedding_train[col_j]` — row-normalized weighted average of trained neighbor coords.
4. `n_epochs`: `if self.n_epochs is None: n_epochs = 100` (else a reduced count); `move_other = False` (training embedding frozen).
5. drive `umap_layout_step` with owners = new points only, training coords read-only.

### Anti-Patterns to Avoid
- **Device RNG for negative sampling** → backend-divergent, breaks D-05. Host SplitMix64 draws into a buffer only.
- **`F::INFINITY` for clip bounds / sigma init** → cpu-MLIR panic at launch (project memory). Use finite literals + statement-`if` clamp; for the smooth-kNN search use a large finite `hi` (umap uses `np.inf` host-side, fine in host Rust f64 — only the DEVICE kernel bans it).
- **Cross-sibling-loop accumulator in the layout kernel** → SILENT miscompile (FINDING 002-B). Recompute per-owner positional values inside the consuming loop.
- **Bare `ABSOLUTE_POS` 1D launch** for the per-owner kernel → MLIR pass failure (FINDING 002-A). Use `CUBE_POS_X`/`UNIT_POS_X==0`.
- **Symmetrizing inside the KNN prim** — it emits directed only (D-04). UMAP owns the t-conorm union.
- **Comparing `embedding_` element-wise to umap-learn** — coordinates can't match (SplitMix64 ≠ Tausworthe). Property-gate only (UMAP-03 / Out-of-Scope table).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| K-nearest-neighbor graph | A new distance/top-k kernel | `prims::knn_graph` (Phase-13) | Per-metric oracle-validated, self-drop-by-identity, memory-gated, cpu-MLIR-safe |
| Symmetric-normalized Laplacian | A hand Laplacian map | `prims::laplacian` | umap's `I−D^-1/2 A D^-1/2` is byte-identical to it; zero-degree guard already correct |
| Symmetric eigendecomposition | A new eig kernel | `prims::eig` (Jacobi) + `recover` | Existing, descending-sorted, `MAX_DIM=64` cap = the spectral fallback boundary |
| Seeded reproducible PRNG | A new RNG | `prims::rng::SplitMix64` / `permutation` | Byte-frozen stream, unbiased `next_below`, the D-05 backbone; device RNG is banned |
| SGD host epoch driver | A bespoke loop | mirror `prims::sgd::sgd_solve` shape | The spike-flag-named precedent; validate→loop→launch→readback |
| `.npz` fixture loading | A parser | `mlrs_core::load_npz` / `OracleCase` | The established oracle path (4/8-byte float arrays) |

**Key insight:** UMAP at the value-gate fixture scale (small n) is mostly host orchestration over prims you already trust. The ONLY genuinely new device code is `umap_layout_step`. Everything deterministic is reuse + host math, which is also why it is value-gatable to ≤1e-5: the host f64 path matches umap-learn's own host (numpy/numba f64) intermediates without device-reduction-order drift.

## Common Pitfalls

### Pitfall 1: cross-sibling-loop accumulator in the layout kernel (SILENT miscompile)
**What goes wrong:** writing the per-owner coordinate delta in one `while` and reading it in a separate sibling `while` compiles, launches, and returns plausible-but-wrong coords. Catches nothing in a happy-path test.
**Why:** FINDING 002-B — cpu-MLIR never propagates the value across sibling loops.
**How to avoid:** accumulate the delta inside the SAME loop that consumes it (the `top_k`/`self_drop_gather` self-contained-nested pattern). Apply the update at the end of each owner's loop body.
**Warning signs:** layout "works" but trustworthiness sits far below umap-learn even with correct gradients — suspect the accumulator before the math.

### Pitfall 2: `F::INFINITY` anywhere in a device kernel
**What goes wrong:** panic at launch on cpu-MLIR.
**Why:** project memory landmine (banned constant).
**How to avoid:** clip with finite literals + statement-`if`; the smooth-kNN search's `hi = inf` is HOST-side f64 only (fine). Never let infinity reach the `umap_layout_step` kernel.

### Pitfall 3: spectral-init operation ORDER and sign convention
**What goes wrong:** wrong order (sign-flip before /dd, or keeping the trivial eigenvector) shifts the init and fails the ≤1e-5 value-gate against umap's spectral coords.
**Why:** the recovery is order-sensitive (documented in `spectral.rs`: slice-smallest → /dd → sign-flip → drop-first).
**How to avoid:** reuse `recover(..., drop_first=true)` verbatim; umap's `argsort(eigenvalues)[1:k]` IS drop-first. Verify the deterministic sign-flip convention matches umap's (umap applies no sign-flip in spectral_layout — see §Open Questions Q3; the value-gate may need to compare up-to-sign per column).

### Pitfall 4: minkowski-p `p` propagation and the `Metric` enum shape
**What goes wrong:** UMAP's `Metric` diverging from the Phase-13 prim's `Metric` (which carries `Minkowski { p: f64 }`) forces a lossy conversion.
**Why:** two enums to keep in sync.
**How to avoid:** mirror the Phase-13 `Metric` shape exactly (Claude's discretion confirms this); map UMAP's `metric=`/`p` straight onto `knn_graph::Metric`.

### Pitfall 5: transform needs query-vs-train KNN, but the prim is X-vs-X
**What goes wrong:** calling `knn_graph` for transform self-graphs the new points instead of querying them against training.
**Why:** the Phase-13 prim only does X-vs-X (self).
**How to avoid:** see §Open Questions Q2 — either add a query-vs-train path or compose `distance` + `top_k` directly in the estimator (no self-drop needed since new≠train). Resolve in the spike.

### Pitfall 6: dumping the WRONG umap-learn intermediate (squared vs true distance, graph orientation)
**What goes wrong:** the layout works in `dist_squared`; the membership works in TRUE metric distance; the KNN oracle stores TRUE distance. Mixing them fails the gate.
**Why:** umap uses different distance spaces per stage.
**How to avoid:** the value-gate fixtures must dump umap-learn's actual per-stage arrays (graph COO `rows/cols/vals`, `sigmas`, `rhos`, `a`, `b`, spectral coords) — NOT recompute. See §Validation Architecture.

## Code Examples

### Membership + t-conorm (host) — VERIFIED umap-learn 0.5.12
```python
# Source: github.com/lmcinnes/umap release-0.5.12 umap/umap_.py
# compute_membership_strengths
val = 1.0 if (knn_dists[i,j] - rhos[i] <= 0.0 or sigmas[i] == 0.0) \
          else np.exp(-((knn_dists[i,j] - rhos[i]) / sigmas[i]))
# fuzzy_simplicial_set union
prod_matrix = result.multiply(transpose)
result = (set_op_mix_ratio * (result + transpose - prod_matrix)
          + (1.0 - set_op_mix_ratio) * prod_matrix)
```

### a/b curve fit target (host LM, D-06) — VERIFIED
```python
# Source: umap/umap_.py find_ab_params
def curve(x, a, b):  return 1.0 / (1.0 + a * x ** (2 * b))
xv = np.linspace(0, spread * 3, 300)
yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
# scipy.optimize.curve_fit(curve, xv, yv)  ->  a, b   (mlrs ports this LM in host Rust f64)
```

### SGD inner gradient (device kernel body, SQUARED distance) — VERIFIED
```text
# Source: umap/layouts.py optimize_layout_euclidean (verified)
# attractive (positive edge j->k):
if dist_squared > 0:  grad = (-2*a*b*pow(dist_squared, b-1)) / (a*pow(dist_squared,b)+1)
else:                 grad = 0
# repulsive (negative sample, n_neg = int((n - eont_neg[i])/epn[i])):
if dist_squared > 0:  grad = (2*gamma*b) / ((0.001+dist_squared)*(a*pow(dist_squared,b)+1))
elif j == k:          skip
else:                 grad = 0
grad_d = clip(grad*(cur_d - other_d), -4, 4)   # grad_d = 4.0 in the dist²==0 repulsive branch
cur_d += grad_d * alpha
if move_other: other_d -= grad_d * alpha
alpha = initial_alpha * (1 - n/n_epochs)
```

### Reuse: spectral init via existing prims (host orchestration)
```rust
// Source: crates/mlrs-algos/src/cluster/spectral_embedding.rs (the pattern to mirror)
let (l, dd) = laplacian::<F>(pool, &g_affinity, n)?;     // umap L = I - D^-1/2 G D^-1/2
let (w_desc, v_desc) = eig::<F>(pool, &l, n, Some(l_out))?;  // DESCENDING, Jacobi, n<=64
let init = recover::<F>(&v_host, &dd_host, n, n_components, /*drop_first=*/true);
// then noisy_scale_coords(max=10, noise=1e-4) via SplitMix64 normal draws
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Element-wise coordinate match to reference | Property/structural gate (trustworthiness/kNN-overlap/ARI) | UMAP's stochastic SGD + non-portable PRNG | Coordinates are not a valid oracle; gate on structure (UMAP-03) |
| Spectral via ARPACK `eigsh` (partial) | Dense Jacobi under `n≤64` cap + random fallback above | mlrs has no Lanczos prim | Spectral init only for small n; larger n uses random init (matches umap's own fallback when eig fails to converge) |

**Deprecated/outdated:** none for this phase. umap-learn 0.5.12 is the pinned reference; do NOT consult newer umap behavior (densmap defaults, etc.) — pin to 0.5.12 source.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` integration tests (`crates/mlrs-algos/tests/umap_test.rs`), oracle `.npz` fixtures via `mlrs_core::load_npz` |
| Config file | none — workspace cargo; fixtures in `tests/fixtures/` |
| Quick run command | `cargo test -p mlrs-algos --features cpu --test umap_test` |
| Full suite command | `cargo test -p mlrs-algos --features cpu` (then opportunistic `--features rocm` for f32) |
| Fixture generator | `python3 scripts/gen_oracle.py` in a `/tmp` venv with `numpy scipy scikit-learn umap-learn==0.5.12` (PEP 668) |

### Phase Requirements → Test Map
| Req | Behavior | Test Type | Gate | Automated Command |
|-----|----------|-----------|------|-------------------|
| UMAP-02 | smooth-kNN ρ/σ per metric | value ≤1e-5 (f64) | dump umap `sigmas`,`rhos` | `cargo test -p mlrs-algos --features cpu --test umap_test -- smooth_knn` |
| UMAP-02 | membership + t-conorm union per metric | value ≤1e-5 | dump umap graph COO `rows/cols/vals` | `… -- fuzzy_union` |
| UMAP-02 | spectral init per metric (n≤64) | value ≤1e-5 (up-to-sign per col, see Q3) | dump umap `spectral_layout` coords | `… -- spectral_init` |
| UMAP-01/02 | a/b LM curve fit | value ≤1e-5 | dump umap `a`,`b` for (min_dist,spread) grid | `… -- ab_fit` |
| UMAP-03 | SGD layout structural | property (trustworthiness/kNN-overlap ≥ umap−ε, ARI in band) | dump umap embedding + labels | `… -- layout_property` |
| UMAP-03 | same-`random_state` reproducibility | byte-identical across 2 runs (per backend,dtype) | self-check, no oracle | `… -- reproducible` |
| UMAP-04 | transform new points | property sub-gate (trustworthiness of new pts ≥ umap−ε) | dump umap transform embedding | `… -- transform_property` |
| UMAP-01 | defaults / build validation | existing shell tests (keep green) | — | already present |

### Per-stage × per-metric fixture matrix (D-02)
For each metric ∈ {euclidean, manhattan, cosine, chebyshev, minkowski(p)}, generate a committed `.npz` dumping umap-learn 0.5.12 internals on a fixed seed/data:
- `gen_umap_fuzzy_<metric>_<dtype>`: `X`, `knn_idx`, `knn_dist`, `sigmas (n)`, `rhos (n)`, graph `rows/cols/vals` (COO), `set_op_mix_ratio`, `local_connectivity`.
- `gen_umap_spectral_<metric>_<dtype>`: the symmetric graph + umap `spectral_layout` coords (n≤64 design).
- `gen_umap_ab_<dtype>`: a grid of `(min_dist, spread) → (a, b)` from `find_ab_params` (metric-independent — one fixture).
- `gen_umap_layout_<metric>_<dtype>`: `X`, umap `embedding_`, true labels (for ARI), fixed `random_state`/`n_epochs` — the property-gate reference.
- `gen_umap_transform_<metric>_<dtype>`: `X_train`, `X_new`, fitted `embedding_`, umap transform output — the transform sub-gate reference.

**How to dump:** call umap-learn's internal functions directly (`from umap.umap_ import fuzzy_simplicial_set, find_ab_params, smooth_knn_dist`; `from umap.spectral import spectral_layout`) — do NOT recompute in numpy (Pitfall 6). Store only 4/8-byte float arrays (`load_npz` constraint — encode indices as float, metric tag in the filename, per the `gen_knn_metric` precedent).

### Property-gate metric definitions (compute in-repo, host)
- **trustworthiness(X, embedding, k):** standard sklearn `manifold.trustworthiness` formula `T = 1 − (2/(nk(2n−3k−1)))·Σ_i Σ_{j∈U_i^k}(r(i,j)−k)` — port the host formula (no sklearn at test time).
- **kNN-overlap:** fraction of each point's k high-D neighbors retained among its k low-D neighbors, averaged.
- **downstream-ARI:** run the existing kmeans on both umap's and mlrs's embedding, compare cluster labels via Adjusted Rand Index (band check).
**Calibration (Spike flag item 2):** on the FIRST fixture run, compute mlrs and umap scores on identical data; set `ε` = (umap_score − mlrs_score) margin with a small safety buffer (kept tight, D-04). Record the calibrated `ε`/band in VALIDATION.md so the gate is reproducible.

### Sampling Rate
- **Per task commit:** the quick command on the touched stage's test.
- **Per wave merge:** full `umap_test` (cpu f64).
- **Phase gate:** full suite green (cpu f64; rocm f32 opportunistic) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/mlrs-algos/tests/umap_test.rs` — extend with the value-gate (5 metrics × 4 stages), property-gate, reproducibility, transform tests (RED-by-design referencing the not-yet-real fit body).
- [ ] `scripts/gen_oracle.py` — add the `gen_umap_*` generators; regen in the `/tmp` venv with `umap-learn==0.5.12`; commit blobs.
- [ ] Property-gate metric helpers (trustworthiness/kNN-overlap/ARI) — host implementations (reuse existing kmeans/ARI helpers if present).
- [ ] Spike harness: confirm `umap_layout_step` launches under `--features cpu` + calibrate thresholds (Spike flag).

## Security Domain

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Geometry + hyperparameter validation BEFORE any device launch (`validate_geometry` precedent in `knn_graph.rs`/`laplacian.rs`); reject `n_neighbors ≥ n`, `n_components ≥ 1`, `min_dist ≤ spread` (shell build-time), Minkowski `p ≥ 1`. Typed `PrimError`/`AlgoError`, no OOB device read. |
| V6 Cryptography | yes | RNG is NON-crypto by design (`SplitMix64`, seeded from the caller's `random_state` u64) — NEVER `OsRng`/`rand` crate (the `rng.rs` ASVS-V6 contract). Reproducibility, not secrecy. |
| V2/V3/V4/V7+ | no | No auth/session/access-control surface (numeric library). |

### Known Threat Patterns for {Rust + CubeCL device kernels}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device GATHER (bad `n_owners`/neg-index) | Tampering / DoS | Host-validate all launch geometry before `unsafe { ArrayArg::from_raw_parts }`; kernel bounds-checks `row < n` and index `< n_vertices` (the `self_drop_gather` precedent) |
| Unbounded LM iteration (a/b non-convergence) | DoS | Iteration cap + `NotConverged`-style typed error (the `eig`/`sgd` `MAX_SWEEPS`/`max_iter` precedent) |
| Biased modulo in negative sampling | Correctness | `SplitMix64::next_below` rejection sampling (NEVER `% n`) — already the `rng.rs` contract |
| Divide-by-zero (zero-degree node, sigma=0) | DoS/NaN | umap's typed-zero guards (`val=1` when sigma=0; `dd=1` for isolated node — already in `laplacian.rs`); the `0.001 + dist²` fudge in the repulsive grad |

## Environment Availability
| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| cubecl-cpu (MLIR) f64 | primary correctness gate | ✓ | 0.10 | — (the gate) |
| rocm gfx1100 f32 | GPU gate (opportunistic) | ✓ | ROCm 7.1.1 | f64-on-rocm SKIPS-with-log |
| Phase-13 `knn_graph` prim | KNN stage | ✓ | in-repo | — |
| `laplacian`/`eig`/`recover` | spectral init | ✓ | in-repo | random init above n=64 |
| numpy/scipy/sklearn (`/tmp` venv) | fixture gen | ✗ (PEP 668; build via venv) | — | `/tmp` venv install |
| umap-learn 0.5.12 (`/tmp` venv) | umap fixture gen | ✗ (install in venv) | pin 0.5.12 | none — REQUIRED for value/property fixtures |
| maturin/pyarrow (PyO3 live test) | NOT this phase | ✗ | — | Phase 16; UMAP PyO3 wrap deferred |

**Missing dependencies with no fallback:** umap-learn 0.5.12 in the oracle venv (required to generate the committed fixtures; a one-time build-side install, not a runtime dep).
**Missing dependencies with fallback:** Jacobi eig above n=64 → random init.

## Assumptions Log
| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Spectral-init disconnected-component handling follows the v2 convention (single-component assumption / random fallback) and does NOT need umap's `multi_component_layout` for the value-gate fixtures | Pattern 4 / Q1 | If a value-gate fixture is disconnected, mlrs spectral coords diverge from umap's per-component layout → gate fails. Mitigate: design fixtures connected, OR implement multi-component. ASSUMED — confirm in spike. |
| A2 | The transform query-vs-train KNN can be composed in-estimator from `distance` + `top_k` (no new prim) since no self-drop is needed (new ≠ train) | Pattern 7 / Q2 | If the prim's tiling/self-drop is entangled, transform needs a small prim extension. ASSUMED — confirm in spike. |
| A3 | umap's `spectral_layout` applies NO deterministic sign-flip, so the value-gate compares spectral coords up-to-sign per column | Pattern 4 / Pitfall 3 / Q3 | Wrong sign convention fails the ≤1e-5 gate. ASSUMED from source read — confirm by dumping and comparing. |
| A4 | `n_epochs=None` → 500 for n≤10000 (verified), and the small-fixture property gate uses the same default | Discretion / verified | Low risk — verified from source; threshold is 10000. |
| A5 | Host LM (Gauss-Newton with LM damping) on the 300-point curve hits a/b ≤1e-5 vs scipy `curve_fit` | D-06 / §Standard Stack | If LM doesn't converge tightly, a/b gate fails. Mitigate: the curve is smooth/well-conditioned; LM with analytic Jacobian is standard. ASSUMED — validate on first fixture. |

## Open Questions
1. **Disconnected-component spectral init.** umap-learn's `spectral_layout` delegates to `multi_component_layout` when the fuzzy graph has >1 connected component. The v2 `SpectralEmbedding` assumes connected.
   - What we know: the existing `laplacian`+`eig`+`recover` path handles the single-component case exactly; the zero-degree guard prevents NaN.
   - What's unclear: whether any value-gate fixture will be disconnected, and whether matching umap's per-component layout is required for ≤1e-5.
   - Recommendation: design the spectral value-gate fixtures to be CONNECTED (n≤64, dense-enough KNN graph); defer `multi_component_layout` (raise to user only if a realistic fixture forces it — Deferred Idea). Confirm in the spike.
2. **Transform query-vs-train KNN.** The Phase-13 prim is X-vs-X. transform needs X_new-vs-X_train.
   - Recommendation: compose `distance(X_new, X_train)` + `top_k(k)` directly in the estimator (no self-drop — new≠train), or add a thin query-vs-train arg to the prim. Decide in the spike (A2).
3. **Spectral sign convention.** umap's `spectral_layout` returns raw eigenvectors (no `_deterministic_vector_sign_flip`); `recover` applies a sign-flip.
   - Recommendation: either compare the spectral value-gate up-to-sign per column, OR skip the sign-flip for the UMAP spectral path (a `recover` flag). Resolve by dumping and diffing (A3).
4. **Property-gate margins ε / ARI band.** Calibrated on first fixture run (D-04 / Spike flag item 2).
   - Recommendation: run mlrs vs umap on identical seeded data; set tight ε with a small buffer; record in VALIDATION.md.

## Sources
### Primary (HIGH confidence)
- umap-learn 0.5.12 source — `umap/umap_.py` (smooth_knn_dist constants `SMOOTH_K_TOLERANCE=1e-5`/`MIN_K_DIST_SCALE=1e-3`/`n_iter=64`, membership formula, t-conorm union, `find_ab_params` curve + `linspace(0, spread*3, 300)`, `n_epochs` 500/200 @ 10000, `make_epochs_per_sample`, `noisy_scale_coords` max=10/noise=1e-4, transform `n_epochs=100`/`move_other=False`/`init_graph_transform`) [VERIFIED: fetched from github release-0.5.12]
- umap-learn 0.5.12 `umap/layouts.py` (attractive/repulsive grad formulas in dist_squared, clip ±4, alpha decay, neg-sample schedule, `tau_rand_int`) [VERIFIED: fetched]
- umap-learn 0.5.12 `umap/spectral.py` (`L = I − D·graph·D` symmetric-normalized, `k=dim+1`, `order=argsort(eig)[1:k]`, multi-component delegation) [VERIFIED: fetched]
- umap-learn 0.5.12 `umap/utils.py` (`tau_rand_int` Tausworthe 3×uint32) [VERIFIED: fetched]
- In-repo prims (read this session): `knn_graph.rs`, `laplacian.rs`, `eig.rs`, `rng.rs`, `sgd.rs`, `spectral_embedding.rs`, `umap.rs`, `umap_test.rs`, `scripts/gen_oracle.py` [VERIFIED: read]
- `spike-findings-mlrs` SKILL + references (cpu-MLIR landmines: 002-A launch shape, 002-B silent cross-sibling miscompile, banned constants) [VERIFIED: read]

### Secondary (MEDIUM confidence)
- CONTEXT.md / REQUIREMENTS.md / ROADMAP.md (D-01..06, UMAP-01..04, Spike flag) [CITED: in-repo planning docs]

### Tertiary (LOW confidence)
- none — all algorithm claims verified against pinned source.

## Metadata
**Confidence breakdown:**
- Standard stack: HIGH — every reuse prim located and read; no new crates.
- Architecture / formulas: HIGH — verified against umap-learn 0.5.12 source line-level.
- Spectral sign + disconnected handling: MEDIUM — sign convention and multi-component require spike confirmation (Q1/Q3).
- Transform query-vs-train KNN: MEDIUM — composition path assumed, spike-confirmable (Q2).
- Property-gate thresholds: LOW-by-design — calibrated empirically on first fixture run (Spike flag).

**Research date:** 2026-06-23
**Valid until:** stable (umap-learn pinned 0.5.12; in-repo prims stable) — ~30 days; re-verify if Phase 13 prims change.
