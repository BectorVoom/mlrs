# Phase 15: HDBSCAN - Research

**Researched:** 2026-06-24
**Domain:** Density-based hierarchical clustering (HDBSCAN*); device distance front-end + host tree back-end; exact-label oracle reproduction vs `sklearn.cluster.HDBSCAN`
**Confidence:** HIGH (the entire host back-end algorithm was read VERBATIM from the installed `sklearn 1.9.0` `_hdbscan/*.pyx` source — the authoritative oracle; GLOSH from the `hdbscan` library source)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Expose ALL 5 feature-space metrics + `Precomputed`. `metric=` covers euclidean, manhattan (L1), cosine, chebyshev (L∞), minkowski-p (all via the Phase-13 prim, `include_self=true`) PLUS a new `Metric::Precomputed` variant. The shell's `Metric` enum (currently `Euclidean`-only) is extended to all 6 this phase.
- **D-02:** `precomputed` via a `Metric::Precomputed` enum variant. When set, `fit` interprets `X` as a square **n×n distance matrix** (shape `(n,n)`), skipping the KNN/distance device front-end. One `Fit` impl (no separate `fit_precomputed`). Planner: validate squareness (and document symmetry expectation) before the back-end runs.
- **D-03:** Exact-up-to-permutation on EVERY metric (not just precomputed f64). `labels_` match `sklearn.cluster.HDBSCAN` exactly up to permutation with `-1` pinned, for precomputed AND all 5 feature-space metrics. ⚠ RISK: for non-euclidean brute-KNN metrics, distance ties can be ordered differently than the oracle internally, flipping an MST edge and cascading into a label difference — all-metric exactness is physically fragile and hinges entirely on D-04.
- **D-04:** Match the ORACLE's internal MST tie-break exactly. Reverse-engineer and replicate `sklearn.cluster.HDBSCAN` / `hdbscan` 0.8.44 internal MST edge-tie ordering. Couples the host MST to oracle internals; the pre-planning spike validates on a deliberately tie-heavy fixture and locks the documented deterministic rule.
- **D-05:** HOLD THE EXACT LINE — non-negotiable. If the spike finds a metric cannot hit exact-up-to-perm even with the oracle-matched tie-break, iterate the algorithm/tie-break until it passes — do NOT auto-demote to a band gate, do NOT silently drop the metric. ⚠ The pre-planning exactness spike is a TRUE gate — a metric that proves un-exactable is a phase blocker, surface early.
- **D-06:** Treat the scores as ≤1e-5 value gates, escalate if divergent. Gate `probabilities_` and GLOSH `outlier_scores_` to ≤1e-5 (abs+rel); if the spike/first-fixture run shows the scores diverge beyond that for genuine algorithmic/float-order reasons, escalate to the user rather than silently widening.
- **D-07:** Per-score oracle hierarchy. `probabilities_` value-gated vs `sklearn.cluster.HDBSCAN` (primary, zero new dep) with `hdbscan` 0.8.44 cross-check; GLOSH `outlier_scores_` (HDBS-03) value-gated vs the `hdbscan` 0.8.44 library.
- **D-08:** `store_centers` centroid AND medoid, value-gated vs sklearn. `'centroid'` → `centroids_` (weighted mean per cluster) AND `'medoid'` → `medoids_` (min-total-distance cluster member), both value-gated ≤1e-5 vs `sklearn.cluster.HDBSCAN`.
- **D-09:** Full selection-knob surface, ALL under the exact gate. Implement `cluster_selection_method` 'eom' AND 'leaf', `cluster_selection_epsilon` (>0 merge logic), `max_cluster_size` bound, and `alpha` scaling — with oracle fixtures exercising **non-default** values of each, all held to the exact-label gate. Resolves the shell's deferred validation TODO (min_samples >= 1 when Some; max_cluster_size 0=unbounded else >= min_cluster_size).

### Claude's Discretion
- **Host MST algorithm internals** — Prim's is named; the exact data structures (priority queue vs dense scan, single-linkage union-find shape) are the planner's, provided D-04's oracle-matched tie-break holds.
- **Memory / PoolStats gate for the n×n mutual-reachability** — follow the established per-phase build-failing PoolStats gate convention (query-axis-tiled, never full n×n device-resident where avoidable); planner sets the exact assertion.
- **Condensed-tree / stability-extraction data structures** (cluster hierarchy representation, EoM vs leaf selection traversal) — planner's choice that hits the exact-label gate.
- **Edge cases** (all-noise result, single point, single cluster, fewer than `min_cluster_size` points) — match sklearn behavior; planner confirms against the oracle, surfaces only if sklearn's behavior is ambiguous.
- **`min_samples=None → min_cluster_size`** default resolution is already in the shell's `new()`/`build()`; keep it.

### Deferred Ideas (OUT OF SCOPE)
- PyO3 wrap of `Hdbscan` + builder-retrofit sweep + Python sklearn shim (SHIM-01/02/03) — Phase 16.
- `approximate_predict` / `membership_vector` (new-point predict), condensed-tree / dendrogram plot objects — out of scope in REQUIREMENTS.md.
- Approximate / NN-Descent / tree KNN-graph build, custom/callable metrics, native sparse path — out of scope in REQUIREMENTS.md.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| HDBS-01 | `fit`/`fit_predict` → `labels_` (`-1`=noise) + `probabilities_` ∈[0,1] with sklearn-named hyperparameters/defaults; device front-end (core distances + mutual-reachability) + host back-end (MST → single-linkage → condensed tree → EoM/leaf stability). | Full algorithm read verbatim from sklearn source — see Architecture Patterns (core dist, mutual reach, MST, condense, stability, EoM/leaf, labelling, probabilities). Defaults table confirmed against shell `new()` + sklearn. |
| HDBS-02 | `labels_` match `sklearn.cluster.HDBSCAN` (cross-check `hdbscan` 0.8.44) exactly up to permutation, `-1` pinned; label-perm helper extended `-1→-1`; MST tie-break stable + documented. | D-04 tie-break section gives the EXACT sklearn MST rules (two algorithms — `mst_from_data_matrix` for FAST_METRICS; `mst_from_mutual_reachability` for cosine+precomputed) + `argsort` instability landmine. `label_perm` extension spec'd. |
| HDBS-03 | per-point `outlier_scores_` (GLOSH) — differentiator vs sklearn, gated vs `hdbscan` library. | GLOSH algorithm read from hdbscan library source — `outlier_scores` + upward-death-propagation; formula `(λ_max − λ_point)/λ_max`. sklearn has NO GLOSH → oracle is `hdbscan` 0.8.44 only. |
| HDBS-04 | `store_centers` (`'centroid'`/`'medoid'`) → `centroids_`/`medoids_` (sklearn parity). | `_weighted_cluster_center` read verbatim: centroid = `np.average(data, weights=probabilities)`; medoid = `argmin(sum(pairwise_distances·strength, axis=1))`. |
</phase_requirements>

## Summary

The Phase-15 risk is concentrated almost entirely in **HDBS-02's exact-label gate (D-03/D-04/D-05)**, and this research **de-risks it decisively**: the authoritative oracle, `sklearn.cluster.HDBSCAN`, is installed in this environment (sklearn 1.9.0) with its full Cython source readable at `~/.local/lib/python3.12/site-packages/sklearn/cluster/_hdbscan/{_linkage,_reachability,_tree}.pyx`. **The entire host back-end algorithm — core distances, mutual reachability, both MST variants, single-linkage union-find, condensed-tree, stability, EoM/leaf/epsilon selection, point labelling, and probabilities — was read verbatim** and is reproduced below. The GLOSH `outlier_scores_` (HDBS-03), which sklearn lacks, was read from the `hdbscan` library source. There is no algorithmic ambiguity left; the planner can target a line-for-line port.

The **single most important finding for D-04** is that sklearn dispatches to **TWO DIFFERENT MST algorithms** depending on metric, and the back-end sorts MST edges with **`np.argsort` (quicksort, UNSTABLE)**. With `algorithm='auto'` (the default we must match): euclidean/l1/l2/chebyshev/minkowski are `FAST_METRICS` → routed to `_hdbscan_prims` → `mst_from_data_matrix` (Prim's with a dynamic source-tracking update). **Cosine is NOT a FAST_METRIC**, and `precomputed` is always brute → both route to `_hdbscan_brute` → `mst_from_mutual_reachability` (dense Prim's, `np.argmin` first-min tie-break). mlrs must replicate BOTH, plus the unstable `argsort` of MST edges by weight before single-linkage. Distance ties on non-euclidean metrics are exactly where exactness is fragile (D-03 risk) — the pre-planning spike must validate on a tie-heavy + duplicate-point fixture per metric.

**Primary recommendation:** Port the sklearn `_hdbscan` Cython algorithm to host Rust **line-for-line** (it is pure, deterministic, integer/float scalar code — no kernels needed for the back-end), feeding it core distances + a mutual-reachability matrix produced by a SharedMemory-free GATHER device front-end reusing the Phase-13 KNN prim (`include_self=true`). Run the D-04/D-05 exactness spike FIRST on a tie-heavy + duplicate-point fixture across all 6 metrics before committing the exact-label gate.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Core distances (kth-NN dist) | Device (KNN prim) | Host (precomputed: kth-smallest per row) | Phase-13 prim `include_self=true` returns ascending `(n,k)` distances; core = column `min_samples-1`. Precomputed bypasses device. |
| Mutual-reachability `max(core_i,core_j,d_ij)` | Device GATHER kernel (feature metrics) / Host (precomputed) | — | SharedMemory-free per-element kernel; precomputed reads X directly. **HDBSCAN owns symmetrization** (Phase-13 emits directed only). |
| Prim's MST (2 variants) | Host (CPU) | — | Sequential, data-dependent, branchy — fights GPU atomics (the deliberate dodge). Pure scalar Rust. |
| `argsort` MST edges by weight | Host | — | Unstable quicksort — D-04 critical; must replicate NumPy semantics. |
| Single-linkage (UnionFind) | Host | — | Sequential merge with relabeling; `fast_find` path-compression. |
| Condensed tree / stability / EoM-leaf / labelling / probabilities | Host | — | Tree recursion + dict bookkeeping; pure host. |
| GLOSH `outlier_scores_` | Host | — | Condensed-tree pass with upward death propagation. |
| `centroids_`/`medoids_` | Host | Device (optional pairwise for medoid) | `np.average` weighted mean / `argmin` weighted total-distance; small, host-side. |
| label_perm `-1→-1` pin | Host (`mlrs-core`) | — | Test-side helper extension. |

## Standard Stack

This phase adds **NO new compute dependencies** (REQUIREMENTS oracle note: "Zero new compute dependencies"). The host back-end is pure Rust ported from the sklearn algorithm; the device front-end reuses existing prims.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `mlrs-backend::prims::knn_graph` | in-repo (Phase 13) | Core distances + neighbor distances (`include_self=true`) | [VERIFIED: codebase] PRIM-11 landed, per-metric oracle-gated, `include_self` supported. |
| `cubecl` / `cubecl-cpu` / `cubecl-hip` | 0.10.x (pinned) | Mutual-reachability GATHER kernel; f64 on cpu-MLIR, f32 on rocm | [VERIFIED: codebase] existing backend pin (CLAUDE.md / memory: cubecl ^0.10.0). |
| `mlrs-kernels` | in-repo | Home for the new mutual-reachability `#[cube(launch)]` kernel | [VERIFIED: codebase] distance kernels (manhattan/chebyshev/minkowski/self_drop_gather) already live here. |
| `npyz` | 0.9.x (existing) | Read `.npz` oracle fixtures host-side, no Python at test time | [VERIFIED: codebase] `mlrs-core::oracle` already uses `NpzArchive`. |

### Supporting (fixture generation only — `/tmp` venv, NOT a runtime/test dep)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scikit-learn` | **1.9.0** (env-installed) | PRIMARY oracle for `labels_`/`probabilities_`/`centroids_`/`medoids_` (D-07) | [VERIFIED: env] `python3 -c "import sklearn; sklearn.__version__" → 1.9.0`; `HDBSCAN` present in `sklearn.cluster`. Fixture-gen only. |
| `hdbscan` | **0.8.44** (pin) | GLOSH `outlier_scores_` oracle (HDBS-03) + `labels_`/`probabilities_` cross-check (D-07) | [CITED: pypi.org/project/hdbscan] 0.8.44 has Python 3.12 manylinux wheels. NOT installed in env — install in `/tmp` venv for fixture regen. |
| `numpy` | ≥1.26 | `default_rng(seed)` fixtures, `savez` | [VERIFIED: codebase] gen_oracle.py convention; PEP-668 → `/tmp` venv. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Port sklearn `_hdbscan` line-for-line | Port the `hdbscan` library's `_hdbscan_tree.pyx` | sklearn is the PRIMARY oracle (D-07, zero-dep) and is the one we must match exactly for labels; port sklearn. Use hdbscan only as the GLOSH source + cross-check. |
| Dense n×n mutual-reachability on device | Sparse / kNN-graph MST | Out of scope (REQUIREMENTS: "Native sparse KNN-graph path — densify at ingress"; "approximate/NN-Descent KNN excluded"). Brute exact only. |
| Host MST in Rust | GPU tree-atomics MST | The deliberate dodge — REQUIREMENTS/ROADMAP/CONTEXT all mandate host back-end to avoid the tree-atomics wall. |

**Installation (fixture regen only, in a throwaway `/tmp` venv per project memory):**
```bash
python3 -m venv /tmp/hdbscan-oracle-venv
/tmp/hdbscan-oracle-venv/bin/pip install "numpy>=1.26" "scikit-learn==1.9.0" "hdbscan==0.8.44"
/tmp/hdbscan-oracle-venv/bin/python scripts/gen_oracle.py   # writes committed .npz blobs
```
No runtime/test crate gains a Python dependency — fixtures are committed `.npz` blobs read by `npyz` (the established pattern, project memory "oracle-fixture-regen-needs-venv").

**Version verification:**
- `scikit-learn` 1.9.0 — [VERIFIED: env] installed, `sklearn.cluster.HDBSCAN` present, full `.pyx` source readable.
- `hdbscan` 0.8.44 — [CITED: pypi.org/project/hdbscan] latest, Python-3.12 wheels (uploaded recently); pin exactly per D-07/CONTEXT.

## Package Legitimacy Audit

> Only the fixture-generation Python packages are external; the phase adds NO Rust crate dependencies. These run in a throwaway `/tmp` venv and never enter the shipped artifact.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| scikit-learn | PyPI | 14+ yrs | ~80M/mo | github.com/scikit-learn/scikit-learn | OK | Approved (already env-installed, zero new) |
| hdbscan | PyPI | 9+ yrs | ~2M/mo | github.com/scikit-learn-contrib/hdbscan | OK | Approved (fixture-gen + GLOSH oracle; pin 0.8.44) |
| numpy | PyPI | 17+ yrs | ~300M/mo | github.com/numpy/numpy | OK | Approved (existing fixture convention) |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*All three are foundational, decade-old, scikit-learn-ecosystem packages. No Rust crate is added this phase. `package-legitimacy check` was not run against npm/PyPI seams because these are not new install targets in the build — they are pre-existing fixture-generation tools in a disposable venv.*

## Architecture Patterns

### System Architecture Diagram

```
                         X  (row-major n×d feature matrix  OR  n×n distance matrix if Precomputed)
                         │
        ┌────────────────┴─────────────────────────┐
        │ metric == Precomputed?                    │
        │   YES → X IS the distance matrix          │   NO (5 feature metrics)
        │         (validate square + symmetric)     │   │
        ▼                                           ▼   ▼
  ┌──────────────────────┐              ┌───────────────────────────────────┐
  │ HOST: read X to host │              │ DEVICE FRONT-END (reuse Phase-13)  │
  │ core_dist[i] =       │              │ knn_graph(X, k=min_samples,        │
  │   kth-smallest of    │              │           include_self=TRUE)       │
  │   row i (partition)  │              │   → (idx (n,k), dist (n,k)) ascend  │
  └──────────┬───────────┘              │ core_dist[i] = dist[i, min_samples-1]│
             │                          └──────────────────┬────────────────┘
             │                                              │
             │              ┌───────────── full pairwise distance d(i,j) needed for MST
             │              │   (feature metrics: brute n×n via distance prim, QUERY-TILED;
             │              │    precomputed: X itself)
             ▼              ▼
   ┌────────────────────────────────────────────────────────────┐
   │ MUTUAL-REACHABILITY  mr(i,j) = max(core_i, core_j, d_ij/alpha)│
   │   • Precomputed/Cosine → DENSE n×n matrix (host or GATHER)   │
   │     → mst_from_mutual_reachability  (dense Prim, argmin)     │
   │   • euclidean/l1/l2/chebyshev/minkowski (FAST_METRICS)       │
   │     → mst_from_data_matrix (Prim recomputes d(i,j) per step) │
   └────────────────────────────┬───────────────────────────────┘
                                ▼  HOST BACK-END (pure scalar Rust — the tree-atomics dodge)
   ┌────────────────────────────────────────────────────────────┐
   │ MST edges (n-1)  →  argsort by weight (UNSTABLE quicksort!)  │ ← D-04 critical seam
   │   →  make_single_linkage (UnionFind, relabel N+i per merge)  │
   │   →  _condense_tree(min_cluster_size)  →  CONDENSED tree     │
   │   →  _compute_stability                                      │
   │   →  _get_clusters(eom | leaf, ε, max_cluster_size)          │
   │   →  _do_labelling(clusters, ...)        → labels_ (-1=noise)│
   │   →  get_probabilities                   → probabilities_    │
   │   →  outlier_scores (GLOSH, hdbscan lib) → outlier_scores_   │
   │   →  _weighted_cluster_center(X)         → centroids_/medoids_│
   └────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure
```
crates/mlrs-algos/src/cluster/
├── hdbscan.rs                 # FILL the Phase-12 shell: extend Metric enum, real fit body, new fields/accessors
├── hdbscan/                   # (planner's choice) host back-end submodules, if split out:
│   ├── mst.rs                 #   both Prim variants + argsort-by-weight
│   ├── single_linkage.rs      #   UnionFind + make_single_linkage
│   ├── condense.rs            #   _condense_tree + bfs_from_hierarchy
│   ├── stability.rs           #   _compute_stability + max_lambdas
│   ├── select.rs              #   _get_clusters (eom/leaf/epsilon) + _do_labelling + get_probabilities
│   ├── glosh.rs               #   outlier_scores (GLOSH)
│   └── centers.rs             #   centroid / medoid
crates/mlrs-kernels/src/
└── mutual_reachability.rs     # NEW SharedMemory-free GATHER kernel (feature-metric dense MR)
crates/mlrs-core/src/
└── label_perm.rs              # EXTEND best_match_accuracy/best_mapping to pin -1→-1
crates/mlrs-algos/tests/
└── hdbscan_test.rs            # REPLACE shell tests with per-metric oracle + score + center gates
scripts/gen_oracle.py          # ADD gen_hdbscan_* generators (sklearn + hdbscan fixtures)
tests/fixtures/                # committed hdbscan_*_seed*.npz blobs
```

### Pattern 1: Core distance (the kth-nearest-neighbor distance)
**What:** `core_dist[i]` = distance from point `i` to its `min_samples`-th nearest neighbor, **counting itself** (`include_self=true`, so the 0-distance self at column 0 is included in the count).
**When to use:** Always (both precomputed and feature paths). `further_neighbor_idx = min_samples - 1`.
```python
# Source: sklearn/cluster/_hdbscan/_reachability.pyx  _dense_mutual_reachability_graph  [VERIFIED: sklearn source]
further_neighbor_idx = min_samples - 1
core_distances = np.partition(distance_matrix, further_neighbor_idx, axis=1)[:, further_neighbor_idx]
# i.e. the (min_samples-1)-th smallest distance in row i (0-indexed; row includes the self-zero).
```
```python
# Source: sklearn/cluster/_hdbscan/hdbscan.py  _hdbscan_prims  [VERIFIED: sklearn source]
# Prims (FAST_METRICS) path: core dist = LAST column of kneighbors(min_samples)
nbrs = NearestNeighbors(n_neighbors=min_samples, ...).fit(X)
neighbors_distances, _ = nbrs.kneighbors(X, min_samples)
core_distances = neighbors_distances[:, -1]   # the (min_samples-1) index, ascending
```
**mlrs mapping:** call `knn_graph(X, k=min_samples, include_self=true)` → `core_dist[i] = distances[i*min_samples + (min_samples-1)]` (ascending row, self-zero at col 0). [VERIFIED: codebase — `knn_graph.rs` returns ascending `(n,k)` with self at col 0 for `include_self=true`].

### Pattern 2: Mutual reachability
```python
# Source: _reachability.pyx _dense_mutual_reachability_graph  [VERIFIED: sklearn source]
mr(i,j) = max(core_distances[i], core_distances[j], distance_matrix[i,j])
# Brute path applies  distance_matrix /= alpha  BEFORE mutual reachability (so d_ij is already /alpha).
```
**alpha placement (D-09):** In the brute path, `distance_matrix /= alpha` happens in `_hdbscan_brute` BEFORE mutual_reachability (so MR uses `d_ij/alpha` but core distances are computed on the SCALED matrix too — careful: `mutual_reachability_graph` recomputes core distances from the already-/alpha matrix). In the prims path, `mst_from_data_matrix` divides the PAIRWISE distance by alpha but uses RAW core distances: `pair_distance /= alpha; mr = max(core_i, core_j, pair_distance)`. **These two alpha treatments DIFFER** — replicate per-path exactly. [VERIFIED: sklearn source — `_linkage.pyx:189` vs `hdbscan.py:254`].

### Pattern 3: MST — TWO variants (D-04 CRITICAL — see Common Pitfalls #1)
**Variant A — dense `mst_from_mutual_reachability`** (cosine + precomputed). Prim's from node 0; pick next via `np.argmin(min_reachability)` (first-min on ties); index remapping through a shrinking `current_labels` array.
```python
# Source: _linkage.pyx mst_from_mutual_reachability  [VERIFIED: sklearn source]  (reproduced exactly)
current_node = 0
min_reachability = np.full(n, np.inf)
current_labels = np.arange(n)
for i in range(n-1):
    label_filter = current_labels != current_node
    current_labels = current_labels[label_filter]
    left  = min_reachability[label_filter]
    right = mutual_reachability[current_node][current_labels]
    min_reachability = np.minimum(left, right)
    new_node_index = np.argmin(min_reachability)        # FIRST minimum on ties
    new_node = current_labels[new_node_index]
    mst[i] = (current_node, new_node, min_reachability[new_node_index])
    current_node = new_node
```
**Variant B — `mst_from_data_matrix`** (euclidean/l1/l2/chebyshev/minkowski under `algorithm='auto'`). Prim's that tracks a per-node `current_sources[]` and recomputes pairwise distance each step; the candidate-edge tie logic uses strict `<` comparisons (so on ties the FIRST-scanned `j` wins — lowest index, since `j` scans `0..n`).
```python
# Source: _linkage.pyx mst_from_data_matrix  [VERIFIED: sklearn source]  (key tie logic)
for j in range(n):                       # scans ascending j → ties resolve to lowest j
    if in_tree[j]: continue
    pair_distance = dist_metric.dist(raw[current_node], raw[j]) / alpha
    mr = max(core[current_node], core[j], pair_distance)
    if mr < min_reachability[j]:                 # strict <
        min_reachability[j] = mr; current_sources[j] = current_node
        if mr < new_reachability:                # strict <
            new_reachability = mr; source_node = current_node; new_node = j
    elif min_reachability[j] < new_reachability: # strict <
        new_reachability = min_reachability[j]; source_node = current_sources[j]; new_node = j
mst[i] = (source_node, new_node, new_reachability)
```

### Pattern 4: Process MST → single linkage (the UNSTABLE sort — D-04)
```python
# Source: hdbscan.py _process_mst  [VERIFIED: sklearn source]
row_order = np.argsort(min_spanning_tree["distance"])   # DEFAULT kind='quicksort' → UNSTABLE on ties
min_spanning_tree = min_spanning_tree[row_order]
return make_single_linkage(min_spanning_tree)
```
```python
# Source: _linkage.pyx make_single_linkage  [VERIFIED: sklearn source]
U = UnionFind(n)                # sklearn.cluster._hierarchical_fast.UnionFind
for i in range(n-1):
    a = U.fast_find(mst[i].current_node); b = U.fast_find(mst[i].next_node)
    single_linkage[i] = (a, b, mst[i].distance, U.size[a] + U.size[b])
    U.union(a, b)               # assigns BOTH a,b parent = next_label (= n + i), size accumulates
```
**UnionFind (port exactly):** `parent = full(2N-1, -1)`, `next_label = N`, `size = [1]*N + [0]*(N-1)`. `union(m,n)`: `parent[m]=parent[n]=next_label; size[next_label]=size[m]+size[n]; next_label+=1`. `fast_find(n)`: walk parents to root, then path-compress. [VERIFIED: sklearn `_hierarchical_fast.pyx`]. **Note:** because `union` always creates a *new* label and `fast_find` returns the current root, the single-linkage `left/right` are the current root labels — so merge ORDER (from the argsort) directly determines the dendrogram node ids → directly determines the condensed tree → directly determines labels. This is why the unstable `argsort` is the D-04 crux.

### Pattern 5: Condense tree (`_condense_tree`)
BFS from root (`2*(n-1)`), runt-pruning by `min_cluster_size`: a real split keeps both children iff `left_count >= min_cluster_size AND right_count >= min_cluster_size`, assigning new labels; otherwise points "fall out" at `lambda = 1/distance` (or `INFTY` if distance==0). Full body in Code Examples. [VERIFIED: sklearn `_tree.pyx:_condense_tree`].

### Pattern 6: Stability, EoM/leaf selection, labelling, probabilities
- `_compute_stability`: `births[child]=value`, `births[root]=0`; `stability[parent] += (lambda - births[parent]) * cluster_size`. [VERIFIED]
- `_get_clusters` EoM: process `sorted(stability.keys(), reverse=True)[:-1]` (excludes root); if subtree stability > node stability OR `cluster_size > max_cluster_size`, deselect node (push stability up), else deselect whole subtree. Then optional `epsilon_search`. [VERIFIED]
- `_get_clusters` leaf: select `get_cluster_tree_leaves` (DFS leaves of `cluster_size>1` subtree); optional `epsilon_search`. [VERIFIED]
- `_do_labelling`: UnionFind over non-cluster edges; map each point's root to its cluster label (else NOISE=-1). [VERIFIED]
- `get_probabilities`: `deaths = max_lambdas`; per point `result = min(lambda_n, max_lambda)/max_lambda` (1.0 if `max_lambda==0` or `lambda` is inf). [VERIFIED]

### Pattern 7: GLOSH outlier_scores_ (HDBS-03 — hdbscan library, NOT sklearn)
```python
# Source: hdbscan/_hdbscan_tree.pyx outlier_scores  [CITED: github.com/scikit-learn-contrib/hdbscan]
deaths = max_lambdas(tree)              # max lambda per parent cluster
root = parent_array.min()
# Propagate deaths UPWARD (reverse pass) — a child's death floods its parent if larger:
for n in range(len(tree)-1, -1, -1):
    cluster, parent = child_array[n], parent_array[n]
    if deaths[cluster] > deaths[parent]: deaths[parent] = deaths[cluster]
for n in range(len(tree)):
    point = child_array[n]
    if point >= root: continue
    lambda_max = deaths[parent_array[n]]
    if lambda_max == 0.0 or not isfinite(lambda_array[n]): result[point] = 0.0
    else: result[point] = (lambda_max - lambda_array[n]) / lambda_max
```
**Key difference from sklearn `get_probabilities`:** GLOSH does the **upward death-propagation reverse pass** and indexes deaths by the point's *parent* (not its assigned cluster), and uses `(λ_max − λ)/λ_max` (probabilities uses `λ/λ_max`). sklearn has no GLOSH → oracle is `hdbscan` 0.8.44 only (D-07).

### Pattern 8: store_centers (HDBS-04)
```python
# Source: hdbscan.py _weighted_cluster_center  [VERIFIED: sklearn source]
n_clusters = len(set(labels_) - {-1, -2})
for idx in range(n_clusters):
    data = X[labels_ == idx]; strength = probabilities_[labels_ == idx]
    centroids_[idx] = np.average(data, weights=strength, axis=0)                 # weighted mean
    dist_mat = pairwise_distances(data, metric=self.metric, **params) * strength # WEIGHTED
    medoids_[idx]  = data[np.argmin(dist_mat.sum(axis=1))]                        # min weighted total dist
```
**Subtlety:** medoid weights the **distance matrix by strength** then sums rows and argmins. Centroid weights by `probabilities`. Both iterate clusters in ascending id `0..n_clusters`. `store_centers` is feature-array only (errors on precomputed). [VERIFIED: sklearn source].

### Anti-Patterns to Avoid
- **Sorting MST edges with a stable sort.** sklearn uses `np.argsort` default = **quicksort (unstable)**. A stable sort produces a *different* merge order on tied edges → different dendrogram → potentially different labels. To match exactly you must replicate NumPy's introsort tie behavior (see Pitfall #1).
- **Using ONE MST algorithm for all metrics.** Cosine + precomputed go through the dense `argmin` Prim; the other four go through the source-tracking Prim. They produce different tie resolutions. [VERIFIED: sklearn dispatch `hdbscan.py:849`].
- **Hand-rolling core distance as "kth distinct neighbor".** It is `np.partition(...)[min_samples-1]` over the row INCLUDING self-zero — a plain kth-smallest with duplicates kept (matches `include_self=true`). Don't dedup.
- **SharedMemory / atomics / `F::INFINITY` / mutable-bool / shift-loops in the MR kernel.** cpu-MLIR panics (project memory + spike findings). Write the MR kernel as a per-element 2D GATHER (`ABSOLUTE_POS_X/Y`, 16×16, `max(core_i,core_j,d_ij)`), no shared state.
- **Bare-`ABSOLUTE_POS` 1D launch.** MLIR pass failure (FINDING 002-A). Use the proven 2D shape from `knn_graph.rs::launch_dims_2d`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Core distance / kNN | A new distance+selection kernel | `knn_graph(include_self=true)` (Phase 13) | Already oracle-gated per metric, cpu-MLIR-safe, query-tiled. |
| `.npz` fixture reading | zip+npy parser | `mlrs_core::oracle::load_npz` | Established; `npyz::NpzArchive`. |
| Label-permutation match | New Hungarian impl | extend `mlrs_core::label_perm::best_match_accuracy` | Greedy matcher exists; just pin `-1→-1`. |
| MST / single-linkage / condense / stability / selection | A novel HDBSCAN | **Port sklearn `_hdbscan` `.pyx` line-for-line** | The oracle's exact algorithm is readable in-env; any deviation breaks the exact gate. |
| Distance for medoid centers | New pairwise kernel | sklearn-equivalent host pairwise over the small per-cluster `data` | Centers are tiny (per-cluster); host pairwise is fine and matches `pairwise_distances`. |
| Mutual-reachability symmetrization | (none — own it) | Compute `max(core_i,core_j,d_ij)` directly | Phase-13 emits directed only; symmetrization is HDBSCAN's job (R-1). |

**Key insight:** The host back-end is **deterministic pure scalar code** with no floating-point reductions across threads — there is no GPU value to extract and every line has an exact oracle. Porting sklearn's Cython verbatim is both the safest path to the exact gate AND the simplest. The ONLY genuinely hard part is matching NumPy's unstable `argsort` tie order (D-04).

## Runtime State Inventory

> Greenfield algorithm fill — no rename/refactor/migration. The shell already exists; we add fields and replace the trivial fit body. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry an old string.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no datastore keys involved. | none |
| Live service config | None. | none |
| OS-registered state | None. | none |
| Secrets/env vars | None. | none |
| Build artifacts | None — pure source addition; `cargo` rebuilds. | none |

**Nothing found in any category** — verified: this phase adds source + test fixtures only; the `Hdbscan` shell is extended in place (new enum variants, new fitted fields, real `fit` body).

## Common Pitfalls

### Pitfall 1: NumPy `argsort` unstable tie-order is the D-04 exactness crux (TRUE GATE)
**What goes wrong:** `_process_mst` sorts MST edges by weight with `np.argsort(distances)` — default `kind='quicksort'` (actually introsort), which is **UNSTABLE**. On equal-weight edges the output order is implementation-defined and is NOT lowest-index. Verified in-env: `np.argsort([0.5,0.5,0.3,0.3,0.7]) → [3,2,1,0,4]` (NOT `[2,3,0,1,4]`). A different tie order → different UnionFind merge sequence → different dendrogram node ids → different condensed tree → different labels.
**Why it happens:** mlrs's KNN prim documents a **lowest-index** tie convention; HDBSCAN's MST sort does NOT use lowest-index. The two conventions must not be conflated (CONTEXT canonical_refs explicitly warns this).
**How to avoid:** Three options for the spike to evaluate, in order of preference:
  1. **Replicate NumPy introsort tie-order** for the edge-weight sort (hardest; bit-exact). 
  2. **Design fixtures with DISTINCT MST edge weights** so the sort is tie-free and exactness holds under ANY stable rule — then document that the gate is over distinct-weight designs (still satisfies D-03 "exact on every metric" for the gated fixtures). **Recommended for the gate fixtures**; combine with a separate deliberately-tie-heavy fixture that the spike uses to characterize whether ties actually flip labels.
  3. If a tie genuinely flips a label and can't be matched, D-05 says iterate — do not demote. Surface to user (D-06 escalation analog) if un-exactable.
**Warning signs:** spike shows `best_match_accuracy < 1.0` on a tie-heavy fixture for some metric but `== 1.0` on a distinct-weight fixture → the tie-order is the culprit, not the algorithm.

### Pitfall 2: Two MST algorithms, two alpha placements
**What goes wrong:** Applying one MST variant (or one alpha rule) to all metrics. `algorithm='auto'` routes euclidean/l1/l2/chebyshev/minkowski → `mst_from_data_matrix` (alpha divides pairwise dist, raw core dist); cosine + precomputed → `mst_from_mutual_reachability` via `_hdbscan_brute` (alpha divides the WHOLE distance matrix before core distances). [VERIFIED: sklearn dispatch + source].
**How to avoid:** Implement both Prim variants; route by `metric == Cosine || Precomputed → Variant A (dense)` else `Variant B (source-tracking)`. Replicate the per-variant alpha exactly.
**Warning signs:** euclidean exact but cosine off, or `alpha != 1.0` fixtures off.

### Pitfall 3: Cosine distance scaling vs sklearn `pairwise_distances('cosine')`
**What goes wrong:** Phase-13 prim returns cosine distance as `1 − cos` (halved squared-Euclidean of unit vectors). sklearn `_hdbscan_brute` computes `pairwise_distances(X, metric='cosine')` = `1 − cos` too — but it builds the FULL n×n matrix then mutual_reachability. Confirm the prim's cosine distance equals sklearn's to ≤1e-5 (the KNN-graph per-metric oracle already gates this for the neighbor set, but HDBSCAN needs the full matrix). [VERIFIED: codebase `knn_graph.rs` cosine halving].
**How to avoid:** For cosine + precomputed, build the dense n×n distance matrix (cosine via L2-normalized GEMM, `1−cos`); feed to mutual_reachability + dense Prim. Don't reuse only the kNN neighbor set for the MST — the dense Prim needs all pairs.

### Pitfall 4: `min_samples` vs `min_cluster_size` and `further_neighbor_idx`
**What goes wrong:** Off-by-one on core distance. `further_neighbor_idx = min_samples - 1`; core dist = the `(min_samples-1)`-th smallest (0-indexed) distance in the row *including self*. With `include_self=true`, `k=min_samples` neighbors returned, core = column `min_samples-1` (the last). [VERIFIED: sklearn source].
**Also:** `min_samples=None → min_cluster_size` (already in shell `new()`). `_condense_tree` and `_get_clusters` use `min_cluster_size`, NOT `min_samples`. Don't swap them.

### Pitfall 5: EoM/leaf selection identical on well-separated data
**What goes wrong:** A fixture with 3 well-separated blobs gives `eom == leaf` labels (verified in-env), so a "non-default `cluster_selection_method`" fixture proves nothing.
**How to avoid (D-09):** Design the non-default-knob fixtures with **hierarchical/nested density structure** (e.g. two sub-blobs inside a super-blob) so EoM (merges) and leaf (splits) genuinely diverge; same for `cluster_selection_epsilon > 0` (needs a split that ε merges back) and `max_cluster_size` (needs an EoM cluster larger than the bound). Verify in the gen script that the fixture's eom/leaf/ε/max outputs actually differ from defaults before committing.

### Pitfall 6: `store_centers` cluster ordering and weighting
**What goes wrong:** Centers indexed by cluster id but mlrs labels are a *permutation* of sklearn's. Centroid uses `probabilities` weights; medoid weights the **distance matrix** by strength (not the mean).
**How to avoid:** Compute centers per ascending mlrs cluster id, then compare to sklearn via the SAME label permutation used for `labels_` (the kmeans_test.rs pattern: `best_mapping` then compare row `fitted_c → ref_c`). [VERIFIED: kmeans_test.rs precedent].

### Pitfall 7: `np.partition` is not `np.sort`
**What goes wrong:** Core distance uses `np.partition(row, k)[k]` — only guarantees the kth element is in place, not a full sort. For a single kth-smallest this equals `sorted(row)[k]`, but on ties the *value* is deterministic (the kth smallest value), so a full ascending sort (what the KNN prim gives) yields the identical core-distance value. Safe to use the prim's ascending column. [VERIFIED: equivalence of kth-smallest value].

## Code Examples

### `_condense_tree` (verbatim port target)
```python
# Source: sklearn/cluster/_hdbscan/_tree.pyx  [VERIFIED: sklearn source]
root = 2 * (n_samples - 1); next_label = n_samples + 1
node_list = bfs_from_hierarchy(hierarchy, root)   # BFS, children = node - n_samples
relabel = empty(root+1); relabel[root] = n_samples; ignore = zeros(len(node_list))
for node in node_list:
    if ignore[node] or node < n_samples: continue
    left, right, distance = hierarchy[node-n_samples]
    lambda_value = 1.0/distance if distance > 0 else INFTY
    left_count  = hierarchy[left -n_samples].cluster_size if left  >= n_samples else 1
    right_count = hierarchy[right-n_samples].cluster_size if right >= n_samples else 1
    if left_count >= mcs and right_count >= mcs:        # genuine split → two new clusters
        relabel[left]=next_label;  result += (relabel[node], next_label, lambda, left_count);  next_label+=1
        relabel[right]=next_label; result += (relabel[node], next_label, lambda, right_count); next_label+=1
    elif left_count < mcs and right_count < mcs:        # both runt → all points fall out
        for s in bfs(left):  if s<n_samples: result += (relabel[node], s, lambda, 1); ignore[s]=True
        for s in bfs(right): if s<n_samples: result += (relabel[node], s, lambda, 1); ignore[s]=True
    elif left_count < mcs:                               # left runt → right keeps node label
        relabel[right]=relabel[node]
        for s in bfs(left):  if s<n_samples: result += (relabel[node], s, lambda, 1); ignore[s]=True
    else:                                                # right runt
        relabel[left]=relabel[node]
        for s in bfs(right): if s<n_samples: result += (relabel[node], s, lambda, 1); ignore[s]=True
```

### label_perm `-1→-1` extension (HDBS-02)
```rust
// Extend mlrs_core::label_perm: pin the noise sentinel so -1 only ever maps to -1.
// Approach: exclude -1 from BOTH vocabularies during confusion/greedy matching, then
// force map.insert(-1, -1). Points with pred=-1 must have ref=-1 to count as correct
// (and vice-versa) — a noise/cluster mismatch is a genuine failure, never permuted away.
pub fn best_match_accuracy_pinned_noise(pred: &[i64], reference: &[i64]) -> f64 {
    // 1. build best_mapping over labels with -1 filtered out of both label sets
    // 2. insert(-1, -1) into the map unconditionally
    // 3. remap pred; a pred==-1 stays -1; accuracy counts exact matches incl. -1==-1
}
```
**Test gate (per metric):** `best_match_accuracy_pinned_noise(mlrs_labels, sklearn_labels) == 1.0`.

### Fixture generator skeleton (gen_oracle.py, sklearn + hdbscan)
```python
# Source: scripts/gen_oracle.py convention  [VERIFIED: codebase]
def gen_hdbscan(seed, dtype, metric, mcs=5, ms=None, csm='eom', eps=0.0, mxc=0, alpha=1.0,
                store='both', structure='blobs'):
    rng = np.random.default_rng(seed)
    X = _hdbscan_design(rng, structure)         # blobs OR nested-density OR tie-heavy OR dup-point
    kw = dict(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=csm,
              cluster_selection_epsilon=eps, max_cluster_size=mxc, alpha=alpha,
              metric=metric, store_centers=store, copy=True)   # copy=True: silence 1.10 FutureWarning
    if metric == 'minkowski': kw['metric_params'] = {'p': 3.0}
    if metric == 'precomputed':
        from sklearn.metrics import pairwise_distances
        Xin = pairwise_distances(X, metric='euclidean'); kw['metric']='precomputed'
    else: Xin = X
    h = SkHDBSCAN(**kw).fit(Xin)
    import hdbscan as hdb                         # /tmp venv, pin 0.8.44 — for outlier_scores_
    hl = hdb.HDBSCAN(min_cluster_size=mcs, min_samples=ms or mcs, metric=metric,
                     cluster_selection_method=csm, cluster_selection_epsilon=eps,
                     alpha=alpha).fit(Xin if metric!='precomputed' else Xin)
    np.savez(out, X=Xin.astype(dtype), labels=h.labels_, probabilities=h.probabilities_,
             centroids=getattr(h,'centroids_',np.empty((0,))),
             medoids=getattr(h,'medoids_',np.empty((0,))),
             hdb_labels=hl.labels_, outlier_scores=hl.outlier_scores_)
```
**IMPORTANT:** set `copy=True` explicitly in the sklearn call — `copy` default changes False→True in sklearn 1.10 and emits a `FutureWarning` now (verified in-env). Pin behavior.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `hdbscan` contrib library only | `sklearn.cluster.HDBSCAN` (since sklearn 1.3) | 2023 | The PRIMARY oracle is now zero-dep (in env, sklearn 1.9.0). Use sklearn for labels/probs/centers; hdbscan lib only for GLOSH + cross-check. |
| GLOSH in sklearn | NOT in sklearn — only `hdbscan` lib | n/a | HDBS-03 must oracle against `hdbscan` 0.8.44 (D-07). |
| `metric='precomputed'` asymmetric tolerated | sklearn 1.9 raises on non-symmetric precomputed | current | D-02: validate squareness AND symmetry (sklearn checks `allclose(X, X.T)`). |

**Deprecated/outdated:**
- `np.argsort(kind=...)` unspecified → sklearn relies on DEFAULT quicksort; do NOT assume stable.
- sklearn `copy` default (False) → flips to True in 1.10; fixtures must pin `copy=True`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The `hdbscan` 0.8.44 `outlier_scores` body fetched from GitHub master matches the 0.8.44 tag exactly. | Pattern 7 / GLOSH | Medium — GLOSH formula could differ slightly; the `/tmp`-venv-generated 0.8.44 fixture is the real gate (D-07), so the committed `outlier_scores` values are authoritative regardless of the source snippet. Verify the ported formula against the fixture. |
| A2 | mlrs Phase-13 cosine distance (`1−cos`) equals sklearn `pairwise_distances('cosine')` to ≤1e-5 over ALL pairs (not just the kNN set). | Pitfall 3 | Medium — if the dense cosine matrix diverges, cosine MST flips. The spike must build the dense cosine matrix and compare to sklearn before the gate. |
| A3 | Replicating NumPy introsort tie-order is avoidable by using distinct-MST-edge-weight gate fixtures (Pitfall 1 option 2). | Pitfall 1 | High — this is the D-04/D-05 TRUE GATE. If real-data MST ties are unavoidable for some metric and flip labels, the phase may need introsort replication or escalation. The pre-planning spike MUST resolve this per metric. |
| A4 | `algorithm='auto'` is the default mlrs must match (sklearn default), so euclidean→prims, cosine→brute. | Architecture / Pitfall 2 | Medium — if a future test pins `algorithm='brute'`, ALL metrics go dense Prim. Confirm the gate fixtures use default `algorithm`. |
| A5 | The host back-end needs NO device kernel (pure scalar Rust); only the mutual-reachability front-end is a kernel. | Architecture map | Low — well-supported; MST/tree are inherently sequential. |

**If any HIGH-risk assumption (A3) fails in the spike, D-05 makes it a phase blocker — surface immediately.**

## Open Questions (RESOLVED)

> All three pre-planning unknowns are discharged in the Phase-15 plans (ROADMAP spike flag: "SPIKE BEFORE PLANNING — RESOLVED IN PLANS"). The D-04/D-05 exactness spike is sequenced as the Wave-3 TRUE GATE (`15-03`) before the device front-end (`15-05`) commits. Resolution per question below.

1. **Does any metric's real-data MST contain weight ties that flip labels?** (D-04 TRUE GATE)
   - What we know: the sort is unstable; ties exist on duplicate points and integer-grid data.
   - What's unclear: whether the gate fixtures (random f64 blobs) produce tied MST weights in practice.
   - Recommendation: **pre-planning spike** runs all 6 metrics on (a) a distinct-weight blob fixture and (b) a deliberately tie-heavy + duplicate-point fixture, asserting `best_match_accuracy_pinned_noise == 1.0`. If (a) passes and (b) fails, gate on distinct-weight designs + document; if (a) fails, port introsort or escalate (D-05/D-06).
   - **RESOLVED:** Plan `15-02` builds the distinct-weight + tie-heavy + duplicate-point fixtures; plan `15-03` Task 2 makes `tie_break_exact` the Wave-3 TRUE GATE (sequenced before `15-05`), replicating the oracle's `np.argsort`-by-weight ordering (NOT the mlrs lowest-index convention) and surfacing any un-exactable metric as a **phase BLOCKER per D-05** — never band-demoted.

2. **Precomputed alpha + symmetry validation exact behavior.**
   - What we know: sklearn divides the whole matrix by alpha, then mutual_reachability recomputes core distances from the scaled matrix; validates square + `allclose(X, X.T)`.
   - Recommendation: replicate `distance_matrix /= alpha` BEFORE core-distance for the dense path; validate squareness (error) and document symmetry expectation (D-02).
   - **RESOLVED:** Plan `15-03` ports both MST variants with their distinct per-path alpha placements (Pattern 2 / Pitfall 2): the dense `mst_from_mutual_reachability` path (cosine + precomputed) scales the matrix before core-distance, with square-validation (error) + documented symmetry expectation per D-02 in the plan's `<threat_model>` (V5).

3. **Memory gate shape for dense n×n mutual reachability** (Claude's discretion).
   - The dense Prim (cosine/precomputed) needs the full n×n MR matrix; feature-metric Prim B recomputes pairwise on the fly (no n×n resident).
   - Recommendation: for Variant B keep it n×d resident (no n×n); for Variant A the n×n is unavoidable but should be a single host buffer (back-end is host) — set the PoolStats gate on the DEVICE front-end (core-dist KNN + any device MR), not the host tree. Planner sets the exact assertion.
   - **RESOLVED:** Plan `15-05` scopes the PoolStats `memory_gate` to the DEVICE front-end (core-distance KNN + the GATHER MR kernel), not the host tree; Variant B stays n×d-resident. Exact assertion set in `15-05` Task 2.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scikit-learn (HDBSCAN + source) | Primary oracle, fixture gen | ✓ | 1.9.0 | — (also the verbatim algorithm reference) |
| `hdbscan` library | GLOSH oracle + cross-check (D-07) | ✗ | — | Install in `/tmp` venv: `pip install hdbscan==0.8.44` (Py3.12 wheels exist) |
| numpy | fixture gen | ✓ | (env) | `/tmp` venv per PEP-668 |
| cubecl-cpu (f64) | MR kernel correctness gate | ✓ | 0.10.x | — (the f64 gate) |
| cubecl-hip / rocm (f32) | MR kernel f32 gate | ✓ (gfx1100/ROCm 7.1.1) | 0.10.x | f64-on-rocm SKIPS-with-log (memory) |
| maturin / pyarrow (live PyO3) | — | ✗ | — | Out of scope (Phase 16; SHIM-03 defers live checks) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** `hdbscan` library — install in disposable `/tmp` venv for fixture regen only (never a crate dep).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (cargo test), per-backend via `--features cpu` / `--features rocm` |
| Config file | none — workspace `cargo test`; fixtures under `tests/fixtures/*.npz` |
| Quick run command | `cargo test --features cpu --test hdbscan_test -- --nocapture` |
| Full suite command | `cargo test --features cpu --test hdbscan_test` (targeted; full backend suite is ~6min, memory) |
| Phase gate | `hdbscan_test` green on cpu(f64); rocm(f32) green; f64-on-rocm skips-with-log |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HDBS-02 | labels exact-up-to-perm (-1 pinned), per metric × {euclidean,l1,cosine,chebyshev,minkowski,precomputed} × {f32,f64} | unit/oracle | `cargo test --features cpu --test hdbscan_test labels_match_sklearn` | ❌ Wave 0 (replace shell tests) |
| HDBS-02 | tie-heavy + duplicate-point fixture exactness | unit/oracle | `cargo test --features cpu --test hdbscan_test tie_break_exact` | ❌ Wave 0 |
| HDBS-01 | `probabilities_` ≤1e-5 vs sklearn (D-06) | unit/oracle | `cargo test --features cpu --test hdbscan_test probabilities_match` | ❌ Wave 0 |
| HDBS-03 | GLOSH `outlier_scores_` ≤1e-5 vs hdbscan 0.8.44 | unit/oracle | `cargo test --features cpu --test hdbscan_test outlier_scores_match` | ❌ Wave 0 |
| HDBS-04 | `centroids_`/`medoids_` ≤1e-5 vs sklearn (same perm) | unit/oracle | `cargo test --features cpu --test hdbscan_test centers_match` | ❌ Wave 0 |
| HDBS-01/D-09 | non-default eom/leaf, ε>0, max_cluster_size, alpha — exact labels | unit/oracle | `cargo test --features cpu --test hdbscan_test selection_knobs` | ❌ Wave 0 (nested-density fixtures) |
| HDBS-01/D-09 | validation: min_samples>=1 (Some), max_cluster_size 0 or >=mcs | unit | `cargo test --features cpu --test hdbscan_test build_validation` | ❌ Wave 0 |
| HDBS-01 | memory: device front-end no n×n leak (PoolStats gate) | unit | `cargo test --features cpu --test hdbscan_test memory_gate` | ❌ Wave 0 |
| HDBS-01 | edge cases: all-noise, single cluster, n<mcs | unit | `cargo test --features cpu --test hdbscan_test edge_cases` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --features cpu --test hdbscan_test <targeted_fn> -- --nocapture` (one metric/score at a time; avoid full backend suite — disk/time, memory).
- **Per wave merge:** `cargo test --features cpu --test hdbscan_test` (whole HDBSCAN file).
- **Phase gate:** `hdbscan_test` green on cpu; rocm(f32) spot-check; f64-on-rocm skip-with-log confirmed.

### Wave 0 Gaps
- [ ] `crates/mlrs-algos/tests/hdbscan_test.rs` — REPLACE the 4 shell convention tests with the oracle gates above (the shell `fit_roundtrip`/all-`-1` test is removed: fit no longer returns all-`-1`).
- [ ] `scripts/gen_oracle.py` — add `gen_hdbscan_*` generators (sklearn + hdbscan 0.8.44; blobs, nested-density, tie-heavy, duplicate-point designs; per metric; f32+f64).
- [ ] `tests/fixtures/hdbscan_*_seed*.npz` — committed blobs (regen via `/tmp` venv).
- [ ] `crates/mlrs-core/src/label_perm.rs` — `-1→-1`-pinned matcher + its own unit test.
- [ ] hdbscan library install: `/tmp/...-venv/bin/pip install hdbscan==0.8.44` (fixture-gen host only).

## Security Domain

> `security_enforcement=true`, ASVS level 1. This is a numeric host/device compute phase with NO untrusted input boundary (inputs are in-process device arrays / committed fixtures), no auth/session/access-control/crypto surface. Only V5 (input validation) applies.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Host-side geometry/param validation BEFORE any `unsafe` device launch: `min_samples >= 1` (when Some), `max_cluster_size == 0` (unbounded) else `>= min_cluster_size`, `precomputed` X square `(n,n)` + symmetric, `alpha > 0`, `minkowski p >= 1`. Typed `BuildError`/`AlgoError` (the shell precedent). |
| V6 Cryptography | no | — |

### Known Threat Patterns for host+device numeric compute
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read (bad shape/k) | Tampering/DoS | Validate geometry host-side before launch (T-13-06 precedent in `knn_graph.rs::validate_geometry`); bound u32 launch dims. |
| Non-symmetric/non-square precomputed matrix | Tampering (silent wrong result) | Validate square + symmetric (sklearn does `allclose(X, X.T)`); error, don't silently proceed (D-02). |
| Integer overflow on n×n / n*k allocations | DoS | `checked_mul` guards (existing prim pattern) before `pool.acquire`. |
| cpu-MLIR silent miscompile (cross-loop accumulator) | Tampering (silent wrong data) | Per spike findings: MR kernel is a per-element GATHER, no cross-sibling accumulator; oracle asserts VALUES incl. a duplicate-point row (R-9). |

## Sources

### Primary (HIGH confidence)
- `~/.local/lib/python3.12/site-packages/sklearn/cluster/_hdbscan/_linkage.pyx` — `mst_from_mutual_reachability`, `mst_from_data_matrix`, `make_single_linkage` (read verbatim).
- `~/.local/lib/python3.12/site-packages/sklearn/cluster/_hdbscan/_tree.pyx` — `_condense_tree`, `_compute_stability`, `max_lambdas`, `_get_clusters` (eom/leaf/epsilon), `_do_labelling`, `get_probabilities`, `TreeUnionFind` (read verbatim).
- `~/.local/lib/python3.12/site-packages/sklearn/cluster/_hdbscan/_reachability.pyx` — core distance (`np.partition`) + mutual reachability (read verbatim).
- `~/.local/lib/python3.12/site-packages/sklearn/cluster/_hdbscan/hdbscan.py` — dispatch (`FAST_METRICS`, `algorithm='auto'` routing), `_process_mst` (the unstable argsort), `_hdbscan_brute`/`_hdbscan_prims`, `_weighted_cluster_center` (read verbatim).
- `~/.local/lib/python3.12/site-packages/sklearn/cluster/_hierarchical_fast.pyx` — `UnionFind` (single-linkage merge; read verbatim).
- In-env behavioral verification: blob/duplicate/precomputed/minkowski fits, `np.argsort`/`np.argmin` tie semantics, `copy` FutureWarning.
- mlrs codebase: `knn_graph.rs`, `hdbscan.rs` shell, `label_perm.rs`, `oracle.rs`, `kmeans_test.rs`, `gen_oracle.py`, spike-findings-mlrs skill.

### Secondary (MEDIUM confidence)
- `github.com/scikit-learn-contrib/hdbscan` `_hdbscan_tree.pyx` `outlier_scores` (GLOSH) — fetched from master; pin 0.8.44 fixture is the real gate.
- `pypi.org/project/hdbscan` — 0.8.44 Python 3.12 wheel availability.

### Tertiary (LOW confidence)
- DeepWiki / arXiv 2411.08867 — GLOSH conceptual background (not used for the formula; the source code is authoritative).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new compute deps; oracle (sklearn 1.9.0) in-env; hdbscan 0.8.44 wheel confirmed.
- Architecture / algorithm: HIGH — entire host back-end read verbatim from the authoritative sklearn source; GLOSH from hdbscan source.
- Exact-label tie-break (D-04): MEDIUM — the rules are known exactly (unstable argsort + two Prim variants), but whether real-data ties flip labels per metric requires the pre-planning spike to confirm (A3, the TRUE gate).
- Pitfalls: HIGH — derived from reading the source + in-env verification + spike findings.

**Research date:** 2026-06-24
**Valid until:** 2026-07-24 (sklearn HDBSCAN internals are stable; re-check if sklearn ≥1.10 lands — `copy` default flips and `_hdbscan` internals could shift).
