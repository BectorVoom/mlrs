# Phase 15: HDBSCAN - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-24
**Phase:** 15-hdbscan
**Areas discussed:** Metric surface + precomputed, Exact-label gate anchoring, probabilities_ + GLOSH band, store_centers + selection knobs

---

## Metric surface + precomputed

### Metric set

| Option | Description | Selected |
|--------|-------------|----------|
| All 5 + precomputed | euclidean/manhattan/cosine/chebyshev/minkowski-p via Phase-13 prim + precomputed; consistent with UMAP D-01 | ✓ |
| Euclidean + precomputed only | sklearn default + exact-gate anchor; smallest surface | |
| All 5, no precomputed | mirror UMAP but drop the exact anchor (not recommended) | |

**User's choice:** All 5 + precomputed → **D-01**

### Precomputed API surface

| Option | Description | Selected |
|--------|-------------|----------|
| Metric::Precomputed variant | X interpreted as n×n distance matrix; single fit entry; mirrors sklearn metric='precomputed' | ✓ |
| Separate fit_precomputed path | distinct typed entry; diverges from sklearn single metric= surface | |

**User's choice:** Metric::Precomputed variant → **D-02**

---

## Exact-label gate anchoring

### Gate anchor

| Option | Description | Selected |
|--------|-------------|----------|
| Precomputed exact + euclidean exact | hard gate on both; other 3 metrics band-gated | |
| Precomputed exact only | hard gate on precomputed; all feature metrics band-gated; matches REQUIREMENTS verbatim | |
| Exact on all metrics | exact-up-to-perm on every metric (highest risk — flagged not recommended) | ✓ |

**User's choice:** Exact on all metrics → **D-03**
**Notes:** Maximal-correctness posture, consistent with broad-scope preference. Risk captured: non-euclidean brute-KNN ties can flip MST edges. Mitigation hinges on D-04.

### MST tie-break rule

| Option | Description | Selected |
|--------|-------------|----------|
| Stable-sort + lowest-index | established mlrs convention; reproducible within mlrs | |
| Match the oracle's exact tie-break | replicate sklearn/hdbscan internal MST tie-ordering; more spike effort | ✓ |

**User's choice:** Match the oracle's exact tie-break → **D-04**

### Fallback policy if a metric is un-exactable

| Option | Description | Selected |
|--------|-------------|----------|
| Demote that metric to band gate | keep exact where achievable; demote the rest | |
| Escalate to user | pause for user decision | |
| Hold the exact line | non-negotiable; iterate until every metric passes | ✓ |

**User's choice:** Hold the exact line → **D-05**
**Notes:** The pre-planning exactness spike becomes a TRUE gate — an un-exactable metric is a phase blocker, surface early.

---

## probabilities_ + GLOSH band

### Band style

| Option | Description | Selected |
|--------|-------------|----------|
| Tight relative-to-oracle (à la UMAP D-04) | small ε relative to oracle, calibrated on first fixture | |
| Tight absolute tolerance | fixed ≤1e-6/1e-5 abs+rel | |
| Near-exact, escalate if not | treat as ≤1e-5 value gate; escalate on algorithmic divergence | ✓ |

**User's choice:** Near-exact, escalate if not → **D-06**

### Score oracle

| Option | Description | Selected |
|--------|-------------|----------|
| probabilities_ vs sklearn, GLOSH vs hdbscan | matches REQUIREMENTS oracle hierarchy | ✓ |
| Both vs hdbscan lib | single consistent reference for both scores | |

**User's choice:** probabilities_ vs sklearn, GLOSH vs hdbscan → **D-07**

---

## store_centers + selection knobs

### store_centers scope

| Option | Description | Selected |
|--------|-------------|----------|
| Both, value-gated vs sklearn | centroid+medoid, ≤1e-5 vs sklearn; full parity | ✓ |
| Both, presence/shape only | structural gate only | |
| Centroid now, medoid deferred | reduced scope; breaks parity | |

**User's choice:** Both, value-gated vs sklearn → **D-08**

### Selection-knob depth

| Option | Description | Selected |
|--------|-------------|----------|
| Full, all under the exact gate | eom+leaf, epsilon, max_cluster_size, alpha; non-default fixtures all exact-gated | ✓ |
| Full impl, defaults-only exact gate | non-default knobs get lighter spot-checks | |
| You decide (planner) | planner sets fixture depth | |

**User's choice:** Full, all under the exact gate → **D-09**

---

## Claude's Discretion

- Host MST algorithm internals (Prim's data structures, union-find shape) — provided D-04's oracle-matched tie-break holds.
- Memory / PoolStats gate for the n×n mutual-reachability — follow the established convention.
- Condensed-tree / stability-extraction data structures (EoM vs leaf traversal).
- Edge cases (all-noise, single point, single cluster, < min_cluster_size points) — match sklearn.
- `min_samples=None → min_cluster_size` resolution — keep the shell's existing rule.

## Deferred Ideas

- PyO3 wrap of `Hdbscan` + builder-retrofit sweep + Python shim — Phase 16.
- `approximate_predict` / `membership_vector`, condensed-tree/dendrogram plot objects — out of scope (REQUIREMENTS).
- Approximate / NN-Descent / tree KNN build, custom/callable metrics, native sparse — out of scope (REQUIREMENTS).
