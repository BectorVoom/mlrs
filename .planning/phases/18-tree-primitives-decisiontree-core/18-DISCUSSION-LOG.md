# Phase 18: Tree Primitives + DecisionTree Core - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-27
**Phase:** 18-tree-primitives-decisiontree-core
**Areas discussed:** Public surface scope, Criterion coverage, Stopping-rule params, Prim shape + memory gate

---

## Public surface scope

| Option | Description | Selected |
|--------|-------------|----------|
| Internal core only | 4 prims + build-loop gated on injected fixed indices; no public estimator, no Python. Matches TREE-02 "core" wording + cuML. | |
| Core + Rust estimator | Internal core PLUS user-facing DecisionTreeClassifier/Regressor (Phase-16 builder/typestate, fit/predict); no Python this phase. | ✓ |
| Full estimator + Python | Core + Rust estimator + PyO3 Python wheel surface. Largest; arguably scope-creep. | |

**User's choice:** Core + Rust estimator
**Notes:** Standalone Rust tree for Rust users; Python deferred (RF Phase 19 is the Python entry point).

### Follow-up — Predict path

| Option | Description | Selected |
|--------|-------------|----------|
| Host traversal now | Simple host walk over SparseTreeNode (colid/threshold, stop at colid==-1, deref value). FIL device version is Phase 20. | ✓ |
| Reuse the witness walk | Promote the Phase-17 Tier-1 witness traversal into predict. | (folded into choice) |
| Defer predict to Phase 20 | Estimator only exposes fitted tree; structure-only gate. Awkward fit-but-can't-predict surface. | |

**User's choice:** Host traversal now
**Notes:** Prefer reusing the already-gated Phase-17 witness walk for the host traversal.

---

## Criterion coverage

| Option | Description | Selected |
|--------|-------------|----------|
| gini + squared_error only | Match the spike witness; defer entropy/log_loss/absolute_error to Phase 19. | |
| Add classifier criteria | gini+entropy+log_loss (clf) + squared_error (reg); defer absolute_error. | |
| Full criterion menu | gini/entropy/log_loss (clf) + squared_error/absolute_error (reg), all in Phase 18. | ✓ |

**User's choice:** Full criterion menu
**Notes:** log_loss ≡ entropy (sklearn alias) → one impurity function. absolute_error is the long-pole (median/MAE, doesn't fit the sum/sum-sq GATHER histogram → own host path + gate). Flagged for the researcher.

---

## Stopping-rule params

| Option | Description | Selected |
|--------|-------------|----------|
| RF-complete set | max_depth, min_samples_split, min_samples_leaf, max_features. | |
| max_depth only | Minimal; RF extends later (splits the same work across two phases). | |
| Full sklearn surface | Above + min_impurity_decrease, max_leaf_nodes, min_weight_fraction_leaf, ccp_alpha. | ✓ |

**User's choice:** Full sklearn surface
**Notes:** Broadens TREE-02; consistent with the owner's broad-surface pattern. Flag REQUIREMENTS/ROADMAP sync.

### Follow-up — Growth order (max_leaf_nodes vs level-wise)

| Option | Description | Selected |
|--------|-------------|----------|
| Dual growth modes | Level-wise default; best-first (priority frontier) when max_leaf_nodes set — matches sklearn. Kernels unchanged; only host scheduling differs. | ✓ |
| Level-wise + drop max_leaf_nodes | Single loop; drop max_leaf_nodes; keep the rest. | |
| Defer max_leaf_nodes | Level-wise now; carry best-first growth to a later sweep. | |

**User's choice:** Dual growth modes
**Notes:** ccp_alpha handled as a post-build pruning rewrite of the node array.

---

## Prim shape + memory gate

### quantiles shape

| Option | Description | Selected |
|--------|-------------|----------|
| Host prim, oracle-gated | quantiles = host function (edges + digitize), gated vs np.percentile/KBinsDiscretizer; other 3 are device kernels. Honors D-10, keeps A2 PASS. | ✓ |
| Device quantiles kernel | On-device digitize/edges; contradicts D-10, re-opens A2 risk. | |
| Host edges + device digitize | Split host edges + thin device digitize kernel. | |

**User's choice:** Host prim, oracle-gated

### Memory gate strictness

| Option | Description | Selected |
|--------|-------------|----------|
| Tight active-frontier assert | tree_hist sized by active frontier; PoolStats asserts peak ≤ frontier_nodes×n_feat×n_bins×buffers (+slack); build fails on cumulative-node regression. | ✓ |
| Loose absolute ceiling | Frontier sizing but assert only a generous byte ceiling; weaker guard. | |
| Frontier sizing, gate later | Defer the PoolStats assertion to Phase 19; contradicts SC-4. | |

**User's choice:** Tight active-frontier assert
**Notes:** Realizes the VERDICT's frontier-only optimization as an enforced invariant; aligns with first-class memory-efficiency value.

---

## Claude's Discretion

- Host frontier-scheduler data structure (priority queue vs sorted frontier for best-first).
- Prim function signatures, scratch-buffer reuse plumbing, fixture seeding, rocm-f32 skip-with-log wiring.
- Module placement of the dual-growth scheduler and ccp_alpha pruning pass.

## Deferred Ideas

- PyO3/Python wheel surface for DecisionTree (RF Phase 19 is the Python entry point).
- Device-batched node traversal (FIL, Phase 20).
- RNG-driven bootstrap/feature sampling + Tier-2 ensemble band gate (Phase 19 RF).
- Multiclass leaf-buffer population beyond the core's predict need (Phase 19+).
- On-device bin-edge computation (host pre-pass chosen; revisit only if A2 mitigation insufficient).
