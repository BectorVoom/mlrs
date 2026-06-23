---
phase: 14-umap
verified: 2026-06-24T03:30:00Z
status: gaps_found
score: 4/4 truths functionally verified on cpu-MLIR; criterion-3 "any backend" intent NOT met (see gaps)
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: none
gaps:
  - truth: "The same random_state reproduces a byte-identical embedding across runs on ANY CubeCL backend (criterion 3 + project core value)"
    status: partial
    reason: >-
      reproducible_f64 PASSES but ONLY on the sequential cpu-MLIR backend. The
      fit-path layout kernel launches one cube per owner (CubeCount::Static(n,1,1))
      with move_other=1, and each owner-cube writes its positive neighbour's
      coordinates (embedding[other_base+d1]). Because the fuzzy graph is symmetric,
      two concurrently-scheduled cubes read-modify-write the same vertex slots with
      NO synchronization. On any genuinely parallel backend (wgpu/rocm/cuda) this is
      an unsynchronized data race → non-deterministic output, breaking the D-05
      byte-identical contract that criterion 3 and the project core value ("running
      on ANY CubeCL backend") require. The contract is GREEN today only because the
      single gate (cpu f64; rocm SKIPS f64-with-log) never runs the kernel on a
      parallel backend.
    artifacts:
      - path: "crates/mlrs-kernels/src/umap_layout.rs"
        issue: "Line 153-157: owner-cube writes a FOREIGN vertex's coords (embedding[other_base+d1]) under move_other==1, racing with that vertex's own cube"
      - path: "crates/mlrs-algos/src/manifold/umap.rs"
        issue: "Line 1265 CubeCount::Static(n,1,1) launches concurrent owner-cubes; line 1293 passes move_other=1 (fit path)"
      - path: "crates/mlrs-algos/tests/umap_test.rs"
        issue: "All gates f64-only via gate_f64; no parallel-backend determinism test exercises the race (race is invisible in CI today)"
    missing:
      - "Make the fit path race-free: either two-pass read-snapshot, or owner-only (move_other=0) over the already-symmetric edge set so no cube writes a foreign vertex (REVIEW CR-01 option b)"
      - "Add a parallel-backend (or concurrency-simulating) determinism test so the D-05 contract is actually exercised off cpu-MLIR"
  - truth: "Umap::fit validates its data-dependent geometry before any device launch (criterion 1 robustness — no panic/OOB on valid-typed input)"
    status: failed
    reason: >-
      Umap::fit calls validate_geometry (n>0, p>0, len) but never validates
      n_components < n. With the default Init::Spectral and small n, run_umap_layout
      → spectral_init → recover computes `let col = n - 1 - r` with r up to
      n_components (drop_first → m = n_components+1). When n_components+1 > n (e.g.
      n=2, n_components=2), `n - 1 - r` underflows usize → panic (debug) or wild OOB
      index (release). The sibling SpectralEmbedding::fit DOES guard this
      (spectral_embedding.rs:155) and the typed error AlgoError::InvalidNComponents
      already exists (error.rs:41) — the UMAP path simply omits the call.
    artifacts:
      - path: "crates/mlrs-algos/src/manifold/umap.rs"
        issue: "Lines 443-450 fit(): only validate_geometry; no n_components < n guard before the spectral launch"
      - path: "crates/mlrs-algos/src/cluster/spectral.rs"
        issue: "Line 91 `let col = n - 1 - r;` underflows usize when r >= n (reachable from UMAP spectral path)"
    missing:
      - "Guard `if self.n_components >= n { return Err(AlgoError::InvalidNComponents{..}) }` in Umap::fit before any launch (the typed error already exists)"
      - "Test constructing Umap with n_components >= n asserting the typed error instead of a panic"
  - truth: "The stochastic SGD layout's per-pair force schedule is a faithful port of umap-learn's optimize_layout_euclidean (criterion 3 'vs umap-learn', not merely a calibration-fitted gate)"
    status: partial
    reason: >-
      The symmetric fuzzy graph contains BOTH (r,c) and (c,r); host_epoch_driver
      builds one positive edge per COO entry keyed by owner=head, so each undirected
      pair yields two owner-edges, and with move_other=1 EACH edge moves both
      endpoints — so every undirected pair is attracted ~2-4× per due-epoch and the
      negative schedule is likewise doubled, vs umap-learn's single-pass head/tail
      loop. The property gate is GREEN, but PROPERTY_EPS=0.02 was calibrated against
      exactly this doubled output (REVIEW CR-03), so the gate validates against
      mlrs's own bug, not umap's actual force schedule. Combined with CR-01 the
      layout dynamics are not a faithful port. The structural property gate (trust /
      overlap / ARI) still PASSES, so criterion 3's literal text (≥ umap − margin)
      is met — this is flagged as a partial/correctness-fidelity gap, not a hard
      criterion failure.
    artifacts:
      - path: "crates/mlrs-algos/src/manifold/umap.rs"
        issue: "Lines 1190-1227 build one positive edge per symmetric-COO entry; line 1293 move_other=1 double-counts each undirected pair"
    missing:
      - "Choose one convention matching umap-learn: directed edge set (one rep per pair) with move_other=1, OR symmetric COO with move_other=0 (also fixes CR-01)"
      - "Re-derive the property-gate calibration AFTER the schedule is corrected; assert per-pair sample count vs expected epochs_per_sample"
deferred: []
---

# Phase 14: UMAP Verification Report

**Phase Goal:** Deliver UMAP `fit`/`fit_transform` → `embedding_` `(n, n_components)` with umap-learn/sklearn-named hyperparameters: KNN graph (reuse Phase 13) → fuzzy simplicial set (smooth-kNN ρ/σ + t-conorm union) → init (random default; spectral via graph-Laplacian + eig under the Jacobi size cap) → vertex-owner GATHER SGD layout kernel with negative sampling. Value-gate deterministic stages 1–4; property-gate the stochastic layout. File-disjoint from HDBSCAN.

**Verified:** 2026-06-24
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

The four roadmap success criteria are **functionally achieved on the cpu-MLIR f64 gate** — confirmed by live test execution. The phase is nonetheless reported `gaps_found` because the code review surfaced three correctness/determinism issues, one of which (CR-01) directly undermines criterion 3's "byte-identical across runs" contract and the project's stated core value ("running on ANY CubeCL backend") — it holds today only because the sole gate is the sequential cpu-MLIR backend. These are reported as gaps for a human decision (they may be accepted as v1-scope follow-up hardening, but they are not invisible: see Gaps + Anti-Patterns).

### Observable Truths

| # | Truth | Status | Evidence |
| - | ----- | ------ | -------- |
| 1 | User can fit/fit_transform → embedding_ (n, n_components) with umap-learn-named hyperparameters + defaults (n_neighbors=15, n_components=2, min_dist=0.1, init='spectral', random_state), min_dist≤spread validated at build | ✓ VERIFIED | `fit_roundtrip` PASS (live, 77.8s) — real finite non-zero (n,2) embedding via full KNN→fuzzy→union→init→a/b→SGD pipeline. Defaults at umap.rs:155-167. `build()` rejects min_dist>spread (umap.rs:376) — `build_rejects_bad_min_dist` PASS. `fit_transform` at umap.rs:187. |
| 2 | Deterministic stages (KNN graph, fuzzy simplicial set, fuzzy-set union, spectral init w/ random fallback above Jacobi cap) value-match umap-learn ≤1e-5 (f64) | ✓ VERIFIED | `smooth_knn_*` 5/5 PASS, `fuzzy_union_*` 5/5 PASS, `ab_fit` PASS (live). `spectral_init_*` 5/5 GREEN per git 61649ca + SUMMARY 03; spectral path also exercised live inside `fit_roundtrip` (n=8 Jacobi eig). 21 oracle fixtures committed (5 metrics × fuzzy/spectral/layout/transform + ab). Random fallback above n=64: `random_init` at umap_init.rs:337. |
| 3 | Stochastic SGD layout passes property/structural gate vs umap-learn 0.5.12 AND same random_state → byte-identical across runs | ⚠ PARTIAL | `reproducible_f64` PASS (live, 696.7s — two real same-seed fits byte-identical). `layout_property_*` 5/5 GREEN per git 6807dd9 + VALIDATION calibration (worst trust margin +0.0007). BUT byte-reproducibility holds ONLY on sequential cpu-MLIR; the fit kernel has a cross-cube write race on parallel backends (CR-01), and the force schedule double-counts edges vs umap (CR-03, masks into the calibrated gate). Literal criterion text met on the gate; "any backend" intent NOT met. |
| 4 | User can embed new data via transform(X_new) against the fitted fuzzy graph, property sub-gated | ✓ VERIFIED | `transform` fully wired: `transform_new_points` → `query_train_knn` (distance+top_k, 5-metric) → `init_graph_transform` → `transform_epoch_driver` (move_other=0, owners=new only) → `umap_layout_step`. `transform_property_*` 5/5 GREEN + transform byte-identical per git 749186b + SUMMARY 05; TRANSFORM_PROPERTY_EPS=0.15 calibrated (VALIDATION). |

**Score:** 4/4 truths functionally verified on cpu-MLIR; criterion-3 "any backend" + force-fidelity intent NOT met (gaps below).

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/mlrs-algos/src/manifold/umap.rs` | real fit/fit_transform/transform + host epoch driver + Metric→knn_graph mapping | ✓ VERIFIED | 1306 lines; full pipeline wired (run_umap_layout, host_epoch_driver, transform_new_points); Metric=5 variants mirroring knn_graph::Metric |
| `crates/mlrs-algos/src/manifold/umap_internals.rs` | smooth_knn_dist, compute_membership_strengths, fuzzy_union, init_graph_transform | ✓ VERIFIED | 334 lines, 4 substantive host fns; value-gated |
| `crates/mlrs-algos/src/manifold/umap_init.rs` | fit_ab (LM), spectral_init, random_init, noisy_scale_coords | ✓ VERIFIED | 368 lines, 9 fns incl. LM curve fit + spectral recover reuse |
| `crates/mlrs-kernels/src/umap_layout.rs` | umap_layout_step<F> vertex-owner GATHER SGD kernel | ⚠ VERIFIED-WITH-RACE | 225 lines; cpu-MLIR-safe (no SharedMemory/atomic/INFINITY); LAUNCHES (3/3 smoke PASS) but foreign-vertex write under move_other=1 (CR-01); powf not floored (WR-05) |
| `crates/mlrs-algos/tests/umap_test.rs` | value/property/reproducibility/transform harness | ✓ VERIFIED | 1010 lines, 32 tests; deterministic + property + repro + transform; asserts real contracts |
| `crates/mlrs-backend/tests/umap_layout_test.rs` | cpu-MLIR launch smoke | ✓ VERIFIED | 177 lines, 3 launch tests PASS (f32+f64, both move modes) |
| `tests/fixtures/umap_*.npz` | umap-learn 0.5.12 oracle blobs | ✓ VERIFIED | 21 committed fixtures (5 metrics × 4 stages + ab) |
| `.planning/phases/14-umap/14-VALIDATION.md` | calibrated ε / ARI-band / transform thresholds | ✓ VERIFIED | PROPERTY_EPS=0.02, ARI_BAND=0.05, TRANSFORM_PROPERTY_EPS=0.15 with measured per-metric margins |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| umap.rs | knn_graph::Metric | 5-variant mirror + mapping | ✓ WIRED | Metric enum 5 variants (umap.rs:58); knn_graph imported (umap.rs:36) |
| umap.rs run_umap_layout | umap_internals + umap_init stages | fit pipeline orchestration | ✓ WIRED | smooth_knn_dist(1065), compute_membership_strengths(1072), fuzzy_union(1081), fit_ab(1086), spectral_init(1103) |
| umap.rs host_epoch_driver | umap_layout_step | per-epoch launch w/ host neg_idx | ⚠ WIRED-WITH-RACE | launch at 1277, move_other=1 at 1293, CubeCount::Static(n,1,1) at 1265 — concurrent foreign-write |
| umap.rs transform | umap_layout_step (move_other=0) | frozen-subset SGD | ✓ WIRED | transform_epoch_driver launch at 833, move_other=0 at 849 |
| umap_init spectral_init | spectral::recover | laplacian→eig→recover (drop_first) | ✓ WIRED | recover reuse; BUT recover has usize underflow when n_components+1>n (CR-02) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| Umap embedding_ | embedding_host | run_umap_layout (real KNN→fuzzy→init→SGD) | Yes — finite non-zero coords (fit_roundtrip asserts) | ✓ FLOWING |
| Umap transform output | combined[n..n+m] | transform_epoch_driver real SGD | Yes — trustworthiness-gated new-pt coords | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Real fit produces finite non-zero embedding | `umap_test fit_roundtrip` | 1 passed (77.8s) | ✓ PASS |
| smooth-kNN ρ/σ value-gate × 5 metrics | `umap_test smooth_knn` | 5 passed | ✓ PASS |
| fuzzy union value-gate × 5 metrics | `umap_test fuzzy_union` | 5 passed | ✓ PASS |
| a/b LM curve fit value-gate | `umap_test ab_fit` | 1 passed | ✓ PASS |
| defaults / build-validation / metric coverage | `umap_test defaults_equal,build_rejects_bad_min_dist,metrics_table_covers` | 3 passed | ✓ PASS |
| kernel launches on cpu-MLIR (f32+f64, both move modes) | `mlrs-backend umap_layout_test` | 3 passed (0.29s) | ✓ PASS |
| same-seed fit byte-identical (cpu-MLIR) | `umap_test reproducible_f64` | 1 passed (696.7s) | ✓ PASS |
| spectral_init value-gate × 5 metrics | not re-run (Jacobi eig ~30min total) | GREEN per git 61649ca + SUMMARY 03; spectral path exercised live in fit_roundtrip | ? SKIP (evidence: git + indirect) |
| layout_property × 5 + transform_property × 5 | not re-run (~28min each family) | GREEN per git 6807dd9 / 749186b + VALIDATION calibration | ? SKIP (evidence: git + calibration doc) |

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared for this phase (Rust cargo-test gate, not a probe-based migration phase). N/A.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| UMAP-01 | 14-01, 14-03, 14-04 | fit/fit_transform → embedding_ with umap-named hyperparams + defaults, min_dist≤spread | ✓ SATISFIED | fit_roundtrip PASS; defaults + build validation verified; fit_transform present |
| UMAP-02 | 14-01, 14-02, 14-03 | deterministic stages value-match umap-learn ≤1e-5 (f64) | ✓ SATISFIED | smooth_knn/fuzzy_union/ab PASS live; spectral GREEN per git; 21 fixtures |
| UMAP-03 | 14-01, 14-04 | stochastic SGD property/structural gate + same-seed reproducibility | ⚠ SATISFIED-ON-GATE | reproducible_f64 PASS + property GREEN, but cpu-only; CR-01 race + CR-03 double-count caveat |
| UMAP-04 | 14-01, 14-05 | transform(X_new) property sub-gate | ✓ SATISFIED | transform wired + transform_property GREEN per git/SUMMARY |

No orphaned requirements — all 4 IDs (UMAP-01..04) declared across plans and mapped to Phase 14 in REQUIREMENTS.md.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| crates/mlrs-kernels/src/umap_layout.rs | 153-157 | Concurrent cross-cube foreign-vertex write (move_other=1) | 🛑 Blocker (CR-01) | Data race on parallel backends; breaks D-05 byte-reproducibility off cpu-MLIR |
| crates/mlrs-algos/src/manifold/umap.rs | 443-450 / spectral.rs:91 | Missing n_components<n guard → usize underflow/OOB | 🛑 Blocker (CR-02) | Panic/OOB on valid-typed input; sibling estimator already guards this |
| crates/mlrs-algos/src/manifold/umap.rs | 1190-1227, 1293 | Symmetric COO + move_other=1 double-counts each undirected edge | ⚠ Warning (CR-03) | Force schedule ≠ umap-learn; property gate calibrated to the divergence |
| crates/mlrs-algos/src/manifold/umap.rs | 734, 1179 | `e / negative_sample_rate as f64` no zero-guard | ⚠ Warning (WR-01) | negative_sample_rate=0 → inf/NaN poisons layout |
| crates/mlrs-kernels/src/umap_layout.rs | 133, 147-152 | `powf(dist_sq, b-1)` not floored; clip not NaN-safe | ⚠ Warning (WR-05) | Tiny dist_sq → inf/NaN passes the statement-if clip, corrupts coord |
| crates/mlrs-algos/src/manifold/umap.rs | 456-457 | full host round-trip of x to retain training rows | ℹ Info (WR-03) | Wasteful device→host→device copy; OOM risk for large n*p |
| crates/mlrs-algos/src/manifold/umap_internals.rs | 3-12 | module doc still says "EMPTY stub" | ℹ Info (IN-04) | Stale doc; file is fully implemented |
| crates/mlrs-algos/src/manifold/umap.rs | 1-9, 578-585 | header + transform doc describe abandoned Phase-12 shell design | ℹ Info (IN-03) | Stale narrative contradicts the real code |

No unreferenced `TBD`/`FIXME`/`XXX` debt markers found in the phase files (the debt-marker BLOCKER gate is clean).

### Human Verification Required

None. All in-scope Phase-14 behaviors have automated Rust verification (value-gate + property-gate + reproducibility), which I executed live or confirmed via git history + calibration docs. (The Python/PyO3 estimator surface is explicitly deferred to Phase 16 per VALIDATION.md and is not in this phase's scope.)

### Gaps Summary

The phase **delivers a working, value-gated, property-gated UMAP on the cpu-MLIR f64 gate** — every success criterion passes its automated test on that backend, verified by live execution (fit_roundtrip, reproducible_f64, smooth_knn ×5, fuzzy_union ×5, ab_fit, kernel launch ×3) and git/calibration evidence for the slow spectral/property/transform families. Files are disjoint from HDBSCAN, all 4 requirements are covered, defaults and build-time validation are correct.

The gaps are correctness/fidelity issues the goal's own wording reaches:

1. **CR-01 (BLOCKER):** Criterion 3 requires byte-identical reproducibility across runs and the project core value requires correctness on **any** CubeCL backend. The fit kernel's concurrent foreign-vertex write is an unsynchronized data race on every parallel backend (wgpu/rocm/cuda); reproducibility passes today only because the lone gate is the sequential cpu-MLIR backend. This is the gap most in tension with the goal — it is not merely performance hardening, it is the determinism contract holding by accident of the test backend. The fix (REVIEW option b: move_other=0 over the symmetric set) is small and also resolves CR-03.

2. **CR-02 (BLOCKER):** `Umap::fit` lacks the `n_components < n` guard that its sibling `SpectralEmbedding::fit` already has; a valid-typed `n_components >= n` underflows usize in `recover` → panic/OOB. The typed error (`AlgoError::InvalidNComponents`) already exists, so this is a one-guard fix plus a test.

3. **CR-03 (WARNING):** The symmetric-COO + move_other=1 schedule double-counts each undirected edge vs umap-learn's single-pass loop; the property gate is GREEN but was calibrated against this divergence, so it validates against mlrs's own behaviour rather than umap's force schedule. Criterion 3's literal "≥ umap − margin" still passes; flagged for fidelity, resolved together with CR-01 by the move_other=0 option.

Recommendation: route to `/gsd-plan-phase --gaps`. CR-01 + CR-03 collapse into one fix (owner-only over the symmetric set), CR-02 is an independent guard + test. If the team decides the parallel-backend determinism is out of v1 cpu+rocm-f32 scope, CR-01/CR-03 can be accepted via a documented override — but note rocm IS a runnable parallel GPU gate per project memory, so the race is reachable there for f32 the moment a rocm UMAP gate is added.

---

_Verified: 2026-06-24_
_Verifier: Claude (gsd-verifier)_
