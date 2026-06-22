# Pitfalls Research

**Domain:** Adding UMAP + HDBSCAN (on a shared KNN-graph primitive), a Rust-native builder-pattern API (retrofit across 30 estimators), and a pure-Python sklearn shim to mlrs (Rust/CubeCL rewrite of cuML), gated cpu(f64)+rocm(f32) against scikit-learn ≤1e-5 / umap-learn property gate / hdbscan exact-label gate
**Researched:** 2026-06-22
**Confidence:** HIGH for cpu-MLIR kernel constraints + builder-retrofit + sklearn-shim traps (grounded in v1/v2 codebase idioms, project memory, and existing `top_k`/`any_estimator!` source); HIGH for the UMAP stochastic-gate strategy and HDBSCAN label-divergence sources (verified against umap-learn validation practice + hdbscan reference source); MEDIUM for exact f32-on-rocm band magnitudes (must be measured empirically per family, as v1/v2 did)

This file is scoped to **adding THESE features to THIS backend** — the cpu-MLIR no-SharedMemory/no-atomics GATHER discipline, the cpu(f64)+rocm(f32) gate with f64-on-rocm skip-with-log, and the oracle discipline (sklearn ≤1e-5 where it exists; umap-learn **property** gate for the stochastic UMAP layout; hdbscan **exact-labels-up-to-permutation** for HDBSCAN). Generic ML/UMAP/HDBSCAN tutorials are omitted. Every pitfall names the feature/phase it hits, the concrete prevention, and a phase. **Phase numbering continues from v2.0: next phase = 12.** A working phase assumption (refine in the roadmap): **Phase 12 = KNN-graph prim, 13 = UMAP, 14 = HDBSCAN, 15 = Rust-native builder retrofit, 16 = pure-Python sklearn shim.**

The two make-or-break items per the downstream consumer — **the UMAP stochastic-gate strategy (Pitfall 5)** and **the cpu-MLIR KNN-graph feasibility (Pitfalls 1–3)** — lead the list.

---

## Critical Pitfalls

### Pitfall 1: KNN-graph construction reaches for cross-unit atomics / SharedMemory and panics at cpu-MLIR launch

**What goes wrong:**
The shared KNN-graph prim is the v3 keystone and the single highest cpu-MLIR risk. Two natural kernel shapes both fail:
- **Heap/insertion-sort per query row with a mutable running heap** — uses a mutable-bool "is this slot used" flag, `F::INFINITY` sentinels to initialise the heap, and a shift-loop to insert into the sorted prefix. All three (mutable bool, `F::INFINITY`, shift-loop) are on the v1/v2 cpu-MLIR forbidden list and panic at *launch* ("failed to run pass" in cubecl_cpu MLIR lowering), not compile.
- **Scatter into a CSR adjacency** where each query writes its k neighbour edges into a shared edge array via an atomic running offset — cross-unit atomic, which cpu-MLIR cannot lower.

**Why it happens:**
KNN is taught as "maintain a max-heap of size k while scanning candidates" — inherently a mutable-accumulator, sentinel-initialised, shift-on-insert structure. And building a graph "feels" like emitting edges into a shared list (scatter). Both instincts are exactly what broke earlier mlrs kernels.

**How to avoid (reuse the v1 `top_k` GATHER prim, do NOT write a new heap kernel):**
mlrs already has a **launch-proven** `top_k` (`crates/mlrs-backend/src/prims/topk.rs`) that returns the k smallest distances + column indices ascending, single-owner per output row, no SharedMemory, no `F::INFINITY`, no shift-loop. The KNN-graph prim should be a thin composition:
1. v1 pairwise `distance` prim → dense `n×n` (or tiled) distance (single-owner per cell, already cpu-MLIR-safe).
2. v1 `top_k` per row → `(indices[n,k], dists[n,k])`. This is the whole neighbour search; no new heap kernel.
3. Materialise the graph as a **dense `[n,k]` index+distance pair** (the format UMAP's fuzzy-set and HDBSCAN's core-distance both consume directly), NOT a scatter-built CSR. Any later CSR/symmetrisation is a *single-owner* transform (Pitfall 3), not a scatter.
Loop guards over candidates use **ascending u32 counters with `if`-guards**, never mutable bool / `while`-on-flag.

**Warning signs:**
Any new `*.rs` kernel module in the KNN-graph prim importing `Atomic` or `SharedMemory`; any `F::INFINITY` literal; any `while` loop with a mutable-bool condition; a kernel that compiles under `--features rocm` but panics at launch under `--features cpu`.

**Phase to address:** Phase 12 (KNN-graph prim). **Spike the compose-from-`top_k` path and confirm `--features cpu` launch BEFORE UMAP/HDBSCAN consume it** — this is the feasibility gate for the whole milestone. Primitive-first discipline (land + standalone-validate the prim before consumers) is non-negotiable here.

---

### Pitfall 2: UMAP SGD layout update wants per-edge scatter into shared embedding coordinates (cpu-MLIR panic + nondeterminism, exactly cuML's bug)

**What goes wrong:**
The UMAP optimisation loop is "for each edge (i,j): pull `y_i`,`y_j` together; for each negative sample (i,c): push apart." The obvious kernel parallelises **over edges** and writes `y[i] += ...` and `y[j] -= ...`. Multiple edge-threads touch the same embedding row → cross-unit atomic / mutable accumulator → cpu-MLIR launch panic (same class as v2 Pitfall 1/2). cuML hit the *same* shape as a non-determinism bug: vertex-parallel races in `optimize_batch_kernel.cuh` (CONCERNS.md: "UMAP Non-Determinism Under Vertex-Parallel Kernels", fixed in 26.06 by enforcing sequential vertex-parallel behaviour). The cuML reference kernel is also 1,165 lines of warp-shuffle + shared-memory collision handling (CONCERNS.md porting risk) — do NOT port it.

**Why it happens:**
Edge-parallel SGD is the textbook UMAP layout. But edge-parallel = multi-writer on vertices, which is both the cpu-MLIR forbidden pattern and a source of the very nondeterminism that makes UMAP hard to gate.

**How to avoid (vertex-owner GATHER, mirroring the v2 SGD two-pass idiom):**
Invert to **one thread per embedding vertex `i`** (single owner of `y[i]`). Each `i`-thread scans its own incident edges (from the dense `[n,k]` KNN-graph, Pitfall 1) and its own assigned negative samples, accumulating the net gradient for `y[i]` into private registers, then writes `y[i]` once. This is the v2 SGD "gradient-build pass = one owner per coordinate, apply pass = one owner" idiom applied to vertices. Negative sampling indices are drawn from the **SplitMix64 device/host PRNG** (project memory: no OsRng; seeded reproducible PRNG), generated single-owner per vertex. This also fixes determinism for free — single-owner writes are race-free, so same seed → identical mlrs embedding across runs (the determinism leg of the property gate, Pitfall 5).

**Warning signs:**
Any UMAP kernel parallelised over edges with `y[i] += / y[j] -=`; any `Atomic`/`SharedMemory` import; embedding differs across two identical-seed runs (race re-introduced); cpu launch panic.

**Phase to address:** Phase 13 (UMAP layout). Spike the vertex-owner SGD kernel on `--features cpu` before wiring the estimator.

---

### Pitfall 3: KNN-graph symmetrisation, self-neighbour inclusion, and tie-handling diverge silently from the references

**What goes wrong:**
Four independent, quiet correctness bugs in the graph stage that both consumers inherit:
1. **Self-neighbour:** A brute-force `top_k` over the full `n×n` distance returns each point as its OWN nearest neighbour (distance 0 at lowest index — and mlrs `top_k` uses a **lowest-index tie-break**, so the self-index always wins the 0-distance tie). UMAP and HDBSCAN both expect the k neighbours *excluding self* (umap-learn) — using `min_samples+1` and dropping the self-column is the reference convention (hdbscan does `tree.query(X, k=min_samples+1)`). Forgetting to drop self silently shifts every neighbourhood by one and corrupts core distances.
2. **Symmetrisation:** UMAP's fuzzy simplicial set is the **fuzzy union** `A + Aᵀ − A∘Aᵀ` of the directed fuzzy graph, NOT a plain `max(A, Aᵀ)` or `(A+Aᵀ)/2`. HDBSCAN's mutual-reachability is `max(core_i, core_j, d_ij)` — symmetric by construction but only if `core` is computed on the self-dropped k-graph. Mixing these up (using UMAP's union for HDBSCAN, or arithmetic mean for UMAP) produces plausible-but-wrong graphs.
3. **Tie-handling vs the reference:** mlrs `top_k` breaks distance ties by **lowest column index**; umap-learn/hdbscan rely on the (different) tie order of pynndescent/KDTree. For **UMAP this is fine** — the property gate (Pitfall 5) tolerates neighbour-set tie differences. For **HDBSCAN it can flip a label** when a tied mutual-reachability edge changes which MST edge is chosen (Pitfall 6).
4. **The symmetrisation transform itself must be GATHER:** computing `A + Aᵀ − A∘Aᵀ` as one-owner-per-`(i,j)` cell reading `A[i,j]` and `A[j,i]` — never a scatter over edges.

**Why it happens:**
"k nearest neighbours" ambiguously includes/excludes self; "symmetrise" sounds like averaging; and the tie-order assumption is invisible until a label flips.

**How to avoid:**
- Query `k+1` then drop the self column (assert column 0 is self at distance ≈0 in a debug check).
- Encode the exact symmetrisation per consumer: **UMAP fuzzy union** (`a+b−a*b` per cell), **HDBSCAN mutual-reachability** (`max(core_i, core_j, d_ij)` per cell). Both as single-owner GATHER transforms.
- Document the lowest-index tie-break as the mlrs convention; rely on UMAP's property gate to absorb it; for HDBSCAN, design the MST tie-break to be deterministic and reference-aligned (Pitfall 6).

**Warning signs:**
Core distances all one-off; UMAP clusters look right but slightly rotated/merged (self not dropped); HDBSCAN label count differs by one on tied-distance fixtures; any `(A+Aᵀ)/2` in UMAP code.

**Phase to address:** Phase 12 (graph format, self-drop, symmetrisation transforms), consumed in 13/14.

---

### Pitfall 4: A dense `n×n` distance / heap tile for the KNN-graph overflows the gfx1100 LDS budget or the per-phase memory gate

**What goes wrong:**
The compose-from-`top_k` path (Pitfall 1) materialises a dense `n×n` distance matrix. At UMAP/HDBSCAN fixture sizes this is `O(n²)` device memory and, if any tiled GEMM/distance stage stages an operand in SharedMemory, a modest f32 tile (128×128 = 64 KiB) already blows gfx1100's 65536 B LDS budget (f64 doubles it) → HIP rejects the launch. Separately the `n×n` buffer can blow the **build-failing PoolStats memory gate** that every phase must pass.

**Why it happens:**
v2 already hit this for kernel/Gram/Laplacian `n×n` operands (v2 Pitfall 11). The KNN-graph is the next `n×n` offender, and memory is a first-class per-phase gate, never deferred.

**How to avoid:**
- Keep the big distance operand in **global** memory (v1 Jacobi precedent); stage only small bounded tiles in SharedMemory, and **none on cpu** (GATHER everywhere).
- Compute LDS bytes = `tile_elems * size_of::<F>()` and assert `< 65536` at kernel-author time.
- For large n, **tile the query axis**: compute distances for a block of rows, run `top_k` on that block, never hold the full `n×n` resident — single-owner per row throughout, keeping the resident set within the PoolStats gate.

**Warning signs:**
Launch failure only on rocm with an LDS/occupancy/resource error; PoolStats gate fails as fixture n grows; works on a small fixture, OOMs on the next size.

**Phase to address:** Phase 12 (set the tiled, global-operand design + memory gate up front).

---

### Pitfall 5: Forcing element-wise 1e-5 on the STOCHASTIC UMAP layout (or picking a too-loose / too-strict property gate) — the make-or-break oracle decision

**What goes wrong:**
UMAP is triply stochastic: random init, SGD edge sampling order, and **negative sampling**, all RNG-driven. mlrs uses **SplitMix64**; umap-learn uses NumPy's RNG. Identical seeds across different PRNGs → different embeddings, so an element-wise ≤1e-5 value oracle against umap-learn reports **total failure on a perfectly correct implementation** — exactly the trap v2 hit with RandomProjection (D-12), where the fix was a property gate, not a value oracle. Two opposite failure modes follow:
- **Too strict:** any attempt to value-match coordinates (or even match umap-learn's embedding via Procrustes alignment to 1e-5) fails forever; the phase stalls chasing a non-existent bug.
- **Too loose:** "it returns a 2-D blob, ship it" with only a smoke test passes a *broken* layout (e.g. one that learned nothing, or collapsed) because nobody checked structure preservation.

**Why it happens:**
The whole v1/v2 oracle reflex is "same input → compare values ≤1e-5." UMAP breaks the unstated determinism-given-input assumption, and the right gate (structure-preservation metrics) is unfamiliar.

**How to avoid (the concrete UMAP property gate — the precedent is RandomProjection D-12):**
Gate the **mathematical/structural contract**, not coordinates. The standard practitioner toolkit for validating a UMAP reimplementation:
1. **Trustworthiness** (`sklearn.manifold.trustworthiness`) — fraction of each point's k-NN preserved from high-D into the embedding. Gate `trustworthiness(X, mlrs_emb, n_neighbors=k) ≥ threshold`, AND assert it is **within a small delta of umap-learn's own trustworthiness on the same data** (e.g. `mlrs_trust ≥ umap_trust − 0.02`). This anchors "as good as the reference" without value-matching. (Literature: different UMAP implementations land at *similar* trustworthiness; that comparability IS the gate.)
2. **k-NN overlap / preservation** — for a held-out sample of points, fraction of high-D k neighbours that remain among the embedding's k neighbours. Gate against an absolute floor and against umap-learn's overlap on the same fixture.
3. **Determinism** — same SplitMix64 seed → byte-identical mlrs embedding across runs (this is *mlrs's* internal contract, value-matchable, and it catches the Pitfall 2 race). ASVS V6 reproducibility.
4. **Cluster-label preservation on a labelled fixture** — embed `make_blobs`/`digits`, run a trivial clustering (or known labels) on the embedding, and require neighbourhood/label structure matches expectation (e.g. silhouette or label-KNN accuracy above a floor). Discrete, robust to RNG.
5. **`fuzzy_simplicial_set` / `smooth_knn_dist` are NOT stochastic** — the fuzzy-graph construction (given the KNN-graph) is deterministic arithmetic. **Value-gate those against umap-learn at ≤1e-5** (like v2 value-matched `johnson_lindenstrauss_min_dim`). This pins the deterministic 80% of UMAP exactly and confines the property gate to the SGD layout only.

Choosing thresholds: anchor every continuous gate to **umap-learn's own score on the identical fixture** (relative floor) rather than a hard absolute, so the gate is neither arbitrarily strict nor a rubber stamp. Pick `n_neighbors` for trustworthiness equal to the UMAP `n_neighbors`. Use multiple seeds and require the gate to hold for all.

**Warning signs:**
Anyone writing a `.npz` value oracle for UMAP `transform()`; "UMAP fails 1e-5 everywhere" (wrong gate chosen); a green test suite where the only UMAP check is `assert emb.shape == (n,2)` (too loose); trustworthiness gate with a hard absolute threshold and no comparison to umap-learn.

**Phase to address:** Phase 13 (decide the exact property-gate spec — metrics, thresholds-relative-to-umap-learn, seeds — IN the phase plan before kernels, so no time is wasted on a value fixture). This is the milestone's flagship oracle decision.

---

### Pitfall 6: HDBSCAN labels are unstable across reimplementations — MST/condensed-tree tie-breaking, noise label, and probabilities diverge

**What goes wrong:**
HDBSCAN's hard gate is **exact labels up to permutation**, but several quiet sources make a correct-looking reimplementation produce *different* labels than the `hdbscan` reference:
1. **MST tie-breaking:** the reference sorts MST edges by mutual-reachability weight with a **stable sort** (`np.argsort(...)`), so equal-weight edges keep input order. A different tie order picks a different spanning tree → a different dendrogram → flipped/merged clusters. Worse, mlrs's KNN tie-break is lowest-index (Pitfall 3) while the reference inherits KDTree/pynndescent neighbour order — so the *inputs* to the MST already differ on ties.
2. **Noise label −1 vs cluster ids:** noise is `−1`; clusters are arbitrary non-negative ids. The label-permutation comparison must treat `−1` as a **fixed, non-permutable** class (noise maps only to noise) while permuting the cluster ids — the v1/v2 `label_perm` best-mapping helper must be extended so `−1` is excluded from the permutation search, else a correct result fails or a wrong one passes.
3. **`probabilities_`:** membership strength per point (distance-based, normalised within cluster); the reference computes it in a Cython module. Re-deriving the formula wrong yields plausible `[0,1]` values that don't match — but probabilities are continuous and RNG-free, so they CAN be value-gated (with an f32 band), unlike labels.
4. **single vs full mutual reachability / `min_samples` vs `min_cluster_size`:** `min_samples` sets the k for **core distance** (`tree.query(X, k=min_samples+1)`); `min_cluster_size` thresholds the condensed tree. Defaulting `min_samples = min_cluster_size` (the reference default) and conflating the two is a classic divergence — using `min_cluster_size` where `min_samples` belongs changes every core distance.
5. **`cluster_selection_method` (eom vs leaf) and `cluster_selection_epsilon`:** the reference's epsilon path is itself incomplete in cuML (CONCERNS.md: cuML's `extract.cuh`/`select.cuh` only approximate epsilon). Match the `hdbscan` Python reference, not cuML, and start with `cluster_selection_epsilon=0` + `eom` to make exact-labels achievable.

**Why it happens:**
The condensed-tree → stability → selection pipeline has many integer/ordering decisions that are deterministic-but-implementation-specific; ties are rare on real data so divergence hides until a tied fixture.

**How to avoid:**
- Reproduce the reference ordering: **stable sort MST edges by weight**, and make the MST construction's tie-break deterministic and documented (Prim's/Borůvka with a defined lowest-index tie-break). Generate oracle fixtures with **distinct distances where possible** (small jitter) so ties don't dominate the gate, AND add a *separate* explicitly-tied fixture to lock the tie convention.
- Extend `label_perm` to **pin `−1`→`−1`** and permute only cluster ids; assert noise sets match exactly before permuting.
- Port `min_samples`/`min_cluster_size`/core-distance semantics line-for-line from the `hdbscan` reference; default `min_samples = min_cluster_size`.
- Value-gate `probabilities_` and the condensed-tree stabilities with a named f32 band; **labels stay the exact hard gate** (the v2 D-08 classifier rule: discrete decisions are integer-exact, continuous outputs get a band).
- Gate against `hdbscan` / `sklearn.cluster.HDBSCAN`, NOT cuML (cuML's epsilon/outlier paths are incomplete — CONCERNS.md).

**Warning signs:**
Label count off by one on tied fixtures; `label_perm` "passing" by mapping noise into a cluster; `probabilities_` in `[0,1]` but not matching; results match on jittered data but diverge on grid/duplicate data; comparing to cuML.

**Phase to address:** Phase 14 (HDBSCAN). Decide the deterministic MST tie-break + the `−1`-pinned `label_perm` extension + the exact-vs-band split in the phase plan.

---

### Pitfall 7: The MST and condensed-tree builders want scatter/atomics (cpu-MLIR) — and parallel core-distance accumulation reorders floats

**What goes wrong:**
- **MST (Borůvka/Prim) on GPU** classically uses atomic min-reductions to find each component's cheapest outgoing edge and scatter to union-find parents — both cross-unit atomics, cpu-MLIR launch panic. cuML's tree algorithms are exactly the atomics-heavy code the milestone is *deliberately dodging* (PROJECT.md defers RandomForest tree construction for this reason).
- **Condensed-tree / stability** is inherently sequential hierarchy traversal — fighting to GPU-parallelise it invites scatter into shared node arrays.
- **Core-distance** parallel reduction across threads reorders float accumulation → f32 results differ run-to-run and from the reference (hdbscan CONCERNS source: "parallel jobs → floating-point accumulation differs across thread orderings").

**Why it happens:**
"GPU MST" literature is atomics-and-union-find; the instinct is to parallelise the whole pipeline on-device.

**How to avoid (split device vs host deliberately):**
- Do the **embarrassingly-parallel, single-owner** parts on device via GATHER: pairwise distance, `top_k` neighbours (Pitfall 1), core distances (one owner per point, ascending scan — deterministic accumulation order, no atomics), mutual-reachability transform (one owner per cell).
- Do the **MST + condensed-tree + stability + selection on the host** (it is `O(n·k)`-ish, sequential, label-exact-sensitive, and tiny vs the distance work). This sidesteps the atomics constraint entirely and makes deterministic tie-breaking trivial — exactly the "dodge tree-atomics risk" framing in PROJECT.md. cuML itself does the hierarchy logic in serial Cython on CPU in the reference.
- Pin core-distance accumulation to a fixed (ascending) order so f32 is reproducible.

**Warning signs:**
Any `Atomic`/union-find scatter in an MST kernel; cpu launch panic on the graph→tree stage; core distances vary across runs.

**Phase to address:** Phase 14 — decide the device/host split (device = distance+knn+core+mutual-reach GATHER; host = MST+tree+stability) in the plan.

---

### Pitfall 8: f32-on-rocm error bands for UMAP/HDBSCAN distance/exp/log-sum accumulation — too-strict gate fails correct code, too-loose hides bugs

**What goes wrong:**
Strict 1e-5 is often physically unreachable in f32 (project memory): ULP exceeds 1e-5 on large magnitudes and error compounds through accumulation-heavy kernels. The v3 features are accumulation- and transcendental-heavy: UMAP's `smooth_knn_dist` binary search + `exp(-(d−ρ)/σ)` membership, the SGD layout's repeated coordinate updates, HDBSCAN's `probabilities_`. Forcing strict 1e-5 fails correct f32 code; loosening the *global* tolerance hides real bugs. **f64-on-rocm is unsupported and must skip-with-log** (cubecl-cpp 0.10 F64 unregistered for HIP) — every new f64 oracle case in Phases 12–16 needs the guard or it FAILS (not skips), looking like a numerical bug.

**Why it happens:**
Copying a non-f64 test forgets the skip guard; reusing one global tolerance is less bookkeeping than per-family bands.

**How to avoid:**
- Mirror the v1/v2 idiom verbatim in **every f64 oracle case**: `if capability::skip_f64_with_log() { return; }` (as in `crates/mlrs-backend/tests/{topk,eig,distance,sgd}_test.rs`). Make it a per-test-file checklist line for Phases 12–16.
- Add **named per-family f32 bands** (continue v1's `Tolerance::for_family` / `docs/tolerance-policy.md`), never a global loosen. Predicted band needs (MEDIUM — measure on rocm):
  | Feature | f32-on-rocm strict 1e-5? | Hard gate kept exact |
  |---|---|---|
  | KNN-graph distances/indices | indices exact; distances band-likely | neighbour **indices** exact (modulo documented tie-break) |
  | UMAP fuzzy set (`smooth_knn_dist`/membership) | band needed (exp + binary search) | value-band vs umap-learn on the deterministic graph |
  | UMAP layout | N/A — **property gate** (Pitfall 5) | trustworthiness/kNN-overlap floors; determinism exact |
  | HDBSCAN core dist / mutual-reach | band likely | feeds exact-label gate |
  | HDBSCAN labels | exact | **exact labels** via `−1`-pinned label_perm |
  | HDBSCAN probabilities_ / stabilities | band needed | named band; labels exact |

**Warning signs:**
A new `*_test.rs` f64 case failing only under `--features rocm` with an F64/HIP registration error; UMAP/HDBSCAN continuous oracle "almost matches" with no named band; a global tolerance being loosened.

**Phase to address:** Every phase 12–16 (skip guard); 13/14 set their family bands during validation.

---

### Pitfall 9: Builder-pattern retrofit explodes into typestate combinatorics and breaks the shipped PyO3 `any_estimator!` machinery

**What goes wrong:**
Retrofitting a Rust-native builder across **30 estimators** has three structural traps:
1. **Typestate explosion:** modelling every hyperparameter's set/unset state in the type system (phantom-typed builders) multiplies types combinatorially — 30 estimators × many params → unmaintainable, slow to compile, and hostile to the macro-generated PyO3 layer.
2. **Breaking `any_estimator!`:** the shipped PyO3 layer (`crates/mlrs-py/src/dispatch.rs`) generates, per estimator, an `Unfit { /* sklearn-named hyperparameters stored verbatim */ } + F32(Estimator<f32>) + F64(Estimator<f64>)` enum and constructs the fitted arm from the **stored verbatim hyperparameters** at `fit`. If the Rust builder changes the *construction surface* (e.g. `Estimator::new(k)` → `Estimator::builder().k(k).build()?`, or makes fields private, or moves defaults into the builder), every `any_estimator!` expansion that does `KMeans::<f32>::new(..)` must change in lockstep, and the "store verbatim hyperparameters then build at fit" contract can silently drift.
3. **Two construction paths fighting:** the sklearn-mirror constructors (consumed via PyO3) and the new idiomatic Rust builders must coexist; if the builder becomes the *only* path, the macro breaks; if they diverge, defaults drift (Pitfall 10).

**Why it happens:**
"Idiomatic Rust builder" tutorials reach for full typestate (compile-time required-field enforcement). And a 30-estimator mechanical sweep is the highest large-blast-radius regression risk in the milestone.

**How to avoid:**
- **Prefer a simple owned-builder (or `derive_builder`-style) over full typestate.** Use the **fit/unfit typestate at the estimator level only** (`Estimator<Unfit>` → `fit` → `Estimator<Fitted>`), which mlrs already expresses via the `Fit<F>` trait + `any_estimator!`'s Unfit/F32/F64 enum — extend that, don't invent per-param phantom types. Required-vs-optional is enforced by `build() -> Result<_, BuildError>` (mlrs already has `BuildError` in `sgd_config.rs`), not by the type system.
- **Keep `new(...)` as a thin wrapper over the builder** so `any_estimator!` and the sklearn-mirror constructors keep working unchanged; the builder is *additive*. Audit every `any_estimator!` call site (`grep -rn any_estimator crates/mlrs-py`) when changing any constructor.
- **Land the builder convention on 1–2 estimators first** (e.g. `NearestNeighbors`, `KMeans`) with the macro still green, write the migration recipe, THEN sweep the remaining 28. Don't sweep blind.

**Warning signs:**
A `PhantomData` per hyperparameter; compile-time blowup; `any_estimator!` expansions failing to compile after a constructor change; the builder being the only way to construct an estimator.

**Phase to address:** Phase 15 (builder convention + retrofit). Establish the convention on a pilot estimator under the green `any_estimator!`/PyO3 suite before the mechanical sweep.

---

### Pitfall 10: Default-value drift between the Rust builder, the sklearn-mirror constructors, and sklearn's actual defaults

**What goes wrong:**
The builder introduces a *new* place to define defaults. If a builder default disagrees with the sklearn default (or with the existing sklearn-mirror constructor), an estimator silently behaves differently depending on construction path — and the PyO3 layer (which must mirror sklearn) ships the wrong default. Example shapes: UMAP `n_neighbors=15`, `min_dist=0.1`, `metric='euclidean'`; HDBSCAN `min_cluster_size=5`, `min_samples=None→min_cluster_size`, `cluster_selection_method='eom'`. A builder defaulting `min_samples=5` instead of `=min_cluster_size` changes results.

**Why it happens:**
Defaults get re-typed in the builder by hand; the `min_samples=None` "derive from min_cluster_size" sklearn convention is easy to hard-code wrong.

**How to avoid:**
- **Single source of truth for defaults:** define them once (a `Default` impl or const block) and have both `new(...)` and `builder()` read from it; never duplicate literal defaults across the two paths.
- Add a test asserting **builder-default == sklearn-default** for every hyperparameter that the PyO3 layer mirrors (a small table-driven test), and that `new()` and `builder().build()` produce identical configs.
- Encode `None`-derived defaults (e.g. HDBSCAN `min_samples`) as an explicit `Option` resolved at `build()`/`fit()`, matching sklearn's resolution point.

**Warning signs:**
The same default literal appearing in two files; PyO3 default ≠ sklearn default; results differing between `new()` and `builder()`.

**Phase to address:** Phase 15 (and 13/14 set the UMAP/HDBSCAN defaults table that the shim mirrors).

---

### Pitfall 11: The pure-Python sklearn shim fails `check_estimator` on get_params types, clone semantics, fit idempotence, and trailing-underscore attribute naming

**What goes wrong:**
The new pure-Python sklearn shim (`get_params`/`set_params`/`check_estimator`) has a well-known cluster of `check_estimator` failure modes, all of which mlrs is exposed to because the estimators are PyO3-backed (state lives in Rust):
1. **`get_params` returns wrong types / not the constructor args:** sklearn requires `get_params()` to return exactly the `__init__` parameters, by their exact names, as the *original Python objects* (ints stay ints, `None` stays `None`) — not Rust-coerced values, not fitted attributes. A PyO3 getter returning a Rust-coerced `u64` where `__init__` took `int`, or omitting a param, fails `check_estimator`.
2. **`clone()` semantics:** sklearn's `clone(est)` calls `get_params(deep=False)` then `type(est)(**params)` and requires the clone be **unfitted** with **identical params** and **no shared mutable state**. If the PyO3 wrapper's `__init__` mutates/normalises params, `clone` produces a non-equal estimator → fails. Params must be stored verbatim (the `any_estimator!` "stored verbatim hyperparameters" contract already does this on the Rust side — mirror it in Python).
3. **Fit idempotence / re-fit:** calling `fit` twice must fully reset state (no accumulation); `check_estimator` calls `fit` on different data and expects clean state. The `any_estimator!` mutex-poisoning guard (WR-02/WR-04) must reset to Unfit on re-fit, not retain the previous monomorphization.
4. **Trailing-underscore convention:** fitted attributes MUST end in `_` (`embedding_`, `labels_`, `probabilities_`, `components_`) and MUST NOT exist before `fit` (accessing them pre-fit must raise `NotFittedError`/`AttributeError`). Estimators must NOT have trailing-underscore attributes set in `__init__`, and constructor params must NOT have trailing underscores. A getter that returns a default pre-fit instead of raising fails `check_estimator`.

**Why it happens:**
`check_estimator` encodes ~40 invariants most reimplementations don't know; the Rust↔Python boundary makes it easy to leak coerced types or fitted state.

**How to avoid:**
- `get_params` returns the verbatim Python `__init__` kwargs (store them on the Python side as given; do NOT round-trip through Rust). `set_params` updates them and invalidates fitted state.
- Implement `__init__` as **pure assignment, no validation/coercion** (sklearn rule: validate in `fit`, not `__init__`) so `clone` round-trips exactly.
- On `fit`, reset to Unfit first; raise `NotFittedError` for `_`-attributes pre-fit; expose fitted state only via `_`-suffixed properties that pull from Rust.
- Run sklearn's `check_estimator` / `parametrize_with_checks` against each shimmed estimator. **Note (deferred reality):** *live* FFI `estimator_checks` needs a maturin+pyarrow host this env lacks (project memory + PROJECT.md) — so author the shim to the documented invariants and route live verification to UAT, the same way v1/v2 did. Pure-Python invariant unit tests (get_params returns ctor kwargs, clone equality, double-fit reset, pre-fit raises) CAN run without the GPU/FFI host and should gate in CI.

**Warning signs:**
`get_params()` returns fitted attrs or coerced types; `clone()` produces an estimator that isn't `==` on params; second `fit` accumulates; `_`-attributes readable before fit; constructor doing validation.

**Phase to address:** Phase 16 (pure-Python sklearn shim). Build the invariant unit tests (FFI-free) into CI; route live `check_estimator` to UAT.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Dense `n×n` distance for the KNN-graph (no ANN / pynndescent) | Reuses v1 distance + `top_k`; 100% cpu-MLIR-safe; exact (not approximate) neighbours | `O(n²)` mem/time; caps UMAP/HDBSCAN fixture size | Acceptable for v3 (matches exact-neighbour oracle discipline); ANN is a later lift |
| MST + condensed-tree + stability on the HOST (not device) | Sidesteps atomics/union-find cpu-MLIR panic; trivial deterministic tie-break; matches reference serial Cython | Host↔device hop for the (small) graph; not GPU-parallel | Acceptable for v3 (the deliberate "dodge tree-atomics" choice); revisit only if graph stage dominates |
| Host-generate SplitMix64 negative-sample indices then upload | Avoids a device RNG kernel for UMAP sampling; trivially reproducible | Host↔device copy crosses the zero-copy boundary | Acceptable if seeded+reproducible and within the PoolStats gate; prefer device RNG if the copy violates the gate |
| `new(...)` kept as a thin wrapper over `builder()` (no full typestate) | Keeps `any_estimator!`/PyO3 + sklearn-mirror constructors working unchanged; small blast radius | Required-field enforcement is runtime (`build()->Result`), not compile-time | Acceptable / preferred — full per-param typestate is the over-engineering trap (Pitfall 9) |
| One global tolerance reused for UMAP/HDBSCAN continuous outputs | Less bookkeeping | Hides genuine f32-on-rocm bands; a too-loose global masks bugs | Never globally loosen; add a named per-family band (Pitfall 8) |
| Shimming `check_estimator` to documented invariants (no live FFI run) | Unblocks the shim without a maturin+pyarrow host | Live sklearn-suite gaps until a UAT host runs it | Acceptable (env constraint, project memory); FFI-free invariant tests must still gate in CI |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| umap-learn (UMAP oracle) | Value-matching the embedding to 1e-5; comparing to cuML | Property gate (trustworthiness/kNN-overlap **relative to umap-learn's own score**) + value-gate only the deterministic `fuzzy_simplicial_set` (Pitfall 5) |
| hdbscan / sklearn.cluster.HDBSCAN (HDBSCAN oracle) | Comparing to cuML (incomplete epsilon/outlier paths); not pinning MST tie-break; `label_perm` permuting `−1` | Gate vs `hdbscan` Python ref; stable-sort MST edges; `−1`-pinned label_perm; jittered + tied fixtures (Pitfall 6) |
| PyO3 `any_estimator!` macro | Changing a constructor signature without updating every macro call site; builder as the only construction path | Builder is additive; `new()` stays; audit `grep -rn any_estimator crates/mlrs-py` on any ctor change (Pitfall 9) |
| sklearn `check_estimator` | `get_params` returns coerced types/fitted attrs; `__init__` validates; `_`-attrs pre-fit | get_params = verbatim ctor kwargs; validate in `fit`; `NotFittedError` pre-fit (Pitfall 11) |
| cpu-MLIR (cubecl-cpu) | New KNN/UMAP/MST kernel with Atomic/SharedMemory/`F::INFINITY`/mutable-bool/shift-loop | Compose from launch-proven `top_k`; single-owner GATHER; host-side hierarchy (Pitfalls 1,2,7) |
| rocm (cubecl-cpp 0.10) | f64 oracle case launched on rocm (F64 unregistered → fail) | `if capability::skip_f64_with_log() { return; }` in every f64 case (Pitfall 8) |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full `n×n` distance resident for KNN-graph | OOM / PoolStats-gate failure as n grows | Tile the query axis; `top_k` per block; keep big operand global, never resident-full | n² exceeds buffer-pool budget |
| Re-running full UMAP layout for the property gate at large n | Test-suite time balloons (backend suite already ~6min, project memory) | Cap fixture n for the property gate; background the full run; targeted post-merge gates | every added UMAP/HDBSCAN `*_test.rs` |
| Host MST on a dense graph at large n | `O(n²)` edges to the host | Build MST from the **k-graph** (`O(n·k)` edges), not the dense matrix | n large with dense edge list |
| f32 core-distance parallel reduction reordered | Non-reproducible f32; label flips | Pin ascending accumulation order, single-owner (Pitfall 7) | parallel float accumulation |

## "Looks Done But Isn't" Checklist

- [ ] **KNN-graph prim:** Composed from launch-proven `top_k` (no new heap kernel); `--features cpu` **launch** verified (not just compile); self-neighbour dropped (query k+1); no `Atomic`/`SharedMemory`/`F::INFINITY`/mutable-bool/shift-loop (Pitfalls 1,3).
- [ ] **UMAP layout:** Vertex-owner GATHER (not edge-scatter); cpu launch passes; same SplitMix64 seed → byte-identical embedding across runs (Pitfall 2).
- [ ] **UMAP gate:** Property gate present (trustworthiness + kNN-overlap **relative to umap-learn**, + determinism); `fuzzy_simplicial_set`/`smooth_knn_dist` value-gated ≤1e-5; NO embedding value oracle; NOT a shape-only smoke test (Pitfall 5).
- [ ] **HDBSCAN:** Stable-sorted MST edges + documented deterministic tie-break; `label_perm` **pins `−1`→`−1`**; `min_samples` vs `min_cluster_size` semantics correct; tested on jittered AND explicitly-tied fixtures; gated vs `hdbscan` not cuML (Pitfall 6).
- [ ] **HDBSCAN device/host split:** distance+knn+core+mutual-reach on device (GATHER), MST+condensed-tree+stability on host; no Atomic/union-find kernel; core-dist accumulation order pinned (Pitfall 7).
- [ ] **Every f64 oracle case:** Has `if capability::skip_f64_with_log() { return; }` (Pitfall 8).
- [ ] **f32 bands:** UMAP fuzzy-set / HDBSCAN probabilities have named bands; labels exact; UMAP layout is property-gated (Pitfall 8).
- [ ] **Builder retrofit:** Piloted on 1–2 estimators with `any_estimator!`/PyO3 suite green BEFORE the 28-estimator sweep; `new()` still works; no per-param `PhantomData` (Pitfall 9).
- [ ] **Defaults:** Single source of truth; builder-default == new()-default == sklearn-default (table test); HDBSCAN `min_samples=None→min_cluster_size` resolved at build/fit (Pitfall 10).
- [ ] **sklearn shim:** `get_params` returns verbatim ctor kwargs; `__init__` pure assignment; double-`fit` resets; `_`-attrs raise pre-fit; FFI-free invariant unit tests gate in CI; live `check_estimator` routed to UAT (Pitfall 11).
- [ ] **Memory gate:** Every new prim/estimator has its build-failing PoolStats gate — not deferred (per-phase discipline).

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| cpu-MLIR panic in KNN/UMAP/MST kernel (1,2,7) | MEDIUM | Compose from `top_k`; invert to single-owner-per-vertex/cell GATHER; move hierarchy logic to host; drop `F::INFINITY`/mutable-bool; re-verify cpu launch |
| Self-neighbour / symmetrisation wrong (3) | LOW | Query k+1 + drop self; apply correct per-consumer symmetrisation (union vs mutual-reach) as GATHER |
| LDS / PoolStats overflow on `n×n` (4) | MEDIUM | Move big operand to global; tile the query axis; assert tile bytes < 65536 |
| UMAP value oracle chosen (5) | LOW | Delete value fixture; add trustworthiness/kNN-overlap-vs-umap-learn + determinism; value-gate only the fuzzy set |
| HDBSCAN labels diverge (6) | MEDIUM | Stable-sort MST + deterministic tie-break; `−1`-pin label_perm; regen jittered+tied fixtures via /tmp venv; gate vs hdbscan |
| f64-on-rocm not skipped (8) | LOW | Add the `skip_f64_with_log` guard line |
| `any_estimator!` broken by builder (9) | MEDIUM-HIGH | Revert constructor to thin wrapper; pilot on 1–2 estimators; audit every macro call site before re-sweeping |
| Default drift (10) | LOW | Single-source defaults; add builder==new==sklearn default test |
| sklearn shim check_estimator fails (11) | LOW-MEDIUM | get_params=verbatim kwargs; move validation to fit; raise NotFittedError pre-fit; reset on re-fit |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. KNN-graph atomics/SharedMemory | Phase 12 | Composed from `top_k`; `--features cpu` launch passes; no forbidden imports/literals |
| 2. UMAP edge-scatter layout | Phase 13 | Vertex-owner GATHER; cpu launch passes; identical-seed reproducibility |
| 3. Self-neighbour / symmetrisation / ties | Phase 12 | Self dropped (k+1); correct per-consumer symmetrisation as GATHER |
| 4. `n×n` LDS / memory-gate overflow | Phase 12 | Tile bytes < 65536; global big operand; PoolStats gate green at fixture sizes |
| 5. UMAP stochastic-gate (make-or-break) | Phase 13 | Property gate (trustworthiness/kNN-overlap vs umap-learn + determinism); fuzzy-set value-gated; no value oracle |
| 6. HDBSCAN label divergence | Phase 14 | Stable-sort MST + deterministic tie-break; `−1`-pinned label_perm; jittered+tied fixtures; vs hdbscan |
| 7. MST/tree atomics + float reorder | Phase 14 | Host-side MST/tree; device GATHER for distance/core/mutual-reach; pinned accumulation |
| 8. f32 bands + f64-on-rocm skip | Phases 12–16 | `skip_f64_with_log` in every f64 case; named family bands; labels exact |
| 9. Builder typestate / `any_estimator!` break | Phase 15 | Pilot estimator green under PyO3 suite; `new()` works; no per-param PhantomData; call sites audited |
| 10. Default-value drift | Phase 15 (+13/14 defaults) | builder==new==sklearn default table test; `min_samples` None-resolution correct |
| 11. sklearn shim check_estimator | Phase 16 | FFI-free invariant tests in CI (get_params/clone/double-fit/pre-fit); live check_estimator → UAT |

## Sources

- v1/v2 codebase idioms (HIGH): `crates/mlrs-backend/src/prims/topk.rs` (launch-proven `top_k`, ascending, **lowest-index tie-break**, no SharedMemory/INFINITY/shift-loop), `prims/distance.rs`, `crates/mlrs-py/src/dispatch.rs` (`any_estimator!` Unfit/F32/F64 + stored-verbatim-hyperparameters + mutex-poisoning WR-02/WR-04), `crates/mlrs-algos/src/linear/sgd_config.rs` (`BuildError`, builder precedent), `tests/{topk,eig,distance,sgd}_test.rs` (`capability::skip_f64_with_log` idiom), `crates/mlrs-core` sign_flip/label_perm/tolerance helpers (D-08).
- Project memory + PROJECT.md (HIGH): cubecl-cpu no-SharedMemory/no-atomics launch panic + GATHER fix; mutable-bool/`F::INFINITY`/shift-loop panic on cpu-MLIR; rocm f64-unsupported skip-with-log (cubecl-cpp 0.10); gfx1100 LDS 65536 B; per-phase PoolStats memory gate; SplitMix64 reproducible PRNG (no OsRng); v2 D-12 RandomProjection property-gate precedent; v2 D-08 exact-label rule; Python-wheel-untestable-in-env (live check_estimator → UAT); oracle-fixture /tmp-venv regen; "dodge tree-atomics risk" framing.
- v2.0 PITFALLS.md (HIGH): two-pass GATHER SGD idiom (vertex-owner UMAP layout mirrors it), zero-degree/`F::INFINITY` guard, LDS-budget audit, f32-band-vs-exact-label policy.
- CONCERNS.md (HIGH, cuML reference behaviour to NOT replicate): UMAP vertex-parallel non-determinism bug (`optimize_batch_kernel.cuh`); cuML HDBSCAN `cluster_selection_epsilon`/outlier-score incompleteness (`extract.cuh`/`select.cuh`/`membership.cuh`) → gate vs hdbscan not cuML; UMAP/SMO/tree CUDA porting risk (don't port the 1,165-line kernel); warp-size NVIDIA assumptions.
- hdbscan reference (HIGH, verified against source): `scikit-learn-contrib/hdbscan/hdbscan_.py` — stable `np.argsort` MST-edge tie order; noise `−1`; `probabilities_` via Cython; `min_samples` (`query(X, k=min_samples+1)`) vs `min_cluster_size`; parallel float-accumulation nondeterminism — https://github.com/scikit-learn-contrib/hdbscan/blob/master/hdbscan/hdbscan_.py
- UMAP validation practice (HIGH/MEDIUM): trustworthiness + kNN-preservation + Shepard/stress as the structure-preservation gate; different UMAP implementations land at *similar* trustworthiness (comparability = the gate); 5–15% coordinate variation across runs → value-matching is impossible — https://medium.com/data-science/on-the-validating-umap-embeddings-2c8907588175 , https://direct.mit.edu/neco/article/33/11/2881/107068/Parametric-UMAP-Embeddings-for-Representation-and
- sklearn estimator contract (HIGH): `check_estimator`/`get_params` returns ctor params; `clone` = `type(est)(**get_params(deep=False))`; validate in `fit` not `__init__`; trailing-`_` fitted attrs; `NotFittedError` pre-fit — https://scikit-learn.org/stable/developers/develop.html

---
*Pitfalls research for: mlrs v3.0 UMAP + HDBSCAN + KNN-graph prim + Rust-native builder retrofit + pure-Python sklearn shim, on cpu(f64)+rocm(f32) / sklearn ≤1e-5 + umap-learn property gate + hdbscan exact-label gate*
*Researched: 2026-06-22*
