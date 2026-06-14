# Pitfalls Research

**Domain:** Adding ~16 sklearn-compatible estimators to mlrs (Rust/CubeCL rewrite of cuML), gated cpu(f64) + rocm(f32) vs the scikit-learn ≤1e-5 oracle
**Researched:** 2026-06-14
**Confidence:** HIGH for backend/cpu-MLIR and oracle traps (grounded in v1 codebase idioms + project memory); HIGH for the sklearn parity math (verified against sklearn source); MEDIUM for the exact f32-on-rocm band magnitudes (must be measured empirically per family, as in v1)

This file is scoped to **these estimators on THIS backend**. Generic ML advice is omitted. Every pitfall names the estimator/phase it hits, gives the concrete GATHER rewrite / sign convention / oracle construction, and maps to a phase. Phase numbers continue v1 numbering: **Phase 7 = Covariance & projection, 8 = Kernel, 9 = Spectral, 10 = SGD/linear-SVM, 11 = Naive Bayes** (per `seeds/v2-breadth-roadmap.md`, "Phase numbering continues from v1.0").

---

## Critical Pitfalls

### Pitfall 1: SGD minibatch weight update naively wants cross-unit atomics (cpu-MLIR launch panic)

**What goes wrong:**
The obvious SGD kernel parallelizes over the `batch_size` samples and has each sample-thread accumulate its gradient contribution into the *shared* weight vector `w[j] += -lr * grad_ij`. On the cpu MLIR backend this either needs `Atomic::add` across units or a SharedMemory reduction with a mutable accumulator — exactly the pattern that COMPILED but PANICKED at launch in v1 ("failed to run pass" in cubecl_cpu MLIR lowering). On rocm it serializes/races.

**Why it happens:**
SGD is "embarrassingly per-sample" so the instinct is one thread per sample. But every sample writes the *same* output cells (the weight vector), which is a multi-writer reduction — the thing cpu-MLIR cannot lower (no cross-unit atomics) and the thing the v1 memory note explicitly warns against (mutable accumulators in SharedMemory).

**How to avoid (GATHER rewrite):**
Split each minibatch step into TWO single-owner passes, mirroring the v1 GATHER idiom (single-owner outputs, only F/u32 accumulators, if-guards, no SharedMemory):
1. **Gradient-build pass** — one thread per *weight coordinate* `j` (single owner of `g[j]`). Each `g[j]` thread reads the whole minibatch and computes `g[j] = (1/B) * Σ_i loss'(margin_i) * x_ij + reg'(w_j)` with an ascending scan over `i`. The margin `margin_i = Σ_j w_j x_ij` is precomputed by a prior GEMV pass (reuse v1 GEMM, single-owner per row). No two threads write the same cell.
2. **Apply pass** — one thread per `j`: `w[j] = w[j] - lr * g[j]` (and the averaged-SGD running mean as a second single-owner buffer).
This makes the whole step a sequence of GATHER kernels with F-only accumulators and no SharedMemory. Loss derivatives (hinge / log / squared / ε-insensitive) become branch-on-`u32`-flag, not mutable-bool while-loops.

**Warning signs:**
Kernel compiles under `--features cuda`/`wgpu` but the `--features cpu` test panics at *launch* (not compile) with an MLIR "failed to run pass" message; or any `Atomic`/`SharedMemory` import in the SGD kernel module.

**Phase to address:** Phase 10 (SGD solver prim). This is the single genuinely-new solver of v2 and the highest-risk cpu-MLIR item — spike the two-pass GATHER kernel before wiring any of the four estimators (MBSGDClassifier/Regressor, LinearSVC/SVR).

---

### Pitfall 2: Naive Bayes per-class accumulation naively wants scatter-add into class bins (cpu-MLIR panic)

**What goes wrong:**
The natural NB fit kernel loops over samples and does `class_sum[y_i, j] += x_ij` — a scatter-add keyed by the label `y_i`. Multiple sample-threads hit the same `(class, feature)` cell → cross-unit atomic / mutable SharedMemory accumulator → cpu-MLIR launch panic, same failure class as Pitfall 1.

**Why it happens:**
Per-class sufficient statistics (class counts, feature sums, feature sum-of-squares for GaussianNB) feel like a histogram, and histograms are the canonical atomics use-case. But the cpu-MLIR backend forbids exactly this.

**How to avoid (GATHER rewrite):**
Invert the parallelism so each output cell has ONE owner:
- One thread per `(class c, feature j)` cell. That thread scans all `n` samples once (ascending) with an `if (y_i == c)` guard and accumulates into its private register, writing `theta[c,j]` exactly once. Class counts are one thread per class with the same guard.
- For GaussianNB also accumulate `Σ x²` in a parallel single-owner buffer (then `var = Σx²/N - mean²`, matching sklearn's biased population variance, see Pitfall 9).
- CategoricalNB: one owner per `(class, feature, category)` cell.
Cost is `O(K·d·n)` reads but n is the only large axis and reads are cheap/coalescable; v2 problem sizes make this fine and it is 100% GATHER (F/u32 accumulators, u32 label compare, if-guard).

**Warning signs:** Any label-indexed write target `out[y_i]`; any `Atomic` in NB kernels.

**Phase to address:** Phase 11 (Naive Bayes family — Gaussian/Multinomial/Bernoulli/Complement/Categorical). "Reductions only" in the seed roadmap is correct *only if* the per-class reductions are written as single-owner GATHER kernels.

---

### Pitfall 3: Graph-Laplacian degree normalization wants a scatter over edges (cpu-MLIR) and silently breaks on zero-degree / disconnected nodes

**What goes wrong:**
Building the normalized Laplacian `L_sym = I - D^{-1/2} W D^{-1/2}` from an affinity `W`, the naive kernel iterates over edges and scatter-adds into a degree vector `D`, then divides. Edge-scatter is multi-writer (cpu-MLIR panic). Separately, when a node has degree 0 (isolated point at the chosen affinity/`n_neighbors`), `D^{-1/2}` is `inf` → NaN rows → eig returns garbage. `F::INFINITY` literals in the kernel are also on the v1 cpu-MLIR forbidden list.

**Why it happens:**
Degree is a row-sum that *looks* like a reduction over a sparse edge list; and dense affinity construction hides the zero-degree case until a sparse/`n_neighbors` graph exposes it.

**How to avoid (GATHER rewrite + guard):**
- Keep `W` dense at v2 sizes and compute `D[i]` as a **single-owner row-reduction** (one thread per row `i` sums row `i`) — reuse the v1 reduce prim, no scatter, no SharedMemory accumulator.
- Compute `d_inv_sqrt[i]` with an explicit guard producing a typed zero, NOT infinity: `if D[i] > eps { rsqrt(D[i]) } else { 0 }`. Do not emit `F::INFINITY`. Then `L[i,j] = (i==j ? 1 : 0) - d_inv_sqrt[i]*W[i,j]*d_inv_sqrt[j]`, one owner per `(i,j)`.
- This matches sklearn's `_set_diag` / degree handling, which guards against zero degrees rather than dividing by them.

**Warning signs:** NaN/inf in the Laplacian; eig returning constant or exploding eigenvectors; any `F::INFINITY` or edge-indexed scatter in the Laplacian kernel.

**Phase to address:** Phase 9 (graph-Laplacian prim → SpectralEmbedding / SpectralClustering).

---

### Pitfall 4: Spectral embedding takes the WRONG eigenvectors (sign + the zero-eigenvector skip + ordering)

**What goes wrong:**
v1's Jacobi eig returns the **full spectrum in descending order** (the convention used by PCA/TruncatedSVD). Spectral methods need the **smallest** nontrivial eigenvectors of the Laplacian, and must **drop the first** (the constant eigenvector at eigenvalue ≈0). Three independent ways to get it wrong: (a) taking the largest instead of smallest, (b) keeping the trivial eigenvector 0, (c) per-component sign disagreeing with sklearn so every oracle value is off by a sign.

**Why it happens:**
Reusing the PCA descending-order eig and "take top-k" muscle memory is exactly backwards for Laplacians. And eigenvectors are only defined up to sign, so even a numerically perfect embedding fails the 1e-5 value gate without sign canonicalization.

**How to avoid (selection + sign convention + oracle):**
- **Selection:** sort ascending, drop index 0 (the ~0 eigenvalue / constant vector), take the next `n_components`. For SpectralClustering the embedding then feeds KMeans (reuse v1) — and the *clustering label* is the witness, compared with the v1 `label_perm` best-mapping helper (labels are permutation-invariant), so embedding sign does NOT need to match there.
- **For SpectralEmbedding's transform output**, the embedding values DO need sign canonicalization: apply sklearn's `_deterministic_vector_sign_flip` rule — flip each eigenvector so its entry of largest absolute value is positive — and reuse the v1 `sign_flip`/`align_sign` helper in the oracle comparison.
- **Degenerate/near-equal eigenvalues:** when eigenvalues cluster, the eigenbasis is only defined up to rotation within the degenerate subspace → value-match is impossible. For those fixtures fall back to a subspace/property test (Pitfall 13/below).

**Warning signs:** Embedding is constant or rank-deficient (kept the trivial vector); oracle off by exact sign; oracle fails only on symmetric/blocky inputs (degenerate spectrum).

**Phase to address:** Phase 9 (selection + sign), with the oracle policy decided up front.

---

### Pitfall 5: IncrementalPCA batch-merge gets the mean-correction factor, stacking, sign, or ddof wrong

**What goes wrong:**
IncrementalPCA is NOT "PCA on a running covariance." sklearn's `partial_fit` does a specific SVD merge; four independent off-by-ways each break 1e-5 parity:
1. Wrong **mean-correction row**: must be `sqrt((n_seen/n_total) * n_batch) * (mean_old - batch_mean)`.
2. Wrong **stack order/scaling**: SVD is taken on `vstack([ S.reshape(-1,1) * components_old , X_centered , mean_correction ])` — previous components scaled by their singular values, then the new centered batch, then the correction row.
3. Wrong **sign**: sklearn uses `svd_flip(U, Vt, u_based_decision=False)` (V-based), whereas standard PCA uses U-based — using the wrong one flips component signs vs the oracle.
4. Wrong **ddof**: `explained_variance_ = S**2 / (n_total - 1)` (ddof=1, Bessel), while GaussianNB/var uses population variance (ddof=0). Mixing these up is a classic parity miss.

**Why it happens:**
Every term is plausible-looking, and the merge math only appears in sklearn's source, not the docstring. The U-vs-V `svd_flip` flag and the ddof choice are silent until the oracle diff appears.

**How to avoid:**
Port `sklearn/decomposition/_incremental_pca.py` line-for-line for the merge (the four quoted facts above are the spec), reuse the v1 Jacobi SVD for the per-batch decomposition, and reuse the v1 `sign_flip` helper with the **V-based** convention in the oracle. Verify the merge against a multi-batch oracle, not just single-batch (single-batch hides the correction-row bug).

**Warning signs:** Single-batch passes but 2+ batches drift; components sign-flipped vs oracle; explained_variance off by exactly an `(n-1)/n` factor (ddof slip).

**Phase to address:** Phase 7 (incremental-SVD merge prim → IncrementalPCA). This is the `[v2-P1]` open research question — settle "full Jacobi per batch vs dedicated update kernel" and the f32-on-rocm stability of the merge here.

---

### Pitfall 6: RandomProjection is value-unmatchable against sklearn (different RNG) — wrong oracle kills the phase

**What goes wrong:**
Trying to oracle-match the actual projection matrix or transformed output against sklearn fails by construction: sklearn fills the Gaussian/sparse matrix with NumPy's MT19937 / its own sparse sampler; mlrs uses a seeded SplitMix64-style device/host PRNG (project memory + `questions.md` note: no OsRng, reproducible seeded PRNG required). Identical seeds across two different PRNGs produce different matrices → every transformed value differs → a value oracle reports total failure on a correct implementation.

**Why it happens:**
The v1 oracle pattern is "same random input → compare values ≤1e-5," and RandomProjection breaks the unstated assumption that the *algorithm* is deterministic given the input. The randomness is internal and RNG-specific.

**How to avoid (property test, not value oracle):**
Test the *mathematical guarantees*, not the values:
- **Shape/density:** Gaussian matrix entries ~ N(0, 1/n_components) (test mean≈0, variance≈1/n_components over the matrix). Sparse matrix has the expected nonzero density `1/sqrt(n_features)` (or `density`) and values in `{-s, 0, +s}` with `s = sqrt(1/(density·n_components))`.
- **Johnson–Lindenstrauss distortion:** sampled pairwise distances are preserved within the JL `eps` bound after projection — the actual contract RandomProjection promises.
- **Determinism:** same seed → identical mlrs output across runs (reproducibility, ASVS V6).
- **`johnson_lindenstrauss_min_dim`** helper value-matches sklearn (it is pure arithmetic, no RNG) — gate that at 1e-5.

**Warning signs:** Anyone writing a `.npz` value oracle for `transform()` output of a projection; "RandomProjection fails 1e-5 everywhere" (means a value oracle was wrongly chosen).

**Phase to address:** Phase 7 (RNG-matrix prim → Gaussian/SparseRandomProjection). Decide the property-test contract in the phase plan so no one wastes time on a value fixture.

---

### Pitfall 7: SGD parity is undefined unless the oracle pins shuffle, schedule, and stopping — and cuML is NOT the oracle

**What goes wrong:**
SGD has three sources of nondeterminism that make a default-config oracle impossible to match: (a) per-epoch sample **shuffling** (RNG-dependent, and mlrs's PRNG ≠ NumPy's), (b) the **learning-rate schedule** (sklearn default `optimal`: `eta = 1/(alpha*(t + t0))` with Bottou's `t0` heuristic — cuML's MBSGD defaults to a *constant* schedule and produces materially different weights, confirmed by RAPIDS issues #2113/#2114), and (c) early stopping via `tol`/`n_iter_no_change` triggering at a different iteration. Picking cuML as the reference, or leaving defaults on, yields a moving target.

**Why it happens:**
The milestone explicitly warns "sklearn (NOT cuML) is the oracle — cuML diverges on SGD loss set + LinearSVC solver," but it's tempting to compare against cuML's MBSGD since the estimator names match. They don't agree.

**How to avoid (deterministic SGD oracle, mirroring v1's LogReg recipe):**
Construct a *pinned* oracle exactly as v1 did for LogReg's L-BFGS:
- `shuffle=False` (removes the RNG mismatch entirely) — this is the single most important knob.
- A **fixed schedule with explicit `eta0`** (`learning_rate='constant'` or `'invscaling'` with a fixed `eta0`), so both sides use the identical, RNG-free `eta_t` sequence. Avoid `optimal` for the oracle (its `t0` heuristic adds a derived constant to reproduce).
- **Fixed `max_iter`, `tol=0` (or very tight), `n_iter_no_change=max_iter`** so both run the same number of steps with no early-stop divergence.
- Same `alpha`, `penalty`, `loss`, `fit_intercept`, `average`.
With shuffle off + fixed schedule + fixed iterations, sklearn and mlrs perform the identical deterministic update sequence → weights value-match. Then add a *separate, looser* property test that shuffled training still converges (decreasing loss), not value-matched. Mirror v1's documented LogReg gauge handling for the multinomial/decision-function comparison.

**Warning signs:** SGD oracle "almost matches" and drifts with `max_iter`; comparing to cuML; oracle uses `shuffle=True`.

**Phase to address:** Phase 10. Write the deterministic-oracle spec into the phase plan before kernels.

---

### Pitfall 8: LedoitWolf shrinkage formula computed on the wrong (non-centered / wrong-normalization) quantities

**What goes wrong:**
The Ledoit–Wolf shrinkage intensity is a ratio of specific Frobenius-norm quantities; getting any normalization wrong yields a plausible but parity-failing `shrinkage_`, and then the whole `covariance_ = (1-δ)·S + δ·μ·I` is off. Common errors: using sample covariance with ddof=1 instead of the **biased** (ddof=0, divide by `n`) empirical covariance sklearn uses; computing `mu = trace(S)/n_features` wrong; forgetting that `delta`/`beta` are clipped so `0 ≤ shrinkage ≤ 1`.

**Why it happens:**
The 2004 paper's `mu/beta/delta` terms aren't spelled out in the sklearn docstring, and EmpiricalCovariance in sklearn uses the **maximum-likelihood (biased, ddof=0)** estimator by default, which contradicts the "use ddof=1" habit.

**How to avoid:**
- EmpiricalCovariance baseline: divide by `n` (ddof=0), centered on the sample mean — match sklearn `empirical_covariance` exactly. Reuse the v1 covariance prim but confirm its normalization is ddof=0 (or parameterize it).
- LedoitWolf: port `ledoit_wolf_shrinkage` directly — `mu = trace(emp_cov)/n_features`; `delta_ = ||emp_cov - mu·I||_F² / n_features`; `beta_` from the per-sample term; `shrinkage = clip(beta/delta, 0, 1)`; `cov = (1-shrinkage)·emp_cov + shrinkage·mu·I`. Then gate `shrinkage_` AND `covariance_` against the oracle.

**Warning signs:** `covariance_` off by an `n/(n-1)` factor (ddof slip); `shrinkage_` outside [0,1]; passes for one `n` but not another (normalization scaling with `n`).

**Phase to address:** Phase 7 (covariance family).

---

### Pitfall 9: Naive Bayes underflow / smoothing / variance-convention mismatches

**What goes wrong:**
NB has a cluster of parity-and-stability traps:
- **Log-space:** computing `P = Π p` in probability space underflows to 0 for moderate `d`. Joint log-likelihood must be accumulated in log space; `predict_proba` needs **log-sum-exp** normalization, not `exp` then divide (which overflows/underflows).
- **Smoothing placement:** Multinomial/Bernoulli use additive `alpha` (`feature_log_prob = log(count + alpha) - log(class_count + alpha·n_features)`); getting the denominator term wrong (forgetting `alpha·n_features`) silently shifts every log-prob.
- **GaussianNB var convention:** uses **population variance (ddof=0)** plus `var_smoothing = 1e-9 * max(var)` added to every variance — omitting `var_smoothing`, or using ddof=1, breaks parity and risks divide-by-zero on constant features.
- **ComplementNB** uses the complement-class statistics with a weight normalization step; **BernoulliNB** binarizes inputs at the `binarize` threshold and includes the `log(1 - p)` term for absent features — both easy to drop.

**Why it happens:**
Each NB variant has its own smoothing/normalization quirk; reusing one variant's code for another silently mis-specifies the math.

**How to avoid:**
- Accumulate joint log-likelihood per class; normalize with a log-sum-exp helper (`m + log Σ exp(x-m)`), implemented as a single-owner GATHER reduction over classes (small K) with an F-only running max+sum — no SharedMemory.
- Encode each variant's exact smoothing/denominator from sklearn source as the spec; add the `var_smoothing` term for GaussianNB and ddof=0 variance.
- Gate `predict` labels (exact) AND `predict_log_proba` values at the f64 tolerance.

**Warning signs:** `predict_proba` rows not summing to 1; `-inf` log-probs; GaussianNB NaN on a constant feature (missing `var_smoothing`); off-by-constant log-probs across all classes (smoothing denominator).

**Phase to address:** Phase 11.

---

### Pitfall 10: f64-on-rocm cases run instead of skip-with-log (gate violation)

**What goes wrong:**
Every new prim's f64 oracle path, if not guarded, will attempt to launch on rocm where cubecl-cpp 0.10 has F64 unregistered for HIP → the test fails (not skips), breaking the established cpu(f64)+rocm(f32) gate. This will hit EVERY new test file in Phases 7–11.

**Why it happens:**
It's a per-file boilerplate that's easy to forget when copying a test from a non-f64 path; the failure looks like a real numerical bug.

**How to avoid:**
Mirror the v1 idiom verbatim: every f64 oracle case begins with
```rust
if capability::skip_f64_with_log() { return; }
```
exactly as in `crates/mlrs-backend/tests/gemm_test.rs`, `eig_test.rs`, `distance_test.rs`, `covariance_test.rs`. f32 cases run on rocm; f64 cases run on cpu and skip-with-log on rocm. Make this a checklist item in every Phase 7–11 plan.

**Warning signs:** A new `*_test.rs` f64 case failing only under `--features rocm` with an F64/HIP registration error.

**Phase to address:** Every phase (7–11); enforce as a per-test-file checklist line.

---

### Pitfall 11: Large kernel/Laplacian/Gram tiles overflow the gfx1100 LDS budget (HIP rejects launch)

**What goes wrong:**
KernelRidge/KernelDensity Gram matrices, the dense Laplacian, and any tiled GEMM/eig that stages a big operand in SharedMemory can exceed gfx1100's 65536 B LDS budget → HIP rejects the launch (v1 hit this in Jacobi and kept the big operand in global). New v2 prims (kernel-matrix, Gram, Laplacian) are prime offenders because n×n is the natural tile.

**Why it happens:**
Tiling tutorials stage the working set in shared memory; on gfx1100 a modest n×n f32 tile (e.g. 128×128 = 64 KiB) already blows the budget, and f64 doubles it.

**How to avoid:**
Keep large operands in **global** memory (v1 Jacobi precedent); only stage small, bounded tiles in SharedMemory — and on cpu, no SharedMemory at all (Pitfalls 1–3). Compute LDS bytes = `tile_elems * size_of::<F>()` and assert it stays under budget at kernel-author time; prefer the GATHER/global pattern that has no SharedMemory dependence so the same kernel runs on cpu too.

**Warning signs:** Launch failure only on rocm with an LDS/occupancy/resource error; kernel works on a small fixture but fails as n grows.

**Phase to address:** Phases 8 (kernel/Gram) and 9 (Laplacian); audit any SharedMemory tile size against 65536 B.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Full Jacobi SVD per IncrementalPCA batch (no dedicated update kernel) | Reuses v1 SVD; fast to ship | More flops than a true rank-update; fine at v2 sizes, costly at streaming scale | Acceptable for v2 (settle in `[v2-P1]`); revisit if streaming-large becomes a requirement |
| Dense affinity/Laplacian (no sparse graph) | Avoids edge-scatter (which cpu-MLIR can't do anyway) and `n_neighbors` sparse kernels | O(n²) memory; caps spectral problem size | Acceptable for v2 (matches the `[v2-P3]` "full-spectrum-then-take-smallest" decision); sparse is a v3 lift |
| Full-spectrum eig then slice smallest (vs shift-invert/Lanczos) | Reuses v1 Jacobi eig directly | O(n³); wrong for large n | Acceptable at v2 sizes per `questions.md`; never for large-graph spectral |
| Host-generate-then-upload RNG matrix for RandomProjection | Avoids a device RNG kernel; trivially reproducible | Host↔device copy crosses the zero-copy boundary | Acceptable if seeded + reproducible (ASVS V6) and within the per-phase memory gate; prefer device RNG only if the copy violates the gate |
| One global tolerance reused for new families | Less bookkeeping | Hides genuine f32-on-rocm bands; a too-loose global masks bugs | Never globally loosen; add a *named per-family* band (Pitfall 12) like v1's D-08 growth point |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows. (mlrs v2 targets oracle-sized fixtures; the relevant "scale" is fixture/test problem size.)

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Kernel-matrix materialized dense for KernelRidge/KernelDensity | OOM / pool-gate failure on larger n | Reuse v1 distance prim; for KDE score, fuse the log-sum-exp so the n×n kernel is never fully resident (stream per query row, single-owner) | n² exceeds buffer pool budget |
| NB GATHER kernel re-reads all n per `(class,feature)` cell | Slow fit on wide+long data | Acceptable at v2 sizes; if it bites, tile the sample axis (still single-owner) | very large K·d·n |
| Backend test suite already slow (v1: cpu ~6min) | CI time balloons as Phases 7–11 add suites | Run targeted post-merge gates, background the full run (project memory: backend-test-suite-slow) | every added `*_test.rs` |
| Full-spectrum eig for spectral at large n | O(n³) blowup | Cap fixture n; defer Lanczos to v3 | large graphs |

---

## "Looks Done But Isn't" Checklist

- [ ] **SGD/NB/Laplacian kernels:** Compiles under cuda/wgpu but never run under `--features cpu` — verify the cpu launch (not just compile) passes; cpu-MLIR panics are launch-time (Pitfalls 1–3).
- [ ] **Every f64 oracle case:** Has the `if capability::skip_f64_with_log() { return; }` guard (Pitfall 10).
- [ ] **IncrementalPCA:** Tested with **2+ batches**, not just one — the mean-correction-row bug is invisible single-batch (Pitfall 5).
- [ ] **RandomProjection:** Has a **property/JL** test, NOT a value oracle; `johnson_lindenstrauss_min_dim` value-matched separately (Pitfall 6).
- [ ] **SGD oracle:** `shuffle=False`, fixed `eta0`/schedule, fixed `max_iter`, `tol=0` — and references sklearn, not cuML (Pitfall 7).
- [ ] **Spectral:** Drops the trivial (≈0) eigenvector, takes *smallest*, sign-canonicalized via `_deterministic_vector_sign_flip` for embedding output (Pitfall 4).
- [ ] **NB predict_proba:** Rows sum to 1, computed via log-sum-exp; GaussianNB has `var_smoothing` (Pitfall 9).
- [ ] **EmpiricalCovariance/LedoitWolf:** ddof=0 (biased) normalization, `shrinkage_ ∈ [0,1]`, both `shrinkage_` and `covariance_` gated (Pitfall 8).
- [ ] **LDS budget:** Any SharedMemory tile size asserted < 65536 B on gfx1100 (Pitfall 11).
- [ ] **Memory gate:** Every new prim/estimator has its build-failing PoolStats gate (v1 per-phase discipline) — not deferred.
- [ ] **f32 band:** Any family that can't hit strict 1e-5 in f32-on-rocm has a *documented, named* band with exact-label/argmax as the hard gate (Pitfall 12).

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| cpu-MLIR atomic/SharedMemory panic (1,2,3) | MEDIUM | Rewrite the kernel as single-owner GATHER (invert parallelism to one-thread-per-output-cell, ascending scan, F/u32 accumulators, drop `F::INFINITY`/mutable-bool); re-validate cpu launch |
| Wrong eigenvector selection/sign (4) | LOW | Switch to ascending sort + drop index 0 + sign_flip helper; clustering path is sign-immune via label_perm |
| IncrementalPCA merge wrong (5) | MEDIUM | Re-port from sklearn `_incremental_pca.py`; add multi-batch oracle |
| RandomProjection value oracle chosen (6) | LOW | Delete value fixture; replace with JL/property test (no kernel change) |
| SGD oracle nondeterministic (7) | LOW | Pin shuffle/schedule/iters in the fixture generator; regen `.npz` via /tmp venv |
| f64-on-rocm not skipped (10) | LOW | Add the skip_f64_with_log guard line |
| LDS overflow (11) | MEDIUM | Move the big operand to global memory; shrink/remove the SharedMemory tile |
| Too-strict f32 gate (12) | LOW | Introduce a named per-family band; keep exact-label/argmax as the hard gate |

---

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. SGD update wants atomics | Phase 10 | Two-pass GATHER kernel passes `--features cpu` launch; no Atomic/SharedMemory imports |
| 2. NB per-class scatter-add | Phase 11 | One-owner-per-(class,feature) kernel; cpu launch passes |
| 3. Laplacian degree scatter + zero-degree | Phase 9 | Row-reduction degree; guarded `d_inv_sqrt` (no INFINITY); no NaN in L |
| 4. Spectral wrong/sign eigenvectors | Phase 9 | Ascending+drop-0 selection; embedding sign-matched; clustering via label_perm |
| 5. IncrementalPCA merge | Phase 7 | 2+ batch oracle at f64 1e-5; V-based svd_flip; ddof=1 explained_variance |
| 6. RandomProjection value-unmatchable | Phase 7 | JL/property test present; no transform value fixture; JL-min-dim value-matched |
| 7. SGD oracle nondeterminism / cuML | Phase 10 | Oracle: shuffle=False, fixed schedule/iters, sklearn ref; weights value-match |
| 8. LedoitWolf/EmpCov normalization | Phase 7 | ddof=0; shrinkage∈[0,1]; covariance_ matches across two n |
| 9. NB underflow/smoothing/variance | Phase 11 | log-sum-exp; var_smoothing; predict_log_proba value-matched; proba sums to 1 |
| 10. f64-on-rocm not skipped | Phases 7–11 | skip_f64_with_log guard in every f64 test case |
| 11. LDS budget overflow | Phases 8, 9 | SharedMemory tile bytes asserted < 65536; big operands in global |
| 12. f32-on-rocm band (below) | Phases 7–11 | Named band documented; exact label/argmax hard-gated |

---

## Pitfall 12 (cross-cutting): Which f32-on-rocm cases need a documented band vs strict 1e-5

**What goes wrong:**
Strict 1e-5 is "often physically unreachable in f32" (project memory): ULP exceeds 1e-5 on large magnitudes, and error compounds through iterative/accumulation-heavy kernels. Forcing strict 1e-5 on rocm-f32 fails correct code; loosening the *global* tolerance hides real bugs. v1's answer is **documented per-family bands** with exact labels/argmax as the hard gate.

**Predicted band needs (MEDIUM confidence — measure empirically, as v1 did):**

| Estimator / prim | f32-on-rocm strict 1e-5? | Why / band rationale | Hard gate to keep exact |
|---|---|---|---|
| EmpiricalCovariance | Likely strict | Single-pass sums, low accumulation | values |
| LedoitWolf | Band likely | Ratio of Frobenius norms amplifies f32 error | `shrinkage_` band; covariance band |
| IncrementalPCA | **Band needed** | Iterative SVD merge across batches compounds f32 error; sign already handled | components band + sign; explained_variance band |
| RandomProjection | N/A (property test) | No value oracle (Pitfall 6) | JL bound + density exact |
| KernelRidge | Band likely | RBF `exp(-γ·d²)` + Cholesky solve compound f32 error | predictions band |
| KernelDensity | **Band needed** | log-sum-exp over n kernels; large dynamic range | log-density band |
| SpectralEmbedding | **Band needed** | eig of Laplacian; near-degenerate spectra | embedding band + sign; or subspace test |
| SpectralClustering | Strict on labels | Labels are discrete | **exact labels** via label_perm (sign/band irrelevant) |
| MBSGD*/LinearSVC/SVR | **Band needed** | Iterative SGD accumulates per-step f32 error over epochs | weights band; **exact predicted labels** (classifier) |
| GaussianNB | Band likely | var + log-Gaussian; var_smoothing helps | log-proba band; **exact labels** |
| Multinomial/Bernoulli/Complement/CategoricalNB | Strict-ish on labels | Log-space sums are stable | **exact labels**; log-proba near-strict |

**Rule (from v1 D-08):** classifiers/clusterers keep **exact predicted labels / argmax** as the non-negotiable hard gate even when continuous outputs get a band; regressors/decompositions get a *named* per-family band documented in `docs/tolerance-policy.md`, never a global loosening. Measure the actual band on rocm hardware during each phase and record it.

**Phase to address:** Each of Phases 7–11 sets its own family band during validation (continue v1's `Tolerance::for_family` growth point).

---

## Sources

- v1 codebase idioms (HIGH): `crates/mlrs-backend/src/prims/reduce.rs` (GATHER / single-owner / pairwise-stable, no mutable SharedMemory accumulator), `tests/{gemm,eig,distance,covariance}_test.rs` (`capability::skip_f64_with_log` idiom), `crates/mlrs-core/src/{tolerance,compare,sign_flip,label_perm}.rs` (tolerance policy D-08, sign-flip + label-perm helpers).
- Project memory (HIGH): cubecl-cpu no-SharedMemory/no-atomics launch panic + GATHER fix idiom; rocm f64-unsupported gate (cubecl-cpp 0.10); gfx1100 LDS 65536 B; f32 band policy; oracle-fixture /tmp-venv regen; cuML diverges from sklearn on SGD/LinearSVC.
- Planning docs (HIGH): `.planning/PROJECT.md`, `seeds/v2-breadth-roadmap.md`, `research/questions.md`, `notes/v3-hard-algorithm-backlog.md`.
- scikit-learn parity math (HIGH, verified against source/docs):
  - IncrementalPCA merge (mean_correction sqrt factor, S·components stacking, `svd_flip(u_based_decision=False)`, `S²/(n_total-1)`, noise_variance) — https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_incremental_pca.py
  - LedoitWolf shrinkage (mu/beta/delta, biased empirical_covariance) — https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf_shrinkage.html , https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
  - SGDClassifier schedule (`optimal` = `1/(alpha*(t+t0))`, Bottou t0; eta0/shuffle/random_state) — https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
  - cuML MBSGD diverges from sklearn (constant default schedule, different results) — https://github.com/rapidsai/cuml/issues/2113 , https://github.com/rapidsai/cuml/issues/2114
  - Spectral: drop the zero eigenvalue / smallest nontrivial eigenvectors — https://en.wikipedia.org/wiki/Spectral_clustering , https://en.wikipedia.org/wiki/Laplacian_matrix

---
*Pitfalls research for: mlrs v2.0 breadth-sweep estimators on cpu(f64)+rocm(f32) / sklearn ≤1e-5 oracle*
*Researched: 2026-06-14*
