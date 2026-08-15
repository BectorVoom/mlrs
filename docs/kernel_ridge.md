# `KernelRidge` — the full parameter surface

`mlrs.KernelRidge` implements the whole `sklearn.kernel_ridge.KernelRidge`
parameter surface: all nine `kernel` strings plus a callable, `alpha` as a
scalar or as one penalty per target, `gamma` / `degree` / `coef0`,
`kernel_params`, and `fit(X, y, sample_weight=…)`.

```python
import mlrs

kr = mlrs.KernelRidge(alpha=1.0, kernel="laplacian", gamma=0.5)
kr.fit(X, y).predict(X_test)
```

It fits the DUAL coefficients — `(K + αI)·dual_coef_ = y` over the `n×n` kernel
matrix, not the `d×d` feature-space Gram — so unlike `Ridge` it has no
intercept, does no centering, and its cost is driven by the SAMPLE count rather
than the feature count. Everything below follows from that.

> **Signature order changed.** The constructor now leads with `alpha`, matching
> sklearn (`KernelRidge(alpha=1, *, kernel=…)`); it previously led with
> `kernel`. Keyword calls are unaffected. A positional
> `KernelRidge("rbf")` now raises a `ValueError` naming `alpha`, rather than
> quietly fitting with a penalty of `"rbf"`.

## `kernel` — the one string-valued parameter

| name | `K(x, y)` | base op | reads |
|---|---|---|---|
| `linear` | `⟨x, y⟩` | GEMM | — |
| `poly` / `polynomial` | `(γ⟨x, y⟩ + c₀)^d` | GEMM + `powf` | `gamma`, `degree`, `coef0` |
| `rbf` | `exp(−γ‖x − y‖²)` | squared-euclidean + `exp` | `gamma` |
| `sigmoid` | `tanh(γ⟨x, y⟩ + c₀)` | GEMM + `tanh` | `gamma`, `coef0` |
| `laplacian` | `exp(−γ‖x − y‖₁)` | L1 pairwise + `exp` | `gamma` |
| `cosine` | `⟨x̂, ŷ⟩` | GEMM over normalised rows | — |
| `chi2` | `exp(−γ Σ (xₖ−yₖ)²/(xₖ+yₖ))` | additive-χ² pairwise + `exp` | `gamma` |
| `additive_chi2` | `−Σ (xₖ−yₖ)²/(xₖ+yₖ)` | additive-χ² pairwise | — |
| `precomputed` | the caller's `K` | — | — |
| *a callable* | whatever it returns | Python pairwise loop | `kernel_params` |

Eight of the nine names are evaluated in Rust
(`mlrs-backend/src/prims/kernel_matrix.rs`). Only one new device kernel was
needed for them — the additive-χ² pairwise sum. `laplacian` re-points an
existing base op (the L1 arm of `metric_distance`) at the map `rbf` already
uses; `cosine` is the linear kernel over L2-normalised rows; and `chi2` is that
same `exp` map with a negated γ. The number of transcendental evaluations in
the file did not grow when the kernel count doubled.

`precomputed` and a callable evaluate no kernel at all. A callable is applied in
Python — through sklearn's own `pairwise_kernels`, so the callable sees exactly
the arguments it would from sklearn — and the resulting matrix is handed to the
Rust engine through the `precomputed` path. `kernel_params` reaches the callable
and is ignored for a string kernel, as it is in sklearn.

### Two sklearn behaviours worth knowing before you hit them

**`chi2` has no `gamma` default.** Every other γ-taking kernel resolves
`gamma=None` to `1/n_features`. `chi2` does not: `KernelRidge._get_kernel`
forwards `self.gamma` unconditionally into `chi2_kernel`'s `K *= gamma`, and
`None` raises there. mlrs raises too, with a message that names the parameter
rather than a numpy dtype error. Resolving it to `1/n_features` would have been
the friendlier behaviour and the wrong one — the same call would then return a
number here and an exception in sklearn.

**`additive_chi2` gives an indefinite Gram.** Its diagonal is zero and every
other entry is non-positive, so `(K + αI)` is indefinite at every `alpha` and
the Cholesky cannot factor it. sklearn catches LAPACK's refusal, re-solves in
the least-squares sense, and warns; mlrs does the same, with the same warning
text, so `pytest.warns(UserWarning, match="Singular matrix")` matches either.
Without that fallback the kernel would raise for every input, which is not
"supported". `sigmoid` reaches the same place for many `(γ, coef0)` choices.

The fallback is two solvers behind one entry point. An indefinite matrix is
usually NONSINGULAR — LAPACK declines it for the pivot's sign, not for rank —
and there the least-squares solution *is* `A⁻¹b`, so the fast path is Gaussian
elimination with partial pivoting. Only a genuinely rank-deficient system (say
`alpha=0` with a linear kernel and `n > d`) falls through to the
symmetric-eigendecomposition pseudo-inverse, which is what produces `lstsq`'s
MINIMUM-NORM answer.

`chi2` and `additive_chi2` require a non-negative `X`, as sklearn's
`check_non_negative` does. That is not defensive: a feature that is zero in both
rows makes the χ² term `0/0`, and the kernel's term guard skips it — with a
negative entry the denominator can pass through zero and the guard would
silently DROP the feature instead of producing a visible infinity.

## `alpha` — the parameter that costs time

`alpha` is a scalar or an array-like of length `n_targets`, and the difference
is not cosmetic:

* **one penalty** (a scalar, or a vector whose entries are all equal) means one
  `(K + αI)` and therefore ONE Cholesky factorization shared across every
  target. Extra targets are nearly free — they are extra right-hand sides.
* **distinct penalties** mean `t` different matrices, so the shared
  factorization is gone and each target pays its own `O(n³)`.

sklearn splits on exactly this test (`(alpha == alpha[0]).all()`) and so does
mlrs. A uniform vector takes the scalar path and produces bit-identical duals,
so writing `alpha=[2.0, 2.0]` instead of `alpha=2.0` costs nothing.

## `sample_weight`

The primal reweighting `Σ wᵢ(yᵢ − f(xᵢ))²` has no `wᵢ` to attach to in the dual,
where the data appear only through `K`. sklearn gets there by a symmetric
similarity transform: with `s = √w`, solve `(SKS + αI)c̃ = S·y` and recover
`c = S·c̃`. mlrs reproduces it verbatim, on the same host pass that injects `α`
— the matrix is already there for that, so weighting costs one multiply per
element and no extra device work.

One consequence is worth stating because the obvious guess is wrong: **a
constant weight is not a no-op.** `α` lands on the diagonal *after* the `S·K·S`
scaling, so a uniform weight `w` divides the effective penalty by `w`.
`sample_weight=np.full(n, 4.0)` with `alpha=1.0` gives exactly `alpha=0.25`
unweighted.

Zero weights drop their samples. All-zero, negative, and non-finite weights are
rejected — sklearn would take the square root of a negative and propagate NaN
through the whole solve.

## `gamma`, `degree`, `coef0`, `kernel_params`

These change what the kernel computes, not how much work it is: `gamma` is one
multiply per element, `coef0` one add, `kernel_params` a dict lookup. `degree`
is the only one with any claim on the clock, and it is evaluated with `powf`
regardless of whether it is an integer — sklearn's interval is `[0, ∞)` over the
REALS, so `degree=0.5` is a legal configuration and is tested as one.

`gamma` is validated in two places for one reason: whether `-1.0` is a legal
coefficient does not depend on the data, so that half is rejected at
construction; whether the value `gamma=None` RESOLVES to is finite cannot be
known until `fit` sees the feature count, so that half stays there.

A `gamma` passed to a kernel that takes none (`linear`, `cosine`,
`additive_chi2`, `precomputed`) is IGNORED, not rejected — sklearn's
`filter_params=True` drops it silently, and being stricter would reject calls
sklearn accepts.

## Measured (ROCm gfx1151, f32, n=700, d=24)

`scripts/bench_kernel_ridge_params.py`, min-of-9 interleaved with 8 s between
reps. Read the caveats under it before quoting any of it.

| kernel | mlrs | sklearn | |
|---|---|---|---|
| linear | 0.034 s | 0.004 s | 0.13x |
| poly | 0.034 s | 0.005 s | 0.13x |
| sigmoid | 0.033 s | 0.005 s | 0.14x |
| cosine | 0.033 s | 0.004 s | 0.13x |
| laplacian | 0.034 s | 0.008 s | 0.24x |
| chi2 | 0.031 s | 0.014 s | 0.44x |
| additive_chi2 | 0.054 s | 0.029 s | 0.55x |
| rbf | 0.171 s | 0.009 s | 0.05x |
| precomputed | 0.034 s | 0.004 s | 0.12x |

The shape is the interesting part, not the ratios. Every GEMM-based family costs
the same 0.033 s, which is the fixed cost of the fit — upload, the host α pass,
the Cholesky — with the kernel evaluation itself invisible underneath it. The
chi² pair is where mlrs closes: its `O(n²·d)` per-element loop is a real cost for
sklearn's Cython and nearly free on a GPU, so `additive_chi2` narrows to 0.55x
from `linear`'s 0.13x. `precomputed` costs the same as `linear`, confirming the
fixed cost reading — removing the kernel evaluation entirely changes nothing.

mlrs LOSES to sklearn at this size, and that is the expected result rather than a
defect: at n=700 the whole problem is a 2 MB Gram and a 5 ms BLAS call, while the
device arm pays an upload, a readback for the α diagonal, and a re-upload. This
estimator's device arm is transfer-bound at small n, the same finding recorded
for Ridge and BayesianRidge.

`degree` is FLAT across 1.0 / 2.0 / 2.5 / 3.0 / 7.0 (0.040–0.042 s), confirming
that a fractional exponent costs no more than an integer one — `powf` does not
care, and neither engine special-cases.

**`alpha`**: the per-target path costs ~1.3–1.7x the scalar path at t = 4/8/16,
which is the extra `t−1` factorizations against a shared kernel matrix. sklearn
shows the same effect far more sharply (12x at t=16), because it has no other
overhead for the extra factorizations to hide behind.

### The rbf row is not comparable to the others

`rbf` is the one family that routes through the `distance()` prim, and that prim
DEGRADES ACROSS REPEATED CALLS in one process. Fitting the same rbf
configuration 25 times in a row climbs monotonically from 0.45 s to 2.80 s while
`poly` in the same loop stays flat at 0.10 s. It is not the machine and not this
estimator: `KernelDensity.score_samples`, which shares the prim and which none of
this work touched, climbs identically (0.095 s → 0.616 s over 13 repeats).

The likely site is `row_reduce(…, ReducePath::Shared)`, which `rbf` calls twice
per fit for the two squared-norm terms and which is already recorded as a
pathology under PyO3. Fixing it belongs with that prim and its other callers, not
here.

Two consequences for reading the table: the 0.171 s rbf figure is a first-call
number and any later rbf measurement in the same process is larger, and the
`alpha` / `sample_weight` sweeps — which use rbf throughout — cannot be read as
absolute times at all. Their INTERNAL comparisons (scalar vs per-target at the
same `t`, adjacent in the run) survive; their cross-sweep magnitudes do not.

### Caveats

This box is shared and was contended for most of this campaign; three unrelated
jobs landed on it mid-session and two earlier runs of this harness produced
self-contradicting numbers because of it. The figures above come from the run
whose cells are internally consistent, and the harness now stamps
`procs_running` at both ends so a contaminated run is visible as one. Treat
these as the shape of the surface, not as benchmark records.

## Where each parameter is honoured

| parameter | layer |
|---|---|
| `kernel` (the nine strings) | Rust — `KernelKind` + `kernel_matrix` |
| `kernel` (a callable) | Python — `pairwise_kernels`, then the Rust `precomputed` path |
| `kernel_params` | Python — only on the callable branch, as in sklearn |
| `alpha` (scalar and per-target) | Rust — the `one_alpha` split in `fit` |
| `gamma` / `degree` / `coef0` | Rust — resolved into the typed `Kernel<F>` at `fit` |
| `sample_weight` | Rust — `KernelRidge::fit_weighted` |

## Tests

| what | where |
|---|---|
| every parameter vs LIVE sklearn | `crates/mlrs-py/python/tests/test_oracle_kernel_ridge_params.py` |
| engine invariants (no sklearn needed) | `crates/mlrs-algos/tests/kernel_ridge_params_test.rs` |
| the committed-fixture oracle | `crates/mlrs-algos/tests/kernel_ridge_test.rs` |
| model-file round-trip | `crates/mlrs-algos/tests/kernel_persist_test.rs` |
| per-parameter timing vs sklearn | `scripts/bench_kernel_ridge_params.py` |

The live-sklearn suite is deliberately not a committed fixture: the two rules
above (`chi2`'s missing default, the indefinite-Gram fallback) are sklearn's
rather than ours, and a fixture would freeze away exactly the thing that has to
keep agreeing as sklearn evolves them.
