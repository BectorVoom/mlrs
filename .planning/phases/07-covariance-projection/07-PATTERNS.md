# Phase 7: Covariance & Projection - Pattern Map

**Mapped:** 2026-06-14
**Files analyzed:** 16 new/modified files
**Analogs found:** 16 / 16 (every new file has a verified v1 analog read this session)

> All analog file paths below were opened and verified to exist this session.
> Line numbers are exact against the current v1 source. Phase 7 is an *assembly*
> phase: no new device kernel — every new file is host-side glue over a validated
> v1 prim or a 1:1 copy of an existing estimator/test/oracle skeleton.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/mlrs-backend/src/prims/rng.rs` | prim (host) | transform (generate) | `crates/mlrs-backend/src/prims/kmeans.rs` (SplitMix64 L658-710) | exact (promotion) |
| `crates/mlrs-backend/src/prims/incremental_svd.rs` | prim (host glue) | transform (merge over svd) | `crates/mlrs-backend/src/prims/svd.rs` + `decomposition/pca.rs` fit | role+flow match |
| `crates/mlrs-backend/src/prims/mod.rs` (modify) | config/index | n/a | self (L12-27) | exact |
| `crates/mlrs-backend/src/prims/kmeans.rs` (modify) | prim (refactor) | n/a | self (L389,398,437 callers; L658-710 struct) | exact |
| `crates/mlrs-algos/src/traits.rs` (modify) | trait surface | n/a | `Fit`/`Transform` in same file (L53-120) | exact |
| `crates/mlrs-algos/src/error.rs` (modify) | error type | n/a | `AlgoError` struct-variants (same file L41-158) | exact |
| `crates/mlrs-algos/src/lib.rs` (modify) | config/index | n/a | self (L32-43) | exact |
| `crates/mlrs-algos/src/covariance/mod.rs` | config/index | n/a | `decomposition/mod.rs` | exact |
| `crates/mlrs-algos/src/covariance/empirical_covariance.rs` | estimator | CRUD/fit (covariance + eig pinvh) | `crates/mlrs-algos/src/decomposition/pca.rs` | role match |
| `crates/mlrs-algos/src/covariance/ledoit_wolf.rs` | estimator | CRUD/fit (covariance + host shrink) | `crates/mlrs-algos/src/decomposition/pca.rs` | role match |
| `crates/mlrs-algos/src/projection/mod.rs` | config/index | n/a | `decomposition/mod.rs` | exact |
| `crates/mlrs-algos/src/projection/gaussian.rs` | estimator | transform (rng + gemm) | `crates/mlrs-algos/src/decomposition/pca.rs` (Transform) | role match |
| `crates/mlrs-algos/src/projection/sparse.rs` | estimator | transform (rng + gemm) | `crates/mlrs-algos/src/decomposition/pca.rs` (Transform) | role match |
| `crates/mlrs-algos/src/decomposition/incremental_pca.rs` | estimator | event-driven (partial_fit stream) | `crates/mlrs-algos/src/decomposition/pca.rs` | exact (mirror) |
| `crates/mlrs-backend/tests/rng_test.rs` | test | property/pool | `crates/mlrs-backend/tests/memory_gate_test.rs` (pool) + `gemm_test.rs` (f64 gate) | role match |
| `crates/mlrs-backend/tests/incremental_svd_test.rs` | test | oracle/multi-batch | `crates/mlrs-backend/tests/svd_test.rs` + `gemm_test.rs` (f64 gate) | role match |
| `crates/mlrs-algos/tests/empirical_covariance_test.rs` | test | oracle (1e-5) | `crates/mlrs-algos/tests/pca_test.rs` | exact |
| `crates/mlrs-algos/tests/ledoit_wolf_test.rs` | test | oracle (1e-5) | `crates/mlrs-algos/tests/pca_test.rs` | exact |
| `crates/mlrs-algos/tests/incremental_pca_test.rs` | test | oracle (1e-5, post align_rows) | `crates/mlrs-algos/tests/pca_test.rs` | exact |
| `crates/mlrs-algos/tests/random_projection_test.rs` | test | property gate + 1 value oracle | `crates/mlrs-algos/tests/pca_test.rs` (structure) | partial |
| `crates/mlrs-py/src/estimators/covariance.rs` + `projection.rs` (or extend `decomposition.rs`) | binding | request-response | `crates/mlrs-py/src/estimators/decomposition.rs` | exact |
| `scripts/gen_oracle.py` (modify) | oracle generator | batch | `gen_pca` (L985-1041) + `main()` (L1088-1163) | exact |

---

## Pattern Assignments

### `crates/mlrs-backend/src/prims/rng.rs` (prim, host generate)

**Analog:** `crates/mlrs-backend/src/prims/kmeans.rs` — promote the `SplitMix64`
struct **verbatim** (RESEARCH Pitfall 7: a verbatim move; do NOT "improve" the
mix or the stream changes and `kmeanspp_test.rs` breaks).

**SplitMix64 to copy verbatim** (`kmeans.rs` L658-710):
```rust
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// UNBIASED uniform integer in [0, bound) via rejection sampling.
    fn next_below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound >= 1, "next_below requires a positive bound");
        if bound == 1 { return 0; }
        let zone = u64::MAX - (u64::MAX % bound);
        loop {
            let v = self.next_u64();
            if v < zone { return v % bound; }
        }
    }
}
```
Make the struct + its methods `pub` on the move (currently private). Use
`next_below` (NOT `next_u64() % n`) for Fisher-Yates — RESEARCH Anti-Pattern
"biased modulo".

**New host-side additions** (Box-Muller Gaussian, Achlioptas sparse,
Fisher-Yates) build on `next_f64` — see RESEARCH Pattern 4 for the exact scaling
(`N(0,1/n_components)` Gaussian; Achlioptas `v = sqrt((1/density)/n_components)`).

**Geometry/hyperparameter guard pattern** — copy the `guard_u32` rejection idiom
(`kmeans.rs` L631-643) and the `validate_geometry` → typed `PrimError` style
(`covariance.rs` L212-262). Validate `density ∈ (0,1]`, `n_components ≥ 1` BEFORE
any allocation (ASVS V5).

**Float bit-cast helpers** — copy `host_to_f64` / `f64_to_host` verbatim
(`kmeans.rs` L717-732; identical block in `svd.rs` L446-461, `pca.rs` L389-404).

**PoolStats memory gate** — host-generate-then-single-upload; one `from_host`
upload of the matrix. See the gate analog under `rng_test.rs` below.

---

### `crates/mlrs-backend/src/prims/incremental_svd.rs` (prim, host merge over svd)

**Analog:** `crates/mlrs-backend/src/prims/svd.rs` (the merge re-runs this) +
`crates/mlrs-algos/src/decomposition/pca.rs` fit (the host-center-then-upload
idiom the merge generalizes).

**The decisive merge math** is RESEARCH Pattern 1 / Code Example "Stacked-matrix
merge". The implementation is host glue:

**SVD call to compose** (`svd.rs` L90-106 signature):
```rust
pub fn svd<F>(
    pool: &mut BufferPool<ActiveRuntime>,
    a: &DeviceArray<ActiveRuntime, F>,
    (rows, cols): (usize, usize),
) -> Result<(DeviceArray<..,F>, DeviceArray<..,F>, DeviceArray<..,F>), PrimError>
```
Returns `(U, S, Vᵀ)` with **S descending** (`svd.rs` L308-323 sort) and `Vᵀ`
row-major (`svd.rs` L298-305). The stacked matrix MUST satisfy
`max(rows,cols) ≤ MAX_ROWS`, `min(rows,cols) ≤ MAX_COLS` — `svd.rs` validates
this at L430-441 and returns `ShapeMismatch`; size fixtures so
`k + batch_size + 1 ≤ MAX_ROWS` and `n_features ≤ MAX_COLS` (RESEARCH A2/Open Q3 —
read `MAX_ROWS`/`MAX_COLS` from `mlrs-kernels`, imported at `svd.rs` L50).

**Host-center-then-upload pattern to mirror** (`pca.rs` L197-208):
```rust
let x_host = x.to_host(pool);
let mut x_centered: Vec<F> = vec![F::from_int(0i64); n_samples * n_features];
for r in 0..n_samples {
    for c in 0..n_features {
        let v = host_to_f64(x_host[r * n_features + c]) - mean64[c];
        x_centered[r * n_features + c] = f64_to_host::<F>(v);
    }
}
let x_c_dev: DeviceArray<ActiveRuntime, F> = DeviceArray::from_host(pool, &x_centered);
let (u, s, vt) = svd::<F>(pool, &x_c_dev, (n_samples, n_features))?;
```
For the merge: build the *stacked* host buffer (rows `0..k` = `singular_values_[i]
* components_[i,:]`; rows `k..k+b` = centered batch; row `k+b` = mean-correction),
upload once via `DeviceArray::from_host`, call `svd`, then apply `align_rows` on
the `Vᵀ` rows (see sign-flip block below). First batch → SVD of centered batch
alone (RESEARCH Pitfall 3). Accumulate the combine math in f64 (`host_to_f64`/
`f64_to_host`), exactly as `pca.rs` does.

**Sign-flip after every batch** (`pca.rs` L226-235, the `u_based_decision=False`
contract):
```rust
let vt_rows: Vec<Vec<f64>> = (0..k)
    .map(|j| (0..n_features).map(|c| host_to_f64(vt_host[j * n_features + c])).collect())
    .collect();
let vt_flipped = align_rows(&vt_rows);   // == sklearn svd_flip(u_based_decision=False)
```

**Scratch release** — release the SVD outputs you don't keep, mirroring `pca.rs`
L259-262:
```rust
u.release_into(pool);
s.release_into(pool);
vt.release_into(pool);
x_c_dev.release_into(pool);
```

**PoolStats memory gate** — second gate (one per new prim, D-10 precedent). Keep
fixtures tiny (RESEARCH: this test is SVD-heavy; the cpu suite is already ~6 min).

---

### `crates/mlrs-algos/src/traits.rs` — add `PartialFit<F>` (modify)

**Analog:** the `Fit` trait in the SAME file (L53-68). Copy its shape exactly —
same bound, same `pool`/`DeviceArray`/explicit-`(rows,cols)` convention, returns
`&mut Self`, errors as `AlgoError`.

**`Fit` trait to mirror** (`traits.rs` L53-68):
```rust
pub trait Fit<F>
where
    F: Float + CubeElement + Pod,
{
    fn fit(
        &mut self,
        pool: &mut BufferPool<ActiveRuntime>,
        x: &DeviceArray<ActiveRuntime, F>,
        y: Option<&DeviceArray<ActiveRuntime, F>>,
        shape: (usize, usize),
    ) -> Result<&mut Self, AlgoError>;
}
```
`PartialFit<F>` is the same with a `partial_fit` method (no `y` for IncrementalPCA;
keep the `Option<&DeviceArray>` slot for the Phase-10 MBSGD reuse per D-01).
Re-export it in `lib.rs` L43 alongside the others.

---

### `crates/mlrs-algos/src/error.rs` — new hyperparameter guards (modify)

**Analog:** the existing struct-variants in the SAME file (L41-158). Copy the
`#[error("...")]` + named-field shape. Add variants for `density`/`batch_size`/
`eps` (JL) per RESEARCH §Security Domain V5.

**Variant style to copy** (`error.rs` L54-60, the simplest scalar guard):
```rust
#[error("estimator '{estimator}': alpha = {alpha} is invalid (must be >= 0)")]
InvalidAlpha {
    estimator: &'static str,
    alpha: f64,
},
```
Reuse `InvalidNComponents` (L41-48), `NotFitted` (L68-73), `Unsupported`
(L82-87), and `Prim(#[from] PrimError)` (L182) AS-IS — no new variant needed for
those. The `#[from] PrimError` wrap lets estimator methods `?` a prim call
directly.

---

### `crates/mlrs-algos/src/covariance/empirical_covariance.rs` (estimator, fit)

**Analog:** `crates/mlrs-algos/src/decomposition/pca.rs` (the canonical estimator
skeleton: struct with `Option<DeviceArray>` fitted slots, host accessors via
`attr`, `Fit` impl with validate-before-launch, device-resident state).

**Struct + accessor skeleton to copy** (`pca.rs` L54-137): the `Option<
DeviceArray<ActiveRuntime, F>>` slots, the per-attr `pub fn attr_name(&self, pool)
-> Result<Vec<F>, AlgoError>`, and the shared `attr` helper:
```rust
fn attr(
    &self,
    slot: &Option<DeviceArray<ActiveRuntime, F>>,
    pool: &BufferPool<ActiveRuntime>,
    operation: &'static str,
) -> Result<Vec<F>, AlgoError> {
    slot.as_ref()
        .map(|a| a.to_host(pool))
        .ok_or(AlgoError::NotFitted { estimator: "empirical_covariance", operation })
}
```

**Validate-before-launch guard** (`pca.rs` L150-179) — copy the geometry/
hyperparameter rejection idiom (reject `n_features == 0`, `x.len() !=
n_samples*n_features` as `AlgoError::Prim(PrimError::ShapeMismatch{..})`).

**`covariance_` via the v1 prim with `ddof=0`** (RESEARCH Code Example
"EmpiricalCovariance fit"; `covariance.rs` L89-95 signature):
```rust
let cov = covariance::<F>(pool, &x_centered, (n, p), /*ddof=*/0, None)?;  // == np.cov(bias=1)
```
**CRITICAL (RESEARCH Pitfall 1 / Anti-Pattern):** use `ddof=0`, NOT `1`. The
covariance prim folds ddof into the scale (`covariance.rs` L182-204).

**`location_`** — `column_reduce(.., ScalarOp::Mean, ReducePath::Shared)` then
`.expect("shared path is never plane-gated to None")` (`pca.rs` L183-191). When
`assume_centered` (D-07): set `location_ = 0` and skip the mean subtraction.

**`precision_` = pinvh via eig (D-05)** — RESEARCH Pattern 2; `eig.rs` L75-89
signature returns `(w descending, V column-major)`:
```rust
let (w, v) = eig::<F>(pool, &cov, p, None)?;  // descending w, V columns (col-major)
// cutoff = rcond * max|w|; inv_w = (|w|>cutoff) ? 1/w : 0; precision_ = V·diag(inv_w)·Vᵀ
```
**V is column-major** (`eig.rs` doc L60-61): `v[c*n+r] = V[r,c]`. Reassemble
`precision_` on the host (p is small — the Gram is p×p), matching the small-n
host-finalize idiom. Reuse the v1 04-03 σ⁺ RCOND constant for the floor
(RESEARCH A1; pin with a rank-deficient `n ≤ p` oracle fixture). Do NOT use
Cholesky (SPD-only, fails on singular — D-05).

**f64 host-combine helpers** — copy `host_to_f64`/`f64_to_host` from `pca.rs`
L389-404.

---

### `crates/mlrs-algos/src/covariance/ledoit_wolf.rs` (estimator, fit)

**Analog:** `pca.rs` (same estimator skeleton as EmpiricalCovariance above).

**Math:** RESEARCH Pattern 3 — exact `ledoit_wolf_shrinkage` β/δ/μ closed form.
`emp_cov` reuses `covariance::<F>(.., ddof=0, ..)` (`covariance.rs` L89). The β/δ
scalar reductions over `X²` and the Gram are a host finalize in f64 (mirror the
kmeans inertia host-sum idiom; use `host_to_f64`). `shrinkage_` ∈ [0,1] by
construction — still apply the `min/max` clip per COV-02. Shares the
`Option<DeviceArray>` + `attr` accessor skeleton (`pca.rs` L54-137).

---

### `crates/mlrs-algos/src/projection/gaussian.rs` + `sparse.rs` (estimator, transform)

**Analog:** `pca.rs` `Transform` impl (L274-331) — `transform == X · componentsᵀ`
is the SAME single GEMM.

**transform GEMM to copy** (`pca.rs` L319-328):
```rust
let z = gemm::<F>(
    pool,
    &x_dev, (n_samples, n_features),
    components, (n_features, nc),
    false,
    true, // components_ buffer is (nc × n_features); transb reads it as componentsᵀ.
    None,
)?;
```
RandomProjection does NOT center (no `mean_` subtraction — drop the `pca.rs`
centering loop). `components_` come from `prims::rng` (Gaussian `N(0,1/
n_components)` / Achlioptas), stored **dense** even for sparse (D-12). Sparse
input densified at the Python ingress (D-12). `n_components='auto'` →
`johnson_lindenstrauss_min_dim` (RESEARCH Pattern 5 / Code Example `jl_min_dim`).

**Property gate (NOT 1e-5)** — D-12: structural properties only (JL bound,
moment stats, seed-repro, `transform == X·componentsᵀ`). Only
`johnson_lindenstrauss_min_dim` is value-matched.

---

### `crates/mlrs-algos/src/decomposition/incremental_pca.rs` (estimator, partial_fit stream)

**Analog:** `crates/mlrs-algos/src/decomposition/pca.rs` — mirror its struct,
accessors, `Transform`/`inverse_transform`, and host-combine helpers EXACTLY;
add the running streaming state (`n_samples_seen_`, `var_`) and the `PartialFit`
impl.

**`fit()` is sklearn-faithful** (D-02): reset state, then loop `partial_fit` over
`gen_batches(n_samples, batch_size)`; `batch_size=None → 5·n_features` (D-03).
Each `partial_fit` calls `incremental_svd::merge` (PRIM-07). `whiten` (D-06)
scales components by `1/sqrt(explained_variance_)` in `transform`, un-whitens in
`inverse_transform`.

**`explained_variance_` uses ddof=1** here (`S²/(n_total−1)`) — DISTINCT from the
covariance estimators' ddof=0 (RESEARCH Pitfall 1). The
`explained_variance_ratio_` denominator is `sum(col_var)·n_total`, NOT the
truncated `S²` sum (RESEARCH Pitfall 6 — different from `pca.rs` L213-224 which
uses the full-spectrum sum). Reuse `pca.rs`'s `inverse_transform` reconstruction
GEMM (L360-369) and mean-broadcast (L371-381) directly.

**Reuse `pca.rs` Transform/inverse_transform verbatim** (L274-385) — the
projection math is identical once components/mean are fitted; only the *fit* path
differs (streaming merge vs single SVD).

---

### Tests — analogs

**`crates/mlrs-algos/tests/{empirical_covariance,ledoit_wolf,incremental_pca}_test.rs`**
**Analog:** `crates/mlrs-algos/tests/pca_test.rs` — copy wholesale.

Reusable scaffolding from `pca_test.rs`:
- `fixture(name)` workspace-root resolver (L45-52).
- `host_to_f64`/`f64_to` bit-casts (L54-68).
- `assert_close` numpy-allclose `|got−exp| ≤ atol + rtol·|exp|` (L73-91).
- `align_matrix_rows` / `align_matrix_cols` for sign-canonicalization (L95-123) —
  IncrementalPCA compares `components_` only AFTER `align_rows` (DECOMP-03).
- The fit-and-promote harness (`fit_pca`, L139-185).
- The dual f32/f64 test pair with the **f64 capability gate** (L192-233):
```rust
if capability::skip_f64_with_log() {
    println!("... f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
    return;
}
```
- `load_npz` / `OracleCase` / `Tolerance` / `F32_TOL` / `F64_TOL` imports
  (L32-33). f64 stays strict `F64_TOL` (1e-5); f32-on-rocm gets a documented
  per-family band (Claude's discretion — set from the standalone prim measurement).

EmpiricalCovariance test adds a **rank-deficient (`n ≤ p`)** case to exercise the
pinvh floor.

**`crates/mlrs-backend/tests/rng_test.rs`**
**Analog:** `memory_gate_test.rs` (the PoolStats gate idiom — `live_bytes`/
`peak_bytes`/`reuses` assertions, L88-128 commentary) + `gemm_test.rs` f64 gate
(L181-184). PRIM-06 distribution + seed-reproducibility (same seed → identical
matrix) + Achlioptas density/value stats + Fisher-Yates bijection, plus
`rng_memory_gate`.

**`crates/mlrs-backend/tests/incremental_svd_test.rs`**
**Analog:** `svd_test.rs` (oracle structure) + `gemm_test.rs` f64 gate. PRIM-07
2+-batch merge vs a host reference; ddof=1; `align_rows` applied; f64 1e-5 /
f32 band; plus `incremental_svd_memory_gate`. Keep fixtures TINY (SVD-heavy).

**`crates/mlrs-algos/tests/random_projection_test.rs`**
**Analog:** `pca_test.rs` (structure) but the gate is a PROPERTY set, not 1e-5
(D-12). Fix the SplitMix64 seed (bit-reproducible per backend) AND average the
distortion/moment statistic over many trials (D-11; planner pins trial count,
e.g. 30-50). One value-oracle blob for `johnson_lindenstrauss_min_dim` only.

---

### `crates/mlrs-py/src/estimators/{covariance,projection}.rs` (or extend `decomposition.rs`)

**Analog:** `crates/mlrs-py/src/estimators/decomposition.rs` (`PyPCA`) — the
canonical per-estimator wrapper. v2 adds ZERO binding infra (RESEARCH).

**`any_estimator!` invocation to copy** (`decomposition.rs` L23-27):
```rust
crate::any_estimator! {
    any:   AnyPca,
    algo:  mlrs_algos::decomposition::pca::Pca,
    unfit: { n_components: usize },
}
```
(macro defined in `crates/mlrs-py/src/dispatch.rs` L84-108.)

**`fit` body to copy** (`decomposition.rs` L60-87): the `py.detach(|| { ... })`
GIL release, `global_pool().lock()`, `float_dtype` dispatch, and — load-bearing —
`crate::capability::guard_f64()?` BEFORE the F64 arm (`decomposition.rs` L77;
guard defined `mlrs-py/src/capability.rs` L37):
```rust
let fitted = py.detach(|| -> PyResult<AnyPca> {
    let mut pool = crate::global_pool().lock().expect("pool mutex");
    match dt {
        FloatDtype::F32 => { let xd = validated_f32(as_f32(&xa)?, &mut pool)?; /* ::<f32>::new(..).fit(..) */ Ok(AnyEst::F32(est)) }
        FloatDtype::F64 => { crate::capability::guard_f64()?; let xd = validated_f64(as_f64(&xa)?, &mut pool)?; /* ::<f64>::new(..).fit(..) */ Ok(AnyEst::F64(est)) }
    }
})?;
```
**dtype-suffixed accessors** (`decomposition.rs` L90-115): `transform_f32`/
`transform_f64` matching on the fitted arm, returning `.to_host_metered(&mut
pool)`. IncrementalPCA additionally needs a `partial_fit` method (same `py.detach`
+ dispatch shape; mutate the fitted arm in place across calls). Errors mapped via
`algo_err_to_py` / `not_fitted` (`decomposition.rs` L16).

Register new `#[pyclass]`es in `crates/mlrs-py/src/estimators/mod.rs`.

---

### `scripts/gen_oracle.py` — 4 new generators + main() wiring (modify)

**Analog:** `gen_pca` (L985-1041) + the `main()` dual-dtype loop (L1088-1163).

**Generator skeleton to copy** (`gen_pca` L1000-1041):
```python
rng = np.random.default_rng(seed)
x = rng.standard_normal(shape)
est = SkEstimator(...).fit(x)
def c(arr):
    # Force C-contiguous so the flat blob matches the row-major Rust contract.
    return np.ascontiguousarray(np.asarray(arr)).astype(dtype)
dtype_tag = {np.float32: "f32", np.float64: "f64"}[dtype]
os.makedirs(_FIXTURE_DIR, exist_ok=True)
np.savez(out_path, X=c(x), ...attrs...)
return out_path
```
**CRITICAL:** the `np.ascontiguousarray` C-contiguous fix (L1007-1015 comment) —
sklearn `components_` is Fortran-order; without forcing C-contiguous the npz
stores the column-major ravel and silently transposes. Apply to `components_` of
IncrementalPCA.

**main() wiring to copy** (L1133-1136, the dual-dtype loop):
```python
for dtype in (np.float32, np.float64):
    print(f"wrote {gen_empirical_covariance(dtype=dtype)}")
```
Add: `gen_empirical_covariance` (store `X`, `covariance_`, `location_`,
`precision_`; add a rank-deficient `n≤p` case — VALUE 1e-5), `gen_ledoit_wolf`
(`X`, `covariance_`, `shrinkage_`; two `n` — VALUE 1e-5), `gen_incremental_pca`
(all attrs + `transform`/`inverse_transform` + `n_samples_seen_`, C-contiguous
`components_`, whiten on/off — VALUE 1e-5 after align_rows), `gen_jl_min_dim`
(value grid — VALUE 1e-5). NO matrix/transform oracle for RandomProjection
(D-12). Regen needs a `/tmp` venv with numpy+scipy+sklearn (PEP 668; project
memory "oracle fixture regen needs venv"); commit the `.npz` blobs.

---

## Shared Patterns

### f64 capability gate (every f64 oracle case)
**Source:** `crates/mlrs-backend/tests/gemm_test.rs` L181-184 (also `pca_test.rs`
L216-219). **Apply to:** every f64 test case in all new test files.
```rust
let backend = capability::active_backend_name();
capability::log_oracle_dtype(capability::FloatKind::F64, backend, "default");
if capability::skip_f64_with_log() {
    println!("... f64 backend={backend}: SKIPPED (no f64 support on this adapter)");
    return;
}
```
cpu runs f64; rocm skips-with-log (project memory: rocm f64 UNSUPPORTED).

### svd_flip sign canonicalization
**Source:** `crates/mlrs-core/src/sign_flip.rs` L60-62 (`align_rows`). **Apply
to:** IncrementalPCA `components_` (after every batch merge AND before oracle
compare), the PCA-style usage in `incremental_svd.rs`, and all PCA-family test
comparisons. It IS sklearn `svd_flip(u_based_decision=False)` — largest-|element|
per row made positive (verified `sign_flip.rs` L21-43 vs sklearn).
```rust
pub fn align_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter().map(|r| align_sign(r)).collect()
}
```

### Validate-hyperparameter-before-launch (ASVS V5)
**Source:** `covariance.rs` `validate_geometry` L212-262 (prim level);
`pca.rs` L150-179 (estimator level); `kmeans.rs` `guard_u32` L631-643.
**Apply to:** every new prim and estimator `fit`/`merge`. Reject as a typed
`PrimError::ShapeMismatch`/`DimMismatch` or `AlgoError` BEFORE any device launch —
never an OOB device read. New guards: `density ∈ (0,1]`, `eps ∈ (0,1)`,
`batch_size ≥ 1`, `n_components` range, stacked `(k+b+1) ≤ MAX_ROWS` /
`n_features ≤ MAX_COLS`.

### Host f64-combine bit-cast helpers
**Source:** identical `host_to_f64`/`f64_to_host` blocks in `pca.rs` L389-404,
`svd.rs` L446-461, `kmeans.rs` L717-732. **Apply to:** every new prim/estimator
that does host-side combine math. Accumulate combine math in f64 regardless of
`F`, then cast back (the RESEARCH Pitfall 4 stability convention).

### Device-resident fitted state + scratch release
**Source:** `pca.rs` L258-269 (store `Some(DeviceArray)`), L259-262
(`release_into(pool)` scratch); `covariance.rs` L149,180 (release transient
scratch at true byte size). **Apply to:** all new estimators (D-03 device
residency) and both new prims (PoolStats memory gates depend on the releases).

### PyO3 GIL-release + dtype dispatch + f64 guard
**Source:** `crates/mlrs-py/src/estimators/decomposition.rs` L60-115 (`PyPCA`);
macro at `dispatch.rs` L84-108; `guard_f64` at `mlrs-py/src/capability.rs` L37.
**Apply to:** every new PyO3 wrapper. `py.detach` closure, `global_pool().lock()`,
`float_dtype` match, `guard_f64()?` before the F64 arm, dtype-suffixed accessors.

### Module-index registration (file-disjoint)
**Source:** `decomposition/mod.rs` (`pub mod pca;`); `prims/mod.rs` L12-27;
`lib.rs` L32-43. **Apply to:** new `covariance/mod.rs` + `projection/mod.rs`
(copy `decomposition/mod.rs` shape), register `pub mod covariance;` /
`pub mod projection;` in `lib.rs`, `pub mod rng;` / `pub mod incremental_svd;` in
`prims/mod.rs`, `pub mod incremental_pca;` in `decomposition/mod.rs`.

---

## No Analog Found

None. Every Phase-7 file maps to a verified v1 analog. The two genuinely-new
behaviors (the Box-Muller/Achlioptas RNG generators and the stacked-matrix SVD
merge) are *host-side compositions* over existing prims with no structural analog,
but their surrounding skeleton (validate → host-combine-in-f64 → upload → prim →
release) is the `pca.rs`/`covariance.rs`/`svd.rs` idiom; the exact math is pinned
in RESEARCH Patterns 1, 3, 4. RandomProjection's test is "partial" only because
its gate is a property set (D-12), not the `pca_test.rs` 1e-5 oracle — the test
*structure* (fixture loader, backend logging, dtype split) still copies
`pca_test.rs`.

---

## Metadata

**Analog search scope:** `crates/mlrs-backend/src/prims/`,
`crates/mlrs-algos/src/`, `crates/mlrs-core/src/`, `crates/mlrs-py/src/`,
`crates/mlrs-backend/tests/`, `crates/mlrs-algos/tests/`, `scripts/gen_oracle.py`.
**Files read this session:** `traits.rs`, `decomposition/pca.rs`,
`prims/covariance.rs`, `prims/svd.rs`, `prims/eig.rs`, `prims/kmeans.rs`
(SplitMix64), `prims/mod.rs`, `core/sign_flip.rs`, `algos/error.rs`,
`algos/lib.rs`, `py/dispatch.rs`, `py/estimators/decomposition.rs`,
`tests/pca_test.rs`, `tests/gemm_test.rs`, `tests/memory_gate_test.rs`,
`scripts/gen_oracle.py` (`gen_pca` + `main`).
**Pattern extraction date:** 2026-06-14
