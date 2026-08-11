# `StackingRegressor` — stacked generalization

`mlrs.StackingRegressor` implements the full
`sklearn.ensemble.StackingRegressor` parameter surface. Base regressors produce
out-of-fold predictions; those become the columns of a meta-feature matrix; a
final regressor is fitted on it.

```python
import mlrs

reg = mlrs.StackingRegressor(
    estimators=[("lr", mlrs.LinearRegression()), ("ridge", mlrs.Ridge())],
    final_estimator=mlrs.Ridge(),
    cv=5,
)
reg.fit(X, y).predict(X_test)
```

Members may be mlrs estimators, scikit-learn estimators, or a mix — the
composition only requires `fit`/`predict` and `is_regressor`.

## Where the work happens

Stacking is a *composition*: the arithmetic already runs inside the composed
estimators. What the meta-estimator owns is structure, and that is in Rust —
the same split `model_selection` uses.

| what | where |
|---|---|
| estimator-name validation | Rust (`stacking_validate_names`) |
| `'drop'` bookkeeping | Rust (`stacking_kept_indices`) |
| `cv="prefit"` classification | Rust (`stacking_cv_is_prefit`) |
| meta-column layout / `_n_feature_outs` | Rust (`stacking_meta_layout`) |
| `get_feature_names_out` strings | Rust (`stacking_feature_names`) |
| fold index generation | Rust (`mlrs.model_selection`) |
| the meta-matrix hstack | numpy |
| base / final `fit` and `predict` | the composed estimators |

The hstack stays in numpy deliberately. It is one `n x width` copy that
`np.hstack` already does at C speed; routing `k + 1` host blocks back through
the Arrow capsule boundary to do the identical copy would add FFI round-trips
and an f64-only constraint for zero arithmetic. Rust still owns the *decision*
and states it executably in
`mlrs_algos::ensemble::stacking::concatenate_predictions`, gated by
`crates/mlrs-algos/tests/stacking_test.rs`.

## Parameters

| parameter | default | notes |
|---|---|---|
| `estimators` | — | `list of (str, estimator)`; an entry may be the string `'drop'` |
| `final_estimator` | `None` | `None` means `sklearn.linear_model.RidgeCV()` — see below |
| `cv` | `None` | int / splitter / iterable of index pairs / `"prefit"`; `None` is 5-fold `KFold` |
| `n_jobs` | `None` | joblib fan-out; see the device caveat below |
| `passthrough` | `False` | append the original `X` columns to the meta features |
| `verbose` | `0` | forwarded to the inner `cross_val_predict` calls |

### The default `final_estimator` is sklearn's `RidgeCV`

sklearn's default is `RidgeCV()`, which selects `alpha` from
`(0.1, 1.0, 10.0)` by leave-one-out generalized cross-validation. mlrs ships
`Ridge`, not `RidgeCV`, and substituting `Ridge(alpha=1.0)` would silently
change every default-constructed stack's predictions relative to the sklearn
baseline users migrate from. So the default is `sklearn.linear_model.RidgeCV()`,
constructed lazily inside `fit`; sklearn is already a hard runtime dependency of
the package. Pass `final_estimator=mlrs.Ridge()` to put the meta-fit on the
device.

### `n_jobs` is ignored when a member holds a device handle

A fitted mlrs estimator owns a compiled `#[pyclass]` wrapping device state.
Neither joblib fan-out is worth taking over one:

* **process backends** (`loky` — joblib's default — `multiprocessing`, `dask`)
  return each worker's result by pickling it, so `n_jobs=2` raises
  `TypeError: cannot pickle 'builtins.Ridge' object`. Unconditional;
* **the threading backend** works, and barely helps. Every device call holds the
  process-global `Mutex<BufferPool>`, so the fan-out cannot overlap the work it
  fans out: six members at `cv=20` on rocm went 1.584 s serial → 1.343 s at
  `n_jobs=4` (1.18x), bit-identical. Picking a backend on the caller's behalf
  for ~18% is a bad trade; finer-grained locking, not a scheduler switch, is
  what would make this parameter pay.

So `mlrs.StackingRegressor` emits a `UserWarning` and fits serially in that
case. `n_jobs` works normally over host (scikit-learn) members.

Historically the threading route did not merely underperform — it aborted the
process, because CubeCL allocates one stream (and one memory arena) per OS
thread. That is fixed by the stream cap (`mlrs_backend::stream_cap`); see
[stream-cap.md](stream-cap.md). The reason `n_jobs` is still reduced is the
mutex, not the crash.

## Parity

`crates/mlrs-py/python/tests/test_oracle_stacking.py` compares against a live
`sklearn.ensemble.StackingRegressor` — 79 cells, run green on **cpu, wgpu and
rocm** with zero skips. Compositions of sklearn members match **exactly**
(`atol=0`), including every rejection message. Compositions of mlrs members
match within `conftest.live_atol()` (1e-5 at f64, 1e-3 on an f32-only backend,
since sklearn always answers in f64).

The two string-valued parameters get dedicated coverage, semantics included —
not just "mlrs == sklearn", which would pass even if both silently ignored the
string:

* `cv="prefit"` — members are reused rather than cloned (`estimators_[i] is
  estimators[i][1]`), never refitted, and the meta features are full-training-set
  predictions. Note the difference is observable in `final_estimator_`, **not**
  in `transform`: `transform` always re-predicts through `estimators_`, which
  under an int `cv` were refitted on the full `X` anyway.
* `estimators=[(name, "drop")]` — no fit, no meta column, no feature name, but
  the slot survives in `named_estimators_` as the literal `'drop'`.

`mlrs.StackingRegressor` also passes sklearn's `parametrize_with_checks` sweep
(57 passed / 1 skipped) at the default `passthrough=False`. At
`passthrough=True` sklearn's **own** `StackingRegressor` fails
`check_{regressor,transformer}_data_not_an_array`, so mlrs is not gated there.

## Measured performance

`scripts/bench_stacking.py`, `n=100000`, `d=64`, two members, min of N fresh
subprocesses. The one-time `_mlrs` extension load (~35–95 ms/process) is
reported separately, not folded into the cells — charging it to the first cell
once made mlrs look 6x slower than it is.

**Host arm** (sklearn members on both sides — isolates the orchestration layer),
min of 5:

| config | sklearn fit | mlrs fit | ratio |
|---|---|---|---|
| `cv=2` | 0.868 s | 0.947 s | 0.92x |
| `cv=3` | 1.227 s | 1.139 s | 1.08x |
| `cv=5` | 1.568 s | 1.705 s | 0.92x |
| `cv=10` | 3.068 s | 3.110 s | 0.99x |
| `cv="prefit"` | 0.017 s | 0.017 s | 0.98x |
| `passthrough=True` (cv=5) | 1.595 s | 1.725 s | 0.92x |
| `n_jobs=2` (cv=5) | 1.172 s | 1.186 s | 0.99x |
| `n_jobs=4` (cv=5) | 0.957 s | 1.065 s | 0.90x |

**Device arm** (mlrs members on rocm gfx1151 vs sklearn members on host),
min of 3:

| config | sklearn fit | mlrs fit | ratio |
|---|---|---|---|
| `cv=2` | 0.713 s | 1.399 s | 0.51x |
| `cv=3` | 1.013 s | 1.455 s | 0.70x |
| `cv=5` | 1.591 s | 1.575 s | 1.01x |
| `cv=10` | 3.168 s | 1.850 s | **1.71x** |
| `cv="prefit"` | 0.046 s | 0.125 s | 0.37x |
| `passthrough=True` (cv=5) | 1.788 s | 1.748 s | 1.02x |
| `n_jobs=2` (cv=5) | 1.219 s | 1.577 s | 0.77x |
| `n_jobs=4` (cv=5) | 1.025 s | 1.596 s | 0.64x |

Reading the ladders:

* **`cv` is the cost driver, and it is linear in the fold count** — on the host
  arm 0.87 → 1.23 → 1.57 → 3.07 s for `k = 2, 3, 5, 10`, exactly the `k + 1`
  base fits the design predicts. `cv="prefit"` costs 0.017 s: ~90x cheaper than
  `cv=5`, because it performs no base fits at all. If a stack is too slow, `cv`
  is the parameter to look at first.
* **The device arm crosses over with `k`.** mlrs's per-fit device overhead is
  fixed, so it loses at `cv=2` (0.51x) and wins at `cv=10` (1.71x): more folds
  amortize the same setup over more arithmetic. `cv=5` is the break-even point
  at this size.
* **`passthrough` is nearly free at fit time** (+3–4% on both implementations)
  but **~3.5x on predict** on the host arm (0.009 → 0.030 s) — that is the extra
  `n x d` copy, and it is the same on both sides.
* **`n_jobs` scales on the host arm** (1.61 → 1.17 → 0.96 s, ~1.7x at 4 jobs)
  and is deliberately **flat on the device arm** (1.63 → 1.58 → 1.60 s), which
  is the serial fallback documented above doing its job rather than a
  regression.
* **The orchestration layer itself is at parity**: on the host arm, where every
  base fit is identical, mlrs is 0.90–1.08x of sklearn across the whole sweep.
