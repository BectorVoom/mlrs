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
| the meta-matrix copy | numpy by default; Rust host / CubeCL device on request |
| base / final `fit` and `predict` | the composed estimators |

## The meta-matrix arms (STACK-META-01)

The meta matrix — `k` prediction blocks side by side, plus `X` under
`passthrough` — can be assembled three ways, selected by the
`MLRS_STACK_META_ENGINE` A/B knob:

| value | arm |
|---|---|
| unset (**default**) | `np.hstack` in the shim |
| `numpy` | the same, forced |
| `host` | `mlrs_algos::ensemble::stacking::concatenate_predictions`, reached through the Arrow capsule |
| `device` | `mlrs_kernels::stacking::stack_meta_block`, one CubeCL launch per block |

All three produce **byte-identical** matrices — the operation has no arithmetic,
so `test_stacking_meta_engine.py` asserts exact equality, not a tolerance, and
`crates/mlrs-backend/tests/stacking_meta_test.rs` does the same against an
independently written host reference. The knob only moves work between them.

`numpy` is the shipping default, and that is a **measurement** (the ladder
below), not an assumption. The two Rust arms start in debt: this is one
`n x width` strided copy of data that is already in host memory (the blocks
arrive as whatever numpy arrays the composed estimators' `predict` returned), so
the host arm adds a capsule crossing each way and the device arm adds an upload
and a download on top of that. The arms exist because that debt is not obviously
unrepayable at large `n`, and because a claim about which copy is faster should
be a number.

The numpy arm also remains the **fallback** for everything the Rust arms cannot
represent: a non-float block, a duck-typed `X` (sklearn's `estimator_checks`
passes one), a block that is not 2-D, or row counts that disagree.
`_meta_via_rust` returns `None` for those rather than raising, so `np.hstack`
handles them exactly as it did before the arms existed.

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

`crates/mlrs-py/python/tests/test_stacking_meta_engine.py` adds 68 more cells
that re-run both string parameters — and the layout equivalences — **once per
meta-assembly arm**, so `host` and `device` are never covered only by synthetic
block data. 147 cells green on cpu, wgpu and rocm, zero skips on all three
(rocm included: the scatter needs no f64 matmul, so it runs there at f64 where
`backend_supports_f64()` alone would have wrongly skipped it).

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
`n=100000`, `d=64`, min of 3, **after LINEAR-ARM-CAL** (see below):

| config | sklearn fit | mlrs fit | ratio | was |
|---|---|---|---|---|
| `cv=2` | 0.761 s | 0.127 s | **5.98x** | 0.51x |
| `cv=3` | 1.341 s | 0.159 s | **8.43x** | 0.70x |
| `cv=5` | 1.545 s | 0.228 s | **6.77x** | 1.01x |
| `cv=10` | 2.999 s | 0.380 s | **7.90x** | 1.71x |
| `cv="prefit"` | 0.043 s | 0.026 s | 1.69x | 0.37x |
| `passthrough=True` (cv=5) | 1.585 s | 0.218 s | **7.27x** | 1.02x |
| `n_jobs=2` (cv=5) | 1.256 s | 0.246 s | **5.11x** | 0.77x |
| `n_jobs=4` (cv=5) | 1.060 s | 0.219 s | **4.85x** | 0.64x |

The `was` column is the same ladder on the same machine with the old dispatch
forced back on (`MLRS_RIDGE_GRAM_HOST=0`), so the two columns are one build and
one box apart, not two campaigns: 0.96–2.45x before, 4.85–8.43x after.

The **cpu** backend wins by 2.00–9.76x at `n=20000`, `d=32` — and that one is a
bug fix rather than a speed-up. `LinearRegression` had no host fit arm, so on
cpu every fold fit went through `center_columns`' per-column round-trip: the old
route did not complete a SINGLE `20000 × 32` fit in 600 s, against 5.5 ms now.

`predict` is unchanged by this work and is the remaining weak spot at
`n=100000`: 0.28–0.61x of sklearn, the same 0.31–0.92x the old dispatch
measured. It is a separate path (`predict_from_host`) and a separate campaign.

### Why the fit numbers moved: the arm was chosen by a constant (LINEAR-ARM-CAL)

Above a fixed dispatch floor, `Ridge` decided host-vs-device from a multiply-add
constant, and `LinearRegression` had no host arm at all. Both were wrong here:
the host arm wins EVERY rung of the `ridge_default_perf_test` A/B on both local
integrated GPUs (rocm 1.6–10x, wgpu 1.8–11x), yet the constant sent 6 of 8 rungs
to the device.

A bigger constant is not the fix, because the two machines this repo has data
for disagree at the *same* shape:

| `100 000 × 64` | host arm | device arm | |
|---|---|---|---|
| Colab T4 (discrete GPU, **2-vCPU** host) | 58.9 ms | 30.7 ms | device wins 1.9x |
| this box (integrated GPU, 16-core host) | 4.9 ms | 13.0 ms | host wins 2.7x |

So the floor stays a constant — it is about launch overhead, which really is
machine-independent — and above it the arm is chosen from rates measured once
per process on the machine that is running: `macs / host_rate` against
`bytes / upload_rate + macs / device_rate`. The model picks device on the T4's
numbers and host on this box's, which is what each one measured.

The probe is two-stage, because a device probe costs a pipeline compile (546 ms
on rocm here): the host rate is always measured (cheap, no device work), and the
device is probed only once the host estimate shows more than 25 ms at stake.
Below that the answer is "host" without touching the device. The bound that
gives up is explicit — up to ~25 ms left on the table, seen once at rocm
`100 000 × 128`, where the device arm would have won by 11%.

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

## Measured: the meta-matrix arms (STACK-META-01)

`scripts/bench_stacking_meta.py --level copy`, min of 5 fresh subprocesses,
five in-process reps each, arms interleaved rather than blocked. Cells are the
assembly ALONE — the numpy arm is `np.hstack`, the Rust arms are the same call
the shim makes, ingress and egress included, because that is what a user pays.
Blocks are one-column predictions (`k` of them); `d > 0` means `passthrough`
with that many `X` columns. **rocm, gfx1151 iGPU, f64, loadavg 3.8:**

| copy | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=2 | **0.003 ms** | 0.027 ms | 0.243 ms | 0.11x | 0.01x |
| n=10 000, k=2 | **0.009 ms** | 0.092 ms | 0.280 ms | 0.10x | 0.03x |
| n=100 000, k=2 | **0.062 ms** | 0.796 ms | 0.733 ms | 0.08x | 0.08x |
| n=1 000 000, k=2 | **0.644 ms** | 4.788 ms | 4.281 ms | 0.13x | 0.15x |
| n=100 000, k=8 | **0.242 ms** | 1.408 ms | 1.819 ms | 0.17x | 0.13x |
| n=100 000, k=2, d=32 | **0.714 ms** | 1.161 ms | 5.211 ms | 0.61x | 0.14x |
| n=1 000 000, k=2, d=32 | **15.1 ms** | 20.5 ms | 45.8 ms | 0.74x | 0.33x |
| n=100 000, k=2, d=128 | **5.60 ms** | 7.98 ms | 18.4 ms | 0.70x | 0.30x |

cpu and wgpu land in the same place (host 0.12–0.70x, device 0.01–0.23x), and
the cpu ladder was re-run with `--cpu-time` on a contended box to confirm the
verdict is not a load artefact: the host/numpy ratios came out identical.

**`np.hstack` wins every cell, so it stays the default.** Reading the ladder:

* **The host arm's floor is a fixed cost, not a slope.** At narrow widths it
  sits at a flat ~0.1x — that is the Arrow capsule crossing and the egress copy,
  which do not shrink with `n`. As the copy grows (`d=32`, `d=128`) the ratio
  climbs toward ~0.7x, i.e. the *copy itself* is competitive; what it cannot
  repay is the two extra buffer traversals the boundary costs. numpy needs one
  pass over each block; the Rust arms need one to hand the blocks over and one
  to bring the matrix back, on top of the same write.
* **The device arm is bus-bound, exactly as a zero-arithmetic kernel must be.**
  It uploads `n x width`, writes it once, and downloads `n x width`. There is no
  arithmetic to amortize that against, so it trails at every size; the best it
  reaches is 0.33x, at the largest passthrough copy where the kernel's bandwidth
  is at its most useful. It would only turn favourable for a caller whose blocks
  are ALREADY device-resident — which today's shim, receiving numpy arrays back
  from each member's `predict`, never is.
* **At fit level the choice is invisible.** `--level fit` ratios bounce between
  0.83x and 1.66x in both directions and do not reproduce run to run: at
  `n=100 000, d=32` the copy is ~1.1 ms against a ~155 ms fit, i.e. under 1%, so
  what that ladder measures is base-fit noise. The copy-level ladder above is
  the discriminating one, which is why it is the one quoted.

The arms remain in the tree because they are cheap to keep (one kernel, one
prim, one dispatcher), because they make the "numpy is faster here" claim a
number that can be re-checked on new hardware with one command, and because the
device arm is the piece a future device-resident stacking path would need.
