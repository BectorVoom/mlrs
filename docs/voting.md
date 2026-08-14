# `VotingRegressor` — prediction voting

`mlrs.VotingRegressor` implements the full
`sklearn.ensemble.VotingRegressor` parameter surface. Every member is fitted on
the whole of `X`; `predict` returns the weighted mean of their predictions.

```python
import mlrs

reg = mlrs.VotingRegressor(
    estimators=[("lr", mlrs.LinearRegression()), ("ridge", mlrs.Ridge())],
    weights=[2.0, 1.0],
)
reg.fit(X, y).predict(X_test)
```

Members may be mlrs estimators, scikit-learn estimators, or a mix — the
composition only requires `fit`/`predict` and `is_regressor`.

The classification counterpart is
[`VotingClassifier`](voting_classifier.md), which adds one string-valued
parameter (`voting`) that forks it into two aggregations with nothing in common.
Everything on this page about `weights`, `'drop'`, `n_jobs` and the
`MLRS_VOTING_ENGINE` knob applies to it unchanged.

Compared with [stacking.md](stacking.md): stacking learns how to combine its
members (a second-stage estimator fitted on out-of-fold predictions); voting is
told, by `weights`. Voting is therefore much cheaper — one fit per member, no
cross-validation — and the two share their `estimators`-list mechanics through
one Python mixin (`_HeterogeneousComposition`) and one Rust rule set, exactly as
sklearn shares them through `_BaseHeterogeneousEnsemble`.

## Where the work happens

Voting is a *composition*: the arithmetic that dominates a fit already runs
inside the members. What the meta-estimator owns is structure plus one small
aggregation, and both are in Rust.

| what | where |
|---|---|
| estimator-name validation | Rust (`stacking_validate_names` — sklearn's shared base-class rule) |
| `'drop'` bookkeeping | Rust (`stacking_kept_indices`) |
| `weights` length rule | Rust (`voting_check_weights`) |
| `_weights_not_none` (the drop filter over `weights`) | Rust (`voting_active_weight_slots`) — see below |
| `get_feature_names_out` strings | Rust (`voting_feature_names`) |
| the aggregation (`transform`, `predict`) | numpy by default; Rust host / CubeCL device on request |
| `_parameter_constraints` | the Python shim — every rule is a predicate on an arbitrary Python object |
| member `fit` and `predict` | the composed estimators |

### The `weights` rule crosses as POSITIONS, not values

`voting_active_weight_slots` answers with the kept *indices*, and the shim then
indexes its own untouched `weights`. That is not a style preference — it is the
only shape that cannot lose a dtype.

`np.average` infers its result dtype from the columns **and** the weights, so a
`float32` weight array over `float32` predictions must leave `predict` in
`float32`. Converting the weights to `f64` on the way to Rust — the obvious thing
to do at an FFI boundary — silently promotes every such problem to `float64`,
and does so while still passing a 1e-5 comparison. The rule that has to live in
Rust (the length check, which entries survive) carries no numbers, so it does
not have to touch them.

The same reasoning applies to any Rust rule over caller-supplied numeric
parameters: cross with indices, not values.

## The aggregation arms (VOTE-01)

Two operations, selected by the `MLRS_VOTING_ENGINE` A/B knob:

```text
  transform(X) -> mat (n × k)   mat[r, j] = predⱼ[r]
  predict(X)   -> avg (n)       avg[r]    = (Σⱼ predⱼ[r]·wⱼ) / (Σⱼ wⱼ)
```

| value | arm |
|---|---|
| unset (**default**) | `np.asarray(...).T` / `np.average` in the shim |
| `numpy` | the same, forced |
| `host` | `mlrs_algos::ensemble::voting::{stack_columns, weighted_average}`, reached through the Arrow capsule |
| `device` | `mlrs_kernels::voting`, `k` accumulate launches plus one divide |

### Why this is not the same question `stacking.md` already answered

The stacking meta matrix is a pure copy — `n × width` up, `n × width` back, no
arithmetic — which is why `np.hstack` beat both Rust arms on every backend
there. `predict` here is a different shape: it **reduces**. It consumes `n · k`
and emits `n`, so the device arm's download is `k` times smaller than its
upload, and there are `k` multiplies and `k − 1` adds per row in between to
amortise the crossing against. That was enough of a difference to be worth
measuring rather than assuming; the ladder is below.

### The arms agree — exactly, with one documented exception

`transform` carries no arithmetic and is **byte-identical** on all three arms.

`predict` reproduces `np.average` operation for operation: the products, then a
left-to-right row sum in member order, then a **division** by the weight sum,
all in the input dtype (no wider accumulator on an f32 problem, no
reciprocal-multiply). So the `numpy` and `host` arms are bit-identical, and the
tests assert exact equality rather than a tolerance — which is the only
assertion strong enough to catch a reassociated accumulation.

The `device` arm is the exception, and it is a hardware fact rather than a bug:
`acc + pred·w` is the canonical fused-multiply-add shape, and a GPU backend
contracts it into one FMA instruction — **one** rounding where numpy performs
two. Measured on rocm gfx1151 at f32 the gap is at most **1 ULP**; the cpu
backend (LLVM at `-O0`) does not contract and comes out bit-exact from the same
source. cubecl exposes no per-kernel `fp-contract` control. So the device arm is
gated at `4 · eps` relative — deliberately much tighter than the project's 1e-5
contract, which is ~80 f32 ULP and would let a genuine accumulation bug through.
The contracted answer is *more* accurate than the reference, not less.

### The numpy fallback

`numpy` also remains the **fallback** for everything the Rust arms cannot
represent: a non-float column (a member is free to return an integer array,
which `np.average` promotes), a column that is not 1-D, lengths that disagree,
or a zero-row problem. `_vote_via_rust` returns `None` for those rather than
raising, so numpy handles them exactly as it did before the arms existed.

## Parameters

| parameter | default | notes |
|---|---|---|
| `estimators` | — | `list of (str, estimator)`; an entry may be the string `'drop'` |
| `weights` | `None` | array-like of length `len(estimators)`; `None` is uniform |
| `n_jobs` | `None` | joblib fan-out over the member fits; see the device caveat below |
| `verbose` | `False` | print each member's fit time as it completes |

### `weights` is indexed against the FULL `estimators` list

`len(weights) == len(estimators)` is checked **before** the `'drop'` filter, and
a dropped entry's weight is then discarded rather than shifting the others
along. That is what makes `set_params(name='drop')` usable on a weighted
ensemble without rewriting `weights`, and it is sklearn's rule — a shim that
filtered before checking would reject a fit sklearn completes.

A weight vector summing to zero is a `ZeroDivisionError` from `predict` (numpy's
own exception and message), **not** a `ValueError` and not an infinity. It
surfaces from `predict` rather than `fit` because sklearn's `fit` only checks
the length. Individual negative weights are legal; only the sum is constrained.

### `n_jobs` is ignored when a member holds a device handle

Same rule and same reason as [stacking.md](stacking.md): a fitted mlrs estimator
owns a compiled `#[pyclass]` wrapping device state, which joblib's process
backends cannot pickle, and mlrs serializes device work behind one pool lock so
a threaded fan-out measures ~1.2x at best. `mlrs.VotingRegressor` emits a
`UserWarning` and fits serially in that case. `n_jobs` works normally over host
(scikit-learn) members — see the ladder below.

### `allow_nan` and `sparse` are derived from the members

This estimator never touches `X` itself; it hands it straight to the members. So
the two input tags are the AND over them, as in sklearn's
`_BaseHeterogeneousEnsemble`. A `VotingRegressor` over sklearn estimators
accepts sparse input; one over mlrs estimators (dense Arrow ingress) does not.
This is load-bearing rather than cosmetic: sklearn's
`check_estimator_sparse_tag` asserts that an estimator declaring `sparse=False`
actually *rejects* sparse input.

## Parity

`crates/mlrs-py/python/tests/test_oracle_voting_regressor.py` compares against a
live `sklearn.ensemble.VotingRegressor` — **104 cells**, green on **cpu, wgpu
and rocm**. Compositions of sklearn members match **exactly** (`atol=0`),
including every rejection message and every result dtype. Compositions of mlrs
members match within `conftest.live_atol()`.

`crates/mlrs-py/python/tests/test_voting_engine.py` re-runs the arm agreement —
and the `'drop'` sentinel — **once per aggregation arm**, so `host` and `device`
are never covered only by synthetic column data: **42 cells**. The single skip in
that file is deliberate: `device` refusing f64 has nothing to assert on a backend
that supports it.

`crates/mlrs-py/python/tests/test_estimator_checks.py` adds sklearn's
`parametrize_with_checks` sweep, and
`test_oracle_stacking{,_classifier}.py` (160 cells) gate the
`_HeterogeneousComposition` split that this estimator's arrival forced out of
`_StackComposition`.

### The string-valued parameter surface

`VotingRegressor(estimators, *, weights=None, n_jobs=None, verbose=False)` has
exactly **one** place a caller supplies a string, and it is not a scalar
parameter: the `'drop'` sentinel inside `estimators`. `weights` is array-like or
`None`, `n_jobs` is an int or `None`, `verbose` is a bool. (This is the whole
difference from `StackingClassifier`, which adds `stack_method` and `cv`.)

`'drop'` is oracle-tested in every combination that can interact with it:

* each position in the list, because an off-by-one in the kept-index bookkeeping
  still produces the right answer when the dropped entry is last;
* both spellings — the constructor argument and `set_params(name='drop')`, which
  reach different code;
* with and without `weights` (asymmetric weights, so a misaligned vector is
  visible);
* through `transform`, `get_feature_names_out`, `named_estimators_` and
  `n_features_in_`;
* the all-dropped rejection, message compared against sklearn's;
* the **near misses** — `'dropped'`, `'DROP'`, `'Drop'`, `' drop'` — which are
  *not* the sentinel and must fall through to the type check. Silently disabling
  one of those is unobservable in the output shape whenever a sibling survives,
  which is why they are tested explicitly. Note the rejection arrives as an
  **`AttributeError`**, not a `ValueError`: `is_regressor` asks the object for
  `__sklearn_tags__` and a `str` has none. The test asserts the exception *type*
  too, so a shim that "helpfully" normalized it would fail;
* and once more per aggregation arm, since `'drop'` changes *which* columns are
  aggregated.

mlrs adds one further string surface of its own, `MLRS_VOTING_ENGINE`
(`numpy` / `host` / `device`, plus the unknown-value fallback). It is a
benchmarking knob rather than a constructor parameter and is covered by the
engine suite.

`mlrs.VotingRegressor` also passes sklearn's `parametrize_with_checks` sweep
with no estimator-specific xfails — the only entries it takes from the shared
xfail map are the ones every mlrs estimator takes.

## Measured: the aggregation arms

`scripts/bench_voting.py --level agg`, min of 5 fresh subprocesses, five
in-process reps each, arms interleaved rather than blocked. Cells are the
aggregation ALONE — the numpy arm is `np.average` / `np.asarray(...).T`, the
Rust arms are the same call the shim makes, ingress and egress included, because
that is what a user pays. The one-time `_mlrs` extension load is reported
separately and excluded from every cell.

**rocm, gfx1151 iGPU, f32, loadavg 3.0–3.8** (the sweep is gated on the box
being quiet — a contended box has inverted a verdict on this project before).

`predict`, **uniform** weights — `np.average(..., weights=None)`, i.e. a mean:

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | **0.005 ms** | 0.011 ms | 0.070 ms | 0.48x | 0.08x |
| n=10 000, k=3 | **0.011 ms** | 0.027 ms | 0.219 ms | 0.42x | 0.05x |
| n=100 000, k=3 | **0.084 ms** | 0.185 ms | 0.377 ms | 0.45x | 0.22x |
| n=1 000 000, k=3 | **1.348 ms** | 2.204 ms | 3.171 ms | 0.61x | 0.42x |
| n=100 000, k=2 | **0.072 ms** | 0.133 ms | 0.253 ms | 0.54x | 0.28x |
| n=100 000, k=8 | **0.141 ms** | 0.548 ms | 0.779 ms | 0.26x | 0.18x |
| n=1 000 000, k=8 | **2.906 ms** | 7.197 ms | 6.174 ms | 0.40x | 0.47x |

`predict`, **weighted** — `np.average(..., weights=[...])`:

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | **0.011 ms** | 0.014 ms | 0.078 ms | 0.82x | 0.14x |
| n=10 000, k=3 | **0.023 ms** | 0.034 ms | 0.152 ms | 0.66x | 0.15x |
| n=100 000, k=3 | **0.157 ms** | 0.229 ms | 0.643 ms | 0.68x | 0.24x |
| n=1 000 000, k=3 | 3.579 ms | **3.228 ms** | 7.582 ms | **1.11x** | 0.47x |
| n=100 000, k=2 | **0.119 ms** | 0.166 ms | 0.516 ms | 0.72x | 0.23x |
| n=100 000, k=8 | **0.431 ms** | 0.682 ms | 1.655 ms | 0.63x | 0.26x |
| n=1 000 000, k=8 | **9.024 ms** | 9.394 ms | 18.405 ms | 0.96x | 0.49x |

`transform` — the pure transpose, no arithmetic:

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | **0.001 ms** | 0.011 ms | 0.079 ms | 0.09x | 0.01x |
| n=10 000, k=3 | **0.002 ms** | 0.019 ms | 0.269 ms | 0.10x | 0.01x |
| n=100 000, k=3 | **0.018 ms** | 0.110 ms | 0.451 ms | 0.16x | 0.04x |
| n=1 000 000, k=3 | **0.643 ms** | 1.550 ms | 5.287 ms | 0.41x | 0.12x |
| n=100 000, k=2 | **0.012 ms** | 0.076 ms | 0.390 ms | 0.16x | 0.03x |
| n=100 000, k=8 | **0.051 ms** | 0.322 ms | 1.771 ms | 0.16x | 0.03x |
| n=1 000 000, k=8 | **1.682 ms** | 8.786 ms | 14.729 ms | 0.19x | 0.11x |

**cpu backend, f64, loadavg 2.6–3.8.** Note that the `device` column here is
**cubecl-cpu**, not a GPU: it spawns one OS thread per unit and JITs at `-O0`
(see `mlrs-cubecl-cpu-execution-model`), so its numbers say nothing about GPU
bandwidth and are reported only for completeness. The `host` column is the one
this table is for — and it is where the largest effect in this document lives.

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| **`predict`, uniform** | | | | | |
| n=1 000, k=3 | **0.005 ms** | 0.011 ms | 0.092 ms | 0.41x | 0.05x |
| n=10 000, k=3 | **0.012 ms** | 0.027 ms | 0.172 ms | 0.45x | 0.07x |
| n=100 000, k=3 | **0.101 ms** | 0.192 ms | 1.032 ms | 0.53x | 0.10x |
| n=1 000 000, k=3 | 2.241 ms | **2.162 ms** | 11.013 ms | **1.04x** | 0.20x |
| n=100 000, k=2 | **0.080 ms** | 0.141 ms | 0.786 ms | 0.57x | 0.10x |
| n=100 000, k=8 | **0.240 ms** | 0.565 ms | 2.693 ms | 0.43x | 0.09x |
| n=1 000 000, k=8 | **6.077 ms** | 6.648 ms | 26.541 ms | 0.91x | 0.23x |
| **`predict`, weighted** | | | | | |
| n=1 000, k=3 | **0.011 ms** | 0.013 ms | 0.082 ms | 0.83x | 0.13x |
| n=10 000, k=3 | **0.023 ms** | 0.029 ms | 0.158 ms | 0.80x | 0.14x |
| n=100 000, k=3 | **0.145 ms** | 0.193 ms | 1.148 ms | 0.75x | 0.13x |
| n=1 000 000, k=3 | 3.821 ms | **2.127 ms** | 11.168 ms | **1.80x** | 0.34x |
| n=100 000, k=2 | **0.110 ms** | 0.139 ms | 0.745 ms | 0.79x | 0.15x |
| n=100 000, k=8 | **0.486 ms** | 0.556 ms | 2.825 ms | 0.87x | 0.17x |
| n=1 000 000, k=8 | 10.532 ms | **7.150 ms** | 25.179 ms | **1.47x** | 0.42x |
| **`transform`** | | | | | |
| n=1 000, k=3 | **0.001 ms** | 0.012 ms | 0.082 ms | 0.10x | 0.01x |
| n=100 000, k=3 | **0.035 ms** | 0.142 ms | 0.829 ms | 0.25x | 0.04x |
| n=1 000 000, k=3 | **1.268 ms** | 3.354 ms | 11.370 ms | 0.38x | 0.11x |
| n=1 000 000, k=8 | **4.140 ms** | 20.922 ms | 36.665 ms | 0.20x | 0.11x |

**`numpy` wins most cells, so it stays the default — but not every cell.**
Reading the ladders:

* **`weights` is a real performance parameter, and the cost is on numpy's
  side.** `np.average(weights=...)` materialises the whole `n × k` product array
  before reducing it, where `weights=None` takes `mean`'s fused path. So
  switching weights on costs **numpy** 1.7x at n=10⁶, k=3 on cpu/f64 (2.24 →
  3.82 ms) and 1.7x at k=8 (6.08 → 10.53 ms) — on rocm/f32, 2.7x and 3.1x. The
  host arm fuses the multiply into the accumulation and materialises nothing, so
  the same switch is nearly free for it (2.16 → 2.13 ms at k=3).
* **That is where the Rust host arm WINS**: **1.80x** at n=10⁶, k=3 and
  **1.47x** at k=8 on cpu/f64; 1.11x and 0.96x on rocm/f32 (a smaller margin
  because f32 halves the bytes numpy wastes on the intermediate). It also
  crosses 1.0 on the *unweighted* path at n=10⁶, k=3 (1.04x). Below ~10⁵ rows it
  loses everywhere, by a fixed amount.
* **The host arm's floor is a fixed cost, not a slope.** At small `n` it sits at
  a flat ~0.1x on `transform` and ~0.45x on `predict`; the ratio climbs with `n`
  because the Arrow capsule crossing and the egress copy do not shrink. The work
  itself is competitive — better than competitive on the weighted path; the two
  extra buffer traversals are what it cannot repay at small sizes.
* **The reduction hypothesis is confirmed — directionally, and not enough to
  win.** On rocm, compare the device column against `transform`, which is the
  same upload with a `k`-times bigger download and no arithmetic: at n=10⁶, k=8
  the device arm reaches **0.47x** of numpy on `predict` against **0.11x** on
  `transform`. Shrinking the download by a factor of `k` is worth roughly 4x in
  relative standing, exactly as predicted — it just starts too far behind. The
  arm is still bus-bound: it uploads `n · k`, and `k` multiplies plus `k − 1`
  adds per row is not arithmetic a GPU needs meaningful time for. It improves
  monotonically with `n` (0.05x → 0.47x), which is the fixed launch and capsule
  cost amortising rather than bandwidth arriving.
* **`transform` behaves exactly like the stacking meta-copy** (0.09–0.41x host),
  which it should — it is the same operation with no arithmetic to fuse.
  `docs/stacking.md`'s conclusion transfers unchanged, and it is the contrast
  that isolates the fusion as the cause of the `predict` win.

**Should the default switch to `host` for large weighted ensembles?** The win is
real and, at 1.80x, larger than anything the stacking ladder found — but it
needs n ≳ 10⁶, and the aggregation is a sub-millisecond slice of a call whose
member `predict`s dominate. Making the default a size-and-weights heuristic
would trade a predictable code path for a fraction of a percent of a real
`predict`. So `numpy` ships, and a caller aggregating millions of rows has
`MLRS_VOTING_ENGINE=host`.

## Measured: `n_jobs`

`scripts/bench_voting.py --level fit`, cpu backend, quiet-gated, min of 3 fresh
subprocesses. Members are **sklearn** estimators, because the fan-out over an
mlrs member is reduced to serial by design and would measure the warning rather
than the parallelism.

**This ladder needs two member sets, and reporting only one of them would have
been wrong.** A voting ensemble fits each member exactly **once** — unlike
stacking, where `cv=k` gives every member `k + 1` fits. So there are only `k`
units of work to spread, and the speedup ceiling is Amdahl's,
`total / slowest`, over the members themselves.

**Balanced members** (`--balanced`: four depth-8 trees at different seeds, so the
ceiling is 4):

| config | n_jobs=None | n_jobs=2 | n_jobs=4 | best/serial |
|---|---|---|---|---|
| n=10 000, d=32 | 456 ms | 342 ms | **232 ms** | 1.96x |
| n=100 000, d=32 | 6 189 ms | 3 373 ms | **1 975 ms** | **3.13x** |
| n=100 000, d=128 | 25 183 ms | 13 268 ms | **7 913 ms** | **3.18x** |

**Mixed members** (the default pool: `LinearRegression`, `Ridge`, a depth-8
`DecisionTreeRegressor`, `Lasso`):

| config | n_jobs=None | n_jobs=2 | n_jobs=4 | best/serial |
|---|---|---|---|---|
| n=10 000, d=32 | **129 ms** | 232 ms | 232 ms | 1.00x |
| n=100 000, d=32 | 1 719 ms | 1 764 ms | **1 712 ms** | 1.00x |
| n=100 000, d=128 | 7 019 ms | 6 733 ms | **6 669 ms** | 1.05x |

Reading them together:

* **`n_jobs` works.** On a balanced ensemble it reaches **3.13–3.18x** at four
  jobs against a ceiling of 4 — normal joblib scaling.
* **The mixed pool's flat result is Amdahl, not a defect.** Timing the members
  individually at these shapes: the depth-8 tree is **94–95%** of the total
  (n=100 000, d=128: tree 6 316 ms of 6 961 ms), so the ceiling is **1.05–1.10x**
  — and the measured 1.00–1.05x lands on it. Nothing is being left on the table.
* **Below ~10⁵ rows the fan-out can cost more than it saves.** At n=10 000 the
  mixed pool goes 129 ms → 232 ms, which is joblib's `loky` process spawn
  (~100 ms) against a fit that only takes 129 ms. Balanced members at the same
  size still win (456 → 232 ms) because there is enough work to hide the spawn.
  Note that both land on the same ~232 ms floor: that is the spawn, not the
  work.

So the useful rule for a `VotingRegressor` is: **`n_jobs` pays in proportion to
how evenly the members' costs are matched.** One dominant member and it buys
nothing, however many cores are free. Check `total / slowest` over the members
before reaching for it.
