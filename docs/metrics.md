# `mlrs.metrics` — the full sklearn parameter surface

`mlrs.metrics` implements eleven `sklearn.metrics` functions with **every
parameter of their `scikit-learn==1.9.0` signatures**. They are free functions,
not estimators: no `output_type` routing, no device buffers, no `fit`.

```python
import numpy as np
import mlrs.metrics as mm

mm.confusion_matrix(y_true, y_pred, normalize="true")
mm.f1_score(y_true, y_pred, average="macro", zero_division=0)
mm.roc_auc_score(y_true, y_proba, multi_class="ovo", average="weighted")
mm.roc_auc_score(y_true, y_score, max_fpr=0.1)          # partial AUC
mm.precision_recall_curve(y_true, y_score, drop_intermediate=True)
mm.r2_score(Y_true, Y_pred, multioutput="variance_weighted")
```

| function | parameters |
|---|---|
| `accuracy_score` | `normalize`, `sample_weight` |
| `confusion_matrix` | `labels`, `sample_weight`, **`normalize`** |
| `precision_score` / `recall_score` / `f1_score` | `labels`, `pos_label`, `average`, `sample_weight`, `zero_division` |
| `log_loss` | `normalize`, `sample_weight`, `labels`, `eps` |
| `roc_auc_score` | `average`, `sample_weight`, **`max_fpr`**, `multi_class`, **`labels`** |
| `precision_recall_curve` | `pos_label`, `sample_weight`, **`drop_intermediate`** |
| `r2_score` | `sample_weight`, **`multioutput`**, **`force_finite`** |
| `mean_squared_error` / `mean_absolute_error` | `sample_weight`, **`multioutput`** |

Bold entries are the parameters added by METR-PARAM-01; the rest shipped with
the original surface. Class labels must be integer (or boolean) valued —
string labels are a separate, unrelated non-goal.

## Where the work happens

Every VALUE decision is in Rust (`crates/mlrs-algos/src/metrics/`). The Python
shim owns only what Rust cannot see: sklearn's parameter-validation messages,
its warnings, and the output shapes.

| what | where |
|---|---|
| the reductions, all averaging modes, `normalize`, `max_fpr`, `drop_intermediate`, `multioutput`, `force_finite` | Rust (`metrics::{classification,regression}`) |
| the `zero_division` policy (value) | Rust (`ZeroDivision`) |
| whether a zero division actually happened (the `'warn'` trigger) | Rust — reported as `PrfResult::zero_division_hit` |
| string → enum parsing, `Vec` ↔ capsule marshalling | the PyO3 layer (`crates/mlrs-py/src/metrics.rs`) |
| parameter-domain errors with sklearn's own messages | the Python shim |
| `UndefinedMetricWarning` / "No positive class found" warnings | the Python shim |
| `pos_label=None` resolution, multiclass `labels` validation, the y_score row-sum check | the Python shim |

### `zero_division='warn'` costs nothing

sklearn's default `zero_division="warn"` is the value `0` plus an
`UndefinedMetricWarning`. Detecting "a denominator was zero" from outside the
metric would mean a second O(n) pass, so the Rust layer reports it: the
precision/recall/f1 entry points return `PrfResult { out, zero_division_hit }`
and the binding hands the flag back with the value. Measured cost of the
default versus an explicit `zero_division=0`: **none** (14.5 ms vs 14.0 ms at
n = 1e6, inside the noise).

## Semantics pinned against scikit-learn 1.9.0

These were read off the pinned version, not derived — each one is a place where
the obvious implementation is wrong:

- **`confusion_matrix(normalize=...)` never produces NaN.** sklearn divides
  under `np.errstate(all="ignore")` and then `np.nan_to_num`s, so a class with
  no true (or no predicted) samples gets an all-zero row (column), not NaN.
- **`roc_auc_score`'s binary positive class is the LARGER label**, not `1`:
  sklearn binarizes against `np.unique(y_true)`, so a `{1, 2}` target's positive
  class is `2`. The `labels` parameter is ignored on the binary path.
- **`max_fpr=1.0` short-circuits** to the full AUC rather than going through the
  (mathematically equal, differently rounded) McClish formula.
- **Multiclass `average` options differ by strategy**: OvR takes
  `'micro'`/`'macro'`/`'weighted'`/`None`, OvO only `'macro'`/`'weighted'`
  (`average=None` raises `NotImplementedError` there — sklearn's own choice).
- **Multiclass `labels` must be sorted and unique**, and `y_score`'s rows must
  sum to 1 — sklearn rejects both otherwise, and so does mlrs.
- **`precision_recall_curve` sets recall to 1.0 everywhere** when `y_true` has
  no positive sample (with a warning), and its precision cell for a zero
  denominator is `0.0` — 1.9.0 changed this from earlier versions.
- **`r2_score` with fewer than 2 samples returns NaN** (plus
  `UndefinedMetricWarning`) before any `multioutput` reduction — even for
  `multioutput='raw_values'`, where the return is a bare scalar NaN.
- **`force_finite=False` is the whole point of the parameter**: a constant
  `y_true` yields `-inf` (imperfect prediction) or `NaN` (exact match) instead of
  the clamped `0.0`/`1.0`.
- **`average='binary'` has two guards, and one of them has a floor**: a target with more
  than two distinct labels raises "Target is multiclass but average='binary'", while
  "pos_label=… is not a valid label" only fires once **two** labels are actually present —
  a single-class target with an absent `pos_label` is a zero-division case, not an error.
  Both guards read the class order the Rust layer already resolved, so neither costs a pass.
- **`multioutput='variance_weighted'` is `r2_score`-only.**
  `mean_squared_error`/`mean_absolute_error` reject the string, and mlrs rejects
  it in Rust rather than inventing a meaning.

### Two deliberate divergences

Both predate this work and both are "gate the error, not a value":

1. `roc_auc_score` with a single class in `y_true` raises `ValueError`; sklearn
   returns `NaN` with an `UndefinedMetricWarning`.
2. `roc_auc_score(multi_class='ovo', sample_weight=...)` raises — as sklearn
   itself does — rather than silently ignoring the weights.

## Performance

Measured on this box (16 cores, **busy** — an unrelated 700 %-CPU job was
running), `scripts/bench_metrics_params.py`, minimum of 5–7 runs with the
mlrs/sklearn arms interleaved, CPU-time clock:

```
python3 scripts/bench_metrics_params.py --level all --repeat 7 --cpu-time
```

### The binding was the whole story, and is not any more

The first run of the harness said `mean_squared_error` took **44 ms** at
n = 1e6 against sklearn's 2.8 ms. None of that was arithmetic: PyO3's `Vec<T>`
extraction walks the Python sequence protocol element by element, ~44 ns each,
so every metric on this surface was paying ~44 ms/million-samples of pure
marshalling. The inputs now cross as zero-copy pyarrow `float64` capsules
(`pa.array` of a contiguous array is ~1 µs at **any** length), and the
precision-recall curve — whose three columns are O(n) long — comes back the same
way instead of as three lists of `PyFloat`s.

| n = 1e6 | before | after | sklearn |
|---|---|---|---|
| `mean_squared_error` | 44.3 ms | 5.6 ms | 7.0 ms |
| `accuracy_score` | 48.8 ms | 14.4 ms | 58.2 ms |

### Which parameters move the clock

**`multi_class` / `average` on multiclass `roc_auc_score` — the big one.**
n = 2e5, CPU-time ms:

| n_classes | ovr macro | ovr weighted | ovr None | ovr micro | ovo macro |
|---|---|---|---|---|---|
| 3 | 33.0 | 32.6 | 31.4 | 35.5 | 42.7 |
| 5 | 47.0 | 46.0 | 46.3 | 55.6 | 80.9 |
| 10 | 87.3 | 90.3 | 88.4 | **259.5** | 162.0 |

`'ovr'` runs `K` binary sweeps over `n` samples and grows linearly in `K`;
`'ovo'` runs `K(K-1)/2` sweeps over the class-PAIR subsets — `n(K-1)` samples of
total sort work, so it is roughly 2x OvR at `K = 10` and pulls further ahead as
`K` grows. `average='micro'` is a third shape entirely: ONE sweep over the
`n*K` raveled indicator/score pairs, which is the cheapest cell at `K = 3` and
**3x the most expensive** at `K = 10`. mlrs is 2.8–3.4x faster than sklearn in
every cell except `ovr micro` at `K = 10` (1.4x), where the single 2M-element
sort dominates.

**`multioutput` — cheap per se, but the 2-D path is where mlrs pulls away.**
n = 5e5, CPU-time ms:

| n_outputs | mse uniform | mse raw_values | r2 variance_weighted | sklearn (mse uniform) |
|---|---|---|---|---|
| 1 | 1.34 | 1.42 | 2.69 | 1.93 |
| 4 | 1.83 | 1.74 | 3.62 | 11.44 |
| 16 | 3.20 | 3.26 | 6.37 | 22.65 |

Choosing between `'raw_values'`, `'uniform_average'`, `'variance_weighted'` and
an explicit weight vector is free — they differ only in an O(k) reduction after
the O(n·k) pass. What is NOT free is the number of outputs, and there mlrs's
single-pass column accumulation beats numpy's temporaries by 6–8.6x (numpy
materializes an `n × k` squared-error array before reducing it).

**`max_fpr` — free.** 60.5 ms at `max_fpr=0.1` versus 67.8 ms for the full AUC
at n = 1e6: the partial path materializes the ROC polyline instead of streaming
the integral, and stops early, which cancels out.

**`drop_intermediate` — free at best, a small tax at worst.** On continuous
scores (1e6 distinct thresholds, 750 575 kept) it costs ~7 % (60.5 → 64.8 ms):
the extra O(m) mask pass is not repaid by the shorter egress unless the drop is
large. On tie-heavy scores (33 distinct thresholds) it is exactly free and
changes nothing. Use it for plot size, not for speed.

**`average` on precision/recall/f1, and `normalize` on `confusion_matrix` —
free.** Every `average` value lands at 14.0–14.3 ms (n = 1e6, K = 4); every
`normalize` value at 14.6–14.9 ms. Both parameters act on the O(K)/O(K²)
bookkeeping that follows an O(n) pass.

### The class-count slope, and what removed it

Resolving "which row of the matrix is this sample" with
`classes.iter().position(...)` is O(K) per sample, so the tabulation was
O(n·K) — a slope in the CLASS COUNT that no parameter controls and that sklearn
(whose tabulation is K-independent) does not have. `confusion_matrix` now
builds a [`ClassIndex`](../crates/mlrs-algos/src/metrics/mod.rs) first: a direct
table indexed by `label - min` when the label span is proportional to the class
count (the normal case — ids are almost always `0..K-1`), a `HashMap` when it is
not (`labels=[0, 1_000_000]` would otherwise allocate a megabyte for two
classes). Lookup is O(1) either way, and a duplicated label still resolves to
its first position, exactly as the scan did.

n = 1e6, wall clock on a quiet box, `--level confusion`:

| n_classes | before | after | vs sklearn (before → after) |
|---|---|---|---|
| 2 | 12.4 ms | 8.7 ms | 4.9x → 6.8x |
| 4 | 15.4 ms | 9.5 ms | 3.8x → 6.2x |
| 32 | 26.4 ms | 12.2 ms | 2.2x → 4.9x |
| 128 | 53.4 ms | 14.0 ms | 1.14x → 4.4x |

The residual 8.7 → 14.0 ms growth is not a per-sample scan any more: it is the
scatter's cache behavior (a `K × K` matrix of separately allocated rows) plus
the O(K²) normalization pass. It is flat in `n`.

The same `ClassIndex` now serves the other three sites that scanned:
`class_bookkeeping` (which feeds precision/recall/f1), `log_loss`'s column
lookup, and multiclass `roc_auc_score`'s `y_true` encoding.

`f1_score`, n = 1e6 (`--level prf`) — the same shape as `confusion_matrix`,
since both are one O(n) tabulation:

| n_classes | before | after | vs sklearn (before → after) |
|---|---|---|---|
| 2 | 11.1 ms | 8.8 ms | 6.3x → 8.1x |
| 4 | 13.9 ms | 9.7 ms | 5.3x → 7.9x |
| 32 | 21.8 ms | 11.7 ms | 4.5x → 8.6x |
| 128 | 39.1 ms | 13.1 ms | 3.0x → 9.3x |

`log_loss`, n = 1e6, `labels=[0..K-1]` (`--level logloss`):

| n_classes | before | after | vs sklearn (before → after) |
|---|---|---|---|
| 2 | 8.1 ms | 5.0 ms | 8.7x → 14.4x |
| 4 | 10.1 ms | 5.0 ms | 8.3x → 16.9x |
| 32 | 39.1 ms | 25.8 ms | 8.4x → 12.7x |
| 128 | 101.8 ms | 27.3 ms | 10.6x → 35.6x |

`log_loss` keeps a K-slope after the fix, and that one is real work rather than
a defect: its input IS an `n × K` probability matrix (1 GB at n = 1e6, K = 128),
so past K ≈ 32 it is memory-bandwidth-bound — which is why K = 128 costs barely
more than K = 32.

**Multiclass `roc_auc_score` measured as a no-op**, and that is the honest
result: 32.5 ms vs 30.0 ms at K = 3 and 246.7 ms vs 245.6 ms at K = 32
(n = 2e5, OvR macro) — run-to-run noise. The encoding pass is O(n·K) with a
tiny constant next to the K binary sweeps that follow it, each an
O(n log n) sort, so the scan was never this metric's bottleneck. It was changed
for consistency (one lookup type across the module), not for speed.

## Oracle tests

Every string-valued parameter is replayed against a committed sklearn fixture,
at both dtypes, from both layers:

- `crates/mlrs-algos/tests/metrics_params_test.rs` (25 tests) — the Rust API.
- `crates/mlrs-py/python/tests/test_oracle_metrics_params.py` (94 tests) — the
  full `numpy -> mlrs.metrics -> _mlrs -> Rust` path, plus the shim-only
  behavior (validation messages, warnings, dtypes, scalar-vs-array returns).

The fixture is `tests/fixtures/metrics_params_{f32,f64}_seed42.npz`, generated by
`scripts/gen_oracle.py::gen_metrics_params`. Its inputs are cast to the target
dtype BEFORE the reference values are computed, so the f32 fixture pins
sklearn's own f32 arithmetic rather than an f64 value the f32 replay could never
reproduce. It deliberately uses 4 classes (so OvO averages 6 class pairs, a
count a per-class loop cannot fake) and carries both a tie-heavy and a
continuous score vector (so `drop_intermediate` has a cell where it actually
drops points).
