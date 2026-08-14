# `VotingClassifier` — soft voting / majority rule

`mlrs.VotingClassifier` implements the full
`sklearn.ensemble.VotingClassifier` parameter surface. Every member is fitted on
the whole of `X` against a label-encoded target; `predict` combines their
answers in one of two ways, chosen by `voting`.

```python
import mlrs

clf = mlrs.VotingClassifier(
    estimators=[("nb", mlrs.GaussianNB()), ("knn", mlrs.KNeighborsClassifier())],
    voting="soft",
    weights=[2.0, 1.0],
)
clf.fit(X, y).predict(X_test)
```

Members may be mlrs estimators, scikit-learn estimators, or a mix — the
composition only requires `fit`/`predict` (plus `predict_proba` under
`voting='soft'`) and `is_classifier`.

It shares its `estimators`-list mechanics with [`VotingRegressor`](voting.md)
and [stacking](stacking.md) through one Python mixin (`_HeterogeneousComposition`
plus `_VoteComposition`) and one Rust rule set, exactly as sklearn shares them
through `_BaseHeterogeneousEnsemble` / `_BaseVoting`. Read
[voting.md](voting.md) first: everything it says about `weights`, `'drop'`,
`n_jobs` and the `MLRS_VOTING_ENGINE` knob applies here unchanged, and this page
covers only what the classifier adds.

## `voting` forks the estimator into two

This is the one parameter that matters most, and it is not a tuning knob — it
selects between two aggregations that share no data path at all:

| method | `voting='hard'` | `voting='soft'` |
|---|---|---|
| `predict` | `argmax_c Σⱼ wⱼ·[predⱼ[r] == c]` | `argmax_c avg[r, c]` |
| `predict_proba` | **absent** (`available_if`) | `avg[r, c] = (Σⱼ probaⱼ[r,c]·wⱼ) / Σⱼ wⱼ` |
| `transform` | the `(n, k)` label matrix | `np.hstack(probas)`, or the raw `(k, n, C)` stack when `flatten_transform=False` |
| `get_feature_names_out` | `votingclassifier_<name>` | `votingclassifier_<name><i>`, `n_classes` per member |
| what crosses the FFI | `uint32` label columns | `n × n_classes` float blocks |

`hasattr(clf, "predict_proba")` is `False` under hard voting, and re-evaluated
per access — so `set_params(voting="soft")` makes it appear, on an already-fitted
estimator, which is sklearn's behaviour.

## Where the work happens

| what | where |
|---|---|
| estimator-name validation | Rust (`stacking_validate_names`) |
| `'drop'` bookkeeping | Rust (`stacking_kept_indices`) |
| `weights` length rule | Rust (`voting_check_weights`) |
| `_weights_not_none` | Rust (`voting_active_weight_slots`) — POSITIONS, not values; see [voting.md](voting.md) |
| the `voting` constraint | Rust (`voting_mode`) — one parse, so the shim's branches read Rust's answer rather than re-comparing the literal |
| `get_feature_names_out` strings and its one rejection | Rust (`voting_classifier_feature_names`, `voting_check_feature_names`) |
| hard `predict` | numpy by default; Rust host / CubeCL device on request |
| soft `predict` / `predict_proba` / flattened `transform` | numpy by default; Rust host / CubeCL device on request |
| hard `transform` | numpy on **every** arm — see below |
| label encoding (`le_`, `classes_`) | `sklearn.preprocessing.LabelEncoder`, as sklearn does |
| `_parameter_constraints` | the Python shim — every rule is a predicate on an arbitrary Python object |

### Hard `transform` stays in numpy, deliberately

It returns the members' **labels**, which are integers. The Rust aggregation
arms are float-typed — they exist to reproduce `np.average` bit for bit — so
`_vote_via_rust` declines an integer column and numpy answers on every arm. The
alternative is a float round-trip that would change the dtype sklearn returns.
This is a documented gap in arm coverage, asserted as such in
`test_voting_classifier_engine.py`, not a latent one.

## The aggregation arms (VOTE-CLF-01)

Same `MLRS_VOTING_ENGINE` knob as the regressor, same three values, same default
(`numpy`). What is new is that there are now four aggregations behind it, and
they are held to different bounds:

| aggregation | numpy vs host | vs device |
|---|---|---|
| hard `predict` | **exact** | **exact** |
| soft `predict_proba` | **exact** | ≤ 4 · eps relative |
| soft `predict` | exact labels | exact labels |
| soft `transform` | **exact** | **exact** |

### Hard voting is exact on the device too, and that is a real claim

The regressor's average is `acc + pred·w`, the canonical fused-multiply-add
shape, which a GPU contracts into one FMA and so rounds once where numpy rounds
twice ([voting.md](voting.md) has the measurement). The hard tally has no such
shape: `vote_bincount_add` adds a **scalar** weight into a bin, one rounding on
every backend. So the device arm is held to equality here, and a drift would
mean the tally or the tie-break is wrong — not the hardware.

The tally width is `f64`, matching `np.bincount(x, weights=w)`'s own accumulator
regardless of the weights' dtype. The uniform case is a sum of `1.0`s and is
exact at either width, which matches numpy's `int64` counting bit for bit.

### `np.bincount`'s length is per row, and it is observable

`np.bincount(x, weights=w)` returns `x.max() + 1` entries, not `n_classes` — so
`argmax` never considers a class above the row's own largest prediction. With
non-negative weights that is invisible: any class present has a count ≥ the
absent classes' implicit `0`, and `argmax` takes the first maximum. With
**negative** weights it decides the answer:

```text
  w = [-1, -2],  row = [0, 0]
  np.bincount(row, weights=w) == [-3.0]        -> argmax 0
  a full-width tally          == [-3.0, 0, 0]  -> argmax 1   (WRONG)
```

sklearn's `weights` constraint is `array-like`, which admits negatives. So both
mlrs arms track each row's label ceiling and bound the argmax by it —
`vote_bincount_add` maintains `hi[r]`, `vote_argmax_bounded` scans `0..=hi[r]`.

### Soft voting needs no new reduction

`np.average(probas, axis=0, weights=w)` over a `(k, n, C)` stack is *exactly*
the regressor's row mean with `n · C` in place of `n`: the reduced axis is still
the member axis and each member still contributes one contiguous block. So
`vote_init_weighted` / `vote_add_weighted` / `vote_divide` are reused unchanged,
and soft voting inherits the regressor's numpy-parity guarantee and its one-ULP
device caveat together.

The one genuinely new kernel on that route is `vote_argmax_rows`, and it is the
reason the soft route has a device arm worth having: it consumes the averaged
`n × C` block **on the device** and emits `n` labels, so `predict` never
downloads the probability matrix at all. numpy cannot do that — it has to
materialise the full average before it can reduce it.

### The numpy fallback

`numpy` remains the **fallback** for everything the Rust arms cannot represent.
Hard voting declines a non-integer label column (so `np.bincount`'s own
`TypeError` is what a caller sees), a **negative** label (so numpy's *"'list'
argument must have no negative elements"* is what a caller sees), a non-1-D
column, mismatched lengths, and an empty query. Soft voting declines a block
that is not 2-D, blocks whose shapes disagree, a non-float promotion, and an
empty query. In every one of those cases mlrs reproduces sklearn exactly,
**including where sklearn itself raises**.

## Parameters

| parameter | default | notes |
|---|---|---|
| `estimators` | — | `list of (str, estimator)`; an entry may be the string `'drop'` |
| `voting` | `'hard'` | `{'hard', 'soft'}`. Forks the estimator; see above |
| `weights` | `None` | indexed against the **full** `estimators` list |
| `n_jobs` | `None` | joblib fan-out over the member fits; reduced to serial over mlrs members |
| `flatten_transform` | `True` | consulted only under `voting='soft'` |
| `verbose` | `False` | one line per member fit |

### `flatten_transform` only exists under soft voting

Under `voting='hard'` sklearn ignores it entirely and returns the `(n, k)` label
matrix either way; a shim that honoured it there would change a shape sklearn
does not. Under `voting='soft'`, `False` returns the raw `(k, n, C)` stack — and
because a 3-D output has no columns, `get_feature_names_out` then raises:

```text
get_feature_names_out is not supported when `voting='soft'` and `flatten_transform=False`
```

### `voting='soft'` is not checked at fit time

sklearn does not verify that every member implements `predict_proba` until
`predict` asks for it, so an `SVC(probability=False)` member **fits fine** and
raises from `predict`. mlrs reproduces the timing, not just the exception:
moving the check into `fit` would reject an ensemble a caller could legitimately
go on to use with `voting='hard'`.

### The target: two exception classes, on purpose

| `y` | exception |
|---|---|
| continuous, or unnameable by `type_of_target` | `ValueError`: *Unknown label type: …* |
| multilabel / multi-output | `NotImplementedError`: *VotingClassifier only supports binary or multiclass …* |

A caller can tell "you gave me nonsense" from "I have not built that", and
collapsing the two would lose that.

### A regressor member is rejected

Unlike `StackingClassifier` — where sklearn deliberately allows a regressor
first layer for ordinal problems — a `VotingClassifier` requires classifiers:
*The estimator LinearRegression should be a classifier.*

## Parity

Every rule above is observable from Python as an exception message, a
`get_feature_names_out()` string, a shape, or a predicted value, and each is
oracle-tested against a **live** sklearn in the same process:

* `crates/mlrs-py/python/tests/test_oracle_voting_classifier.py` — the full
  parameter surface, both string-valued parameters, on the default arm;
* `crates/mlrs-py/python/tests/test_voting_classifier_engine.py` — the same
  string parameters re-run on the `host` and `device` arms, plus the arms'
  agreement bounds;
* `crates/mlrs-algos/tests/voting_test.rs` — the Rust core's rules and host
  aggregations;
* `crates/mlrs-backend/tests/voting_test.rs` — the CubeCL kernels, live-launched;
* `crates/mlrs-py/python/tests/test_estimator_checks.py` — sklearn's own
  `check_estimator` suite, entered **twice** (once per `voting` value), because
  the two routes reach different checks.

### Landmine: sklearn's `StrOptions` message is not deterministic

`The 'voting' parameter … must be a str among {…}` renders its options by
iterating a Python `set`, whose order for these two strings changes with
`PYTHONHASHSEED`. The oracle parses the option set out of both messages rather
than comparing them as text — comparing them literally would be a coin flip.
Same trap as `stack_method` in [stacking.md](stacking.md).

### The string-valued parameter surface

| string | where it is validated | rejection |
|---|---|---|
| `voting='hard' \| 'soft'` | Rust `voting_mode`, at sklearn's point in `fit` | `InvalidParameterError` with the `StrOptions` text |
| `estimators=[(name, 'drop')]` | Rust `stacking_kept_indices` | a near-miss (`'dropped'`, `'DROP'`) falls through to the classifier type check and raises sklearn's own `AttributeError` |

<!-- MEASUREMENTS -->

## Measured: the aggregation arms

`scripts/bench_voting_classifier.py --level agg`, cpu backend. Every cell runs in
a fresh subprocess and the table reports the minimum of 3; cells are interleaved
across arms, not blocked, so a drifting machine penalizes all of them equally.
The one-time `_mlrs` load is warmed outside every timed region.

**These are CPU-time cells, not wall clock, and that is deliberate.** The box was
co-tenanted with another session throughout this run (loadavg 100-260), which is
the condition this project has twice recorded as capable of *inverting* a verdict
(`mlrs-cpu-bench-separate-processes`). CPU time is the load-robust metric, and it
proved it here: the same cell measured in two independent runs hours apart, under
loadavg 176 and loadavg 224, came out at 155.10x and 156.70x. A wall-clock run
taken between them produced one cell at 3 749x — a pure contention artifact — and
is not reported.

Note also that the ladder cannot be gated behind a "wait for a quiet box" check:
on the cpu backend the **device arm is itself the load**, since cubecl-cpu runs
one OS thread per unit and pushes `procs_running` past 260 on its own.

### `voting='hard'` — the ladder that matters

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | 1.687 ms | **0.073 ms** | 1.643 ms | **23.1x** | 1.0x |
| n=10 000, k=3 | 16.553 ms | **0.168 ms** | 2.321 ms | **98.4x** | 7.1x |
| n=100 000, k=3 | 167.372 ms | **1.068 ms** | 21.238 ms | **156.7x** | 7.9x |
| n=1 000 000, k=3 | 1 643.400 ms | **11.788 ms** | 221.783 ms | **139.4x** | 7.4x |
| n=100 000, k=2 | 163.651 ms | **1.000 ms** | 3.819 ms | **163.7x** | 42.9x |
| n=100 000, k=8 | 245.900 ms | **2.771 ms** | 10.558 ms | **88.8x** | 23.3x |
| n=1 000 000, k=8 | 2 500.565 ms | **32.169 ms** | 237.550 ms | **77.7x** | 10.5x |

Uniform weights; the weighted ladder is the same shape (sklearn's `bincount` takes
the `weights` argument either way, so its per-row Python loop costs the same).

**Two orders of magnitude, and the reason is structural rather than clever.**
sklearn's hard route is

```python
np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=w)), 1, predictions)
```

— a **Python-level loop over `n` rows**, allocating a fresh `bincount` array per
row. It is the one place in either voting estimator where sklearn is not already
running vectorised numpy, so the host arm is not beating a tuned kernel; it is
replacing an interpreter loop with a single pass and a reused scratch tally.

This is the exact opposite of what [voting.md](voting.md) concluded for the
regressor, where `np.average` is already vectorised and the Rust arms start an
Arrow round-trip in debt. Same estimator family, same knob, opposite verdict —
which is why `voting` had to be measured rather than assumed.

**The device arm wins too, but by far less** (7-43x), and for the usual reason:
`n · k` labels up, an `n · n_bins` tally allocated on device, `n` back, against a
host arm that never crosses the bus at all. It is not the arm to reach for here.

The weighted ladder was interrupted by the co-tenant after its first rows; what it
did record (`23.08x → 32.08x` at n=1 000 as the weights turn on) is the same shape,
which is expected — `np.bincount` takes the `weights` argument on both paths, so
sklearn's per-row Python loop costs the same either way.

### `voting='soft'` — no new verdict, and that is the point

Soft voting IS the regressor's reduction with `n · n_classes` elements per member
(see above), so it inherits [voting.md](voting.md)'s ladder rather than
establishing its own. Reproduced here to confirm that the extra axis does not
change the shape — **wall clock, on a contended box**, so read the ratios as
approximate:

`predict_proba`, weighted — `np.average(probas, axis=0, weights=w)`:

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | **0.015 ms** | 0.028 ms | 0.151 ms | 0.53x | 0.10x |
| n=10 000, k=3 | **0.075 ms** | 0.125 ms | 0.416 ms | 0.60x | 0.18x |
| n=100 000, k=3 | **1.153 ms** | 1.284 ms | 9.923 ms | 0.90x | 0.12x |
| n=1 000 000, k=3 | 15.537 ms | **7.460 ms** | 38.665 ms | **2.08x** | 0.40x |
| n=100 000, k=8 | **2.922 ms** | 4.122 ms | 8.493 ms | 0.71x | 0.34x |
| n=1 000 000, k=8 | 45.622 ms | **20.751 ms** | 235.962 ms | **2.20x** | 0.19x |

`transform` (`flatten_transform=True`) — the `np.hstack` copy:

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | **0.015 ms** | 0.017 ms | 0.095 ms | 0.90x | 0.16x |
| n=10 000, k=3 | 0.099 ms | **0.071 ms** | 0.363 ms | 1.39x | 0.27x |
| n=100 000, k=3 | **0.960 ms** | 0.979 ms | 3.477 ms | 0.98x | 0.28x |
| n=1 000 000, k=3 | 174.814 ms | **146.325 ms** | 369.112 ms | 1.19x | 0.47x |
| n=100 000, k=8 | **7.050 ms** | 11.488 ms | 27.475 ms | 0.61x | 0.26x |
| n=1 000 000, k=8 | **80.880 ms** | 187.231 ms | 980.457 ms | 0.43x | 0.08x |

Same conclusion as the regressor's, for the same reason: numpy is already
vectorised here, the Rust arms start an Arrow round-trip in debt, and only at
n ≳ 10⁶ does the host arm's single pass repay it. The device arm loses
everywhere on this backend.

**`predict` under soft voting is deliberately NOT tabulated.** Its ladder was the
last to run and the co-tenant had by then pushed the box to loadavg 255; the
recorded cells are self-inconsistent (numpy reads 6.97 / 8.27 / 9.98 ms at
n=10⁵ for k=8 / 3 / 2, when the cost must *increase* with `k`), so the table
would be an artifact rather than a measurement. What is *not* in question is the
structural claim, which `crates/mlrs-backend/tests/voting_test.rs` gates directly:
the device arm fuses the argmax into the reduction, so it downloads `n` `u32`s
where numpy must materialise the whole `n × n_classes` average first. Whether
that pays on a given backend is still an open number.

### Should the default change?

`numpy` still ships for every aggregation, and for soft voting the ladder above
says so outright. Hard voting is the one place where the Rust arm is worth a
caller's `MLRS_VOTING_ENGINE=host` on its own: at n=10⁵ it turns a 167 ms
aggregation into a 1 ms one, and the win grows with `n`. It is not made the
default only because that would split the knob's meaning across the two `voting`
values — a caller who wants it can set it, and the ladder above is the argument.

## Measured: what `voting` costs a whole `predict`

`scripts/bench_voting_classifier.py --level call`, cpu backend, four linear-time
sklearn members. This is the ladder above with the members' own
`predict` / `predict_proba` put back in, which is what a caller actually pays.

The box was contended for this run and only the smallest row is clean enough to
quote (the n=10⁵ rows record the host arm as *slower* than numpy on the same
work, which is not physical). At n=10 000, d=32, k=4, `n_classes`=3:

| arm | `voting='hard'` | `voting='soft'` | soft/hard |
|---|---|---|---|
| `numpy` | 36.50 ms | 13.31 ms | **0.36x** |
| `host` | **11.84 ms** | 12.65 ms | 1.07x |

Two things fall out of that, and both matter more than the microbenchmark:

1. **On the default arm, `voting='hard'` — sklearn's own default — makes a whole
   `predict` 2.8x SLOWER than `'soft'`**, on the same members and the same data.
   Nothing about majority voting is intrinsically more expensive than averaging
   probabilities; the entire gap is sklearn's per-row Python loop.
2. **`MLRS_VOTING_ENGINE=host` removes it.** The hard call drops 3.1x
   (36.50 → 11.84 ms) and the two `voting` values land at parity (1.07x), which
   is what they should cost relative to each other. The aggregation was ~70% of
   that call; the ladder above says it is ~30% at n=10⁵.

So the arm is not a micro-optimisation for hard voting — it is the difference
between `voting` being a modelling choice and `voting` being a performance
choice.

## Measured: `n_jobs`

Not tabulated here, and deliberately so. Two reasons:

* **`voting` does not enter a fit at all** — every member is fitted identically
  either way — so this ladder is [voting.md](voting.md)'s, unchanged: a voting
  ensemble fits each member exactly **once**, so the ceiling is Amdahl's
  `total / slowest` over the members themselves, not `k`.
* **The fit ladder cannot be run on a contended box.** `--cpu-time`, this
  harness's remedy everywhere else, is *invalid* here: joblib's workers are
  separate processes and `time.process_time()` only sees the parent, so an
  `n_jobs=2` cell measures the parent's bookkeeping alone and renders as a
  128x-342x "speedup". The harness now **refuses** `--level fit --cpu-time`
  rather than print it, because a plausible-looking wrong number is worse than
  no number. Run it in wall clock on a quiet box.
