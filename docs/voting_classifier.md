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

**The tables below use two different clocks, and each says which.** The box was
co-tenanted with another session for part of this work — the condition this
project has twice recorded as capable of *inverting* a verdict
(`mlrs-cpu-bench-separate-processes`) — and it does not have one right answer:

* the **hard** ladder was taken under load (loadavg 100-260) and is reported in
  **CPU time**, the load-robust metric. It earned that trust: the same cell
  measured in two independent runs hours apart, under loadavg 176 and 224, came
  out at 155.10x and 156.70x. A wall-clock run taken between them produced one
  cell at 3 749x — a pure artifact — and is not reported.
* the **soft** ladders were re-run later on a **quiet box in wall clock**, which
  is the preferred metric when it is available. That re-run was not cosmetic: the
  contended data had the host arm *losing* at n=10⁵ where a quiet box shows it
  winning, and had soft `predict` at 0.24x where it is really 1.8-4.2x. Those
  tables were withheld until they could be measured properly.

CPU time is also **invalid** for one level of this harness — see `n_jobs` below.

Note that the ladder cannot simply be gated behind a "wait for a quiet box"
check: on the cpu backend the **device arm is itself the load**, since cubecl-cpu
runs one OS thread per unit and pushes `procs_running` past 260 on its own. The
cheap in-band check is the reported `_mlrs` load spread — 690-750 ms across a
whole run means nothing else was competing; 700 ms to 31 s means something was.

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
change the shape — **wall clock on a quiet box** (the `_mlrs` load varied only
690-746 ms across the whole run, which is the cheapest available proof that
nothing else was competing):

`predict_proba`, weighted — `np.average(probas, axis=0, weights=w)`:

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | **0.014 ms** | 0.017 ms | 0.140 ms | 0.82x | 0.10x |
| n=10 000, k=3 | **0.055 ms** | 0.074 ms | 0.486 ms | 0.74x | 0.11x |
| n=100 000, k=3 | 0.785 ms | **0.691 ms** | 3.089 ms | 1.14x | 0.25x |
| n=1 000 000, k=3 | 15.167 ms | **6.979 ms** | 34.972 ms | **2.17x** | 0.43x |
| n=100 000, k=2 | 0.500 ms | **0.449 ms** | 2.988 ms | 1.11x | 0.17x |
| n=100 000, k=8 | **2.384 ms** | 2.710 ms | 7.823 ms | 0.88x | 0.30x |
| n=1 000 000, k=8 | 34.684 ms | **19.907 ms** | 88.819 ms | **1.74x** | 0.39x |

The crossover is at roughly `n · k ≈ 3 · 10⁵` elements: below it numpy's already
vectorised `np.average` beats an Arrow round-trip, above it the host arm's single
pass repays the crossing. (An earlier contended run put the crossover a decade
higher and had the host arm *losing* at n=10⁵ — which is exactly the kind of
verdict shift this project has recorded before from a busy box, and the reason
this ladder was re-run rather than published as measured.)

`transform` (`flatten_transform=True`) — the `np.hstack` copy:

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | **0.010 ms** | 0.016 ms | 0.100 ms | 0.60x | 0.10x |
| n=10 000, k=3 | 0.085 ms | **0.070 ms** | 0.251 ms | 1.20x | 0.34x |
| n=100 000, k=3 | 0.898 ms | **0.754 ms** | 3.090 ms | 1.19x | 0.29x |
| n=1 000 000, k=3 | **12.929 ms** | 14.711 ms | 37.916 ms | 0.88x | 0.34x |
| n=100 000, k=2 | 0.557 ms | **0.504 ms** | 2.281 ms | 1.10x | 0.24x |
| n=100 000, k=8 | 4.459 ms | **3.705 ms** | 7.762 ms | 1.20x | 0.57x |
| n=1 000 000, k=8 | 57.408 ms | **53.729 ms** | 107.321 ms | 1.07x | 0.53x |

This is the copy-shaped half, and it behaves like one: the two host-side arms sit
within ±20% of each other at every size, because both are ultimately one pass of
`memcpy`-shaped work and neither has any arithmetic to be better at. That is
`docs/stacking.md`'s conclusion reproduced — a pure copy gives the Rust arm
nothing to amortise its Arrow round-trip against. The device arm loses
everywhere on this backend, as it does for stacking's meta-matrix.

### What the soft ladders add up to

`numpy` stays the default for soft voting, but not uniformly — the three
aggregations rank exactly as their arithmetic content predicts:

| aggregation | what it does | host arm at scale |
|---|---|---|
| `transform` | copy only | ~1.0-1.2x (a wash) |
| `predict_proba` | reduce `k → 1` | 1.7-2.2x |
| `predict` | reduce, then reduce again | **1.8-4.2x** |

The more a call reduces, the more the single-pass Rust arm is worth — which is
the same principle [voting.md](voting.md) used to predict that the regressor's
`predict` would fare better than its `transform`, now confirmed across a third
and fourth operation.

`predict`, weighted — `argmax(np.average(...), axis=1)`, and **the one soft
aggregation where the Rust arms win outright**:

| n, k | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=3 | **0.017 ms** | 0.019 ms | 0.164 ms | 0.93x | 0.10x |
| n=10 000, k=3 | 0.103 ms | **0.084 ms** | 0.396 ms | 1.23x | 0.26x |
| n=100 000, k=3 | 2.743 ms | **0.798 ms** | 3.314 ms | **3.44x** | 0.83x |
| n=1 000 000, k=3 | 26.120 ms | **8.301 ms** | 38.379 ms | **3.15x** | 0.68x |
| n=100 000, k=2 | 2.265 ms | **0.545 ms** | 2.958 ms | **4.15x** | 0.77x |
| n=100 000, k=8 | 4.950 ms | **2.714 ms** | 8.008 ms | 1.82x | 0.62x |
| n=1 000 000, k=8 | 45.285 ms | **21.021 ms** | 93.108 ms | **2.15x** | 0.49x |

Compare it against `predict_proba` directly above — same members, same weights,
same reduction, and the host arm goes from 1.1-2.2x to **1.8-4.2x**. The extra
margin is the argmax: numpy must materialise the whole `n × n_classes` average
before it can reduce it, while the host arm walks each row once. The `device`
arm shows the same effect from the other side — 0.49-0.83x here against
0.10-0.43x for `predict_proba`, because `vote_argmax_rows` consumes the average
on the device and downloads `n` `u32`s instead of `n · n_classes` floats. It
still does not overtake numpy on this backend, but the fusion is worth roughly a
3x improvement in its position, which is what that kernel exists to buy.

(This is the ladder an earlier contended run got completely wrong — it recorded
the host arm at 0.24x, i.e. losing by 4x, where a quiet box shows it winning by
2-4x. It was withheld rather than published; the numbers above are the re-run.)

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

`scripts/bench_voting_classifier.py --level fit`, **wall clock on a quiet box**,
min of 3. `voting` does not enter a fit at all — every member is fitted
identically either way — so this is [voting.md](voting.md)'s ladder, and it
reproduces its conclusion.

**This needs two member pools, and reporting only one of them would have been
wrong.** A voting ensemble fits each member exactly **once**, so there are only
`k` units of work to spread and the speedup ceiling is Amdahl's
`total / slowest` over the members themselves — not `k`.

**Mixed pool** (the default: members of wildly different cost, which is what an
ensemble usually looks like):

| config | `n_jobs=None` | `n_jobs=2` | `n_jobs=4` | best/serial |
|---|---|---|---|---|
| n=10 000, d=32, k=4 | **201.4 ms** | 293.9 ms | 273.8 ms | 1.00x |
| n=100 000, d=32, k=4 | 2 579.6 ms | 2 225.3 ms | **2 182.2 ms** | 1.18x |
| n=100 000, d=128, k=4 | 9 588.8 ms | 8 466.1 ms | **8 268.0 ms** | 1.16x |

**Balanced pool** (`--balanced`: four depth-8 trees at different seeds, so the
ceiling really is `k` = 4):

| config | `n_jobs=None` | `n_jobs=2` | `n_jobs=4` | best/serial |
|---|---|---|---|---|
| n=10 000, d=32, k=4 | 631.2 ms | 434.4 ms | **323.9 ms** | **1.95x** |
| n=100 000, d=32, k=4 | 7 853.5 ms | 4 952.7 ms | **2 694.2 ms** | **2.91x** |
| n=100 000, d=128, k=4 | 31 294.8 ms | 16 497.8 ms | **9 852.0 ms** | **3.18x** |

Read together, those say something the mixed pool alone would have got wrong.
On the mixed pool `n_jobs` looks nearly useless (1.16-1.18x, and an outright
**loss** at n=10 000 where process spawn costs more than the ~200 ms of work it
splits). On the balanced pool the identical machinery delivers **3.18x of a 4x
ceiling**. The fan-out is not weak — one member dominates. A ladder run only on
the mixed pool would have reported "`n_jobs` does nothing" and hidden *why*,
which is why the harness carries `--balanced` at all.

### `--cpu-time` is refused at this level, on purpose

This is the one ladder where the contended-box remedy cannot be applied.
`time.process_time()` measures the calling process, joblib runs the fan-out in
forked workers, and the parent therefore sees only its own bookkeeping — so a
CPU-time `n_jobs=2` cell renders as a **128x-342x "speedup"**. Those were the
numbers an earlier run of this harness actually produced. `--level fit
--cpu-time` now errors out with that explanation rather than printing them,
because a plausible-looking wrong number is worse than no number.
