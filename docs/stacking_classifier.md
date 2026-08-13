# `StackingClassifier` — stacked generalization for classification

`mlrs.StackingClassifier` implements the full
`sklearn.ensemble.StackingClassifier` parameter surface. Base classifiers
produce out-of-fold **responses** — probabilities, margins or labels, chosen by
`stack_method` — those become the columns of a meta-feature matrix, and a final
classifier is fitted on it.

```python
import mlrs

clf = mlrs.StackingClassifier(
    estimators=[("nb", mlrs.GaussianNB()), ("knn", mlrs.KNeighborsClassifier())],
    final_estimator=mlrs.LogisticRegression(),
    cv=5,
)
clf.fit(X, y).predict(X_test)
```

Members may be mlrs estimators, scikit-learn estimators, or a mix. A
**regressor** is also accepted as a base estimator — sklearn allows it for
ordinal problems, and mlrs matches that rather than tightening the rule.

This is the classifier twin of [`stacking.md`](stacking.md); everything the two
share (the `'drop'` sentinel, the name rules, `cv="prefit"`, the meta-matrix
arms) is documented there and only summarised here.

## Where the work happens

Stacking is a *composition*: the arithmetic already runs inside the composed
estimators. What the meta-estimator owns is structure, and that is in Rust.

| what | where |
|---|---|
| estimator-name validation | Rust (`stacking_validate_names`) |
| `'drop'` bookkeeping | Rust (`stacking_kept_indices`) |
| `cv="prefit"` classification | Rust (`stacking_cv_is_prefit`) |
| **`stack_method` validation** | **Rust (`stacking_stack_method`)** |
| **the `"auto"` fallback chain** | **Rust (`stacking_resolve_stack_methods`)** |
| **the dropped-column rule** | **Rust (`stacking_classifier_meta_slices`)** |
| meta-column layout / `_n_feature_outs` | Rust (`stacking_meta_layout`) |
| `get_feature_names_out` strings | Rust (`stacking_feature_names`) |
| fold index generation | Rust (`mlrs.model_selection`) |
| the meta-matrix copy | numpy by default; `host` / `device` arms available |
| base / final `fit` and response | the composed estimators |

The three bold rows are what the classifier adds over the regressor. They all
live in `mlrs_algos::ensemble::stacking` and are gated by
`crates/mlrs-algos/tests/stacking_test.rs`; the composition mechanics the two
classes share live on one mixin (`_StackComposition` in `mlrs/ensemble.py`), so
a rule has one definition rather than two that can drift.

## Parameters

| parameter | default | notes |
|---|---|---|
| `estimators` | — | `list of (str, estimator)`; an entry may be the string `'drop'`; a regressor member is legal |
| `final_estimator` | `None` | `None` means `sklearn.linear_model.LogisticRegression()` — see below |
| `cv` | `None` | int / splitter / iterable of index pairs / `"prefit"`; `None` is 5-fold `StratifiedKFold` |
| `stack_method` | `"auto"` | `"auto"` / `"predict_proba"` / `"decision_function"` / `"predict"` |
| `n_jobs` | `None` | joblib fan-out; ignored when a member holds a device handle |
| `passthrough` | `False` | append the original `X` columns to the meta features |
| `verbose` | `0` | forwarded to the inner `cross_val_predict` calls |

### `stack_method` — the parameter that makes this class different

`"auto"` is resolved **per member**, taking the first of `predict_proba`,
`decision_function`, `predict` that the member implements. A stack of a
`GaussianNB` and a `LinearSVC` therefore reports

```python
clf.stack_method_ == ["predict_proba", "decision_function"]
```

A named method every member must implement, or `fit` raises sklearn's
`Underlying estimator {name} does not implement the method {method}.` A
**dropped** entry is never asked, so a stack whose only proba-less member is
`'drop'` fits fine under `stack_method="predict_proba"`.

The method decides the meta matrix's width, and one case drops a column:

| method | binary `y` | `K`-class `y` |
|---|---|---|
| `predict_proba` | **1 column** (first dropped) | `K` columns |
| `decision_function` | 1 column (the response is 1-D) | `K` columns |
| `predict` | 1 column | 1 column |

The binary `predict_proba` drop is sklearn's rule and is not an optimisation:
`p(y=0) = 1 - p(y=1)`, so keeping both would hand the final estimator two
perfectly collinear features. mlrs states it once, in Rust
(`classifier_meta_slices`), together with the multilabel case — where a member
whose `predict_proba` returns one array per target contributes one meta block
per target, each with its first column dropped.

> **Caveat on mlrs members and `decision_function`.** `mlrs.LogisticRegression`
> exposes `predict_proba` but not `decision_function` (sklearn's exposes both),
> so `stack_method="decision_function"` over that member raises where sklearn
> would not. `mlrs.RidgeClassifier` and `mlrs.LinearSVC` do expose it. This is a
> gap in the member, not in the stack.

### `y` is label-encoded, and `classes_` is how it comes back

Every base estimator and the final estimator see `0..n_classes-1`; `predict`
maps back through `classes_` in the caller's own dtype (strings, booleans,
negative ints all round-trip). A multilabel-indicator `y` is encoded column by
column and `classes_` becomes a list of per-column arrays.

### The default `final_estimator` is sklearn's `LogisticRegression`

Same reasoning as the regressor's `RidgeCV` default: substituting
`mlrs.LogisticRegression` would silently move every default-constructed stack
off the sklearn baseline users migrate from, which is exactly the divergence the
1e-5 parity contract exists to prevent. sklearn is already a hard runtime
dependency. Pass `final_estimator=mlrs.LogisticRegression()` to put the meta fit
on the device.

### `n_jobs` is ignored when a member holds a device handle

Identical to the regressor — see [stacking.md](stacking.md#n_jobs-is-ignored-when-a-member-holds-a-device-handle).
The stack emits a `UserWarning` and fits serially. `n_jobs` works normally over
host (scikit-learn) members.

## The meta-matrix copy: numpy, host, or device

`MLRS_STACK_META_ENGINE` selects the arm exactly as it does for the regressor
(`numpy` — the default — / `host` / `device`); all three produce **bit-identical**
matrices, since the operation carries no arithmetic. The classifier is the
harder exercise of that scatter and is covered per arm in
`test_stacking_meta_engine.py`:

* its blocks are **multi-column** (`n_classes` per member under `predict_proba`),
  so a wrong row stride or a transposed block is visible where a stack of
  one-column regressor blocks could not show it;
* the binary `predict_proba` block handed to Rust is a **slice view**
  (`proba[:, 1:]`) — non-contiguous, non-zero offset — which is exactly the shape
  an ingress path can get wrong while passing on contiguous input.

## Parity

`crates/mlrs-py/python/tests/test_oracle_stacking_classifier.py` compares against
a live `sklearn.ensemble.StackingClassifier` — **81 cells, green on cpu, wgpu
and rocm with zero skips**. Compositions of sklearn members match **exactly**
(`atol=0`), including every rejection message; compositions of mlrs members match
within `conftest.live_atol()`.

`test_stacking_meta_engine.py` adds 38 classifier cells that re-run the four
`stack_method` values, both class counts and both `passthrough` settings **once
per meta arm** (106 cells in that file in total, green on all three backends).

`mlrs.StackingClassifier` also passes sklearn's `parametrize_with_checks` sweep —
**60 passed / 1 skipped**, with no estimator-specific xfails.

The three string-valued parameters get dedicated semantic coverage, not just
"mlrs == sklearn" (which would pass even if both silently ignored the string):

* `stack_method` — the resolved `stack_method_` list is asserted per member; the
  dropped binary probability column is pinned against the base estimator's own
  `predict_proba(X)[:, 1]`; and `predict` vs `predict_proba` is shown to change
  the model, not just the shape.
* `cv="prefit"` — members are reused rather than cloned, never refitted, and the
  meta features are full-training-set responses. As in the regressor, the
  difference is observable in `final_estimator_` (its coefficient, 5.63 vs 2.66
  on the suite's design) and **not** in `transform`.
* `estimators=[(name, "drop")]` — no fit, no meta column, no feature name, but
  the slot survives in `named_estimators_` as the literal `'drop'`.

### Landmine: sklearn's `StrOptions` message is not deterministic

`The 'stack_method' parameter of StackingClassifier must be a str among {…}`
renders its options by iterating a Python `set`, whose order for these strings
changes with `PYTHONHASHSEED` — two runs of the *same* sklearn call produce
different text. The oracle test therefore compares the message with the option
set parsed out; comparing it literally would be a coin flip. mlrs emits the
options in declaration order.

## Measured performance

`scripts/bench_stacking_classifier.py`, two members, min of N fresh subprocesses,
implementations interleaved. The one-time `_mlrs` extension load
(~90–160 ms/process) is reported separately rather than folded into the cells.

Members are deliberately **closed-form** — two `GaussianNB`s for the
probabilistic pair, two `RidgeClassifier`s for the margin pair, a
`RidgeClassifier` as the meta learner. A first draft used `LogisticRegression`
members and was unusable: the same configuration measured 2.26 s in one sweep
and 3.33 s in the next, because the member's convergence path varies run to run
and swamped every effect the parameters have. These ladders are about the
composition, so the members must not add noise of their own.

**Host arm** (scikit-learn members on BOTH sides — isolates mlrs's orchestration:
Rust `StratifiedKFold`, Rust method resolution, Rust meta-layout against
sklearn's Python equivalents, with every base fit identical).
`n=100000, d=32`, min of 5, loadavg 0.3:

| config | sklearn fit | mlrs fit | ratio | sklearn pred | mlrs pred |
|---|---|---|---|---|---|
| `cv=2` (binary) | 0.157 s | 0.187 s | 0.84x | 0.037 s | 0.038 s |
| `cv=3` (binary) | 0.194 s | 0.232 s | 0.84x | 0.038 s | 0.041 s |
| `cv=5` (binary) | 0.294 s | 0.328 s | 0.89x | 0.038 s | 0.039 s |
| `cv=10` (binary) | 0.531 s | 0.576 s | 0.92x | 0.038 s | 0.040 s |
| `cv="prefit"` (binary) | 0.050 s | 0.062 s | 0.81x | 0.040 s | 0.041 s |
| `stack_method="auto"` (5-class) | 0.351 s | 0.388 s | 0.90x | 0.086 s | 0.097 s |
| `stack_method="predict"` (5-class) | 0.308 s | 0.369 s | 0.83x | 0.084 s | 0.079 s |
| `stack_method="predict_proba"` (5-class) | 0.349 s | 0.387 s | 0.90x | 0.086 s | 0.092 s |
| `stack_method="decision_function"` (5-class) | 0.443 s | 0.485 s | 0.91x | 0.018 s | 0.013 s |
| `n_classes=2`, `predict_proba` | 0.294 s | 0.327 s | 0.90x | 0.037 s | 0.038 s |
| `n_classes=5`, `predict_proba` | 0.345 s | 0.388 s | 0.89x | 0.085 s | 0.092 s |
| `n_classes=10`, `predict_proba` | 0.413 s | 0.453 s | 0.91x | 0.145 s | 0.154 s |
| `passthrough=True` (5-class) | 0.367 s | 0.409 s | 0.90x | 0.109 s | 0.099 s |
| `n_jobs=2` (binary, cv=5) | 0.401 s | 0.400 s | 1.00x | 0.042 s | 0.038 s |
| `n_jobs=4` (binary, cv=5) | 0.414 s | 0.373 s | 1.11x | 0.040 s | 0.040 s |

Reading the ladder:

* **`cv` is the cost driver and is linear in the fold count** — 0.157 → 0.194 →
  0.294 → 0.531 s for `k = 2, 3, 5, 10`, about +47 ms per extra fold, exactly the
  `k + 1` base fits the design predicts. `cv="prefit"` costs 0.050 s because it
  performs no base fits at all. If a stack is too slow, `cv` is the first
  parameter to look at.

  The `"prefit"` *ratio* is about the members, not about stacking: it is ~6x
  cheaper than `cv=5` with the closed-form members used here and ~90x cheaper
  with [stacking.md](stacking.md)'s more expensive ones. Both are honest — what
  `"prefit"` removes is the base fits, so the ratio reports how expensive those
  were. The same caveat applies in reverse to `n_jobs` below.
* **`stack_method` moves `predict`, not `fit`.** Fit differs by under 10%
  between `predict` and `predict_proba`; *predict* differs by 4x across the
  `n_classes` ladder (0.037 → 0.085 → 0.145 s) because that is where the meta
  matrix's width lands — `k * n_classes` columns instead of `k`. The margin pair's
  much cheaper predict (0.013–0.018 s) is `RidgeClassifier`'s own
  `decision_function` being cheaper than `GaussianNB`'s `predict_proba`, not a
  property of the stack.
* **`passthrough` is nearly free at fit time** (+6%) and **+31% on predict** —
  the extra `n x d` copy, and the same on both implementations.
* **`n_jobs` LOSES here, and that is a real result, not noise**: 0.294 s serial →
  0.401 s at 2 → 0.414 s at 4, the same shape on both implementations. With
  members this cheap, joblib's dispatch costs more than a fold fit saves.
  **The parameter's sign flips with member cost**: the same curve read at a
  different point — [stacking.md](stacking.md)'s ladder, with expensive
  members — measures ~1.7x at 4 jobs. Neither number is a property of the
  stacking layer, so do not carry either one over to a stack of different
  members without re-measuring.
* **The orchestration layer is consistently ~10-18% behind sklearn's**
  (0.80–0.92x across every cell). That is mlrs's own cost, measured with every
  base fit held identical; cheap closed-form members expose it more than
  expensive ones do.

  **Root-caused, and it is not in the stacking layer.** A parallel n-sweep
  (5k → 400k, trivial members, `scripts/bench_stacking_overhead.py`) separated
  the two candidate shapes: on `cv="prefit"` the delta against sklearn bounces
  around zero and changes sign — the shared composition is at parity — while on
  `cv=5` it is flat at ~0.39 ms per 1000 samples across a 20x range of `n`.
  So the whole gap is **per-sample and inside
  `mlrs.model_selection.cross_val_predict`**, which every stack calls once per
  member. Profiling at n=200000 puts most of it in the SPLITTER's egress:
  `kfold_split` returns `Vec<(Vec<i64>, Vec<i64>)>`, i.e. ~1 million boxed
  `PyLong`s at k=5, and the `numpy.asarray` calls that turn them straight back
  into arrays cost more again. Handing index arrays across as Arrow buffers in
  both directions is the known fix; it is a `model_selection` change, not a
  stacking one, so nothing on this page depends on it.

**Device arm** (mlrs members in the mlrs stack vs scikit-learn members in the
sklearn stack — the end-to-end deployment comparison, dominated by the members).
rocm, gfx1151 iGPU, `n=100000, d=32`, min of 3, loadavg 0.3:

| config | sklearn fit | mlrs fit | ratio | sklearn pred | mlrs pred | ratio |
|---|---|---|---|---|---|---|
| `cv=2` (binary) | 0.159 s | 0.117 s | **1.36x** | 0.041 s | 0.119 s | 0.35x |
| `cv=3` (binary) | 0.198 s | 0.131 s | **1.50x** | 0.039 s | 0.125 s | 0.31x |
| `cv=5` (binary) | 0.290 s | 0.164 s | **1.77x** | 0.040 s | 0.128 s | 0.31x |
| `cv=10` (binary) | 0.531 s | 0.224 s | **2.37x** | 0.040 s | 0.126 s | 0.32x |
| `cv="prefit"` (binary) | 0.050 s | 0.082 s | 0.61x | 0.040 s | 0.080 s | 0.50x |
| `stack_method="auto"` (5-class, proba) | 0.347 s | 0.257 s | 1.35x | 0.083 s | 0.358 s | 0.23x |
| `stack_method="predict"` (5-class) | 0.306 s | 0.221 s | 1.39x | 0.081 s | 0.305 s | 0.26x |
| `stack_method="predict_proba"` (5-class) | 0.347 s | 0.259 s | 1.34x | 0.086 s | 0.341 s | 0.25x |
| `stack_method="decision_function"` (5-class) | 0.465 s | 0.221 s | **2.11x** | 0.018 s | 0.131 s | 0.13x |
| `n_classes=2`, `predict_proba` | 0.293 s | 0.161 s | **1.81x** | 0.040 s | 0.133 s | 0.30x |
| `n_classes=5`, `predict_proba` | 0.345 s | 0.256 s | 1.35x | 0.085 s | 0.342 s | 0.25x |
| `n_classes=10`, `predict_proba` | 0.407 s | 0.641 s | 0.63x | 0.184 s | 0.490 s | 0.38x |
| `passthrough=True` (5-class) | 0.380 s | 0.493 s | 0.77x | 0.111 s | 0.345 s | 0.32x |
| `n_jobs=2` (binary, cv=5) | 0.301 s | 0.183 s | 1.65x | 0.046 s | 0.140 s | 0.33x |
| `n_jobs=4` (binary, cv=5) | 0.410 s | 0.181 s | **2.27x** | 0.044 s | 0.142 s | 0.31x |

Reading the ladder:

* **The fit win grows with the fold count** — 1.36x → 1.50x → 1.77x → 2.37x for
  `k = 2, 3, 5, 10` — because mlrs's per-fit device overhead is fixed and more
  folds amortize it over more work. `cv="prefit"` is the same fact from the
  other end: with no base fits at all, only the fixed cost remains, so it reads
  0.61x.
* **`n_jobs` is flat on this arm** (0.178 / 0.183 / 0.181 s) — the documented
  serial fallback doing its job, not a regression. It reads as a *growing* win
  against sklearn only because sklearn's own `n_jobs` is losing on these cheap
  members.
* **Two cells invert, and it is a MEMBER, not the composition.** `n_classes=10`
  (0.63x) and `passthrough=True` (0.77x) both go the wrong way. Swapping only
  the final estimator — `mlrs.RidgeClassifier` → sklearn's, everything else
  held — moves the 10-class fit 0.4025 s → 0.4244 s, so the meta fit is not the
  cost. Timing one member at the same shape locates it exactly:

  | `n_classes` | sklearn fit | mlrs fit | ratio | sklearn `predict_proba` | mlrs `predict_proba` | ratio |
  |---|---|---|---|---|---|---|
  | 2 | 0.0161 s | 0.0021 s | **7.65x** | 0.0090 s | 0.0356 s | 0.25x |
  | 5 | 0.0161 s | 0.0016 s | **10.03x** | 0.0193 s | 0.0785 s | 0.25x |
  | 10 | 0.0164 s | 0.0016 s | **10.27x** | 0.0383 s | 0.1492 s | 0.26x |

  `mlrs.GaussianNB`'s **fit** is 7.6–10.3x faster than sklearn's and flat in
  `n_classes` — that is where the whole device win comes from. Its
  **`predict_proba`** is ~4x *slower* at every class count and grows linearly
  with `n_classes`. A stack under `stack_method="predict_proba"` pays that over
  all `n` rows per member for the out-of-fold responses and again on every
  `transform`, so at 10 classes two members contribute ~0.3 s to a fit that
  otherwise takes 0.4 s — enough to flip the cell. It is also the whole of the
  0.11–0.50x `predict` column.

  This is a member-level optimisation target (`predict_proba`, not `fit`), of the
  same kind meta-estimators are good at exposing: a stack calls a member many
  times, so a per-call gap that is invisible in a single fit becomes the
  headline. It is not a defect in the stacking layer, and nothing in this
  document's parity section depends on it.

  **FIXED (PERF-GNB-01).** `mlrs.GaussianNB.predict_proba` now runs at
  **1.6–4.0x of sklearn** rather than 0.25x — 3.6–8.0x faster than the numbers in
  the table above — after hoisting a per-element `ln` out of the
  joint-log-likelihood loop, halving the softmax's transcendental work, and
  moving the NB predict surface off the Python-list egress. The two inverted
  cells above therefore no longer describe the current tree; the *reasoning* is
  kept because the attribution technique (swap one component, then time a single
  member at the ladder's shape) is what found it.

> The margin-pair rows use `mlrs.RidgeClassifier`, whose dispatch is being
> reworked on a parallel branch (a measured host/device cost model in place of a
> constant floor). Treat those cells as a **pre-fix lower bound**; re-measure
> after that merges.

### The meta-matrix arms, in the classifier's shape

`scripts/bench_stacking_meta.py --level copy --cols 10` — ten columns per block,
i.e. a 10-class `predict_proba` stack, against the same harness's one-column
regressor shape. rocm gfx1151, f64, min of 5 fresh processes × 5 inner reps,
loadavg 0.8:

| copy | numpy | host | device | host/np | dev/np |
|---|---|---|---|---|---|
| n=1 000, k=2x10 | **0.007 ms** | 0.015 ms | 0.092 ms | 0.49x | 0.08x |
| n=10 000, k=2x10 | **0.062 ms** | 0.064 ms | 0.278 ms | 0.96x | 0.22x |
| n=100 000, k=2x10 | 0.681 ms | **0.672 ms** | 2.380 ms | **1.01x** | 0.29x |
| n=1 000 000, k=2x10 | **9.80 ms** | 11.96 ms | 28.21 ms | 0.82x | 0.35x |
| n=100 000, k=8x10 | **5.41 ms** | 6.41 ms | 11.10 ms | 0.84x | 0.49x |
| n=100 000, k=2x10, d=32 | 2.212 ms | **2.078 ms** | 7.780 ms | **1.06x** | 0.28x |
| n=1 000 000, k=2x10, d=32 | **27.2 ms** | 30.8 ms | 73.7 ms | 0.88x | 0.37x |
| n=100 000, k=2x10, d=128 | **6.74 ms** | 9.03 ms | 20.5 ms | 0.75x | 0.33x |

**Widening the blocks moves the host arm to parity.** On the regressor's
one-column ladder it runs at 0.08–0.74x of `np.hstack`
([stacking.md](stacking.md)); here it is 0.49–1.06x and it *beats* numpy in two
cells. That is the same explanation from the other side: the host arm's deficit
is a FIXED cost — the Arrow capsule crossing plus the egress copy — so ten times
the payload per crossing amortizes it away. The device arm improves for the same
reason (0.08–0.49x against 0.01–0.33x) but stays bus-bound, as a kernel with no
arithmetic to amortize its transfers must.

`numpy` still ships as the default: parity in two cells is not a reason to move,
and the numpy path is also the fallback for everything the Rust arms decline
(non-float blocks — which is every `stack_method="predict"` stack, since those
meta columns are encoded integer labels — duck-typed `X`, mismatched row counts).
But "the host arm loses" is only true at one column per member; at classifier
widths it is a wash.
