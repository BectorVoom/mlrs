# Upstream scikit-learn Issues

mlrs is gated against scikit-learn as its oracle, so a defect in scikit-learn is
also a decision point for mlrs: reproduce it, or diverge and say why. This
document is the record of those decisions.

**The rule.** mlrs matches scikit-learn's *semantics*, not its *bugs*. Where
scikit-learn's behaviour is a genuine property of the algorithm, mlrs reproduces
it exactly — including its errors. Where the behaviour is an oversight (a
validation gap, an inconsistency between code paths), mlrs implements what the
estimator means and records the divergence here.

**The invariant every divergence must preserve:** anything that runs under
scikit-learn must still run under mlrs, with the same results. Divergences may
only ever *widen* what is accepted, never narrow it. `SK-001` below is checked
against that invariant by
`test_oracle_cluster.py::test_hdbscan_brute_accepts_tree_only_metric_aliases`,
and the full accept/reject matrix is swept in the same file.

| ID | Component | Status | mlrs response |
| --- | --- | --- | --- |
| [SK-001](#sk-001) | `cluster.HDBSCAN` — `algorithm='brute'` rejects four metrics its own `metric` constraint accepts | Reported, sklearn 1.9.0 | Diverges: accepts `infinity` / `p` |

---

## SK-001

**`HDBSCAN`: `algorithm="brute"` rejects four metrics its own `metric`
constraint accepts, with an error that leaks `pairwise_distances`' parameter
list.**

Affects scikit-learn 1.9.0. Found while implementing mlrs's HDBSCAN parameter
surface and sweeping the full `algorithm` x `metric` matrix for parity
(HDBS-PARAMS).

### What mlrs does about it

mlrs **accepts** `algorithm='brute'` with `metric='infinity'` and `metric='p'`,
where scikit-learn raises.

Those two strings are tree-only spellings of `chebyshev` and `minkowski`. mlrs
resolves both to the same `Metric` enum value on every route
(`crates/mlrs-py/src/estimators/cluster.rs::parse_hdbscan_metric`), and its
`Algorithm` is value-neutral by construction — every route computes identical
core distances, gated as bit-identical equality rather than to a tolerance (see
`cluster::hdbscan::Algorithm`). So there is no route on which mlrs *cannot*
serve them, and refusing the pair would have meant inventing a restriction the
engine does not have.

The shim originally mirrored the rejection, on the reasoning that a drop-in
replacement should fail wherever scikit-learn fails. That was wrong, and is the
reason this document exists: parity is worth having with the estimator's
semantics, not with a gap in its validation.

mlrs **does** reproduce the neighbouring rejections — `kd_tree`/`ball_tree` with
`cosine`/`precomputed` — because those are real: a KD/ball box bound is only a
valid lower bound for a distance that aggregates monotonely over the feature
axes, which normalized cosine is not and a precomputed matrix has no axes for.
mlrs rejects that pair at `build()`, before any data is touched, where
scikit-learn rejects it at `fit`.

### The report

<sub>The remainder of this section is the issue body as submitted, kept verbatim
so it stays comparable with whatever upstream replies.</sub>

#### Describe the bug

`HDBSCAN`'s `metric` parameter constraint is the union of the tree metrics and
the pairwise metrics:

```python
# sklearn/cluster/_hdbscan/hdbscan.py
FAST_METRICS = set(KDTree.valid_metrics + BallTree.valid_metrics)          # L65
...
"metric": [StrOptions(FAST_METRICS | set(_VALID_METRICS) | {"precomputed"}),  # L651
           callable],
```

but `algorithm="brute"` can only serve the `_VALID_METRICS` half, because it
routes through `pairwise_distances`. Four metrics fall in the gap:

```python
>>> from sklearn.neighbors import KDTree, BallTree
>>> from sklearn.metrics.pairwise import _VALID_METRICS
>>> sorted(set(KDTree.valid_metrics + BallTree.valid_metrics) - set(_VALID_METRICS))
['infinity', 'p', 'pyfunc', 'sokalmichener']
```

For these, `HDBSCAN(algorithm="brute", metric=...)` raises
`InvalidParameterError` from *inside* `pairwise_distances`. Two things make this
worse than a plain "unsupported combination":

1. **The error names the wrong function and the wrong parameter set.** It reports
   "The `'metric'` parameter of **`pairwise_distances`**" and then lists
   `pairwise_distances`' metrics — a set that is neither the one `HDBSCAN`
   documents nor the one its own constraint accepts. A user who passed a metric
   that `HDBSCAN` explicitly validated as legal is told it is illegal, by a
   function they never called.

2. **The tree paths already do this correctly**, so the inconsistency is
   internal to one estimator. `algorithm="kd_tree"`/`"ball_tree"` validate the
   metric up front and raise a clear, actionable `ValueError` naming HDBSCAN's
   own parameter (L816-L822):

   ```
   ValueError: cosine is not a valid metric for a KDTree-based algorithm.
   Please select a different metric.
   ```

   `algorithm="brute"` has no equivalent check and falls through.

The result is that `algorithm=` silently changes which values of `metric=` are
legal, in a direction that is not documented and not discoverable from the
constraint. `algorithm="auto"` masks it entirely, because for all four of these
metrics `auto` routes to a tree (L849: `if issparse(X) or self.metric not in
FAST_METRICS: -> brute`), so the failure only appears once a user pins
`algorithm="brute"` — typically for a reason unrelated to the metric, such as
avoiding tree construction on small or high-dimensional data.

#### Steps/Code to Reproduce

```python
import numpy as np
from sklearn.cluster import HDBSCAN

X = np.random.default_rng(0).random((50, 3))

# 'infinity' is accepted by HDBSCAN's own `metric` constraint, and works on
# every algorithm except 'brute'.
HDBSCAN(algorithm="auto",      metric="infinity").fit(X)   # OK
HDBSCAN(algorithm="kd_tree",   metric="infinity").fit(X)   # OK
HDBSCAN(algorithm="ball_tree", metric="infinity").fit(X)   # OK
HDBSCAN(algorithm="brute",     metric="infinity").fit(X)   # InvalidParameterError
```

Full matrix (`p` behaves as `infinity`; `sokalmichener` and `pyfunc` are
BallTree-only, so `kd_tree` correctly rejects them with its clear message):

| metric | auto | brute | kd_tree | ball_tree |
|---|---|---|---|---|
| `infinity` | OK | **InvalidParameterError** | OK | OK |
| `p` | OK | **InvalidParameterError** | OK | OK |
| `sokalmichener` | OK | **InvalidParameterError** | ValueError (clear) | OK |
| `pyfunc` | OK | **InvalidParameterError** | ValueError (clear) | OK |

#### Expected Results

Either of the following would be consistent; I have no strong preference, though
(1) is the smaller change and matches what the tree paths already do:

1. **Validate in `fit`, symmetrically with the tree paths.** Alongside the
   existing `kd_tree`/`ball_tree` checks at L816-L822, add:

   ```python
   if self.algorithm == "brute" and self.metric not in _VALID_METRICS:
       raise ValueError(
           f"{self.metric} is not a valid metric for a brute-force algorithm. "
           f"Please select a different metric."
       )
   ```

   This yields the same shape of message users already get from the tree paths,
   and keeps the failure inside `HDBSCAN` rather than in a helper.

2. **Resolve the aliases before dispatching.** `infinity` and `p` are pure
   aliases of `chebyshev` and `minkowski` in the tree metric tables, so the
   brute path could normalize them and succeed rather than fail. That would fix
   two of the four cases and make `algorithm=` genuinely orthogonal to `metric=`
   for them; `sokalmichener`/`pyfunc` would still need option (1).

In both cases, documenting on the `metric` parameter that some accepted values
are algorithm-dependent would help — the docstring currently says only that
`metric` must be "one of the options allowed by
:func:`~sklearn.metrics.pairwise_distances`", which is itself inaccurate for the
tree-only metrics that the constraint accepts and the tree paths handle.

#### Actual Results

```python-traceback
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    HDBSCAN(algorithm="brute", metric="infinity").fit(X)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File ".../sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
  File ".../sklearn/cluster/_hdbscan/hdbscan.py", line 864, in fit
    self._single_linkage_tree_ = mst_func(**kwargs)
                                 ~~~~~~~~^^^^^^^^^^
  File ".../sklearn/cluster/_hdbscan/hdbscan.py", line 251, in _hdbscan_brute
    distance_matrix = pairwise_distances(
        X, metric=metric, n_jobs=n_jobs, **metric_params
    )
  File ".../sklearn/utils/_param_validation.py", line 208, in wrapper
    validate_parameter_constraints(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        parameter_constraints, params, caller_name=func.__qualname__
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File ".../sklearn/utils/_param_validation.py", line 98, in validate_parameter_constraints
    raise InvalidParameterError(
    ...<2 lines>...
    )
sklearn.utils._param_validation.InvalidParameterError: The 'metric' parameter of
pairwise_distances must be a str among {'nan_euclidean', 'chebyshev',
'mahalanobis', 'matching', 'correlation', 'minkowski', 'braycurtis', 'cosine',
'hamming', 'wminkowski', 'jaccard', 'sokalsneath', 'precomputed', 'manhattan',
'l1', 'seuclidean', 'rogerstanimoto', 'sqeuclidean', 'l2', 'russellrao', 'dice',
'cityblock', 'canberra', 'haversine', 'euclidean', 'yule'} or a callable.
Got 'infinity' instead.
```

#### Versions

```shell
System:
    python: 3.14.6 (main, Jun 11 2026, 00:00:00) [GCC 16.1.1 20260515 (Red Hat 16.1.1-2)]
executable: /usr/bin/python
   machine: Linux-7.1.5-201.fc44.x86_64-x86_64-with-glibc2.43

Python dependencies:
      sklearn: 1.9.0
          pip: 26.0.1
   setuptools: 83.0.0
        numpy: 2.4.6
        scipy: 1.18.0
       Cython: None
       pandas: 3.0.5
   matplotlib: 3.11.1
       joblib: 1.5.3
threadpoolctl: 3.6.0
     narwhals: 2.24.0

Built with OpenMP: True
```

---

## Adding an entry

1. Reproduce the defect from a clean interpreter and capture the real traceback
   — never a reconstructed one.
2. Establish that it *is* a defect rather than intended behaviour, by finding
   the inconsistency: a code path that disagrees with another, a constraint that
   disagrees with an implementation, or documentation that disagrees with both.
   "Surprising" is not enough.
3. Decide mlrs's response, and check it against the widening invariant at the
   top of this file.
4. Add the row to the table, write the section, and point the implementation's
   comment at the ID so a future reader can get from the code to the reasoning.
