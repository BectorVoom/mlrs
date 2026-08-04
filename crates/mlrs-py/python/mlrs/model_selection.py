"""``mlrs.model_selection`` — sklearn-faithful splitters + search (MODSEL-01/02).

============================  =============================================
name                          provenance
============================  =============================================
:func:`train_test_split`      implemented here
:class:`KFold`                implemented here
:class:`StratifiedKFold`      implemented here
``GridSearchCV``              re-exported from sklearn (see the passthrough
``RandomizedSearchCV``        section at the foot of this module for why)
``cross_val_score``
``cross_validate``
============================  =============================================

The three splitters carry the **complete** sklearn parameter surface and work
over numpy / pandas / polars / pyarrow / python-list / scipy-sparse containers.

None of this delegates to the compiled ``_mlrs`` extension — it is pure host
index bookkeeping plus a container gather. There is no per-element float
arithmetic to move onto a device, and the wall cost is dominated by
pandas/polars/arrow's own gather, so importing this module (and calling it)
works on a tree where ``_mlrs`` was never built. Like :mod:`mlrs.metrics` these
are plain functions/classes, NOT :class:`~mlrs.base.MlrsBase` subclasses.

## MT19937-host-match decision (REQUIREMENTS MODSEL-01)

MODSEL-01 requires that the ``shuffle=True`` reproducibility story be recorded
as either *bit-for-bit vs sklearn* or *property-gated*. **mlrs chooses
bit-for-bit, for every splitter here.**

``random_state`` is resolved through ``sklearn.utils.check_random_state`` into a
legacy ``numpy.random.RandomState`` (MT19937), and the draws are issued in
exactly the order sklearn issues them:

* :func:`train_test_split` — ``ShuffleSplit._iter_indices`` /
  ``StratifiedShuffleSplit._iter_indices``;
* :class:`KFold` — one in-place ``rng.shuffle(indices)`` (NOT
  ``rng.permutation``; they consume the stream differently);
* :class:`StratifiedKFold` — one ``rng.shuffle`` per class, in class order,
  over that class's fold-label block.

Consequently every splitter returns the *same rows* as its sklearn counterpart
for the same arguments — index-for-index, not merely distributionally. That is
what the test file asserts, directly against a live sklearn rather than against
a stored fixture.

The alternative (a SplitMix64 device/host RNG like ``tsne.rs``'s
``init='random'`` seed) was rejected: it would buy nothing — the split is O(n)
integer work — while silently changing every user's train/test rows relative to
the sklearn baseline they are migrating from.

## Container support (D-03 "egress mirrors ingress")

Each input array is gathered with its own native row-take and comes back as the
**same container type it went in as**:

===========================  =========================================
input                        gather used
===========================  =========================================
``numpy.ndarray``            ``X[indices]``
``pandas`` DataFrame/Series  ``X.take(indices, axis=0)`` (positional)
``polars`` DataFrame         ``X[indices]``
``polars`` Series            ``X.gather(indices)``
``pyarrow`` Table/RecordBatch/Array/ChunkedArray
                             ``X.take(indices)``
``scipy.sparse`` matrix      ``X[indices]`` (after ``.tocsr()``)
``list`` / ``tuple`` / ``range`` / other sequence
                             ``[X[i] for i in indices]``
===========================  =========================================

The pandas row is reached by DUCK-TYPING on ``.iloc``, not by
``isinstance``, so a pandas-API frame that is not a pandas instance — modin,
cudf, a test double — takes the same positional ``.take`` path it does under
sklearn. That distinction is load-bearing rather than cosmetic: the generic
fallback below is ``X[indices]``, which on a dataframe means COLUMN selection
by label, so an ``isinstance`` gate would silently gather such a frame along
the wrong axis instead of failing.

polars/pyarrow/scipy are *never imported* by this module — they are detected
through ``sys.modules``, so the check cannot pull in a library the user does not
have installed. (pandas is generally already in ``sys.modules`` regardless,
since ``sklearn.utils`` imports it; this module still never imports it itself.)
A container mlrs has not been taught about falls through to the generic
``X[indices]`` (anything exposing ``.shape``) or the python-sequence path.

.. note::
   Row-take is POSITIONAL everywhere, matching sklearn: a pandas object keeps
   its original (now shuffled) index labels rather than being re-indexed
   ``0..n-1``, and a polars frame — which has no index — simply keeps its rows.

.. note::
   ``pandas.Index`` is supported here but NOT by sklearn, which raises
   ``TypeError`` on it (its generic branch evaluates ``X[indices, ...]``).
   ``pyarrow.RecordBatch`` likewise comes back as a ``RecordBatch``, where
   sklearn degrades it to a ``StructArray``. Both are deliberate improvements
   on the parity baseline — they turn an error / a lossy conversion into the
   container the caller passed in, and neither can change which ROWS are
   selected.
"""

import numbers
import sys
import warnings
from itertools import chain
from math import ceil, floor

import numpy as np
from sklearn.utils import check_array, check_random_state

__all__ = [
    # splitters implemented here (MODSEL-01)
    "train_test_split",
    "KFold",
    "StratifiedKFold",
    # sklearn-delegation passthrough (MODSEL-02) — see the section comment
    "GridSearchCV",
    "RandomizedSearchCV",
    "cross_val_score",
    "cross_validate",
    "InvalidParameterError",
]

# sklearn's `train_test_split` default when NEITHER test_size nor train_size is
# given (`_validate_shuffle_split(..., default_test_size=0.25)`).
_DEFAULT_TEST_SIZE = 0.25


class InvalidParameterError(ValueError, TypeError):
    """A parameter did not satisfy its declared constraint.

    Mirrors ``sklearn.utils._param_validation.InvalidParameterError``, including
    its double base — sklearn raises an exception that is BOTH a ``ValueError``
    and a ``TypeError`` for a constraint violation, so migrating code guarding
    ``train_test_split`` with either ``except ValueError`` or ``except
    TypeError`` keeps catching it. Raising a plain ``ValueError`` here would
    silently break the ``except TypeError`` callers.

    sklearn splits its two error classes by *what* was wrong, and mlrs follows:

    * a value that violates the parameter's declared TYPE/RANGE constraint
      (``test_size=0``, ``test_size='half'``, ``random_state=-3``) raises this,
      before the data is looked at;
    * a value that is well-formed but wrong FOR THIS DATA
      (``test_size=500`` on 100 rows, float sizes summing above 1) raises a
      plain ``ValueError`` from :func:`_validate_shuffle_split`.
    """


# --------------------------------------------------------------------------- #
# parameter constraints (sklearn's `@validate_params` decorator, inlined)
# --------------------------------------------------------------------------- #


def _check_size_param(name, value):
    """``test_size`` / ``train_size`` must be ``None``, an int >= 1, or a float
    in ``(0, 1)`` — sklearn's ``Interval`` constraint pair.

    The float arm tests ``isinstance(value, float)`` rather than
    ``numbers.Real``, matching sklearn's ``RealNotInt`` (which only registers
    ``float``). That is not pedantry: it makes ``np.float32(0.25)`` a rejected
    value on both sides instead of one that quietly rounds to a different
    ``ceil``/``floor`` boundary than the ``float`` the caller wrote.
    """
    if value is None:
        return
    if isinstance(value, numbers.Integral) and value >= 1:
        return
    if isinstance(value, float) and 0.0 < value < 1.0:
        return
    raise InvalidParameterError(
        f"The {name!r} parameter of train_test_split must be a float in the "
        f"range (0.0, 1.0), an int in the range [1, inf) or None. "
        f"Got {value!r} instead."
    )


def _check_random_state_param(random_state):
    """``random_state`` must be ``None``, an int in ``[0, 2**32 - 1]``, or a
    ``numpy.random.RandomState`` — sklearn's ``"random_state"`` constraint."""
    if random_state is None or isinstance(random_state, np.random.RandomState):
        return
    if isinstance(random_state, numbers.Integral) and 0 <= random_state <= 2**32 - 1:
        return
    raise InvalidParameterError(
        "The 'random_state' parameter of train_test_split must be an int in "
        "the range [0, 4294967295], an instance of "
        f"'numpy.random.mtrand.RandomState' or None. Got {random_state!r} instead."
    )


def _check_stratify_param(stratify):
    """``stratify`` must be array-like or ``None`` — sklearn's ``"array-like"``
    constraint.

    Without this gate a scalar ``stratify`` would fall through to
    ``_num_samples``, which raises a bare ``TypeError`` — the mirror image of
    the problem :class:`InvalidParameterError` exists to avoid, since an
    ``except ValueError`` caller would then miss it.

    ``np.isscalar`` is what excludes a ``str`` (which does have ``__len__``),
    matching sklearn's ``_is_arraylike_not_scalar``.
    """
    if stratify is None:
        return
    arraylike = (
        hasattr(stratify, "__len__")
        or hasattr(stratify, "shape")
        or hasattr(stratify, "__array__")
    )
    if arraylike and not np.isscalar(stratify):
        return
    raise InvalidParameterError(
        "The 'stratify' parameter of train_test_split must be an array-like "
        f"or None. Got {stratify!r} instead."
    )


def _check_shuffle_param(shuffle):
    """``shuffle`` must be a ``bool`` or ``numpy.bool_``."""
    if isinstance(shuffle, (bool, np.bool_)):
        return
    raise InvalidParameterError(
        "The 'shuffle' parameter of train_test_split must be an instance of "
        f"'bool' or an instance of 'numpy.bool'. Got {shuffle!r} instead."
    )


# --------------------------------------------------------------------------- #
# container detection — sys.modules only, never an import
# --------------------------------------------------------------------------- #


def _is_pandas(X):
    """``X`` is a pandas DataFrame / Series / Index (pandas already imported)."""
    pd = sys.modules.get("pandas")
    if pd is None:
        return False
    return isinstance(X, (pd.DataFrame, pd.Series, pd.Index))


def _is_polars(X):
    """``X`` is a polars DataFrame / Series (polars already imported)."""
    pl = sys.modules.get("polars")
    if pl is None:
        return False
    return isinstance(X, (pl.DataFrame, pl.Series))


def _is_pyarrow(X):
    """``X`` is a pyarrow Table / RecordBatch / Array / ChunkedArray."""
    pa = sys.modules.get("pyarrow")
    if pa is None:
        return False
    return isinstance(X, (pa.Table, pa.RecordBatch, pa.Array, pa.ChunkedArray))


def _is_sparse(X):
    """``X`` is a scipy sparse matrix/array (scipy.sparse already imported).

    ``scipy.sparse`` is imported transitively by sklearn, so the module lookup
    effectively always succeeds in-process; the guard is for the (hypothetical)
    scipy-free install.
    """
    sp = sys.modules.get("scipy.sparse")
    if sp is None:
        return False
    return bool(sp.issparse(X))


# --------------------------------------------------------------------------- #
# length + indexability (sklearn `_num_samples` / `indexable` semantics)
# --------------------------------------------------------------------------- #


def _num_samples(x):
    """Number of rows in ``x`` — sklearn ``utils.validation._num_samples``.

    Prefers ``shape[0]`` (numpy / pandas / polars / pyarrow Table+RecordBatch /
    sparse) and falls back to ``len`` (pyarrow Array + ChunkedArray, python
    sequences). A 0-d array and a bare estimator are both rejected, matching
    sklearn's messages so a mis-call reads the same as it would there.
    """
    if hasattr(x, "fit") and callable(x.fit):
        # an estimator was passed by mistake
        raise TypeError(f"Expected sequence or array-like, got estimator {x}")
    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(f"Expected sequence or array-like, got {type(x)}")
    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                f"Singleton array {x!r} cannot be considered a valid collection."
            )
        if isinstance(x.shape[0], numbers.Integral):
            return int(x.shape[0])
    try:
        return len(x)
    except TypeError as exc:
        raise TypeError(f"Expected sequence or array-like, got {type(x)}") from exc


def _make_indexable(X):
    """Coerce one input into something row-indexable (sklearn ``_make_indexable``).

    ``None`` passes through (and is gathered back out as ``None``); a sparse
    matrix is converted to CSR (COO/DOK cannot be row-sliced); anything already
    supporting ``__getitem__`` or ``.iloc`` — which covers every container in
    the table above, plus ``range`` and duck-typed dataframes — is left
    untouched; anything else is materialized with ``np.array``.

    The predicate is ``__getitem__ or iloc``, matching sklearn, and NOT
    ``__getitem__ or __array__``: an object exposing only ``__array__`` (no
    ``__getitem__``) is not row-indexable, so it has to be materialized here.
    Admitting it on the strength of ``__array__`` alone would send it to
    :func:`_safe_indexing` un-materialized, where every branch then fails.
    """
    if X is None:
        return None
    if _is_sparse(X):
        return X.tocsr()
    if hasattr(X, "__getitem__") or hasattr(X, "iloc"):
        return X
    return np.array(X)


def _check_consistent_length(*arrays):
    """Raise unless every non-``None`` input has the same number of rows."""
    lengths = [_num_samples(a) for a in arrays if a is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r"
            % [int(length) for length in lengths]
        )


def _indexable(*arrays):
    """``_make_indexable`` every input, then length-check them together."""
    result = [_make_indexable(X) for X in arrays]
    _check_consistent_length(*result)
    return result


# --------------------------------------------------------------------------- #
# container-aware positional row gather
# --------------------------------------------------------------------------- #


def _safe_indexing(X, indices):
    """Positional row-take of ``X`` at integer ``indices``, container-preserving.

    The axis-0 / integer-key subset of sklearn's private ``_safe_indexing`` that
    ``train_test_split`` actually needs, with polars handled natively instead of
    through sklearn's narwhals dependency (mlrs does not depend on narwhals).
    See the module docstring's container table.
    """
    if X is None:
        return None

    indices = np.asarray(indices)
    if indices.dtype.kind not in ("i", "u"):
        # every caller in this module builds integer index arrays; a non-integer
        # key here means an internal invariant broke, not user error.
        raise TypeError(
            f"mlrs.model_selection: row indices must be integers, got dtype "
            f"{indices.dtype!r}"
        )

    if hasattr(X, "iloc") or _is_pandas(X):
        # DUCK-TYPED, not isinstance: `.iloc` is sklearn's own pandas probe, and
        # it is what admits pandas-API frames that are not pandas instances
        # (modin, cudf, test doubles like sklearn's `MockDataFrame`). Keying on
        # `isinstance(X, pd.DataFrame)` instead would drop those through to the
        # generic `X[indices]` branch below — which on a dataframe is COLUMN
        # selection by label, so a frame with integer column labels and at least
        # as many columns as rows would be silently gathered along the wrong
        # axis and returned with the train and test sets fully overlapping.
        #
        # `_is_pandas` still runs, for `pd.Index` — which has no `.iloc` but
        # does have a positional `.take` (and which sklearn itself fails on:
        # its generic branch does `X[indices, ...]`, raising TypeError).
        #
        # `.take` is POSITIONAL and returns a proper copy (no
        # SettingWithCopyWarning), unlike `.iloc[...]` — same choice sklearn makes.
        return X.take(indices, axis=0)

    if _is_polars(X):
        pl = sys.modules["polars"]
        if isinstance(X, pl.Series):
            return X.gather(indices)
        # polars DataFrame has no `.take`; `df[int_sequence]` IS the row gather.
        return X[indices]

    if _is_pyarrow(X):
        # Table / RecordBatch / Array / ChunkedArray all expose `.take`, and each
        # returns its own type — so the pyarrow container is mirrored exactly.
        return X.take(indices)

    if _is_sparse(X):
        return X[indices]

    if hasattr(X, "shape"):
        return X[indices]

    # python list / tuple / range / any other plain sequence -> list (sklearn's
    # `_list_indexing` behavior, including for `range`).
    return [X[i] for i in indices]


# --------------------------------------------------------------------------- #
# train/test size resolution (sklearn `_validate_shuffle_split`, verbatim rules)
# --------------------------------------------------------------------------- #


def _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=None):
    """Resolve ``(n_train, n_test)`` from the size parameters.

    Mirrors ``sklearn.model_selection._split._validate_shuffle_split`` rule for
    rule, including the asymmetric rounding that makes a float split
    reproducible against sklearn: the TEST count is ``ceil``-ed and the TRAIN
    count is ``floor``-ed.

    A ``float`` size is a fraction in the open interval ``(0, 1)``; an ``int``
    size is an absolute count in ``[1, n_samples)``. Omitting one makes it the
    complement of the other; omitting both applies ``default_test_size``.
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (test_size_type == "i" and (test_size >= n_samples or test_size <= 0)) or (
        test_size_type == "f" and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            f"test_size={test_size} should be either positive and smaller"
            f" than the number of samples {n_samples} or a float in the "
            "(0, 1) range"
        )

    if (train_size_type == "i" and (train_size >= n_samples or train_size <= 0)) or (
        train_size_type == "f" and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            f"train_size={train_size} should be either positive and smaller"
            f" than the number of samples {n_samples} or a float in the "
            "(0, 1) range"
        )

    if train_size is not None and train_size_type not in ("i", "f"):
        raise ValueError(f"Invalid value for train_size: {train_size}")
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError(f"Invalid value for test_size: {test_size}")

    if train_size_type == "f" and test_size_type == "f" and train_size + test_size > 1:
        raise ValueError(
            f"The sum of test_size and train_size = {train_size + test_size}, "
            "should be in the (0, 1) range. Reduce test_size and/or train_size."
        )

    if test_size_type == "f":
        n_test = ceil(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    if train_size_type == "f":
        n_train = floor(train_size * n_samples)
    elif train_size_type == "i":
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        # `int(...)`, because sklearn formats this one with `%d` while n_train /
        # n_test are still floats — so the message reads "= 150", not "= 150.0".
        # Every OTHER message in this function uses `{}`/`.format`, which does
        # render the float. Verbatim-compared in
        # `test_error_data_relative_messages_match_sklearn_verbatim`.
        raise ValueError(
            f"The sum of train_size and test_size = {int(n_train + n_test)}, "
            "should be smaller than the number of "
            f"samples {n_samples}. Reduce test_size and/or "
            "train_size."
        )

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            f"With n_samples={n_samples}, test_size={test_size} and "
            f"train_size={train_size}, the resulting train set will be empty. "
            "Adjust any of the aforementioned parameters."
        )

    return n_train, n_test


# --------------------------------------------------------------------------- #
# index generation — MT19937 draw order matched to sklearn (see module docstring)
# --------------------------------------------------------------------------- #


def _shuffle_split_indices(n_samples, n_train, n_test, random_state):
    """Unstratified shuffle split — ``ShuffleSplit._iter_indices``' draw order.

    ONE ``rng.permutation(n_samples)``; TEST takes the leading ``n_test``
    entries and TRAIN the following ``n_train``. The test-first slicing is not
    cosmetic — it is what makes the indices identical to sklearn's when
    ``n_train + n_test < n_samples`` (a sub-sampling split).
    """
    rng = check_random_state(random_state)
    permutation = rng.permutation(n_samples)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test : n_test + n_train]
    return ind_train, ind_test


def _approximate_mode(class_counts, n_draws, rng):
    """Per-class draw counts — ``sklearn.utils._indexing._approximate_mode``.

    Approximate mode of the multivariate hypergeometric given by
    ``class_counts`` and ``n_draws``: take each class's exact proportional share
    floored, then hand out the leftover draws by descending fractional
    remainder, breaking ties within one remainder value RANDOMLY (``rng.choice``
    without replacement) so repeated ties do not bias the same classes.

    That tie-break draw is part of the sklearn RNG stream, which is why this is
    reimplemented here rather than approximated — skipping it would desync every
    subsequent permutation.
    """
    rng = check_random_state(rng)
    continuous = class_counts / class_counts.sum() * n_draws
    floored = np.floor(continuous)
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        for value in values:
            (inds,) = np.where(remainder == value)
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(int)


def _stratified_shuffle_split_indices(y, n_train, n_test, random_state):
    """Stratified shuffle split — ``StratifiedShuffleSplit._iter_indices``.

    Class shares are allocated with :func:`_approximate_mode` (train first, then
    test from what remains), each class's members are permuted independently,
    and finally the assembled train and test index lists are each permuted once
    more. The draw order — ``n_i``, ``t_i``, per-class permutations in sorted
    class order, train permutation, test permutation — is the sklearn order.

    A 2-D ``y`` is the multilabel case: each row collapses to a
    space-joined string so a distinct label COMBINATION becomes one stratum
    (sklearn's own encoding, ``" ".join(row.astype("str"))``).
    """
    y = check_array(y, input_name="y", ensure_2d=False, dtype=None)

    if y.ndim == 2:
        # multilabel: one stratum per distinct label-set row
        y = np.array([" ".join(row.astype("str")) for row in y])

    classes, y_indices, class_counts = np.unique(
        y, return_inverse=True, return_counts=True
    )
    y_indices = y_indices.reshape(-1)
    n_classes = classes.shape[0]

    if np.min(class_counts) < 2:
        too_few_classes = classes[class_counts < 2].tolist()
        raise ValueError(
            "The least populated classes in y have only 1"
            " member, which is too few. The minimum"
            " number of groups for any class cannot"
            " be less than 2. Classes with too few"
            " members are: %s" % (too_few_classes,)
        )
    if n_train < n_classes:
        raise ValueError(
            "The train_size = %d should be greater or "
            "equal to the number of classes = %d" % (n_train, n_classes)
        )
    if n_test < n_classes:
        raise ValueError(
            "The test_size = %d should be greater or "
            "equal to the number of classes = %d" % (n_test, n_classes)
        )

    # sorted row positions of each class, in class order (stable sort keeps the
    # within-class order deterministic before the per-class permutation).
    class_indices = np.split(
        np.argsort(y_indices, kind="stable"), np.cumsum(class_counts)[:-1]
    )

    rng = check_random_state(random_state)

    n_i = _approximate_mode(class_counts, n_train, rng)
    class_counts_remaining = class_counts - n_i
    t_i = _approximate_mode(class_counts_remaining, n_test, rng)

    train = []
    test = []
    for i in range(n_classes):
        permutation = rng.permutation(class_counts[i])
        perm_indices_class_i = class_indices[i].take(permutation, mode="clip")
        train.extend(perm_indices_class_i[: n_i[i]])
        test.extend(perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]])

    return rng.permutation(train), rng.permutation(test)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """Split arrays or matrices into random train and test subsets.

    Signature- and result-compatible with
    ``sklearn.model_selection.train_test_split``: for the same ``random_state``
    the selected ROWS are identical, not merely similarly distributed (see the
    module docstring's MT19937-host-match note).

    Parameters
    ----------
    *arrays : sequence of indexables with the same first dimension
        numpy arrays, pandas DataFrames/Series, polars DataFrames/Series,
        pyarrow Tables/RecordBatches/Arrays/ChunkedArrays, scipy-sparse
        matrices, and plain python sequences may be mixed freely in one call;
        each is returned as its own container type. ``None`` entries pass
        through as ``None``. At least one array is required.

    test_size : float or int, default=None
        ``float`` in ``(0, 1)`` — the fraction of rows in the test split
        (rounded UP). ``int`` — the absolute test-row count. ``None`` — the
        complement of ``train_size``, or ``0.25`` if ``train_size`` is also
        ``None``.

    train_size : float or int, default=None
        ``float`` in ``(0, 1)`` — the fraction of rows in the train split
        (rounded DOWN). ``int`` — the absolute train-row count. ``None`` — the
        complement of ``test_size``.

        Giving both a ``train_size`` and a ``test_size`` that do not sum to the
        full data is legal and produces a SUB-SAMPLED split: the unallocated
        rows appear in neither output.

    random_state : int, ``numpy.random.RandomState`` instance or None, default=None
        Seeds the shuffle. Pass an int for a reproducible split; ``None`` draws
        from the global numpy random state (so successive calls differ).

    shuffle : bool, default=True
        When ``False`` the split is the contiguous head/tail of the data —
        ``train = rows[:n_train]``, ``test = rows[n_train:n_train + n_test]`` —
        and ``random_state`` is unused. ``stratify`` is not permitted.

        Tested by IDENTITY against ``False``, as sklearn does: ``numpy.False_``
        is not the ``False`` singleton, so it SHUFFLES. Inherited deliberately
        so the split matches sklearn's (see the ``shuffle is False`` comment in
        the body).

    stratify : array-like, default=None
        Class labels to stratify on, one per row (typically ``y``). The class
        proportions of the full data are preserved in both splits as closely as
        integer counts allow. A 2-D ``stratify`` is treated as multilabel: each
        distinct label-set row is its own stratum. Every class needs at least 2
        members, and both splits must be at least as large as the class count.

    Returns
    -------
    splitting : list, length ``2 * len(arrays)``
        ``[a0_train, a0_test, a1_train, a1_test, ...]`` — the train and test
        parts of each input, in input order.

    Examples
    --------
    >>> import numpy as np
    >>> from mlrs.model_selection import train_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), np.arange(5)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.33, random_state=42)
    >>> X_train
    array([[4, 5],
           [0, 1],
           [6, 7]])
    >>> y_test
    array([1, 4])

    A polars frame and a numpy target split together, each coming back in its
    own container:

    >>> import polars as pl                                # doctest: +SKIP
    >>> df = pl.DataFrame({"a": range(100), "b": range(100)})   # doctest: +SKIP
    >>> df_tr, df_te, y_tr, y_te = train_test_split(        # doctest: +SKIP
    ...     df, np.arange(100), test_size=0.2, random_state=0, stratify=None)
    """
    # Parameter constraints run FIRST, before the "no arrays" check and before
    # the data is touched at all — sklearn's `@validate_params` decorator runs
    # ahead of the function body, so `train_test_split(test_size=0)` reports the
    # bad `test_size`, not the missing arrays.
    _check_size_param("test_size", test_size)
    _check_size_param("train_size", train_size)
    _check_random_state_param(random_state)
    _check_shuffle_param(shuffle)
    _check_stratify_param(stratify)

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = _indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=_DEFAULT_TEST_SIZE
    )

    # `is False`, not `not shuffle` — IDENTITY, deliberately, because that is
    # what sklearn tests. The observable consequence is that `np.False_` (which
    # is not the `False` singleton) takes the SHUFFLING branch on both sides:
    # `shuffle=np.any(mask)` shuffles even when the mask is all-False. That is
    # surprising, but a bit-for-bit contract means inheriting sklearn's quirks
    # too — "fixing" it here would hand a caller different rows than sklearn
    # gives them, and would raise on `shuffle=np.False_, stratify=y`, a call
    # sklearn accepts. Pinned by `test_parity_numpy_bool_shuffle_is_identity`.
    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )
        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)
    elif stratify is not None:
        # length-check stratify against the data before it drives the split
        # (sklearn gets this from `indexable(X, y, groups)` inside `cv.split`).
        _check_consistent_length(arrays[0], stratify)
        train, test = _stratified_shuffle_split_indices(
            stratify, n_train, n_test, random_state
        )
    else:
        train, test = _shuffle_split_indices(
            n_samples, n_train, n_test, random_state
        )

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


# --------------------------------------------------------------------------- #
# cross-validation splitters (MODSEL-01: KFold / StratifiedKFold)
# --------------------------------------------------------------------------- #


class _BaseKFold:
    """Shared construction, validation and ``split`` driver for the k-fold
    splitters.

    Deliberately NOT a subclass of ``sklearn.model_selection.BaseCrossValidator``
    — mlrs is a ground-up rewrite, and sklearn consumers (``check_cv``,
    ``GridSearchCV``, ``cross_val_score``) duck-type on ``split`` +
    ``get_n_splits`` rather than on the base class. The integration is covered by
    tests that hand an mlrs splitter to sklearn's own ``GridSearchCV``.

    The constructor validation mirrors sklearn's ``_BaseKFold.__init__``
    exactly, INCLUDING two rules that differ from
    :func:`train_test_split`'s parameter constraints:

    * ``shuffle`` must be a strict ``bool`` and a violation raises **TypeError**
      (``train_test_split`` accepts ``numpy.bool_`` and raises
      :class:`InvalidParameterError`);
    * passing a ``random_state`` while ``shuffle=False`` is an ERROR, not a
      silently-ignored argument — it is the classic "my folds aren't random"
      bug, and sklearn refuses it.
    """

    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                f"{n_splits} of type {type(n_splits)} was passed."
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                f" got n_splits={n_splits}."
            )

        # TypeError, not ValueError — sklearn's own choice here.
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be True or False; got {shuffle}")

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                "Setting a random_state has no effect since shuffle is "
                "False. You should leave "
                "random_state to its default (None), or set shuffle=True."
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """The number of splitting iterations, i.e. ``n_splits``.

        The arguments exist for sklearn API compatibility and are ignored.
        """
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each of the ``n_splits`` folds.

        ``X`` may be any container :func:`train_test_split` accepts — only its
        ROW COUNT is read here, so a polars frame or a pyarrow table works
        exactly as a numpy array does. The yielded values are always integer
        numpy index arrays; pass them to :func:`_safe_indexing` (or the
        container's own take) to materialize the folds.
        """
        arrays = _indexable(X, y, groups)
        X, y, groups = arrays[0], arrays[1], arrays[2]
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater"
                f" than the number of samples: n_samples={n_samples}."
            )
        indices = np.arange(n_samples)
        for test_mask in self._iter_test_masks(X, y, groups):
            yield indices[np.logical_not(test_mask)], indices[test_mask]

    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Boolean test masks, derived from the subclass's test indices."""
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError

    def __repr__(self):
        return (
            f"{type(self).__name__}(n_splits={self.n_splits}, "
            f"random_state={self.random_state}, shuffle={self.shuffle})"
        )


class KFold(_BaseKFold):
    """K-Folds cross-validator — sklearn-compatible, bit-for-bit.

    Splits the data into ``n_splits`` consecutive folds (shuffled first if
    ``shuffle=True``); each fold serves once as the test set while the
    remaining ``n_splits - 1`` form the train set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. At least 2, and never more than ``n_samples``.
    shuffle : bool, default=False
        Shuffle the row order before splitting into folds. Note this shuffles
        WITHIN the data, not the fold order; the folds themselves are still
        contiguous slices of the (possibly shuffled) index array.
    random_state : int, RandomState instance or None, default=None
        Only meaningful with ``shuffle=True``; passing it with
        ``shuffle=False`` raises ``ValueError`` rather than being ignored.

    Examples
    --------
    >>> import numpy as np
    >>> from mlrs.model_selection import KFold
    >>> X = np.arange(10).reshape(5, 2)
    >>> kf = KFold(n_splits=5)
    >>> for train, test in kf.split(X):
    ...     print(test)
    [0]
    [1]
    [2]
    [3]
    [4]

    Notes
    -----
    The shuffle uses ``rng.shuffle(indices)`` — an IN-PLACE Fisher-Yates on the
    index array — not ``rng.permutation(n)``. The two consume the MT19937
    stream differently, so the distinction is what keeps the folds identical to
    sklearn's for a given ``random_state``.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        # The first `n_samples % n_splits` folds get one extra row, so the fold
        # sizes differ by at most 1 and every row is used exactly once.
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class StratifiedKFold(_BaseKFold):
    """Stratified K-Folds cross-validator — sklearn-compatible, bit-for-bit.

    Like :class:`KFold`, but each fold preserves the class proportions of the
    full ``y``.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds; must not exceed the size of the smallest class (a
        ``UserWarning`` is emitted when it exceeds the smallest class count,
        and a ``ValueError`` when it exceeds EVERY class count).
    shuffle : bool, default=False
        Shuffle each class's rows before assigning them to folds. This does NOT
        shuffle the data globally — stratification is preserved either way.
    random_state : int, RandomState instance or None, default=None
        Only meaningful with ``shuffle=True``; see :class:`KFold`.

    Examples
    --------
    >>> import numpy as np
    >>> from mlrs.model_selection import StratifiedKFold
    >>> X = np.zeros((10, 2))
    >>> y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> skf = StratifiedKFold(n_splits=5)
    >>> for train, test in skf.split(X, y):
    ...     print(test, y[test])
    [0 5] [0 1]
    [1 6] [0 1]
    [2 7] [0 1]
    [3 8] [0 1]
    [4 9] [0 1]

    Notes
    -----
    Two details of sklearn's algorithm are load-bearing for index parity and are
    reproduced verbatim:

    1. Classes are encoded by **order of appearance in y**, not lexicographic
       order. sklearn gets this by inverting ``np.unique(..., return_index=True)``
       a second time; encoding lexicographically instead would permute which
       fold each class block lands in.
    2. Fold allocation is a **round robin over the sorted encoded y**
       (``y_order[i::n_splits]``), then each class's fold labels are shuffled
       independently when ``shuffle=True`` — one ``rng.shuffle`` per class, in
       class order.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _make_test_folds(self, X, y=None):
        from sklearn.utils.multiclass import type_of_target
        from sklearn.utils.validation import column_or_1d

        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                f"Supported target types are: {allowed_target_types}. "
                f"Got {type_of_target_y!r} instead."
            )

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        y_inv = y_inv.reshape(-1)
        # y_inv encodes y in LEXICOGRAPHIC order; re-map through the sorted
        # first-appearance positions so classes are encoded by ORDER OF
        # APPEARANCE instead (see the class docstring, note 1).
        _, class_perm = np.unique(y_idx, return_inverse=True)
        class_perm = class_perm.reshape(-1)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )

        # Round robin over the sorted labels: allocation[i, k] is how many rows
        # of class k belong to test fold i.
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        # Assign each class's rows to folds in contiguous blocks (preserving the
        # original data order as far as stratification allows), then break that
        # ordering per class when shuffle=True.
        test_folds = np.empty(len(y), dtype="i")
        for k in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Yield ``(train, test)`` index arrays; ``y`` is REQUIRED (it is what
        the stratification is computed from). ``groups`` is ignored, with a
        warning, matching sklearn."""
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {type(self).__name__}",
                UserWarning,
            )
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


# --------------------------------------------------------------------------- #
# hyper-parameter search — sklearn-delegation passthrough (MODSEL-02)
# --------------------------------------------------------------------------- #
#
# `GridSearchCV` / `RandomizedSearchCV` / `cross_val_score` / `cross_validate`
# are RE-EXPORTED from sklearn unchanged, deliberately.
#
# They are pure orchestration: clone the estimator, loop over a parameter grid
# and a CV splitter, call `fit`/`score`, and tabulate `cv_results_`. There is no
# mlrs-specific numerics in any of it, and every mlrs estimator already
# satisfies the contract they need — `BaseEstimator` `get_params`/`set_params`
# (so `clone` works), `fit`/`predict`, and a `score` from the sklearn
# Regressor/Classifier mixins the shims inherit. Reimplementing them would add a
# large surface with no correctness or performance upside, and would drift from
# sklearn's `cv_results_` layout that downstream tooling reads.
#
# What this module DOES own is the piece that has to interoperate: the mlrs
# splitters above are accepted as `cv=` by these functions (sklearn's `check_cv`
# duck-types on `split`/`get_n_splits`). That integration is what the tests
# exercise — an mlrs estimator + an mlrs splitter driven by sklearn's search.
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
)
