"""``mlrs.model_selection`` — the complete sklearn ``model_selection`` surface,
backed by Rust (MODSEL-01/02, MODSEL-RS-01..08).

Every public name in :mod:`sklearn.model_selection` is implemented here. The
*algorithms* live in Rust (``mlrs_algos::model_selection``, reached through the
compiled ``_mlrs`` extension); this module owns the sklearn-compatible classes,
the parameter validation, the container handling, and the one thing Rust cannot
own — calling a user's estimator.

===============================  =========================================
what                             where the work happens
===============================  =========================================
splitter index generation        Rust (``mlrs_algos::model_selection::split``)
``shuffle=True`` randomness      Rust (a numpy-MT19937 reimplementation)
``ParameterGrid``/``Sampler``    Rust combinatorics, Python values
search + halving schedules       Rust
score aggregation / ranking      Rust
learning-curve tick resolution   Rust
permutation-test p-value         Rust
decision-threshold tuning        Rust
``fit`` / ``predict`` / scoring  Python (it is the user's estimator)
container row gather             Python (native pandas/polars/pyarrow takes)
===============================  =========================================

## Parity contract: host-match, not merely distributional

For the same arguments, an mlrs splitter selects the **same rows** as sklearn's,
index for index — including under ``shuffle=True``. ``random_state`` is resolved
through :func:`sklearn.utils.check_random_state` into a legacy
``numpy.random.RandomState`` (MT19937), whose 624-word state is handed to Rust,
advanced there through a bit-exact reimplementation of numpy's ``shuffle`` /
``permutation`` / ``randint``, and **written back** into the caller's
``RandomState``. Three consequences worth stating explicitly:

* an ``int`` ``random_state`` reproduces sklearn's split exactly;
* a ``RandomState`` *instance* is left advanced exactly as sklearn would have
  left it, so :class:`RepeatedKFold` (one generator shared across repeats) and
  :class:`ParameterSampler` (which interleaves ``scipy`` ``rvs`` draws that only
  Python can make) both stay in step;
* ``random_state=None`` draws from — and advances — numpy's global singleton,
  again exactly as sklearn does. Nothing is reproducible in either
  implementation under ``None``, so there is no parity to lose.

The alternative (a fast native RNG) was rejected: it would buy nothing — a split
is O(n) integer work — while silently changing every user's train/test rows
relative to the sklearn baseline they are migrating from.

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

The pandas row is reached by DUCK-TYPING on ``.iloc``, not by ``isinstance``, so
a pandas-API frame that is not a pandas instance — modin, cudf, a test double —
takes the same positional ``.take`` path it does under sklearn. That distinction
is load-bearing rather than cosmetic: the generic fallback below is
``X[indices]``, which on a dataframe means COLUMN selection by label, so an
``isinstance`` gate would silently gather such a frame along the wrong axis
instead of failing.

polars/pyarrow/scipy are *never imported* by this module — they are detected
through ``sys.modules``, so the check cannot pull in a library the user does not
have installed. (pandas is generally already in ``sys.modules`` regardless,
since ``sklearn.utils`` imports it; this module still never imports it itself.)

.. note::
   Row-take is POSITIONAL everywhere, matching sklearn: a pandas object keeps
   its original (now shuffled) index labels rather than being re-indexed
   ``0..n-1``, and a polars frame — which has no index — simply keeps its rows.

.. note::
   ``pandas.Index`` is supported here but NOT by sklearn, which raises
   ``TypeError`` on it (its generic branch evaluates ``X[indices, ...]``).
   ``pyarrow.RecordBatch`` likewise comes back as a ``RecordBatch``, where
   sklearn degrades it to a ``StructArray``. Both are deliberate improvements on
   the parity baseline — they turn an error / a lossy conversion into the
   container the caller passed in, and neither can change which ROWS are
   selected.

## Labels cross into Rust as codes

``y`` and ``groups`` may be strings, objects, floats or 2-D multi-label rows.
Rather than teach Rust about Python object ordering, this module factorizes them
with ``np.unique(..., return_inverse=True)`` and passes the integer codes. That
is exactly the encoding sklearn's splitters derive internally, so the codes carry
all the information the algorithms use and none of the representation they don't.

.. note::
   Unlike the previous pure-Python version of this module, importing it now
   requires the compiled ``_mlrs`` extension, since the algorithms live there.
   The import is LAZY (at first call, not at module import), so
   ``import mlrs.model_selection`` still works on a tree where the extension has
   not been built — only calling into it does not.
"""

import contextlib
import inspect
import numbers
import sys
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from math import ceil

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    clone,
    is_classifier,
)
from sklearn.metrics import check_scoring
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

__all__ = [
    # splitters
    "BaseCrossValidator",
    "BaseShuffleSplit",
    "GroupKFold",
    "GroupShuffleSplit",
    "KFold",
    "LeaveOneGroupOut",
    "LeaveOneOut",
    "LeavePGroupsOut",
    "LeavePOut",
    "PredefinedSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "TimeSeriesSplit",
    "check_cv",
    "train_test_split",
    # parameter iterables
    "ParameterGrid",
    "ParameterSampler",
    # search
    "GridSearchCV",
    "RandomizedSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    # validation
    "cross_val_predict",
    "cross_val_score",
    "cross_validate",
    "learning_curve",
    "permutation_test_score",
    "validation_curve",
    # decision thresholds
    "FixedThresholdClassifier",
    "TunedThresholdClassifierCV",
    # displays
    "LearningCurveDisplay",
    "ValidationCurveDisplay",
    # errors
    "InvalidParameterError",
]

# sklearn's `train_test_split` default when NEITHER test_size nor train_size is
# given (`_validate_shuffle_split(..., default_test_size=0.25)`).
_DEFAULT_TEST_SIZE = 0.25


# --------------------------------------------------------------------------- #
# the Rust bridge
# --------------------------------------------------------------------------- #


def _ext():
    """The compiled ``_mlrs`` extension, imported on first use.

    Deferred rather than imported at module scope so ``import
    mlrs.model_selection`` — and therefore ``import mlrs`` — still succeeds on a
    tree where ``maturin develop`` has not run; only *calling* a splitter then
    fails, with the extension's own import error rather than a confusing
    ``AttributeError`` deep inside a split.
    """
    from . import _mlrs

    return _mlrs


class _RustRng:
    """A caller's ``numpy.random.RandomState``, borrowed by Rust and handed back.

    sklearn passes a LIVE generator through its splitters, and callers observe
    the advancement — so this is a borrow, not a copy. On construction it lifts
    the 624-word MT19937 state into the Rust generator; :meth:`sync` writes the
    advanced words back into the same ``RandomState`` object, leaving it exactly
    where sklearn would have left it.

    Use through :func:`_rust_rng`, which guarantees the write-back even if the
    splitter raises.
    """

    def __init__(self, random_state):
        self.rs = check_random_state(random_state)
        state = self.rs.get_state(legacy=True)
        if state[0] != "MT19937":
            # `check_random_state` only ever returns a legacy RandomState, so
            # this means someone handed us a hand-built object. Fail loudly
            # rather than reinterpreting another generator's words as MT19937.
            raise ValueError(
                "mlrs.model_selection requires a legacy numpy RandomState "
                f"(MT19937); got bit generator {state[0]!r}."
            )
        self._tail = tuple(state[3:])
        self.handle = _ext().NumpyRandomState(
            np.asarray(state[1], dtype=np.uint32).tolist(), int(state[2])
        )

    def sync(self):
        """Write the advanced Rust state back into the caller's generator."""
        key, pos = self.handle.get_state()
        self.rs.set_state(
            ("MT19937", np.asarray(key, dtype=np.uint32), int(pos), *self._tail)
        )

    def reload(self):
        """Re-read the caller's generator into Rust.

        Needed when Python itself drew from the generator in between — the
        ``scipy`` ``rvs`` calls :class:`ParameterSampler` interleaves with its
        own index draws are the only case.
        """
        state = self.rs.get_state(legacy=True)
        self._tail = tuple(state[3:])
        self.handle = _ext().NumpyRandomState(
            np.asarray(state[1], dtype=np.uint32).tolist(), int(state[2])
        )


@contextlib.contextmanager
def _rust_rng(random_state):
    """Borrow ``random_state`` for the duration of the block, then sync it back."""
    bridge = _RustRng(random_state)
    try:
        yield bridge
    finally:
        bridge.sync()


def _emit(warnings_from_rust):
    """Re-raise Rust-reported warnings through Python's ``warnings`` machinery.

    Rust returns warning TEXT rather than emitting it, because a ``log::warn!``
    on the Rust side is invisible to ``pytest.warns``, to
    ``warnings.simplefilter("error")``, and to anyone filtering by category —
    all of which sklearn users rely on.
    """
    for message in warnings_from_rust:
        warnings.warn(message, UserWarning, stacklevel=3)


def _as_index(indices):
    """A Rust index list as a numpy ``intp`` array (numpy's own index dtype)."""
    return np.asarray(indices, dtype=np.intp)


def _codes(values, *, name="y"):
    """``np.unique(values, return_inverse=True)`` codes as ``int64``.

    This is the encoding every Rust splitter expects (see the module docstring).
    A 2-D ``y`` is first collapsed to one string per row, matching
    ``StratifiedShuffleSplit``'s multi-label handling — ``" ".join`` rather than
    ``str(row)`` because the latter elides rows longer than 1000 entries into an
    ellipsis, silently merging distinct label sets into one stratum.
    """
    arr = np.asarray(values)
    if arr.ndim == 2:
        arr = np.array([" ".join(row.astype(str)) for row in arr])
    _, inverse = np.unique(arr, return_inverse=True)
    return np.asarray(inverse, dtype=np.int64).ravel()


def _check_target_is_discrete(y, splitter):
    """Reject a continuous target for the stratified splitters, as sklearn does."""
    target_type = type_of_target(y)
    allowed = ("binary", "multiclass")
    if target_type not in allowed:
        raise ValueError(
            f"Supported target types are: {allowed}. Got {target_type!r} instead."
        )


def _build_repr(self):
    """sklearn's splitter ``__repr__``: the class name plus its ctor arguments.

    Reads the signature rather than ``__dict__`` so the printed order matches
    the constructor and inherited-but-unset attributes do not leak in.
    """
    signature = inspect.signature(type(self).__init__)
    names = sorted(
        name
        for name, param in signature.parameters.items()
        if name != "self" and param.kind is not param.VAR_KEYWORD
    )
    params = [f"{name}={getattr(self, name, None)!r}" for name in names]
    return f"{type(self).__name__}({', '.join(params)})"


# --------------------------------------------------------------------------- #
# parameter constraints, container detection and the positional row gather
# --------------------------------------------------------------------------- #


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
# train/test size plumbing
# --------------------------------------------------------------------------- #


def _size_kwargs(prefix, value):
    """Split a ``test_size``/``train_size`` into the int/float pair Rust takes.

    The dispatch is on ``np.asarray(value).dtype.kind``, exactly as sklearn's
    ``_validate_shuffle_split`` does — so ``np.int64(20)`` counts as an int and
    ``np.float32(0.2)`` as a float, while a ``bool`` (kind ``'b'``) matches
    neither and is rejected with sklearn's message instead of being coerced to
    0/1 rows.
    """
    if value is None:
        return {f"{prefix}_int": None, f"{prefix}_float": None}
    kind = np.asarray(value).dtype.kind
    if kind in ("i", "u"):
        return {f"{prefix}_int": int(value), f"{prefix}_float": None}
    if kind == "f":
        return {f"{prefix}_int": None, f"{prefix}_float": float(value)}
    raise ValueError(f"Invalid value for {prefix}: {value}")


def _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=None):
    """Resolve ``(n_train, n_test)`` — sklearn's ``_validate_shuffle_split``.

    Kept as a module-level function (rather than folded into each splitter)
    because :func:`train_test_split` calls it FIRST and then hands the resolved
    absolute counts to the underlying shuffle splitter. Without that ordering
    the splitter's own ``_default_test_size`` (0.1, or 0.2 for groups) would
    apply instead of ``train_test_split``'s documented 0.25.
    """
    return _ext().validate_shuffle_split(
        n_samples,
        default_test_size=(
            _DEFAULT_TEST_SIZE if default_test_size is None else float(default_test_size)
        ),
        **_size_kwargs("test_size", test_size),
        **_size_kwargs("train_size", train_size),
    )


# --------------------------------------------------------------------------- #
# splitter base classes
# --------------------------------------------------------------------------- #


class BaseCrossValidator(metaclass=ABCMeta):
    """Base class for the index-yielding cross-validators.

    Deliberately NOT a subclass of ``sklearn.model_selection.BaseCrossValidator``
    — mlrs is a ground-up rewrite, and sklearn consumers (``check_cv``,
    ``GridSearchCV``, ``cross_val_score``) duck-type on ``split`` +
    ``get_n_splits`` rather than on the base class. The integration is covered
    by tests that hand an mlrs splitter to sklearn's own ``GridSearchCV``.
    """

    @abstractmethod
    def split(self, X=None, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each split."""

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """The number of splitting iterations."""

    def __repr__(self):
        return _build_repr(self)


class BaseShuffleSplit(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for the random-permutation splitters.

    Unlike the fold-based family these do NOT partition the data: successive
    splits are independent draws, `train` and `test` need not cover every row,
    and the indices come back in **permutation order** rather than ascending.
    """

    def __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self._default_test_size = 0.1

    def get_n_splits(self, X=None, y=None, groups=None):
        """The number of re-shuffling and splitting iterations."""
        return self.n_splits


class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Shared construction and validation for the k-fold splitters.

    The constructor validation mirrors sklearn's ``_BaseKFold.__init__``
    exactly, INCLUDING two rules that differ from :func:`train_test_split`'s
    parameter constraints:

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

    def _check_n_samples(self, n_samples):
        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater"
                f" than the number of samples: n_samples={n_samples}."
            )

    @contextlib.contextmanager
    def _maybe_rng(self):
        """A Rust generator handle when shuffling, ``None`` when not.

        Skipping the bridge entirely for ``shuffle=False`` matters: building it
        would touch (and, on write-back, re-set) numpy's global singleton for a
        splitter that sklearn guarantees never consumes randomness.
        """
        if not self.shuffle:
            yield None
            return
        with _rust_rng(self.random_state) as bridge:
            yield bridge.handle


def _warn_unused_groups(splitter, groups):
    """sklearn's `_UnsupportedGroupCVMixin` warning, for the splitters that
    accept ``groups`` only for signature compatibility.

    Raised when ``split`` is CALLED rather than when its result is first
    iterated — which is why every such ``split`` returns a generator instead of
    being one. A caller who builds a splitter, passes groups by mistake and
    hands the iterator to something else would otherwise never see it.
    """
    if groups is not None:
        warnings.warn(
            f"The groups parameter is ignored by {type(splitter).__name__}",
            UserWarning,
            stacklevel=3,
        )


def _yield_splits(trains, tests):
    """Yield the Rust index lists as numpy arrays, split by split."""
    for train, test in zip(trains, tests):
        yield _as_index(train), _as_index(test)


# --------------------------------------------------------------------------- #
# fold splitters
# --------------------------------------------------------------------------- #


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
    Even with ``shuffle=True`` the yielded indices are ASCENDING: shuffling
    changes *which* rows land in a fold, not the order they are reported in.
    That is sklearn's behavior (its test sets go through a boolean mask), and
    code that zips a split against another array depends on it.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each of the ``n_splits`` folds.

        ``X`` may be any container the module docstring lists — only its ROW
        COUNT is read here, so a polars frame or a pyarrow table works exactly
        as a numpy array does.
        """
        _warn_unused_groups(self, groups)
        X, y, groups = _indexable(X, y, groups)
        n_samples = _num_samples(X)
        self._check_n_samples(n_samples)
        with self._maybe_rng() as rng:
            trains, tests, msgs = _ext().kfold_split(
                n_samples, self.n_splits, self.shuffle, rng
            )
        _emit(msgs)
        return _yield_splits(trains, tests)


class GroupKFold(_BaseKFold):
    """K-fold variant where the same group never spans train and test.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds; never more than the number of distinct groups.
    shuffle : bool, default=False
        Selects a genuinely different algorithm rather than adding noise to one:
        ``False`` greedily balances folds by ROW count (heaviest group onto the
        lightest fold), ``True`` permutes the distinct groups and cuts them into
        ``n_splits`` contiguous chunks — which balances GROUP count instead.
    random_state : int, RandomState instance or None, default=None
        Only meaningful with ``shuffle=True``.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)``; ``groups`` is required."""
        X, y, groups = _indexable(X, y, groups)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        n_samples = _num_samples(X)
        self._check_n_samples(n_samples)
        codes = _codes(groups, name="groups")
        with self._maybe_rng() as rng:
            trains, tests, msgs = _ext().group_kfold_split(
                codes.tolist(), self.n_splits, self.shuffle, rng
            )
        _emit(msgs)
        return _yield_splits(trains, tests)


class StratifiedKFold(_BaseKFold):
    """K-fold variant preserving each class's share of the data in every fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Shuffle each class's fold assignments before laying them down. The
        shuffle is per-class, not global — that is what keeps the class
        proportions intact while still varying the composition.
    random_state : int, RandomState instance or None, default=None
        Only meaningful with ``shuffle=True``.

    Notes
    -----
    Classes are re-encoded by **order of first appearance** in ``y``, not
    lexicographically, because that is the order sklearn lays the per-class fold
    blocks down in. Passing ``y`` through :func:`numpy.unique` and using those
    codes directly would assign different rows to different folds.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        """Yield ``(train_indices, test_indices)``; ``y`` is required."""
        X, y, groups = _indexable(X, y, groups)
        _warn_unused_groups(self, groups)
        if y is None:
            raise ValueError("The 'y' parameter should not be None.")
        _check_target_is_discrete(y, self)
        n_samples = _num_samples(X)
        self._check_n_samples(n_samples)
        codes = _codes(y)
        with self._maybe_rng() as rng:
            trains, tests, msgs = _ext().stratified_kfold_split(
                codes.tolist(), self.n_splits, self.shuffle, rng
            )
        _emit(msgs)
        return _yield_splits(trains, tests)


class StratifiedGroupKFold(_BaseKFold):
    """Folds that keep groups intact AND preserve the class distribution.

    Greedy: the most lopsided group (highest per-class count standard deviation)
    is placed first, into whichever fold minimizes the resulting class
    imbalance, with ties broken toward the emptier fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Randomize the order groups are considered in, which changes the greedy
        outcome among ties.
    random_state : int, RandomState instance or None, default=None
        Only meaningful with ``shuffle=True``.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        """Yield ``(train_indices, test_indices)``; ``y`` and ``groups`` required."""
        X, y, groups = _indexable(X, y, groups)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        if y is None:
            raise ValueError("The 'y' parameter should not be None.")
        _check_target_is_discrete(y, self)
        n_samples = _num_samples(X)
        self._check_n_samples(n_samples)
        with self._maybe_rng() as rng:
            trains, tests, msgs = _ext().stratified_group_kfold_split(
                _codes(y).tolist(),
                _codes(groups, name="groups").tolist(),
                self.n_splits,
                self.shuffle,
                rng,
            )
        _emit(msgs)
        return _yield_splits(trains, tests)


class TimeSeriesSplit(_BaseKFold):
    """Forward-chaining splits: the test window always follows the train window.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits.
    max_train_size : int, default=None
        Cap on the training window, making it a sliding rather than expanding
        window. A ``max_train_size`` of 0 is IGNORED (sklearn tests it for
        truthiness), not treated as "train on nothing".
    test_size : int, default=None
        Size of each test window; defaults to ``n_samples // (n_splits + 1)``.
    gap : int, default=0
        Rows dropped between the end of train and the start of test, for series
        where adjacent samples leak into each other.
    """

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        # `shuffle` is meaningless for a time series, so this bypasses
        # `_BaseKFold.__init__`'s shuffle/random_state rules rather than
        # pretending to accept them.
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                f"{n_splits} of type {type(n_splits)} was passed."
            )
        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                f" got n_splits={int(n_splits)}."
            )
        self.n_splits = int(n_splits)
        self.shuffle = False
        self.random_state = None
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)``, both ascending and disjoint."""
        _warn_unused_groups(self, groups)
        X, y, groups = _indexable(X, y, groups)
        trains, tests, msgs = _ext().time_series_split(
            _num_samples(X),
            self.n_splits,
            self.max_train_size,
            self.test_size,
            self.gap,
        )
        _emit(msgs)
        return _yield_splits(trains, tests)


# --------------------------------------------------------------------------- #
# leave-out splitters (streamed, not materialized)
# --------------------------------------------------------------------------- #


class LeaveOneOut(BaseCrossValidator):
    """Leave-One-Out cross-validator: ``n_samples`` splits of one test row each.

    Notes
    -----
    Splits are generated ONE AT A TIME rather than materialized: the full set is
    ``n_samples`` train vectors of ``n_samples - 1`` entries, which is quadratic
    in the row count and is the reason sklearn keeps this lazy too.
    """

    def get_n_splits(self, X, y=None, groups=None):
        """``n_samples`` — every row is held out once."""
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return _num_samples(X)

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)``, one held-out row at a time."""
        _warn_unused_groups(self, groups)
        X, y, groups = _indexable(X, y, groups)
        n_samples = _num_samples(X)

        def stream():
            for i in range(n_samples):
                train, test = _ext().leave_one_out_split_at(n_samples, i)
                yield _as_index(train), _as_index(test)

        return stream()


class LeavePOut(BaseCrossValidator):
    """Leave-P-Out cross-validator: every ``p``-subset of rows is held out once.

    Parameters
    ----------
    p : int
        Size of each test set. Must be strictly less than ``n_samples``.

    Notes
    -----
    There are ``comb(n_samples, p)`` splits — 161 700 for ``n_samples=100,
    p=3`` — so splits are streamed rather than materialized. The `i`-th split is
    unranked directly from `i`, which reproduces
    :func:`itertools.combinations`' lexicographic order without enumerating the
    ones before it.
    """

    def __init__(self, p):
        self.p = p

    def get_n_splits(self, X, y=None, groups=None):
        """``comb(n_samples, p)``."""
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return int(_ext().leave_p_out_n_splits(_num_samples(X), self.p))

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each ``p``-subset."""
        _warn_unused_groups(self, groups)
        X, y, groups = _indexable(X, y, groups)
        n_samples = _num_samples(X)
        total = _ext().leave_p_out_n_splits(n_samples, self.p)

        def stream():
            for i in range(total):
                train, test = _ext().leave_p_out_split_at(n_samples, self.p, i)
                yield _as_index(train), _as_index(test)

        return stream()


class LeaveOneGroupOut(BaseCrossValidator):
    """Hold out one whole group per split."""

    def get_n_splits(self, X=None, y=None, groups=None):
        """The number of distinct groups."""
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        return len(np.unique(np.asarray(groups)))

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)``, one group held out at a time."""
        X, y, groups = _indexable(X, y, groups)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        trains, tests, msgs = _ext().leave_one_group_out_split(
            _codes(groups, name="groups").tolist()
        )
        _emit(msgs)
        return _yield_splits(trains, tests)


class LeavePGroupsOut(BaseCrossValidator):
    """Hold out every combination of ``n_groups`` whole groups.

    Parameters
    ----------
    n_groups : int
        Number of groups held out per split; must be strictly less than the
        number of distinct groups.
    """

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def get_n_splits(self, X=None, y=None, groups=None):
        """``comb(n_distinct_groups, n_groups)``."""
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        codes = _codes(groups, name="groups")
        return int(_ext().leave_p_groups_out_n_splits(codes.tolist(), self.n_groups))

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each group combination."""
        X, y, groups = _indexable(X, y, groups)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        codes = _codes(groups, name="groups").tolist()
        total = _ext().leave_p_groups_out_n_splits(codes, self.n_groups)

        def stream():
            for i in range(total):
                train, test = _ext().leave_p_groups_out_split_at(codes, self.n_groups, i)
                yield _as_index(train), _as_index(test)

        return stream()


class PredefinedSplit(BaseCrossValidator):
    """Splits taken verbatim from a caller-supplied fold assignment.

    Parameters
    ----------
    test_fold : array-like of shape (n_samples,)
        Fold id per row. Rows tagged ``-1`` are NEVER in any test set — they
        are training-only, which is how a fixed validation split is expressed.
    """

    def __init__(self, test_fold):
        self.test_fold = np.array(test_fold, dtype=int)
        self.test_fold = np.asarray(self.test_fold).ravel()
        self.unique_folds = np.unique(self.test_fold)
        self.unique_folds = self.unique_folds[self.unique_folds != -1]

    def get_n_splits(self, X=None, y=None, groups=None):
        """The number of distinct non-negative fold ids."""
        return len(self.unique_folds)

    def split(self, X=None, y=None, groups=None):
        """Yield ``(train_indices, test_indices)``, one per distinct fold id."""
        if groups is not None:
            warnings.warn(
                f"The groups parameter is ignored by {type(self).__name__}",
                UserWarning,
                stacklevel=2,
            )
        trains, tests, msgs = _ext().predefined_split(self.test_fold.tolist())
        _emit(msgs)
        return _yield_splits(trains, tests)


# --------------------------------------------------------------------------- #
# shuffle splitters
# --------------------------------------------------------------------------- #


class ShuffleSplit(BaseShuffleSplit):
    """Independent random train/test draws.

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling and splitting iterations.
    test_size : float or int, default=None
        A float is a proportion (rounded UP); an int is an absolute row count.
        Defaults to 0.1 when ``train_size`` is also ``None``.
    train_size : float or int, default=None
        A float is a proportion (rounded DOWN); an int is an absolute count.
        Defaults to the complement of ``test_size``.
    random_state : int, RandomState instance or None, default=None

    Notes
    -----
    The yielded indices are in PERMUTATION order, not ascending — this family
    reports its draw order directly. Sorting them would break index-for-index
    compatibility with sklearn.

    Successive splits are independent draws, so a row can appear in several test
    sets or none; this is not a partition and cannot drive
    :func:`cross_val_predict`.
    """

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` for each random draw."""
        _warn_unused_groups(self, groups)
        X, y, groups = _indexable(X, y, groups)
        with _rust_rng(self.random_state) as bridge:
            trains, tests, msgs = _ext().shuffle_split(
                _num_samples(X),
                self.n_splits,
                rng=bridge.handle,
                **_size_kwargs("test_size", self.test_size),
                **_size_kwargs("train_size", self.train_size),
            )
        _emit(msgs)
        return _yield_splits(trains, tests)


class GroupShuffleSplit(BaseShuffleSplit):
    """A :class:`ShuffleSplit` over the *groups*, expanded back to rows.

    ``test_size`` / ``train_size`` are proportions of the DISTINCT GROUPS, not
    of the rows, so a 0.2 test size on ragged groups rarely yields 20% of the
    data. The default ``test_size`` is 0.2 here (0.1 for :class:`ShuffleSplit`).

    Notes
    -----
    Output is ASCENDING, unlike its :class:`ShuffleSplit` parent: groups are
    drawn in permutation order but rows are recovered by a mask.
    """

    def __init__(self, n_splits=5, *, test_size=None, train_size=None, random_state=None):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.2

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)``; ``groups`` is required."""
        X, y, groups = _indexable(X, y, groups)
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        with _rust_rng(self.random_state) as bridge:
            trains, tests, msgs = _ext().group_shuffle_split(
                _codes(groups, name="groups").tolist(),
                self.n_splits,
                rng=bridge.handle,
                **_size_kwargs("test_size", self.test_size),
                **_size_kwargs("train_size", self.train_size),
            )
        _emit(msgs)
        return _yield_splits(trains, tests)


class StratifiedShuffleSplit(BaseShuffleSplit):
    """Random train/test draws that preserve the class distribution.

    Notes
    -----
    Output is in permutation order, and both sides are permuted a second time at
    the end so the classes are interleaved rather than blocked.
    """

    def split(self, X, y, groups=None):
        """Yield ``(train_indices, test_indices)``; ``y`` is required."""
        X, y, groups = _indexable(X, y, groups)
        _warn_unused_groups(self, groups)
        if y is None:
            raise ValueError("The 'y' parameter should not be None.")
        codes = _codes(y)
        _check_least_populated_class(y, codes)
        with _rust_rng(self.random_state) as bridge:
            trains, tests, msgs = _ext().stratified_shuffle_split(
                codes.tolist(),
                self.n_splits,
                rng=bridge.handle,
                **_size_kwargs("test_size", self.test_size),
                **_size_kwargs("train_size", self.train_size),
            )
        _emit(msgs)
        return _yield_splits(trains, tests)


def _check_least_populated_class(y, codes):
    """Reject a singleton class before Rust sees only its code.

    Raised here rather than in Rust so the message can name the offending
    LABELS (``['rare']``) rather than their factorization codes (``[2]``), which
    is what sklearn prints and what the user can act on.
    """
    arr = np.asarray(y)
    if arr.ndim == 2:
        arr = np.array([" ".join(row.astype(str)) for row in arr])
    classes, counts = np.unique(arr, return_counts=True)
    if counts.size and counts.min() < 2:
        too_few = classes[counts < 2].tolist()
        raise ValueError(
            "The least populated classes in y have only 1"
            " member, which is too few. The minimum"
            " number of groups for any class cannot"
            " be less than 2. Classes with too few"
            " members are: %s" % (too_few,)
        )


# --------------------------------------------------------------------------- #
# repeated splitters
# --------------------------------------------------------------------------- #


class RepeatedKFold(BaseCrossValidator):
    """``n_repeats`` independent shuffled :class:`KFold` runs.

    Parameters
    ----------
    n_splits : int, default=5
    n_repeats : int, default=10
    random_state : int, RandomState instance or None, default=None

    Notes
    -----
    All repeats draw from ONE generator, continuing where the previous repeat
    stopped. Re-seeding per repeat would make every repeat identical while still
    satisfying every per-repeat invariant — a failure that is invisible unless
    two repeats are compared.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError("Number of repetitions must be of Integral type.")
        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """``n_splits * n_repeats``."""
        return self.n_splits * self.n_repeats

    def split(self, X, y=None, groups=None):
        """Yield ``(train_indices, test_indices)`` across every repeat."""
        _warn_unused_groups(self, groups)
        X, y, groups = _indexable(X, y, groups)
        with _rust_rng(self.random_state) as bridge:
            trains, tests, msgs = _ext().repeated_kfold_split(
                _num_samples(X), self.n_splits, self.n_repeats, bridge.handle
            )
        _emit(msgs)
        return _yield_splits(trains, tests)


class RepeatedStratifiedKFold(BaseCrossValidator):
    """``n_repeats`` independent shuffled :class:`StratifiedKFold` runs.

    Shares one generator across repeats, like :class:`RepeatedKFold`.
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError("Number of repetitions must be of Integral type.")
        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """``n_splits * n_repeats``."""
        return self.n_splits * self.n_repeats

    def split(self, X, y, groups=None):
        """Yield ``(train_indices, test_indices)``; ``y`` is required."""
        _warn_unused_groups(self, groups)
        X, y, groups = _indexable(X, y, groups)
        _check_target_is_discrete(y, self)
        with _rust_rng(self.random_state) as bridge:
            trains, tests, msgs = _ext().repeated_stratified_kfold_split(
                _codes(y).tolist(), self.n_splits, self.n_repeats, bridge.handle
            )
        _emit(msgs)
        return _yield_splits(trains, tests)

# --------------------------------------------------------------------------- #
# train_test_split
# --------------------------------------------------------------------------- #


def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """Split arrays into random train and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with the same first dimension
        Any mix of the containers in the module docstring's table. Each one is
        gathered with its own native row-take and comes back as the same type.
    test_size : float or int, default=None
        A float in ``(0, 1)`` is a proportion of the rows, rounded UP; an int is
        an absolute row count. Defaults to the complement of ``train_size``, or
        0.25 when neither is given.
    train_size : float or int, default=None
        A float is a proportion, rounded DOWN; an int is an absolute count.
    random_state : int, RandomState instance or None, default=None
        An int reproduces sklearn's split exactly.
    shuffle : bool, default=True
        With ``False`` the split is a plain prefix/suffix cut, and ``stratify``
        is rejected.
    stratify : array-like, default=None
        Class labels to preserve the distribution of.

    Returns
    -------
    splitting : list, length ``2 * len(arrays)``
        ``[a_train, a_test, b_train, b_test, ...]``.

    Examples
    --------
    >>> import numpy as np
    >>> from mlrs.model_selection import train_test_split
    >>> X = np.arange(10).reshape(5, 2)
    >>> y = [0, 1, 0, 1, 0]
    >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
    >>> len(X_tr), len(X_te)
    (3, 2)

    Notes
    -----
    The sizes are resolved against a 0.25 default and handed to the underlying
    shuffle splitter as ABSOLUTE counts, so this function does not inherit
    :class:`ShuffleSplit`'s own 0.1 default.
    """
    # Parameter constraints run FIRST — sklearn checks them in a decorator, so
    # they fire before the body's "no arrays" check. This is also the split
    # sklearn draws between `InvalidParameterError` (a malformed argument) and
    # `ValueError` (an argument that is wrong for this data).
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

    if stratify is not None:
        _check_consistent_length(arrays[0], stratify)

    # `shuffle is False`, NOT `not shuffle`: sklearn tests the singleton, so
    # `np.False_` — which is falsy but is not `False` — takes the SHUFFLING
    # path there. Rewriting this as a truthiness test would silently change the
    # rows for anyone passing a numpy boolean.
    if shuffle is False:
        if stratify is not None:
            raise ValueError(
                "Stratified train/test split is not implemented for shuffle=False"
            )
        # No randomness is consumed, so the generator is not even touched.
        train, test = _ext().train_test_split_indices(
            n_samples,
            shuffle=False,
            stratify=None,
            rng=None,
            **_size_kwargs("test_size", test_size),
            **_size_kwargs("train_size", train_size),
        )
    else:
        codes = None
        if stratify is not None:
            codes = _codes(stratify, name="stratify")
            _check_least_populated_class(stratify, codes)
            codes = codes.tolist()
        with _rust_rng(random_state) as bridge:
            train, test = _ext().train_test_split_indices(
                n_samples,
                shuffle=True,
                stratify=codes,
                rng=bridge.handle,
                **_size_kwargs("test_size", test_size),
                **_size_kwargs("train_size", train_size),
            )

    train, test = _as_index(train), _as_index(test)
    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


# `train_test_split` is not a test — some collectors key on the leading `t`.
train_test_split.__test__ = False


# --------------------------------------------------------------------------- #
# check_cv
# --------------------------------------------------------------------------- #


class _CVIterableWrapper(BaseCrossValidator):
    """Wrap a plain iterable of ``(train, test)`` index pairs as a splitter."""

    def __init__(self, cv):
        self.cv = list(cv)

    def get_n_splits(self, X=None, y=None, groups=None):
        """The number of ``(train, test)`` pairs the iterable held."""
        return len(self.cv)

    def split(self, X=None, y=None, groups=None):
        """Yield the wrapped pairs as numpy index arrays."""
        for train, test in self.cv:
            yield np.asarray(train), np.asarray(test)


def check_cv(cv=5, y=None, *, classifier=False, shuffle=False, random_state=None):
    """Resolve a ``cv`` argument into a splitter object.

    Parameters
    ----------
    cv : int, cross-validation generator or iterable, default=5
        An int means that many folds; a splitter is passed through untouched;
        an iterable of ``(train, test)`` index pairs is wrapped.
    y : array-like, default=None
        The target. Only consulted to decide whether an int ``cv`` becomes a
        :class:`StratifiedKFold`.
    classifier : bool, default=False
        Whether the estimator is a classifier. An int ``cv`` becomes a
        :class:`StratifiedKFold` only when this is ``True`` AND ``y`` is a
        binary/multiclass target — a continuous ``y`` falls back to
        :class:`KFold` even for a classifier.
    shuffle : bool, default=False
    random_state : int, RandomState instance or None, default=None

    Returns
    -------
    checked_cv : a cross-validator with a ``split`` method.
    """
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        if classifier and (y is not None) and (
            type_of_target(y, input_name="y") in ("binary", "multiclass")
        ):
            return StratifiedKFold(cv, shuffle=shuffle, random_state=random_state)
        return KFold(cv, shuffle=shuffle, random_state=random_state)

    if not hasattr(cv, "split") or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError(
                "Expected `cv` as an integer, a cross-validation object "
                "(from sklearn.model_selection), or an iterable yielding "
                f"(train, test) splits as arrays of indices. Got {cv}."
            )
        return _CVIterableWrapper(cv)

    return cv  # already a splitter — mlrs's or sklearn's


# --------------------------------------------------------------------------- #
# ParameterGrid / ParameterSampler
# --------------------------------------------------------------------------- #


class ParameterGrid:
    """The cartesian product of a parameter grid, enumerated in sklearn's order.

    Parameters
    ----------
    param_grid : dict of str to sequence, or list of such dicts
        A list of dicts is a UNION of grids, searched one after another — the
        way to search parameters that only make sense together (an ``rbf``
        kernel's ``gamma``, say).

    Examples
    --------
    >>> from mlrs.model_selection import ParameterGrid
    >>> list(ParameterGrid({"a": [1, 2], "b": ["x"]}))
    [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}]

    Notes
    -----
    Keys are sorted and the LAST one varies fastest. That order is not cosmetic:
    :class:`RandomizedSearchCV` samples *indices into this enumeration*, so a
    different order would draw a different subset from the same seed.
    """

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            # A single dict is a grid of one — wrap it so the rest of the class
            # only ever handles the list form.
            param_grid = [param_grid]
        if not isinstance(param_grid, Iterable):
            raise TypeError(f"Parameter grid should be a dict or a list, got: {param_grid!r}")

        param_grid = list(param_grid)
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Parameter grid is not a dict ({grid!r})")
            for key, value in grid.items():
                if isinstance(value, np.ndarray) and value.ndim > 1:
                    raise ValueError(
                        f"Parameter array for {key!r} should be one-dimensional, got:"
                        f" {value!r} with shape {value.shape}"
                    )
                if isinstance(value, str) or not isinstance(value, (np.ndarray, Sequence)):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} needs to be a list or a"
                        f" numpy array, but got {value!r} (of type "
                        f"{type(value).__name__}) instead. Single values "
                        "need to be wrapped in a list with one element."
                    )
                if len(value) == 0:
                    raise ValueError(
                        f"Parameter grid for parameter {key!r} need "
                        f"to be a non-empty sequence, got: {value!r}"
                    )
        self.param_grid = param_grid

    def _sorted_items(self):
        """Per sub-grid, the ``(keys, value_lists)`` in sorted-key order."""
        out = []
        for grid in self.param_grid:
            items = sorted(grid.items())
            keys = [k for k, _ in items]
            values = [v for _, v in items]
            out.append((keys, values))
        return out

    def _value_counts(self):
        """Per sub-grid, the value count per sorted key — the Rust wire form."""
        return [[len(v) for v in values] for _, values in self._sorted_items()]

    def __len__(self):
        """Number of candidates across every sub-grid."""
        return _ext().parameter_grid_size(self._value_counts())

    def __getitem__(self, ind):
        """The ``ind``-th candidate as a parameter dict."""
        layout = self._sorted_items()
        found = _ext().parameter_grid_nth(self._value_counts(), ind)
        if found is None:
            raise IndexError("ParameterGrid index out of range")
        grid_idx, value_indices = found
        keys, values = layout[grid_idx]
        return {k: values[i][j] for i, (k, j) in enumerate(zip(keys, value_indices))}

    def __iter__(self):
        """Iterate over every candidate, in sklearn's order."""
        for i in range(len(self)):
            yield self[i]


class ParameterSampler:
    """``n_iter`` parameter settings sampled from distributions or lists.

    Parameters
    ----------
    param_distributions : dict or list of dicts
        Values are either lists (sampled without replacement across the implied
        grid) or objects with an ``rvs`` method (a ``scipy.stats``
        distribution, sampled independently each draw).
    n_iter : int
        Number of settings to draw. Capped at the grid size — with a
        ``UserWarning`` — when every value is a list.
    random_state : int, RandomState instance or None, default=None

    Notes
    -----
    When any value is a distribution the draw loop runs in Python, because only
    Python can call ``scipy``'s ``rvs``. The generator is handed back and forth
    around each such call so the interleaved stream stays identical to
    sklearn's — that is why a ``RandomState`` instance passed here comes back
    advanced by exactly the right number of draws.
    """

    def __init__(self, param_distributions, n_iter, *, random_state=None):
        if not isinstance(param_distributions, (Mapping, Iterable)):
            raise TypeError(
                "Parameter distribution is not a dict or a list ({!r})".format(
                    param_distributions
                )
            )
        if isinstance(param_distributions, Mapping):
            param_distributions = [param_distributions]

        param_distributions = list(param_distributions)
        for dist in param_distributions:
            if not isinstance(dist, dict):
                raise TypeError(f"Parameter distribution is not a dict ({dist!r})")
            for key in dist:
                if not isinstance(dist[key], Iterable) and not hasattr(dist[key], "rvs"):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} is not iterable "
                        f"or a distribution (value={dist[key]})"
                    )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _is_all_lists(self):
        """True when no value is a distribution — the finite-grid fast path."""
        return all(
            all(not hasattr(v, "rvs") for v in dist.values())
            for dist in self.param_distributions
        )

    def __len__(self):
        """The number of settings that will actually be drawn."""
        if self._is_all_lists():
            return min(self.n_iter, len(ParameterGrid(self.param_distributions)))
        return self.n_iter

    def __iter__(self):
        """Yield the sampled parameter dicts."""
        if self._is_all_lists():
            grid = ParameterGrid(self.param_distributions)
            with _rust_rng(self.random_state) as bridge:
                indices, warning = _ext().sample_parameter_grid_indices(
                    grid._value_counts(), self.n_iter, bridge.handle
                )
            if warning is not None:
                warnings.warn(warning, UserWarning, stacklevel=2)
            for i in indices:
                yield grid[i]
            return

        dists = self.param_distributions
        drawn = []
        with _rust_rng(self.random_state) as bridge:
            for _ in range(self.n_iter):
                # `rng.choice(list_of_dicts)` with the legacy generator is one
                # masked `randint` draw — matched here exactly.
                dist = dists[bridge.handle.randint(len(dists))]
                params = {}
                for k, v in sorted(dist.items()):
                    if hasattr(v, "rvs"):
                        # Hand the stream to numpy so scipy draws from the same
                        # generator sklearn would have, then take it back.
                        bridge.sync()
                        params[k] = v.rvs(random_state=bridge.rs)
                        bridge.reload()
                    else:
                        params[k] = v[bridge.handle.randint(len(v))]
                drawn.append(params)
        yield from drawn

# --------------------------------------------------------------------------- #
# fit/score plumbing (the one part Rust cannot own)
# --------------------------------------------------------------------------- #


def _parallel(n_jobs, pre_dispatch, verbose=0):
    """A joblib ``Parallel``. joblib arrives with scikit-learn, never on its own."""
    from joblib import Parallel

    return Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)


def _delayed(func):
    from joblib import delayed

    return delayed(func)


def _resolve_scorers(estimator, scoring):
    """Normalize ``scoring`` into ``(name -> scorer, is_multimetric)``.

    A single scorer keeps the plain ``test_score`` key; a list/tuple/set/dict
    produces ``test_<name>`` keys. The distinction is what a caller's
    ``cv_results_`` indexing depends on, so it is carried explicitly rather than
    inferred later from the dict size.
    """
    if scoring is None or isinstance(scoring, str) or callable(scoring):
        return {"score": check_scoring(estimator, scoring)}, False
    if isinstance(scoring, (list, tuple, set)):
        names = list(scoring)
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate elements in {scoring!r}.")
        return {name: check_scoring(estimator, name) for name in names}, True
    if isinstance(scoring, dict):
        return {
            name: check_scoring(estimator, value) for name, value in scoring.items()
        }, True
    raise ValueError(
        "scoring must be None, a string, a callable, a list/tuple/set of "
        f"strings, or a dict; got {scoring!r}"
    )


def _score_all(scorers, estimator, X, y, score_params, error_score):
    """Run every scorer, returning ``{name: float}``.

    A scorer that raises is reported as ``error_score`` (with a warning) rather
    than aborting the whole cross-validation, matching sklearn — a single
    degenerate fold should not lose the other four folds' results.
    """
    scores = {}
    for name, scorer in scorers.items():
        try:
            value = scorer(estimator, X, y, **score_params) if y is not None else scorer(
                estimator, X, **score_params
            )
            scores[name] = float(value)
        except Exception as exc:
            if error_score == "raise":
                raise
            warnings.warn(
                f"Scoring failed. The score on this train-test partition for "
                f"these parameters will be set to {error_score}. Details: \n"
                f"{type(exc).__name__}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            scores[name] = float(error_score)
    return scores


def _index_params(params, indices, n_samples):
    """Index the row-aligned entries of ``params`` (sample weights and friends).

    Only values whose length matches the data are indexed; a scalar or a
    differently-shaped argument is passed through untouched, which is how a
    ``fit`` keyword like ``classes=`` survives.
    """
    if not params:
        return {}
    out = {}
    for key, value in params.items():
        if value is None or np.isscalar(value):
            out[key] = value
            continue
        try:
            length = _num_samples(value)
        except TypeError:
            out[key] = value
            continue
        out[key] = _safe_indexing(value, indices) if length == n_samples else value
    return out


def _fit_and_score(
    estimator,
    X,
    y,
    *,
    scorers,
    train,
    test,
    parameters=None,
    fit_params=None,
    score_params=None,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    error_score=np.nan,
    split_progress=None,
    verbose=0,
):
    """Fit one estimator on one split and score it — sklearn's ``_fit_and_score``.

    Returns a result dict. A failed FIT is recorded as ``error_score`` for every
    scorer (with a ``FitFailedWarning``) unless ``error_score="raise"``, so a
    search over a parameter that is invalid for part of the grid still reports
    the candidates that did work.
    """
    from sklearn.exceptions import FitFailedWarning

    fit_params = fit_params or {}
    score_params = score_params or {}
    n_samples = _num_samples(X)

    if parameters is not None:
        # `clone` first so a parameter set never leaks into the next candidate.
        estimator = estimator.set_params(**clone(parameters, safe=False))

    X_train = _safe_indexing(X, train)
    X_test = _safe_indexing(X, test)
    y_train = None if y is None else _safe_indexing(y, train)
    y_test = None if y is None else _safe_indexing(y, test)
    fit_params_train = _index_params(fit_params, train, n_samples)
    score_params_test = _index_params(score_params, test, n_samples)

    result = {}
    start_time = time.time()
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params_train)
        else:
            estimator.fit(X_train, y_train, **fit_params_train)
    except Exception as exc:
        fit_time = time.time() - start_time
        if error_score == "raise":
            raise
        result["fit_time"] = fit_time
        result["score_time"] = 0.0
        result["test_scores"] = {name: float(error_score) for name in scorers}
        if return_train_score:
            result["train_scores"] = dict(result["test_scores"])
        result["fit_error"] = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "Estimator fit failed. The score on this train-test partition for "
            f"these parameters will be set to {error_score}. Details: \n"
            f"{result['fit_error']}",
            FitFailedWarning,
            stacklevel=2,
        )
    else:
        fit_time = time.time() - start_time
        score_start = time.time()
        result["test_scores"] = _score_all(
            scorers, estimator, X_test, y_test, score_params_test, error_score
        )
        result["score_time"] = time.time() - score_start
        result["fit_time"] = fit_time
        if return_train_score:
            result["train_scores"] = _score_all(
                scorers,
                estimator,
                X_train,
                y_train,
                _index_params(score_params, train, n_samples),
                error_score,
            )
        result["fit_error"] = None

    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = result["fit_time"]
        result["score_time"] = result["score_time"]
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


# --------------------------------------------------------------------------- #
# cross_validate / cross_val_score / cross_val_predict
# --------------------------------------------------------------------------- #


def cross_validate(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    return_indices=False,
    error_score=np.nan,
):
    """Evaluate metrics by cross-validation, recording fit and score times.

    Parameters mirror :func:`sklearn.model_selection.cross_validate`. ``params``
    is forwarded to the estimator's ``fit``, with any row-aligned entry (a
    ``sample_weight``, say) indexed per fold.

    Returns
    -------
    scores : dict of str to ndarray
        ``fit_time`` / ``score_time`` / ``test_score`` (or ``test_<name>`` for
        multimetric scoring), plus ``train_*`` / ``estimator`` / ``indices``
        when requested.
    """
    X, y, groups = _indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, multimetric = _resolve_scorers(estimator, scoring)
    splits = list(cv.split(X, y, groups))

    results = _parallel(n_jobs, pre_dispatch, verbose)(
        _delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorers=scorers,
            train=train,
            test=test,
            fit_params=params,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            error_score=error_score,
            verbose=verbose,
        )
        for train, test in splits
    )

    out = {
        "fit_time": np.array([r["fit_time"] for r in results]),
        "score_time": np.array([r["score_time"] for r in results]),
    }
    for name in scorers:
        key = f"test_{name}" if multimetric else "test_score"
        out[key] = np.array([r["test_scores"][name] for r in results])
        if return_train_score:
            train_key = f"train_{name}" if multimetric else "train_score"
            out[train_key] = np.array([r["train_scores"][name] for r in results])
    if return_estimator:
        out["estimator"] = [r["estimator"] for r in results]
    if return_indices:
        out["indices"] = {
            "train": [train for train, _ in splits],
            "test": [test for _, test in splits],
        }
    return out


def cross_val_score(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
):
    """Cross-validated scores for a single metric — the ``test_score`` column of
    :func:`cross_validate`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from mlrs.model_selection import cross_val_score
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(30, 3))
    >>> y = X @ [1.0, 2.0, 3.0]
    >>> scores = cross_val_score(Ridge(), X, y, cv=3)
    >>> scores.shape
    (3,)
    """
    if isinstance(scoring, (list, tuple, set, dict)):
        raise ValueError(
            "cross_val_score accepts a single metric; pass a list or dict of "
            "scorers to cross_validate instead."
        )
    result = cross_validate(
        estimator,
        X,
        y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
    )
    return result["test_score"]


def _enforce_prediction_order(fold_classes, predictions, all_classes, method):
    """Widen a fold's probability/decision output back to the full class set.

    A fold whose training rows happened to miss a class produces a NARROWER
    matrix than the other folds, and stacking those would misalign every column
    after the gap — silently, since the shapes only disagree in one axis that
    `np.concatenate` is not checking. Each present class is placed at its
    position in the global class list and the rest are filled with the neutral
    value for the method (0 for a probability, the dtype minimum for a score),
    matching sklearn's `_enforce_prediction_order`.
    """
    predictions = np.asarray(predictions)
    if len(fold_classes) == len(all_classes):
        return predictions
    # Widening keeps the columns aligned, but the fold's model never saw the
    # missing class, so its probabilities are not comparable with the other
    # folds'. sklearn warns here and so does mlrs — a silently-widened column of
    # zeros reads as "confidently not this class".
    warnings.warn(
        f"Number of classes in training fold ({len(fold_classes)}) does not "
        f"match total number of classes ({len(all_classes)}). Results may not "
        "be appropriate for your use case. To fix this, use a cross-validation "
        "technique resulting in properly stratified folds",
        RuntimeWarning,
        stacklevel=2,
    )
    fill = 0.0 if method == "predict_proba" else np.finfo(predictions.dtype).min
    widened = np.full(
        (_num_samples(predictions), len(all_classes)), fill, dtype=predictions.dtype
    )
    # `all_classes` is sorted (`np.unique`), so `searchsorted` maps each of the
    # fold's labels to its column in the global layout.
    widened[:, np.searchsorted(all_classes, fold_classes)] = predictions
    return widened


def _fit_and_predict(estimator, X, y, train, test, params, method, all_classes):
    """Fit on ``train`` and call ``method`` on ``test``."""
    n_samples = _num_samples(X)
    X_train = _safe_indexing(X, train)
    X_test = _safe_indexing(X, test)
    fit_params = _index_params(params, train, n_samples)
    if y is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, _safe_indexing(y, train), **fit_params)
    predictions = getattr(estimator, method)(X_test)
    if method in ("decision_function", "predict_proba", "predict_log_proba") and (
        all_classes is not None
    ):
        fold_classes = getattr(estimator, "classes_", None)
        if fold_classes is not None and np.ndim(predictions) == 2:
            predictions = _enforce_prediction_order(
                np.asarray(fold_classes), predictions, all_classes, method
            )
    return predictions


def cross_val_predict(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    method="predict",
):
    """Cross-validated predictions: each row predicted by the fold that held it out.

    Notes
    -----
    The splitter must PARTITION the rows — every row in exactly one test set —
    so :class:`ShuffleSplit` and friends are rejected. The check runs in Rust
    and raises sklearn's own "only works for partitions" message.
    """
    X, y, groups = _indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, groups))
    n_samples = _num_samples(X)

    # Rust validates the partition and hands back the scatter map in one pass.
    inverse = _ext().partition_inverse(
        [np.asarray(test).astype(np.int64).tolist() for _, test in splits], n_samples
    )

    all_classes = None
    if method != "predict" and y is not None:
        all_classes = np.unique(np.asarray(y))

    predictions = _parallel(n_jobs, pre_dispatch, verbose)(
        _delayed(_fit_and_predict)(
            clone(estimator), X, y, train, test, params, method, all_classes
        )
        for train, test in splits
    )
    stacked = np.concatenate([np.asarray(p) for p in predictions], axis=0)
    return stacked[np.asarray(inverse, dtype=np.intp)]


# --------------------------------------------------------------------------- #
# learning_curve / validation_curve / permutation_test_score
# --------------------------------------------------------------------------- #


def learning_curve(
    estimator,
    X,
    y,
    *,
    groups=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=None,
    scoring=None,
    exploit_incremental_learning=False,
    n_jobs=None,
    pre_dispatch="all",
    verbose=0,
    shuffle=False,
    random_state=None,
    error_score=np.nan,
    return_times=False,
    params=None,
):
    """Train and test scores as a function of the training-set size.

    Returns
    -------
    train_sizes_abs : ndarray
    train_scores : ndarray of shape (n_ticks, n_cv_folds)
    test_scores : ndarray of shape (n_ticks, n_cv_folds)
    fit_times, score_times : ndarray, only when ``return_times=True``

    Notes
    -----
    ``train_sizes`` resolution (fraction vs absolute, truncation, clipping and
    de-duplication) happens in Rust; a float dtype means fractions of the
    maximum training size, an integer dtype means absolute row counts, and the
    two are not interchangeable.

    ``exploit_incremental_learning`` is accepted for signature compatibility;
    passing ``True`` raises rather than silently ignoring it, because the whole
    point of the flag is a different (``partial_fit``-driven) cost profile.
    """
    if exploit_incremental_learning:
        raise NotImplementedError(
            "mlrs.model_selection.learning_curve does not implement "
            "exploit_incremental_learning=True; use "
            "sklearn.model_selection.learning_curve for the partial_fit path."
        )

    X, y, groups = _indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _resolve_scorers(estimator, scoring)
    cv_iter = list(cv.split(X, y, groups))

    n_max_training_samples = min(len(train) for train, _ in cv_iter)
    sizes = np.asarray(train_sizes)
    kwargs = (
        {"fractions": [float(v) for v in sizes.ravel()]}
        if np.issubdtype(sizes.dtype, np.floating)
        else {"absolute": [int(v) for v in sizes.ravel()]}
    )
    train_sizes_abs, warning = _ext().translate_train_sizes(
        n_max_training_samples, **kwargs
    )
    if warning is not None:
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    train_sizes_abs = np.asarray(train_sizes_abs)

    if shuffle:
        with _rust_rng(random_state) as bridge:
            cv_iter = [
                (train[np.asarray(bridge.handle.permutation(len(train)))], test)
                for train, test in cv_iter
            ]

    work = [
        (train[:n_train], test)
        for train, test in cv_iter
        for n_train in train_sizes_abs
    ]
    results = _parallel(n_jobs, pre_dispatch, verbose)(
        _delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorers=scorers,
            train=train,
            test=test,
            fit_params=params,
            return_train_score=True,
            return_times=True,
            error_score=error_score,
        )
        for train, test in work
    )

    name = next(iter(scorers))
    n_ticks = len(train_sizes_abs)
    # `work` is fold-major; the public layout is tick-major, hence the transpose.
    train_scores = np.array([r["train_scores"][name] for r in results]).reshape(
        -1, n_ticks
    ).T
    test_scores = np.array([r["test_scores"][name] for r in results]).reshape(
        -1, n_ticks
    ).T
    if return_times:
        fit_times = np.array([r["fit_time"] for r in results]).reshape(-1, n_ticks).T
        score_times = np.array([r["score_time"] for r in results]).reshape(-1, n_ticks).T
        return train_sizes_abs, train_scores, test_scores, fit_times, score_times
    return train_sizes_abs, train_scores, test_scores


def validation_curve(
    estimator,
    X,
    y,
    *,
    param_name,
    param_range,
    groups=None,
    cv=None,
    scoring=None,
    n_jobs=None,
    pre_dispatch="all",
    verbose=0,
    error_score=np.nan,
    params=None,
):
    """Train and test scores as a function of one hyper-parameter.

    Returns
    -------
    train_scores, test_scores : ndarray of shape (n_param_values, n_cv_folds)
    """
    X, y, groups = _indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _resolve_scorers(estimator, scoring)
    cv_iter = list(cv.split(X, y, groups))

    results = _parallel(n_jobs, pre_dispatch, verbose)(
        _delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorers=scorers,
            train=train,
            test=test,
            parameters={param_name: value},
            fit_params=params,
            return_train_score=True,
            error_score=error_score,
        )
        for value in param_range
        for train, test in cv_iter
    )
    name = next(iter(scorers))
    n_folds = len(cv_iter)
    train_scores = np.array([r["train_scores"][name] for r in results]).reshape(
        -1, n_folds
    )
    test_scores = np.array([r["test_scores"][name] for r in results]).reshape(-1, n_folds)
    return train_scores, test_scores


def _permutation_test_score(estimator, X, y, groups, cv, scorers, params):
    """The mean cross-validated score for one (possibly permuted) target."""
    avg = []
    name = next(iter(scorers))
    for train, test in cv.split(X, y, groups):
        result = _fit_and_score(
            clone(estimator),
            X,
            y,
            scorers=scorers,
            train=train,
            test=test,
            fit_params=params,
            error_score="raise",
        )
        avg.append(result["test_scores"][name])
    return float(np.mean(avg))


def _shuffle_target(y, groups, rng_handle):
    """Permute ``y`` — WITHIN each group when ``groups`` is given.

    The group-wise variant is what keeps a grouped permutation test honest: a
    global shuffle would also destroy the group structure, so a low p-value
    could not be attributed to the label/feature relationship.
    """
    y = np.asarray(y)
    if groups is None:
        return y[np.asarray(rng_handle.permutation(len(y)), dtype=np.intp)]
    indices = np.arange(len(y))
    for group in np.unique(np.asarray(groups)):
        this = np.where(np.asarray(groups) == group)[0]
        indices[this] = this[np.asarray(rng_handle.permutation(len(this)), dtype=np.intp)]
    return y[indices]


def permutation_test_score(
    estimator,
    X,
    y,
    *,
    groups=None,
    cv=None,
    n_permutations=100,
    n_jobs=None,
    random_state=0,
    verbose=0,
    scoring=None,
    params=None,
):
    """Significance of a cross-validated score against permuted targets.

    Returns
    -------
    score : float
        The cross-validated score on the true target.
    permutation_scores : ndarray of shape (n_permutations,)
    pvalue : float
        ``(C + 1) / (n_permutations + 1)`` where ``C`` counts permutations
        scoring at least as well as the true target — so the best attainable
        p-value is ``1 / (n_permutations + 1)``, never 0.
    """
    X, y, groups = _indexable(X, y, groups)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorers, _ = _resolve_scorers(estimator, scoring)

    score = _permutation_test_score(clone(estimator), X, y, groups, cv, scorers, params)

    # The permutations are drawn UP FRONT, in order, so the target sequence does
    # not depend on how joblib happens to schedule the workers.
    with _rust_rng(random_state) as bridge:
        permuted_targets = [
            _shuffle_target(y, groups, bridge.handle) for _ in range(n_permutations)
        ]

    permutation_scores = _parallel(n_jobs, "2*n_jobs", verbose)(
        _delayed(_permutation_test_score)(
            clone(estimator), X, y_perm, groups, cv, scorers, params
        )
        for y_perm in permuted_targets
    )
    permutation_scores = np.array(permutation_scores)
    pvalue = _ext().permutation_pvalue(score, permutation_scores.tolist())
    return score, permutation_scores, pvalue

# --------------------------------------------------------------------------- #
# hyper-parameter search
# --------------------------------------------------------------------------- #


def _check_refit(search, attr):
    """Guard a delegated method behind ``refit`` having produced an estimator."""
    if not search.refit:
        raise AttributeError(
            f"This {type(search).__name__} instance was initialized with "
            f"`refit=False`. {attr} is available only after refitting on the "
            "best parameters. You can refit an estimator manually using the "
            "`best_params_` attribute"
        )


class BaseSearchCV(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Shared fit/refit/delegation machinery for the search estimators.

    Subclasses implement :meth:`_run_search`, which decides *which* candidates
    to evaluate and calls the supplied ``evaluate_candidates`` closure with
    them. Everything the closure does — scheduling folds, reducing scores,
    ranking, assembling ``cv_results_`` — is shared, and the reductions
    themselves run in Rust.
    """

    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    @abstractmethod
    def _run_search(self, evaluate_candidates):
        """Call ``evaluate_candidates(candidates)`` one or more times."""

    def _select_best_index(self, refit, refit_metric, results):
        """Which row of ``cv_results_`` wins. Overridden by the halving search."""
        if callable(refit):
            return int(refit(results))
        return int(_ext().best_index(list(results[f"mean_test_{refit_metric}"])))

    def _resolve_refit_metric(self, scorers, multimetric):
        if not multimetric:
            return "score"
        if self.refit is not False and not (
            isinstance(self.refit, str) and self.refit in scorers
        ) and not callable(self.refit):
            raise ValueError(
                "For multi-metric scoring, the parameter refit must be set to a "
                "scorer key or a callable to refit an estimator with the best "
                "parameter setting on the whole data and make the best_* "
                f"attributes available for that metric. {self.refit!r} was passed."
            )
        return self.refit if isinstance(self.refit, str) else next(iter(scorers))

    def fit(self, X, y=None, **params):
        """Run the search, then (unless ``refit=False``) refit on the full data.

        ``**params`` is forwarded to the estimator's ``fit``, except ``groups``,
        which goes to the splitter.
        """
        groups = params.pop("groups", None)
        estimator = self.estimator
        X, y, groups = _indexable(X, y, groups)

        scorers, multimetric = _resolve_scorers(estimator, self.scoring)
        refit_metric = self._resolve_refit_metric(scorers, multimetric)

        base_cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.multimetric_ = multimetric
        self.scorer_ = scorers if multimetric else scorers["score"]

        all_candidate_params = []
        all_out = []
        all_more_results = {}
        results = {}

        def evaluate_candidates(candidate_params, cv=None, more_results=None):
            """Score every candidate on every split and refresh ``cv_results_``.

            Returns the running results so a ``_run_search`` implementation
            (successive halving) can decide what to do next from them.
            """
            nonlocal results
            cv = base_cv if cv is None else cv
            candidate_params = list(candidate_params)
            splits = list(cv.split(X, y, groups))
            self.n_splits_ = len(splits)
            if not candidate_params:
                raise ValueError("No fits were performed. Was the CV iterator empty?")

            out = _parallel(self.n_jobs, self.pre_dispatch, self.verbose)(
                _delayed(_fit_and_score)(
                    clone(estimator),
                    X,
                    y,
                    scorers=scorers,
                    train=train,
                    test=test,
                    parameters=parameters,
                    fit_params=params,
                    return_train_score=self.return_train_score,
                    return_times=True,
                    return_parameters=False,
                    error_score=self.error_score,
                )
                for parameters in candidate_params
                for train, test in splits
            )
            if len(out) != len(candidate_params) * len(splits):
                raise ValueError(
                    "cv.split and cv.get_n_splits returned inconsistent results."
                )
            _warn_about_fit_failures(out, self.error_score)

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)
            for key, value in (more_results or {}).items():
                all_more_results.setdefault(key, []).extend(value)

            results = self._format_results(
                all_candidate_params, len(splits), all_out, all_more_results, scorers
            )
            return results

        self._run_search(evaluate_candidates)

        self.cv_results_ = results
        # With MULTIMETRIC scoring and `refit=False` there is no single metric
        # to be "best" by, so sklearn leaves the `best_*` attributes unset
        # rather than silently picking the first scorer — mirrored here, since
        # code that reads `best_score_` under those settings is asking the
        # wrong question and should fail rather than get an arbitrary answer.
        if self.refit or not multimetric:
            self.best_index_ = self._select_best_index(self.refit, refit_metric, results)
            if not callable(self.refit):
                self.best_score_ = results[f"mean_test_{refit_metric}"][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(estimator).set_params(
                **clone(self.best_params_, safe=False)
            )
            refit_start = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **params)
            else:
                self.best_estimator_.fit(X, **params)
            self.refit_time_ = time.time() - refit_start
            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_
        return self

    def _format_results(self, candidate_params, n_splits, out, more_results, scorers):
        """Assemble ``cv_results_`` from the flat per-(candidate, split) results.

        The mean/std/rank reduction is Rust's — including sklearn's two
        easy-to-miss rules: the std is a POPULATION std, and a NaN score
        (a failed fold) ranks below every finite score rather than being
        dropped.
        """
        n_candidates = len(candidate_params)
        results = {key: np.asarray(value) for key, value in more_results.items()}

        def reduce_column(values, key_name):
            flat = [float(v) for v in values]
            mean, std, rank = _ext().summarize_scores(flat, n_candidates, n_splits)
            for split_i in range(n_splits):
                results[f"split{split_i}_{key_name}"] = np.array(
                    [flat[c * n_splits + split_i] for c in range(n_candidates)]
                )
            results[f"mean_{key_name}"] = np.asarray(mean)
            results[f"std_{key_name}"] = np.asarray(std)
            return rank

        # timings: mean/std only (sklearn does not rank or split them out)
        for time_key in ("fit_time", "score_time"):
            values = [float(r[time_key]) for r in out]
            mean, std, _ = _ext().summarize_scores(values, n_candidates, n_splits)
            results[f"mean_{time_key}"] = np.asarray(mean)
            results[f"std_{time_key}"] = np.asarray(std)

        for name in scorers:
            key = f"test_{name}" if self.multimetric_ else "test_score"
            rank = reduce_column([r["test_scores"][name] for r in out], key)
            results[f"rank_{key}"] = np.asarray(rank, dtype=np.int32)
            if self.return_train_score:
                train_key = f"train_{name}" if self.multimetric_ else "train_score"
                reduce_column([r["train_scores"][name] for r in out], train_key)

        results["params"] = candidate_params
        # `param_<name>` columns are MASKED: a candidate from a different
        # sub-grid legitimately has no value for a parameter, and a masked entry
        # says "not applicable" where a `None` would say "explicitly None".
        param_names = sorted({k for params in candidate_params for k in params})
        for name in param_names:
            column = np.ma.MaskedArray(
                np.empty(n_candidates, dtype=object), mask=True, dtype=object
            )
            for i, candidate in enumerate(candidate_params):
                if name in candidate:
                    column[i] = candidate[name]
            results[f"param_{name}"] = column
        return results

    # ---- delegation to best_estimator_ ----

    def _delegate(self, name, X, **kwargs):
        check_is_fitted(self)
        _check_refit(self, name)
        return getattr(self.best_estimator_, name)(X, **kwargs)

    def predict(self, X):
        """Call ``predict`` on the refit best estimator."""
        return self._delegate("predict", X)

    def predict_proba(self, X):
        """Call ``predict_proba`` on the refit best estimator."""
        return self._delegate("predict_proba", X)

    def predict_log_proba(self, X):
        """Call ``predict_log_proba`` on the refit best estimator."""
        return self._delegate("predict_log_proba", X)

    def decision_function(self, X):
        """Call ``decision_function`` on the refit best estimator."""
        return self._delegate("decision_function", X)

    def transform(self, X):
        """Call ``transform`` on the refit best estimator."""
        return self._delegate("transform", X)

    def inverse_transform(self, X):
        """Call ``inverse_transform`` on the refit best estimator."""
        return self._delegate("inverse_transform", X)

    def score_samples(self, X):
        """Call ``score_samples`` on the refit best estimator."""
        return self._delegate("score_samples", X)

    def score(self, X, y=None, **params):
        """Score the refit best estimator with the search's own scorer."""
        check_is_fitted(self)
        _check_refit(self, "score")
        scorer = self.scorer_
        if self.multimetric_:
            metric = self.refit if isinstance(self.refit, str) else next(iter(scorer))
            scorer = scorer[metric]
        return scorer(self.best_estimator_, X, y, **params)

    @property
    def classes_(self):
        """The refit best estimator's classes."""
        _check_refit(self, "classes_")
        return self.best_estimator_.classes_

    @property
    def n_features_in_(self):
        """The refit best estimator's input feature count."""
        check_is_fitted(self, "best_estimator_" if self.refit else "cv_results_")
        return self.best_estimator_.n_features_in_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        sub = self.estimator.__sklearn_tags__()
        tags.estimator_type = sub.estimator_type
        tags.classifier_tags = sub.classifier_tags
        tags.regressor_tags = sub.regressor_tags
        return tags


def _warn_about_fit_failures(results, error_score):
    """Summarize the failed fits once, instead of once per fold."""
    from sklearn.exceptions import FitFailedWarning

    failed = [r["fit_error"] for r in results if r.get("fit_error") is not None]
    if not failed:
        return
    if error_score == "raise":  # pragma: no cover - re-raised at the fit site
        return
    warnings.warn(
        f"\n{len(failed)} fits failed out of a total of {len(results)}.\n"
        "The score on these train-test partitions for these parameters will be "
        f"set to {error_score}.\n"
        "If these failures are not expected, you can try to debug them by "
        "setting error_score='raise'.\n\n"
        f"Below is the first traceback:\n{failed[0]}",
        FitFailedWarning,
        stacklevel=2,
    )


class GridSearchCV(BaseSearchCV):
    """Exhaustive search over a parameter grid, cross-validated.

    Parameters mirror :class:`sklearn.model_selection.GridSearchCV`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from mlrs.model_selection import GridSearchCV
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(40, 3))
    >>> y = X @ [1.0, 2.0, 3.0]
    >>> search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0]}, cv=3).fit(X, y)
    >>> sorted(search.best_params_)
    ['alpha']
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Evaluate every point of the grid, in one batch."""
        evaluate_candidates(ParameterGrid(self.param_grid))


class RandomizedSearchCV(BaseSearchCV):
    """Randomized search over parameter distributions, cross-validated.

    Parameters mirror :class:`sklearn.model_selection.RandomizedSearchCV`.
    With an integer ``random_state`` the sampled candidates are identical to
    sklearn's, in the same order.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Evaluate ``n_iter`` sampled candidates, in one batch."""
        evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )


# --------------------------------------------------------------------------- #
# successive halving
# --------------------------------------------------------------------------- #


class _SubsampleMetaSplitter:
    """Wrap a splitter so each fold is subsampled to a fraction of its rows.

    This is how successive halving spends fewer *resources* on early rounds
    when ``resource="n_samples"``: the candidate set shrinks and the data grows.
    The subsample is a shuffle-then-truncate (sklearn's ``resample(replace=False)``),
    drawn from the search's own ``random_state`` — so with an int seed every
    round is reproducible.
    """

    def __init__(self, *, base_cv, fraction, subsample_test, random_state):
        self.base_cv = base_cv
        self.fraction = fraction
        self.subsample_test = subsample_test
        self.random_state = random_state

    def _resample(self, indices):
        n = int(self.fraction * indices.shape[0])
        with _rust_rng(self.random_state) as bridge:
            order = np.asarray(bridge.handle.permutation(indices.shape[0]), dtype=np.intp)
        return indices[order][:n]

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in self.base_cv.split(X, y, groups):
            train_idx = self._resample(np.asarray(train_idx))
            if self.subsample_test:
                test_idx = self._resample(np.asarray(test_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.base_cv.get_n_splits(X, y, groups)


class BaseSuccessiveHalving(BaseSearchCV):
    """Shared schedule and elimination logic for the two halving searches.

    Each round fits every surviving candidate on ``n_resources`` rows and keeps
    the top ``1/factor`` of them. The schedule — how many rounds, how many rows
    each, how many candidates survive — is derived in Rust from ``factor``,
    ``min_resources``, ``max_resources`` and ``aggressive_elimination``.
    """

    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=5,
        verbose=0,
        random_state=None,
        error_score=np.nan,
        return_train_score=True,
        max_resources="auto",
        min_resources="exhaust",
        resource="n_samples",
        factor=3,
        aggressive_elimination=False,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.random_state = random_state
        self.max_resources = max_resources
        self.resource = resource
        self.factor = factor
        self.min_resources = min_resources
        self.aggressive_elimination = aggressive_elimination

    def _select_best_index(self, refit, refit_metric, results):
        """The best candidate of the LAST round only.

        A candidate that scored well on 20 rows has not earned a win over one
        measured on 2000, so unlike :class:`BaseSearchCV` this does not rank
        across rounds.
        """
        last_iter = np.max(results["iter"])
        last_iter_rows = np.flatnonzero(results["iter"] == last_iter)
        scores = np.asarray(results[f"mean_test_{refit_metric}"])[last_iter_rows]
        return int(last_iter_rows[int(_ext().best_index(list(scores)))])

    def _check_input_parameters(self, X, y, groups, base_cv):
        if self.resource != "n_samples" and self.resource not in self.estimator.get_params():
            raise ValueError(
                f"Cannot use resource={self.resource} which is not supported "
                f"by estimator {type(self.estimator).__name__}"
            )
        self._n_samples_orig = _num_samples(X)

        self.max_resources_ = self.max_resources
        if self.max_resources_ == "auto":
            if self.resource != "n_samples":
                raise ValueError(
                    "resource can only be 'n_samples' when max_resources='auto'"
                )
            self.max_resources_ = self._n_samples_orig

        # `smallest` is `n_splits * 2`, times the class count for a classifier —
        # enough rows that every fold can hold at least a couple of each class.
        n_splits = base_cv.get_n_splits(X, y, groups)
        smallest = n_splits * 2
        if is_classifier(self.estimator) and y is not None:
            smallest *= len(np.unique(np.asarray(y)))
        if self.resource != "n_samples":
            smallest = 1
        self._smallest_resources = smallest

    def _run_search(self, evaluate_candidates):
        base_cv = check_cv(self.cv, self._y, classifier=is_classifier(self.estimator))
        candidate_params = list(self._generate_candidate_params())

        if self.resource != "n_samples" and any(
            self.resource in candidate for candidate in candidate_params
        ):
            raise ValueError(
                f"Cannot use parameter {self.resource} as the resource since "
                "it is part of the searched parameters."
            )

        kind = (
            "fixed"
            if isinstance(self.min_resources, numbers.Integral)
            else str(self.min_resources)
        )
        (
            self.min_resources_,
            self.n_required_iterations_,
            self.n_possible_iterations_,
            resources_per_iter,
            candidates_per_iter,
        ) = _ext().halving_schedule(
            len(candidate_params),
            self.factor,
            kind,
            int(self.min_resources) if kind == "fixed" else 0,
            int(self.max_resources_),
            bool(self.aggressive_elimination),
            int(self._smallest_resources),
        )
        self.n_iterations_ = len(resources_per_iter)
        self.n_resources_ = []
        self.n_candidates_ = []

        for itr, n_resources in enumerate(resources_per_iter):
            n_candidates = len(candidate_params)
            self.n_resources_.append(n_resources)
            self.n_candidates_.append(n_candidates)

            if self.resource == "n_samples":
                cv = _SubsampleMetaSplitter(
                    base_cv=base_cv,
                    fraction=n_resources / self._n_samples_orig,
                    # The TEST side is subsampled too, so the cost of a round is
                    # proportional to `n_resources` end to end; scoring every
                    # candidate on the full test set would defeat the point.
                    subsample_test=True,
                    random_state=self.random_state,
                )
            else:
                cv = base_cv
                candidate_params = [
                    {**candidate, self.resource: n_resources}
                    for candidate in candidate_params
                ]

            more_results = {
                "iter": [itr] * n_candidates,
                "n_resources": [n_resources] * n_candidates,
            }
            results = evaluate_candidates(
                candidate_params, cv=cv, more_results=more_results
            )

            n_to_keep = int(ceil(n_candidates / self.factor))
            this_iter = np.flatnonzero(np.asarray(results["iter"]) == itr)
            metric = "score" if not self.multimetric_ else next(iter(self.scorer_))
            scores = np.asarray(results[f"mean_test_{metric}"])[this_iter]
            keep = _ext().top_k(list(scores), n_to_keep)
            candidate_params = [results["params"][this_iter[i]] for i in keep]

    def fit(self, X, y=None, **params):
        """Run the halving search, then refit on the best of the LAST round."""
        groups = params.get("groups", None)
        base_cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self._check_input_parameters(X, y, groups, base_cv)
        # `_run_search` needs `y` to re-derive the (stratified) base splitter.
        self._y = y
        return super().fit(X, y, **params)


class HalvingGridSearchCV(BaseSuccessiveHalving):
    """Grid search with successive halving.

    Rounds start with every grid point on a small slice of the data and end with
    a few candidates on all of it. Parameters mirror
    :class:`sklearn.model_selection.HalvingGridSearchCV`.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        *,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        self.param_grid = param_grid

    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)


class HalvingRandomSearchCV(BaseSuccessiveHalving):
    """Randomized search with successive halving.

    ``n_candidates="exhaust"`` draws ``max_resources_ // min_resources_``
    candidates — the number whose last round can use the whole budget.
    Parameters mirror :class:`sklearn.model_selection.HalvingRandomSearchCV`.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_candidates="exhaust",
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="smallest",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        super().__init__(
            estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates

    def _check_input_parameters(self, X, y, groups, base_cv):
        if self.min_resources == self.n_candidates == "exhaust":
            # Each would be derived from the other — there is nothing to solve.
            raise ValueError(
                "n_candidates and min_resources cannot be both set to 'exhaust'."
            )
        super()._check_input_parameters(X, y, groups, base_cv)

    def _generate_candidate_params(self):
        n_candidates = self.n_candidates
        if n_candidates == "exhaust":
            min_resources = (
                int(self.min_resources)
                if isinstance(self.min_resources, numbers.Integral)
                else self._smallest_resources
            )
            n_candidates = _ext().exhaust_n_candidates(
                int(self.max_resources_), int(min_resources)
            )
        return ParameterSampler(
            self.param_distributions, n_candidates, random_state=self.random_state
        )

# --------------------------------------------------------------------------- #
# decision-threshold classifiers
# --------------------------------------------------------------------------- #


def _resolve_response_method(estimator, response_method):
    """Pick the scoring method to threshold on.

    ``"auto"`` prefers ``predict_proba`` and falls back to
    ``decision_function``, matching sklearn — the two live on different scales
    (a probability in ``[0, 1]`` versus an unbounded margin), which is why the
    default threshold below depends on which one was chosen.
    """
    if response_method == "auto":
        candidates = ["predict_proba", "decision_function"]
    elif isinstance(response_method, str):
        candidates = [response_method]
    else:
        candidates = list(response_method)
    for name in candidates:
        if hasattr(estimator, name):
            return name
    raise AttributeError(
        f"{type(estimator).__name__} has none of the following attributes: "
        f"{', '.join(candidates)}."
    )


def _binary_scores(estimator, X, response_method, pos_label=None):
    """The per-row score of the positive class, as a 1-D array."""
    values = getattr(estimator, response_method)(X)
    values = np.asarray(values)
    if response_method == "predict_proba":
        classes = np.asarray(estimator.classes_)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError(
                "Decision-threshold tuning is only defined for binary "
                f"classification; got {values.shape[1] if values.ndim == 2 else 1} "
                "columns from predict_proba."
            )
        col = 1 if pos_label is None else int(np.flatnonzero(classes == pos_label)[0])
        return values[:, col]
    return values.ravel()


def _threshold_to_labels(scores, threshold, classes, pos_label=None):
    """Map thresholded scores back to the estimator's own class labels."""
    classes = np.asarray(classes)
    if pos_label is None:
        mapping = np.array([0, 1])
    else:
        pos_idx = int(np.flatnonzero(classes == pos_label)[0])
        neg_idx = int(np.flatnonzero(classes != pos_label)[0])
        mapping = np.array([neg_idx, pos_idx])
    return classes[mapping[(np.asarray(scores) >= threshold).astype(int)]]


class FixedThresholdClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Binary classifier whose decision threshold is set by the caller.

    Parameters
    ----------
    estimator : estimator
        A binary classifier exposing ``predict_proba`` or ``decision_function``.
    threshold : {"auto"} or float, default="auto"
        ``"auto"`` is 0.5 for a probability response and 0.0 for a decision
        function — the two natural neutral points, which is why the default is
        response-dependent rather than a single number.
    pos_label : int, float, bool or str, default=None
        Which class counts as positive. ``None`` means the second class in
        ``classes_``.
    response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
    """

    def __init__(self, estimator, *, threshold="auto", pos_label=None, response_method="auto"):
        self.estimator = estimator
        self.threshold = threshold
        self.pos_label = pos_label
        self.response_method = response_method

    def fit(self, X, y, **params):
        """Fit the wrapped estimator; the threshold itself needs no fitting."""
        self.estimator_ = clone(self.estimator).fit(X, y, **params)
        self.classes_ = self.estimator_.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                "Only binary classification is supported. Got "
                f"{len(self.classes_)} classes."
            )
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self

    def _effective_threshold(self, response_method):
        if self.threshold != "auto":
            return float(self.threshold)
        return 0.5 if response_method == "predict_proba" else 0.0

    def predict(self, X):
        """Predict class labels at the configured threshold."""
        check_is_fitted(self, "estimator_")
        method = _resolve_response_method(self.estimator_, self.response_method)
        scores = _binary_scores(self.estimator_, X, method, self.pos_label)
        threshold = self._effective_threshold(method)
        # The comparison itself is Rust's `>= threshold` (a tie is POSITIVE),
        # so the boundary rule is defined in exactly one place.
        indicator = np.asarray(_ext().apply_threshold(scores.tolist(), threshold))
        classes = np.asarray(self.classes_)
        if self.pos_label is None:
            return classes[indicator]
        pos_idx = int(np.flatnonzero(classes == self.pos_label)[0])
        neg_idx = int(np.flatnonzero(classes != self.pos_label)[0])
        return classes[np.array([neg_idx, pos_idx])[indicator]]

    def predict_proba(self, X):
        """Delegate to the wrapped estimator — thresholding does not change it."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        """Delegate to the wrapped estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_log_proba(X)

    def decision_function(self, X):
        """Delegate to the wrapped estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(X)


class TunedThresholdClassifierCV(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Binary classifier whose decision threshold is tuned by cross-validation.

    Parameters
    ----------
    estimator : estimator
    scoring : str or callable, default="balanced_accuracy"
        The objective the threshold maximizes.
    response_method : {"auto", "decision_function", "predict_proba"}, default="auto"
    thresholds : int or array-like, default=100
        An int builds a ``linspace`` over the union of the folds' score ranges;
        an array is used verbatim.
    cv : int, float, cross-validation generator, iterable or "prefit", default=None
        ``None`` is 5-fold stratified. A float is a single stratified split with
        that test fraction. ``"prefit"`` skips fitting and tunes on the data
        passed to :meth:`fit` — which must therefore be held out.
    refit : bool, default=True
        Refit the estimator on the whole dataset after tuning.
    n_jobs : int, default=None
    random_state : int, RandomState instance or None, default=None
    store_cv_results : bool, default=False
        Keep the threshold/score curve in ``cv_results_``.

    Notes
    -----
    Each fold produces its own threshold vector (the distinct scores its
    validation rows happened to take), so the folds cannot be averaged
    elementwise. The curves are interpolated onto one common grid first — in
    Rust, with numpy's clamped-endpoint ``interp`` semantics — and only then
    averaged.
    """

    def __init__(
        self,
        estimator,
        *,
        scoring="balanced_accuracy",
        response_method="auto",
        thresholds=100,
        cv=None,
        refit=True,
        n_jobs=None,
        random_state=None,
        store_cv_results=False,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.response_method = response_method
        self.thresholds = thresholds
        self.cv = cv
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.store_cv_results = store_cv_results

    def _curve_for_split(self, estimator, X_val, y_val, score_func, sign):
        """One fold's ``(thresholds, objective_scores)`` curve."""
        method = _resolve_response_method(estimator, self.response_method)
        scores = _binary_scores(estimator, X_val, method)
        if isinstance(self.thresholds, numbers.Integral):
            grid = np.linspace(np.min(scores), np.max(scores), int(self.thresholds))
        else:
            grid = np.asarray(self.thresholds, dtype=float)
        objective = [
            sign * score_func(y_val, _threshold_to_labels(scores, th, estimator.classes_))
            for th in grid
        ]
        return np.asarray(grid, dtype=float), np.asarray(objective, dtype=float)

    def _resolve_cv(self, X, y):
        if isinstance(self.cv, numbers.Real) and not isinstance(self.cv, numbers.Integral):
            return StratifiedShuffleSplit(
                n_splits=1, test_size=float(self.cv), random_state=self.random_state
            )
        if self.cv is None:
            return StratifiedKFold(n_splits=5)
        return check_cv(self.cv, y, classifier=True)

    def fit(self, X, y, **params):
        """Tune the threshold by cross-validation, then (optionally) refit."""
        from sklearn.metrics import get_scorer

        X, y, _ = _indexable(X, y, None)
        classes = np.unique(np.asarray(y))
        if len(classes) != 2:
            raise ValueError(
                "Only binary classification is supported. Got "
                f"{len(classes)} classes."
            )

        scorer = get_scorer(self.scoring) if isinstance(self.scoring, str) else self.scoring
        score_func = getattr(scorer, "_score_func", None)
        sign = getattr(scorer, "_sign", 1)
        if score_func is None:
            raise ValueError(
                "TunedThresholdClassifierCV needs a metric it can evaluate at "
                "many thresholds; pass a scoring string (or a scorer built by "
                "sklearn.metrics.make_scorer), not a bare callable."
            )

        if self.cv == "prefit":
            check_is_fitted(self.estimator)
            self.estimator_ = self.estimator
            curves = [self._curve_for_split(self.estimator_, X, y, score_func, sign)]
        else:
            cv = self._resolve_cv(X, y)
            curves = []
            for train, val in cv.split(X, y):
                fitted = clone(self.estimator).fit(
                    _safe_indexing(X, train), _safe_indexing(y, train), **params
                )
                curves.append(
                    self._curve_for_split(
                        fitted,
                        _safe_indexing(X, val),
                        _safe_indexing(y, val),
                        score_func,
                        sign,
                    )
                )
            self.estimator_ = (
                clone(self.estimator).fit(X, y, **params) if self.refit else fitted
            )

        grid_kwargs = (
            {"grid_count": int(self.thresholds)}
            if isinstance(self.thresholds, numbers.Integral)
            else {"grid_explicit": [float(v) for v in np.asarray(self.thresholds)]}
        )
        best_threshold, best_score, thresholds, scores = _ext().tune_threshold(
            [t.tolist() for t, _ in curves],
            [s.tolist() for _, s in curves],
            **grid_kwargs,
        )
        self.best_threshold_ = float(best_threshold)
        self.best_score_ = float(best_score)
        if self.store_cv_results:
            self.cv_results_ = {
                "thresholds": np.asarray(thresholds),
                "scores": np.asarray(scores),
            }

        self.classes_ = np.asarray(self.estimator_.classes_)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        return self

    def predict(self, X):
        """Predict class labels at the tuned threshold."""
        check_is_fitted(self, "best_threshold_")
        method = _resolve_response_method(self.estimator_, self.response_method)
        scores = _binary_scores(self.estimator_, X, method)
        indicator = np.asarray(
            _ext().apply_threshold(scores.tolist(), self.best_threshold_)
        )
        return self.classes_[indicator]

    def predict_proba(self, X):
        """Delegate to the wrapped estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def predict_log_proba(self, X):
        """Delegate to the wrapped estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_log_proba(X)

    def decision_function(self, X):
        """Delegate to the wrapped estimator."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(X)


# --------------------------------------------------------------------------- #
# curve displays
# --------------------------------------------------------------------------- #


class _CurveDisplayMixin:
    """Shared plotting for the two curve displays.

    matplotlib is imported LAZILY, inside :meth:`plot`. It is not an mlrs
    dependency, and importing it at module scope would make
    ``import mlrs.model_selection`` fail — or pull in a GUI backend — on a
    headless install that only ever wanted to split some data.
    """

    def _plot_curve(self, x_values, x_label, *, ax=None, negate_score=False,
                    score_name=None, score_type="both", std_display_style="fill_between",
                    line_kw=None, fill_between_kw=None, errorbar_kw=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        self.score_name = score_name or self.score_name or "Score"
        line_kw = {} if line_kw is None else line_kw
        fill_between_kw = {"alpha": 0.5} if fill_between_kw is None else fill_between_kw
        errorbar_kw = {} if errorbar_kw is None else errorbar_kw

        wanted = {"train": "Train", "test": "Test"}
        if score_type == "train":
            wanted = {"train": "Train"}
        elif score_type == "test":
            wanted = {"test": "Test"}

        self.lines_, self.errorbar_, self.fill_between_ = [], [], []
        for key, label in wanted.items():
            scores = self.train_scores if key == "train" else self.test_scores
            scores = -np.asarray(scores) if negate_score else np.asarray(scores)
            mean, std = scores.mean(axis=1), scores.std(axis=1)
            if std_display_style == "errorbar":
                container = ax.errorbar(x_values, mean, std, label=label, **errorbar_kw)
                self.errorbar_.append(container)
            else:
                (line,) = ax.plot(x_values, mean, label=label, **line_kw)
                self.lines_.append(line)
                if std_display_style == "fill_between":
                    self.fill_between_.append(
                        ax.fill_between(
                            x_values,
                            mean - std,
                            mean + std,
                            color=line.get_color(),
                            **fill_between_kw,
                        )
                    )
        ax.set_xlabel(x_label)
        ax.set_ylabel(self.score_name)
        ax.legend()
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


class LearningCurveDisplay(_CurveDisplayMixin):
    """Learning-curve visualization.

    Parameters
    ----------
    train_sizes : ndarray of shape (n_ticks,)
    train_scores, test_scores : ndarray of shape (n_ticks, n_cv_folds)
    score_name : str, default=None

    Examples
    --------
    >>> from sklearn.linear_model import Ridge  # doctest: +SKIP
    >>> from mlrs.model_selection import LearningCurveDisplay  # doctest: +SKIP
    >>> LearningCurveDisplay.from_estimator(Ridge(), X, y, cv=3)  # doctest: +SKIP
    """

    def __init__(self, *, train_sizes, train_scores, test_scores, score_name=None):
        self.train_sizes = train_sizes
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.score_name = score_name

    def plot(self, ax=None, *, negate_score=False, score_name=None, score_type="both",
             std_display_style="fill_between", line_kw=None, fill_between_kw=None,
             errorbar_kw=None):
        """Draw the curve onto ``ax`` (a fresh figure when ``ax`` is ``None``)."""
        return self._plot_curve(
            self.train_sizes,
            "Number of samples in the training set",
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )

    @classmethod
    def from_estimator(cls, estimator, X, y, *, groups=None,
                       train_sizes=np.linspace(0.1, 1.0, 5), cv=None, scoring=None,
                       exploit_incremental_learning=False, n_jobs=None,
                       pre_dispatch="all", verbose=0, shuffle=False, random_state=None,
                       error_score=np.nan, fit_params=None, ax=None, negate_score=False,
                       score_name=None, score_type="both",
                       std_display_style="fill_between", line_kw=None,
                       fill_between_kw=None, errorbar_kw=None):
        """Compute the learning curve with :func:`learning_curve`, then plot it."""
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator,
            X,
            y,
            groups=groups,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            exploit_incremental_learning=exploit_incremental_learning,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
            shuffle=shuffle,
            random_state=random_state,
            error_score=error_score,
            params=fit_params,
        )
        display = cls(
            train_sizes=train_sizes_abs,
            train_scores=train_scores,
            test_scores=test_scores,
            score_name=score_name,
        )
        return display.plot(
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )


class ValidationCurveDisplay(_CurveDisplayMixin):
    """Validation-curve visualization.

    Parameters
    ----------
    param_name : str
    param_range : array-like
    train_scores, test_scores : ndarray of shape (n_param_values, n_cv_folds)
    score_name : str, default=None
    """

    def __init__(self, *, param_name, param_range, train_scores, test_scores, score_name=None):
        self.param_name = param_name
        self.param_range = param_range
        self.train_scores = train_scores
        self.test_scores = test_scores
        self.score_name = score_name

    def plot(self, ax=None, *, negate_score=False, score_name=None, score_type="both",
             std_display_style="fill_between", line_kw=None, fill_between_kw=None,
             errorbar_kw=None):
        """Draw the curve onto ``ax`` (a fresh figure when ``ax`` is ``None``)."""
        return self._plot_curve(
            self.param_range,
            f"{self.param_name}",
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )

    @classmethod
    def from_estimator(cls, estimator, X, y, *, param_name, param_range, groups=None,
                       cv=None, scoring=None, n_jobs=None, pre_dispatch="all",
                       verbose=0, error_score=np.nan, fit_params=None, ax=None,
                       negate_score=False, score_name=None, score_type="both",
                       std_display_style="fill_between", line_kw=None,
                       fill_between_kw=None, errorbar_kw=None):
        """Compute the validation curve with :func:`validation_curve`, then plot it."""
        train_scores, test_scores = validation_curve(
            estimator,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
            error_score=error_score,
            params=fit_params,
        )
        display = cls(
            param_name=param_name,
            param_range=param_range,
            train_scores=train_scores,
            test_scores=test_scores,
            score_name=score_name,
        )
        return display.plot(
            ax=ax,
            negate_score=negate_score,
            score_name=score_name,
            score_type=score_type,
            std_display_style=std_display_style,
            line_kw=line_kw,
            fill_between_kw=fill_between_kw,
            errorbar_kw=errorbar_kw,
        )
