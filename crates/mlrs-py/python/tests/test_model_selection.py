"""``mlrs.model_selection`` tests (MODSEL-01 / MODSEL-02).

Covers :func:`train_test_split`, :class:`KFold`, :class:`StratifiedKFold`, and
the ``GridSearchCV`` / ``RandomizedSearchCV`` / ``cross_val_score`` /
``cross_validate`` sklearn-delegation passthrough.

Four gates, matching the roadmap's Phase-24 criterion 4 ("gated structurally
with a recorded MT19937-host-match decision, and run GridSearchCV/
RandomizedSearchCV over mlrs estimators via the sklearn-delegation
passthrough"):

1. **Bit-for-bit oracle** — the recorded decision is *host-match*, so the oracle
   is ``sklearn.model_selection.train_test_split`` itself, called live with the
   same arguments. Every parameter combination must select the SAME ROWS, not
   merely a same-sized/same-balance split. No stored ``.npz`` fixture: an index
   permutation is exactly reproducible, so a live compare is strictly stronger
   (it re-checks parity against the installed sklearn on every run).
2. **Structural** — sizes, disjointness (no leakage), coverage, and stratum
   balance, asserted independently of sklearn so a *shared* misunderstanding
   between the two implementations still fails.
3. **Container** — pandas / polars / pyarrow / list / range / sparse round-trip
   with the container type preserved and the right rows in the right order.
4. **Interop** — the mlrs splitters survive sklearn's ``check_cv`` unchanged and
   actually drive sklearn's ``cross_val_score`` / ``GridSearchCV``, which is
   what makes the passthrough a real integration rather than a re-export.

These exercise the PURE-PYTHON layer only — ``mlrs.model_selection`` never
touches the compiled ``_mlrs`` extension — so the whole file runs on a tree
where ``maturin develop`` has not been run.
"""

import numpy as np
import pyarrow as pa  # a hard mlrs runtime dependency, never optional
import pytest
import sklearn.model_selection as skm
from sklearn.model_selection import KFold as sk_KFold
from sklearn.model_selection import StratifiedKFold as sk_StratifiedKFold
from sklearn.model_selection import train_test_split as sk_train_test_split

from mlrs.model_selection import (
    GridSearchCV,
    InvalidParameterError,
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

# pandas and polars are OPTIONAL for mlrs (`model_selection` detects them via
# sys.modules and never imports them). Skip only the tests that need them —
# a module-level `importorskip` would take the parity and structural gates down
# with them, which are the gates that actually protect the split logic.
try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised on a pandas-free install
    pd = None
try:
    import polars as pl
except ImportError:  # pragma: no cover - exercised on a polars-free install
    pl = None

needs_pandas = pytest.mark.skipif(pd is None, reason="pandas is not installed")
needs_polars = pytest.mark.skipif(pl is None, reason="polars is not installed")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

N = 120


@pytest.fixture
def xy():
    """A deterministic (X, y) pair: 120 rows, 4 features, 3 imbalanced classes."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 4))
    y = np.repeat([0, 1, 2], [60, 40, 20])
    return X, y


def _rows(a):
    """Row indices recovered from a split of ``np.arange(n)``-tagged data."""
    return np.asarray(a).ravel().astype(np.int64)


# --------------------------------------------------------------------------- #
# 1. bit-for-bit parity vs sklearn (the recorded MT19937-host-match decision)
# --------------------------------------------------------------------------- #

# Every parameter combination the function accepts, crossed with several seeds.
_PARITY_CASES = [
    pytest.param({}, id="all-defaults"),
    pytest.param({"test_size": 0.2}, id="test_size-float"),
    pytest.param({"test_size": 0.33}, id="test_size-float-ceil"),
    pytest.param({"test_size": 7}, id="test_size-int"),
    pytest.param({"train_size": 0.6}, id="train_size-float-floor"),
    pytest.param({"train_size": 90}, id="train_size-int"),
    pytest.param({"train_size": 0.5, "test_size": 0.25}, id="both-float-subsample"),
    pytest.param({"train_size": 50, "test_size": 30}, id="both-int-subsample"),
    pytest.param({"train_size": 0.7, "test_size": 20}, id="mixed-float-int"),
    pytest.param({"shuffle": False}, id="shuffle-False"),
    pytest.param({"shuffle": False, "test_size": 0.1}, id="shuffle-False-sized"),
]


@pytest.mark.parametrize("kwargs", _PARITY_CASES)
@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_parity_with_sklearn_unstratified(xy, kwargs, seed):
    X, y = xy
    got = train_test_split(X, y, random_state=seed, **kwargs)
    want = sk_train_test_split(X, y, random_state=seed, **kwargs)
    assert len(got) == len(want) == 4
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="all-defaults"),
        pytest.param({"test_size": 0.2}, id="test_size-float"),
        pytest.param({"test_size": 30}, id="test_size-int"),
        pytest.param({"train_size": 0.5}, id="train_size-float"),
        pytest.param({"train_size": 60, "test_size": 30}, id="both-int-subsample"),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_parity_with_sklearn_stratified(xy, kwargs, seed):
    X, y = xy
    got = train_test_split(X, y, random_state=seed, stratify=y, **kwargs)
    want = sk_train_test_split(X, y, random_state=seed, stratify=y, **kwargs)
    assert len(got) == len(want) == 4
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_parity_stratified_ties_break_identically():
    """Equal class counts make `_approximate_mode`'s tie-break RNG draw load-
    bearing; if mlrs skipped it the stream would desync and rows would differ."""
    y = np.repeat([0, 1, 2, 3], 10)
    idx = np.arange(y.size)
    for seed in range(12):
        got = train_test_split(idx, random_state=seed, stratify=y, test_size=0.5)
        want = sk_train_test_split(idx, random_state=seed, stratify=y, test_size=0.5)
        assert len(got) == len(want) == 2
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])


def test_parity_multilabel_stratify():
    """A 2-D `stratify` strata-fies on the label COMBINATION, like sklearn."""
    rng = np.random.default_rng(7)
    y2d = rng.integers(0, 2, size=(90, 3))
    idx = np.arange(90)
    got = train_test_split(idx, random_state=3, stratify=y2d, test_size=0.3)
    want = sk_train_test_split(idx, random_state=3, stratify=y2d, test_size=0.3)
    assert len(got) == len(want) == 2
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_parity_with_randomstate_instance(xy):
    """A `RandomState` INSTANCE advances across calls — both sides identically."""
    X, y = xy
    mine = np.random.RandomState(11)
    theirs = np.random.RandomState(11)
    for _ in range(3):
        got = train_test_split(X, y, random_state=mine, test_size=0.25)
        want = sk_train_test_split(X, y, random_state=theirs, test_size=0.25)
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g, w)


def test_parity_single_array(xy):
    X, _ = xy
    got = train_test_split(X, random_state=5)
    want = sk_train_test_split(X, random_state=5)
    assert len(got) == 2
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def test_parity_many_arrays(xy):
    X, y = xy
    w = np.linspace(0.0, 1.0, N)
    got = train_test_split(X, y, w, random_state=9, test_size=0.4)
    want = sk_train_test_split(X, y, w, random_state=9, test_size=0.4)
    assert len(got) == 6
    for g, wv in zip(got, want):
        np.testing.assert_array_equal(g, wv)


# --------------------------------------------------------------------------- #
# 2. structural gates (independent of sklearn)
# --------------------------------------------------------------------------- #


def test_structural_sizes_and_no_leakage(xy):
    X, y = xy
    idx = np.arange(N)
    tr, te = train_test_split(idx, random_state=0, test_size=0.25)
    assert len(te) == 30  # ceil(0.25 * 120)
    assert len(tr) == 90
    assert set(tr).isdisjoint(set(te))
    assert set(tr) | set(te) == set(range(N))


def test_structural_subsample_leaves_rows_unallocated():
    idx = np.arange(N)
    tr, te = train_test_split(idx, random_state=0, train_size=50, test_size=30)
    assert len(tr) == 50 and len(te) == 30
    assert set(tr).isdisjoint(set(te))
    assert len(set(tr) | set(te)) == 80  # 40 rows in neither split


def test_structural_float_rounding_is_ceil_test_floor_train():
    """`test_size` rounds UP, `train_size` rounds DOWN — the asymmetry matters
    for odd row counts."""
    idx = np.arange(10)
    _, te = train_test_split(idx, random_state=0, test_size=0.33)
    assert len(te) == 4  # ceil(3.3)
    tr, _ = train_test_split(idx, random_state=0, train_size=0.33)
    assert len(tr) == 3  # floor(3.3)


def test_structural_shuffle_false_is_head_and_tail():
    idx = np.arange(10)
    tr, te = train_test_split(idx, shuffle=False, test_size=0.3)
    np.testing.assert_array_equal(tr, np.arange(7))
    np.testing.assert_array_equal(te, np.arange(7, 10))


def test_structural_shuffle_true_actually_shuffles():
    idx = np.arange(N)
    tr, _ = train_test_split(idx, random_state=0)
    assert not np.array_equal(np.sort(tr), tr), "train indices came out ordered"


def test_structural_stratify_preserves_class_balance(xy):
    _, y = xy
    idx = np.arange(N)
    tr, te = train_test_split(idx, random_state=0, test_size=0.25, stratify=y)
    for split in (tr, te):
        counts = np.bincount(y[split], minlength=3)
        share = counts / counts.sum()
        # population shares are 0.5 / 0.3333 / 0.1667
        np.testing.assert_allclose(share, [0.5, 1 / 3, 1 / 6], atol=0.02)
    assert set(tr).isdisjoint(set(te))


def test_structural_random_state_none_varies(xy):
    X, _ = xy
    idx = np.arange(N)
    a, _ = train_test_split(idx)
    b, _ = train_test_split(idx)
    assert not np.array_equal(a, b)


def test_structural_same_seed_is_reproducible(xy):
    X, y = xy
    first = train_test_split(X, y, random_state=77, test_size=0.3, stratify=y)
    second = train_test_split(X, y, random_state=77, test_size=0.3, stratify=y)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# 3. container support — pandas / polars / pyarrow / list / range / sparse
# --------------------------------------------------------------------------- #


def _expected_rows(random_state, n=N, **kwargs):
    """The row indices mlrs will select, obtained by splitting `arange(n)`."""
    return train_test_split(np.arange(n), random_state=random_state, **kwargs)


def test_container_numpy_2d_and_1d(xy):
    X, y = xy
    tr_i, te_i = _expected_rows(4, test_size=0.25)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=4, test_size=0.25)
    assert isinstance(Xtr, np.ndarray) and Xtr.shape == (90, 4)
    np.testing.assert_array_equal(Xtr, X[tr_i])
    np.testing.assert_array_equal(Xte, X[te_i])
    np.testing.assert_array_equal(ytr, y[tr_i])
    np.testing.assert_array_equal(yte, y[te_i])


@needs_pandas
def test_container_pandas_dataframe_and_series(xy):
    X, y = xy
    df = pd.DataFrame(X, columns=list("abcd"))
    ser = pd.Series(y, name="target")
    tr_i, te_i = _expected_rows(4, test_size=0.25)

    dtr, dte, str_, ste = train_test_split(df, ser, random_state=4, test_size=0.25)
    assert isinstance(dtr, pd.DataFrame) and isinstance(str_, pd.Series)
    assert list(dtr.columns) == list("abcd")
    # positional take -> ORIGINAL index labels are carried along, not reset
    np.testing.assert_array_equal(dtr.index.to_numpy(), tr_i)
    np.testing.assert_array_equal(dte.index.to_numpy(), te_i)
    np.testing.assert_allclose(dtr.to_numpy(), X[tr_i])
    np.testing.assert_array_equal(ste.to_numpy(), y[te_i])
    assert str_.name == "target"


@needs_pandas
def test_container_pandas_non_default_index_is_preserved(xy):
    X, _ = xy
    labels = [f"row{i}" for i in range(N)]
    df = pd.DataFrame(X, index=labels)
    tr_i, _ = _expected_rows(1, test_size=0.5)
    dtr, _ = train_test_split(df, random_state=1, test_size=0.5)
    assert list(dtr.index) == [labels[i] for i in tr_i]


@needs_polars
def test_container_polars_dataframe_and_series(xy):
    X, y = xy
    df = pl.DataFrame({f"c{j}": X[:, j] for j in range(4)})
    ser = pl.Series("target", y)
    tr_i, te_i = _expected_rows(4, test_size=0.25)

    dtr, dte, str_, ste = train_test_split(df, ser, random_state=4, test_size=0.25)
    assert isinstance(dtr, pl.DataFrame) and isinstance(str_, pl.Series)
    assert dtr.shape == (90, 4) and dte.shape == (30, 4)
    assert dtr.columns == [f"c{j}" for j in range(4)]
    np.testing.assert_allclose(dtr.to_numpy(), X[tr_i])
    np.testing.assert_allclose(dte.to_numpy(), X[te_i])
    np.testing.assert_array_equal(str_.to_numpy(), y[tr_i])
    np.testing.assert_array_equal(ste.to_numpy(), y[te_i])
    assert str_.name == "target"


@needs_pandas
@needs_polars
def test_container_polars_matches_pandas_rows(xy):
    """The same split applied to the same data in two containers must select
    identical rows — the gather is container-specific, the indices are not."""
    X, y = xy
    df_pd = pd.DataFrame(X)
    df_pl = pl.DataFrame({f"c{j}": X[:, j] for j in range(4)})
    ptr, pte = train_test_split(df_pd, random_state=8, test_size=0.3, stratify=y)
    ltr, lte = train_test_split(df_pl, random_state=8, test_size=0.3, stratify=y)
    np.testing.assert_allclose(ptr.to_numpy(), ltr.to_numpy())
    np.testing.assert_allclose(pte.to_numpy(), lte.to_numpy())


def test_container_pyarrow_table_array_chunked(xy):
    X, y = xy
    table = pa.table({f"c{j}": X[:, j] for j in range(4)})
    arr = pa.array(y)
    chunked = pa.chunked_array([y[:50], y[50:]])
    tr_i, te_i = _expected_rows(4, test_size=0.25)

    ttr, tte, atr, ate, ctr, cte = train_test_split(
        table, arr, chunked, random_state=4, test_size=0.25
    )
    assert isinstance(ttr, pa.Table)
    assert isinstance(atr, pa.Array)
    assert isinstance(ctr, pa.ChunkedArray)
    np.testing.assert_allclose(ttr.column("c0").to_numpy(), X[tr_i, 0])
    np.testing.assert_allclose(tte.column("c3").to_numpy(), X[te_i, 3])
    np.testing.assert_array_equal(atr.to_numpy(), y[tr_i])
    np.testing.assert_array_equal(cte.to_numpy(), y[te_i])


def test_container_python_list_and_range():
    data = list("abcdefghij")
    tr, te = train_test_split(data, random_state=0, test_size=0.3)
    assert isinstance(tr, list) and isinstance(te, list)
    assert len(tr) == 7 and len(te) == 3
    assert set(tr) | set(te) == set(data)

    rtr, rte = train_test_split(range(10), shuffle=False, test_size=0.3)
    assert rtr == [0, 1, 2, 3, 4, 5, 6] and rte == [7, 8, 9]


def test_container_scipy_sparse(xy):
    sparse = pytest.importorskip("scipy.sparse")
    X, _ = xy
    csr = sparse.csr_matrix(X)
    tr_i, te_i = _expected_rows(4, test_size=0.25)
    Str, Ste = train_test_split(csr, random_state=4, test_size=0.25)
    assert sparse.issparse(Str)
    np.testing.assert_allclose(Str.toarray(), X[tr_i])
    np.testing.assert_allclose(Ste.toarray(), X[te_i])


def test_container_coo_sparse_is_converted_to_csr(xy):
    """COO cannot be row-sliced; `_make_indexable` converts it to CSR first."""
    sparse = pytest.importorskip("scipy.sparse")
    X, _ = xy
    coo = sparse.coo_matrix(X)
    tr_i, _ = _expected_rows(4, test_size=0.25)
    Str, _ = train_test_split(coo, random_state=4, test_size=0.25)
    np.testing.assert_allclose(Str.toarray(), X[tr_i])


@needs_pandas
@needs_polars
def test_container_mixed_in_one_call(xy):
    """numpy + pandas + polars + pyarrow + list in a single split — every output
    keeps its own type and all five select the SAME rows."""
    X, y = xy
    df_pd = pd.DataFrame(X)
    df_pl = pl.DataFrame({"v": y})
    tbl = pa.table({"v": y})
    lst = list(range(N))
    tr_i, te_i = _expected_rows(2, test_size=0.2)

    out = train_test_split(X, df_pd, df_pl, tbl, lst, random_state=2, test_size=0.2)
    assert len(out) == 10
    np_tr, np_te, pd_tr, _, pl_tr, _, pa_tr, _, l_tr, _ = out
    assert isinstance(pd_tr, pd.DataFrame)
    assert isinstance(pl_tr, pl.DataFrame)
    assert isinstance(pa_tr, pa.Table)
    assert isinstance(l_tr, list)
    np.testing.assert_allclose(np_tr, X[tr_i])
    np.testing.assert_allclose(pd_tr.to_numpy(), X[tr_i])
    np.testing.assert_array_equal(pl_tr["v"].to_numpy(), y[tr_i])
    np.testing.assert_array_equal(pa_tr.column("v").to_numpy(), y[tr_i])
    assert l_tr == list(tr_i)
    np.testing.assert_allclose(np_te, X[te_i])


def test_container_none_passes_through(xy):
    X, _ = xy
    Xtr, Xte, ntr, nte = train_test_split(X, None, random_state=0)
    assert ntr is None and nte is None
    assert Xtr.shape[0] == 90


@needs_pandas
def test_container_stratify_may_be_a_pandas_series(xy):
    X, y = xy
    ser = pd.Series(y)
    got = train_test_split(np.arange(N), random_state=6, stratify=ser, test_size=0.25)
    want = train_test_split(np.arange(N), random_state=6, stratify=y, test_size=0.25)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


def test_container_string_labels_stratify():
    y = np.array(["cat"] * 40 + ["dog"] * 30 + ["fox"] * 20)
    idx = np.arange(y.size)
    tr, te = train_test_split(idx, random_state=0, stratify=y, test_size=0.25)
    for split in (tr, te):
        vals, counts = np.unique(y[split], return_counts=True)
        assert list(vals) == ["cat", "dog", "fox"]
        np.testing.assert_allclose(counts / counts.sum(), [4 / 9, 3 / 9, 2 / 9], atol=0.05)


# --------------------------------------------------------------------------- #
# 4. parameter validation / error surface
# --------------------------------------------------------------------------- #


def test_error_no_arrays():
    with pytest.raises(ValueError, match="At least one array required"):
        train_test_split()


def test_error_inconsistent_lengths(xy):
    X, y = xy
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        train_test_split(X, y[:-1])


def test_error_stratify_length_mismatch(xy):
    X, y = xy
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        train_test_split(X, stratify=y[:-1])


def test_error_shuffle_false_with_stratify(xy):
    X, y = xy
    with pytest.raises(ValueError, match="not implemented for shuffle=False"):
        train_test_split(X, y, shuffle=False, stratify=y)


def test_error_shuffle_not_boolean(xy):
    X, _ = xy
    with pytest.raises(InvalidParameterError, match="'shuffle' parameter"):
        train_test_split(X, shuffle="yes")


@pytest.mark.parametrize("bad", [0, -1, 1.5, 0.0, 1.0, N, N + 5])
def test_error_bad_test_size(xy, bad):
    X, _ = xy
    with pytest.raises(ValueError, match="test_size"):
        train_test_split(X, test_size=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5, 0.0, 1.0, N])
def test_error_bad_train_size(xy, bad):
    X, _ = xy
    with pytest.raises(ValueError, match="train_size"):
        train_test_split(X, train_size=bad)


def test_error_non_numeric_size(xy):
    X, _ = xy
    with pytest.raises(InvalidParameterError, match="'test_size' parameter"):
        train_test_split(X, test_size="half")


@pytest.mark.parametrize(
    "bad",
    [np.float32(0.25), np.array(0.25), np.array(3), "half", 0, -1, 1.5, 0.0, 1.0],
)
@pytest.mark.parametrize("name", ["test_size", "train_size"])
def test_error_size_type_constraint_matches_sklearn(xy, name, bad):
    """Constraint violations must be rejected on BOTH sides, with an exception
    that is a ValueError AND a TypeError (sklearn's `InvalidParameterError`).

    `np.float32(0.25)` is the one that matters beyond hygiene: accepting it
    would round the ceil/floor boundary off the float64 value the caller wrote.
    """
    X, _ = xy
    with pytest.raises(InvalidParameterError) as mine:
        train_test_split(X, **{name: bad})
    with pytest.raises((ValueError, TypeError)) as theirs:
        sk_train_test_split(X, **{name: bad})
    # sklearn's InvalidParameterError subclasses BOTH; ours must too, or an
    # `except TypeError` caller silently stops catching it.
    assert isinstance(mine.value, (ValueError, TypeError))
    assert isinstance(theirs.value, ValueError) and isinstance(theirs.value, TypeError)


@pytest.mark.parametrize("bad", [3, 1.5, "abc", object()])
def test_error_stratify_must_be_array_like(xy, bad):
    """A scalar `stratify` must raise the same double-based error sklearn does —
    NOT the bare TypeError `_num_samples` would produce further down."""
    X, _ = xy
    with pytest.raises(InvalidParameterError, match="'stratify' parameter"):
        train_test_split(X, stratify=bad)
    with pytest.raises((ValueError, TypeError)) as theirs:
        sk_train_test_split(X, stratify=bad)
    assert isinstance(theirs.value, ValueError) and isinstance(theirs.value, TypeError)


@pytest.mark.parametrize("bad", ["x", 1.5, -3, 2**32])
def test_error_random_state_constraint(xy, bad):
    X, _ = xy
    with pytest.raises(InvalidParameterError, match="'random_state' parameter"):
        train_test_split(X, random_state=bad)
    with pytest.raises((ValueError, TypeError)):
        sk_train_test_split(X, random_state=bad)


def test_error_params_validated_before_missing_arrays():
    """sklearn validates keyword params in a decorator, i.e. BEFORE the body's
    "no arrays" check — so a bad size wins over a missing array."""
    with pytest.raises(InvalidParameterError, match="'test_size' parameter"):
        train_test_split(test_size=0)
    with pytest.raises((ValueError, TypeError)):
        sk_train_test_split(test_size=0)


def test_error_float_sizes_sum_above_one(xy):
    X, _ = xy
    with pytest.raises(ValueError, match="should be in the \\(0, 1\\) range"):
        train_test_split(X, train_size=0.8, test_size=0.3)


def test_error_int_sizes_sum_above_n(xy):
    X, _ = xy
    with pytest.raises(ValueError, match="should be smaller than the number"):
        train_test_split(X, train_size=100, test_size=50)


def test_error_empty_train_set():
    with pytest.raises(ValueError, match="train set will be empty"):
        train_test_split(np.arange(4), test_size=0.99)


def test_error_stratify_singleton_class():
    y = np.array([0] * 10 + [1] * 10 + [2])
    with pytest.raises(ValueError, match="only 1 member"):
        train_test_split(np.arange(y.size), stratify=y, test_size=0.3)


def test_error_stratify_split_smaller_than_class_count():
    y = np.repeat(np.arange(10), 4)
    with pytest.raises(ValueError, match="test_size = .* greater or equal"):
        train_test_split(np.arange(y.size), stratify=y, test_size=5)


def test_error_train_size_smaller_than_class_count():
    y = np.repeat(np.arange(10), 4)
    with pytest.raises(ValueError, match="train_size = .* greater or equal"):
        train_test_split(np.arange(y.size), stratify=y, train_size=5, test_size=20)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"test_size": 500}, id="test_size-exceeds-n"),
        pytest.param({"train_size": 500}, id="train_size-exceeds-n"),
        pytest.param({"train_size": 0.8, "test_size": 0.3}, id="float-sum-above-1"),
        pytest.param({"train_size": 100, "test_size": 50}, id="int-sum-above-n"),
        pytest.param({"test_size": 0.999}, id="empty-train"),
    ],
)
def test_error_data_relative_messages_match_sklearn_verbatim(xy, kwargs):
    """A well-formed-but-wrong-for-this-data size raises a plain ValueError on
    both sides, with the SAME message text — asserted verbatim, not just by
    exception class, so a wording drift in either side is caught."""
    X, _ = xy
    with pytest.raises(ValueError) as mine:
        train_test_split(X, **kwargs)
    with pytest.raises(ValueError) as theirs:
        sk_train_test_split(X, **kwargs)
    assert str(mine.value) == str(theirs.value)
    # this class of error is NOT a constraint violation, so it must NOT be a
    # TypeError — that is what distinguishes it from InvalidParameterError.
    assert not isinstance(mine.value, TypeError)
    assert not isinstance(theirs.value, TypeError)


def test_parity_numpy_bool_shuffle_is_identity(xy):
    """`np.False_` is not the `False` singleton, so sklearn's `shuffle is False`
    test SHUFFLES. mlrs inherits that identity check verbatim; this pins the
    quirk on both sides so a "helpful" `not shuffle` rewrite fails here."""
    X, y = xy
    got = train_test_split(X, y, shuffle=np.False_, random_state=0)
    want = sk_train_test_split(X, y, shuffle=np.False_, random_state=0)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)
    # ...and it really did shuffle, rather than coincidentally matching a
    # head/tail split.
    assert not np.array_equal(got[0], X[: len(got[0])])

    # np.False_ + stratify is therefore a LEGAL call (it never reaches the
    # shuffle=False guard) — sklearn accepts it, so mlrs must not raise.
    got = train_test_split(X, y, shuffle=np.False_, stratify=y, random_state=0)
    want = sk_train_test_split(X, y, shuffle=np.False_, stratify=y, random_state=0)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)

    # a real `False` still takes the head/tail path
    tr, _ = train_test_split(np.arange(10), shuffle=False, test_size=0.3)
    np.testing.assert_array_equal(tr, np.arange(7))


def test_container_duck_typed_dataframe_takes_the_pandas_path():
    """A pandas-API object that is NOT a pandas instance must be gathered by
    ROW via `.take`, like sklearn does.

    `sklearn.utils._mocking.MockDataFrame` has `.iloc`/`.take`/`.shape` but no
    `__getitem__`. Dispatching on `isinstance(X, pd.DataFrame)` would drop it
    into the generic `X[indices]` branch, which on a dataframe selects COLUMNS
    by label — for a square-ish frame with integer column labels that returns
    silently-wrong data with train and test fully overlapping, rather than
    raising.
    """
    mock = pytest.importorskip("sklearn.utils._mocking")
    data = np.arange(240).reshape(20, 12)
    frame = mock.MockDataFrame(data)
    tr_i, te_i = _expected_rows(0, n=20)

    got_tr, got_te = train_test_split(frame, random_state=0)
    want_tr, want_te = sk_train_test_split(frame, random_state=0)

    assert type(got_tr) is type(frame)
    assert got_tr.shape == want_tr.shape == (15, 12)  # rows taken, not columns
    np.testing.assert_array_equal(np.asarray(got_tr), data[tr_i])
    np.testing.assert_array_equal(np.asarray(got_te), data[te_i])
    np.testing.assert_array_equal(np.asarray(got_tr), np.asarray(want_tr))
    # no leakage — the failure mode this guards against overlapped them entirely
    assert set(map(tuple, np.asarray(got_tr))).isdisjoint(
        set(map(tuple, np.asarray(got_te)))
    )


def test_container_array_only_object_is_materialized():
    """An object exposing only `__array__` (no `__getitem__`, no `.iloc`) is not
    row-indexable, so `_make_indexable` must materialize it with `np.array` —
    matching sklearn, which returns an ndarray rather than raising."""

    class ArrayOnly:
        def __init__(self, values):
            self._values = values

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self._values, dtype=dtype)

        def __len__(self):
            return len(self._values)

    obj = ArrayOnly(np.arange(20))
    got = train_test_split(obj, random_state=0)
    want = sk_train_test_split(obj, random_state=0)
    assert isinstance(got[0], np.ndarray)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])


# --------------------------------------------------------------------------- #
# 5. namespace wiring
# --------------------------------------------------------------------------- #


def test_submodule_is_reachable_from_package_root():
    import mlrs

    assert mlrs.model_selection.train_test_split is train_test_split
    # deliberately NOT hoisted into the top-level estimator namespace
    assert not hasattr(mlrs, "train_test_split")


def test_module_does_not_require_the_compiled_extension(monkeypatch):
    """Importing and CALLING the module must not touch `mlrs._mlrs` — break the
    loader and the split still works."""
    import mlrs

    def boom():
        raise ImportError("extension deliberately unavailable")

    monkeypatch.setattr(mlrs, "_load_ext", boom)
    tr, te = train_test_split(np.arange(20), random_state=0)
    assert len(tr) == 15 and len(te) == 5


# --------------------------------------------------------------------------- #
# 6. KFold / StratifiedKFold (MODSEL-01)
# --------------------------------------------------------------------------- #

_KF_SHAPES = [(10, 2), (10, 5), (23, 3), (23, 7), (100, 5), (97, 4)]


@pytest.mark.parametrize("n,k", _KF_SHAPES)
@pytest.mark.parametrize("seed", [None, 0, 42])
def test_kfold_parity_with_sklearn(n, k, seed):
    """Same folds as sklearn, index-for-index. `seed=None` exercises the
    unshuffled path (a random_state with shuffle=False is an error)."""
    X = np.arange(n * 3).reshape(n, 3).astype(float)
    if seed is None:
        mine, theirs = KFold(k), sk_KFold(k)
    else:
        mine = KFold(k, shuffle=True, random_state=seed)
        theirs = sk_KFold(k, shuffle=True, random_state=seed)
    got, want = list(mine.split(X)), list(theirs.split(X))
    assert len(got) == len(want) == k
    for (gtr, gte), (wtr, wte) in zip(got, want):
        np.testing.assert_array_equal(gtr, wtr)
        np.testing.assert_array_equal(gte, wte)


@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("seed", [None, 0, 42])
def test_stratified_kfold_parity_with_sklearn(xy, k, seed):
    X, y = xy
    if seed is None:
        mine, theirs = StratifiedKFold(k), sk_StratifiedKFold(k)
    else:
        mine = StratifiedKFold(k, shuffle=True, random_state=seed)
        theirs = sk_StratifiedKFold(k, shuffle=True, random_state=seed)
    got, want = list(mine.split(X, y)), list(theirs.split(X, y))
    assert len(got) == len(want) == k
    for (gtr, gte), (wtr, wte) in zip(got, want):
        np.testing.assert_array_equal(gtr, wtr)
        np.testing.assert_array_equal(gte, wte)


def test_stratified_kfold_class_encoding_is_by_appearance_not_lexicographic():
    """sklearn encodes classes by ORDER OF APPEARANCE in y, not sorted order.
    Labels chosen so the two orders differ ('zebra' appears first but sorts
    last) — a lexicographic encoding would permute the fold blocks."""
    y = np.array(["zebra"] * 10 + ["apple"] * 10 + ["mango"] * 10)
    X = np.zeros((30, 2))
    got = list(StratifiedKFold(3).split(X, y))
    want = list(sk_StratifiedKFold(3).split(X, y))
    for (gtr, gte), (wtr, wte) in zip(got, want):
        np.testing.assert_array_equal(gtr, wtr)
        np.testing.assert_array_equal(gte, wte)


@pytest.mark.parametrize("n,k", _KF_SHAPES)
def test_kfold_structural_partition(n, k):
    """Every row is tested exactly once; folds differ in size by at most 1;
    train and test never overlap."""
    X = np.zeros((n, 2))
    seen = []
    for train, test in KFold(k).split(X):
        assert set(train).isdisjoint(set(test))
        assert len(train) + len(test) == n
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(n))
    sizes = [len(te) for _, te in KFold(k).split(X)]
    assert max(sizes) - min(sizes) <= 1


def test_stratified_kfold_preserves_class_balance(xy):
    _, y = xy
    X = np.zeros((y.size, 2))
    full = np.bincount(y) / y.size
    for _, test in StratifiedKFold(4).split(X, y):
        share = np.bincount(y[test], minlength=3) / len(test)
        np.testing.assert_allclose(share, full, atol=0.03)


def test_kfold_get_n_splits_and_repr():
    kf = KFold(4, shuffle=True, random_state=3)
    assert kf.get_n_splits() == 4
    assert kf.get_n_splits(np.zeros((10, 2)), None, None) == 4
    assert repr(kf) == "KFold(n_splits=4, random_state=3, shuffle=True)"
    assert repr(StratifiedKFold(2)) == (
        "StratifiedKFold(n_splits=2, random_state=None, shuffle=False)"
    )


@pytest.mark.parametrize("cls", [KFold, StratifiedKFold])
def test_kfold_constructor_validation_matches_sklearn(cls):
    sk_cls = sk_KFold if cls is KFold else sk_StratifiedKFold
    # n_splits must be Integral
    for bad in (2.5, "3", None):
        with pytest.raises(ValueError, match="must be of Integral type"):
            cls(bad)
        with pytest.raises(ValueError):
            sk_cls(bad)
    # at least 2 folds
    for bad in (1, 0, -1):
        with pytest.raises(ValueError, match="at least one"):
            cls(bad)
    # shuffle must be a strict bool -> TypeError (NOT ValueError, and NOT
    # np.bool_-tolerant, unlike train_test_split's constraint)
    for bad in ("yes", 1, np.True_):
        with pytest.raises(TypeError, match="shuffle must be True or False"):
            cls(3, shuffle=bad)
        with pytest.raises(TypeError):
            sk_cls(3, shuffle=bad)
    # random_state without shuffle is an ERROR, not a silent no-op
    with pytest.raises(ValueError, match="no effect since shuffle is"):
        cls(3, shuffle=False, random_state=0)
    with pytest.raises(ValueError):
        sk_cls(3, shuffle=False, random_state=0)


@pytest.mark.parametrize("cls", [KFold, StratifiedKFold])
def test_kfold_more_splits_than_samples(cls):
    X = np.zeros((4, 2))
    y = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="Cannot have number of splits"):
        list(cls(5).split(X, y))


def test_stratified_kfold_rejects_continuous_target():
    X = np.zeros((20, 2))
    y = np.linspace(0.0, 1.0, 20)
    with pytest.raises(ValueError, match="Supported target types"):
        list(StratifiedKFold(3).split(X, y))


def test_stratified_kfold_warns_on_small_class_and_errors_when_all_too_small():
    X = np.zeros((12, 2))
    y = np.array([0] * 10 + [1, 1])  # class 1 has only 2 members
    with pytest.warns(UserWarning, match="least populated class"):
        list(StratifiedKFold(3).split(X, y))
    # every class smaller than n_splits -> hard error
    y_tiny = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="cannot be greater than"):
        list(StratifiedKFold(3).split(np.zeros((4, 2)), y_tiny))


def test_stratified_kfold_warns_when_groups_passed(xy):
    X, y = xy
    with pytest.warns(UserWarning, match="groups parameter is ignored"):
        list(StratifiedKFold(3).split(X, y, groups=np.arange(len(y))))


@needs_polars
def test_kfold_split_accepts_non_numpy_containers(xy):
    """`split` reads only the ROW COUNT of X, so a polars/pandas/pyarrow X
    yields the same integer index folds a numpy X does."""
    X, y = xy
    frames = [pl.DataFrame({f"c{j}": X[:, j] for j in range(4)}), pa.table({"a": X[:, 0]})]
    if pd is not None:
        frames.append(pd.DataFrame(X))
    want = list(KFold(4).split(X))
    for frame in frames:
        got = list(KFold(4).split(frame))
        for (gtr, gte), (wtr, wte) in zip(got, want):
            np.testing.assert_array_equal(gtr, wtr)
            np.testing.assert_array_equal(gte, wte)


def test_kfold_folds_are_reproducible_and_seed_sensitive():
    X = np.zeros((50, 2))
    a = list(KFold(5, shuffle=True, random_state=1).split(X))
    b = list(KFold(5, shuffle=True, random_state=1).split(X))
    c = list(KFold(5, shuffle=True, random_state=2).split(X))
    for (x, _), (y_, _) in zip(a, b):
        np.testing.assert_array_equal(x, y_)
    assert any(not np.array_equal(x, z) for (x, _), (z, _) in zip(a, c))


# --------------------------------------------------------------------------- #
# 7. search passthrough (MODSEL-02) + sklearn interop
# --------------------------------------------------------------------------- #


def test_search_passthrough_is_sklearn_itself():
    """These are deliberately re-exported, not reimplemented — pin that, so a
    future 'helpful' reimplementation is a conscious decision."""
    import mlrs.model_selection as ms

    assert ms.GridSearchCV is skm.GridSearchCV
    assert ms.RandomizedSearchCV is skm.RandomizedSearchCV
    assert ms.cross_val_score is skm.cross_val_score
    assert ms.cross_validate is skm.cross_validate


@pytest.mark.parametrize("cls", [KFold, StratifiedKFold])
def test_mlrs_splitters_are_accepted_by_sklearn_check_cv(cls):
    """sklearn duck-types `cv=` on split/get_n_splits — an mlrs splitter must
    pass through `check_cv` unchanged rather than being silently replaced."""
    from sklearn.model_selection import check_cv

    splitter = cls(3)
    assert check_cv(splitter, y=np.array([0, 1, 0, 1]), classifier=True) is splitter


def test_mlrs_kfold_drives_sklearn_cross_val_score():
    """End-to-end: an mlrs splitter feeding sklearn's cross_val_score, with the
    fold count and scores matching the sklearn splitter exactly."""
    from sklearn.linear_model import Ridge as SkRidge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.normal(size=60)
    got = cross_val_score(SkRidge(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    want = cross_val_score(SkRidge(), X, y, cv=sk_KFold(5, shuffle=True, random_state=0))
    assert got.shape == (5,)
    np.testing.assert_allclose(got, want)


def test_mlrs_stratified_kfold_drives_sklearn_gridsearch():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=80) > 0).astype(int)
    grid = {"C": [0.1, 1.0]}
    gs = GridSearchCV(
        LogisticRegression(), grid, cv=StratifiedKFold(4, shuffle=True, random_state=0)
    )
    gs.fit(X, y)
    assert gs.n_splits_ == 4
    assert gs.best_params_["C"] in (0.1, 1.0)
    assert len(gs.cv_results_["mean_test_score"]) == 2


def test_train_test_split_then_kfold_compose(xy):
    """The two surfaces compose: hold out a test set, then CV the train set."""
    X, y = xy
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    assert len(X_tr) == 90 and len(X_te) == 30
    folds = list(StratifiedKFold(3).split(X_tr, y_tr))
    assert len(folds) == 3
    covered = sorted(np.concatenate([te for _, te in folds]).tolist())
    assert covered == list(range(len(X_tr)))
