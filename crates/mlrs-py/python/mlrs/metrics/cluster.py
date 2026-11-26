"""``mlrs.metrics.cluster`` — sklearn's ``sklearn.metrics.cluster`` surface.

Supervised scores (labellings vs labellings) are functions of one sparse
contingency table built in Rust (``mlrs_algos::metrics::cluster``); the
unsupervised scores take the feature matrix, and ``silhouette_samples`` streams
its pairwise distances block by block so the ``n × n`` matrix never exists.

Labels may be any integer (or boolean) values; they are relabelled to
``0..K-1`` in Rust.
"""

import warnings

import numpy as np

from ._base import _ext, _owned, _pa

__all__ = [
    "adjusted_mutual_info_score",
    "adjusted_rand_score",
    "calinski_harabasz_score",
    "completeness_score",
    "consensus_score",
    "contingency_matrix",
    "davies_bouldin_score",
    "entropy",
    "expected_mutual_information",
    "fowlkes_mallows_score",
    "homogeneity_completeness_v_measure",
    "homogeneity_score",
    "mutual_info_score",
    "normalized_mutual_info_score",
    "pair_confusion_matrix",
    "rand_score",
    "silhouette_samples",
    "silhouette_score",
    "v_measure_score",
]


def _labels(y, name):
    a = np.asarray(y)
    if a.ndim != 1:
        raise ValueError("%s must be 1D: shape is %r" % (name, a.shape))
    if a.dtype.kind == "f":
        if not np.all(a == np.round(a)):
            warnings.warn(
                "Clustering metrics expects discrete values but received continuous values for "
                "label, and continuous values for target",
                UserWarning,
            )
    elif a.dtype.kind not in "iub":
        raise ValueError("mlrs clustering metrics take integer (or boolean) labels; got dtype %s" % a.dtype)
    return a


def _pair(labels_true, labels_pred):
    lt = _labels(labels_true, "labels_true")
    lp = _labels(labels_pred, "labels_pred")
    if len(lt) != len(lp):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % ([len(lt), len(lp)],)
        )
    return _pa(lt), _pa(lp)


def _check_average_method(m):
    if m not in ("min", "geometric", "arithmetic", "max"):
        raise ValueError("'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'")


def contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype=np.int64):
    """Matches ``sklearn.metrics.cluster.contingency_matrix``. ``sparse=True``
    returns a ``scipy.sparse.csr_array``."""
    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")
    a, b = _pair(labels_true, labels_pred)
    flat, rows, cols = _ext().contingency_matrix(a, b)
    dense = np.asarray(flat, dtype=np.int64).reshape(rows, cols).astype(dtype)
    if sparse:
        import scipy.sparse as sp

        return sp.csr_array(dense)
    if eps is not None:
        dense = dense + eps
    return dense


def pair_confusion_matrix(labels_true, labels_pred):
    """Matches ``sklearn.metrics.cluster.pair_confusion_matrix``."""
    a, b = _pair(labels_true, labels_pred)
    return np.asarray(_ext().pair_confusion_matrix(a, b), dtype=np.int64).reshape(2, 2)


def _score(name, labels_true, labels_pred, average_method="arithmetic"):
    a, b = _pair(labels_true, labels_pred)
    return float(_ext().cluster_score(name, a, b, average_method))


def rand_score(labels_true, labels_pred):
    """Matches ``sklearn.metrics.rand_score``."""
    return _score("rand", labels_true, labels_pred)


def adjusted_rand_score(labels_true, labels_pred):
    """Matches ``sklearn.metrics.adjusted_rand_score``."""
    return _score("adjusted_rand", labels_true, labels_pred)


def mutual_info_score(labels_true, labels_pred, *, contingency=None):
    """Matches ``sklearn.metrics.mutual_info_score``, including the
    ``contingency=`` form (dense or scipy-sparse counts)."""
    if contingency is None:
        return _score("mutual_info", labels_true, labels_pred)
    c = contingency.toarray() if hasattr(contingency, "toarray") else np.asarray(contingency)
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("Expected 2D array, got %dD array instead" % c.ndim)
    return float(_ext().mutual_info_score_contingency(_pa(c.ravel()), c.shape[0], c.shape[1]))


def adjusted_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic"):
    """Matches ``sklearn.metrics.adjusted_mutual_info_score``."""
    _check_average_method(average_method)
    return _score("adjusted_mutual_info", labels_true, labels_pred, average_method)


def normalized_mutual_info_score(labels_true, labels_pred, *, average_method="arithmetic"):
    """Matches ``sklearn.metrics.normalized_mutual_info_score``."""
    _check_average_method(average_method)
    return _score("normalized_mutual_info", labels_true, labels_pred, average_method)


def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0):
    """Matches ``sklearn.metrics.homogeneity_completeness_v_measure``."""
    a, b = _pair(labels_true, labels_pred)
    h, c, v = _ext().homogeneity_completeness_v_measure(a, b, float(beta))
    return float(h), float(c), float(v)


def homogeneity_score(labels_true, labels_pred):
    """Matches ``sklearn.metrics.homogeneity_score``."""
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[0]


def completeness_score(labels_true, labels_pred):
    """Matches ``sklearn.metrics.completeness_score``."""
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[1]


def v_measure_score(labels_true, labels_pred, *, beta=1.0):
    """Matches ``sklearn.metrics.v_measure_score``."""
    return homogeneity_completeness_v_measure(labels_true, labels_pred, beta=beta)[2]


def fowlkes_mallows_score(labels_true, labels_pred):
    """Matches ``sklearn.metrics.fowlkes_mallows_score``."""
    return _score("fowlkes_mallows", labels_true, labels_pred)


def entropy(labels):
    """Matches ``sklearn.metrics.cluster.entropy`` (natural log)."""
    return float(_ext().cluster_entropy(_pa(_labels(labels, "labels"))))


def expected_mutual_information(contingency, n_samples):
    """Matches ``sklearn.metrics.cluster.expected_mutual_information`` (the
    hypergeometric expectation of the MI, computed in Rust over the table's
    marginals)."""
    c = contingency.toarray() if hasattr(contingency, "toarray") else np.asarray(contingency)
    c = np.asarray(c, dtype=np.float64)
    return float(_ext().expected_mutual_information(_pa(c.ravel()), c.shape[0], c.shape[1], int(n_samples)))


def _check_X_labels(X, labels):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Expected 2D array, got %dD array instead" % X.ndim)
    lab = np.asarray(labels).ravel()
    if X.shape[0] != len(lab):
        raise ValueError(
            "Found input variables with inconsistent numbers of samples: %r" % ([X.shape[0], len(lab)],)
        )
    n_labels = len(np.unique(lab))
    if not 1 < n_labels < X.shape[0]:
        raise ValueError(
            "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)" % n_labels
        )
    return np.ascontiguousarray(X), lab


def silhouette_samples(X, labels, *, metric="euclidean", **kwds):
    """Matches ``sklearn.metrics.silhouette_samples`` for every string
    ``metric`` the pairwise engine supports and ``metric='precomputed'``.
    A callable ``metric`` is evaluated through :func:`pairwise_distances`."""
    from .pairwise import _metric_params, pairwise_distances

    X, lab = _check_X_labels(X, labels)
    ext = _ext()
    if metric == "precomputed":
        atol = np.finfo(X.dtype).eps * 100
        if np.any(np.abs(np.diagonal(X)) > atol):
            raise ValueError(
                "The precomputed distance matrix contains non-zero elements on the diagonal. "
                "Use np.fill_diagonal(X, 0)."
            )
        out = ext.silhouette_samples(_pa(X.ravel()), X.shape[1], _pa(lab), "euclidean", True)
        return _owned(out)
    if callable(metric):
        D = pairwise_distances(X, metric=metric, **kwds)
        out = ext.silhouette_samples(_pa(np.ascontiguousarray(D).ravel()), D.shape[1], _pa(lab), "euclidean", True)
        return _owned(out)
    name, p, w, v, vi = _metric_params(metric, X, None, kwds)
    out = ext.silhouette_samples(_pa(X.ravel()), X.shape[1], _pa(lab), name, False, p, w, v, vi)
    return _owned(out)


def silhouette_score(X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds):
    """Matches ``sklearn.metrics.silhouette_score``, including ``sample_size``
    (drawn with sklearn's own ``random_state.permutation`` so a seeded call
    picks the same subsample)."""
    if sample_size is not None:
        from sklearn.utils import check_random_state

        X = np.asarray(X)
        labels = np.asarray(labels)
        rs = check_random_state(random_state)
        indices = rs.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return float(np.mean(silhouette_samples(X, labels, metric=metric, **kwds)))


def calinski_harabasz_score(X, labels):
    """Matches ``sklearn.metrics.calinski_harabasz_score``."""
    X, lab = _check_X_labels(X, labels)
    return float(_ext().cluster_dispersion("calinski_harabasz", _pa(X.ravel()), _pa(lab)))


def davies_bouldin_score(X, labels):
    """Matches ``sklearn.metrics.davies_bouldin_score``."""
    X, lab = _check_X_labels(X, labels)
    return float(_ext().cluster_dispersion("davies_bouldin", _pa(X.ravel()), _pa(lab)))


def consensus_score(a, b, *, similarity="jaccard"):
    """Matches ``sklearn.metrics.consensus_score``: the best one-to-one
    matching (Hungarian algorithm, in Rust) of two bicluster sets, each given
    as ``(rows, columns)`` boolean membership arrays."""
    a_rows, a_cols = (np.asarray(m, dtype=np.float64) for m in a)
    b_rows, b_cols = (np.asarray(m, dtype=np.float64) for m in b)
    n_a, n_b = a_rows.shape[0], b_rows.shape[0]
    ext = _ext()
    if similarity == "jaccard":
        return float(
            ext.consensus_score_jaccard(
                _pa(a_rows.ravel()),
                _pa(a_cols.ravel()),
                _pa(b_rows.ravel()),
                _pa(b_cols.ravel()),
                n_a,
                n_b,
                a_rows.shape[1],
                a_cols.shape[1],
            )
        )
    sim = np.array(
        [[similarity(a_rows[i], a_cols[i], b_rows[j], b_cols[j]) for j in range(n_b)] for i in range(n_a)],
        dtype=np.float64,
    )
    return float(ext.consensus_from_similarity(sim.ravel().tolist(), n_a, n_b))
