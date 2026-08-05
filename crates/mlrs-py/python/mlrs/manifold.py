"""Manifold-learning estimator shims (PY-01/PY-02) delegating to ``_mlrs``.

UMAP -> ``TransformerMixin`` (gives ``fit`` + ``transform`` + ``fit_transform``).
sklearn/umap-learn-faithful ``__init__`` stores every ctor arg verbatim under
the SAME name (purity rule — the AST gate enforces it). ``fit`` is unsupervised
(``y=None``); ``transform`` embeds new points via the fitted model;
``fit_transform`` returns the training embedding; ``embedding_`` is the fitted
training embedding.

Defaults mirror ``PyUMAP`` ``#[new]`` at
``crates/mlrs-py/src/estimators/manifold.rs:166-167`` (umap-learn defaults). The
forwarders target the Plan-10 ``transform_f{32,64}`` / ``fit_transform_f{32,64}``
``#[pymethods]`` (16-10-SUMMARY).
"""

import numpy as np
from sklearn.base import TransformerMixin

from .base import MlrsBase


def _flat(v):
    """Row-major flatten of a ``metric_params`` payload (``V`` is a vector,
    ``VI`` a square matrix) into the ``list[float]`` the extension takes."""
    return np.ascontiguousarray(np.asarray(v, dtype=np.float64)).ravel().tolist()


def _pca_init(x, n_components):
    """t-SNE's ``init='pca'`` embedding: the top-``n_components`` principal
    scores, rescaled so ``std(y[:, 0]) == 1e-4``.

    Only reached on the callable-``metric`` path, where the Rust side is handed
    a precomputed distance matrix and so no longer has the feature space to
    project (see ``TSNE._resolve_metric``). It is the same computation
    ``manifold::tsne::pca_init`` performs, expressed over the original design —
    up to a per-axis sign, which the descent is equivariant to because its
    gradient depends on the embedding only through pairwise differences.
    """
    arr = np.asarray(x, dtype=np.float64)
    centered = arr - arr.mean(axis=0, keepdims=True)
    k = min(int(n_components), *centered.shape)
    # `full_matrices=False` keeps this the thin SVD, so the cost is set by
    # min(n, p) rather than by the larger dimension.
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :k] * s[:k]
    if scores.shape[1] < int(n_components):
        raise ValueError(
            f"init='pca' needs n_components <= min(n_samples, n_features) = "
            f"{min(centered.shape)}, got {n_components}"
        )
    std0 = scores[:, 0].std()
    return scores / std0 * 1e-4 if std0 > 0 else scores * 1e-4


class UMAP(TransformerMixin, MlrsBase):
    """Uniform Manifold Approximation and Projection (MANIFOLD-01).

    ``UMAP(n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0,
    metric="euclidean", n_epochs=None, init="spectral", random_state=None,
    learning_rate=1.0, set_op_mix_ratio=1.0, local_connectivity=1.0,
    repulsion_strength=1.0, negative_sample_rate=5, a=None, b=None)``.
    """

    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        spread=1.0,
        metric="euclidean",
        n_epochs=None,
        init="spectral",
        random_state=None,
        learning_rate=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        a=None,
        b=None,
        output_type="input",
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.spread = spread
        self.metric = metric
        self.n_epochs = n_epochs
        self.init = init
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.a = a
        self.b = b
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().UMAP(
            self.n_neighbors,
            self.n_components,
            self.min_dist,
            self.spread,
            self.metric,
            self.n_epochs,
            self.init,
            self.random_state,
            self.learning_rate,
            self.set_op_mix_ratio,
            self.local_connectivity,
            self.repulsion_strength,
            self.negative_sample_rate,
            self.a,
            self.b,
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def transform(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("transform")(xa, rows, cols)
        return self._to_output(
            out, (rows, self.n_components), X, self._np_float()
        )

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.embedding_

    @property
    def embedding_(self):
        out = self._suffixed("embedding")()
        return self._to_output(
            out, (-1, self.n_components), None, self._np_float()
        )


class TSNE(MlrsBase):
    """t-distributed Stochastic Neighbor Embedding (TSNE-01 / TSNE-PARAMS).

    Mirrors ``sklearn.manifold.TSNE``'s full 1.9.0 parameter surface::

        TSNE(n_components=2, *, perplexity=30.0, early_exaggeration=12.0,
             learning_rate="auto", max_iter=1000, n_iter_without_progress=300,
             min_grad_norm=1e-7, metric="euclidean", metric_params=None,
             init="pca", verbose=0, random_state=None, method="barnes_hut",
             angle=0.5, n_jobs=None)

    ``method='barnes_hut'`` is the default, as in sklearn, and is a genuinely
    different algorithm from ``'exact'``: an ``O(n log n)`` quadtree-summarized
    gradient over a sparse k-nearest-neighbour ``P`` rather than the dense
    ``O(n²)`` objective. ``metric`` accepts every string sklearn does — including
    ``'precomputed'`` and the six metrics scipy evaluates on a boolean cast —
    plus a **callable**, which is realized by evaluating it into a dense
    distance matrix and fitting that as ``metric='precomputed'`` (the same
    values sklearn's own callable path produces). ``init`` accepts ``'pca'``,
    ``'random'``, or an ``(n_samples, n_components)`` array.

    No out-of-sample ``transform`` (sklearn parity — TSNE re-embeds via
    ``fit_transform``).

    Two deliberate divergences, both wall-clock only:

    * ``n_jobs=None`` uses every core rather than joblib's single worker. Every
      parallel reduction here runs in point order, so the worker count cannot
      change a value.
    * mlrs accepts ``metric='seuclidean'`` and ``'matching'`` with
      ``method='barnes_hut'``; sklearn 1.9.0 raises for those two pairs because
      its ``NearestNeighbors`` cannot take them, not because they are undefined.
    """

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        init="pca",
        verbose=0,
        random_state=None,
        method="barnes_hut",
        angle=0.5,
        n_jobs=None,
        output_type="input",
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.output_type = output_type

    def _resolve_metric(self, X):
        """Return ``(metric_name, X_for_fit, init_override)``.

        A callable ``metric`` is evaluated into a dense square distance matrix
        here and fitted as ``'precomputed'``. That is not a shortcut: sklearn's
        own callable path also reduces to "evaluate every pair", both for the
        exact method (``pairwise_distances`` with a callable) and for
        Barnes-Hut (``NearestNeighbors`` falls back to brute force), so the
        distances the fit sees are the same ones.

        The reduction has one consequence that must NOT be inherited, though.
        ``metric='precomputed'`` forbids ``init='pca'`` — there is no feature
        space left to project — but a *callable* metric does not: sklearn keeps
        the original ``X`` around and PCA-initializes from it. Since ``'pca'``
        is the DEFAULT init, letting the reduction leak that restriction would
        make ``TSNE(metric=my_callable)`` fail on the default configuration
        while sklearn succeeds. So the PCA init is computed HERE, from the
        original design, and passed through as an explicit init array — which
        is exactly the embedding the non-callable path would have produced
        (up to a per-axis sign, which the descent is equivariant to).
        """
        if not callable(self.metric):
            return self.metric, X, None

        arr = np.asarray(X, dtype=np.float64)
        n = arr.shape[0]
        dist = np.empty((n, n), dtype=arr.dtype)
        for i in range(n):
            dist[i, i] = 0.0
            for j in range(i + 1, n):
                v = float(self.metric(arr[i], arr[j]))
                dist[i, j] = v
                dist[j, i] = v

        init_override = None
        if isinstance(self.init, str) and self.init == "pca":
            init_override = _pca_init(arr, int(self.n_components))
        return "precomputed", dist, init_override

    def _init_payload(self, rows, override=None):
        """Split ``init`` into the ``(name, flat_array)`` pair the extension
        takes. sklearn allows an ndarray there; the sentinel ``"array"`` is how
        that crosses a typed boundary. ``override`` carries the callable-metric
        path's pre-computed PCA init (see ``_resolve_metric``)."""
        init = self.init if override is None else override
        if isinstance(init, str):
            return init, None
        arr = np.asarray(init, dtype=np.float64)
        if arr.shape != (rows, int(self.n_components)):
            raise ValueError(
                f"init array has shape {arr.shape}, expected "
                f"{(rows, int(self.n_components))}"
            )
        return "array", np.ascontiguousarray(arr).ravel().tolist()

    def fit(self, X, y=None):
        metric, X, init_override = self._resolve_metric(X)
        xa, rows, cols = self._normalize(X)
        lr = None if self.learning_rate == "auto" else float(self.learning_rate)
        seed = None if self.random_state is None else int(self.random_state)
        mp = self.metric_params or {}
        unknown = set(mp) - {"p", "V", "VI", "w"}
        if unknown:
            raise ValueError(
                f"metric_params: unsupported key(s) {sorted(unknown)}; "
                "expected a subset of {'p', 'V', 'VI', 'w'}"
            )
        if "w" in mp:
            raise ValueError(
                "metric_params['w'] belongs to 'wminkowski', which scipy "
                "removed; it cannot be evaluated"
            )
        init_name, init_array = self._init_payload(rows, init_override)
        obj = self._ext().TSNE(
            int(self.n_components),
            float(self.perplexity),
            float(self.early_exaggeration),
            lr,
            int(self.max_iter),
            int(self.n_iter_without_progress),
            float(self.min_grad_norm),
            metric,
            None if mp.get("p") is None else float(mp["p"]),
            None if mp.get("V") is None else _flat(mp["V"]),
            None if mp.get("VI") is None else _flat(mp["VI"]),
            init_name,
            init_array,
            int(self.verbose),
            seed,
            self.method,
            float(self.angle),
            None if self.n_jobs is None else int(self.n_jobs),
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.embedding_

    @property
    def embedding_(self):
        out = self._suffixed("embedding")()
        return self._to_output(
            out, (-1, self.n_components), None, self._np_float()
        )

    @property
    def kl_divergence_(self):
        self._check_fitted()
        return self._mlrs_obj.kl_divergence_()

    @property
    def n_iter_(self):
        self._check_fitted()
        return self._mlrs_obj.n_iter_()
