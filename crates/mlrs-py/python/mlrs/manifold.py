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

from sklearn.base import TransformerMixin

from .base import MlrsBase


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
    """t-distributed Stochastic Neighbor Embedding (TSNE-01, exact method).

    ``TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,
    learning_rate="auto", max_iter=1000, init="pca", random_state=None,
    method="exact", metric="euclidean")``.

    Scope: ``method='exact'`` and ``metric='euclidean'`` only (any other value
    raises at construction inside ``fit``). ``learning_rate`` accepts the
    sklearn ``"auto"`` sentinel or a positive float. No out-of-sample
    ``transform`` (sklearn parity — TSNE re-embeds via ``fit_transform``).
    """

    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1000,
        init="pca",
        random_state=None,
        method="exact",
        metric="euclidean",
        output_type="input",
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.method = method
        self.metric = metric
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        lr = None if self.learning_rate == "auto" else float(self.learning_rate)
        seed = None if self.random_state is None else int(self.random_state)
        obj = self._ext().TSNE(
            int(self.n_components),
            float(self.perplexity),
            float(self.early_exaggeration),
            lr,
            int(self.max_iter),
            self.init,
            seed,
            self.method,
            self.metric,
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
