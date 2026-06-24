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
