"""Latent Dirichlet allocation with a sparse variational E-step in Rust."""

import itertools

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy import sparse, special
from sklearn.utils import check_random_state

from ._decomposition_common import DecompositionTransformer, choice, count


def _expectation(values):
    return special.digamma(values) - special.digamma(values.sum(axis=1, keepdims=True))


def _bound_term(values, prior, expectation):
    width = values.shape[1]
    return (
        np.sum((prior - values) * expectation)
        + special.gammaln(values).sum()
        - special.gammaln(values.sum(axis=1)).sum()
        + len(values)
        * (special.gammaln(width * prior) - width * special.gammaln(prior))
    )


class LatentDirichletAllocation(DecompositionTransformer):
    """Variational topic model with batch and online learning."""

    _accept_sparse = True
    _nonnegative = True

    def __init__(
        self,
        n_components=10,
        *,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1000000.0,
        perp_tol=0.1,
        mean_change_tol=0.001,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _setup(self, X):
        k = count(self.n_components, "n_components")
        count(self.max_doc_update_iter, "max_doc_update_iter", minimum=0)
        choice(self.learning_method, "learning_method", ("batch", "online"))
        if (
            not 0 <= self.learning_decay <= 1
            or self.learning_offset <= 0
            or self.mean_change_tol < 0
            or self.perp_tol < 0
            or self.total_samples <= 0
        ):
            raise ValueError(
                "Invalid LDA learning rate, tolerance, or total sample count."
            )
        self.doc_topic_prior_ = (
            1 / k if self.doc_topic_prior is None else self.doc_topic_prior
        )
        self.topic_word_prior_ = (
            1 / k if self.topic_word_prior is None else self.topic_word_prior
        )
        if not 0 < self.doc_topic_prior_ <= 1 or not 0 < self.topic_word_prior_ <= 1:
            raise ValueError("Dirichlet priors must be in (0, 1].")
        self.random_state_ = check_random_state(self.random_state)
        self.n_iter_ = 0
        self.n_batch_iter_ = 1
        self.components_ = self.random_state_.gamma(100, 0.01, (k, X.shape[1])).astype(
            X.dtype
        )
        self.exp_dirichlet_component_ = np.exp(_expectation(self.components_))

    def _e_step(self, X, *, training):
        from . import _mlrs

        X = sparse.csr_matrix(X)
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
        k = len(self.components_)
        shape = (X.shape[0], k)
        initial = (
            self.random_state_.gamma(100, 0.01, shape)
            if training
            else np.ones(shape, dtype=X.dtype)
        )
        weights = self.exp_dirichlet_component_.T.ravel().tolist()

        def infer(batch, starts):
            doc, stats = _mlrs.decomp_lda_infer(
                batch.indptr.tolist(),
                batch.indices.tolist(),
                batch.data.tolist(),
                weights,
                starts.ravel().tolist(),
                k,
                self.doc_topic_prior_,
                self.max_doc_update_iter,
                self.mean_change_tol,
                np.finfo(X.dtype).eps,
                training,
            )
            return np.asarray(doc, dtype=X.dtype).reshape(batch.shape[0], k), stats

        jobs = min(effective_n_jobs(self.n_jobs), X.shape[0])
        if jobs == 1:
            result = [infer(X, initial)]
        else:
            edges = np.linspace(0, X.shape[0], jobs + 1, dtype=int)
            result = Parallel(n_jobs=jobs, prefer="threads")(
                delayed(infer)(X[start:end], initial[start:end])
                for start, end in itertools.pairwise(edges)
            )
        docs = np.concatenate([doc for doc, _ in result])
        if not training:
            return docs, None
        stats = np.sum([values for _, values in result], axis=0)
        return docs, stats.reshape(X.shape[1], k).T.astype(X.dtype)

    def _update(self, X, *, total_samples, online):
        _, stats = self._e_step(X, training=True)
        if online:
            weight = (self.learning_offset + self.n_batch_iter_) ** -self.learning_decay
            estimate = self.topic_word_prior_ + stats * (total_samples / X.shape[0])
            self.components_ = (1 - weight) * self.components_ + weight * estimate
        else:
            self.components_ = self.topic_word_prior_ + stats
        self.exp_dirichlet_component_ = np.exp(_expectation(self.components_))
        self.n_batch_iter_ += 1

    def fit(self, X, y=None):
        X = self._fit_matrix(X)
        steps = count(self.max_iter, "max_iter", minimum=0)
        batch = count(self.batch_size, "batch_size")
        self._setup(X)
        self._record_fit(X, len(self.components_))
        previous = np.inf
        for epoch in range(steps):
            if self.learning_method == "online":
                for i in range(0, X.shape[0], batch):
                    self._update(
                        X[i : i + batch], total_samples=X.shape[0], online=True
                    )
            else:
                self._update(X, total_samples=X.shape[0], online=False)
            self.n_iter_ = epoch + 1
            if self.verbose:
                print(f"Iteration {self.n_iter_}")
            if self.evaluate_every > 0 and self.n_iter_ % self.evaluate_every == 0:
                current = self.perplexity(X)
                if abs(previous - current) < self.perp_tol:
                    break
                previous = current
        self.bound_ = self.perplexity(X)
        return self

    def partial_fit(self, X, y=None):
        batch = count(self.batch_size, "batch_size")
        if hasattr(self, "components_"):
            X = self._validate_transform(X)
        else:
            X = self._fit_matrix(X)
            self._setup(X)
            self._record_fit(X, len(self.components_))
        for i in range(0, X.shape[0], batch):
            self._update(
                X[i : i + batch], total_samples=self.total_samples, online=True
            )
        return self

    def _documents(self, X):
        X = self._validate_transform(X)
        docs, _ = self._e_step(X, training=False)
        return X, docs

    def transform(self, X, *, normalize=True):
        _, docs = self._documents(X)
        return docs / docs.sum(axis=1, keepdims=True) if normalize else docs

    def _score(self, X, docs, ratio):
        doc_topics = _expectation(docs)
        word_topics = _expectation(self.components_)
        obs = sparse.csr_matrix(X)
        score = 0.0
        for row, (start, end) in enumerate(zip(obs.indptr, obs.indptr[1:])):
            weights = doc_topics[row, :, None] + word_topics[:, obs.indices[start:end]]
            score += obs.data[start:end] @ special.logsumexp(weights, axis=0)
        score += _bound_term(docs, self.doc_topic_prior_, doc_topics)
        return score * ratio + _bound_term(
            self.components_, self.topic_word_prior_, word_topics
        )

    def score(self, X, y=None):
        X, docs = self._documents(X)
        return self._score(X, docs, 1)

    def perplexity(self, X, sub_sampling=False):
        X, docs = self._documents(X)
        ratio = self.total_samples / X.shape[0] if sub_sampling else 1
        return np.exp(-self._score(X, docs, ratio) / (X.sum() * ratio))
