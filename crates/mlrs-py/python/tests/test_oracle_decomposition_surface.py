"""Live scikit-learn oracle coverage for the added decomposition estimators."""

import numpy as np
import pytest
from mlrs import decomposition
from scipy import sparse
from sklearn import decomposition as reference
from sklearn.base import clone
from sklearn.exceptions import NotFittedError


@pytest.fixture
def samples():
    return np.random.RandomState(5).normal(size=(35, 8))


@pytest.mark.parametrize(
    "name, params",
    [
        ("FactorAnalysis", {"svd_method": "lapack"}),
        ("FactorAnalysis", {"rotation": "varimax"}),
        ("KernelPCA", {"kernel": "rbf", "fit_inverse_transform": True}),
        (
            "KernelPCA",
            {"kernel": "cosine", "eigen_solver": "randomized", "random_state": 42},
        ),
        ("KernelPCA", {"kernel": "rbf", "eigen_solver": "arpack", "random_state": 42}),
        ("DictionaryLearning", {"random_state": 42, "max_iter": 10}),
        (
            "DictionaryLearning",
            {
                "random_state": 42,
                "max_iter": 10,
                "fit_algorithm": "cd",
                "positive_code": True,
                "transform_algorithm": "lasso_cd",
            },
        ),
        (
            "MiniBatchDictionaryLearning",
            {"random_state": 42, "max_iter": 10, "batch_size": 5},
        ),
        ("SparsePCA", {"random_state": 42, "max_iter": 10}),
        ("MiniBatchSparsePCA", {"random_state": 42, "max_iter": 10}),
    ],
)
def test_fit_and_transform_match_oracle(samples, name, params):
    expected = getattr(reference, name)(3, **params)
    actual = clone(getattr(decomposition, name)(3, **params))
    with pytest.raises(NotFittedError):
        actual.transform(samples)
    fit = actual.fit_transform(samples)
    np.testing.assert_allclose(fit, expected.fit_transform(samples), atol=1e-5)
    np.testing.assert_allclose(
        actual.transform(samples[:5]),
        expected.transform(samples[:5]),
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        actual.get_feature_names_out(), expected.get_feature_names_out()
    )
    with pytest.raises(ValueError, match="features"):
        actual.transform(samples[:, :-1])
    if params.get("fit_inverse_transform") or "PCA" in name and name != "KernelPCA":
        np.testing.assert_allclose(
            actual.inverse_transform(fit),
            expected.inverse_transform(expected.transform(samples)),
            atol=1e-5,
        )


@pytest.mark.parametrize("algorithm", ["parallel", "deflation"])
def test_ica_sources_and_inverse(algorithm):
    rng = np.random.RandomState(12)
    original = np.column_stack([rng.laplace(size=350), rng.uniform(-2, 2, 350)])
    X = original @ np.array([[1.0, 0.2], [0.5, 1.5]])
    options = {"algorithm": algorithm, "random_state": 0, "tol": 1e-8}
    estimator = decomposition.FastICA(**options)
    sources = estimator.fit_transform(X)
    expected = reference.FastICA(**options).fit_transform(X)
    directions = np.corrcoef(sources.T, expected.T)[:2, 2:]
    np.testing.assert_allclose(np.abs(directions).max(axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(estimator.inverse_transform(sources), X, atol=1e-10)
    whitening, unmixing, transformed, mean, n_iter = decomposition.fastica(
        X,
        **options,
        return_X_mean=True,
        return_n_iter=True,
    )
    np.testing.assert_allclose((X - mean) @ (unmixing @ whitening).T, transformed)
    assert n_iter == estimator.n_iter_


@pytest.mark.parametrize(
    "algorithm", ["lars", "lasso_lars", "lasso_cd", "omp", "threshold"]
)
def test_sparse_coding_algorithms(samples, algorithm):
    dictionary = np.random.RandomState(2).normal(size=(12, samples.shape[1]))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    expected = reference.sparse_encode(
        samples, dictionary, algorithm=algorithm, alpha=0.2
    )
    result = decomposition.sparse_encode(
        samples,
        dictionary,
        algorithm=algorithm,
        alpha=0.2,
        n_jobs=2,
    )
    np.testing.assert_allclose(result, expected, atol=1e-5)
    coder = decomposition.SparseCoder(
        dictionary, transform_algorithm=algorithm, transform_alpha=0.2, split_sign=True
    )
    split = coder.transform(samples)
    assert (split >= 0).all()
    np.testing.assert_allclose(
        coder.inverse_transform(split), expected @ dictionary, atol=1e-5
    )


@pytest.mark.parametrize(
    "solver,beta,sparse_input",
    [
        ("cd", "frobenius", False),
        ("mu", "frobenius", False),
        ("mu", "kullback-leibler", True),
        ("mu", "itakura-saito", False),
    ],
)
def test_nmf_solvers(samples, solver, beta, sparse_input):
    X = np.abs(samples) + 0.01
    if sparse_input:
        X = sparse.csr_matrix(X * (X > 0.4))
    options = {
        "n_components": 3,
        "random_state": 42,
        "max_iter": 40,
        "tol": 0,
        "solver": solver,
        "beta_loss": beta,
    }
    expected = reference.NMF(**options)
    model = decomposition.NMF(**options)
    W = model.fit_transform(X)
    np.testing.assert_allclose(W, expected.fit_transform(X), atol=1e-5)
    np.testing.assert_allclose(model.components_, expected.components_, atol=1e-5)
    np.testing.assert_allclose(model.transform(X), expected.transform(X), atol=1e-5)
    W, H, n_iter = decomposition.non_negative_factorization(X, **options)
    assert n_iter == model.n_iter_
    np.testing.assert_allclose(W @ H, model.inverse_transform(W), atol=1e-8)


@pytest.mark.parametrize("fresh_restarts", [False, True])
def test_minibatch_nmf_fit_and_continue(samples, fresh_restarts):
    X = np.abs(samples)
    options = {
        "n_components": 3,
        "batch_size": 7,
        "max_iter": 4,
        "random_state": 42,
        "tol": 0,
        "fresh_restarts": fresh_restarts,
    }
    expected = reference.MiniBatchNMF(**options)
    model = decomposition.MiniBatchNMF(**options)
    np.testing.assert_allclose(
        model.fit_transform(X), expected.fit_transform(X), atol=1e-5
    )
    for batch in (X[:7], X[7:14]):
        assert model.partial_fit(batch) is model
        expected.partial_fit(batch)
    assert model.n_steps_ == expected.n_steps_
    np.testing.assert_allclose(model.components_, expected.components_, atol=1e-5)
    np.testing.assert_allclose(model.transform(X), expected.transform(X), atol=1e-5)


@pytest.mark.parametrize(
    "method,dtype", [("batch", np.float64), ("online", np.float32)]
)
def test_lda_sparse_and_online(samples, method, dtype):
    rng = np.random.RandomState(12)
    X = sparse.csr_matrix(rng.poisson(2, samples.shape).astype(dtype))
    options = {
        "n_components": 3,
        "max_iter": 5,
        "learning_method": method,
        "batch_size": 7,
        "random_state": 42,
    }
    expected = reference.LatentDirichletAllocation(**options).fit(X)
    model = decomposition.LatentDirichletAllocation(**options).fit(X)
    assert model.transform(X).dtype == dtype
    assert model.n_iter_ == expected.n_iter_
    assert model.n_batch_iter_ == expected.n_batch_iter_
    np.testing.assert_allclose(model.transform(X), expected.transform(X), atol=1e-5)
    np.testing.assert_allclose(model.perplexity(X), expected.perplexity(X), rtol=1e-5)
    model.partial_fit(X[:7])
    expected.partial_fit(X[:7])
    # Online updates can move the float32 document convergence check by one step.
    atol = 3e-5 if dtype == np.float32 else 1e-5
    np.testing.assert_allclose(model.transform(X), expected.transform(X), atol=atol)


def test_dictionary_function_entry_points(samples):
    options = {"n_components": 3, "max_iter": 4, "alpha": 0.7, "random_state": 42}
    expected = reference.dict_learning(samples, **options, return_n_iter=True)
    actual = decomposition.dict_learning(samples, **options, return_n_iter=True)
    for result, oracle in zip(actual, expected):
        np.testing.assert_allclose(result, oracle, atol=1e-5)
    expected = reference.dict_learning_online(samples, **options, batch_size=5)
    actual = decomposition.dict_learning_online(samples, **options, batch_size=5)
    for result, oracle in zip(actual, expected):
        np.testing.assert_allclose(result, oracle, atol=1e-5)


def test_randomized_svd_sparse_wide_input(samples):
    from sklearn.utils.extmath import randomized_svd

    X = sparse.csr_matrix(samples.T)
    expected = randomized_svd(X, 3, n_oversamples=2, random_state=9)
    result = decomposition.randomized_svd(X, 3, n_oversamples=2, random_state=9)
    for actual, oracle in zip(result, expected):
        np.testing.assert_allclose(actual, oracle, atol=1e-5)


def test_public_decomposition_entries():
    required = {
        "DictionaryLearning",
        "FactorAnalysis",
        "FastICA",
        "IncrementalPCA",
        "KernelPCA",
        "LatentDirichletAllocation",
        "MiniBatchDictionaryLearning",
        "MiniBatchNMF",
        "MiniBatchSparsePCA",
        "NMF",
        "PCA",
        "SparseCoder",
        "SparsePCA",
        "TruncatedSVD",
        "dict_learning",
        "dict_learning_online",
        "fastica",
        "non_negative_factorization",
        "randomized_svd",
        "sparse_encode",
    }
    assert required <= set(decomposition.__all__)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rank_deficient", [False, True])
@pytest.mark.parametrize("solver", ["auto", "dense"])
def test_linear_kernel_pca_tall(dtype, rank_deficient, solver):
    X = np.random.RandomState(11).normal(size=(260, 8)).astype(dtype) + 2
    if rank_deficient:
        X[:, 1:] = X[:, :1]
    options = dict(
        n_components=3, kernel="linear", eigen_solver=solver,
        remove_zero_eig=True, random_state=0
    )
    expected = reference.KernelPCA(**options).fit(X)
    actual = decomposition.KernelPCA(**options).fit(X)
    np.testing.assert_allclose(actual.eigenvalues_, expected.eigenvalues_, rtol=1e-5)
    np.testing.assert_allclose(actual.transform(X[:8]), expected.transform(X[:8]), atol=2e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_omp_preserves_dtype_and_support(dtype):
    rng = np.random.RandomState(19)
    X = rng.normal(size=(27, 32)).astype(dtype)
    dictionary = rng.normal(size=(40, 32)).astype(dtype)
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    kwargs = dict(algorithm="omp", n_nonzero_coefs=8, n_jobs=2)
    expected = reference.sparse_encode(X, dictionary, **kwargs)
    actual = decomposition.sparse_encode(X, dictionary, **kwargs)
    assert actual.dtype == dtype
    assert actual.flags.writeable
    np.testing.assert_array_equal(actual != 0, expected != 0)
    np.testing.assert_allclose(actual, expected, atol=2e-5)


def test_nmf_float32_shuffled_update():
    X = np.asfortranarray(
        np.abs(np.random.RandomState(21).normal(size=(40, 15))), dtype=np.float32
    )
    options = dict(
        n_components=4,
        init="random",
        random_state=0,
        max_iter=20,
        tol=0,
        shuffle=True,
        alpha_W=0.02,
        alpha_H=0.03,
        l1_ratio=0.5,
    )
    expected = reference.NMF(**options)
    actual = decomposition.NMF(**options)
    actual_weights = actual.fit_transform(X)
    expected_weights = expected.fit_transform(X)
    assert actual_weights.dtype == expected_weights.dtype == np.float32
    np.testing.assert_allclose(actual_weights, expected_weights, atol=2e-5)
    np.testing.assert_allclose(actual.components_, expected.components_, atol=2e-5)
