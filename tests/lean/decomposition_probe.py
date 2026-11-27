"""Collect observations from the installed mlrs package for Decomposition.lean."""

import json

import numpy as np
from mlrs import decomposition


def main():
    samples = np.array([[3.0, -2.0, 0.5], [-0.5, 2.0, -4.0]])
    codes = decomposition.sparse_encode(
        samples, np.eye(3), algorithm="lasso_lars", alpha=1.0
    )

    model = decomposition.NMF(n_components=1, init="random", random_state=0)
    weights = model.fit_transform(np.array([[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]))

    _, singular_values, _ = decomposition.randomized_svd(
        np.diag([5.0, 3.0, 1.0]), 2, random_state=0
    )

    counts = np.array([[5, 1, 0], [0, 1, 5], [2, 1, 2]])
    topics = decomposition.LatentDirichletAllocation(
        n_components=2, max_iter=5, random_state=0
    ).fit_transform(counts)

    print(
        json.dumps(
            {
                "sparseCodes": codes.tolist(),
                "nmfWeights": weights.tolist(),
                "nmfComponents": model.components_.tolist(),
                "singularValues": singular_values.tolist(),
                "topicProbabilities": topics.tolist(),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
