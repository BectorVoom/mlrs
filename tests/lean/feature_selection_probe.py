"""Exercise mlrs on sparse inputs and report the observations to Lean."""

import json
import warnings

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

from mlrs import feature_selection as fs


def main():
    x = sparse.csr_matrix(
        [[0, 3, 0, 0], [0, 3, 1, 1], [1, 3, 0, 2],
         [4, 3, 1, 0], [4, 3, 0, 1], [3, 3, 1, 2]]
    )
    y = [0, 0, 0, 1, 1, 1]
    variance = fs.VarianceThreshold(threshold=0.5).fit(x.tocsc())
    top = fs.SelectKBest(fs.chi2, k=2).fit(x, y)
    selected = top.transform(x)
    restored = top.inverse_transform(selected)
    meta = fs.RFE(Ridge(), n_features_to_select=1).fit(x, x[:, 0].toarray().ravel())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        empty_selector = fs.SelectKBest(fs.chi2, k=0).fit(x, y)
        empty = empty_selector.transform(sparse.csr_array(x))
    empty_restored = empty_selector.inverse_transform(empty)

    def score_sparse(matrix, target):
        if not sparse.issparse(matrix) or target is not None:
            raise ValueError("expected sparse input and no target")
        return np.asarray(matrix.sum(axis=0)).ravel()

    custom = fs.SelectKBest(score_sparse, k=1).fit(x.tocoo())
    dense_top = fs.SelectKBest(fs.chi2, k=2, output_type="numpy").fit(x, y)
    dense_result = dense_top.transform(x)
    dense_restored = dense_top.inverse_transform(selected)
    print(json.dumps({
        "variances": variance.variances_.tolist(),
        "varianceMask": variance.get_support().tolist(),
        "scores": top.scores_.tolist(),
        "mask": top.get_support().tolist(),
        "selected": selected.toarray().tolist(),
        "restored": restored.toarray().tolist(),
        "selectedIsSparse": sparse.issparse(selected),
        "restoredIsSparse": sparse.issparse(restored),
        "emptyShape": empty.shape,
        "emptyRestoredShape": empty_restored.shape,
        "emptyRestoredNnz": empty_restored.nnz,
        "emptyRestoredIsSparseArray": isinstance(empty_restored, sparse.sparray),
        "emptyWarning": any("No features were selected" in str(w.message) for w in caught),
        "customMask": custom.get_support().tolist(),
        "metaMask": meta.get_support().tolist(),
        "metaIsSparse": sparse.issparse(meta.transform(x)),
        "denseOutput": isinstance(dense_result, np.ndarray)
        and isinstance(dense_restored, np.ndarray)
        and np.array_equal(dense_result, selected.toarray())
        and np.array_equal(dense_restored, restored.toarray()),
    }, allow_nan=False))


if __name__ == "__main__":
    main()
