"""Kernel-ridge estimator shim (PY-01/PY-02) delegating to ``_mlrs``.

KernelRidge -> ``RegressorMixin``. sklearn-faithful ``__init__`` stores every
ctor arg verbatim under the SAME name (purity rule — the AST gate enforces it).
``fit`` normalizes via the base, constructs ``_mlrs.KernelRidge``, stores the
handle on ``self._mlrs_obj`` and returns ``self`` (PY-01). ``predict`` forwards
to the dtype-suffixed accessor (D-06); ``dual_coef_`` materializes the fitted
dual coefficients.

Defaults mirror ``PyKernelRidge`` ``#[new]`` at
``crates/mlrs-py/src/estimators/kernel.rs:147-149``.
"""

import numpy as np
from sklearn.base import RegressorMixin

from .base import MlrsBase


class KernelRidge(RegressorMixin, MlrsBase):
    """Kernel ridge regression (KERNEL-01).

    ``KernelRidge(kernel="linear", alpha=1.0, gamma=None, degree=3.0, coef0=1.0)``.
    """

    def __init__(
        self,
        kernel="linear",
        alpha=1.0,
        gamma=None,
        degree=3.0,
        coef0=1.0,
        output_type="input",
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.output_type = output_type

    def fit(self, X, y):
        xa, rows, cols = self._normalize(X)
        ya = self._normalize_y(y, dtype=self._x_float(xa))
        obj = self._ext().KernelRidge(
            self.kernel, self.alpha, self.gamma, self.degree, self.coef0
        )
        obj.fit(xa, ya, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def predict(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("predict")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())

    @property
    def dual_coef_(self):
        return self._to_output(
            self._suffixed("dual_coef")(), (-1,), None, self._np_float()
        )

    @staticmethod
    def _x_float(xa):
        return np.float32 if xa.type.bit_width == 32 else np.float64
