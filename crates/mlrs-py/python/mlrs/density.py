"""Kernel-density estimator shim (PY-01/PY-02) delegating to ``_mlrs``.

KernelDensity subclasses :class:`MlrsBase` only — sklearn's ``KernelDensity`` is
an unsupervised density estimator with no scoring/transformer family mixin (its
public surface is ``fit`` + ``score_samples``). The ``__init__`` stores every
ctor arg verbatim under the SAME name (purity rule — the AST gate enforces it).
``fit`` is unsupervised (``y=None``); ``score_samples`` returns the per-query
log-densities via the dtype-suffixed accessor (D-06 / D-12).

Defaults mirror ``PyKernelDensity`` ``#[new]`` at
``crates/mlrs-py/src/estimators/kernel.rs:361-363``.
"""

from .base import MlrsBase


class KernelDensity(MlrsBase):
    """Kernel density estimation (KERNEL-02).

    ``KernelDensity(kernel="gaussian", bandwidth=1.0, bandwidth_rule="numeric")``.
    No standalone ``predict`` — the public surface is ``fit`` + ``score_samples``.
    """

    def __init__(
        self,
        kernel="gaussian",
        bandwidth=1.0,
        bandwidth_rule="numeric",
        output_type="input",
    ):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_rule = bandwidth_rule
        self.output_type = output_type

    def fit(self, X, y=None):
        xa, rows, cols = self._normalize(X)
        obj = self._ext().KernelDensity(
            self.kernel, self.bandwidth, self.bandwidth_rule
        )
        obj.fit(xa, rows, cols)
        self._mlrs_obj = obj
        self._post_fit(cols)
        return self

    def score_samples(self, X):
        xa, rows, cols = self._check_predict_X(X)
        out = self._suffixed("score_samples")(xa, rows, cols)
        return self._to_output(out, (rows,), X, self._np_float())
