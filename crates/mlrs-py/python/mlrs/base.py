"""Pure-Python base estimator for the mlrs shim (D-01 / D-03).

``MlrsBase`` subclasses sklearn ``BaseEstimator`` *directly* (D-01) rather than
re-implementing ``get_params`` / ``set_params`` / ``clone`` / ``__repr__`` in
Rust — those come for free given a faithful ``__init__`` (every constructor
argument stored verbatim under the same name; validation deferred to ``fit``;
``fit`` returns ``self``). See RESEARCH 06 §Hyperparameter Mapping + the
``__init__`` purity rule (Pitfall 4).

It adds:
  * an ``output_type='input'`` constructor param (stored verbatim) + the
    ``_normalize`` / ``_normalize_y`` ingress and ``_to_output`` egress helpers
    delegating to :mod:`mlrs._io` (D-02 / D-03 mirror routing — narrowed to the
    numpy + pyarrow set, the mlrs analog of cuML ``base.py::_get_output_type``).
  * ``_check_fitted`` via ``sklearn.utils.validation.check_is_fitted`` so reading
    a fitted attribute / calling ``predict`` before ``fit`` raises sklearn's
    ``NotFittedError`` (T-06-13).
  * a ``__sklearn_tags__`` override turning off the sparse / array-api / NaN
    checks mlrs intentionally does not support (RESEARCH 06 §estimator_checks).
"""

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from . import _io


class MlrsBase(BaseEstimator):
    """sklearn-compatible base for every mlrs estimator.

    Invariants (enforced by every subclass):
      - ``__init__`` stores each ctor arg verbatim, same name, nothing else
        (sklearn ``check_no_attributes_set_in_init`` /
        ``check_parameters_default_constructible``); ``output_type='input'`` is
        the only param this base contributes.
      - ``fit`` returns ``self`` (PY-01).
      - Fitted attributes end with ``_`` and raise ``NotFittedError`` before
        ``fit`` (via :meth:`_check_fitted`).
    """

    def __init__(self, output_type="input"):
        self.output_type = output_type

    # -- ingress (D-02) ---------------------------------------------------- #

    def _normalize(self, X, dtype=None):
        """numpy/list/pyarrow ``X`` -> ``(fresh pyarrow array, rows, cols)``."""
        return _io.normalize_X(X, dtype=dtype)

    def _normalize_y(self, y, dtype):
        """1-D target ``y`` -> a fresh-contiguous pyarrow float array."""
        return _io.normalize_y(y, dtype=dtype)

    # -- egress (D-03) ----------------------------------------------------- #

    def _resolve_output_type(self, input_obj):
        """The egress container for ``input_obj`` under ``self.output_type``."""
        return _io.resolve_output_type(input_obj, self.output_type)

    def _to_output(self, buf, shape, input_obj, dtype):
        """Wrap a host buffer back into the resolved output container (D-03)."""
        ot = self._resolve_output_type(input_obj)
        return _io.to_output(buf, shape, ot, dtype)

    # -- fitted-state contract (T-06-13) ----------------------------------- #

    def _check_fitted(self):
        """Raise ``NotFittedError`` if ``fit`` has not run on this estimator.

        Subclasses store the compiled ``_mlrs`` handle under ``self._mlrs_obj``
        once fitted; ``check_is_fitted`` keys on that trailing-underscore-free
        private attribute via the explicit ``attributes`` argument.
        """
        check_is_fitted(self, attributes="_mlrs_obj")

    # -- sklearn >=1.6 tags (RESEARCH §estimator_checks) ------------------- #

    def __sklearn_tags__(self):
        """Disable sparse / array-api / NaN checks mlrs does not support.

        mlrs ingests dense Arrow only; the Rust bridge hard-rejects nulls; there
        is no array-api dispatch. Turning these tags off keeps the estimator_checks
        harness from running checks mlrs intentionally fails by design.
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        tags.array_api_support = False
        return tags
