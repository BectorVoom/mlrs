"""Pure-Python base estimator for the mlrs shim (D-01).

This is the importable *shell* only. Plan 04 fills in:
  - ``output_type`` routing (numpy / pyarrow), mirroring cuML
    ``internals/base.py`` ``_set_output_type`` / ``_get_output_type``
    (D-03 egress; default ``"input"`` → infer from the container the data
    arrived in, narrowed to numpy + pyarrow only).
  - The ``_to_output`` egress wrapper (cuML ``CumlArray.to_output`` analog).

Design (D-01): mlrs subclasses sklearn ``BaseEstimator`` *directly* rather than
re-implementing ``get_params`` / ``set_params`` / ``clone`` / ``__repr__`` in
Rust. Those come for free from sklearn given a faithful ``__init__`` (every
constructor argument stored verbatim under the same name; validation deferred
to ``fit``; ``fit`` returns ``self``). See RESEARCH 06 §Hyperparameter Mapping
and the ``__init__`` purity rule (Pitfall 4).
"""

from sklearn.base import BaseEstimator


class MlrsBase(BaseEstimator):
    """sklearn-compatible base for every mlrs estimator.

    Plan 04 adds ``output_type`` storage and the device<->host egress
    plumbing. At Wave 0 this is a thin subclass so the family modules import
    and the 12 estimators are constructible pure-Python shells.

    Invariants (enforced from Wave 0 onward):
      - ``__init__`` stores each ctor arg verbatim, same name, nothing else
        (sklearn ``check_no_attributes_set_in_init`` /
        ``check_parameters_default_constructible``).
      - ``fit`` returns ``self`` (PY-01).
      - Fitted attributes end with ``_`` and raise ``NotFittedError`` before
        ``fit`` (filled in Plan 04 via ``check_is_fitted``).
    """
