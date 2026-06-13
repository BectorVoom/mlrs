"""Input/output normalization for the mlrs shim (D-02 / D-03).

Importable *shell* only — Plan 04 fills in:
  - ``to_pyarrow(X)``: normalize numpy arrays / Python lists into a
    *freshly-contiguous* 1-D pyarrow float array (the D-02 row-major flatten).
    The Rust bridge ``validate_f32`` / ``validate_f64`` HARD-REJECTS sliced /
    offset arrays (it requires the values view to cover the whole backing
    buffer), so the shim must hand a fresh ``pa.array(np.ascontiguousarray(X)
    .ravel())`` — never a zero-copy slice of a larger numpy buffer
    (RESEARCH 06 Pitfall 3).
  - ``resolve_dtype(X)``: pick f32 vs f64. Default to f64 *where supported*;
    on an f64-incapable backend (rocm) default to f32 by querying
    ``mlrs.backend_supports_f64()`` so integer/list inputs don't hit the
    D-04 error (RESEARCH 06 Pitfall 5).
  - Output routing (D-03): wrap returned host buffers back into the configured
    ``output_type`` (numpy or pyarrow), mirroring cuML
    ``base.py::_get_output_type`` + ``array.py::to_output``.

No logic lands here at Wave 0; the wrappers in Plan 04 import these helpers.
"""
