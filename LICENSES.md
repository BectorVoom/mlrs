# Licenses

This project depends on the following third-party libraries and frameworks. All dependencies are listed with their respective licenses.

## Direct Dependencies (Apache 2.0)

### CubeCL (0.10.0)
- **License**: Apache License 2.0
- **URL**: https://github.com/tracel-ai/cubecl
- **Purpose**: Compute kernels written in CubeCL and made generic over floating-point type (f32/f64) and CubeCL runtime for deployment on CUDA, ROCm, wgpu, or CPU

### Apache Arrow (59)
- **License**: Apache License 2.0
- **URL**: https://github.com/apache/arrow
- **Purpose**: Zero-copy data interchange layer between Rust and Python/PyArrow

### mimalloc (0.1)
- **License**: MIT License
- **URL**: https://github.com/microsoft/mimalloc
- **Purpose**: Custom global allocator wired into mlrs-py (with local_dynamic_tls for dlopen safety)

### cubecl-macros (0.10.0)
- **License**: Apache License 2.0
- **URL**: https://github.com/tracel-ai/cubecl
- **Purpose**: CubeCL macro system for kernel construction

## Language Bindings

### PyO3 (0.28)
- **License**: MIT/Apache 2.0 (dual-licensed)
- **URL**: https://github.com/PyO3/PyO3
- **Purpose**: Python bindings for the Rust core, enabling the generation of sklearn-compatible Python estimators
- **Note**: Pinned to 0.28 due to Arrow's pyarrow feature transitively depending on pyo3 0.28.x (ABI compatibility requirement)

## Testing & Development

### scikit-learn (>=1.6)
- **License**: BSD 3-Clause
- **URL**: https://github.com/scikit-learn/scikit-learn
- **Purpose**: Primary oracle for numerical correctness (≤ 1e-5 absolute/relative error tolerance)
- **Role**: Provides sklearn-compatible estimator surface for validation and API reference

### tracel-tblgen-rs (20.1.4-7)
- **License**: Apache License 2.0
- **URL**: https://github.com/tracel-ai/tracel-tblgen-rs
- **Purpose**: Code generation for MLIR dialects

## Open-Source Acknowledgements

### cuML (RAPIDS cuML)
- **License**: Apache 2.0 (underlying algorithms)
- **URL**: https://github.com/rapidsai/cuml
- **Role**: Source of algorithm inspiration and validation baseline
- **Note**: mlrs is a ground-up rewrite of cuML algorithms in Rust, matching numerical results within 1e-5 tolerance. While algorithm design draws from cuML's scikit-learn-compatible API, this is an independent implementation without direct code copying.

### NumPy (>=2.0.0)
- **License**: BSD 3-Clause
- **URL**: https://github.com/numpy/numpy
- **Purpose**: Array manipulation and numerical computation

## License Compatibility

All dependencies are OSI-approved open source licenses that are compatible with:
- Apache License 2.0 (primary project license)
- MIT License (allowed as dual-licensed in PyO3)

The project follows Apache 2.0 licensing for all source code, with dependencies explicitly documented as required by their respective licenses.