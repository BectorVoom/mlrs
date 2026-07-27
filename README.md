# mlrs

mlrs is a ground-up rewrite of RAPIDS cuML's machine-learning algorithms in Rust.
Compute kernels are written once in [CubeCL](https://github.com/tracel-ai/cubecl) and made
generic over both the floating-point type (`f32`/`f64`) and the CubeCL runtime, so the same
algorithm runs on CUDA, ROCm, wgpu, or CPU selected at build time via Cargo features. It ships
sklearn-compatible Python estimators (via PyO3) so data scientists on Python ≥ 3.12 can `pip
install` the package for their backend and use familiar `fit`/`predict`/`transform` APIs.

**Core Value:** **Correct, memory-efficient ML algorithms that match scikit-learn within 1e-5, running on any
CubeCL backend from a single generic codebase.** If everything else fails, the numerical results
must be right and the backend abstraction must hold.

## Dependencies & Attributions

### Primary Oracles & Inspirations
- **scikit-learn** (BSD 3-Clause): All 30+ estimators validated for numerical correctness (≤ 1e-5 tolerance)
- **RAPIDS cuML** (Apache 2.0): Algorithm inspiration for API design and correctness baseline

### Runtime & Libraries
- **CubeCL** (Apache 2.0): Compute kernel runtime for CUDA/ROCm/wgpu/CPU
- **PyO3** (MIT/Apache 2.0): Python bindings layer
- **Apache Arrow** (Apache 2.0): Zero-copy data interchange
- **mimalloc** (MIT): Global allocator with dlopen-safe configuration

### Usage & Validation
- **Python ≥3.12**: Required for estimators
- **NumPy >=2.0.0**: Array operations

## License & Attribution

Please see [LICENSES.md](./LICENSES.md) for detailed license information and third-party attributions, including dependencies on scikit-learn, RAPIDS cuML, CubeCL, and other open-source projects that enable this implementation.

## Quick Start

For usage information and examples, please refer to the project documentation and tests in the repository.