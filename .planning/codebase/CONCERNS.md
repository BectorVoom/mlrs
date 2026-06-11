# Codebase Concerns

**Analysis Date:** 2026-06-11
**Codebase:** RAPIDS cuML (`cuml-main/`) — version 26.08.00, branch `main`

---

## Tech Debt

**Multi-GPU Stream Management (MG files):**
- Issue: Multiple `_mg.cu` files manually allocate CUDA streams outside `raft::handle_t`, with repeated TODO comments saying streams should come from the handle instead.
- Files: `cpp/src/glm/ols_mg.cu` (lines 108, 192), `cpp/src/glm/ridge_mg.cu` (lines 236, 323), `cpp/src/pca/pca_mg.cu` (lines 144, 285, 415, 482), `cpp/src/solver/cd_mg.cu` (lines 277, 368), `cpp/src/tsvd/tsvd_mg.cu` (lines 134, 229, 315, 466, 508)
- Impact: Stream lifecycle is not managed by the central resource handle, making it fragile under multi-GPU teardown and incompatible with future raft handle evolution.
- Fix approach: Route all stream creation/sync through `raft::resource::get_stream_pool(handle)` and remove manual stream arrays.

**`#TODO: Replace with public header when ready` Pattern:**
- Issue: Multiple modules use `raft::linalg::detail::cublas_wrappers.hpp` (a private/internal raft header) directly, bypassing the public API. These are annotated with `#TODO: Replace with public header when ready`.
- Files: `cpp/src/holtwinters/runner.cuh` (line 16), `cpp/src/glm/qn/simple_mat/dense.hpp` (line 17), `cpp/src/svm/smosolver.cuh` (line 10), `cpp/src/solver/lars_impl.cuh` (line 27), `cpp/src/arima/batched_kalman.cu` (line 20), `cpp/src/holtwinters/internal/hw_decompose.cuh` (lines 12, 14)
- Impact: Private raft headers can be restructured without public API guarantees, meaning these files break silently on raft upgrades.
- Fix approach: Track raft public API additions for cuBLAS/cuSolver/cuSPARSE wrappers and migrate each call site.

**Softmax GLM Kernel Fusion:**
- Issue: The softmax gradient computation in QN GLM involves multiple sequential operations that should be kernel-fused. Multiple TODOs identify unfused steps, unnecessary intermediate reads, and an unoptimized large-class code path.
- Files: `cpp/src/glm/qn/glm_softmax.cuh` (lines 30, 33, 51, 90), `cpp/src/glm/qn/glm_base.cuh` (lines 44, 45, 77, 135)
- Impact: Performance regression for multinomial logistic regression with many classes. Memory bandwidth bottleneck.
- Fix approach: Introduce CUTLASS-based fused GEMM+bias broadcast kernel; split small/large-class code paths.

**MLCommon Legacy Namespace:**
- Issue: ~404 references to `MLCommon::` namespace remain in `cpp/src/`, specifically for `MLCommon::Matrix` (250 refs), `MLCommon::LinAlg` (92 refs), and `MLCommon::TimeSeries` (15 refs). This is a pre-raft namespace that has been partially migrated.
- Files: `cpp/src/glm/ols_mg.cu`, `cpp/src/glm/ridge_mg.cu`, `cpp/src/pca/pca_mg.cu`, `cpp/src/solver/cd_mg.cu`, and other MG files.
- Impact: These references tie the build to internal cuML header paths (`cpp/include/cuml/prims/opg/`) rather than raft, blocking full raft unification.
- Fix approach: Audit `MLCommon::Matrix::Data<T>` and `MLCommon::Matrix::PartDescriptor` and replace with raft distributed matrix types.

**Genetic Programming Memory Transfer:**
- Issue: `genetic.cu` and `program.cu` have TODOs noting excessive implicit host-device memory transfers with no better approach identified.
- Files: `cpp/src/genetic/genetic.cu` (lines 217, 512, 541), `cpp/src/genetic/program.cu` (line 225), `cpp/src/genetic/fitness.cuh` (line 197)
- Impact: High data-copy overhead during fitness evaluation; degrades throughput on large populations.
- Fix approach: Batch fitness evaluations into a single kernel; keep program trees in device memory across generations.

**PCA Multi-GPU Thrust Calls:**
- Issue: Two functions in PCA multi-GPU sign-flip code use Thrust instead of native CUDA kernels or raft prims, annotated as needing replacement.
- Files: `cpp/src/pca/sign_flip_mg.cu` (lines 36, 90)
- Impact: Thrust launch overhead in a hot path; harder to control stream affinity in MG context.
- Fix approach: Replace with raft `raft::linalg::` primitives operating on the `raft::handle_t` stream.

**HDBSCAN `cluster_selection_epsilon` (Epsilon Search) Incomplete:**
- Issue: The eom (excess of mass) cluster selection path in HDBSCAN has a `// TODO: Implement the cluster_selection_epsilon / epsilon_search` comment; only a partial approximation is in place.
- Files: `cpp/src/hdbscan/detail/extract.cuh` (line 142), `cpp/src/hdbscan/detail/select.cuh` (line 418)
- Impact: `cluster_selection_epsilon` results may diverge from the reference `hdbscan` Python library for non-zero epsilon values.
- Fix approach: Implement the full EOMSC epsilon traversal as in the reference implementation's `epsilon_search`.

**HDBSCAN Outlier Score Not Computed:**
- Issue: `cpp/src/hdbscan/detail/membership.cuh` line 37 marks outlier score computation as TODO; the soft clustering membership returns zeros for outlier scores.
- Files: `cpp/src/hdbscan/detail/membership.cuh` (line 37)
- Impact: Users relying on `outlier_scores_` attribute get incorrect (zeroed) output.

**Deprecated Thrust `FlagHeads` in Decision Tree:**
- Issue: Uses Thrust's deprecated `FlagHeads` primitive instead of the replacement `SubtractLeft`, with a TODO acknowledging this.
- Files: `cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh` (line 226)
- Impact: Will break when CCCL removes `FlagHeads` in a future release.
- Fix approach: Replace with `cub::DevicePartition::Flagged` or `cub::DeviceSelect::Flagged` with `SubtractLeft` semantics.

---

## Known Bugs

**KMeans int32 Index Overflow (Recently Fixed, Risk Remains):**
- Symptoms: Large datasets (many rows × clusters) cause index overflow in KMeans int32 index calculations.
- Files: `cpp/src/kmeans/` (fixed in 26.06 per CHANGELOG — `Fix KMeans int32 index overflow check`)
- Trigger: Very large n_samples or n_clusters exceeding int32 range.
- Workaround: Use dataset sizes safely below INT32_MAX per-cluster-count products.

**UMAP Non-Determinism Under Vertex-Parallel Kernels:**
- Symptoms: UMAP produced non-sequential (non-deterministic) results due to vertex-parallel kernel race conditions.
- Files: `cpp/src/umap/simpl_set_embed/optimize_batch_kernel.cuh`
- Trigger: Fixed in 26.06 (`Add vertex-parallel kernels to UMAP to enforce sequential behavior`), but non-determinism can re-emerge from cuRand order changes.
- Workaround: Set `random_state` explicitly and verify reproducibility across runs.

**Holtwinters NaN Errors (GCC 9.3 Regression):**
- Symptoms: Statsmodels-based tests produce NaN for certain Holtwinters configurations when compiled with GCC 9.3.
- Files: `python/cuml/tests/test_holtwinters.py` (lines 393, 425) — marked `xfail`
- Trigger: Issue #3384; optimizer divergence on specific time series.
- Workaround: Use GCC >= 10 or avoid Holtwinters on pathological time series.

**ARIMA Exogenous Coefficient Interpolation:**
- Symptoms: ARIMA solves exogenous coefficients using interpolation rather than fitting only valid rows, which may produce incorrect estimates for gapped data.
- Files: `cpp/src/arima/batched_arima.cu` (line 953)
- Trigger: Time series with missing exogenous values.

**FIL Buffer Bounds OOB (Recently Fixed):**
- Symptoms: FIL (Forest Inference Library) had buffer bounds assertions that could trigger out-of-bounds memory access on malformed Treelite model input.
- Files: `python/cuml/cuml/fil/` — fixed in 26.06 (`Fix FIL buffer bounds assertions`)
- Trigger: Integer overflow in Treelite model with extremely wide trees.
- Workaround: Validate Treelite model input before passing to FIL.

**Spectral Clustering Known Failure:**
- Symptoms: Test `test_spectral_clustering.py` has an xfail for issue #7714.
- Files: `python/cuml/tests/test_spectral_clustering.py` (line 246)
- Trigger: Unknown — tracked in GitHub issue #7714.

---

## Security Considerations

**No Secrets in Build System:**
- Risk: CMake files fetch dependencies from GitHub via HTTPS at build time (RAFT, cuVS, nvForest, CCCL, treelite). A compromised pinned tag or GitHub repository could inject malicious code.
- Files: `cpp/cmake/thirdparty/get_raft.cmake`, `cpp/cmake/thirdparty/get_cuvs.cmake`, `cpp/cmake/thirdparty/get_nvforest.cmake`
- Current mitigation: Tags are pinned to `${rapids-cmake-checkout-tag}` (synchronized with RAPIDS version). No hash pinning.
- Recommendations: Add CMake SHA256 hash verification for CPM-downloaded sources. Consider vendoring or using a private mirror.

**Vulnerability Disclosure:**
- Process: Reports routed to NVIDIA PSIRT via `https://www.nvidia.com/en-us/security/report-vulnerability/` or `psirt@nvidia.com`. Documented in `SECURITY.md`.
- No public vulnerability tracker is maintained in the repository itself.

**CUDA Memory Safety:**
- Risk: Raw `cudaMalloc`/`cudaMemcpy` calls in the Barnes-Hut TSNE (cannylab) code bypass RMM and have no bounds checking.
- Files: `cpp/src/tsne/cannylab/bh.cu` (lines 908–987, 1079–1086)
- Current mitigation: None — this is embedded third-party code (Texas State University ECL-BH, BSD-3-Clause).
- Recommendations: Migrate to `rmm::device_uvector` allocations inside a raft handle to get RMM error handling.

**Deprecation Warning Suppression:**
- Risk: Build option `DISABLE_DEPRECATION_WARNINGS` defaults to `ON` in `cpp/CMakeLists.txt` (line 56). This hides all deprecation warnings during compilation, masking silent API drift from raft/CCCL/CUDA Toolkit updates.
- Files: `cpp/CMakeLists.txt` (line 56)
- Recommendations: Enable deprecation warnings in CI and fix outstanding usages before they become compilation errors.

---

## Performance Bottlenecks

**Barnes-Hut TSNE (cannylab) — Unmanaged Memory:**
- Problem: The entire BH-TSNE implementation (`bh.cu`) uses raw `cudaMalloc` rather than RMM pooled allocation. Each TSNE call allocates and frees GPU memory from scratch.
- Files: `cpp/src/tsne/cannylab/bh.cu`
- Cause: Inherited from the embedded ECL-BH third-party code; not refactored.
- Improvement path: Wrap allocations in `rmm::device_buffer` with a stream-aware pool to avoid repeated CUDA context synchronization on alloc/free.

**QN Solver Workspace Allocation:**
- Problem: The QN solver (L-BFGS) workspace allocation happens inside the solve loop rather than being pre-allocated outside. Marked as TODO.
- Files: `cpp/src/glm/qn/qn_solvers.cuh` (line 415)
- Cause: Historical incremental development; workspace growth logic mixed with solver loop.
- Improvement path: Compute max workspace size before the solve loop and allocate once.

**ARIMA Batched Kalman — Shared Memory Usage:**
- Problem: Two inner loops in the Kalman filter kernel could use shared memory for the `R` matrix instead of precomputed global memory reads, but this is unimplemented.
- Files: `cpp/src/arima/batched_kalman.cu` (lines 550, 628)
- Cause: Optimization not yet implemented.
- Improvement path: Allocate `R` in `__shared__` and broadcast across warps in the same ARIMA batch.

**TSNE KL Divergence Not Computed:**
- Problem: Both exact TSNE and Barnes-Hut TSNE have `// TODO: Calculate Kullback-Leibler divergence` at the end of their gradient loops. The KL divergence loss is not returned.
- Files: `cpp/src/tsne/exact_kernels.cuh` (line 256), `cpp/src/tsne/barnes_hut_tsne.cuh` (line 275)
- Cause: Optimization of convergence metric deferred.
- Improvement path: Accumulate KL divergence in an `atomicAdd` reduction during the attractive force kernel.

**SVM Kernel Cache — Unused Update Path:**
- Problem: A `// TODO: utilize cache read without update` comment in the SVM kernel cache indicates that reads that should be no-ops still trigger write-backs.
- Files: `cpp/src/svm/kernelcache.cuh` (line 545)
- Cause: Conservative cache invalidation logic.
- Improvement path: Add a read-only cache access path that skips the LRU write-back for cache hits.

---

## Fragile Areas

**cuml.accel Drop-in Replacement Layer:**
- Files: `python/cuml/cuml/accel/estimator_proxy.py`, `python/cuml/cuml/accel/accelerator.py`, `python/cuml/cuml/accel/_overrides/sklearn/`
- Why fragile: Uses metaclass-based `ProxyBaseMeta` wrapping to intercept sklearn estimator calls. Relies on internal sklearn attribute conventions (`_get_tags`, `_cpu_class`, `_parent_callback_ctx`). Python < 3.13 `inspect.signature` bug is worked around with an XXX comment. sklearn version-specific deprecation sentinels (`"deprecated"` string values) are hard-coded.
- Safe modification: Any change to `ProxyBaseMeta.__call__` or `reconstruct` must be tested against the full cuml_accel_tests suite. Avoid touching the metaclass `__instancecheck__` path.
- Test coverage: `python/cuml/cuml_accel_tests/` and `python/cuml/cuml_accel_tests/upstream/scikit-learn/` — but several upstream sklearn tests are marked xfail.

**UMAP Large-n Dispatch (`dispatch_to_uint64_t`):**
- Files: `cpp/src/umap/umap.cuh` (line 25), `cpp/src/umap/umap.cu`
- Why fragile: Every UMAP entry point checks `dispatch_to_uint64_t(n, n_neighbors, n_components)` to switch between `int` and `uint64_t` index types. This doubles template instantiation and means any code path untested with large indices can silently overflow or use the wrong path.
- Safe modification: Always test UMAP with both small and large `n_rows` (above and below the `int32` threshold).
- Test coverage: Not explicitly parametrized in `python/cuml/tests/test_umap.py` for the large-index path.

**TSNE Barnes-Hut Cannylab Embedding:**
- Files: `cpp/src/tsne/cannylab/bh.cu`
- Why fragile: 1,111-line third-party CUDA file from Texas State University (ECL-BH, BSD-3-Clause, adapted from 2010-2020). Uses manual CUDA memory management, float-only arithmetic, and fixed warp-count FACTOR macros. Has no raft or RMM integration.
- Safe modification: Do not modify this file for performance; replace the entire BH TSNE with a new raft-native implementation if needed.
- Test coverage: `cpp/tests/sg/tsne_test.cu` — single-GPU only.

**FIL `compat.py` Module — Scheduled Removal:**
- Files: `python/cuml/cuml/fil/compat.py`
- Why fragile: The entire module is marked `TODO(26.10): This module will be removed in 26.10`. All five public symbols (`ForestInference`, `set_fil_device_type`, `.load()`, `.load_from_sklearn()`, `.load_from_treelite_model()`) emit `FutureWarning` and delegate to a new API. Any code depending on `cuml.fil.ForestInference` will break in 26.10.
- Safe modification: Migrate all callers to the new `cuml.fil` API before 26.10.

**Holtwinters Internal Headers:**
- Files: `cpp/src/holtwinters/runner.cuh`, `cpp/src/holtwinters/internal/hw_decompose.cuh`, `cpp/src/holtwinters/internal/hw_optim.cuh`
- Why fragile: Uses private raft cuBLAS/cuSolver wrappers via detail headers, and multiple functions are annotated `#TODO: Call from public API when ready`. Internal call sites inside the `.cuh` files call these raft detail functions directly.
- Test coverage: `cpp/tests/sg/holtwinters_test.cu` and `python/cuml/tests/test_holtwinters.py` — has known xfail cases.

---

## Scaling Limits

**KMeans/UMAP int32 Index Space:**
- Current capacity: `int32_t` indices by default, dispatching to `uint64_t` only in UMAP via `dispatch_to_uint64_t`.
- Limit: KMeans overflows for `n_samples * n_clusters > INT32_MAX` (recently patched in 26.06). Other algorithms remain int32-limited.
- Scaling path: Systematic index type templating across all algorithms, mirroring the UMAP pattern.

**Single-Node Memory Ceiling:**
- Current capacity: All single-GPU algorithms allocate from RMM-managed GPU VRAM. No disk-backed or CPU-hosted overflow.
- Limit: Datasets exceeding GPU VRAM are not handled; the job fails with an OOM error.
- Scaling path: Multi-GPU (MG) variants exist for GLM, PCA, TSVD, KNN, but not for SVM, TSNE, UMAP, or genetic programming.

**Dask Multi-GPU (MG) Gaps:**
- Current capacity: Dask-distributed wrappers exist in `python/cuml/cuml/dask/` for cluster, decomposition, linear model, manifold (UMAP only), and neighbors.
- Limit: TSNE, SVM, Holtwinters, ARIMA, Genetic Programming have no MG/Dask variants.
- Scaling path: Not planned in visible roadmap; would require C++ MG backend implementations.

---

## Dependencies at Risk

**RAPIDS Synchronized Version Pinning (All Core Deps):**
- Risk: RAFT, cuVS, nvForest, RMM, CCCL are all pinned to `${rapids-cmake-checkout-tag}`, which resolves to the same RAPIDS version as cuML (26.08.00). This means ALL these libraries must be upgraded in lockstep. There is no independent version compatibility range.
- Files: `cpp/cmake/thirdparty/get_raft.cmake`, `cpp/cmake/thirdparty/get_cuvs.cmake`, `cpp/cmake/thirdparty/get_nvforest.cmake`, `cpp/cmake/thirdparty/get_rmm.cmake`
- Impact: Consumers cannot pin cuML 26.08 against a patched raft or RMM; all RAPIDS packages must match.
- Migration plan: None practical — this is a deliberate RAPIDS ecosystem design decision.

**nvForest (New Dependency, 26.06):**
- Risk: `nvforest` was adopted in 26.06 (`Adopt nvForest for random forest inference`) to replace the old FIL C++ backend. It is a new NVIDIA-internal library (`rapidsai/nvforest`), not widely used outside cuML.
- Files: `cpp/cmake/thirdparty/get_nvforest.cmake`, `cpp/CMakeLists.txt` (lines 73–79, 278–279)
- Impact: Any Rust port that wants to use forest inference cannot simply call treelite; it must also understand or wrap nvForest's C++ ABI.
- Migration plan: Treelite remains the model serialization format; nvForest handles inference. A Rust port should target the treelite model format and implement its own inference.

**CUDA Toolkit 13.x Requirement:**
- Risk: `pyproject.toml` pins `cuda-toolkit[cublas,cufft,curand,cusolver,cusparse]==13.*` and `cuda-python>=13.0.1,<14.0`. CUDA 12.x is no longer supported.
- Files: `python/cuml/pyproject.toml`
- Impact: Hardware or cloud environments locked to CUDA 12.x are incompatible with cuML 26.08.
- Migration plan: No fallback; upgrade CUDA Toolkit.

**cuDF Version Pinning:**
- Risk: `cudf==26.8.*` is a hard pin. cuML internals (`python/cuml/cuml/internals/input_utils.py`, `array.py`) have deep integration with cuDF's internal APIs, including the DataFrame Interchange Protocol (now deprecated in cuDF itself).
- Files: `python/cuml/cuml/internals/validation.py` (line 246 — suppresses `Pandas4Warning` and Interchange Protocol deprecation)
- Impact: cuML will break against cuDF 27.x if cuDF removes the Interchange Protocol or changes array conversion APIs.
- Migration plan: Migrate to Array API Standard (`numpy.array_api`) and cuDF public pandas-compatible API.

**treelite `>=4.7.0,<5.0.0`:**
- Risk: treelite 5.x already exists upstream. The upper bound `<5.0.0` will prevent compatibility with treelite 5.x until cuML explicitly upgrades.
- Files: `python/cuml/pyproject.toml`
- Impact: Projects that upgrade treelite independently will be incompatible with cuML 26.08.
- Migration plan: Track treelite 5.x API changes and bump the constraint.

**sklearn Deprecation Churn:**
- Risk: Multiple files track sklearn deprecations with explicit version comments: `multi_class` (deprecated 1.5, removed 1.8), `penalty` (deprecated 1.8, removed 1.10), `_get_tags` (deprecated 1.6, removed 1.7), `sample_weight` in KMeans predict (removed 1.5).
- Files: `python/cuml/cuml/linear_model/logistic_regression.py` (lines 163–193), `python/cuml/cuml/internals/validation.py` (lines 62–74), `python/cuml/cuml/accel/_overrides/sklearn/cluster.py` (lines 25–27)
- Impact: Each sklearn minor release can break cuml.accel compatibility.
- Migration plan: Pin `scikit-learn>=1.6` (already done in `pyproject.toml`) and triage deprecated attribute access before each sklearn release.

---

## Missing Critical Features

**KL Divergence Output from TSNE:**
- Problem: TSNE fit does not return the final KL divergence loss value; the Python API marks `kl_divergence_` as "experimental" in `python/cuml/cuml/manifold/t_sne.pyx` (line 709).
- Blocks: Users cannot monitor convergence quality or compare TSNE runs programmatically.

**HDBSCAN Outlier Scores:**
- Problem: `outlier_scores_` is not computed in the GPU HDBSCAN soft clustering path (C++ TODO at `cpp/src/hdbscan/detail/membership.cuh` line 37).
- Blocks: Anomaly detection use-cases that rely on HDBSCAN outlier scores.

**TSNE Distance Metrics (Cosine etc.):**
- Problem: Five TODO comments in `cpp/src/tsne/exact_kernels.cuh` (lines 180, 222, 256, 291, 333) note that only Euclidean distance is implemented; cosine and other metrics are marked as future work.
- Blocks: NLP/embedding use-cases that require cosine TSNE.

**Genetic Programming Multi-Class:**
- Problem: `cpp/src/genetic/genetic.cu` (lines 512, 541) has two TODOs for multi-class classification support.
- Blocks: Symbolic regression for multi-class targets.

**Dask Naive Bayes Sparse Arrays:**
- Problem: `python/cuml/cuml/dask/naive_bayes/naive_bayes.py` (line 173) has a TODO noting cupy sparse arrays are not fully supported under Dask for Naive Bayes.
- Blocks: Distributed sparse text classification.

---

## Test Coverage Gaps

**C++ MG Tests Disabled by Default:**
- What's not tested: Multi-GPU algorithm correctness (`BUILD_CUML_MG_TESTS=OFF` by default in `cpp/CMakeLists.txt` line 48).
- Files: `cpp/tests/mg/`
- Risk: MG-specific stream management bugs (see Tech Debt section) go undetected in standard CI.
- Priority: High — MG code paths diverge significantly from single-GPU paths.

**UMAP Large-Index Path:**
- What's not tested: The `uint64_t` index dispatch branch in UMAP is not covered by parametrized tests.
- Files: `cpp/src/umap/umap.cu`, `python/cuml/tests/test_umap.py`
- Risk: Silent int32 truncation for large datasets.
- Priority: High.

**float16 Input Not Supported:**
- What's not tested: `python/cuml/tests/test_input_utils.py` (lines 214, 247) explicitly marks float16 as `xfail` with reason "float16 not yet supported by numba/cuDF".
- Files: `python/cuml/cuml/internals/input_utils.py`, `python/cuml/cuml/internals/array.py`
- Risk: Any caller passing float16 arrays will get an error or silent conversion, not GPU half-precision acceleration.
- Priority: Medium — important for inference use-cases with quantized models.

**TSNE BH Kernel (cannylab) — No Unit Tests:**
- What's not tested: The Barnes-Hut tree construction kernels in `cpp/src/tsne/cannylab/bh.cu` have no isolated unit tests; they are only covered by end-to-end TSNE tests.
- Files: `cpp/tests/sg/tsne_test.cu`
- Risk: Kernel-level regressions (e.g., BH tree correctness) are invisible until result quality degrades.
- Priority: Medium.

**Dask Eager Path Removed:**
- What's not tested: `python/cuml/cuml/dask/common/base.py` (line 304) has `# TODO: Add eager path back in`. The Dask eager execution path was removed and not replaced.
- Files: `python/cuml/cuml/dask/common/base.py`
- Risk: Dask-based estimators may be slower than expected because all operations go through the Dask scheduler even for small inputs.
- Priority: Low.

---

## Porting Risks for Rust Reimplementation

**CUDA Kernel Complexity:**
- TSNE: Two entirely separate CUDA kernel implementations (exact O(n²) in `cpp/src/tsne/exact_kernels.cuh` and Barnes-Hut O(n log n) in `cpp/src/tsne/cannylab/bh.cu` + `cpp/src/tsne/barnes_hut_kernels.cuh`, `cpp/src/tsne/fft_tsne.cuh`). The BH implementation is 1,111 lines of raw CUDA with shared memory, warp-sync primitives, and atomic operations.
- SVM: The SMO solver (`cpp/src/svm/smosolver.cuh`, 507 lines) uses a custom CUDA working set selection kernel (`cpp/src/svm/smoblocksolve.cuh`) with shared memory tiling that is non-trivial to express in Rust CUDA bindings.
- UMAP: The optimize_batch kernel (`cpp/src/umap/simpl_set_embed/optimize_batch_kernel.cuh`, 1,165 lines) contains warp-level shuffle operations and shared memory collision handling.
- Recommendation: For a Rust port, target cuVS/RAFT via C FFI for ANN and distance kernels rather than reimplementing; focus custom CUDA work on algorithms absent from cuVS.

**Heavy Template Metaprogramming:**
- 471 template declarations across `.cuh`/`.hpp` files; 6 files use `enable_if`/`type_traits`/SFINAE explicitly. The primary dimension of template instantiation is `<float>` vs `<double>` on all algorithm entry points.
- Rust impact: Generics over `f32`/`f64` are straightforward, but SFINAE-based dispatch patterns (e.g., `dispatch_to_uint64_t`) must be reimplemented as Rust enum dispatch or const-generic branching.
- Files: `cpp/src/umap/umap.cuh`, `cpp/src/decisiontree/decisiontree.cuh`, `cpp/src/explainer/tree_shap.cu`

**CUDA Toolkit Library Dependencies:**
- cuBLAS: Used directly in `cpp/src/svm/svc_impl.cuh`, `cpp/src/svm/svr_impl.cuh` (raw `cublas_v2.h`), and indirectly via raft wrappers in ARIMA, Holtwinters, LARS, SVM SMO.
- cuFFT: Used in TSNE FFT implementation (`cpp/src/tsne/fft_tsne.cuh`).
- cuRAND: Used in UMAP embedding (`cpp/src/umap/simpl_set_embed/algo.cuh`) and SHAP explainer (`cpp/src/explainer/kernel_shap.cu`).
- Rust impact: All three must be called from Rust via `cuda-sys` or equivalent bindings. cuBLAS and cuFFT have community Rust bindings (`cublas-sys`, `cufft-sys`) but these are not well-maintained.

**RAPIDS Ecosystem Coupling (Python Layer):**
- The Python layer depends on cuDF, cupy, RMM, pylibraft, numba-cuda — all RAPIDS packages with synchronized version pins. A Rust reimplementation of the C++ layer would still need to expose a Python API compatible with these data structures.
- The `cuml.accel` metaclass proxy system is ~400 lines of Python metaclass machinery that would need to be reimplemented or abandoned in favor of a simpler dispatch layer.
- Files: `python/cuml/cuml/accel/estimator_proxy.py`, `python/cuml/cuml/internals/array.py`, `python/cuml/cuml/internals/input_utils.py`

**Warp-Size Assumptions:**
- 17 occurrences of warp-size-dependent logic (`warpSize`, `laneId`, warp shuffle) across CUDA kernel files. All assume 32-thread warps (NVIDIA-only). A port targeting future AMD ROCm/HIP would require warp-size portability guards.
- Files: Include `cpp/src/svm/smoblocksolve.cuh`, `cpp/src/tsne/barnes_hut_kernels.cuh`, `cpp/src/umap/simpl_set_embed/optimize_batch_kernel.cuh`.

**Multi-GPU Communication Stack:**
- MG algorithms require NCCL + UCX + MPI. The C++ MG communicator is in `cpp/src/` under the `SINGLEGPU=OFF` path and ties to `raft::comms::`. A Rust MG port would need to wrap NCCL directly or use an existing Rust NCCL binding.
- Files: `cpp/CMakeLists.txt` (lines 48, 52, 209–220), `cpp/src/glm/ols_mg.cu`, `cpp/src/pca/pca_mg.cu`

**nvForest Forest Inference ABI:**
- The new nvForest library (`cpp/cmake/thirdparty/get_nvforest.cmake`) is NVIDIA-internal and has no stable public C ABI documented in this repo. A Rust port wanting GPU forest inference must either use the C++ nvForest headers or implement its own FIL-equivalent from the treelite model format.

---

*Concerns audit: 2026-06-11*
