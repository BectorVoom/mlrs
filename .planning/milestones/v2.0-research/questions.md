# Research Questions

Open questions surfaced during exploration that need a deeper investigation pass before
planning. Resolve before the phase that depends on them.

## v2 milestone (breadth sweep) — see [[../seeds/v2-breadth-roadmap]]

- **[v2-P1] Incremental SVD in CubeCL.** Can IncrementalPCA's batched/streaming SVD update
  (sklearn's incremental_pca merges the running SVD with each new batch) be expressed on
  the existing Jacobi SVD primitive, or does it need a dedicated update kernel? What's the
  numerical-stability story for the merge step under f32 on rocm?

- **[v2-P1] RNG-matrix generator on device.** v1 used a host SplitMix64 PRNG read back per
  draw (k-means++). RandomProjection needs a full (n_features × n_components) Gaussian /
  sparse matrix — is host-generate-then-upload acceptable, or is a device RNG kernel worth
  it? (Note: ASVS V6 — no OsRng; reproducible seeded PRNG required.)

- **[v2-P2] Kernel-matrix primitive design.** One prim covering linear/RBF/poly/sigmoid
  kernels over pairwise distance, reused by KernelRidge + KernelDensity (+ future kernel
  SVM). Confirm it composes from the v1 distance prim without new SharedMemory/atomics
  (cpu-MLIR constraint).

- **[v2-P4] SGD solver under cpu-MLIR constraints.** A minibatch SGD solver (hinge / log /
  squared / ε-insensitive losses) for MBSGD* + LinearSVC/SVR. Does the update loop fit the
  no-SharedMemory / no-cross-unit-atomics cpu-MLIR pattern (cf. v1 GATHER-kernel idiom)?
  How is sklearn's SGD parity defined (learning-rate schedule, averaging, tol/n_iter_no_change)?

- **[v2-P3] Graph Laplacian + smallest-eigenpairs.** SpectralEmbedding/SpectralClustering
  need the *smallest* nontrivial eigenvectors of the Laplacian; v1's Jacobi eig returns the
  full spectrum (descending). Is full-spectrum-then-take-smallest acceptable at v2 problem
  sizes, or is a shift-invert / Lanczos path needed?
