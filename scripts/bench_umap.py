#!/usr/bin/env python3
"""UMAP `fit_transform` wall-clock harness — mlrs (cpu backend) vs umap-learn.

sklearn ships no UMAP, so the reference engine is `umap-learn` 0.5.x — the same
library the committed oracle fixtures under `tests/fixtures/umap_*.npz` were
generated with, and the CPU reference CLAUDE.md names for this estimator.

Both engines run the SAME splitmix64 blob ladder with matched hyperparameters
(`n_neighbors`, `n_components`, `min_dist`, `n_epochs`, `metric`,
`random_state`), so the number compares the whole fit path directly: ingress ->
kNN graph -> smooth-kNN rho/sigma -> membership -> t-conorm union -> a/b fit ->
init -> SGD layout -> embedding egress.

Both engines pay a large ONE-TIME cost on the first call in a process — mlrs the
cubecl-cpu client construction + kernel JIT, umap-learn its numba JIT of the
layout/nn-descent kernels. Neither is part of the algorithm, so the harness warms
both before timing and reports them separately (`--show-cold`).

    python3 scripts/bench_umap.py                 # default ladder
    python3 scripts/bench_umap.py --n 2000 5000   # custom row ladder

Embedding QUALITY is printed alongside the timings as the correctness
cross-check — a speedup that destroys the manifold is not a speedup. Both
columns (`trust`, `overlap`) are measured for EACH engine against the ORIGINAL
data, which is the same pair of structural quantities `umap_test`'s own property
gate uses. They are never compared embedding-to-embedding: UMAP is stochastic and
the two engines do not share an RNG stream, so their coordinates differ by
construction. Read the gate as RELATIVE — mlrs's numbers against umap-learn's on
the same row — never as an absolute floor.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

# (n_rows, n_features, n_neighbors, n_blobs)
CONFIGS = [
    (1_000, 8, 15, 6),
    (2_000, 8, 15, 6),
    (5_000, 16, 15, 8),
    (10_000, 16, 30, 8),
]

N_EPOCHS = 200


def _splitmix64_block(seed: int, count: int) -> np.ndarray:
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        state = (np.uint64(seed) + idx * np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        z = state
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
        return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def _uniform01(seed: int, count: int) -> np.ndarray:
    return (_splitmix64_block(seed, count) >> np.uint64(11)) / float(1 << 53)


def make_blobs(n: int, d: int, k: int, seed: int = 42) -> np.ndarray:
    """Well-separated blobs so the embedding has real manifold structure."""
    centers = (_uniform01(seed + 1, k * d) * 20.0).reshape(k, d)
    noise = (_uniform01(seed, n * d) - 0.5) * 2.0
    labels = np.arange(n, dtype=np.int64) % k
    return (centers[labels] + noise.reshape(n, d)).astype(np.float64)


def knn_indices(x: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's `k` nearest neighbours (self dropped)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    return nn.kneighbors(x, return_distance=False)[:, 1:]


def neighbor_overlap(x: np.ndarray, emb: np.ndarray, k: int) -> float:
    """Mean fraction of each row's high-dimensional kNN that survive into `emb`.

    This is the repo's own structural gate (`umap_test::knn_overlap`), measured
    against the ORIGINAL data — not embedding-vs-embedding. Two independent UMAP
    runs never agree coordinate-wise (different RNG streams by construction), so
    comparing the two embeddings to each other would score a correct engine as
    wrong; how much of the input neighbourhood each embedding preserves is the
    quantity that actually says whether the manifold survived.
    """
    ix, ie = knn_indices(x, k), knn_indices(emb, k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, ie)]))


def trustworthiness(x: np.ndarray, emb: np.ndarray, k: int) -> float:
    from sklearn.manifold import trustworthiness as tw

    return float(tw(x, emb, n_neighbors=k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=None, help="row ladder override")
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-quality", action="store_true", help="skip the quality columns")
    ap.add_argument(
        "--show-cold",
        action="store_true",
        help="report the one-time engine warm-up costs and exit",
    )
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    import umap as umap_learn

    import mlrs

    configs = CONFIGS
    if args.n:
        configs = [(n, 16, 15, 8) for n in args.n]

    # One-time warm-up for BOTH engines on a trivial problem, so neither the
    # cubecl-cpu client/JIT nor numba's JIT lands inside a timed cell.
    warm = make_blobs(128, 4, 2)

    def _cold_warm(fn):
        t0 = time.perf_counter()
        fn()
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        fn()
        return cold, time.perf_counter() - t0

    ml_cold, ml_warm = _cold_warm(
        lambda: mlrs.UMAP(n_neighbors=5, n_epochs=5, random_state=42).fit_transform(warm)
    )
    ul_cold, ul_warm = _cold_warm(
        lambda: umap_learn.UMAP(n_neighbors=5, n_epochs=5, random_state=42).fit_transform(warm)
    )
    print(
        f"mlrs one-time backend init: {ml_cold - ml_warm:.4f} s "
        f"(first call {ml_cold:.4f} s, warm {ml_warm:.4f} s)"
    )
    print(
        f"umap-learn one-time numba JIT: {ul_cold - ul_warm:.4f} s "
        f"(first call {ul_cold:.4f} s, warm {ul_warm:.4f} s)"
    )
    print("both are process-wide warm-ups, excluded below")
    if args.show_cold:
        return

    header = (
        f"{'n':>7} {'d':>4} {'nn':>4} | {'engine':>10} {'fit(s)':>10} "
        f"{'speedup':>8} {'trust':>7} {'overlap':>8}"
    )
    print(f"metric={args.metric}  n_epochs={args.epochs}  repeat={args.repeat}")
    print(header)
    print("-" * len(header))

    for n, d, nn, k in configs:
        x = make_blobs(n, d, k)

        def timed(fn):
            best = float("inf")
            out = None
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                out = fn()
                best = min(best, time.perf_counter() - t0)
            return out, best

        ul_emb, ul_s = timed(
            lambda: umap_learn.UMAP(
                n_neighbors=nn,
                n_components=2,
                min_dist=0.1,
                metric=args.metric,
                n_epochs=args.epochs,
                random_state=42,
            ).fit_transform(x)
        )
        ul_emb = np.asarray(ul_emb, dtype=np.float64)
        ul_tw = 1.0 if args.no_quality else trustworthiness(x, ul_emb, nn)
        ul_ov = 1.0 if args.no_quality else neighbor_overlap(x, ul_emb, nn)
        print(
            f"{n:>7} {d:>4} {nn:>4} | {'umap-learn':>10} {ul_s:>10.4f} "
            f"{1.0:>8.2f} {ul_tw:>7.4f} {ul_ov:>8.4f}"
        )

        ml_emb, ml_s = timed(
            lambda: mlrs.UMAP(
                n_neighbors=nn,
                n_components=2,
                min_dist=0.1,
                metric=args.metric,
                n_epochs=args.epochs,
                random_state=42,
            ).fit_transform(x)
        )
        ml_emb = np.asarray(ml_emb, dtype=np.float64)
        ml_tw = 1.0 if args.no_quality else trustworthiness(x, ml_emb, nn)
        ml_ov = 1.0 if args.no_quality else neighbor_overlap(x, ml_emb, nn)
        print(
            f"{n:>7} {d:>4} {nn:>4} | {'mlrs':>10} {ml_s:>10.4f} "
            f"{ul_s / ml_s:>8.2f} {ml_tw:>7.4f} {ml_ov:>8.4f}"
        )


if __name__ == "__main__":
    main()
