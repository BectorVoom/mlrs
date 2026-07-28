#!/usr/bin/env python3
"""HDBSCAN `fit_predict` wall-clock harness — mlrs (cpu backend) vs scikit-learn.

Both engines run the SAME splitmix64 blob ladder with matched hyperparameters
(`min_cluster_size`, `min_samples`, `metric`), so the number compares the whole
prediction path directly: ingress -> core distances -> mutual-reachability MST ->
single linkage -> condensed tree -> EoM selection -> label egress.

`fit_predict` is HDBSCAN's ONLY prediction surface (neither sklearn's HDBSCAN nor
mlrs has a standalone `predict` — the labels come out of the fit), so it is the
call this harness times.

mlrs pays a ONE-TIME backend init on the first call in a process (cubecl-cpu
client construction plus its first device read-back). It is not part of the
algorithm and is shared with every other mlrs estimator, so the harness warms it
before timing and reports it separately — see `--show-cold`.

    python3 scripts/bench_hdbscan.py                 # default ladder
    python3 scripts/bench_hdbscan.py --n 4000 8000   # custom row ladder

Agreement (ARI vs sklearn labels) is printed alongside the timings as the
correctness cross-check — a speedup that changes the clustering is not a
speedup.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# (n_rows, n_features, min_cluster_size, n_blobs)
CONFIGS = [
    (2_000, 8, 5, 6),
    (5_000, 8, 5, 6),
    (10_000, 16, 10, 8),
    (20_000, 16, 25, 8),
]

MASK = (1 << 64) - 1


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
    """Well-separated blobs so the condensed tree has real cluster structure."""
    centers = (_uniform01(seed + 1, k * d) * 20.0).reshape(k, d)
    noise = (_uniform01(seed, n * d) - 0.5) * 2.0
    labels = np.arange(n, dtype=np.int64) % k
    return (centers[labels] + noise.reshape(n, d)).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=None, help="row ladder override")
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument(
        "--show-cold",
        action="store_true",
        help="report the one-time mlrs backend init cost and exit",
    )
    args = ap.parse_args()

    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    from sklearn.metrics import adjusted_rand_score

    import mlrs

    configs = CONFIGS
    if args.n:
        configs = [(n, 16, 10, 8) for n in args.n]

    # One-time backend init (cubecl-cpu client + first read-back), measured on a
    # trivial fit so none of the cost is HDBSCAN's own work.
    warm = make_blobs(64, 4, 2)
    t0 = time.perf_counter()
    mlrs.HDBSCAN(min_cluster_size=5, min_samples=5).fit_predict(warm)
    cold_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    mlrs.HDBSCAN(min_cluster_size=5, min_samples=5).fit_predict(warm)
    warm_s = time.perf_counter() - t0
    print(f"mlrs one-time backend init: {cold_s - warm_s:.4f} s "
          f"(first call {cold_s:.4f} s, warm {warm_s:.4f} s) — "
          "process-wide, shared by every mlrs estimator, excluded below")
    if args.show_cold:
        return

    header = (
        f"{'n':>7} {'d':>4} {'mcs':>5} | {'engine':>8} {'fit_pred(s)':>11} "
        f"{'speedup':>8} {'clusters':>9} {'noise%':>7} {'ARI':>7}"
    )
    print(f"metric={args.metric}  repeat={args.repeat}")
    print(header)
    print("-" * len(header))

    for n, d, mcs, k in configs:
        x = make_blobs(n, d, k)

        def timed(fn):
            best = float("inf")
            out = None
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                out = fn()
                best = min(best, time.perf_counter() - t0)
            return out, best

        sk_lab, sk_s = timed(
            lambda: SkHDBSCAN(min_cluster_size=mcs, metric=args.metric).fit_predict(x)
        )
        n_cl = int(sk_lab.max()) + 1 if sk_lab.size else 0
        noise = 100.0 * float((sk_lab == -1).mean())
        print(
            f"{n:>7} {d:>4} {mcs:>5} | {'sklearn':>8} {sk_s:>10.4f} "
            f"{1.0:>8.2f} {n_cl:>9} {noise:>7.2f} {1.0:>7.4f}"
        )

        ml_lab, ml_s = timed(
            lambda: mlrs.HDBSCAN(min_cluster_size=mcs, metric=args.metric).fit_predict(x)
        )
        ml_lab = np.asarray(ml_lab)
        n_cl = int(ml_lab.max()) + 1 if ml_lab.size else 0
        noise = 100.0 * float((ml_lab == -1).mean())
        ari = adjusted_rand_score(sk_lab, ml_lab)
        print(
            f"{n:>7} {d:>4} {mcs:>5} | {'mlrs':>8} {ml_s:>10.4f} "
            f"{sk_s / ml_s:>8.2f} {n_cl:>9} {noise:>7.2f} {ari:>7.4f}"
        )


if __name__ == "__main__":
    main()
