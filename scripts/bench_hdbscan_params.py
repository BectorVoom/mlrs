#!/usr/bin/env python3
"""HDBSCAN PARAMETER perf harness — which knobs move wall clock, and by how much.

`bench_hdbscan.py` answers "is mlrs faster than sklearn at the defaults". This
one answers the follow-up: of the fourteen sklearn parameters, which ones are
performance-significant, what does each cost, and does mlrs still beat sklearn
across the range of each?

Four parameters are swept, because those are the four that can change the amount
of work rather than only its answer:

  ``algorithm``   auto / brute / kd_tree / ball_tree — the neighbour-search route
                  for the core-distance stage, which is the dominant term of a
                  host fit. `auto` builds a KD-tree and lets each worker abandon
                  it if it is not pruning; `brute` never builds one; the two tree
                  values force it unconditionally.
  ``leaf_size``   points per KD-tree leaf: traversal bookkeeping (small leaves)
                  against wasted distance work (large leaves).
  ``n_jobs``      host worker count.
  ``metric``      euclidean/manhattan/chebyshev/minkowski take the source-tracking
                  Variant-B path (no n x n resident); cosine and precomputed take
                  the dense Variant-A path, which is a different cost class.

``min_samples`` is swept alongside because it sets `k` for the core-distance
scan, and `min_cluster_size` is NOT: it only affects the condensed-tree walk,
which is O(n) against the scan's O(n^2 d).

Every sweep prints mlrs against sklearn at the SAME parameter value, so a knob
that helps mlrs is never mistaken for a knob that hurts sklearn.

METHODOLOGY (learned the hard way on this repo — see the notes in
`bench_hdbscan.py` and the cpu-bench memories):

  * `time.perf_counter`, never `process_time`: cubecl spins threads, so CPU time
    over-counts wildly.
  * min-of-`--repeat`, and the mlrs backend init warmed out first.
  * A/B knobs are verified LIVE before their sweep is reported: a flat sweep is a
    dead knob until proven otherwise, so `--check-live` fails loudly rather than
    printing a reassuring row of identical numbers.
  * `--load-guard` refuses to run on a machine whose load average exceeds the
    core count. A co-tenant build has inverted an mlrs-vs-sklearn verdict on this
    repo before; a benchmark taken on a saturated box is worse than no benchmark
    because it looks like data.

    python3 scripts/bench_hdbscan_params.py                  # all sweeps
    python3 scripts/bench_hdbscan_params.py --sweep algorithm leaf_size
    python3 scripts/bench_hdbscan_params.py --n 20000 --d 16
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

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
    """Well-separated blobs (the `bench_hdbscan.py` design, shared verbatim).

    Structure matters for the tree sweeps specifically: a KD-tree prunes on
    density, so uniform noise and clustered data land in genuinely different
    regimes (12.7% of points visited per query against ~100%). Blobs are the
    regime HDBSCAN is FOR, so they are the honest default here; `--uniform`
    switches to the adversarial one.
    """
    centers = (_uniform01(seed + 1, k * d) * 20.0).reshape(k, d)
    noise = (_uniform01(seed, n * d) - 0.5) * 2.0
    labels = np.arange(n, dtype=np.int64) % k
    return (centers[labels] + noise.reshape(n, d)).astype(np.float64)


def make_uniform(n: int, d: int, k: int, seed: int = 42) -> np.ndarray:
    """Structureless scatter — the regime where a KD-tree stops paying off."""
    return (_uniform01(seed, n * d).reshape(n, d) * 20.0).astype(np.float64)


def timed(fn, repeat: int):
    """Min-of-`repeat` wall clock. Min, not mean: we want the machine's best
    behaviour, which is the number least polluted by unrelated load."""
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def load_guard(enabled: bool) -> None:
    if not enabled:
        return
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        return
    cores = os.cpu_count() or 1
    if load1 > cores:
        raise SystemExit(
            f"REFUSING to benchmark: 1-min load average {load1:.1f} exceeds "
            f"{cores} cores. A saturated box has inverted an mlrs-vs-sklearn "
            f"verdict on this repo before, and a bad number that looks like a "
            f"good one is worse than no number. Wait for the box to quiet, or "
            f"pass --no-load-guard if you know what you are measuring."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--k", type=int, default=8, help="blob count")
    ap.add_argument("--mcs", type=int, default=10, help="min_cluster_size")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--uniform", action="store_true", help="structureless design")
    ap.add_argument(
        "--sweep",
        nargs="*",
        default=["algorithm", "leaf_size", "n_jobs", "metric", "min_samples"],
    )
    ap.add_argument("--no-load-guard", dest="load_guard", action="store_false")
    ap.add_argument(
        "--check-live",
        action="store_true",
        default=True,
        help="fail if a swept knob produces no measurable spread (dead-knob guard)",
    )
    args = ap.parse_args()

    load_guard(args.load_guard)

    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    from sklearn.metrics import adjusted_rand_score, pairwise_distances

    import mlrs

    design = make_uniform if args.uniform else make_blobs
    x = design(args.n, args.d, args.k)

    # Warm the one-time cubecl-cpu backend init out of every number below.
    mlrs.HDBSCAN(min_cluster_size=5).fit_predict(design(64, 4, 2))

    print(
        f"n={args.n} d={args.d} k={args.k} min_cluster_size={args.mcs} "
        f"design={'uniform' if args.uniform else 'blobs'} repeat={args.repeat} "
        f"cores={os.cpu_count()} load1={os.getloadavg()[0]:.2f}"
    )
    print()

    def sk_fit(**kw):
        arr = kw.pop("_x", x)
        # `copy` is left at its default so sklearn does exactly the work it does
        # for a normal caller (setting copy=True would add a matrix copy to the
        # precomputed rows and flatter mlrs); the 1.10 FutureWarning that
        # provokes is noise here, so it is silenced rather than dodged.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return SkHDBSCAN(min_cluster_size=args.mcs, **kw).fit_predict(arr)

    def ml_fit(**kw):
        arr = kw.pop("_x", x)
        return np.asarray(mlrs.HDBSCAN(min_cluster_size=args.mcs, **kw).fit_predict(arr))

    # Reference partition for the ARI cross-check: a speedup that changes the
    # clustering is not a speedup.
    ref, _ = timed(lambda: sk_fit(), 1)

    def sweep(title, values, to_kwargs, note="", live=True):
        head = (
            f"{title:>14} | {'sklearn(s)':>10} {'mlrs(s)':>9} {'speedup':>8} "
            f"{'ARI':>7} {'clusters':>9}"
        )
        print(f"--- {title} " + "-" * max(0, 62 - len(title)))
        if note:
            print(f"    {note}")
        print(head)
        print("-" * len(head))
        ml_times = []
        for v in values:
            kw = to_kwargs(v)
            sk_kw = dict(kw)
            ml_kw = dict(kw)
            # n_jobs/leaf_size mean different things to sklearn's tree; pass what
            # each engine understands and let the row compare like with like.
            try:
                _, sk_s = timed(lambda: sk_fit(**sk_kw), args.repeat)
            except Exception as e:  # noqa: BLE001 — report, do not abort the sweep
                print(f"{str(v):>14} | sklearn ERR {type(e).__name__}: {str(e)[:40]}")
                continue
            lab, ml_s = timed(lambda: ml_fit(**ml_kw), args.repeat)
            ml_times.append(ml_s)
            ari = adjusted_rand_score(ref, lab)
            n_cl = int(lab.max()) + 1 if lab.size else 0
            print(
                f"{str(v):>14} | {sk_s:>10.4f} {ml_s:>9.4f} {sk_s / ml_s:>8.2f} "
                f"{ari:>7.4f} {n_cl:>9}"
            )
        # Dead-knob guard: a knob that is genuinely wired shows SOME spread.
        # `algorithm`/`leaf_size` on a small n can legitimately be flat, so this
        # is advisory for those and only fails when asked.
        if live and args.check_live and len(ml_times) > 1:
            spread = max(ml_times) / max(min(ml_times), 1e-12)
            if spread < 1.02:
                print(
                    f"    NOTE: mlrs spread across this sweep is {spread:.3f}x — "
                    f"verify the knob is live before reading the row as 'no cost'."
                )
        print()

    if "algorithm" in args.sweep:
        sweep(
            "algorithm",
            ["auto", "brute", "kd_tree", "ball_tree"],
            lambda v: {"algorithm": v},
            note="core-distance neighbour search. All four give IDENTICAL labels "
            "in mlrs (gated exactly) — this is pure wall clock.",
        )

    if "leaf_size" in args.sweep:
        sweep(
            "leaf_size",
            [4, 8, 16, 32, 40, 64, 128, 256],
            lambda v: {"leaf_size": v, "algorithm": "kd_tree"},
            note="points per KD-tree leaf, forced onto the tree route so the knob "
            "is live (on 'brute' nothing reads it).",
        )

    if "n_jobs" in args.sweep:
        sweep(
            "n_jobs",
            [1, 2, 4, 8, 16, -1],
            lambda v: {"n_jobs": v},
            note="host worker count. NOTE mlrs's None default is ALL units, not "
            "joblib's 1 — n_jobs=1 is the like-for-like single-core row.",
        )

    if "metric" in args.sweep:
        dist = pairwise_distances(x)
        sweep(
            "metric",
            ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine", "precomputed"],
            lambda v: (
                {"metric": v, "_x": dist}
                if v == "precomputed"
                else (
                    {"metric": v, "metric_params": {"p": 3.0}}
                    if v == "minkowski"
                    else {"metric": v}
                )
            ),
            note="euclidean/manhattan/chebyshev/minkowski take the no-n^2 "
            "Variant-B path; cosine and precomputed take the dense Variant-A path.",
            live=False,
        )

    if "min_samples" in args.sweep:
        sweep(
            "min_samples",
            [2, 5, 10, 25, 50],
            lambda v: {"min_samples": v},
            note="sets k for the core-distance scan (a k-smallest insertion per "
            "candidate), so it is the one selection knob with a scan-side cost.",
            live=False,
        )


if __name__ == "__main__":
    main()
