#!/usr/bin/env python3
"""Linear-SVM **predict** wall-clock: mlrs (cpu backend) vs scikit-learn.

The SVM-PRED-CPU probe — the `bench_linear_predict_cpu.py` twin for
`LinearSVR.predict` (float margins) and `LinearSVC.predict` (int32 labels).
Same cost model: on the cpu backend a `DeviceArray` upload is a pure `memcpy`
that costs more than the whole prediction, so what is being timed here is
almost entirely ingress/egress, not arithmetic.

Both engines are handed the SAME splitmix64 design matrix (`make_regression`
from `bench_linear.py`) and timed through their own public Python API, so each
pays its own validation and ingestion.

    .venv/bin/python scripts/bench_svm_predict_cpu.py \
        [--reps 7] [--cold] [--estimator LinearSVC] [--engine mlrs]

Every config is fitted on a SMALL subset (`--fit-rows`, default 300): the
linear-SVM `fit` is a host-orchestrated L-BFGS over a device GEMM and takes
minutes at scale on the cpu backend, and the fitted coefficients have no effect
whatsoever on how long `predict` takes.

The `--engine` note from `bench_linear_predict_cpu.py` applies verbatim: both
engines run their own thread pool and OpenBLAS keeps its workers SPINNING after
a call, so interleaving them in one process taxes whichever runs second. Re-run
a suspicious rung with `--engine mlrs` / `--engine sklearn` in separate
processes.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bench_linear import make_regression

# (rows, features) of the timed predict batch.
CONFIGS = [
    (10_000, 16),
    (100_000, 16),
    (100_000, 64),
    (1_000_000, 16),
    (200_000, 64),
]


def best_of(fn, reps):
    """(best, first) wall-clock seconds over `reps` calls."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), times[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--cold", action="store_true", help="also print first-call time")
    ap.add_argument("--fit-rows", type=int, default=300)
    ap.add_argument(
        "--estimator",
        default="LinearSVR",
        choices=["LinearSVR", "LinearSVC"],
        help="which linear SVM to time",
    )
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument(
        "--max-iter", type=int, default=2000, help="fit budget (predict cost is unaffected)"
    )
    # The default tol=1e-4 does not reach mlrs' f32 convergence gate on this
    # tiny fit subset; predict cost is identical either way, so the probe asks
    # for a shallower solve rather than a longer one.
    ap.add_argument("--tol", type=float, default=1e-2)
    args = ap.parse_args()

    import mlrs
    import sklearn.svm as sksvm

    Sk = getattr(sksvm, args.estimator)
    Ml = getattr(mlrs, args.estimator)
    classify = args.estimator == "LinearSVC"

    dt = np.dtype(args.dtype)
    both = args.engine == "both"
    header = f"{'m':>9} {'n':>4} | {'mlrs (s)':>10} {'sklearn (s)':>11}"
    if both:
        header += f" {'speedup':>8}"
    if args.cold:
        header += f" | {'cold (s)':>9}"
    print(f"{args.estimator} predict, {args.dtype}, cpu backend")
    print(header)
    print("-" * len(header))

    for m, n in CONFIGS:
        x, y = make_regression(m, n)
        x = np.ascontiguousarray(x.astype(dt, copy=False))
        # LinearSVC needs a label vector; sign of the regression target is a
        # perfectly ordinary linearly-separable-ish binary problem.
        yt = (y > np.median(y)).astype(np.int32) if classify else y.astype(dt, copy=False)

        f = args.fit_rows
        sk = Sk(max_iter=args.max_iter, tol=args.tol).fit(x[:f], yt[:f])
        ml = (
            Ml(max_iter=args.max_iter, tol=args.tol).fit(x[:f], yt[:f])
            if args.engine != "sklearn"
            else None
        )

        agree = float("nan")
        if both:
            # Agreement gate — a fast wrong answer is not a win. Labels must
            # match exactly; margins within a relative band.
            a, b = ml.predict(x), sk.predict(x)
            if classify:
                agree = float(np.mean(a == b))
            else:
                agree = float(np.max(np.abs(a - b))) / max(
                    1e-12, float(np.max(np.abs(b)))
                )

        ml_s, ml_cold = (
            best_of(lambda: ml.predict(x), args.reps)
            if args.engine != "sklearn"
            else (float("nan"), float("nan"))
        )
        sk_s = (
            best_of(lambda: sk.predict(x), args.reps)[0]
            if args.engine != "mlrs"
            else float("nan")
        )

        line = f"{m:>9} {n:>4} | {ml_s:>10.4f} {sk_s:>11.4f}"
        if both:
            line += f" {sk_s / ml_s:>7.2f}x"
        if args.cold:
            line += f" | {ml_cold:>9.4f}"
        if both:
            line += (
                f"   (label agree {agree:.4f})" if classify else f"   (rel err {agree:.1e})"
            )
        print(line, flush=True)


if __name__ == "__main__":
    main()
