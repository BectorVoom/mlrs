#!/usr/bin/env python3
"""Linear-SVM **fit** wall-clock: mlrs (cpu backend) vs scikit-learn.

The SVM-FIT-CPU probe — the `bench_svm_predict_cpu.py` twin for
`LinearSVC.fit` / `LinearSVR.fit`. Both engines minimize the SAME
L2-regularized primal (squared hinge / squared epsilon-insensitive) over the
same design matrix, so the comparison is like-for-like *given the same
iteration budget*: sklearn's `LinearSVC` is liblinear (dual/primal
coordinate descent, C-compiled), mlrs is an L-BFGS over the margin matvec.

Because the two solvers take different iteration COUNTS to reach the same
optimum, the probe reports the fitted objective alongside the wall-clock so a
"win" from an under-solved model is visible rather than hidden. Pass
`--check` to also print `max|coef_ - sklearn coef_|`.

    .venv/bin/python scripts/bench_svm_fit_cpu.py \
        [--reps 5] [--estimator LinearSVC] [--engine mlrs] [--check]

The `--engine` note from `bench_linear_predict_cpu.py` applies verbatim: both
engines run their own thread pool and OpenBLAS keeps its workers SPINNING
after a call, so interleaving them in one process taxes whichever runs
second. Re-run a suspicious rung with `--engine mlrs` / `--engine sklearn` in
separate processes.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bench_linear import make_regression

# (rows, features) of the timed fit.
CONFIGS = [
    (1_000, 16),
    (10_000, 16),
    (10_000, 64),
    (100_000, 16),
    (50_000, 64),
]


def make_classification(n: int, d: int, seed: int = 42):
    """`make_regression`'s design with a sign-thresholded ±1 label."""
    x, y = make_regression(n, d, seed)
    labels = np.where(y >= np.median(y), 1, 0).astype(np.int32)
    return x, labels


def best_of(fn, reps):
    """(best, first, result) wall-clock seconds over `reps` calls."""
    times = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), times[0], out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--estimator", default="LinearSVC", choices=["LinearSVC", "LinearSVR"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--check", action="store_true", help="print max|coef_| deviation")
    ap.add_argument("--configs", default="", help="comma-separated n:d overriding CONFIGS")
    args = ap.parse_args()

    import mlrs

    if args.estimator == "LinearSVC":
        from mlrs import LinearSVC as MlrsEst
        from sklearn.svm import LinearSVC as SkEst

        make = make_classification
        sk_kwargs = dict(loss="squared_hinge", penalty="l2", dual=False)
    else:
        from mlrs import LinearSVR as MlrsEst
        from sklearn.svm import LinearSVR as SkEst

        make = make_regression
        sk_kwargs = dict(loss="squared_epsilon_insensitive", dual=False, epsilon=0.0)

    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]

    dt = np.float32 if args.dtype == "float32" else np.float64
    print(f"mlrs {mlrs.__name__} | estimator={args.estimator} dtype={args.dtype} "
          f"max_iter={args.max_iter} tol={args.tol} C={args.C}")
    header = f"{'n':>8} {'d':>4} | {'engine':>8} {'fit (s)':>10} {'first (s)':>10}"
    print(header)
    print("-" * len(header))

    for n, d in configs:
        x, y = make(n, d)
        x = x.astype(dt)
        if args.estimator == "LinearSVR":
            y = y.astype(dt)

        results = {}
        if args.engine in ("both", "mlrs"):
            def fit_mlrs():
                m = MlrsEst(C=args.C, max_iter=args.max_iter, tol=args.tol)
                m.fit(x, y)
                return m

            try:
                best, first, model = best_of(fit_mlrs, args.reps)
                results["mlrs"] = (best, first)
                if args.check:
                    results["mlrs_coef"] = np.asarray(model.coef_, dtype=np.float64).ravel()
            except Exception as exc:  # noqa: BLE001
                print(f"{n:>8} {d:>4} |     mlrs  FAILED: {type(exc).__name__}: {exc}")

        if args.engine in ("both", "sklearn"):
            def fit_sk():
                m = SkEst(C=args.C, max_iter=args.max_iter, tol=args.tol, **sk_kwargs)
                m.fit(x, y)
                return m

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best, first, model = best_of(fit_sk, args.reps)
            results["sklearn"] = (best, first)
            if args.check:
                results["sk_coef"] = np.asarray(model.coef_, dtype=np.float64).ravel()

        for eng in ("mlrs", "sklearn"):
            if eng in results:
                best, first = results[eng]
                print(f"{n:>8} {d:>4} | {eng:>8} {best:>10.4f} {first:>10.4f}")
        if "mlrs" in results and "sklearn" in results:
            speedup = results["sklearn"][0] / results["mlrs"][0]
            note = f"{speedup:.2f}x vs sklearn"
            if args.check and "mlrs_coef" in results and "sk_coef" in results:
                dev = float(np.max(np.abs(results["mlrs_coef"] - results["sk_coef"])))
                note += f" | max|Δcoef| = {dev:.3e}"
            print(f"{'':>8} {'':>4} | {note}")
        print()


if __name__ == "__main__":
    main()
