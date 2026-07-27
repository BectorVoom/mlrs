#!/usr/bin/env python3
"""LinearRegression **predict** wall-clock: mlrs (cpu backend) vs scikit-learn.

The LINEAR-PRED-CPU probe. `scripts/bench_linear.py` walks the whole fit+predict
ladder; this one isolates `predict` on the cpu backend, where the cost model is
completely different from a GPU's — there the operand has to cross a bus anyway,
here a `DeviceArray` upload is a pure `memcpy` that costs more than the whole
prediction (see `crates/mlrs-backend/src/prims/linear_predict.rs`).

Both engines are handed the SAME splitmix64 design matrix (`make_regression` is
imported from `bench_linear.py`, byte-identical to
`crates/mlrs-algos/tests/linear_regression_perf_test.rs`) and timed through their
own public Python API, so each pays its own validation and ingestion.

    .venv/bin/python scripts/bench_linear_predict_cpu.py \
        [--reps 7] [--cold] [--estimator Ridge] [--engine mlrs]

Every config is fitted on a SMALL subset (`--fit-rows`, default 400): `fit` on
the cpu backend runs the Gram+eig path through the cubecl-cpu JIT and takes
minutes, and the fitted coefficients have no effect whatsoever on how long
`predict` takes. Pair this with the prim-level probe when attributing a change:

    cargo test -p mlrs-backend --release --features cpu \
      --test linear_predict_perf_test -- --ignored --nocapture
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
    ap.add_argument("--fit-rows", type=int, default=400)
    ap.add_argument(
        "--estimator",
        default="LinearRegression",
        choices=["LinearRegression", "Ridge", "Lasso", "ElasticNet"],
        help="which dense linear regressor to time (all four share one predict path)",
    )
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument(
        "--engine",
        default="both",
        choices=["both", "mlrs", "sklearn"],
        help=(
            "Time only one engine. Both engines run their own thread pool, and "
            "OpenBLAS keeps its workers SPINNING after a call, so interleaving "
            "them in one process taxes whichever runs second — at the rungs "
            "where the whole predict is a few hundred microseconds that is "
            "most of the measurement. Run the two engines in separate "
            "processes when a rung's numbers look implausible."
        ),
    )
    args = ap.parse_args()

    import mlrs
    import sklearn.linear_model as sklm

    # Hyperparameters are irrelevant to predict cost (it is one matvec over the
    # fitted coefficients), so both engines take their defaults; the penalized
    # models only differ in how `fit` produced those coefficients.
    SkLR = getattr(sklm, args.estimator)
    MlLR = getattr(mlrs, args.estimator)

    dt = np.dtype(args.dtype)
    both = args.engine == "both"
    header = f"{'m':>9} {'n':>4} | {'mlrs (s)':>10} {'sklearn (s)':>11}"
    if both:
        header += f" {'speedup':>8}"
    if args.cold:
        header += f" | {'cold (s)':>9}"
    print(header)
    print("-" * len(header))

    for m, n in CONFIGS:
        x, y = make_regression(m, n)
        x = np.ascontiguousarray(x.astype(dt, copy=False))
        y = y.astype(dt, copy=False)
        f = args.fit_rows
        sk = SkLR().fit(x[:f], y[:f])
        ml = MlLR().fit(x[:f], y[:f]) if args.engine != "sklearn" else None

        rel = float("nan")
        if both:
            # Agreement gate — a fast wrong answer is not a win.
            a, b = ml.predict(x), sk.predict(x)
            rel = float(np.max(np.abs(a - b))) / max(1e-12, float(np.max(np.abs(b))))

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
        print(line + (f"   (rel err {rel:.1e})" if both else ""), flush=True)


if __name__ == "__main__":
    main()
