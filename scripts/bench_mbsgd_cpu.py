#!/usr/bin/env python3
"""MBSGD **fit** wall-clock: mlrs (cpu backend) vs scikit-learn ``SGDClassifier``.

The MBSGD-FIT-CPU probe. Both engines run the SAME per-sample SGD recurrence
(sklearn ``_plain_sgd``; mlrs ``sgd_solve`` at ``batch_size=1``), so the
comparison is like-for-like *only when the iteration budget is pinned on both
sides* — hence the defaults here:

    tol = 0 / None   both engines run exactly ``max_iter`` epochs
    shuffle = False  mlrs' prim consumes rows in natural order (no permutation)

Run with ``--defaults`` to instead time each engine at ITS OWN library defaults
(mlrs stops on max|Δcoef|, sklearn on the loss plateau) — a user-facing number
where the epoch counts legitimately differ.

    .venv/bin/python scripts/bench_mbsgd_cpu.py [--reps 3] [--check]
                     [--engine mlrs|sklearn|both] [--loss hinge|log]

The ``--engine`` caveat from ``bench_linear_predict_cpu.py`` applies verbatim:
OpenBLAS keeps its workers SPINNING after a call, so interleaving both engines
in one process taxes whichever runs second. Re-run a suspicious rung with
``--engine mlrs`` / ``--engine sklearn`` in separate processes.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bench_linear import make_regression

# (rows, features, max_iter) of the timed fit. `max_iter` shrinks as `n·d`
# grows so a single rung stays inside a few seconds at the pinned budget.
CONFIGS = [
    (1_000, 16, 50),
    (10_000, 16, 20),
    (10_000, 64, 20),
    (50_000, 16, 10),
    (50_000, 64, 5),
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
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument("--loss", default="hinge", choices=["hinge", "log", "squared_hinge"])
    ap.add_argument("--penalty", default="l2", choices=["l2", "l1", "elasticnet"])
    ap.add_argument("--learning-rate", default="optimal",
                    choices=["optimal", "constant", "invscaling"])
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--eta0", type=float, default=0.01)
    ap.add_argument("--max-iter", type=int, default=0,
                    help="override the per-config epoch budget")
    ap.add_argument("--defaults", action="store_true",
                    help="time each engine at its own library defaults (tol/shuffle on)")
    ap.add_argument("--check", action="store_true", help="print max|Δcoef| vs sklearn")
    ap.add_argument("--configs", default="", help="comma-separated n:d:max_iter")
    args = ap.parse_args()

    import mlrs
    from mlrs import MBSGDClassifier as MlrsEst
    from sklearn.linear_model import SGDClassifier as SkEst

    # mlrs names the logistic loss "log"; sklearn renamed it "log_loss".
    sk_loss = {"hinge": "hinge", "log": "log_loss", "squared_hinge": "squared_hinge"}[args.loss]

    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]

    dt = np.float32 if args.dtype == "float32" else np.float64
    mode = "library defaults" if args.defaults else "pinned budget (tol=0, shuffle=off)"
    print(f"mlrs {mlrs.__name__} | loss={args.loss} penalty={args.penalty} "
          f"lr={args.learning_rate} alpha={args.alpha} dtype={args.dtype} | {mode}")
    header = (f"{'n':>7} {'d':>4} {'iter':>5} | {'engine':>8} "
              f"{'fit (s)':>10} {'first (s)':>10}")
    print(header)
    print("-" * len(header))

    for n, d, cfg_iter in configs:
        max_iter = args.max_iter or cfg_iter
        x, y = make_classification(n, d)
        x = np.ascontiguousarray(x.astype(dt))

        common = dict(
            loss=None, penalty=args.penalty, alpha=args.alpha,
            learning_rate=args.learning_rate, eta0=args.eta0, max_iter=max_iter,
        )
        common.pop("loss")

        results = {}
        if args.engine in ("both", "mlrs"):
            def fit_mlrs():
                m = MlrsEst(loss=args.loss, batch_size=1,
                            tol=(1e-3 if args.defaults else 0.0),
                            shuffle=args.defaults, **common)
                m.fit(x, y)
                return m

            try:
                best, first, model = best_of(fit_mlrs, args.reps)
                results["mlrs"] = (best, first)
                if args.check:
                    results["mlrs_coef"] = np.asarray(model.coef_, dtype=np.float64).ravel()
            except Exception as exc:  # noqa: BLE001
                print(f"{n:>7} {d:>4} {max_iter:>5} |     mlrs  FAILED: "
                      f"{type(exc).__name__}: {exc}")

        if args.engine in ("both", "sklearn"):
            def fit_sk():
                m = SkEst(loss=sk_loss,
                          tol=(1e-3 if args.defaults else None),
                          shuffle=args.defaults, n_iter_no_change=5, **common)
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
                print(f"{n:>7} {d:>4} {max_iter:>5} | {eng:>8} "
                      f"{best:>10.4f} {first:>10.4f}")
        if "mlrs" in results and "sklearn" in results:
            speedup = results["sklearn"][0] / results["mlrs"][0]
            note = f"{speedup:.2f}x vs sklearn"
            if args.check and "mlrs_coef" in results and "sk_coef" in results:
                a, b = results["mlrs_coef"], results["sk_coef"]
                dev = float(np.max(np.abs(a - b)))
                rel = dev / max(1e-30, float(np.max(np.abs(b))))
                note += f" | max|Δcoef| = {dev:.3e} (rel {rel:.3e})"
            print(f"{'':>7} {'':>4} {'':>5} | {note}")
        print()


if __name__ == "__main__":
    main()
