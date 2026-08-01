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

On a machine with unrelated load, prefer the default INTERLEAVED schedule
(engines alternate rep by rep, so a load burst hits both) and read the
``cpu (s)`` column — ``time.process_time`` excludes time the process spent
descheduled. Both engines are single-threaded on this path, so CPU time and
wall clock agree on an idle box and CPU time is the one to trust on a busy
one. ``--schedule blocked`` restores the run-all-reps-of-one-engine-first
order.
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


def timed_call(fn):
    """(wall, cpu, result) seconds for ONE call."""
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn()
    return time.perf_counter() - w0, time.process_time() - c0, out


class Samples:
    """Per-engine timing accumulator: min wall, min cpu, first wall, last model."""

    def __init__(self):
        self.wall = []
        self.cpu = []
        self.model = None

    def add(self, wall, cpu, model):
        self.wall.append(wall)
        self.cpu.append(cpu)
        self.model = model

    @property
    def best(self):
        return min(self.wall)

    @property
    def best_cpu(self):
        return min(self.cpu)

    @property
    def first(self):
        return self.wall[0]


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
    ap.add_argument("--schedule", default="interleaved",
                    choices=["interleaved", "blocked"],
                    help="alternate engines rep by rep (default) or run each engine's "
                         "reps back to back")
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
              f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10}")
    print(header)
    print("-" * len(header))

    import warnings

    engines = [e for e in ("mlrs", "sklearn")
               if args.engine in ("both", e)]

    for n, d, cfg_iter in configs:
        max_iter = args.max_iter or cfg_iter
        x, y = make_classification(n, d)
        x = np.ascontiguousarray(x.astype(dt))

        common = dict(
            penalty=args.penalty, alpha=args.alpha,
            learning_rate=args.learning_rate, eta0=args.eta0, max_iter=max_iter,
        )

        def fit_mlrs():
            m = MlrsEst(loss=args.loss, batch_size=1,
                        tol=(1e-3 if args.defaults else 0.0),
                        shuffle=args.defaults, **common)
            m.fit(x, y)
            return m

        def fit_sk():
            m = SkEst(loss=sk_loss,
                      tol=(1e-3 if args.defaults else None),
                      shuffle=args.defaults, n_iter_no_change=5, **common)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(x, y)
            return m

        fits = {"mlrs": fit_mlrs, "sklearn": fit_sk}
        samples = {e: Samples() for e in engines}
        failed = {}

        # Interleaved by default: a load burst from an unrelated process hits
        # both engines rather than taxing whichever happens to run second.
        order = (
            [e for _ in range(args.reps) for e in engines]
            if args.schedule == "interleaved"
            else [e for e in engines for _ in range(args.reps)]
        )
        for eng in order:
            if eng in failed:
                continue
            try:
                samples[eng].add(*timed_call(fits[eng]))
            except Exception as exc:  # noqa: BLE001
                failed[eng] = f"{type(exc).__name__}: {exc}"

        for eng, msg in failed.items():
            print(f"{n:>7} {d:>4} {max_iter:>5} | {eng:>8}  FAILED: {msg}")

        ok = [e for e in engines if e not in failed]
        for eng in ok:
            s = samples[eng]
            print(f"{n:>7} {d:>4} {max_iter:>5} | {eng:>8} "
                  f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f}")
        if len(ok) == 2:
            wall_x = samples["sklearn"].best / samples["mlrs"].best
            cpu_x = samples["sklearn"].best_cpu / samples["mlrs"].best_cpu
            note = f"{wall_x:.2f}x wall / {cpu_x:.2f}x cpu vs sklearn"
            if args.check:
                a = np.asarray(samples["mlrs"].model.coef_, dtype=np.float64).ravel()
                b = np.asarray(samples["sklearn"].model.coef_, dtype=np.float64).ravel()
                dev = float(np.max(np.abs(a - b)))
                rel = dev / max(1e-30, float(np.max(np.abs(b))))
                note += f" | max|Δcoef| = {dev:.3e} (rel {rel:.3e})"
            print(f"{'':>7} {'':>4} {'':>5} | {note}")
        print()


if __name__ == "__main__":
    main()
