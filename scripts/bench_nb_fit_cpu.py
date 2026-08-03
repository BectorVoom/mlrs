#!/usr/bin/env python3
"""GaussianNB / MultinomialNB / ComplementNB **fit** wall-clock vs scikit-learn.

The NB-FIT-CPU probe for the three variants whose fits used to run through the
``class_grouped_sum`` device GATHER — a per-class row gather + upload +
``column_reduce`` launch + read-back, i.e. ``n_classes`` cubecl-cpu kernels over
the whole design matrix (GaussianNB paid it TWICE, for ``sum`` and ``sumsq``,
plus a third COLUMN-strided pass for ``epsilon_``). They now run one row-major
host sweep, like their BernoulliNB / CategoricalNB siblings — which have their
own scripts (``bench_bernoulli_nb_cpu.py`` / ``bench_categorical_nb_cpu.py``);
this one covers the remaining three.

    .venv/bin/python scripts/bench_nb_fit_cpu.py [--reps 3] [--check]
                     [--estimator all|gaussian|multinomial|complement]
                     [--engine mlrs|sklearn|both] [--dtype float64|float32]

Each estimator gets the input it is FOR: GaussianNB real-valued standard-normal
features, Multinomial/Complement non-negative integer counts. The ladder widens
as well as lengthens — a wide-``d`` rung is the only thing that exposes a
column-strided pass (it moves ``O(n · d^2)`` bytes), which is what GaussianNB's
``epsilon_`` pass was.

The ``--engine`` caveat from ``bench_linear_predict_cpu.py`` applies verbatim:
BLAS/threading state left behind by one engine can tax whichever runs second.
Re-run a suspicious rung with ``--engine mlrs`` / ``--engine sklearn`` in
separate processes.

On a machine with unrelated load, prefer the default INTERLEAVED schedule
(engines alternate rep by rep, so a load burst hits both) and read the
``cpu (s)`` column — ``time.process_time`` excludes time the process spent
descheduled. NOTE that mlrs' sweep IS multi-threaded above ~32k elements while
sklearn's is single-threaded numpy (its count GEMM is not), so the two columns
bracket the answer rather than agreeing.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

# (rows, features, n_classes) of the timed fit.
CONFIGS = [
    (1_000, 8, 3),
    (10_000, 16, 4),
    (50_000, 32, 5),
    (100_000, 64, 10),
    (500_000, 8, 2),
    (100_000, 128, 4),
    (50_000, 256, 4),
    (20_000, 512, 4),
    (100_000, 32, 20),
]


def make_data(kind: str, n: int, d: int, n_classes: int, seed: int = 42):
    """The input each estimator is actually for.

    ``gaussian`` gets real-valued standard-normal features (negatives included —
    GaussianNB's sweep rejects only non-finite values); ``multinomial`` /
    ``complement`` get non-negative integer counts, which is what their
    ``check_non_negative`` parity demands.
    """
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        x = rng.standard_normal((n, d))
    else:
        x = rng.poisson(2.0, size=(n, d)).astype(np.float64)
    y = rng.integers(0, n_classes, size=n)
    return x, y


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


def max_deviation(mlrs_model, sk_model, x, sample=512):
    """max |delta log P(c | x)| between the two fitted models on `sample` rows.

    The fitted tables are what this benchmark's optimization touches, but the
    mlrs shim does not surface them (only the predict surface is public), so the
    check reads them THROUGH ``predict_log_proba``. A row subset keeps
    ``--check`` from dominating the timing run on the large rungs.
    """
    xs = x[:sample]
    a = np.asarray(mlrs_model.predict_log_proba(xs), dtype=np.float64)
    b = np.asarray(sk_model.predict_log_proba(xs), dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument(
        "--estimator",
        default="all",
        choices=["all", "gaussian", "multinomial", "complement"],
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="print max|dlog P(c|x)| vs sklearn (reads the fitted tables through "
        "predict_log_proba — the shim does not surface them directly)",
    )
    ap.add_argument("--configs", default="", help="comma-separated n:d:n_classes")
    ap.add_argument(
        "--schedule",
        default="interleaved",
        choices=["interleaved", "blocked"],
        help="alternate engines rep by rep (default) or run each engine's "
        "reps back to back",
    )
    args = ap.parse_args()

    import mlrs.naive_bayes as mlrs_nb
    import sklearn.naive_bayes as sk_nb

    kinds = (
        ["gaussian", "multinomial", "complement"]
        if args.estimator == "all"
        else [args.estimator]
    )
    names = {
        "gaussian": "GaussianNB",
        "multinomial": "MultinomialNB",
        "complement": "ComplementNB",
    }

    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]

    dt = {"float64": np.float64, "float32": np.float32}[args.dtype]
    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]

    for kind in kinds:
        name = names[kind]
        MlrsEst = getattr(mlrs_nb, name)
        SkEst = getattr(sk_nb, name)
        print(f"\n=== {name} | dtype={args.dtype}")
        header = (
            f"{'n':>7} {'d':>4} {'C':>3} | {'engine':>8} "
            f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10}"
        )
        print(header)
        print("-" * len(header))

        for n, d, n_classes in configs:
            x, y = make_data(kind, n, d, n_classes)
            x = np.ascontiguousarray(x.astype(dt))

            fits = {
                "mlrs": lambda: MlrsEst().fit(x, y),
                "sklearn": lambda: SkEst().fit(x, y),
            }
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
                print(f"{n:>7} {d:>4} {n_classes:>3} | {eng:>8}  FAILED: {msg}")

            ok = [e for e in engines if e not in failed]
            for eng in ok:
                s = samples[eng]
                print(
                    f"{n:>7} {d:>4} {n_classes:>3} | {eng:>8} "
                    f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f}"
                )
            if len(ok) == 2:
                wall_x = samples["sklearn"].best / samples["mlrs"].best
                cpu_x = samples["sklearn"].best_cpu / samples["mlrs"].best_cpu
                note = f"{wall_x:.2f}x wall / {cpu_x:.2f}x cpu vs sklearn"
                if args.check:
                    dev = max_deviation(
                        samples["mlrs"].model, samples["sklearn"].model, x
                    )
                    note += f" | max|dlog P(c|x)| = {dev:.3e}"
                print(f"{'':>7} {'':>4} {'':>3} | {note}")
            print()


if __name__ == "__main__":
    main()
