#!/usr/bin/env python3
"""BernoulliNB **fit** wall-clock: mlrs (cpu backend) vs scikit-learn.

The BERNNB-FIT-CPU probe. Both engines compute the same thing — the binarized
per-``(class, feature)`` occurrence counts smoothed into ``feature_log_prob_`` —
so this is like-for-like with no iteration budget to pin, and ``--check``
asserts the fitted tables agree.

sklearn's fit is ``binarize(X, threshold)`` (a full ``n x d`` copy) followed by
``safe_sparse_dot(Y.T, X)``, a dense ``C x n`` by ``n x d`` BLAS GEMM — so its
cost grows with the CLASS COUNT as well as with ``n * d``. mlrs accumulates one
row into one class accumulator, which is ``C`` times less arithmetic; the
``C = 20`` rung is where that shows up, and the ``d >= 128`` rungs are what
caught the ``O(n * d^2)`` regression in the CategoricalNB sibling. Keep both
when trimming the ladder.

    .venv/bin/python scripts/bench_bernoulli_nb_cpu.py [--reps 3] [--check]
                     [--engine mlrs|sklearn|both] [--dtype float64|float32]

The ``--engine`` caveat from ``bench_linear_predict_cpu.py`` applies verbatim:
BLAS/threading state left behind by one engine can tax whichever runs second.
Re-run a suspicious rung with ``--engine mlrs`` / ``--engine sklearn`` in
separate processes.

On a machine with unrelated load, prefer the default INTERLEAVED schedule
(engines alternate rep by rep, so a load burst hits both) and read the
``cpu (s)`` column — ``time.process_time`` excludes time the process spent
descheduled. NOTE that mlrs' fit pass IS multi-threaded above ~32k elements
while sklearn's binarize is single-threaded numpy (its GEMM is not), so the two
columns bracket the answer rather than agreeing.
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


def make_binary(n: int, d: int, n_classes: int, density: float = 0.3, seed: int = 42):
    """A 0/1 occurrence matrix (the shape BernoulliNB is meant for) + labels.

    Values are exactly 0.0/1.0, so the default ``binarize=0.0`` threshold is a
    no-op on the DATA and both engines count the same occurrences. The
    threshold pass still runs in both — it is part of the fit either way.
    """
    rng = np.random.default_rng(seed)
    x = (rng.random((n, d)) < density).astype(np.float64)
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


def max_table_deviation(mlrs_model, sk_model, x, sample=512):
    """max |delta log P(c | x)| between the two fitted models on `sample` rows.

    The fitted ``feature_log_prob_`` tables are what this benchmark's
    optimization touches, but the mlrs shim does not surface them (only the
    predict surface is public), so the check reads them THROUGH
    ``predict_log_proba``: the joint log-likelihood is a plain sum of table
    entries plus the class log-prior, so a divergence in a table shows up here
    undamped. A row subset keeps ``--check`` from dominating the timing run.
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
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--density", type=float, default=0.3)
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

    import mlrs
    from mlrs.naive_bayes import BernoulliNB as MlrsEst
    from sklearn.naive_bayes import BernoulliNB as SkEst

    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]

    dt = {"float64": np.float64, "float32": np.float32}[args.dtype]
    print(
        f"mlrs {mlrs.__name__} | alpha={args.alpha} "
        f"dtype={args.dtype} density={args.density}"
    )
    header = (
        f"{'n':>7} {'d':>4} {'C':>3} | {'engine':>8} "
        f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10}"
    )
    print(header)
    print("-" * len(header))

    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]

    for n, d, n_classes in configs:
        x, y = make_binary(n, d, n_classes, density=args.density)
        x = np.ascontiguousarray(x.astype(dt))

        def fit_mlrs():
            return MlrsEst(alpha=args.alpha).fit(x, y)

        def fit_sk():
            return SkEst(alpha=args.alpha).fit(x, y)

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
                dev = max_table_deviation(
                    samples["mlrs"].model, samples["sklearn"].model, x
                )
                note += f" | max|dlog P(c|x)| = {dev:.3e}"
            print(f"{'':>7} {'':>4} {'':>3} | {note}")
        print()


if __name__ == "__main__":
    main()
