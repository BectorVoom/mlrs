#!/usr/bin/env python3
"""BayesianRidge **fit** wall-clock: mlrs (cpu backend) vs scikit-learn.

The LINEAR-06 cpu probe. Both engines run the SAME evidence iteration (MacKay
1992) to the SAME stopping rule, so the comparison is like-for-like at library
defaults — no budget pinning is needed, unlike the SGD probe. What differs is
only HOW each step is evaluated:

    sklearn   thin SVD of X up front, then per iteration an O(n·d) residual
              pass (`y - X @ coef_`) plus an O(d^2) posterior mean
    mlrs      one O(n·d^2/2) Gram sweep + one O(d^3) eigendecomposition up
              front, then per iteration O(d) in the eigenbasis (the residual
              identity `yTy - 2*w.Xty + w.Gw` is diagonal there) plus the
              O(d^2) `coef_` materialization the L1 convergence test needs

So the expected shape of the result is that mlrs' margin GROWS with
`n_samples` (sklearn's per-iteration cost scales with it, mlrs' does not) and
SHRINKS with `n_features` (mlrs' eigenwork is cubic in it).

    .venv/bin/python scripts/bench_bayesian_ridge_cpu.py [--reps 5] [--check]
                     [--engine mlrs|sklearn|both] [--dtype float64]

The `--engine` caveat from `bench_linear_predict_cpu.py` applies verbatim:
OpenBLAS keeps its workers SPINNING after a call, so interleaving both engines
in one process taxes whichever runs second. Re-run a suspicious rung with
`--engine mlrs` / `--engine sklearn` in separate processes.

On a machine with unrelated load, prefer the default INTERLEAVED schedule
(engines alternate rep by rep, so a load burst hits both) and read the
`cpu (s)` column -- `time.process_time` excludes time the process spent
descheduled, and it is the number to trust on a busy box (the caveat recorded
in `mlrs-hgb-cpu-bench-caveat`). Note that CPU time also CHARGES mlrs for its
worker threads, so on the wide rungs `cpu (s)` is a deliberately pessimistic
reading of the mlrs column rather than a like-for-like one.

`--check` prints max|Δcoef| against sklearn plus the two fitted precisions and
the iteration count, so a rung that wins on time but drifts on numerics is
visible in the same line. A speed number is only meaningful if `n_iter_`
matches: an engine that stops earlier is solving a different problem.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from bench_linear import make_regression as _make_regression_low_noise

# (rows, features) of the timed fit. The ladder deliberately walks BOTH axes
# independently: `n` at fixed `d` isolates the per-iteration residual pass that
# the eigenbasis removes, and `d` at fixed `n` isolates the O(d^3) eigenwork
# that is the price of removing it.
CONFIGS = [
    (1_000, 8),
    (10_000, 8),
    (10_000, 64),
    (100_000, 16),
    (100_000, 64),
    (50_000, 128),
    (20_000, 256),
]


def make_design(n: int, d: int, noise: float, seed: int = 42):
    """`bench_linear.make_regression`'s design, with the noise level dialled.

    The shared generator fixes the noise at `0.01`, which makes the evidence
    iteration converge in two or three steps — so a benchmark built on it
    measures the ONE-TIME work (sklearn's SVD against the Gram sweep) and barely
    touches the per-iteration cost at all.

    That matters because the two engines differ in BOTH places, and only the
    per-iteration difference scales with `n_samples`: sklearn recomputes
    `y - X @ coef_` every step at `O(n*d)`, mlrs evaluates the same residual in
    the eigenbasis at `O(d)`. `--noise` exists so the iteration-heavy regime can
    be measured rather than assumed. A larger noise level leaves more variance
    unexplained, which makes the evidence updates take longer to settle.
    """
    x, y = _make_regression_low_noise(n, d, seed)
    if noise <= 0.01:
        return x, y
    rng = np.random.default_rng(seed + 1000)
    return x, (y + np.float32(noise) * rng.standard_normal(n).astype(np.float32)).astype(np.float32)


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
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--compute-score", action="store_true",
                    help="also accumulate the log marginal likelihood each iteration")
    ap.add_argument("--check", action="store_true",
                    help="print max|Δcoef|, the fitted precisions and n_iter_ vs sklearn")
    ap.add_argument("--noise", type=float, default=0.01,
                    help="target noise sigma; higher leaves more variance unexplained and "
                         "so takes MORE evidence iterations (default 0.01 converges in ~3)")
    ap.add_argument("--configs", default="", help="comma-separated n:d")
    ap.add_argument("--schedule", default="interleaved",
                    choices=["interleaved", "blocked"],
                    help="alternate engines rep by rep (default) or run each engine's "
                         "reps back to back")
    args = ap.parse_args()

    import mlrs
    from mlrs import BayesianRidge as MlrsEst
    from sklearn.linear_model import BayesianRidge as SkEst

    configs = CONFIGS
    if args.configs:
        configs = [tuple(int(v) for v in c.split(":")) for c in args.configs.split(",")]

    dt = np.float32 if args.dtype == "float32" else np.float64
    print(f"mlrs {mlrs.__name__} | max_iter={args.max_iter} tol={args.tol} "
          f"noise={args.noise} compute_score={args.compute_score} dtype={args.dtype}")
    header = (f"{'n':>7} {'d':>4} | {'engine':>8} "
              f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10} {'n_iter':>7}")
    print(header)
    print("-" * len(header))

    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]
    common = dict(
        max_iter=args.max_iter, tol=args.tol, compute_score=args.compute_score
    )

    for n, d in configs:
        x, y = make_design(n, d, args.noise)
        x = np.ascontiguousarray(x.astype(dt))
        y = np.ascontiguousarray(y.astype(dt))

        def fit_mlrs():
            m = MlrsEst(**common)
            m.fit(x, y)
            return m

        def fit_sk():
            m = SkEst(**common)
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
            print(f"{n:>7} {d:>4} | {eng:>8}  FAILED: {msg}")

        ok = [e for e in engines if e not in failed]
        for eng in ok:
            s = samples[eng]
            print(f"{n:>7} {d:>4} | {eng:>8} "
                  f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f} "
                  f"{int(s.model.n_iter_):>7}")
        if len(ok) == 2:
            wall_x = samples["sklearn"].best / samples["mlrs"].best
            cpu_x = samples["sklearn"].best_cpu / samples["mlrs"].best_cpu
            note = f"{wall_x:.2f}x wall / {cpu_x:.2f}x cpu vs sklearn"
            mm, sm = samples["mlrs"].model, samples["sklearn"].model
            # A speed number is only meaningful if both engines ran the same
            # number of iterations — flag it loudly rather than folding a
            # different stopping point into the ratio.
            if int(mm.n_iter_) != int(sm.n_iter_):
                note += f"  [!] n_iter_ differs: mlrs={mm.n_iter_} sklearn={sm.n_iter_}"
            if args.check:
                a = np.asarray(mm.coef_, dtype=np.float64).ravel()
                b = np.asarray(sm.coef_, dtype=np.float64).ravel()
                dev = float(np.max(np.abs(a - b)))
                rel = dev / max(1e-30, float(np.max(np.abs(b))))
                note += (
                    f" | max|Δcoef| = {dev:.3e} (rel {rel:.3e})"
                    f" | alpha_ {mm.alpha_:.6g}/{sm.alpha_:.6g}"
                    f" lambda_ {mm.lambda_:.6g}/{sm.lambda_:.6g}"
                )
            print(f"{'':>7} {'':>4} | {note}")
        print()


if __name__ == "__main__":
    main()
