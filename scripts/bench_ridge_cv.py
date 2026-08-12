#!/usr/bin/env python3
"""``RidgeCV`` wall-clock comparison: mlrs (cpu backend) vs scikit-learn.

    python3 scripts/bench_ridge_cv.py                 # the whole suite
    python3 scripts/bench_ridge_cv.py --only alphas   # one section
    MLRS_BENCH_REPS=9 python3 scripts/bench_ridge_cv.py

``RidgeCV``'s cost is dominated by ONE parameter — ``len(alphas)`` — and by
which engine ``cv`` selects, so the ladders are built around those rather than
around the design shape alone:

* **shape**    — the default ``RidgeCV()`` at a spread of ``(n, d)``.
* **alphas**   — the same design at 1 … 200 alphas. This is the section that
  shows the algorithmic difference: sklearn's default dense route re-forms an
  ``n x d`` product per alpha, mlrs forms the eigenbasis projection once.
* **gcv_mode** — all four spellings, both libraries. mlrs's are ONE code path
  (see ``ridge_cv.rs``), sklearn's are three different LAPACK calls with very
  different costs, so this section is also the honest denominator: the win is
  quoted against sklearn's BEST mode as well as against its default.
* **cv**       — the explicit-``GridSearchCV`` arm at 3/5/10 folds.
* **scoring**  — ``None`` (scored in Rust) vs a string scorer (scored in Python,
  which costs a per-alpha callback and an ``n``-vector crossing the boundary).
* **targets / weights / store_cv_results** — the remaining parameters, to show
  which ones actually move the number.

## Each library is timed in its OWN PROCESS, and that is not fussiness

Timing the two alternately in one process INVERTED three verdicts while this
script was being written: ``n=10 000, d=64`` read as a 0.64x loss interleaved
and a 3.9x win measured alone. The cause is not mysterious — numpy's OpenBLAS
keeps its worker threads SPINNING for a while after a parallel region ends, so
whichever library runs second is handed a machine whose cores are still busy.
mlrs's own ``std::thread::scope`` workers do the same to sklearn. Running each
library in a fresh process is the only arrangement where neither inherits the
other's thread pool ([[mlrs-cpu-bench-separate-processes]]).

``--same-process`` reproduces the bad arrangement on purpose, for anyone who
wants to see the effect rather than take it on trust. (It is the WEAKER form:
whole sections rather than alternating laps, and it still moves the numbers.)

Requires numpy + scikit-learn + a built mlrs cpu extension.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPS = int(os.environ.get("MLRS_BENCH_REPS", "7"))


def _splitmix64_block(seed: int, count: int) -> np.ndarray:
    """Counter-based splitmix64 — the workspace's shared deterministic stream,
    so a Rust probe and this script fit the same numbers."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        state = (np.uint64(seed) + idx * np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        z = state
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
        return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def _uniform_pm1(seed: int, count: int) -> np.ndarray:
    u = (_splitmix64_block(seed, count) >> np.uint64(11)) / float(1 << 53)
    return u * 2.0 - 1.0


def make_regression(n: int, d: int, n_targets: int = 1, seed: int = 42):
    x = _uniform_pm1(seed, n * d).reshape(n, d)
    coef = _uniform_pm1(seed + 1, d * n_targets).reshape(d, n_targets)
    noise = _uniform_pm1(seed + 2, n * n_targets).reshape(n, n_targets)
    y = x @ coef + 0.5 + 0.05 * noise
    if n_targets == 1:
        y = y[:, 0]
    return np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(
        y, dtype=np.float64
    )


def measure(fn, reps: int = REPS) -> float:
    """Min-of-N wall clock in ms, after a discarded warmup.

    The MINIMUM, not the mean: unrelated load can only ever ADD time, so the
    smallest lap is the least contaminated estimate of the work itself.
    """
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


# ---------------------------------------------------------------------------
# The ladders. Each returns {label: milliseconds} for ONE library.
# ---------------------------------------------------------------------------


def bench_shape(RidgeCV, reps):
    out = {}
    for n, d in [
        (1_000, 16),
        (10_000, 16),
        (10_000, 64),
        (100_000, 16),
        (100_000, 64),
        (200_000, 64),
        (50_000, 128),
        (20_000, 256),
    ]:
        X, y = make_regression(n, d)
        out[f"n={n:,} d={d}"] = measure(lambda: RidgeCV().fit(X, y), reps)
    return out


def bench_alphas(RidgeCV, reps):
    out = {}
    X, y = make_regression(100_000, 64)
    for k in (1, 3, 10, 30, 100, 200):
        a = np.logspace(-3, 3, k)
        out[f"n=100k d=64  len(alphas)={k}"] = measure(
            lambda a=a: RidgeCV(alphas=a).fit(X, y), reps
        )
    X, y = make_regression(20_000, 256)
    for k in (1, 3, 10, 50):
        a = np.logspace(-3, 3, k)
        out[f"n=20k  d=256 len(alphas)={k}"] = measure(
            lambda a=a: RidgeCV(alphas=a).fit(X, y), max(3, reps // 2)
        )
    return out


def bench_gcv_mode(RidgeCV, reps):
    out = {}
    X, y = make_regression(100_000, 64)
    alphas = np.logspace(-3, 3, 30)
    for mode in (None, "auto", "eigen", "svd"):
        out[f"tall  gcv_mode={mode!r}"] = measure(
            lambda m=mode: RidgeCV(alphas=alphas, gcv_mode=m).fit(X, y), reps
        )
    Xw, yw = make_regression(400, 2_000)
    for mode in (None, "svd"):
        out[f"wide  gcv_mode={mode!r}"] = measure(
            lambda m=mode: RidgeCV(alphas=alphas, gcv_mode=m).fit(Xw, yw), 3
        )
    return out


def bench_cv(RidgeCV, reps):
    out = {}
    X, y = make_regression(100_000, 64)
    alphas = np.logspace(-3, 3, 30)
    for folds in (3, 5, 10):
        out[f"cv={folds}"] = measure(
            lambda f=folds: RidgeCV(alphas=alphas, cv=f).fit(X, y), 3
        )
    return out


def bench_scoring(RidgeCV, reps):
    out = {}
    X, y = make_regression(100_000, 64)
    alphas = np.logspace(-3, 3, 30)
    for sc in (None, "r2", "neg_mean_squared_error", "neg_mean_absolute_error"):
        out[f"scoring={sc!r}"] = measure(
            lambda s=sc: RidgeCV(alphas=alphas, scoring=s).fit(X, y), reps
        )
    return out


def bench_misc(RidgeCV, reps):
    out = {}
    X, y = make_regression(100_000, 64)
    alphas = np.logspace(-3, 3, 30)
    w = np.abs(_uniform_pm1(7, X.shape[0])) + 0.1

    out["fit_intercept=False"] = measure(
        lambda: RidgeCV(alphas=alphas, fit_intercept=False).fit(X, y), reps
    )
    out["sample_weight"] = measure(
        lambda: RidgeCV(alphas=alphas).fit(X, y, sample_weight=w), reps
    )
    out["store_cv_results=True"] = measure(
        lambda: RidgeCV(alphas=alphas, store_cv_results=True).fit(X, y), reps
    )

    Xm, ym = make_regression(100_000, 64, n_targets=5)
    out["5 targets"] = measure(lambda: RidgeCV(alphas=alphas).fit(Xm, ym), 3)
    out["5 targets, alpha_per_target"] = measure(
        lambda: RidgeCV(alphas=alphas, alpha_per_target=True).fit(Xm, ym), 3
    )

    Xp, yp = make_regression(200_000, 64)
    Xq, _ = make_regression(1_000_000, 64, seed=99)
    est = RidgeCV(alphas=alphas).fit(Xp, yp)
    out["predict n=1,000,000 d=64"] = measure(lambda: est.predict(Xq), 3)
    return out


SECTIONS = {
    "shape": bench_shape,
    "alphas": bench_alphas,
    "gcv_mode": bench_gcv_mode,
    "cv": bench_cv,
    "scoring": bench_scoring,
    "misc": bench_misc,
}

TITLES = {
    "shape": "shape — RidgeCV() defaults (3 alphas)",
    "alphas": "alphas — the parameter that dominates the fit",
    "gcv_mode": "gcv_mode — mlrs is ONE route; sklearn's three are three LAPACK calls",
    "cv": "cv — the explicit GridSearchCV arm (n=100k d=64, 30 alphas)",
    "scoring": "scoring — None is scored in Rust, a string scorer in Python",
    "misc": "the remaining parameters (n=100k d=64, 30 alphas)",
}


# Fraction of the machine that OTHER processes may consume during a timing run
# before it is reported as UNTRUSTED.
#
# The measurement has to be FOREIGN cpu, not load average. A run that started at
# 3.4 and ended at 259 (another agent's test suite restarting mid-run) produced
# a table where sklearn was 50% SLOWER at 3 alphas than at 1 -- arithmetically
# impossible, and it would have been read as a win. But a load-average
# threshold cannot catch that without also firing on every honest run: a
# 16-thread benchmark on a 16-core box IS a load of ~16, so the first version of
# this guard flagged a run on a provably idle machine and would have trained
# whoever read it to ignore the banner.
#
# So: sample total busy jiffies from /proc/stat, subtract this process tree's
# own cpu time, and divide by (wall x cores). That is the share of the machine
# somebody else took, which is exactly the quantity that distorts a
# parallel-vs-single-threaded comparison.
FOREIGN_LIMIT = float(os.environ.get("MLRS_BENCH_FOREIGN_LIMIT", "0.10"))


def _cpu_sample():
    """(wall seconds, system busy seconds, own cpu seconds) right now."""
    import resource

    with open("/proc/stat") as fh:
        parts = [float(v) for v in fh.readline().split()[1:]]
    ticks = os.sysconf("SC_CLK_TCK")
    total = sum(parts) / ticks
    idle = (parts[3] + (parts[4] if len(parts) > 4 else 0.0)) / ticks
    own = sum(
        getattr(resource.getrusage(who), f)
        for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
        for f in ("ru_utime", "ru_stime")
    )
    return time.perf_counter(), total - idle, own


def foreign_share(before, after):
    """Fraction of the machine consumed by OTHER processes between two samples."""
    wall = after[0] - before[0]
    busy = after[1] - before[1]
    own = after[2] - before[2]
    cores = os.cpu_count() or 1
    if wall <= 0:
        return 0.0
    return max(0.0, (busy - own)) / (wall * cores)


def run_one(lib: str, names) -> dict:
    """One library, this process. Brackets the timings with a cpu sample."""
    if lib == "mlrs":
        import mlrs

        RidgeCV = mlrs.RidgeCV
    else:
        from sklearn.linear_model import RidgeCV
    before = _cpu_sample()
    out = {n: SECTIONS[n](RidgeCV, REPS) for n in names}
    return {"timings": out, "foreign": foreign_share(before, _cpu_sample())}


def child(lib: str, names) -> dict:
    env = dict(os.environ, MLRS_BENCH_CHILD=lib)
    proc = subprocess.run(
        [sys.executable, __file__, *sum((["--only", n] for n in names), []),
         "--_child", lib],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{lib} child failed")
    return json.loads(proc.stdout)


def report(names, sk, me, foreign):
    hot = [x for x in foreign if x > FOREIGN_LIMIT]
    losses = []
    for n in names:
        print()
        print(TITLES[n])
        print("  " + "-" * 84)
        for label in sk[n]:
            a, b = sk[n][label], me[n][label]
            ratio = a / b if b else float("nan")
            flag = "" if ratio >= 1.0 else "   <-- LOSS"
            if ratio < 1.0:
                losses.append((n, label, ratio))
            print(
                f"  {label:<32} sklearn {a:9.2f} ms   mlrs {b:9.2f} ms"
                f"   {ratio:6.2f}x{flag}"
            )
    print()
    if losses:
        print("LOSSES:")
        for n, label, r in losses:
            print(f"  {n}/{label}: {r:.2f}x")
    else:
        print("no losses in this run")
    print()
    print(
        "foreign cpu during each child (share of the machine taken by OTHER "
        "processes): " + ", ".join(f"{x:.1%}" for x in foreign)
    )
    if hot:
        print(
            f"*** UNTRUSTED: other processes took up to {max(hot):.0%} of the "
            f"machine (limit {FOREIGN_LIMIT:.0%}). This compares a parallel "
            "engine with a mostly single-threaded one, so contention changes "
            "the RATIO, not just the variance. Re-run on a quiet machine. ***"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SECTIONS), action="append")
    ap.add_argument(
        "--same-process",
        dest="same_process",
        action="store_true",
        help="time both libraries in ONE process (the arrangement that lies)",
    )
    ap.add_argument("--_child", choices=("mlrs", "sklearn"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    names = args.only or list(SECTIONS)

    if args._child:
        json.dump(run_one(args._child, names), sys.stdout)
        return 0

    print(f"reps: {REPS}   numpy {np.__version__}   cores: {os.cpu_count()}")
    print(f"load: {os.getloadavg()}")
    if args.same_process:
        print("MODE: both libraries in one process (results are NOT trustworthy)")
        sk = run_one("sklearn", names)
        me = run_one("mlrs", names)
    else:
        sk = child("sklearn", names)
        me = child("mlrs", names)
    report(names, sk["timings"], me["timings"], [sk["foreign"], me["foreign"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
