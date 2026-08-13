#!/usr/bin/env python3
"""Is `StackingRegressor`'s orchestration overhead FIXED or PER-SAMPLE? (STACK-OVERHEAD)

With cheap members, `mlrs.StackingRegressor` measures ~0.80-0.92x of
scikit-learn — a stable loss in the composition layer rather than in anything
the members do (expensive members hide it: they turn the same overhead into a
few percent). This harness answers the one question that decides where to look
for it, and that no single problem size can answer:

    Is the overhead a FIXED cost per fit, or does it scale with n?

## Why a ratio at one `n` cannot tell you

A fixed ~10 ms overhead shows up as 0.81x on a 50 ms fit and 0.90x on a 300 ms
one. A per-sample overhead shows up as roughly the SAME ratio at both. Read at a
single size, those two are indistinguishable — which is exactly the trap that
made a `cv="prefit"` cell look like it exonerated the FFI index marshalling in
`cross_val_predict` (~2n boxed Python ints per call).

So this sweeps `n` and reports the **absolute delta**, not just the ratio:

    delta_ms      = mlrs_ms - sklearn_ms
    delta_per_1k  = delta_ms / (n / 1000)

* **Fixed cost** → `delta_ms` flat as `n` grows, `delta_per_1k` falling, ratio
  drifting toward 1.00.
* **Per-sample cost** → `delta_per_1k` flat, `delta_ms` growing linearly, ratio
  roughly constant.

## Two routes, because they differ by exactly one thing

* `cv="prefit"` — the composition WITHOUT `cross_val_predict`: no fold fits, no
  index marshalling, no joblib fan-out.
* `cv=5` — the same composition WITH it.

Subtracting the two isolates `cross_val_predict`'s own contribution, which is
where the `.tolist()` boundary lives. Note the prefit route still runs the
member `predict`s, so it is "no cross-validation", not "no work".

## Members are DELIBERATELY degenerate

`DummyRegressor` fits in one pass over `y` and predicts a constant, so the
members contribute almost nothing and the composition IS the measurement. That
is a control, not a recommendation — `--members ridge` runs the realistic shape
for comparison, and the interesting number is how much of the delta survives
when the members get expensive.

## Measurement discipline

Fresh subprocess per cell, min of `--repeat`, arms interleaved. The warm-up is a
full fit at the REAL `(n, d)`, not a small one: an arm or device pipeline chosen
by SHAPE is not warmed by a smaller fit, and the compile then lands inside the
timed region and reads as a per-cell cost that min-of-N cannot remove. The
signature of that mistake is a cost FLAT in the parameter being swept.

Usage
-----
    python3 scripts/bench_stacking_overhead.py                  # both routes
    python3 scripts/bench_stacking_overhead.py --members ridge
    python3 scripts/bench_stacking_overhead.py --repeat 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

SEED = 42
N_LADDER = [5_000, 20_000, 50_000, 100_000, 200_000, 400_000]
D_DEFAULT = 16


def design(n, d, dtype):
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, d)).astype(dtype)
    w = rng.standard_normal(d).astype(dtype)
    y = (X @ w + 0.05 * rng.standard_normal(n)).astype(dtype)
    return X, y


def build(impl, members, cv, X, y):
    """The estimator under test. Both implementations get the SAME member
    classes — this harness measures the composition, so the members must not be
    a difference between the arms."""
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge

    if members == "dummy":
        pair = [("a", DummyRegressor()), ("b", DummyRegressor(strategy="median"))]
    else:
        pair = [("a", Ridge(alpha=1.0)), ("b", Ridge(alpha=2.0))]
    final = Ridge(alpha=1.0)

    if cv == "prefit":
        pair = [(name, est.fit(X, y)) for name, est in pair]

    if impl == "sklearn":
        from sklearn.ensemble import StackingRegressor
    else:
        from mlrs import StackingRegressor
    return StackingRegressor(pair, final_estimator=final, cv=cv)


def run_cell(impl, members, cv, n, d, reps):
    dtype = np.float64
    startup_s = 0.0
    if impl == "mlrs":
        import mlrs

        t0 = time.perf_counter()
        mlrs._load_ext()
        startup_s = time.perf_counter() - t0
        if not mlrs.backend_supports_f64():
            dtype = np.float32

    X, y = design(n, d, dtype)

    # Warm at the REAL shape (see the module docstring).
    build(impl, members, cv, X, y).fit(X, y)

    best = float("inf")
    for _ in range(reps):
        est = build(impl, members, cv, X, y)
        t0 = time.perf_counter()
        est.fit(X, y)
        best = min(best, time.perf_counter() - t0)
    return best, startup_s


def spawn(impl, members, cv, n, d, reps):
    out = subprocess.run(
        [
            sys.executable, __file__, "--cell",
            "--impl", impl, "--members", members, "--cv", str(cv),
            "--n", str(n), "--d", str(d), "--inner", str(reps),
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return None


HEADER = (
    f"{'n':>9}{'sklearn':>11}{'mlrs':>11}{'ratio':>8}"
    f"{'delta ms':>11}{'delta/1k':>11}"
)

VERDICT = (
    "# FIXED overhead -> 'delta ms' flat, 'delta/1k' falling, ratio -> 1.00\n"
    "# PER-SAMPLE     -> 'delta/1k' flat, 'delta ms' linear in n"
)


def row(n, sk_ms, ml_ms):
    delta = ml_ms - sk_ms
    return (
        f"{n:>9}{sk_ms:>10.2f}m{ml_ms:>10.2f}m{sk_ms / ml_ms:>7.2f}x"
        f"{delta:>10.2f}m{delta / (n / 1000):>10.3f}m"
    )


def sweep(args, cv):
    """One route's ladder. Returns `{n: (sklearn_ms, mlrs_ms)}` so the caller can
    difference two routes without re-running either."""
    print(f"\n=== cv={cv!r}, members={args.members}, d={args.d}, min of {args.repeat} ===")
    print(HEADER)
    print("-" * len(HEADER))

    out = {}
    for n in args.n_ladder:
        best = {}
        for _ in range(args.repeat):
            for impl in ("sklearn", "mlrs"):  # interleaved, not blocked
                got = spawn(impl, args.members, cv, n, args.d, args.inner)
                if got is None:
                    continue
                prev = best.get(impl, float("inf"))
                best[impl] = min(prev, got["seconds"])
        if len(best) < 2:
            print(f"{n:>9}{'cell failed':>11}")
            continue
        sk, ml = best["sklearn"] * 1e3, best["mlrs"] * 1e3
        out[n] = (sk, ml)
        print(row(n, sk, ml))

    print(VERDICT)
    return out


def difference(cv_route, prefit_route):
    """`cv=k` MINUS `prefit`, per implementation — cross_val_predict's own cost.

    The two routes differ by exactly one thing: the cross-validated fit. So this
    subtraction is the only column that can carry the FFI index marshalling
    (~2n boxed Python ints per call), and it gets its own table rather than
    being left to the reader.

    Both minuend and subtrahend are minima of separate process sets, so this
    difference is noisier than either row — read its SHAPE across n, not any one
    cell.
    """
    shared = sorted(set(cv_route) & set(prefit_route))
    if not shared:
        return
    print("\n=== cv MINUS prefit — cross_val_predict's own contribution ===")
    print(HEADER)
    print("-" * len(HEADER))
    for n in shared:
        sk = cv_route[n][0] - prefit_route[n][0]
        ml = cv_route[n][1] - prefit_route[n][1]
        if sk <= 0 or ml <= 0:
            print(f"{n:>9}{'non-positive (noise dominates)':>32}")
            continue
        print(row(n, sk, ml))
    print(VERDICT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--impl", choices=["sklearn", "mlrs"], default="mlrs")
    p.add_argument("--members", choices=["dummy", "ridge"], default="dummy")
    p.add_argument("--cv", default="5")
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--d", type=int, default=D_DEFAULT)
    p.add_argument("--inner", type=int, default=3, help="in-process reps per cell")
    p.add_argument("--repeat", type=int, default=3, help="fresh processes per cell")
    p.add_argument("--routes", default="prefit,5", help="comma-separated cv values")
    args = p.parse_args()
    args.n_ladder = N_LADDER

    if args.cell:
        cv = args.cv if args.cv == "prefit" else int(args.cv)
        seconds, startup_s = run_cell(
            args.impl, args.members, cv, args.n, args.d, args.inner
        )
        print(json.dumps({"seconds": seconds, "startup_s": startup_s}))
        return 0

    results = {}
    for route in args.routes.split(","):
        cv = route if route == "prefit" else int(route)
        results[route] = sweep(args, cv)
    # The difference is only meaningful between a cross-validated route and the
    # prefit one; with any other pair the two ladders do not differ by exactly
    # cross_val_predict.
    if "prefit" in results:
        for route, table in results.items():
            if route != "prefit":
                difference(table, results["prefit"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
