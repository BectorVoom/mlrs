#!/usr/bin/env python3
"""Meta-matrix assembly arms of ``StackingRegressor`` (STACK-META-01).

``mlrs.StackingRegressor`` can build its meta-feature matrix three ways, chosen
by ``MLRS_STACK_META_ENGINE``:

    ==========  ====================================================
    arm         what performs the copy
    ==========  ====================================================
    ``numpy``   ``np.hstack`` in the shim — the shipping DEFAULT
    ``host``    ``concatenate_predictions`` in ``mlrs-algos``, reached
                through the Arrow capsule boundary
    ``device``  the CubeCL ``stack_meta_block`` scatter: upload each
                block, ``k (+1)`` launches, read the matrix back
    ==========  ====================================================

The three produce byte-identical matrices (``test_stacking_meta_engine.py``
gates that), so this harness answers the only remaining question: which is
fastest, and by how much.

## Why the answer is not obvious enough to skip measuring

The operation has NO arithmetic — it is one ``n x width`` strided copy. So the
two Rust arms start in debt (a capsule crossing each way; the device arm also an
upload and a download), and the interesting question is whether a parallel or
device-bandwidth copy can repay that at large ``n``. Nothing about that is
predictable from first principles, and "it's obviously a memcpy" is the kind of
assumption that has been wrong on this project's cpu backend before.

## Two measurements, because they answer different questions

* ``--level copy`` (default) — the assembly ALONE, on synthetic blocks. This is
  the arm comparison, undiluted.
* ``--level fit`` — a whole ``StackingRegressor.fit`` per arm. Says whether the
  copy is a large enough share of a real fit for the arm choice to be visible
  at all. (Spoiler shape: with ``cv=k`` the fit performs ``k + 1`` base fits per
  member, so the copy is a rounding error unless ``n`` is huge and ``cv`` tiny.)

## Measurement discipline

Every cell runs in a FRESH subprocess (``--cell``) and the harness reports the
MINIMUM of ``--repeat`` runs; cells are interleaved across arms rather than run
in blocks, so a drifting machine penalizes all three equally. Both rules are
load-bearing on this project's cpu backend, where in-process interleaving and a
busy box have each inverted a verdict before.

The one-time ``_mlrs`` extension load is warmed OUTSIDE every timed region and
reported separately: it is ~35-95 ms, which is larger than the entire copy at
most sizes, and charging it to the first cell once made mlrs look 6x slower than
it is (see ``bench_stacking.py``).

Usage
-----
    python3 scripts/bench_stacking_meta.py                    # copy ladder
    python3 scripts/bench_stacking_meta.py --level fit
    python3 scripts/bench_stacking_meta.py --repeat 7 --k 4

Requires numpy + mlrs (built for the backend under test).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

SEED = 42
ARMS = ["numpy", "host", "device"]

#: ``(n_rows, k_blocks, n_features)`` — ``n_features = 0`` means
#: ``passthrough=False``. The ladder walks the copy from "smaller than the FFI
#: crossing" to "big enough for bandwidth to matter".
#:
#: Columns per block are ``--cols`` (default 1, the regressor's shape). A
#: ``StackingClassifier`` contributes ``n_classes`` columns per member under
#: ``predict_proba``, so ``--cols 10`` is the shape a 10-class stack assembles
#: and the one where the scatter has the most to move (STACK-CLF-01).
COPY_LADDER = [
    (1_000, 2, 0),
    (10_000, 2, 0),
    (100_000, 2, 0),
    (1_000_000, 2, 0),
    (100_000, 8, 0),
    (100_000, 2, 32),
    (1_000_000, 2, 32),
    (100_000, 2, 128),
]

#: ``(n_rows, n_features, cv)`` for the end-to-end level.
FIT_LADDER = [
    (100_000, 32, 2),
    (100_000, 32, 5),
    (1_000_000, 32, 2),
]


def blocks_for(n, k, d, dtype, cols=1):
    """``k`` prediction blocks of ``cols`` columns, plus ``X`` when ``d > 0``."""
    rng = np.random.default_rng(SEED)
    preds = [rng.standard_normal((n, cols)).astype(dtype) for _ in range(k)]
    X = rng.standard_normal((n, d)).astype(dtype) if d else None
    return preds, X


def time_copy(arm, n, k, d, dtype, reps, cols=1):
    """Seconds for one meta-matrix assembly on `arm`, min over `reps` in-cell.

    The in-cell minimum is on top of the across-process minimum, not instead of
    it: the copy is short enough at small ``n`` that a single sample is mostly
    scheduler noise.
    """
    import mlrs

    preds, X = blocks_for(n, k, d, dtype, cols)
    given = preds + ([X] if X is not None else [])
    pred_cols = [cols] * k
    passthrough = X is not None
    n_features = d if passthrough else 0

    if arm == "numpy":
        def once():
            return np.hstack(given)
    else:
        def once():
            return mlrs.ensemble._meta_via_rust(
                given, pred_cols, n_features, passthrough, arm
            )

    warm = once()
    if warm is None:
        raise RuntimeError(f"arm {arm!r} declined these blocks")

    best, best_cpu = float("inf"), float("inf")
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        out = once()
        best = min(best, time.perf_counter() - w0)
        best_cpu = min(best_cpu, time.process_time() - c0)
    return best, best_cpu, float(np.asarray(out, dtype=np.float64).sum())


def time_fit(arm, n, d, cv, dtype, reps):
    """Seconds for a whole ``StackingRegressor.fit`` with `arm` selected."""
    import mlrs
    from sklearn.linear_model import LinearRegression, Ridge

    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((n, d)).astype(dtype)
    w = rng.standard_normal(d).astype(dtype)
    y = (X @ w + 0.05 * rng.standard_normal(n)).astype(dtype)

    def build():
        return mlrs.StackingRegressor(
            [("lr", LinearRegression()), ("ridge", Ridge(alpha=1.0))],
            final_estimator=Ridge(alpha=1.0),
            cv=cv,
            passthrough=True,
        )

    build().fit(X[:64], y[:64])

    best, best_cpu = float("inf"), float("inf")
    for _ in range(reps):
        est = build()
        w0, c0 = time.perf_counter(), time.process_time()
        est.fit(X, y)
        best = min(best, time.perf_counter() - w0)
        best_cpu = min(best_cpu, time.process_time() - c0)
    return best, best_cpu, float(np.asarray(est.predict(X[:64]), dtype=np.float64).mean())


def cell(args):
    """One measurement, in this fresh interpreter, as a JSON line."""
    os.environ["MLRS_STACK_META_ENGINE"] = args.arm

    t0 = time.perf_counter()
    import mlrs

    mlrs._load_ext()
    startup_s = time.perf_counter() - t0

    resolved = mlrs._load_ext().stacking_meta_engine()
    # A knob that did not take would make the whole sweep a comparison of numpy
    # against numpy and report it as "no difference". Fail loudly instead.
    if resolved != args.arm:
        print(json.dumps({"error": f"knob resolved to {resolved!r}, wanted {args.arm!r}"}))
        return 1

    dtype = np.float64 if mlrs.backend_supports_f64() else np.float32
    if args.level == "copy":
        seconds, cpu_seconds, checksum = time_copy(
            args.arm, args.n, args.k, args.d, dtype, args.inner, args.cols
        )
    else:
        seconds, cpu_seconds, checksum = time_fit(
            args.arm, args.n, args.d, args.cv, dtype, args.inner
        )

    print(
        json.dumps(
            {
                "arm": args.arm,
                "seconds": seconds,
                "cpu_seconds": cpu_seconds,
                "checksum": checksum,
                "startup_s": startup_s,
                "dtype": np.dtype(dtype).name,
                "loadavg": os.getloadavg()[0],
            }
        )
    )
    return 0


def spawn(arm, level, n, k, d, cv, inner, cols=1):
    """Run one cell in a FRESH interpreter and parse its JSON line."""
    argv = [
        sys.executable, __file__, "--cell",
        "--arm", arm, "--level", level,
        "--n", str(n), "--k", str(k), "--d", str(d),
        "--cv", str(cv), "--inner", str(inner), "--cols", str(cols),
    ]
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            payload = json.loads(line)
            if "error" in line and "seconds" not in payload:
                raise RuntimeError(payload["error"])
            return payload
    return None


def sweep(args):
    ladder = COPY_LADDER if args.level == "copy" else FIT_LADDER
    if args.k:
        ladder = [(n, args.k, d) for n, _, d in ladder] if args.level == "copy" else ladder

    clock = "cpu_seconds" if args.cpu_time else "seconds"
    print(
        f"# level={args.level} repeat={args.repeat} inner={args.inner} "
        f"clock={'CPU time' if args.cpu_time else 'wall'} loadavg={os.getloadavg()[0]:.1f}"
    )
    if not args.cpu_time and os.getloadavg()[0] > 4:
        # A co-tenanted box has INVERTED a verdict on this project before
        # (`mlrs-cpu-bench-separate-processes`). Say so on the output itself,
        # so a ladder read months later carries its own caveat.
        print("# WARNING: loadavg > 4 — wall-clock cells are contended; re-run with --cpu-time")
    header = (
        f"{'config':<28}{'numpy':>12}{'host':>12}{'device':>12}"
        f"{'host/np':>10}{'dev/np':>10}"
    )
    print(header)
    print("-" * len(header))

    startups = []
    for row in ladder:
        if args.level == "copy":
            n, k, d = row
            cv = 0
            label = f"n={n:,} k={k}x{args.cols} d={d}"
        else:
            n, d, cv = row
            k = 2
            label = f"n={n:,} d={d} cv={cv}"

        best = {arm: float("inf") for arm in ARMS}
        checks = {}
        for _ in range(args.repeat):
            for arm in ARMS:  # interleaved, not blocked
                got = spawn(arm, args.level, n, k, d, cv, args.inner, args.cols)
                if got is None:
                    continue
                best[arm] = min(best[arm], got[clock])
                checks[arm] = got["checksum"]
                startups.append(got["startup_s"])

        def fmt(v):
            return f"{v * 1000:>11.3f}m" if v != float("inf") else f"{'n/a':>12}"

        def ratio(v):
            if v == float("inf") or best["numpy"] == float("inf") or v == 0:
                return f"{'n/a':>10}"
            return f"{best['numpy'] / v:>9.2f}x"

        print(
            f"{label:<28}{fmt(best['numpy'])}{fmt(best['host'])}{fmt(best['device'])}"
            f"{ratio(best['host'])}{ratio(best['device'])}"
        )
        # A checksum split means an arm computed something else; the ladder is
        # meaningless then, so say so on the line itself.
        distinct = {round(v, 6) for v in checks.values()}
        if len(distinct) > 1:
            print(f"    !! checksums disagree: {checks}")

    if startups:
        print(
            f"\n# _mlrs load (excluded from every cell): "
            f"min {min(startups) * 1000:.1f} ms, max {max(startups) * 1000:.1f} ms"
        )
    print("# ratios are numpy/arm — above 1.00x means the arm BEAT np.hstack")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arm", choices=ARMS, default="numpy")
    p.add_argument("--level", choices=["copy", "fit"], default="copy")
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--k", type=int, default=0, help="blocks (copy level)")
    p.add_argument(
        "--cols",
        type=int,
        default=1,
        help=(
            "columns per prediction block (copy level). 1 is a regressor's "
            "shape; n_classes is a StackingClassifier's under predict_proba"
        ),
    )
    p.add_argument("--d", type=int, default=0, help="passthrough width; 0 = off")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--repeat", type=int, default=3, help="fresh processes per cell")
    p.add_argument("--inner", type=int, default=5, help="in-process reps per cell")
    p.add_argument(
        "--cpu-time",
        action="store_true",
        help=(
            "report process CPU time instead of wall clock — the metric to use on "
            "a contended box. Note it INFLATES the cpu-backend device arm, which "
            "spends CPU on every cubecl unit thread; on a real GPU it understates "
            "the device arm instead, so wall clock is the metric there."
        ),
    )
    args = p.parse_args()

    if args.cell:
        return cell(args)
    sweep(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
