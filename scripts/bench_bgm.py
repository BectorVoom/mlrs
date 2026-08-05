#!/usr/bin/env python3
"""sklearn `BayesianGaussianMixture` wall-clock baseline for the MIX-02 probe.

The counterpart of ``crates/mlrs-algos/tests/bayesian_mixture_perf_test.rs``.
Both sides build the SAME design from the same counter-based splitmix64 stream
(``make_blobs`` below is byte-identical to the Rust one, and to
``bench_gmm.py``'s — the two mixture estimators are benchmarked on one design),
fit with the same hyperparameters, and print the same table, so the two columns
can be divided.

What is measured, and why each ladder exists:

``fit``
    The whole variational fit including the initialization. This is the number a
    user experiences.
``predict`` / ``score_samples``
    Inference on a fitted model — the E-step alone, with no M-step and no
    initialization, so it isolates the Mahalanobis kernel plus the expected-log
    corrections.

The ladders sweep the parameters that change what the fit COSTS:

* ``covariance_type`` — ``full``/``tied`` are ``O(n·k·d²)``, ``diag``/
  ``spherical`` ``O(n·k·d)``. The single biggest lever.
* ``n_features`` — quadratic for ``full``/``tied``; also the axis the ``O(d)``
  digamma sum per component rides on.
* ``n_components`` — linear in the sweep but multiplies the quadratic term, and
  it is the parameter users of THIS estimator deliberately over-provision
  (the prior prunes what is not needed).
* ``init_params`` — ``kmeans`` runs a whole Lloyd fit before the loop starts;
  the sparse routes run none but need far more iterations.
* ``weight_concentration_prior_type`` — the one string parameter whose cost is
  not obvious. The stick-breaking branch adds a ``k``-length scan plus ``k``
  ``betaln`` evaluations per iteration against an ``O(n·k·d²)`` sweep; the
  ladder exists to SHOW that is free rather than to assume it, and repeats at
  ``k=32`` where the ``k``-scaling term is four times longer.
* ``n_init`` — a pure multiplier, and the cheapest way to check the restart
  loop leaks no per-restart setup.
* the ``vb`` ladder — ``tol=0`` with a fixed ``max_iter``, which removes both
  the initialization and the convergence RATE from the timer. This is the
  ladder that attributes a win to the loop itself; everywhere else a difference
  can come from converging in fewer iterations, which is why ``n_iter`` is
  printed on both sides.

Run:
    python3 scripts/bench_bgm.py            # the default ladders
    python3 scripts/bench_bgm.py --reps 7   # more repeats on a noisy box

Timing methodology (learned the hard way — [[mlrs-cpu-bench-separate-processes]],
[[mlrs-bench-verify-knob-is-live]]): min-of-N repeats after one discarded warmup,
and BOTH wall clock and process CPU time are reported. A box with a co-tenant
job has inverted an mlrs-vs-sklearn verdict before; if `cpu/wall` is far above
1.0 for one engine and near 1.0 for the other, the wall-clock ratio is measuring
thread count, not efficiency.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

# --- the shared dataset generator (byte-identical to the Rust probe) -------- #

_GOLDEN = 0x9E3779B97F4A7C15


def _stream(seed: int, count: int) -> np.ndarray:
    """The first `count` outputs of the Rust probe's counter-based splitmix64.

    Vectorized, and it has to be: the largest rung draws 6.4M values, which a
    scalar Python loop turns into tens of seconds PER RUNG. That is safe here
    precisely because the generator is COUNTER-based: its state is the closed
    form `seed + (i+1)*GOLDEN mod 2**64`, so the whole sequence is addressable
    without iterating. The Rust probe's `shared_design_matches_the_python_probe`
    test pins the two against each other so this cannot drift.
    """
    i = np.arange(1, count + 1, dtype=np.uint64)
    z = (np.uint64(seed) + i * np.uint64(_GOLDEN)).astype(np.uint64)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def make_blobs(n: int, d: int, k: int, seed: int) -> np.ndarray:
    """`k` well-separated isotropic blobs — the same array the Rust probe builds.

    Consumes the stream in the SAME order the Rust `make_blobs` does: `k*d`
    center coordinates first, then two uniforms per element of the design (the
    Box-Muller pair), row-major.
    """
    n_center = k * d
    u = _stream(seed, n_center + 2 * n * d)
    centers = ((u[:n_center] * 2.0 - 1.0) * 10.0).reshape(k, d)
    rest = u[n_center:].reshape(n, d, 2)
    u1 = np.maximum(rest[:, :, 0], 2.2250738585072014e-308)
    u2 = rest[:, :, 1]
    g = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return centers[np.arange(n) % k] + g


# --- timing ---------------------------------------------------------------- #


def _time(fn, reps: int) -> tuple[float, float]:
    """min-of-`reps` (wall_ms, cpu_ms), after one discarded warmup."""
    fn()
    best_wall = float("inf")
    best_cpu = float("inf")
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        fn()
        w1, c1 = time.perf_counter(), time.process_time()
        best_wall = min(best_wall, (w1 - w0) * 1e3)
        best_cpu = min(best_cpu, (c1 - c0) * 1e3)
    return best_wall, best_cpu


DP = "dirichlet_process"
DD = "dirichlet_distribution"


def main() -> None:
    from sklearn.mixture import BayesianGaussianMixture

    # Every `vb` rung deliberately runs to `max_iter` without converging, which
    # sklearn reports as a ConvergenceWarning per fit. That is the POINT of the
    # ladder, so the warning is noise here.
    warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # (label, n, d, k, covariance_type, init_params, prior_type, max_iter,
    #  n_init, tol)
    rungs = []
    # Ladder 1: covariance_type at a fixed, realistic geometry.
    for cov in ("full", "tied", "diag", "spherical"):
        rungs.append((f"cov={cov}", 20000, 16, 8, cov, "kmeans", DP, 100, 1, 1e-3))
    # Ladder 2: n_features (the quadratic axis for full/tied).
    for d in (4, 16, 64, 128):
        rungs.append((f"d={d}", 20000, d, 8, "full", "kmeans", DP, 100, 1, 1e-3))
    # Ladder 3: n_components.
    for k in (2, 8, 32):
        rungs.append((f"k={k}", 20000, 16, k, "full", "kmeans", DP, 100, 1, 1e-3))
    # Ladder 4: n_samples.
    for n in (2000, 20000, 200000):
        rungs.append((f"n={n}", n, 16, 8, "full", "kmeans", DP, 100, 1, 1e-3))
    # Ladder 5: init_params.
    for init in ("kmeans", "k-means++", "random", "random_from_data"):
        rungs.append((f"init={init}", 20000, 16, 8, "full", init, DP, 100, 1, 1e-3))
    # Ladder 6: weight_concentration_prior_type, at two `k` (the axis its extra
    # work scales on).
    rungs.append(("prior=dp", 20000, 16, 8, "full", "kmeans", DP, 100, 1, 1e-3))
    rungs.append(("prior=dd", 20000, 16, 8, "full", "kmeans", DD, 100, 1, 1e-3))
    rungs.append(("prior=dp k=32", 20000, 16, 32, "full", "kmeans", DP, 100, 1, 1e-3))
    rungs.append(("prior=dd k=32", 20000, 16, 32, "full", "kmeans", DD, 100, 1, 1e-3))
    # Ladder 7: n_init (the restart multiplier).
    for ni in (1, 3):
        rungs.append((f"n_init={ni}", 20000, 16, 8, "full", "kmeans", DP, 100, ni, 1e-3))
    # Ladder 8: the variational loop in ISOLATION.
    for cov in ("full", "tied", "diag", "spherical"):
        rungs.append((f"vb cov={cov}", 20000, 16, 8, cov, "random", DP, 50, 1, 0.0))
    for cov in ("full", "tied"):
        rungs.append((f"vb {cov} d=64", 20000, 64, 8, cov, "random", DP, 50, 1, 0.0))
    rungs.append(("vb full k=32", 20000, 16, 32, "full", "random", DP, 50, 1, 0.0))

    print(
        f"{'rung':<26} {'fit ms':>10} {'cpu ms':>10} "
        f"{'pred ms':>10} {'score ms':>10} {'n_iter':>7}"
    )
    print("-" * 78)
    for label, n, d, k, cov, init, prior, max_iter, n_init, tol in rungs:
        x = make_blobs(n, d, k, args.seed)

        def do_fit():
            return BayesianGaussianMixture(
                n_components=k,
                covariance_type=cov,
                init_params=init,
                weight_concentration_prior_type=prior,
                max_iter=max_iter,
                n_init=n_init,
                tol=tol,
                reg_covar=1e-6,
                random_state=0,
            ).fit(x)

        fit_wall, fit_cpu = _time(do_fit, args.reps)
        est = do_fit()
        pred_wall, _ = _time(lambda: est.predict(x), args.reps)
        score_wall, _ = _time(lambda: est.score_samples(x), args.reps)
        print(
            f"{label:<26} {fit_wall:>10.2f} {fit_cpu:>10.2f} "
            f"{pred_wall:>10.2f} {score_wall:>10.2f} {est.n_iter_:>7}"
        )


if __name__ == "__main__":
    main()
