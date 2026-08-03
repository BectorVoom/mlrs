#!/usr/bin/env python3
"""SpectralEmbedding **fit** wall-clock: mlrs (cpu backend) vs scikit-learn.

The SPECTRAL-PERF-CPU probe. Both engines run the same three stages — build the
affinity, form the symmetric-normalized Laplacian, extract its smallest
`n_components + 1` eigenpairs — and the two agree on the eigenspace to ~1e-15,
so this is a like-for-like comparison at library defaults. What differs is HOW
each stage is evaluated:

    sklearn   `kneighbors_graph` (or `rbf_kernel`), then ARPACK in SHIFT-INVERT
              mode (`eigsh(L, k, sigma=-1e-5, which="LM", tol=0)`), which
              factorizes `L - sigma*I` before it can iterate
    mlrs      a KD-tree-routed parallel host kNN scan, the affinity kept SPARSE
              in CSR, and a thick-restart Lanczos whose only contact with the
              matrix is a matvec — so there is no factorization at all

The expected shape is therefore that mlrs' margin GROWS with `n_samples`
(sklearn pays a sparse factorization that scales worse than a matvec) and that
the `rbf` rungs favor mlrs further still, because a dense affinity makes
sklearn's factorization dense while mlrs' matvec stays a plain `O(n^2)` pass.

    .venv/bin/python scripts/bench_spectral_embedding_cpu.py [--reps 5] [--check]
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
worker threads while sklearn's ARPACK stage is single-threaded, so `cpu (s)` is
a deliberately pessimistic reading of the mlrs column rather than a
like-for-like one.

`--check` prints the sign-aligned max|Δembedding| against sklearn AND the
principal-angle subspace mismatch, so a rung that wins on time but drifts on
numerics is visible in the same line.

**A speed number here is only meaningful if the graph is CONNECTED.** A
disconnected affinity graph has one zero Laplacian eigenvalue per component, so
the requested eigenvectors span a fully degenerate null space and are defined
only up to an arbitrary rotation within it — sklearn and mlrs will then return
different (both correct) bases and the elementwise `--check` number is
meaningless. The rung line reports the component count and flags anything above
1, which is the same condition sklearn warns about. The default ladder uses
uniform-random data precisely because it keeps the kNN graph connected; a
`make_blobs`-style ladder does NOT (well-separated clusters are exactly the case
that disconnects, which is the opposite of the usual fixture intuition).
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

# (rows, features, n_components, affinity) of the timed fit. The ladder walks
# `n` at fixed `d` (isolating the eigensolver, which is where the old dense
# `O(n^3)` path died), `d` at fixed `n` (isolating the kNN scan), and both
# affinities (sparse CSR matvec vs dense).
CONFIGS = [
    (500, 8, 2, "nearest_neighbors"),
    (2_000, 16, 2, "nearest_neighbors"),
    (5_000, 16, 2, "nearest_neighbors"),
    (10_000, 16, 2, "nearest_neighbors"),
    (5_000, 64, 4, "nearest_neighbors"),
    (500, 8, 2, "rbf"),
    (2_000, 16, 4, "rbf"),
]


def make_design(n, d, seed=0):
    """Uniform-random rows.

    Deliberately NOT `make_blobs`: separated clusters disconnect the kNN graph,
    which makes the embedding degenerate and the accuracy check vacuous (see the
    module docstring).
    """
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((n, d)))


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


def sign_aligned_max_err(a, b):
    """max|a - b| after aligning each column's sign.

    An eigenvector is defined only up to a global sign, and both engines apply
    sklearn's deterministic flip — but the flip keys on the argmax entry, so a
    near-tie in |value| can legitimately land on opposite signs. Aligning here
    keeps the number about the SUBSPACE rather than about that tie-break.
    """
    err = 0.0
    for c in range(a.shape[1]):
        u, v = a[:, c], b[:, c]
        if float(np.dot(u, v)) < 0.0:
            v = -v
        err = max(err, float(np.max(np.abs(u - v))))
    return err


def subspace_mismatch(a, b):
    """`1 - sigma_min` of the principal angles between the two column spaces.

    This is the number that stays meaningful when the spectrum is degenerate:
    the individual eigenvectors are then arbitrary within the invariant
    subspace, but the subspace itself is not.
    """
    qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64))
    qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64))
    return 1.0 - float(np.linalg.svd(qa.T @ qb, compute_uv=False).min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--engine", default="both", choices=["both", "mlrs", "sklearn"])
    ap.add_argument("--n-neighbors", type=int, default=0,
                    help="0 (default) leaves n_neighbors=None, which sklearn "
                         "resolves to max(n_samples // 10, 1)")
    ap.add_argument("--check", action="store_true",
                    help="print sign-aligned max|Δembedding| and the subspace "
                         "mismatch vs sklearn")
    ap.add_argument("--configs", default="",
                    help="comma-separated n:d:k:affinity")
    ap.add_argument("--schedule", default="interleaved",
                    choices=["interleaved", "blocked"],
                    help="alternate engines rep by rep (default) or run each "
                         "engine's reps back to back")
    args = ap.parse_args()

    import mlrs
    from mlrs.cluster import SpectralEmbedding as MlrsEst
    from sklearn.manifold import SpectralEmbedding as SkEst

    configs = CONFIGS
    if args.configs:
        configs = []
        for c in args.configs.split(","):
            n, d, k, aff = c.split(":")
            configs.append((int(n), int(d), int(k), aff))

    dt = np.float32 if args.dtype == "float32" else np.float64
    nn = args.n_neighbors or None
    print(f"mlrs {mlrs.__name__} | n_neighbors={nn if nn else 'None (n//10)'} "
          f"dtype={args.dtype} reps={args.reps} schedule={args.schedule}")
    header = (f"{'n':>7} {'d':>4} {'k':>3} {'affinity':>18} | {'engine':>8} "
              f"{'fit (s)':>10} {'cpu (s)':>10} {'first (s)':>10}")
    print(header)
    print("-" * len(header))

    engines = [e for e in ("mlrs", "sklearn") if args.engine in ("both", e)]

    for n, d, k, aff in configs:
        x = np.ascontiguousarray(make_design(n, d).astype(dt))
        common = dict(n_components=k, affinity=aff, n_neighbors=nn, random_state=0)

        def fit_mlrs():
            m = MlrsEst(**common)
            m.fit(x)
            return m

        def fit_sk():
            with warnings.catch_warnings():
                # The "not fully connected" warning is advisory in sklearn and
                # changes nothing; the rung line reports the component count
                # explicitly instead of letting a warning scroll past.
                warnings.simplefilter("ignore")
                m = SkEst(**common)
                m.fit(x)
                return m

        fits = {"mlrs": fit_mlrs, "sklearn": fit_sk}
        samples = {e: Samples() for e in engines}
        failed = {}

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
            print(f"{n:>7} {d:>4} {k:>3} {aff:>18} | {eng:>8}  FAILED: {msg}")

        ok = [e for e in engines if e not in failed]
        for eng in ok:
            s = samples[eng]
            print(f"{n:>7} {d:>4} {k:>3} {aff:>18} | {eng:>8} "
                  f"{s.best:>10.4f} {s.best_cpu:>10.4f} {s.first:>10.4f}")
        if len(ok) == 2:
            wall_x = samples["sklearn"].best / samples["mlrs"].best
            cpu_x = samples["sklearn"].best_cpu / samples["mlrs"].best_cpu
            note = f"{wall_x:.2f}x wall / {cpu_x:.2f}x cpu vs sklearn"
            mm, sm = samples["mlrs"].model, samples["sklearn"].model
            ncc = int(mm._mlrs_obj.n_graph_components())
            if ncc > 1:
                note += (f"  [!] graph has {ncc} components — the kept eigenspace "
                         f"is degenerate, so any elementwise check below is "
                         f"meaningless (compare the subspace instead)")
            if args.check:
                a = np.asarray(mm.embedding_, dtype=np.float64)
                b = np.asarray(sm.embedding_, dtype=np.float64)
                note += (
                    f" | max|Δemb| = {sign_aligned_max_err(a, b):.3e}"
                    f" | subspace = {subspace_mismatch(a, b):.3e}"
                )
                if aff == "nearest_neighbors":
                    note += f" | n_neighbors_ {mm.n_neighbors_}/{sm.n_neighbors_}"
                else:
                    note += f" | gamma_ {mm.gamma_:.6g}/{sm.gamma_:.6g}"
            print(f"{'':>7} {'':>4} {'':>3} {'':>18} | {note}")
        print()


if __name__ == "__main__":
    main()
