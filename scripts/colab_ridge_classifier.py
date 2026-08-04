"""Colab driver for the `RidgeClassifier` on-device CUDA probe (RIDGECLF-CUDA).

Runtime must be **GPU** (Runtime → Change runtime type → T4 GPU).

Sibling of ``colab_ridge_default.py`` / ``colab_ridge_positive.py``. Those
measured the single-target `Ridge` fit; this one measures the classifier, whose
arithmetic differs from it in exactly the way that decides whether a GPU can
win:

  * **fit** is ``O(n·d² + n·d·k)`` over the same ``n·d`` transfer — the shared
    Gram is formed ONCE and all ``k`` ``Xᵀy`` columns ride along, so a
    26-class fit does ~2× the arithmetic of a plain `Ridge` at ``d = 64`` for
    the same bytes moved;
  * **predict** is ``O(m·d·k)`` over the same ``m·d`` transfer, and the fused
    classify kernel brings back ``m`` `i32` labels rather than ``m·k`` floats.
    That matters because `Ridge`'s single-target device predict measured
    **10–23× slower** than this crate's own host matvec on a P100 — `predict`
    is the one linear-model op whose compute-to-transfer ratio a GPU cannot
    improve, and ``k`` targets is the only thing that changes it.

One-time setup, then two cells:

  1. In Colab's left sidebar click the 🔑 (Secrets) icon → "Add new secret".
     Name it exactly ``GH_TOKEN``. Value: a GitHub token with write access to
     BectorVoom/mlrs ("Contents: Read and write" on that one repo is enough).
     Toggle "Notebook access" on.
  2. CELL 1 — installs Rust, clones the work branch, first build (~8 min).
  3. CELL 2 — runs the measurements and PUSHES the log to the
     ``results/ridge-classifier-t4`` branch. Nothing to copy by hand.

The log carries the source commit SHA it was produced from. That is
load-bearing, not decoration: a stale checkout silently produces
plausible-looking numbers for the wrong code.

**Read the ratio carefully.** The ``host`` column is the code the **cpu**
backend runs, executing on the Colab VM's 2-vCPU Xeon @2GHz — roughly 4× slower
than the 16-thread dev box (measured in RIDGE-DEFAULT-CUDA). So a ratio here is
"T4 vs THIS VM's CPU", and the local ladder below is the baseline it has to be
quoted against:

    cargo test -p mlrs-algos --release --features cpu \
      --test ridge_classifier_cuda_perf_test -- --ignored --nocapture
"""

BRANCH = "worktree-ridge-classifier-cuda"
RESULTS = "results/ridge-classifier-t4"
REPO = "https://github.com/BectorVoom/mlrs.git"

CELL_1 = (
    r"""
#@title CELL 1 — one-time setup (Rust + clone + first build)
import os, subprocess

BRANCH = "%(branch)s"
REPO   = "%(repo)s"

def sh(cmd, check=True):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-8000:])
    if check and r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")

sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
sh("nvcc --version | tail -2", check=False)
sh("nproc && lscpu | grep -E 'Model name|^CPU\\(s\\):'")

if not os.path.exists("/root/.cargo/bin/cargo"):
    sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
       "| sh -s -- -y --profile minimal --default-toolchain stable")
os.environ["PATH"] = "/root/.cargo/bin:" + os.environ["PATH"]
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
sh("cargo --version && rustc --version")

if not os.path.exists("/content/mlrs/.git"):
    sh(f"git clone --depth 50 -b {BRANCH} {REPO} /content/mlrs")
sh(f"cd /content/mlrs && git fetch origin {BRANCH} && "
   f"git checkout -B {BRANCH} origin/{BRANCH} && git log --oneline -1")

# Build every binary CELL 2 runs, so CELL 2 is pure measurement.
sh("cd /content/mlrs && cargo test -p mlrs-algos --release --features cuda "
   "--test ridge_classifier_cuda_perf_test --no-run")
sh("cd /content/mlrs && cargo test -p mlrs-algos --release --features cuda "
   "--test ridge_classifier_test --test ridge_classifier_device_test --no-run")
sh("cd /content/mlrs && cargo test -p mlrs-backend --release --features cuda "
   "--test gram_test --no-run", check=False)
print("SETUP OK — run CELL 2")
"""
    % {"branch": BRANCH, "repo": REPO}
)

CELL_2 = (
    r"""
#@title CELL 2 — measure on the T4 and push the log (run after every push)
import os, subprocess, io, time

BRANCH  = "%(branch)s"
RESULTS = "%(results)s"
os.environ["PATH"] = "/root/.cargo/bin:" + os.environ["PATH"]
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")

TOKEN = None
try:
    from google.colab import userdata
    TOKEN = userdata.get("GH_TOKEN")
except Exception as e:
    print(f"(Colab secret GH_TOKEN unavailable: {e})")
if not TOKEN:
    import getpass
    TOKEN = getpass.getpass("GitHub token (write access to BectorVoom/mlrs): ")
TOKEN = TOKEN.strip()

log = io.StringIO()
def cap(s=""):
    print(s)
    log.write(s + "\n")

def sh(cmd, echo=True):
    if echo:
        cap(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.replace(TOKEN, "***") if TOKEN else r.stdout
    cap(out[-24000:])
    return r.returncode

def out_of(cmd):
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return r.stdout.strip()

sh(f"cd /content/mlrs && git fetch origin {BRANCH} && "
   f"git checkout -B {BRANCH} origin/{BRANCH}")

sha  = out_of("cd /content/mlrs && git rev-parse HEAD")
desc = out_of("cd /content/mlrs && git log --oneline -1")
gpu  = out_of("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
cpu  = out_of("lscpu | grep 'Model name' | head -1")
cap("=" * 72)
cap("mlrs RidgeClassifier on-device (fit + predict) CUDA probe")
cap(f"utc      : {time.strftime('%%Y-%%m-%%dT%%H:%%M:%%SZ', time.gmtime())}")
cap(f"commit   : {sha}")
cap(f"          {desc}")
cap(f"gpu      : {gpu}")
cap(f"cpu      : {cpu} ({out_of('nproc')} threads)")
cap("=" * 72)

LADDER = ("cd /content/mlrs && MLRS_RIDGECLF_REPS=9 {env} cargo test -p mlrs-algos "
          "--release --features cuda --test ridge_classifier_cuda_perf_test "
          "-- --ignored --nocapture")

# A. CORRECTNESS FIRST. A perf number from a wrong kernel is worse than no
#    number, and this is the first cuda run of the fused multi-target Gram, the
#    multi-RHS Cholesky at k > 1, and the fused classify kernel.
cap("\n" + "-" * 72)
cap("A. correctness on the T4 (sklearn oracle + device/host arm equivalence)")
cap("-" * 72)
sh("cd /content/mlrs && cargo test -p mlrs-algos --release --features cuda "
   "--test ridge_classifier_test --test ridge_classifier_device_test "
   "-- --nocapture", echo=False)

# B. The headline: both fit arms and both predict arms forced at every rung on
#    the same machine — the only comparison that answers "does the gpu beat the
#    cpu path".
cap("\n" + "-" * 72)
cap("B. fit + predict ladders, both arms forced")
cap("-" * 72)
sh(LADDER.format(env=""), echo=False)

# C. The Gram formation is an ADAPTER property (the shared-staged / register-
#    tiled / blocked crossover measured on the local iGPU does not transfer),
#    and the multi-target path pairs it with a SECOND pass (`xty_multi_blocked`)
#    that has never been swept anywhere.
for label, env in [("C1. register-tiled Gram forced", "LR_GRAM_TILED=1"),
                   ("C2. blocked Gram forced",        "LR_GRAM_BLOCKED=1")]:
    cap("\n" + "-" * 72)
    cap(label)
    cap("-" * 72)
    sh(LADDER.format(env=env), echo=False)

# D. The wide (global-memory, column-distributed) Cholesky arm below its cap.
#    RIDGE-DEFAULT-CUDA measured 1.87x at 10k x 64 for rhs=1 and did not ship it
#    for want of a second adapter; a multi-RHS solve spends k times as long in
#    the triangular solves, so the case is stronger here if it holds at all.
cap("\n" + "-" * 72)
cap("D. wide Cholesky arm forced at every order")
cap("-" * 72)
sh(LADDER.format(env="MLRS_CHOLESKY_WIDE=1"), echo=False)

# E. The predict dispatch gate itself. `device_predict_applicable` ships a
#    threshold on n_targets; forcing both directions at every rung is what
#    places it (and what would catch it being placed wrong).
for label, env in [("E1. predict forced to the DEVICE arm", "MLRS_RIDGECLF_PREDICT_DEVICE=1"),
                   ("E2. predict forced to the HOST arm",   "MLRS_RIDGECLF_PREDICT_DEVICE=0")]:
    cap("\n" + "-" * 72)
    cap(label)
    cap("-" * 72)
    sh(LADDER.format(env=env), echo=False)

# --- push the log to its own branch so nothing has to be pasted back ---------
name = f"ridge_classifier_t4_{sha[:7]}.log"
os.makedirs("/content/out", exist_ok=True)
with open(f"/content/out/{name}", "w") as f:
    f.write(log.getvalue())

remote = f"https://x-access-token:{TOKEN}@github.com/BectorVoom/mlrs.git"
subprocess.run(
    f"cd /content/mlrs && git config user.email colab@local && "
    f"git config user.name colab && "
    f"git checkout -B {RESULTS} && mkdir -p results && "
    f"cp /content/out/{name} results/{name} && "
    f"git add results/{name} && git commit -m 'RidgeClassifier T4 log {sha[:7]}' && "
    f"git push -f {remote} {RESULTS}",
    shell=True, text=True)
print(f"\npushed results/{name} to branch {RESULTS}")
print("fetch it with:  git fetch origin "
      "'refs/heads/results/*:refs/remotes/origin/results/*'")
"""
    % {"branch": BRANCH, "results": RESULTS}
)


def main() -> None:
    print(__doc__)
    print("=" * 72)
    print("CELL 1")
    print("=" * 72)
    print(CELL_1)
    print("=" * 72)
    print("CELL 2")
    print("=" * 72)
    print(CELL_2)


if __name__ == "__main__":
    main()
