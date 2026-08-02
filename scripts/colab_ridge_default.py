"""Colab driver for the `Ridge()` (positive=False) CUDA perf probe
(RIDGE-DEFAULT-CUDA).

Runtime must be **GPU** (Runtime → Change runtime type → T4 GPU).

Sibling of ``colab_ridge_positive.py``; that one measured the `positive=True`
arm, this one measures the DEFAULT — `solver='cholesky'`, which is what
`Ridge()` with no arguments runs. Two things changed under it and both need a
T4 number:

  * the fit is now fully device-resident (fused centering, `α` added inside the
    Cholesky kernel, the intercept dot on-device), so nothing crosses the bus
    between the upload and `coef_`;
  * `d > 64` runs at all — the shared-memory factorization cap used to reject it
    with `NotSquare`, and `d = 128`/`256` is the only regime where a device fit
    can beat a host one.

One-time setup, then two cells:

  1. In Colab's left sidebar click the 🔑 (Secrets) icon → "Add new secret".
     Name it exactly ``GH_TOKEN``. Value: a GitHub token with write access to
     BectorVoom/mlrs (a fine-grained token with "Contents: Read and write" on
     that one repo is enough). Toggle "Notebook access" on.
  2. CELL 1 — installs Rust, clones the work branch, first build (~8 min).
  3. CELL 2 — runs the measurements and PUSHES the log to the
     ``results/ridge-default-t4`` branch. Nothing to copy by hand.

The log carries the source commit SHA it was produced from. That is
load-bearing, not decoration: a stale checkout silently produces
plausible-looking numbers for the wrong code, and the SHA is what makes that
detectable instead of misleading.

The probe prints TWO ladders. The first is what a Python caller gets (whichever
arm ``host_fit_applicable`` picks). The second forces BOTH arms at every rung on
the same machine, which is the only comparison that answers "does the gpu beat
the cpu" — the T4's own host arm is a 2-vCPU Xeon and is roughly 4× slower than
the 16-thread dev box, so the Colab ratio flatters the GPU and the local CPU
ladder is the baseline to quote alongside it.

    # the local CPU baseline the T4 numbers are divided against
    cargo test -p mlrs-algos --release --features cpu \
      --test ridge_default_perf_test -- --ignored --nocapture
"""

BRANCH = "perf/ridge-positive-cuda"
RESULTS = "results/ridge-default-t4"
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
   "--test ridge_default_perf_test --no-run")
sh("cd /content/mlrs && cargo test -p mlrs-algos --release --features cuda "
   "--test ridge_test --test ridge_params_test --test ridge_host_fit_test --no-run")
sh("cd /content/mlrs && cargo test -p mlrs-backend --release --features cuda "
   "--test cholesky_test --no-run")
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

# --- token: Colab Secrets first, hand entry only as a fallback -------------
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
cap("mlrs Ridge() [positive=False, solver=cholesky] CUDA probe")
cap(f"utc      : {time.strftime('%%Y-%%m-%%dT%%H:%%M:%%SZ', time.gmtime())}")
cap(f"commit   : {sha}")
cap(f"          {desc}")
cap(f"gpu      : {gpu}")
cap(f"cpu      : {cpu} ({out_of('nproc')} threads)")
cap("=" * 72)

LADDER = ("cd /content/mlrs && MLRS_RIDGE_REPS=9 {env} cargo test -p mlrs-algos "
          "--release --features cuda --test ridge_default_perf_test "
          "-- --ignored --nocapture")

# A. CORRECTNESS FIRST. A perf number from a wrong kernel is worse than no
#    number, and this is the first cuda run of both the wide Cholesky arm and
#    the fused-centering default path.
cap("\n" + "-" * 72)
cap("A. correctness on the T4 (wide Cholesky arm, fused default path)")
cap("-" * 72)
sh("cd /content/mlrs && cargo test -p mlrs-backend --release --features cuda "
   "--test cholesky_test -- --nocapture", echo=False)
sh("cd /content/mlrs && cargo test -p mlrs-algos --release --features cuda "
   "--test ridge_test --test ridge_params_test --test ridge_host_fit_test "
   "-- --nocapture", echo=False)

# B. The headline ladder plus the forced-arm A/B, default dispatch.
cap("\n" + "-" * 72)
cap("B. whole-fit ladder — default dispatch")
cap("-" * 72)
sh(LADDER.format(env=""), echo=False)

# C. Per-phase attribution. The laps are drained (a one-element blocking
#    read-back, NOT client.sync(), which returns a future and measures enqueue
#    time), so `preprocess`/`solve`/`tail` split honestly — at the cost of
#    forbidding overlap, which is why the ladder above is the number to quote.
cap("\n" + "-" * 72)
cap("C. per-phase attribution (RIDGE_PROFILE=1, drained laps)")
cap("-" * 72)
sh(LADDER.format(env="RIDGE_PROFILE=1 MLRS_RIDGE_REPS=1"), echo=False)

# D. The Gram formation is an ADAPTER property — the shared-staged/tiled/blocked
#    crossover measured on the local iGPU does not transfer, and the default
#    path now runs the FUSED (centering-in-kernel) variant of all three, which
#    has never been swept on a T4.
for label, env in [("D1. register-tiled Gram forced", "LR_GRAM_TILED=1"),
                   ("D2. blocked Gram forced",        "LR_GRAM_BLOCKED=1")]:
    cap("\n" + "-" * 72)
    cap(label)
    cap("-" * 72)
    sh(LADDER.format(env=env), echo=False)

# E. Does the narrow (shared-memory, unit-0-serial) factorization still win
#    below d=64, or does the wide one? The dispatch threshold is MAX_DIM today
#    because that is where the narrow kernel STOPS WORKING, not because anyone
#    measured a crossover.
cap("\n" + "-" * 72)
cap("E. wide Cholesky arm forced at every order")
cap("-" * 72)
sh(LADDER.format(env="MLRS_CHOLESKY_WIDE=1"), echo=False)

# --- push the log ----------------------------------------------------------
stamp = time.strftime("%%Y%%m%%dT%%H%%M%%SZ", time.gmtime())
name  = f"ridge_default_t4_{stamp}_{sha[:8]}.log"
os.makedirs("/content/out", exist_ok=True)
with open(f"/content/out/{name}", "w") as f:
    f.write(log.getvalue())

push_url = f"https://x-access-token:{TOKEN}@github.com/BectorVoom/mlrs.git"
os.makedirs("/content/results", exist_ok=True)
# A dedicated results branch, never the code branch: the two are written by
# different machines and would otherwise race.
for c in [["git", "init", "-q", "/content/results"],
          ["git", "-C", "/content/results", "config", "user.email", "colab@local"],
          ["git", "-C", "/content/results", "config", "user.name",  "colab-t4"]]:
    subprocess.run(c, check=False, capture_output=True)
subprocess.run(["git", "-C", "/content/results", "remote", "remove", "origin"],
               capture_output=True)
subprocess.run(["git", "-C", "/content/results", "remote", "add", "origin", push_url],
               capture_output=True)          # never echoed — carries the token
subprocess.run(["git", "-C", "/content/results", "fetch", "--depth", "1",
                "origin", RESULTS], capture_output=True)
subprocess.run(["git", "-C", "/content/results", "checkout", "-B", RESULTS,
                f"origin/{RESULTS}"], capture_output=True)
subprocess.run(["cp", f"/content/out/{name}", "/content/results/"], check=True)
subprocess.run(["git", "-C", "/content/results", "add", "-A"], check=True)
subprocess.run(["git", "-C", "/content/results", "commit", "-q", "-m",
                f"T4 ridge(default) probe @ {sha[:8]}"], capture_output=True)
p = subprocess.run(["git", "-C", "/content/results", "push", "-u", "origin", RESULTS],
                   capture_output=True, text=True)
msg = (p.stdout + p.stderr).replace(TOKEN, "***")
print(msg[-3000:])
if p.returncode:
    print(f"\n!! PUSH FAILED ({p.returncode}) — check the token has "
          f"'Contents: Read and write' on BectorVoom/mlrs")
else:
    print(f"\nPUSHED  branch={RESULTS}  file={name}")
"""
    % {"branch": BRANCH, "results": RESULTS}
)

if __name__ == "__main__":
    print(CELL_1)
    print("\n# ---------------------------------------------------------\n")
    print(CELL_2)
