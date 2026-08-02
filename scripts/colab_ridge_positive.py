"""Colab driver for the `Ridge(positive=True)` CUDA perf probe (RIDGE-POS-CUDA).

Runtime must be **GPU** (Runtime → Change runtime type → T4 GPU).

One-time setup, then two cells:

  1. In Colab's left sidebar click the 🔑 (Secrets) icon → "Add new secret".
     Name it exactly ``GH_TOKEN``. Value: a GitHub token with write access to
     BectorVoom/mlrs (a fine-grained token with "Contents: Read and write" on
     that one repo is enough). Toggle "Notebook access" on.
     This is stored by Colab, not in the notebook, and survives new sessions —
     so it is pasted once, ever.
  2. CELL 1 — installs Rust, clones the work branch, first build (~5 min).
  3. CELL 2 — runs the four measurements and PUSHES the log to the
     ``results/ridge-positive-t4`` branch. Nothing to copy by hand.

The log carries the source commit SHA it was produced from. That is
load-bearing, not decoration: a stale checkout silently produces
plausible-looking numbers for the wrong code, and the SHA is what makes that
detectable instead of misleading.

The local CPU baseline these numbers are divided against comes from the same
probe:

    cargo test -p mlrs-algos --release --features cpu \
      --test ridge_positive_perf_test -- --ignored --nocapture
"""

CELL_1 = r"""
#@title CELL 1 — one-time setup (Rust + clone + first build)
import os, subprocess

BRANCH = "perf/ridge-positive-cuda"
REPO   = "https://github.com/BectorVoom/mlrs.git"

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

# Build both test binaries now, so CELL 2 is pure measurement.
sh("cd /content/mlrs && cargo test -p mlrs-algos --release --features cuda "
   "--test ridge_positive_perf_test --no-run")
sh("cd /content/mlrs && cargo test -p mlrs-backend --release --features cuda "
   "--test gram_perf_test --no-run")
print("SETUP OK — run CELL 2")
"""

CELL_2 = r"""
#@title CELL 2 — measure on the T4 and push the log (run after every push)
import os, subprocess, io, time

BRANCH  = "perf/ridge-positive-cuda"
RESULTS = "results/ridge-positive-t4"
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
    cap(out[-20000:])
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
cap(f"mlrs Ridge(positive=True) CUDA probe")
cap(f"utc      : {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
cap(f"commit   : {sha}")
cap(f"          {desc}")
cap(f"gpu      : {gpu}")
cap(f"cpu      : {cpu} ({out_of('nproc')} threads)")
cap("=" * 72)

RIDGE = ("cd /content/mlrs && MLRS_POS_REPS=9 {env} cargo test -p mlrs-algos --release "
         "--features cuda --test ridge_positive_perf_test -- --ignored --nocapture")
GRAM  = ("cd /content/mlrs && MLRS_POS_REPS=9 cargo test -p mlrs-backend --release "
         "--features cuda --test gram_perf_test -- --ignored --nocapture")

# The whole-fit ladder under each dispatch arm. Neither Gram formation may be
# assumed on this device: the crossover between them is an adapter property
# (the local iGPU puts it between d=64 and d=128) and has never been measured
# on a T4.
for label, env in [("A. whole fit — default dispatch",      ""),
                   ("B. whole fit — tiled Gram forced",     "LR_GRAM_TILED=1"),
                   ("C. whole fit — blocked Gram forced",   "LR_GRAM_BLOCKED=1"),
                   ("D. whole fit — host arm forced",       "MLRS_RIDGE_GRAM_HOST=1")]:
    cap("\n" + "-" * 72)
    cap(label)
    cap("-" * 72)
    sh(RIDGE.format(env=env), echo=False)

# The Gram phase in isolation, sweeping the tiled kernel's cube width — this is
# what locates the tiled/blocked crossover on THIS device, which is the number
# the dispatch threshold needs.
cap("\n" + "-" * 72)
cap("E. Gram phase only — formation + cube-width sweep")
cap("-" * 72)
sh(GRAM, echo=False)

# --- push the log ----------------------------------------------------------
stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
name  = f"ridge_positive_t4_{stamp}_{sha[:8]}.log"
os.makedirs("/content/out", exist_ok=True)
with open(f"/content/out/{name}", "w") as f:
    f.write(log.getvalue())

push_url = f"https://x-access-token:{TOKEN}@github.com/BectorVoom/mlrs.git"
os.makedirs("/content/results", exist_ok=True)
# A dedicated results branch, never the code branch: the two are written by
# different machines and would otherwise race.
cmds = [
    ["git", "init", "-q", "/content/results"],
    ["git", "-C", "/content/results", "config", "user.email", "colab@local"],
    ["git", "-C", "/content/results", "config", "user.name",  "colab-t4"],
]
for c in cmds:
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
                f"T4 ridge(positive) probe @ {sha[:8]}"], capture_output=True)
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

if __name__ == "__main__":
    print(CELL_1)
    print("\n# ---------------------------------------------------------\n")
    print(CELL_2)
