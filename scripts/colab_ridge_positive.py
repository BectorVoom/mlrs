"""Colab driver for the `Ridge(positive=True)` CUDA perf probe (RIDGE-POS-CUDA).

Paste CELL 1 once per Colab session (installs Rust, clones the work branch,
first build ~5 min), then re-run CELL 2 after every push to re-measure — the
incremental rebuild is seconds, not minutes.

Runtime must be **GPU** (Runtime → Change runtime type → T4 GPU).

The numbers this prints are compared against the LOCAL 16-thread cpu baseline
produced by:

    cargo test -p mlrs-algos --release --features cpu \
      --test ridge_positive_perf_test -- --ignored --nocapture

so both sides run the identical probe and the identical dispatch branch.
"""

CELL_1 = r"""
#@title CELL 1 — one-time setup (Rust + clone + first build)
import os, subprocess, textwrap

BRANCH = "perf/ridge-positive-cuda"
REPO   = "https://github.com/BectorVoom/mlrs.git"

def sh(cmd, **kw):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    print(r.stdout[-8000:])
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {cmd}")

sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
sh("nvcc --version | tail -2 || true")
sh("nproc && lscpu | grep -E 'Model name|^CPU\\(s\\):'")

if not os.path.exists("/root/.cargo/bin/cargo"):
    sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
       "| sh -s -- -y --profile minimal --default-toolchain stable")
os.environ["PATH"] = "/root/.cargo/bin:" + os.environ["PATH"]
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
sh("cargo --version && rustc --version")

if not os.path.exists("/content/mlrs/.git"):
    sh(f"git clone --depth 50 -b {BRANCH} {REPO} /content/mlrs")
sh(f"cd /content/mlrs && git fetch origin {BRANCH} && git checkout -B {BRANCH} origin/{BRANCH} && git log --oneline -1")

# Build only (so CELL 2's timing run is not polluted by compile output).
sh("cd /content/mlrs && cargo test -p mlrs-algos --release --features cuda "
   "--test ridge_positive_perf_test --no-run")
print("SETUP OK")
"""

CELL_2 = r"""
#@title CELL 2 — re-measure (run after every push)
import os, subprocess
BRANCH = "perf/ridge-positive-cuda"
os.environ["PATH"] = "/root/.cargo/bin:" + os.environ["PATH"]
os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")

def sh(cmd):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-20000:])
    return r.returncode

sh(f"cd /content/mlrs && git fetch origin {BRANCH} && "
   f"git checkout -B {BRANCH} origin/{BRANCH} && git log --oneline -1")
sh("cd /content/mlrs && MLRS_POS_REPS=9 cargo test -p mlrs-algos --release "
   "--features cuda --test ridge_positive_perf_test -- --ignored --nocapture")
"""

if __name__ == "__main__":
    print(CELL_1)
    print("\n# ---------------------------------------------------------\n")
    print(CELL_2)
