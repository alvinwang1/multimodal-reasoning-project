"""Final-report GPU runs on Modal (one NVIDIA L4): src.final.run_all inside a Modal container.

One-time, on your machine:   pip install modal   then   modal setup   (opens the browser to log in)

    modal run scripts/modal_run.py --sections preflight      # CPU-only check that every item loads (~10 min)
    modal run --detach scripts/modal_run.py                  # everything on an L4 (~8 h); keeps going if you
                                                             # close the laptop; resumable: rerun to continue
    modal run scripts/modal_run.py --sections "9 9b"         # only some sections
    modal run scripts/modal_run.py --limit 2                 # quick end-to-end pass on 2 items per section
    modal run scripts/modal_run.py --progress                # passes done per section (CPU, ~1 min)
    modal run --detach scripts/modal_run.py --sections protocol   # why accuracy is below Qwen's published score (~1 h);
                                                             # try --limit 2 first (~5 min)
    modal run scripts/modal_run.py --sections router         # Idea 1 router study on the score bank (CPU, ~10 min)
    modal volume get mmml-final results/final/final_results.tar.gz .    # the results, when it has finished
    tar xzf final_results.tar.gz                             # in the repo root: files land in results/

The code in src/, configs/ and data/ is uploaded from this checkout at launch (no push needed); the commit and any
uncommitted changes are recorded in results/final/logs/runs.log. Results, logs and the model/dataset downloads live on
the Modal Volume "mmml-final", so a rerun skips finished passes and does not download again. Watch the logs in the
terminal, or after --detach under Apps on modal.com. Sections, commands and the L4 check are those of
src/final/run_all.py (the same sections as colab/run_final.ipynb).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME, VOLUME_NAME, GPU = "mmml-final", "mmml-final", "L4"
REPO_DIR, VOL_DIR = "/repo", "/vol"  # inside the container: the code, and the Volume (results + downloads)
CPU_ONLY = {"preflight", "report", "router"}  # sections that need no GPU
SKIP_PACKAGES = re.compile(r"^(gradio|empath)\b")  # annotation tool / Empath: Assignment 2 analyses B and C only
LOCAL_REPO = Path(__file__).resolve().parents[1]
IGNORE = ["**/__pycache__", "**/*.pyc", "**/.DS_Store"]


def run_packages():
    """The team's pinned versions (requirements.txt), minus packages the runs never import."""
    lines = (LOCAL_REPO / "requirements.txt").read_text().splitlines()
    return [ln.split("#")[0].strip() for ln in lines
            if ln.split("#")[0].strip() and not SKIP_PACKAGES.match(ln.strip())]


image = (modal.Image.debian_slim(python_version="3.12")
         .env({"HF_HOME": f"{VOL_DIR}/hf_cache", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}))
if modal.is_local():  # inside the container the image is already built
    image = (image.uv_pip_install(*run_packages())
             .add_local_dir(LOCAL_REPO / "src", f"{REPO_DIR}/src", copy=True, ignore=IGNORE)
             .add_local_dir(LOCAL_REPO / "configs", f"{REPO_DIR}/configs", copy=True, ignore=IGNORE)
             .add_local_dir(LOCAL_REPO / "data", f"{REPO_DIR}/data", copy=True, ignore=IGNORE + ["cache", "omnibench"]))

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def _run(args, commit):
    """Run src.final.run_all with results and downloads on the Volume; returns its exit code."""
    os.makedirs(f"{VOL_DIR}/results", exist_ok=True)
    if not os.path.lexists(f"{REPO_DIR}/results"):
        os.symlink(f"{VOL_DIR}/results", f"{REPO_DIR}/results")
    cmd = [sys.executable, "-m", "src.final.run_all", *args]
    print("$ python " + " ".join(cmd[1:]), flush=True)
    env = {**os.environ, "MMML_COMMIT": commit, "MMML_JOBS": "4"}  # MMML_JOBS: worker processes (both functions: 4 CPUs)
    rc = subprocess.run(cmd, cwd=REPO_DIR, env=env).returncode
    volume.commit()  # background commits already run every few seconds; this is the final one
    return rc


@app.function(image=image, gpu=GPU, cpu=4.0, memory=24 * 1024, timeout=24 * 3600, retries=2,
              volumes={VOL_DIR: volume})
def run_gpu(args: list, commit: str) -> int:
    # retries: if Modal stops the container (e.g. preemption), it starts again and run_all resumes where it stopped;
    # a failed section returns its exit code instead of raising, so it is not retried
    return _run(args, commit)


@app.function(image=image, cpu=4.0, memory=16 * 1024, timeout=4 * 3600, volumes={VOL_DIR: volume})
def run_cpu(args: list, commit: str) -> int:
    return _run(args, commit)


def local_commit():
    def git(*a):
        return subprocess.run(["git", "-C", str(LOCAL_REPO), *a], capture_output=True, text=True).stdout.strip()
    try:
        head = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except OSError:
        return "unknown"
    return (head + (" + uncommitted changes" if dirty else "")) if head else "unknown"


@app.local_entrypoint()
def main(sections: str = "", limit: int = 0, progress: bool = False):
    commit = local_commit()
    if progress:
        run_cpu.remote(["--progress"], commit)
        return
    names = sections.replace(",", " ").split()
    args = (["--sections", *names] if names else []) + (["--limit", str(limit)] if limit else [])
    cpu_only = bool(names) and set(names) <= CPU_ONLY
    print(f"code {commit} | {'CPU' if cpu_only else GPU} | sections {' '.join(names) or 'all'}"
          + (f" | first {limit} items" if limit else ""), flush=True)
    rc = (run_cpu if cpu_only else run_gpu).remote(args, commit)
    if rc:
        print(f"\nThe run stopped with exit code {rc}: the error is in the output above (and in results/final/logs/ "
              f"on the Volume). Run the same command again after fixing it; finished passes are skipped.")
    if not rc and "protocol" in names:
        print(f"\nDone. Download the protocol check (its summary is also in the output above):\n"
              f"  modal volume get --force {VOLUME_NAME} results/final/protocol_check.jsonl results/final/\n"
              f"  modal volume get --force {VOLUME_NAME} results/final/protocol_check.md results/final/")
    if not rc and "router" in names:
        print(f"\nDone. Download the Idea 1 results (report, tables, figures, question embeddings):\n"
              f"  modal volume get --force {VOLUME_NAME} results/final/idea1_results.tar.gz .\n"
              f"then unpack them in the repo root:  tar xzf idea1_results.tar.gz")
    if not rc and (not names or "report" in names):
        print(f"\nDone. Download the results:\n  modal volume get {VOLUME_NAME} results/final/final_results.tar.gz .\n"
              f"then unpack them in the repo root:  tar xzf final_results.tar.gz")
