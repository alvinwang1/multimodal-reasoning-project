"""Final-report GPU runs on one machine, e.g. an AWS g6.2xlarge (1x NVIDIA L4): every section of
colab/run_final.ipynb, in order, with the same commands and config lists (the notebook is the Colab fallback).

    python -m src.final.run_all                    # everything (~8 h on an L4); resumable
    python -m src.final.run_all --sections 6 7     # only some sections (always run in the order below)
    python -m src.final.run_all --limit 2          # quick end-to-end pass: the first 2 items of every section
    python -m src.final.run_all --progress         # passes done per section (safe while a run is going)
    python -m src.final.run_all --dry-run          # print the commands only

Sections: preflight (CPU: every item loads and builds its prompts), smoke (2 items, plus 1 item through the Idea 3 and
Idea 4 code, into results/final/smoke/), then 6, 7, 8, 9, 9b, 9c, 10, 11 as in the notebook, and report (score-bank
coverage, progress, and results/final/final_results.tar.gz with the 4 result files and the logs, to copy back).

Resumable: after a crash, a reboot or Ctrl-C, run the same command again; finished passes are skipped. The run stops at
the first section that fails (--keep-going moves on to the next one). Output is shown live and appended to
results/final/logs/<section>.log; results/final/logs/runs.log records the commit, GPU and arguments of every run.
The GPU sections only start if nvidia-smi names the expected GPU (--gpu, default L4), so all timings share one GPU type.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tarfile
import time
from collections import Counter
from datetime import datetime

from src.common import REPO_ROOT, load_config, read_jsonl, repo_path

# The same lists as colab/run_final.ipynb §0 (keep the two in sync)
PROBE_CELLS = "v64_a0.5 v64_a0 v0_a0.5".split()
OTHER_CELLS = "v0_a0.25 v64_a0.25 v64_a1 v256_a0 v256_a0.25 v256_a0.5 v256_a1 v768_a0.25 v768_a0.5".split()
SELECT_CELLS = "v0_a0.25c v0_a0.5c v0_a0.25e v0_a0.5e v64_a0.25e v64_a0.5e".split()
IDEA4_CONFIGS = ("img_route img_route_fb img_qdesc img_pipe gencap_pipe img_masked cap_masked "  # new pipeline + ablations
                 "img_asr img_gendesc gen_cap_desc gen_cap_asr").split()                         # midterm baselines
IDEA4_LEN_BASES = "img_qdesc img_pipe gencap_pipe img_gendesc gen_cap_desc gen_cap_asr".split()  # Qwen-generated text
ABLATION_LENS = (80, 120)  # generation-length ablation; 256 is §9 itself (final.idea4_gen_lens)
IDEA4_LEN_CONFIGS = [f"{c}_t{n}" for n in ABLATION_LENS for c in IDEA4_LEN_BASES]
IDEA4_THR_CONFIGS = "img_tags img_tagsq gencap_tagsq gencap_aud".split()  # routing branches (speech-threshold sweep)

NOISE = r"UNEXPECTED|^token2wav|^talker|Model config|^Key |^-+\+|^Notes:|can be ignored|LOAD REPORT|^\s*$|Loading weights"
FILTER = re.compile(NOISE)                                       # same filter as the notebook
FILTER_GEN = re.compile(NOISE + r"|warn|^\[transformers\]")      # idea4_generate (Whisper/AST warnings)
SMOKE_DIR, LOG_DIR, BUNDLE = "results/final/smoke", "results/final/logs", "results/final/final_results.tar.gz"
ORDER = ["preflight", "smoke", "6", "7", "8", "9", "9b", "9c", "10", "11", "report"]
ALIASES = {"5": "smoke", "12": "report"}


def sections(cfg, limit):
    """section -> (log name, title, commands)"""
    py = [sys.executable, "-m"]
    lim = ["--limit", str(limit)] if limit else []
    f = cfg["final"]

    def score_i4(configs, surrogates=f["idea4_surrogates"], out=f["idea4_out"], extra=lim):
        return py + ["src.analysis_a.run", "--split", "idea4", "--configs", *configs, "--perm-configs", "all",
                     "--surrogates", surrogates, "--out", out] + extra

    def idea3(cells, extra=lim, out=()):
        return py + ["src.idea3.run", "--split", "final", "--cells", *cells, *out] + extra

    return {
        "preflight": ("00_preflight", "every item loads and builds its prompts (CPU, ~5 min)",
                      [py + ["src.final.preflight"] + lim]),
        "smoke": ("05_smoke", "smoke test: 2 items, and 1 item through the Idea 3 / Idea 4 code (~10 min)", [
            py + ["src.analysis_a.run", "--split", "final", "--limit", "2", "--configs", "full", "cap_trans",
                  "--out", f"{SMOKE_DIR}/smoke_l4.jsonl"],
            idea3(["v64_a0.5e", "v0_a0.25"], extra=["--limit", "1"], out=["--out", f"{SMOKE_DIR}/idea3_smoke.jsonl"]),
            py + ["src.final.idea4_generate", "--limit", "1", "--out", f"{SMOKE_DIR}/idea4_smoke.json"],
            score_i4(IDEA4_CONFIGS + IDEA4_LEN_CONFIGS + IDEA4_THR_CONFIGS, f"{SMOKE_DIR}/idea4_smoke.json",
                     f"{SMOKE_DIR}/idea4_smoke.jsonl", ["--limit", "1"])]),
        "6": ("06_base_bank", "base bank: 7 input formats x 3 orders on all items (~1.6 h)",
              [py + ["src.analysis_a.run", "--split", "final", "--perm-configs", "all", "--out", f["base_out"]] + lim]),
        "7": ("07_probe_cells", "probe cells (64,1/2), (64,0), (0,1/2) (~0.7 h)", [idea3(PROBE_CELLS)]),
        "8": ("08_idea4_generate", "Idea 4: generate the text for 242 items (~55 min)",
              [py + ["src.final.idea4_generate", "--out", f["idea4_surrogates"]] + lim]),
        "9": ("09_idea4_score", "Idea 4: score the 11 configs (~35 min)", [score_i4(IDEA4_CONFIGS)]),
        "9b": ("09b_idea4_lengths", "Idea 4: generation-length ablation, 80 and 120 tokens (~30 min)",
               [score_i4(IDEA4_LEN_CONFIGS)]),
        "9c": ("09c_idea4_threshold", "Idea 4: speech-threshold sweep inputs (~12 min)", [score_i4(IDEA4_THR_CONFIGS)]),
        "10": ("10_idea3_grid", "Idea 3: the other 9 budget-grid cells (~2 h)", [idea3(OTHER_CELLS)]),
        "11": ("11_idea3_audio_select", "Idea 3: centered and loudest audio windows (~1 h)", [idea3(SELECT_CELLS)]),
        "report": ("12_coverage", "score-bank coverage and the results bundle", [py + ["src.final.bank"]]),
    }


def progress(cfg):
    """Passes done per section (unique item x config x order; cut-off lines ignored), as the notebook's progress()."""
    f = cfg["final"]
    n_items = len({i for fold in json.loads(repo_path(f["folds"]).read_text())["folds"] for i in fold})
    n_i4 = len(json.loads(repo_path(f["idea4_ids"]).read_text())["ids"])
    pa, p3 = cfg["analysis_a"]["n_perms"], cfg["idea3"]["n_perms"]

    def keys(path, key):
        return {(r["id"], r[key], r["perm"]) for r in read_jsonl(repo_path(path)) if {"id", key, "perm"} <= r.keys()}

    base = keys(f["base_out"], "config")
    i3 = Counter(c for _, c, _ in keys(f"{cfg['idea3']['out_dir']}/raw_final.jsonl", "cell"))
    i4 = Counter(c for _, c, _ in keys(f["idea4_out"], "config"))
    tp = repo_path(f["idea4_surrogates"])
    text = len(json.loads(tp.read_text())) if tp.exists() and tp.stat().st_size else 0
    rows = [("6  base bank", len(base), n_items * len(cfg["analysis_a"]["configs"]) * pa),
            ("7  probe cells", sum(i3[c] for c in PROBE_CELLS), n_items * len(PROBE_CELLS) * p3),
            ("8  idea4 text (items)", text, n_i4),
            ("9  idea4 scoring", sum(i4[c] for c in IDEA4_CONFIGS), n_i4 * len(IDEA4_CONFIGS) * pa),
            ("9b idea4 lengths", sum(i4[c] for c in IDEA4_LEN_CONFIGS), n_i4 * len(IDEA4_LEN_CONFIGS) * pa),
            ("9c idea4 threshold", sum(i4[c] for c in IDEA4_THR_CONFIGS), n_i4 * len(IDEA4_THR_CONFIGS) * pa),
            ("10 idea3 grid", sum(i3[c] for c in OTHER_CELLS), n_items * len(OTHER_CELLS) * p3),
            ("11 audio selection", sum(i3[c] for c in SELECT_CELLS), n_items * len(SELECT_CELLS) * p3)]
    for name, n, total in rows:
        print(f"  {name:22s} {n:>7,} / {total:>7,}  {100 * n / total:5.1f}%")
    return rows


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = out.stdout.strip().splitlines()
    return names[0].strip() if out.returncode == 0 and names else None


def commit():
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    try:
        head = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except OSError:
        return "unknown"
    return (head + (" + uncommitted changes" if dirty else "")) if head else "unknown"


def run_cmd(cmd, log_path):
    """Run one command: filtered output live on screen and appended to the section log. Returns the exit code."""
    flt = FILTER_GEN if "src.final.idea4_generate" in cmd else FILTER
    shown = "python " + " ".join(cmd[1:])
    with open(log_path, "a") as log:
        log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S}  {shown}\n")
        log.flush()
        print(f"$ {shown}", flush=True)
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            for line in proc.stdout:
                if not flt.search(line.rstrip("\n")):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                    log.flush()
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise


def bundle(cfg):
    """The result files and logs in one archive, to copy back (unpack it in the repo root)."""
    f = cfg["final"]
    parts = [f["base_out"], f"{cfg['idea3']['out_dir']}/raw_final.jsonl", f["idea4_out"], f["idea4_surrogates"],
             LOG_DIR]
    out = repo_path(BUNDLE)
    with tarfile.open(out, "w:gz") as tar:
        for rel in parts:
            if repo_path(rel).exists():
                tar.add(repo_path(rel), arcname=rel)
    print(f"bundle: {out} ({out.stat().st_size / 2**20:.1f} MB). Copy it to your Mac with\n"
          f"  scp -i <your-key.pem> <user>@<instance-address>:{out} .\n"
          f"then unpack it in the repo root:  tar xzf final_results.tar.gz", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sections", nargs="*", default=None, help=f"any of: {' '.join(ORDER)} (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items of each section (a quick pass)")
    ap.add_argument("--gpu", default="L4", help="the GPU the runs must use (substring of its name; default L4)")
    ap.add_argument("--any-gpu", action="store_true", help="skip the GPU check")
    ap.add_argument("--keep-going", action="store_true", help="after a failed section, continue with the next one")
    ap.add_argument("--progress", action="store_true", help="only show how far each section has got")
    ap.add_argument("--dry-run", action="store_true", help="only print the commands")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.progress:
        progress(cfg)
        return

    wanted = [ALIASES.get(s, s) for s in (args.sections or ORDER)]
    unknown = [s for s in wanted if s not in ORDER]
    if unknown:
        ap.error(f"unknown section(s) {unknown}; choose from {' '.join(ORDER)}")
    todo = [s for s in ORDER if s in wanted]  # dependencies: always in this order (9, 9b, 9c need 8)
    secs = sections(cfg, args.limit)
    if args.dry_run:
        for s in todo:
            print(f"# {s}: {secs[s][1]}")
            for cmd in secs[s][2]:
                print("python " + " ".join(cmd[1:]))
        return

    gpu = gpu_name()
    if any(s not in ("preflight", "report") for s in todo) and not args.any_gpu \
            and (gpu is None or args.gpu.lower() not in gpu.lower()):
        raise SystemExit(f"GPU check: these runs must use an {args.gpu}, but this machine has "
                         f"{gpu or 'no GPU that nvidia-smi can see'}. Use the right instance type (g6.2xlarge for "
                         "an L4), or --gpu NAME if the whole run uses another GPU.")
    logs = repo_path(LOG_DIR)
    logs.mkdir(parents=True, exist_ok=True)
    repo_path(SMOKE_DIR).mkdir(parents=True, exist_ok=True)
    info = {"start": datetime.now().isoformat(timespec="seconds"), "commit": commit(), "gpu": gpu,
            "host": socket.gethostname(), "sections": todo, "limit": args.limit, "argv": sys.argv[1:]}
    with open(logs / "runs.log", "a") as fh:
        fh.write(json.dumps(info) + "\n")
    print(f"commit {info['commit']} | GPU {gpu} | sections {' '.join(todo)}" +
          (f" | first {args.limit} items only" if args.limit else ""), flush=True)

    failed = []
    try:
        for s in todo:
            log_name, title, cmds = secs[s]
            print(f"\n===== {s}: {title}   [{datetime.now():%a %H:%M}]", flush=True)
            t0, rc = time.time(), 0
            for cmd in cmds:
                rc = run_cmd(cmd, logs / f"{log_name}.log")
                if rc:
                    break
            if s == "report" and not rc:
                bundle(cfg)
            print(f"===== {s}: {'done' if not rc else f'FAILED (exit code {rc})'} in "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if rc:
                failed.append(s)
                if not args.keep_going:
                    break
    except KeyboardInterrupt:
        print("\nstopped (Ctrl-C). Run the same command again to resume.")
        sys.exit(130)
    print("\nprogress:")
    progress(cfg)
    if failed:
        raise SystemExit(f"\nfailed: {' '.join(failed)}. The error is at the end of its log in {LOG_DIR}/. "
                         "After fixing it, run the same command again: finished passes are skipped.")


if __name__ == "__main__":
    main()
