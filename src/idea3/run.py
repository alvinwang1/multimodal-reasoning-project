"""Idea 3 runner: score every (example, budget cell, permutation) and append to JSONL.

Same scorer, prompts and option permutations as Analysis A; only the image token cap and the
audio length change. Resumable: entries already in the output are skipped.

    python -m src.idea3.run --split dev --limit 2 --dry-run   # build inputs, no model
    python -m src.idea3.run --split dev                       # grid minus Analysis A corners + ablation
    python -m src.idea3.run --split test                      # full grid + surrogates on held-out items

Default cells (both splits): the full 4x4 grid plus the three surrogate cells (image+transcript,
caption+audio, caption+transcript); dev also gets the centered-window audio ablations v0_a0.25c,
v0_a0.5c. Dev re-runs Analysis A's corners so the whole grid comes from one device and dtype;
the analysis reports agreement with the Analysis A run.
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np

from src.analysis_a.prompts import CONFIGS, build_messages, permutation, surrogate_texts
from src.common import append_jsonl, load_config, read_jsonl, repo_path, seed_everything
from src.data import OmniBenchSubset, build_subset, load_subset_ids, text_records
from src.idea3.cells import SURROGATE_CELLS, TOKENS_PER_PATCH, grid_cells, parse_cell, truncate_audio

log = logging.getLogger("idea3")
CONFIG_BY_REPS = {reps: name for name, reps in CONFIGS.items()}
ABLATION_CELLS = ["v0_a0.25c", "v0_a0.5c"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--cells", nargs="*", default=None, help="override the default cells")
    ap.add_argument("--out", default=None, help="output JSONL (default <out_dir>/raw_<split>.jsonl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-perms", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--dry-run", action="store_true", help="build inputs and print the jobs; no model")
    return ap.parse_args()


def test_ids(cfg, data=None):
    """Held-out subset: same audio-type balance as dev, disjoint from it. Built once, then read."""
    icfg = cfg["idea3"]
    path = repo_path(icfg["test_subset"])
    if path.exists():
        return json.loads(path.read_text())["ids"]
    from src.data import load_omnibench
    ds = data.ds if data is not None else load_omnibench(cfg)
    dev = set(load_subset_ids(cfg))
    records, _ = text_records(ds)
    pool = [r for r in records if r["index"] not in dev]
    by_type, notes = build_subset(pool, cfg["subset"]["n"], icfg["test_seed"])
    ids = sorted(i for v in by_type.values() for i in v)
    path.write_text(json.dumps({"seed": icfg["test_seed"], "n": len(ids), "disjoint_from": cfg["subset"]["path"],
                                "per_audio_type": {t: len(v) for t, v in by_type.items()},
                                "notes": notes, "ids": ids}, indent=2))
    log.info("wrote %d held-out ids to %s", len(ids), path)
    return ids


def default_cells(cfg, split):
    cells = grid_cells(cfg["idea3"]["image_budgets"], cfg["idea3"]["audio_fracs"]) + SURROGATE_CELLS
    return cells + ABLATION_CELLS if split == "dev" else cells


def cell_inputs(ex, cell, cfg):
    """-> (example with truncated audio, Analysis A config name, max_pixels or None, audio seconds used)."""
    c = parse_cell(cell)
    img = None if c["img"] == 0 else ("text" if c["img"] == "cap" else "raw")
    aud = None if c["aud"] == 0 else ("text" if c["aud"] == "trans" else "raw")
    ex = dict(ex)
    sr = cfg["dataset"]["audio_sr"]
    if aud == "raw":
        ex["audio"] = truncate_audio(ex["audio"], c["aud"], sr, cfg["idea3"]["min_audio_sec"], c["center"])
    max_pixels = c["img"] * TOKENS_PER_PATCH if img == "raw" else None
    audio_sec = len(ex["audio"]) / sr if aud == "raw" else 0.0
    return ex, CONFIG_BY_REPS[(img, aud)], max_pixels, audio_sec


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    icfg = cfg["idea3"]
    seed = cfg["seed"]
    seed_everything(seed)

    data = OmniBenchSubset(cfg)
    ids = data.ids if args.split == "dev" else test_ids(cfg, data)
    if args.limit:
        ids = ids[: args.limit]
    cells = args.cells or default_cells(cfg, args.split)
    for c in cells:
        parse_cell(c)
    n_perms = args.n_perms or icfg["n_perms"]
    out = repo_path(args.out or f"{icfg['out_dir']}/raw_{args.split}.jsonl")

    done = {(r["id"], r["cell"], r["perm"]) for r in read_jsonl(out)}
    todo = [(i, c, p) for i in ids for c in cells for p in range(n_perms) if (i, c, p) not in done]
    log.info("%s: %d examples x %d cells x %d perms | %d done, %d to run -> %s",
             args.split, len(ids), len(cells), n_perms, len(done), len(todo), out)
    if not todo:
        return

    if args.dry_run:
        for idx in ids[:2]:
            ex = data[idx]
            for cell in cells:
                ex_c, config, max_pixels, audio_sec = cell_inputs(ex, cell, cfg)
                kinds = [c["type"] for c in build_messages(ex_c, config, list(range(4)))[1]["content"]]
                log.info("  id %d %-12s -> %-9s max_pixels=%s audio=%.2fs/%.2fs content=%s", idx, cell, config,
                         max_pixels, audio_sec, len(ex["audio"]) / cfg["dataset"]["audio_sr"], kinds)
        return

    from src.analysis_a.model import OmniScorer, map_back
    scorer = OmniScorer(cfg, device=args.device, dtype=args.dtype)
    default_max_pixels = scorer.max_pixels
    log.info("model on %s (%s)", scorer.device, scorer.dtype)

    warmup = icfg["warmup"] if args.warmup is None else args.warmup
    if warmup:
        _, inputs = scorer.prepare(build_messages(data[todo[0][0]], "full", list(range(4))))
        for k in range(warmup):
            log.info("warmup %d/%d: %.0f ms", k + 1, warmup, scorer.score(inputs)["latency_ms"])

    by_example = {}
    for j in todo:
        by_example.setdefault(j[0], []).append(j)

    t_start, n_run = time.time(), 0
    for idx, jobs in by_example.items():
        ex = data[idx]
        perms = {}
        for p in range(n_perms):
            perms[p] = permutation(idx, p, seed, prior=[perms[q] for q in range(p)])
        for (_, cell, p) in jobs:
            perm = perms[p]
            ex_c, config, max_pixels, audio_sec = cell_inputs(ex, cell, cfg)
            scorer.max_pixels = max_pixels or default_max_pixels
            t0 = time.perf_counter()
            _, inputs = scorer.prepare(build_messages(ex_c, config, perm))
            preproc_ms = (time.perf_counter() - t0) * 1000
            res = scorer.score(inputs)
            if not np.all(np.isfinite(res["letter_logits"])):
                raise SystemExit(f"non-finite logits for id {idx} {cell} perm {p} in {scorer.dtype}; "
                                 "retry with --dtype float32")
            probs = map_back(res["probs_display"], perm)
            pred = int(np.argmax(probs))
            append_jsonl(out, {
                "id": idx, "split": args.split, "cell": cell, "config": config, "perm": p, "perm_order": perm,
                "audio_type": ex["audio type"], "task_type": ex["task type"],
                "answer_idx": ex["answer_idx"], "pred_idx": pred, "correct": pred == ex["answer_idx"],
                "probs": probs, "max_prob": float(max(probs)),
                "letter_logits": res["letter_logits"], "top_vocab_is_letter": res["top_vocab_is_letter"],
                **scorer.token_counts(inputs, surrogate_texts(ex_c, config)),
                "latency_ms": res["latency_ms"], "preproc_ms": preproc_ms,
                "peak_mem_mb": res["peak_mem_mb"], "baseline_mem_mb": res["baseline_mem_mb"],
                "max_pixels": max_pixels, "audio_sec_used": audio_sec,
                "audio_sec": len(ex["audio"]) / cfg["dataset"]["audio_sr"],
                "device": str(scorer.device), "dtype": str(scorer.dtype).replace("torch.", ""),
                "model": cfg["model"]["hf_id"],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            n_run += 1
            if n_run % 20 == 0 or n_run == len(todo):
                el = time.time() - t_start
                log.info("%d/%d  %.2fs/pass  ETA %.1f min", n_run, len(todo), el / n_run,
                         el / n_run * (len(todo) - n_run) / 60)
    log.info("done: %d records written to %s", n_run, out)


if __name__ == "__main__":
    main()
