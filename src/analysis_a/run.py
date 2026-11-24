"""Analysis A runner: score every (example, config, permutation) and append to JSONL.

Resumable: entries already in --out are skipped, so the job can be killed and restarted.

    python -m src.analysis_a.run --limit 5 --out results/analysis_a/smoke.jsonl --check-generate
    python -m src.analysis_a.run                       # full subset, all configs
    python -m src.analysis_a.run --n-perms 3 --configs full cap_trans
"""
import argparse
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np

from src.analysis_a.model import OmniScorer, map_back
from src.analysis_a.prompts import build_messages, config_spec, permutation, surrogate_fields, surrogate_texts
from src.common import append_jsonl, load_config, read_jsonl, repo_path, seed_everything
from src.data import OmniBenchSubset

log = logging.getLogger("analysis_a")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    ap.add_argument("--out", default=None, help="output JSONL (default from config)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N subset examples")
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="explicit OmniBench indices")
    ap.add_argument("--split", choices=["train", "test", "all", "final", "idea4"], default=None,
                    help="midterm train/test/all split; final = every item in the final-report folds; "
                         "idea4 = the final Idea 4 subset")
    ap.add_argument("--surrogates", default=None, help="JSON of generated surrogates (midterm Idea 4)")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="config names: base configs, gen_* / Idea 4 configs (add _t<n> for the generation-length "
                         "ablation, e.g. img_qdesc_t80), or bud_i{r}_a{r}")
    ap.add_argument("--n-perms", type=int, default=None, help="option permutations (incl. original)")
    ap.add_argument("--perm-configs", nargs="*", default=None, help="configs run under all perms ('all')")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None, help="bfloat16 / float16 / float32")
    ap.add_argument("--check-generate", action="store_true",
                    help="also greedy-generate and compare with the logit argmax (slower)")
    return ap.parse_args()


def jobs_for(idx, configs, perm_configs, n_perms):
    for c in configs:
        for p in (range(n_perms) if c in perm_configs else [0]):
            yield (idx, c, p)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    acfg = cfg["analysis_a"]
    seed = cfg["seed"]
    seed_everything(seed)

    out = repo_path(args.out or acfg["out"])
    configs = args.configs or acfg["configs"]
    for c in configs:
        config_spec(c)  # fail fast on typos
    perm_configs = args.perm_configs if args.perm_configs is not None else acfg["perm_configs"]
    if perm_configs == ["all"]:
        perm_configs = list(configs)
    n_perms = args.n_perms or acfg["n_perms"]
    warmup = acfg["warmup"] if args.warmup is None else args.warmup

    data = OmniBenchSubset(cfg)
    ids = args.ids or data.ids
    if args.split == "final":
        ids = sorted(i for f in json.loads(repo_path(cfg["final"]["folds"]).read_text())["folds"] for i in f)
    elif args.split == "idea4":
        ids = json.loads(repo_path(cfg["final"]["idea4_ids"]).read_text())["ids"]
    elif args.split:
        split = json.loads(repo_path(cfg["midterm"]["split"]).read_text())
        ids = split["train"] + split["test"] if args.split == "all" else split[args.split]
    if args.surrogates and not repo_path(args.surrogates).exists():
        raise SystemExit(f"--surrogates file not found: {args.surrogates} (final Idea 4: run src.final.idea4_generate first)")
    surrogates = json.loads(repo_path(args.surrogates).read_text()) if args.surrogates else {}
    if args.limit:
        ids = ids[: args.limit]
    # every generated text these configs read must be in --surrogates (checked before the model loads)
    missing = sorted({(i, f) for i in ids for c in configs for f in surrogate_fields(c)
                      if f not in surrogates.get(str(i), {})})
    if missing:
        raise SystemExit(f"{len(missing)} surrogate field(s) missing for these configs, e.g. id {missing[0][0]}: "
                         f"{missing[0][1]!r}. Pass --surrogates with a complete file (final Idea 4: let "
                         "src.final.idea4_generate finish first).")

    done = {(r["id"], r["config"], r["perm"]) for r in read_jsonl(out)}
    todo = [j for i in ids for j in jobs_for(i, configs, perm_configs, n_perms) if j not in done]
    log.info("%d examples x %s | perms=%d on %s | %d done, %d to run -> %s",
             len(ids), configs, n_perms, perm_configs, len(done), len(todo), out)
    if not todo:
        return

    scorer = OmniScorer(cfg, device=args.device, dtype=args.dtype)
    log.info("model on %s (%s); letter ids %s", scorer.device, scorer.dtype, scorer.letter_ids)
    for s, tid, dec in scorer.letter_token_report():
        log.info("  tokenize %-5r -> %s %s", s, tid, dec)

    if warmup:
        ex = data[todo[0][0]]
        _, inputs = scorer.prepare(build_messages(ex, "full", list(range(4))))
        for k in range(warmup):
            r = scorer.score(inputs)
            log.info("warmup %d/%d: %.0f ms", k + 1, warmup, r["latency_ms"])

    by_example = {}
    for j in todo:
        by_example.setdefault(j[0], []).append(j)

    t_start, n_run = time.time(), 0
    for idx, jobs in by_example.items():
        t0 = time.perf_counter()
        ex = data[idx]  # decode image bytes + load/resample audio: part of end-to-end latency
        decode_ms = (time.perf_counter() - t0) * 1000
        ex.update(surrogates.get(str(idx), {}))
        perms = {}
        for p in range(n_perms):
            perms[p] = permutation(idx, p, seed, prior=[perms[q] for q in range(p)])
        for (_, config, p) in jobs:
            perm = perms[p]
            messages = build_messages(ex, config, perm)
            t0 = time.perf_counter()
            _, inputs = scorer.prepare(messages)
            preproc_ms = (time.perf_counter() - t0) * 1000
            res = scorer.score(inputs)
            probs = map_back(res["probs_display"], perm)
            pred = int(np.argmax(probs))
            rec = {
                "id": idx, "config": config, "perm": p, "perm_order": perm,
                "img_ratio": config_spec(config)["img_ratio"], "aud_ratio": config_spec(config)["aud_ratio"],
                "audio_type": ex["audio type"], "task_type": ex["task type"],
                "answer_idx": ex["answer_idx"], "pred_idx": pred,
                "correct": pred == ex["answer_idx"], "probs": probs,
                "max_prob": float(max(probs)),
                "probs_display": res["probs_display"], "letter_logits": res["letter_logits"],
                "top_vocab_token": res["top_vocab_token"], "top_vocab_is_letter": res["top_vocab_is_letter"],
                **scorer.token_counts(inputs, surrogate_texts(ex, config)),
                "latency_ms": res["latency_ms"], "preproc_ms": preproc_ms, "decode_ms": decode_ms,
                "peak_mem_mb": res["peak_mem_mb"], "baseline_mem_mb": res["baseline_mem_mb"],
                "audio_sec": len(ex["audio"]) / cfg["dataset"]["audio_sr"],
                "device": str(scorer.device), "dtype": str(scorer.dtype).replace("torch.", ""),
                "model": cfg["model"]["hf_id"], "max_pixels": cfg["model"]["max_pixels"],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if args.check_generate:
                rec["generated"] = scorer.generate_text(inputs)
            append_jsonl(out, rec)
            n_run += 1
            if n_run % 10 == 0 or n_run == len(todo):
                el = time.time() - t_start
                eta = el / n_run * (len(todo) - n_run)
                log.info("%d/%d  %.1fs/pass  elapsed %.1f min  ETA %.1f min",
                         n_run, len(todo), el / n_run, el / 60, eta / 60)
    log.info("done: %d records written to %s", n_run, out)


if __name__ == "__main__":
    main()
