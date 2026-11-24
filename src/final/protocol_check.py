"""Why is our full-input accuracy (42.3%) below Qwen's published OmniBench score for Qwen2.5-Omni-3B (52.19%)?

Our evaluation differs from a typical OmniBench evaluation in three ways; this run measures each one on all items:
  image size    we cap images at 768 tokens (configs/default.yaml model.max_pixels); qwen-omni-utils' default
                allows up to 16,384 tokens per image
  scoring       we read the answer from the letter logits of one forward pass; a typical evaluation lets the model
                write its answer and parses the letter (or the option text) from it
  option order  we average 3 option orders; in OmniBench's original order the answer is D for ~49% of items

Each pass is one greedy generate() call on the full input (image + audio + question). Its first-step logits give the
letter-logit answer (as in the score bank) and its text gives the written answer, so both scorings see the same input.
Conditions: image cap {768, qwen_default} x option order {0 = original, 1 = the score bank's first shuffle}, i.e.
4 passes per item (about 1 h on an L4). The 768-cap letter-logit answers should reproduce the score bank's `full`
rows; the summary checks that.

    python -m src.final.protocol_check               # run (GPU; resumable), then write the summary
    python -m src.final.protocol_check --limit 2     # quick check: the first 2 items
    python -m src.final.protocol_check --summary     # summary only (CPU), from results/final/protocol_check.jsonl
    On Modal: modal run --detach scripts/modal_run.py --sections protocol

Output: results/final/protocol_check.jsonl (one row per item x image cap x order) and results/final/protocol_check.md.
"""
import argparse
import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from src.common import append_jsonl, load_config, read_jsonl, repo_path, seed_everything
from src.data import AUDIO_TYPES, LETTERS

log = logging.getLogger("protocol_check")
OUT, SUMMARY = "results/final/protocol_check.jsonl", "results/final/protocol_check.md"
TOKEN_PX = 28 * 28  # pixels per image token after Qwen's 2x2 patch merge
CAPS = {"768": 768, "qwen_default": 16384}  # image tokens; 16384 * 28 * 28 is qwen-omni-utils' MAX_PIXELS
PERMS = [0, 1]
# Qwen2.5-Omni README (github.com/QwenLM/Qwen2.5-Omni), OmniBench, Qwen2.5-Omni-3B
PUBLISHED = {"speech": 0.5214, "sound event": 0.5208, "music": 0.5283, "overall": 0.5219}

# written answer -> displayed letter: "B" / "(B)" / "B." ; "B. text" / "B) text" / "B: text" ; "The answer is B"
LETTER_PATTERNS = [re.compile(r"^\W*([A-D])\W*$"), re.compile(r"^\W*([A-D])[.):]"),
                   re.compile(r"(?i:answer|option|choice)\s*(?i:is)?\s*:?\s*\(?([A-D])\b")]


def parse_answer(text, shown):
    """Written answer -> (displayed option index or None, how it was parsed). shown: option texts in display order.
    Falls back to the option text when exactly one option appears in the answer; otherwise unparsed (= wrong)."""
    t = text.strip()
    for k, pat in enumerate(LETTER_PATTERNS):
        m = pat.search(t)
        if m:
            return LETTERS.index(m.group(1)), f"letter{k}"
    low = t.lower()
    hits = [j for j, o in enumerate(shown) if str(o).strip() and str(o).strip().lower() in low]
    if len(hits) == 1:
        return hits[0], "option_text"
    return None, "unparsed"


def answer(scorer, inputs, eos, max_new_tokens):
    """One greedy generate(): letter logits of the first new token, and the written answer."""
    import torch

    from src.analysis_a.model import PeakMemory, sync

    tok = scorer.processor.tokenizer
    with torch.inference_mode():
        inputs = inputs.to(scorer.device)
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(scorer.dtype)
        n_in = inputs["input_ids"].shape[1]
        sync(scorer.device)
        with PeakMemory(scorer.device) as mem:
            t0 = time.perf_counter()
            out = scorer.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos,
                                        pad_token_id=eos[1], return_dict_in_generate=True, output_logits=True)
            sync(scorer.device)
            ms = (time.perf_counter() - t0) * 1000
        first = out.logits[0][0].float()  # raw logits of the first new token (before any logits processor)
        gen = out.sequences[0, n_in:]
    return {"letter_logits": first[scorer.letter_ids].cpu().tolist(), "top_vocab_token": tok.decode([int(first.argmax())]),
            "generated": tok.decode(gen, skip_special_tokens=True), "gen_new_tokens": int(gen.numel()),
            "gen_ms": ms, "peak_mem_mb": mem.peak_mb}


def run(args, cfg):
    import torch

    from src.analysis_a.model import OmniScorer
    from src.analysis_a.prompts import build_messages, permutation
    from src.data import OmniBenchSubset

    seed = cfg["seed"]
    seed_everything(seed)
    out = repo_path(args.out)
    ids = args.ids or sorted(i for f in json.loads(repo_path(cfg["final"]["folds"]).read_text())["folds"] for i in f)
    if args.limit:
        ids = ids[: args.limit]
    done = {(r["id"], r["image_cap"], r["perm"]) for r in read_jsonl(out) if "image_cap" in r}
    todo = [(i, c, p) for i in ids for c in args.caps for p in args.perms if (i, c, p) not in done]
    log.info("%d items x caps %s x orders %s | %d done, %d to run -> %s", len(ids), args.caps, args.perms,
             len(done), len(todo), out)
    if not todo:
        return
    cap_px = {c: CAPS[c] * TOKEN_PX for c in CAPS}
    if cfg["model"]["max_pixels"] != cap_px["768"]:
        raise SystemExit(f"configs model.max_pixels is {cfg['model']['max_pixels']}, not the score bank's "
                         f"768-token cap ({cap_px['768']}): the 768 condition would not match the bank")
    try:  # the documented default of qwen-omni-utils; warn if this version has another one
        from qwen_omni_utils.v2_5.vision_process import MAX_PIXELS
        if MAX_PIXELS != cap_px["qwen_default"]:
            log.info("note: qwen-omni-utils MAX_PIXELS is %d (%d tokens), not %d; using %d", MAX_PIXELS,
                     MAX_PIXELS // TOKEN_PX, cap_px["qwen_default"], cap_px["qwen_default"])
    except ImportError:
        pass

    data = OmniBenchSubset(cfg)
    scorer = OmniScorer(cfg)
    base_px = scorer.max_pixels
    tok = scorer.processor.tokenizer
    eos = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    log.info("model on %s (%s); letter ids %s; eos %s", scorer.device, scorer.dtype, scorer.letter_ids, eos)

    by_item = {}
    for j in todo:
        by_item.setdefault(j[0], []).append(j)
    t_start, n_run = time.time(), 0
    for idx, jobs in by_item.items():
        ex = data[idx]
        perms = {}
        for p in range(max(args.perms) + 1):  # the same permutations as the score bank (src.analysis_a.run)
            perms[p] = permutation(idx, p, seed, prior=[perms[q] for q in range(p)])
        for _, cap, p in jobs:
            perm = perms[p]
            messages = build_messages(ex, "full", perm)
            for c in messages[-1]["content"]:
                if c["type"] == "image":
                    c["max_pixels"] = cap_px[cap]
            rec = {"id": idx, "image_cap": cap, "max_pixels": cap_px[cap], "perm": p, "perm_order": perm,
                   "audio_type": ex["audio type"], "task_type": ex["task type"], "answer_idx": ex["answer_idx"],
                   "image_size": list(ex["image"].size)}
            scorer.max_pixels = cap_px[cap]  # OmniScorer.prepare uses min(this, the message's max_pixels)
            try:
                t0 = time.perf_counter()
                _, inputs = scorer.prepare(messages)
                rec["preproc_ms"] = (time.perf_counter() - t0) * 1000
                rec.update(scorer.token_counts(inputs))
                rec.update(answer(scorer, inputs, eos, args.max_new_tokens))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec["error"] = "CUDA out of memory"
                log.info("id %d cap %s order %d: CUDA out of memory (image %s); recorded as an error", idx, cap, p,
                         rec["image_size"])
            finally:
                scorer.max_pixels = base_px
            if "error" not in rec:
                j, how = parse_answer(rec["generated"], [ex["options"][o] for o in perm])
                k = int(np.argmax(rec["letter_logits"]))
                rec.update({"logit_pred_idx": perm[k], "logit_correct": perm[k] == ex["answer_idx"],
                            "gen_pred_idx": None if j is None else perm[j], "parse": how,
                            "gen_correct": j is not None and perm[j] == ex["answer_idx"]})
            rec.update({"model": cfg["model"]["hf_id"], "dtype": str(scorer.dtype).replace("torch.", ""),
                        "device": str(scorer.device), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            append_jsonl(out, rec)
            n_run += 1
            if n_run % 20 == 0 or n_run == len(todo):
                el = time.time() - t_start
                log.info("%d/%d  %.2fs/pass  elapsed %.1f min  ETA %.1f min", n_run, len(todo), el / n_run, el / 60,
                         el / n_run * (len(todo) - n_run) / 60)
    log.info("done: %d passes written to %s", n_run, out)


def summarize(cfg, path, md_path, n_boot):
    import pandas as pd

    from src.final.stats import Resampler, holm, md_table, pct, with_ci

    df = pd.DataFrame(read_jsonl(path))
    if df.empty:
        print(f"no rows yet in {path}")
        return
    df = df.drop_duplicates(["id", "image_cap", "perm"], keep="last")
    err = df[df["error"].notna()] if "error" in df else df.iloc[:0]
    ok = df.drop(err.index)
    caps = [c for c in CAPS if c in set(ok.image_cap)]
    perms = sorted(int(p) for p in ok.perm.unique())
    long = pd.concat([ok.assign(scoring="letter logits", correct=ok.logit_correct.astype(float)),
                      ok.assign(scoring="written answer", correct=ok.gen_correct.astype(float))])
    wide = long.pivot_table(index="id", columns=["scoring", "image_cap", "perm"], values="correct",
                            aggfunc="mean").dropna()  # items with every condition, so all comparisons are paired
    n = len(wide)
    if not n:
        print("no item has every condition yet")
        return
    rs = Resampler(n, n_boot, [cfg["seed"], n])
    order_name = {0: "original order", 1: "shuffled order"}
    cap_name = {"768": "768 tokens (ours)", "qwen_default": "Qwen default (16,384)"}
    md = [f"# Protocol check: full input, {n} items ({n_boot:,} bootstrap resamples; 95% CIs in brackets)\n",
          "Generated by `python -m src.final.protocol_check --summary`. Qwen's published Qwen2.5-Omni-3B OmniBench "
          f"score: {pct(PUBLISHED['overall'])}% overall.\n"]

    def side(cols):
        return wide[cols].mean(axis=1).to_numpy()

    # accuracy per condition
    rows = []
    for cap in caps:
        for sc in ("letter logits", "written answer"):
            cols = [(sc, cap, p) for p in perms]
            m = np.c_[[wide[c].to_numpy() for c in cols] + ([side(cols)] if len(perms) > 1 else [])].T
            lo, hi = rs.ci(m)
            row = {"image cap": cap_name[cap], "scoring": sc}
            for j, p in enumerate(perms):
                row[order_name.get(p, f"order {p}")] = with_ci(m[:, j].mean(), lo[j], hi[j])
            if len(perms) > 1:
                row["mean of the orders"] = with_ci(m[:, -1].mean(), lo[-1], hi[-1])
            rows.append(row)
    md += ["## Accuracy\n", md_table(pd.DataFrame(rows)), ""]

    # paired comparisons
    def cols(sc, cap, ps=None):
        return [(sc, cap, p) for p in (perms if ps is None else ps) if (sc, cap, p) in wide.columns]

    L, W, D = "letter logits", "written answer", "qwen_default"
    specs = [("image cap: Qwen default - 768 (letter logits)", cols(L, D), cols(L, "768")),
             ("image cap: Qwen default - 768 (written answer)", cols(W, D), cols(W, "768")),
             ("scoring: written - letter logits (768 cap)", cols(W, "768"), cols(L, "768")),
             ("scoring: written - letter logits (Qwen default cap)", cols(W, D), cols(L, D)),
             ("order: original - shuffled (letter logits, 768 cap)", cols(L, "768", [0]), cols(L, "768", [1])),
             ("order: original - shuffled (written, Qwen default cap)", cols(W, D, [0]), cols(W, D, [1])),
             ("typical protocol (written, Qwen default cap, original order) - ours (letter logits, 768, both orders)",
              cols(W, D, [0]), cols(L, "768"))]
    rows = []
    for name, a, b in specs:
        if not a or not b or set(a) == set(b):
            continue
        d = side(a) - side(b)
        lo, hi = rs.ci(d)
        rows.append({"A - B": name, "A": pct(side(a).mean()), "B": pct(side(b).mean()),
                     "difference (pts)": with_ci(d.mean(), lo, hi, signed=True), "p": rs.p_signflip(d)})
    if rows:
        t = pd.DataFrame(rows)
        t["p_holm"] = [f"{p:.4f}" for p in holm(t.p)]
        md += ["## What each difference is worth\n", md_table(t.drop(columns="p")), ""]

    # per audio type: the typical protocol and ours, next to the published numbers
    atype = ok.drop_duplicates("id").set_index("id").audio_type.reindex(wide.index)
    typical, ours = cols(W, D, [0]), cols(L, "768")
    if typical and ours:
        rows = []
        for t in AUDIO_TYPES + ["overall"]:
            mask = np.ones(n, bool) if t == "overall" else (atype == t).to_numpy()
            if not mask.any():
                continue
            r = Resampler(int(mask.sum()), n_boot, [cfg["seed"], int(mask.sum())])
            m = np.c_[side(typical)[mask], side(ours)[mask]]
            lo, hi = r.ci(m)
            rows.append({"audio type": t, "items": int(mask.sum()), "published (Qwen)": pct(PUBLISHED[t]),
                         "typical protocol": with_ci(m[:, 0].mean(), lo[0], hi[0]),
                         "ours (letter logits, 768, both orders)": with_ci(m[:, 1].mean(), lo[1], hi[1])})
        md += ["## By audio type\n", md_table(pd.DataFrame(rows)), ""]

    # checks: same answers as the score bank, written-answer parsing, letters picked, image tokens, errors
    bank = pd.DataFrame(read_jsonl(repo_path(cfg["final"]["base_out"])))
    if len(bank) and "768" in caps:
        b = bank[bank.config == "full"].drop_duplicates(["id", "perm"], keep="last").set_index(["id", "perm"]).pred_idx
        o = ok[ok.image_cap == "768"].set_index(["id", "perm"]).logit_pred_idx
        common = o.index.intersection(b.index)
        agree = (o.loc[common] == b.loc[common]).mean() if len(common) else float("nan")
        md.append(f"- Score-bank check: the 768-cap letter-logit answer equals the bank's `full` answer on "
                  f"{100 * agree:.1f}% of {len(common)} passes (expected ~100%).")
    parse = ok.groupby("image_cap").parse.value_counts().unstack(fill_value=0)
    md.append("- Written-answer parsing (passes): " + "; ".join(
        f"{cap_name.get(c, c)}: " + ", ".join(f"{k} {v}" for k, v in parse.loc[c].items() if v) for c in parse.index))
    unparsed = ok[ok.parse == "unparsed"].generated.head(3).tolist()
    if unparsed:
        md.append("- Unparsed examples (counted wrong): " + " | ".join(repr(u[:80]) for u in unparsed))
    for cap in caps:
        for sc, col in (("letter logits", "logit_pred_idx"), ("written answer", "gen_pred_idx")):
            parts = []
            for p in perms:
                g = ok[(ok.image_cap == cap) & (ok.perm == p) & ok[col].notna()]
                c = Counter(LETTERS[list(o).index(int(x))] for o, x in zip(g.perm_order, g[col]))
                tot = max(sum(c.values()), 1)
                parts.append(f"{order_name.get(p, p)} " + " ".join(f"{L_}{100 * c[L_] / tot:.0f}" for L_ in LETTERS))
            md.append(f"- Letters picked (%), {cap_name[cap]}, {sc}: " + "; ".join(parts))
    gold = ok[(ok.perm == 0)].drop_duplicates("id")
    if len(gold):
        g = Counter(LETTERS[a] for a in gold.answer_idx)
        md.append("- Gold letter in the original order (%): " + " ".join(f"{L_}{100 * g[L_] / len(gold):.0f}"
                                                                        for L_ in LETTERS))
    for cap in caps:
        g = ok[ok.image_cap == cap]
        md.append(f"- {cap_name[cap]}: image tokens median {g.n_image_tok.median():.0f}, max {g.n_image_tok.max():.0f}; "
                  f"generate() median {g.gen_ms.median():.0f} ms; peak GPU memory {g.peak_mem_mb.max():.0f} MB")
    md.append(f"- Errors: {len(err)} pass(es)" + (f" ({Counter(err.error).most_common()})" if len(err) else ""))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten: {md_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--summary-out", default=SUMMARY)
    ap.add_argument("--limit", type=int, default=None, help="only the first N items")
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="explicit OmniBench indices")
    ap.add_argument("--caps", nargs="*", default=list(CAPS), choices=list(CAPS))
    ap.add_argument("--perms", type=int, nargs="*", default=PERMS, help="option orders (0 = original)")
    ap.add_argument("--max-new-tokens", type=int, default=32, help="length of the written answer")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--summary", action="store_true", help="only summarize the existing output (CPU)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if not args.summary:
        run(args, cfg)
    summarize(cfg, repo_path(args.out), repo_path(args.summary_out), args.n_boot)


if __name__ == "__main__":
    main()
