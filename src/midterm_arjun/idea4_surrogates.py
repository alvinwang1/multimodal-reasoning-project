"""Idea 4 baseline: raw modalities vs. *generated* text surrogates (test split, no training).

Compares the backbone's accuracy with OmniBench's human-written surrogates (gold caption / transcript,
from the Idea-1 run) against model-generated ones (Whisper ASR, Qwen2.5-Omni captions and audio
descriptions; src/midterm/surrogates.py), and counts the generation cost.

    python -m src.midterm.idea4_surrogates
"""
import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis_a.prompts import config_spec
from src.common import (AUDIO_TYPE_COLORS, COL_W, CONFIG_LABELS, INK, INK_2, PALETTE, load_config, read_jsonl,
                        repo_path, save_fig, set_plot_style)
from src.data import AUDIO_TYPES, load_omnibench, text_records

GOLD = ["full", "img_trans", "cap_aud", "cap_trans", "img_only", "aud_only", "q_only"]
GEN = ["gen_cap_desc", "gen_cap_asr", "img_gendesc", "img_asr", "gencap_aud"]
LABELS = {**CONFIG_LABELS, "gen_cap_desc": "Gen. caption + Gen. audio desc.",
          "gen_cap_asr": "Gen. caption + ASR", "img_gendesc": "Image + Gen. audio desc.",
          "img_asr": "Image + ASR", "gencap_aud": "Gen. caption + Audio"}
PAIRS = [("gen_cap_desc", "cap_trans"), ("gen_cap_asr", "cap_trans"), ("img_gendesc", "img_trans"),
         ("img_asr", "img_trans"), ("gencap_aud", "cap_aud"), ("gen_cap_desc", "full"), ("img_asr", "full")]


def tokens(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def token_f1(pred, ref):
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def gen_cost_ms(config, g):
    """Generation latency needed before answering with this config."""
    spec = config_spec(config)
    ms = 0.0
    if spec["img"] == "gen":
        ms += g["caption_ms"]
    if spec["aud"] == "gen_desc":
        ms += g["desc_ms"]
    if spec["aud"] == "asr":
        ms += g["asr_ms"]
    return ms


def paired_boot(a, b, seed=42, n=5000):
    d = (a - b).values
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_plot_style()
    out = repo_path(cfg["midterm"]["out_dir"]) / "idea4"
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(repo_path(cfg["midterm"]["split"]).read_text())
    test = set(split["test"])

    gen_text = json.loads(repo_path(cfg["midterm"]["surrogates"]).read_text())
    base = pd.DataFrame(read_jsonl(repo_path(cfg["analysis_a"]["out"])))
    surr = pd.DataFrame(read_jsonl(repo_path(cfg["midterm"]["surrogate_out"])))
    df = pd.concat([base[base.config.isin(GOLD)], surr[surr.config.isin(GEN)]], ignore_index=True)
    df = df[df.id.isin(test)].drop_duplicates(["id", "config", "perm"], keep="last")
    counts = df.groupby("id").size()
    complete = [i for i in counts[counts == (len(GOLD) + len(GEN)) * 3].index if str(i) in gen_text]
    df = df[df.id.isin(complete)]
    n = len(complete)
    print(f"complete test items: {n}/{len(test)}")
    if n == 0:
        return

    acc = df.groupby(["id", "config"]).correct.mean().unstack()
    p0 = df[df.perm == 0].set_index(["id", "config"])
    meta = p0.xs("full", level="config")[["audio_type", "task_type"]]
    g = pd.DataFrame(gen_text).T
    g.index = g.index.astype(int)
    g = g.loc[complete]
    for c in ["caption_ms", "desc_ms", "asr_ms", "caption_new_tok", "desc_new_tok"]:
        g[c] = g[c].astype(float)

    rows = []
    for c in GOLD + GEN:
        lat = df[df.config == c].groupby("id").latency_ms.mean()
        gen_ms = pd.Series({i: gen_cost_ms(c, g.loc[i]) for i in complete})
        spec = config_spec(c)
        gen_tok = (g.caption_new_tok * (spec["img"] == "gen") + g.desc_new_tok * (spec["aud"] == "gen_desc"))
        row = {"config": c, "label": LABELS[c], "surrogates": "generated" if c in GEN else "gold/raw",
               "acc": acc[c].mean(), "answer_tokens": p0.xs(c, level="config").n_total_tok.mean(),
               "answer_latency_s": lat.mean() / 1000, "generation_s": gen_ms.mean() / 1000,
               "total_latency_s": (lat + gen_ms).mean() / 1000, "generated_tokens": gen_tok.mean()}
        for at in AUDIO_TYPES:
            row[f"acc_{at.split()[0]}"] = acc.loc[meta.audio_type == at, c].mean()
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "results_test.csv", index=False)
    print("\nTest accuracy (mean over 3 orders) and cost; generation cost counts Whisper/Qwen generation time:")
    print(table.drop(columns=["config"]).round(3).to_string(index=False))

    comp = []
    for a, b in PAIRS:
        for at in ["all"] + AUDIO_TYPES:
            m = slice(None) if at == "all" else (meta.audio_type == at)
            d, lo, hi = paired_boot(acc.loc[m, a], acc.loc[m, b])
            comp.append({"generated": LABELS[a], "reference": LABELS[b], "audio_type": at, "diff_pts": d * 100,
                         "ci_lo": lo * 100, "ci_hi": hi * 100})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "paired_differences.csv", index=False)
    print("\nPaired accuracy difference, generated - reference (points, bootstrap 95% CI):")
    print(comp[comp.audio_type == "all"].round(1).to_string(index=False))

    # surrogate quality vs. the gold (human-written) surrogates
    records, _ = text_records(load_omnibench(cfg))
    by_id = {r["index"]: r for r in records}
    q = pd.DataFrame({
        "audio_type": meta.audio_type,
        "asr_words": [len(tokens(g.at[i, "asr"])) for i in complete],
        "asr_f1_vs_gold_transcript": [token_f1(g.at[i, "asr"], by_id[i]["audio content"]) for i in complete],
        "desc_f1_vs_gold_transcript": [token_f1(g.at[i, "gen_audio_desc"], by_id[i]["audio content"]) for i in complete],
        "caption_f1_vs_gold_caption": [token_f1(g.at[i, "gen_caption"], by_id[i]["image content"]) for i in complete],
        "gen_caption_words": [len(tokens(g.at[i, "gen_caption"])) for i in complete],
        "gold_caption_words": [len(tokens(by_id[i]["image content"])) for i in complete],
        "drop_text_route": acc.loc[complete, "cap_trans"] - acc.loc[complete, "gen_cap_desc"],
    }, index=complete)
    q["asr_near_empty"] = q.asr_words < 3
    qual = q.groupby("audio_type").agg(
        asr_words=("asr_words", "median"), asr_near_empty=("asr_near_empty", "mean"),
        asr_f1=("asr_f1_vs_gold_transcript", "mean"), desc_f1=("desc_f1_vs_gold_transcript", "mean"),
        caption_f1=("caption_f1_vs_gold_caption", "mean"), gen_caption_words=("gen_caption_words", "median"),
        gold_caption_words=("gold_caption_words", "median")).reindex(AUDIO_TYPES)
    qual.to_csv(out / "surrogate_quality.csv")
    q.to_csv(out / "surrogate_quality_per_item.csv")
    print("\nSurrogate quality by audio type (token F1 against the human-written surrogate):")
    print(qual.round(3).to_string())
    r = np.corrcoef(q.desc_f1_vs_gold_transcript, q.drop_text_route)[0, 1] if q.drop_text_route.std() > 0 else np.nan
    print(f"correlation(audio-description F1, accuracy drop gold->generated text route): r = {r:.2f}")

    # failure cases: the text route works with gold surrogates but fails with generated ones
    fails = q[(acc.loc[complete, "cap_trans"] >= 2 / 3) & (acc.loc[complete, "gen_cap_desc"] <= 1 / 3)]
    fc = pd.DataFrame([{
        "id": i, "audio_type": meta.at[i, "audio_type"], "task_type": meta.at[i, "task_type"],
        "question": by_id[i]["question"], "gold_transcript": by_id[i]["audio content"],
        "gen_audio_desc": g.at[i, "gen_audio_desc"], "asr": g.at[i, "asr"],
        "gold_caption": by_id[i]["image content"][:300], "gen_caption": str(g.at[i, "gen_caption"])[:300],
        "acc_gold_text": acc.at[i, "cap_trans"], "acc_gen_text": acc.at[i, "gen_cap_desc"],
        "acc_full": acc.at[i, "full"]} for i in fails.index])
    fc.to_csv(out / "failure_cases.csv", index=False)
    print(f"\nitems where gold text works but generated text fails: {len(fc)} "
          f"({fc.audio_type.value_counts().to_dict() if len(fc) else {}}) -> failure_cases.csv")

    # ---------------- figures
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    show = [("cap_trans", "Gold caption + transcript", INK_2), ("gen_cap_desc", "Gen. caption + gen. desc.", PALETTE[0]),
            ("gen_cap_asr", "Gen. caption + ASR", PALETTE[1]), ("full", "Image + Audio (raw)", PALETTE[2])]
    x = np.arange(len(AUDIO_TYPES) + 1)
    w = 0.2
    for k, (c, lab, col) in enumerate(show):
        vals = [acc.loc[meta.audio_type == at, c].mean() * 100 for at in AUDIO_TYPES] + [acc[c].mean() * 100]
        ax.bar(x + (k - 1.5) * w, vals, width=w - 0.02, color=col, label=lab)
    ax.set_xticks(x, AUDIO_TYPES + ["all"])
    ax.set_ylabel("Test accuracy (%)")
    ax.axhline(25, color=INK_2, ls="--", lw=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    save_fig(fig, out, "gold_vs_generated")

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for _, r_ in table.iterrows():
        col = PALETTE[0] if r_.surrogates == "generated" else INK_2
        ax.scatter(r_.total_latency_s, r_.acc * 100, s=20, color=col, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(r_.label, (r_.total_latency_s, r_.acc * 100), xytext=(3, 2), textcoords="offset points",
                    fontsize=4.8, color=INK if r_.surrogates == "generated" else INK_2)
    ax.set_xlabel("Total latency per question (s), incl. generating surrogates")
    ax.set_ylabel("Test accuracy (%)")
    save_fig(fig, out, "accuracy_vs_total_latency")

    (out / "summary.json").write_text(json.dumps({"n_test_items": n, "results": table.round(4).to_dict(orient="records"),
                                                  "paired": comp.round(3).to_dict(orient="records"),
                                                  "quality": qual.round(4).reset_index().to_dict(orient="records")},
                                                 indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
