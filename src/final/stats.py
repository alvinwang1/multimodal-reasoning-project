"""Confidence intervals for the final-report numbers (CPU only: reads the downloaded result files, no model).

    python -m src.final.stats                  # tables in results/final/stats/ and a summary on screen
    python -m src.final.stats --n-boot 2000    # quicker, slightly noisier intervals

Outputs (results/final/stats/):
  accuracy.csv       every representation in the score bank (src/final/bank.py): accuracy, 95% CI, mean token cost
  by_audio_type.csv  the 7 base configs per audio type
  option_order.csv   the base configs in OmniBench's original option order vs the two shuffled orders
  comparisons.csv    the paired comparisons the report relies on (A - B on the same items)
  report.md          all of the above as markdown tables

Method. The unit is the item: an item's accuracy is the mean over its 3 option orders, so the orders of one item are
never counted as independent samples. 95% CIs are percentile bootstrap intervals over items (one set of resamples per
item set, shared by all its columns). A paired comparison uses the per-item difference A - B: a bootstrap CI of the
mean difference and a two-sided sign-flip permutation p-value; p_holm is Holm-adjusted within each family (base,
order, idea3, idea4, length). "clear" means p_holm < 0.05; when it is not clear, the CI shows how large a difference
the data still allow.
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from src.common import load_config, read_jsonl, repo_path
from src.data import AUDIO_TYPES, LETTERS
from src.final.bank import load_bank, matrix

OUT_DIR = "results/final/stats"
BASE = ["full", "img_trans", "cap_aud", "cap_trans", "img_only", "aud_only", "q_only"]
LEN_BASES = ["img_qdesc", "img_pipe", "gencap_pipe", "img_gendesc", "gen_cap_desc", "gen_cap_asr"]
LEN_ABLATION = (80, 120)  # the unsuffixed Idea 4 configs are the 256-token point

# (family, A, B, what A - B measures). Names as in the bank: base configs, "i3:" Idea 3 cells, "i4:" Idea 4 configs.
COMPARISONS = [
    ("base", "img_trans", "full", "human audio transcript instead of the raw audio"),
    ("base", "cap_aud", "full", "human image caption instead of the raw image"),
    ("base", "cap_trans", "full", "human caption + transcript instead of both raw inputs"),
    ("base", "full", "img_only", "adding the raw audio to the image"),
    ("base", "full", "aud_only", "adding the raw image to the audio"),
    ("base", "img_only", "q_only", "raw image alone vs the question alone"),
    ("base", "aud_only", "q_only", "raw audio alone vs the question alone"),
    ("idea3", "i3:v256_a1", "full", "256 instead of 768 image tokens"),
    ("idea3", "i3:v64_a1", "full", "64 instead of 768 image tokens"),
    ("idea3", "i3:v768_a0.5", "full", "first half of the audio only"),
    ("idea3", "i3:v768_a0.25", "full", "first quarter of the audio only"),
    ("idea3", "i3:v256_a0.5", "full", "256 image tokens and the first half of the audio"),
    ("idea3", "i3:v0_a0.25c", "i3:v0_a0.25", "centered vs leading quarter of the audio (no image)"),
    ("idea3", "i3:v0_a0.25e", "i3:v0_a0.25", "loudest vs leading quarter of the audio (no image)"),
    ("idea3", "i3:v0_a0.5c", "i3:v0_a0.5", "centered vs leading half of the audio (no image)"),
    ("idea3", "i3:v0_a0.5e", "i3:v0_a0.5", "loudest vs leading half of the audio (no image)"),
    ("idea3", "i3:v64_a0.25e", "i3:v64_a0.25", "loudest vs leading quarter of the audio (64 image tokens)"),
    ("idea3", "i3:v64_a0.5e", "i3:v64_a0.5", "loudest vs leading half of the audio (64 image tokens)"),
    ("idea4", "i4:img_pipe", "full", "pipeline text instead of the raw audio"),
    ("idea4", "i4:img_pipe", "img_trans", "pipeline text vs the human transcript"),
    ("idea4", "i4:img_pipe", "i4:img_asr", "pipeline text vs the Whisper transcript alone"),
    ("idea4", "i4:img_pipe", "i4:img_masked", "pipeline text vs the human transcript without option words"),
    ("idea4", "i4:img_route_fb", "i4:img_route", "raw-audio fallback for broken transcripts"),
    ("idea4", "i4:img_qdesc", "i4:img_gendesc", "question-guided vs generic audio description"),
    ("idea4", "i4:gencap_pipe", "cap_trans", "all generated text vs all human text"),
    ("idea4", "i4:img_masked", "img_trans", "masking the option words in the human transcript"),
    ("idea4", "i4:cap_masked", "cap_trans", "masking the option words (human caption + transcript)"),
    ("idea4", "i4:img_masked", "full", "masked human transcript vs the raw audio"),
] + [("length", f"i4:{b}_t{n}", f"i4:{b}", f"{n} vs 256 generated tokens") for b in LEN_BASES for n in LEN_ABLATION]


class Resampler:
    """Bootstrap weights and sign flips for one item set, from a fixed seed (reruns give identical intervals)."""

    def __init__(self, n, n_boot, seed):
        rng = np.random.default_rng(seed)
        self.n = n
        self.w = rng.multinomial(n, np.full(n, 1 / n), size=n_boot) / n  # (n_boot, n): row k = weights of resample k
        self.signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_boot, n))

    def ci(self, values):
        """values: (n,) or (n, k) -> percentile 95% CI of the mean(s)."""
        lo, hi = np.percentile(self.w @ np.asarray(values, dtype=float), [2.5, 97.5], axis=0)
        return lo, hi

    def p_signflip(self, d):
        """Two-sided paired permutation p-value for mean(d) = 0."""
        d = np.asarray(d, dtype=float)
        obs, hits = abs(d.mean()), 0
        for k in range(0, len(self.signs), 2500):
            null = np.abs(self.signs[k:k + 2500].astype(float) @ d) / self.n
            hits += int((null >= obs - 1e-9).sum())
        return (1 + hits) / (1 + len(self.signs))


def holm(p):
    p = np.asarray(p, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pct(x, signed=False):
    return f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"


def with_ci(est, lo, hi, signed=False):
    return f"{pct(est, signed)} [{pct(lo, signed)}, {pct(hi, signed)}]"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples and sign flips (default 10000)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(cfg)
    if bank.empty:
        raise SystemExit("the score bank is empty: unpack final_results.tar.gz in the repo root first")
    acc = matrix(bank, "acc", complete=False)
    cost = matrix(bank, "cost", complete=False)
    i4_ids = sorted(bank.loc[bank.source == "idea4", "id"].unique())
    samplers = {}

    def sampler(n):  # resamples depend only on the number of items (columns of one table share them)
        if n not in samplers:
            samplers[n] = Resampler(n, args.n_boot, [cfg["seed"], n])
        return samplers[n]

    # ---- accuracy of every representation (Idea 4 reps and the base configs on the Idea 4 items in their own rows)
    rows = []
    groups = [("all items", BASE + sorted(c for c in acc.columns if c.startswith("i3:")), acc.index),
              ("Idea 4 items", BASE + sorted(c for c in acc.columns if c.startswith("i4:")), i4_ids)]
    for items, reps, ids in groups:
        m = acc.loc[ids, reps].dropna()
        lo, hi = sampler(len(m)).ci(m.to_numpy())
        for j, rep in enumerate(reps):
            rows.append({"items": items, "rep": rep, "n_items": len(m), "acc": m[rep].mean(), "lo": lo[j], "hi": hi[j],
                         "cost": cost.loc[m.index, rep].mean()})
    acc_tab = pd.DataFrame(rows)
    acc_tab.to_csv(out_dir / "accuracy.csv", index=False)

    # ---- base configs per audio type
    atype = bank.drop_duplicates("id").set_index("id").audio_type
    rows = []
    for t in AUDIO_TYPES:
        m = acc.loc[acc.index[atype.reindex(acc.index).to_numpy() == t], BASE].dropna()
        lo, hi = sampler(len(m)).ci(m.to_numpy())
        rows += [{"audio_type": t, "rep": rep, "n_items": len(m), "acc": m[rep].mean(), "lo": lo[j], "hi": hi[j]}
                 for j, rep in enumerate(BASE)]
    type_tab = pd.DataFrame(rows)
    type_tab.to_csv(out_dir / "by_audio_type.csv", index=False)

    # ---- option order: the original OmniBench order (perm 0) vs the two shuffled orders, from the per-pass rows
    raw = pd.DataFrame(read_jsonl(repo_path(cfg["final"]["base_out"])))
    raw = raw.drop_duplicates(["id", "config", "perm"], keep="last")
    per_perm = raw.pivot_table(index="id", columns=["config", "perm"], values="correct", aggfunc="mean").dropna()
    rs = sampler(len(per_perm))
    rows = []
    for c in BASE:
        p0, sh = per_perm[(c, 0)].to_numpy(), per_perm[[(c, 1), (c, 2)]].mean(axis=1).to_numpy()
        lo, hi = rs.ci(np.c_[p0, sh])
        d = p0 - sh
        dlo, dhi = rs.ci(d)
        rows.append({"family": "order", "rep": c, "n_items": len(d), "acc_original": p0.mean(), "lo_original": lo[0],
                     "hi_original": hi[0], "acc_shuffled": sh.mean(), "lo_shuffled": lo[1], "hi_shuffled": hi[1],
                     "diff": d.mean(), "diff_lo": dlo, "diff_hi": dhi, "p": rs.p_signflip(d)})
    order_tab = pd.DataFrame(rows)
    order_tab["p_holm"] = holm(order_tab.p)
    order_tab.to_csv(out_dir / "option_order.csv", index=False)
    full0 = raw[(raw.config == "full") & (raw.perm == 0)]
    gold = Counter(LETTERS[o.index(a)] for o, a in zip(full0.perm_order, full0.answer_idx))
    pick = {p: Counter(LETTERS[o.index(a)] for o, a in zip(g.perm_order, g.pred_idx))
            for p, g in raw[raw.config == "full"].groupby("perm")}

    # ---- paired comparisons
    rows = []
    for fam, a, b, what in COMPARISONS:
        if a not in acc.columns or b not in acc.columns:
            print(f"skipped {a} - {b}: not in the bank")
            continue
        m = acc[[a, b]].dropna()
        r = sampler(len(m))
        d = (m[a] - m[b]).to_numpy()
        lo, hi = r.ci(d)
        rows.append({"family": fam, "A": a, "B": b, "what": what, "n_items": len(m), "acc_A": m[a].mean(),
                     "acc_B": m[b].mean(), "diff": d.mean(), "lo": lo, "hi": hi, "p": r.p_signflip(d),
                     "items_differ": int((d != 0).sum())})
    cmp_tab = pd.DataFrame(rows)
    cmp_tab["p_holm"] = cmp_tab.groupby("family").p.transform(holm)
    cmp_tab["clear"] = cmp_tab.p_holm < 0.05
    cmp_tab.to_csv(out_dir / "comparisons.csv", index=False)

    # ---- report.md
    n = len(full0)
    md = [f"# Confidence intervals ({args.n_boot:,} bootstrap resamples over items; 95% CIs in brackets)\n",
          "Accuracy is the mean over the 3 option orders of each item. Paired comparisons: A - B on the same items, "
          "sign-flip permutation p-value, Holm-adjusted within each family. Generated by `python -m src.final.stats`.\n",
          "## Paired comparisons\n"]
    for fam, g in cmp_tab.groupby("family", sort=False):
        md += [f"### {fam}\n", md_table(pd.DataFrame({
            "A - B": g.A + " - " + g.B, "what": g.what, "items": g.n_items, "A": g.acc_A.map(pct), "B": g.acc_B.map(pct),
            "difference (pts)": [with_ci(*v, signed=True) for v in zip(g["diff"], g.lo, g.hi)],
            "items differ": g.items_differ, "p_holm": g.p_holm.map(lambda p: f"{p:.4f}"),
            "clear": g.clear.map({True: "yes", False: "no"})})), ""]
    md += ["## Option order\n",
           f"Gold answer letter in OmniBench's original order (n={n}): "
           + ", ".join(f"{L} {100 * gold[L] / n:.1f}%" for L in LETTERS)
           + f". Always answering {max(gold, key=gold.get)} would score {100 * max(gold.values()) / n:.1f}% in that order.\n",
           "Letters the model picks with the full input: " + "; ".join(
               f"order {p}: " + ", ".join(f"{L} {100 * c[L] / sum(c.values()):.0f}%" for L in LETTERS)
               for p, c in pick.items()) + "\n",
           md_table(pd.DataFrame({
               "config": order_tab.rep,
               "original order": [with_ci(*v) for v in zip(order_tab.acc_original, order_tab.lo_original, order_tab.hi_original)],
               "shuffled orders": [with_ci(*v) for v in zip(order_tab.acc_shuffled, order_tab.lo_shuffled, order_tab.hi_shuffled)],
               "original - shuffled": [with_ci(*v, signed=True) for v in zip(order_tab["diff"], order_tab.diff_lo, order_tab.diff_hi)],
               "p_holm": order_tab.p_holm.map(lambda p: f"{p:.4f}")})), ""]
    md += ["## Accuracy by audio type (base configs)\n", md_table(type_tab.assign(
        acc=[with_ci(*v) for v in zip(type_tab.acc, type_tab.lo, type_tab.hi)]).pivot(
        index="rep", columns="audio_type", values="acc").reindex(BASE).reset_index().rename(columns={"rep": "config"})),
        "", "n items: " + ", ".join(f"{t} {type_tab[type_tab.audio_type == t].n_items.iloc[0]}" for t in AUDIO_TYPES), ""]
    md += ["## Accuracy of every representation\n"]
    for items, g in acc_tab.groupby("items", sort=False):
        md += [f"### {items} (n={g.n_items.iloc[0]})\n", md_table(pd.DataFrame({
            "rep": g.rep, "accuracy": [with_ci(*v) for v in zip(g.acc, g.lo, g.hi)],
            "token cost vs full": g.cost.map(lambda c: f"{c:.2f}")})), ""]
    (out_dir / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwritten: {out_dir}/report.md, accuracy.csv, by_audio_type.csv, option_order.csv, comparisons.csv")


if __name__ == "__main__":
    main()
