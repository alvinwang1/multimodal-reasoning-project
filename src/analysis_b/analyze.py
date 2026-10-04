"""Analysis B: staged-reveal human annotations.

    python -m src.analysis_b.analyze

Reports
  - sufficiency prediction: stage-0 "is text enough?" vs. whether text actually sufficed
    (the annotator's own text-only answer being correct)
  - steps (reveals) and token cost per item and per correct answer; step costs are the
    per-example modality token counts measured in Analysis A (results/analysis_a/modality_costs.json)
  - breakdowns by audio type and task type
  - Fleiss' kappa on the shared items (text-enough judgment, final answer)
  - human text-enough vs. the model's text-sufficient labels (Analysis A)
Writes results/analysis_b/human_text_sufficient.json for `analysis_c ... --labels human`.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from statsmodels.stats.proportion import proportion_confint

from src.common import (AUDIO_TYPE_COLORS, COL_W, CONFIG_COLORS, CONFIG_LABELS, CONFIG_ORDER, INK, INK_2,
                        PALETTE, load_config, read_jsonl, repo_path, save_fig, set_plot_style)
from src.data import AUDIO_TYPES, load_omnibench, text_records

REVEAL_PATTERNS = ["text only", "+image", "+audio", "+both"]


def load_annotations(cfg):
    bcfg = cfg["analysis_b"]
    rows = []
    for a in bcfg["annotators"]:
        for r in read_jsonl(repo_path(bcfg["results_dir"]) / f"{a}.jsonl"):
            rows.append(r)
    if not rows:
        raise SystemExit("no annotations found")
    df = pd.DataFrame(rows).drop_duplicates(["annotator", "id"], keep="last")
    return df


def item_cost(r, costs, means):
    c = costs.get(str(r["id"]), means)
    tok = c["base_text"] + c["caption"] + c["transcript"]
    tok += c["image"] * r["revealed"]["image"] + c["audio"] * r["revealed"]["audio"]
    full = c["base_text"] + c["image"] + c["audio"]
    return tok, tok / full


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_plot_style()
    out = repo_path(cfg["analysis_b"]["results_dir"])

    df = load_annotations(cfg)
    records, _ = text_records(load_omnibench(cfg))
    gold = {r["index"]: r["answer_idx"] for r in records}

    mc_path = repo_path("results/analysis_a/modality_costs.json")
    if not mc_path.exists():
        raise SystemExit("run `python -m src.analysis_a.analyze` first (needs modality_costs.json)")
    mc = json.loads(mc_path.read_text())
    costs, means = mc["per_example"], mc["mean"]

    df["gold"] = df.id.map(gold)
    df["final_correct"] = df.final.map(lambda f: f["answer_idx"]) == df.gold
    df["text_enough"] = df.stage0.map(lambda s: s["text_enough"])
    df["conf0"] = df.stage0.map(lambda s: s["conf"])
    df["stage0_answer"] = df.stage0.map(lambda s: s["answer_idx"])
    df["stage0_correct"] = np.where(df.stage0_answer.isna(), np.nan, df.stage0_answer == df.gold)
    # Did text actually suffice for this annotator? Their text-only answer: the stage-0 answer if given,
    # else the final answer when they answered without revealing anything.
    df["text_only_correct"] = np.where(~df.stage0_answer.isna(), df.stage0_correct,
                                       np.where(df.n_reveals == 0, df.final_correct, np.nan))
    df["img"] = df.revealed.map(lambda r: r["image"])
    df["aud"] = df.revealed.map(lambda r: r["audio"])
    df["pattern"] = [REVEAL_PATTERNS[int(i) + 2 * int(a)] for i, a in zip(df.img, df.aud)]
    df["first_reveal"] = df.steps.map(lambda s: next((x["action"] for x in s if x["action"] != "answer"), "none"))
    tc = [item_cost(r, costs, means) for _, r in df.iterrows()]
    df["tokens"], df["cost"] = [t for t, _ in tc], [c for _, c in tc]
    n_missing_cost = (~df.id.astype(str).isin(costs)).sum()
    if n_missing_cost:
        print(f"WARNING: {n_missing_cost} annotated items lack per-example costs; using mean costs for them")

    print(f"{len(df)} annotations from {df.annotator.nunique()} annotators on {df.id.nunique()} items")
    print(df.groupby("annotator").agg(n=("id", "size"), acc=("final_correct", "mean"),
                                      reveals=("n_reveals", "mean"), minutes=("duration_s", lambda x: x.sum() / 60))
          .round(2).to_string())

    # 1. sufficiency prediction
    known = df.dropna(subset=["text_only_correct"])
    y_true = known.text_only_correct.astype(bool)
    y_pred = known.text_enough.astype(bool)
    suff = {
        "n_with_text_only_answer": int(len(known)),
        "accuracy": float((y_true == y_pred).mean()) if len(known) else None,
        "confusion[true][pred]": confusion_matrix(y_true, y_pred, labels=[False, True]).tolist() if len(known) else None,
        "text_only_acc_when_said_yes": float(known[known.text_enough].text_only_correct.mean()) if known.text_enough.any() else None,
        "text_only_acc_when_said_no": float(known[~known.text_enough].text_only_correct.mean()) if (~known.text_enough).any() else None,
        "said_text_enough_rate": float(df.text_enough.mean()),
    }
    print("\nSufficiency prediction (stage-0 judgment vs. own text-only correctness):")
    print(json.dumps(suff, indent=2))

    # 1b. per-annotator breakdown
    model_lab = {}
    model_path = repo_path("results/text_sufficient_labels.json")
    if model_path.exists():
        model_lab = {int(k): v for k, v in json.loads(model_path.read_text())["labels"].items()}
    per = []
    for a, g in df.groupby("annotator"):
        k = g.dropna(subset=["text_only_correct"])
        s0 = g.dropna(subset=["stage0_answer"])
        changed = s0[s0.stage0_answer != s0.final.map(lambda f: f["answer_idx"])]
        lo, hi = proportion_confint(int(g.final_correct.sum()), len(g), method="wilson")
        m = g[g.id.isin(model_lab)]
        per.append({
            "annotator": a, "n": len(g),
            "final_acc": g.final_correct.mean(), "final_ci_lo": lo, "final_ci_hi": hi,
            "acc_shared": g[g.shared].final_correct.mean(), "acc_unique": g[~g.shared].final_correct.mean(),
            **{f"acc_{t.split()[0]}": g[g.audio_type == t].final_correct.mean() for t in AUDIO_TYPES},
            "stage0_answered": len(s0) / len(g), "stage0_acc": (s0.stage0_answer == s0.gold).mean(),
            "said_text_enough": g.text_enough.mean(),
            "suff_pred_acc": (k.text_only_correct.astype(bool) == k.text_enough).mean() if len(k) else np.nan,
            "changed_after_reveal": len(changed) / max(len(s0), 1),
            "wrong_to_right": int(((changed.stage0_answer != changed.gold) & changed.final_correct).sum()),
            "right_to_wrong": int(((changed.stage0_answer == changed.gold) & ~changed.final_correct).sum()),
            "conf0_mean": g.conf0.mean(), "conf_final_mean": g.final.map(lambda f: f["conf"]).mean(),
            "acc_when_conf_ge4": g[g.final.map(lambda f: f["conf"]) >= 4].final_correct.mean(),
            "acc_when_conf_le3": g[g.final.map(lambda f: f["conf"]) <= 3].final_correct.mean(),
            "share_answer_D": (g.final.map(lambda f: f["answer_idx"]) == 3).mean(),
            "agree_model_text_suff": (m.text_enough.astype(int) == m.id.map(model_lab)).mean() if len(m) else np.nan,
            "median_s_per_item": g.duration_s.median(), "total_min": g.duration_s.sum() / 60,
        })
    per = pd.DataFrame(per).set_index("annotator")
    per.to_csv(out / "by_annotator.csv")
    print("\nPer annotator:\n" + per.T.round(2).to_string())

    # pairwise Cohen's kappa on the shared items
    sh = df[df.shared]
    pair_rows = []
    names = sorted(sh.annotator.unique())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A = sh[sh.annotator == a].set_index("id")
            B = sh[sh.annotator == b].set_index("id")
            common = A.index.intersection(B.index)
            fa = A.loc[common].final.map(lambda f: f["answer_idx"])
            fb = B.loc[common].final.map(lambda f: f["answer_idx"])
            pair_rows.append({
                "pair": f"{a}-{b}", "n": len(common),
                "final_agree": (fa == fb).mean(), "final_kappa": cohen_kappa_score(fa, fb),
                "text_enough_agree": (A.loc[common].text_enough == B.loc[common].text_enough).mean(),
                "text_enough_kappa": cohen_kappa_score(A.loc[common].text_enough, B.loc[common].text_enough),
            })
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "pairwise_agreement_shared.csv", index=False)
    print("\nPairwise agreement on shared items:\n" + pairs.round(2).to_string(index=False))

    # per shared item: who got it right, majority vote
    sh_items = sh.assign(fin=sh.final.map(lambda f: f["answer_idx"])).groupby("id").agg(
        audio_type=("audio_type", "first"), task_type=("task_type", "first"), gold=("gold", "first"),
        n_correct=("final_correct", "sum"), n_distinct_answers=("fin", "nunique"),
        majority=("fin", lambda x: x.mode().iloc[0] if x.value_counts().iloc[0] > 1 else -1),
        n_text_enough=("text_enough", "sum"))
    sh_items["majority_correct"] = sh_items.majority == sh_items.gold
    sh_items.to_csv(out / "shared_items.csv")
    print("\nShared items:\n" + sh_items.to_string())
    print(f"majority-vote accuracy on shared items: {sh_items.majority_correct.mean():.2f} "
          f"(ties/no majority counted wrong: {(sh_items.majority == -1).sum()})")

    # 2. steps and cost
    n_correct = int(df.final_correct.sum())
    cost_summary = {
        "final_acc": float(df.final_correct.mean()),
        "mean_reveals": float(df.n_reveals.mean()),
        "mean_tokens": float(df.tokens.mean()),
        "mean_cost_vs_full": float(df.cost.mean()),
        "tokens_per_correct": float(df.tokens.sum() / max(n_correct, 1)),
        "cost_per_correct": float(df.cost.sum() / max(n_correct, 1)),
        "reveal_patterns": df.pattern.value_counts().to_dict(),
    }
    print("\nSteps / cost:")
    print(json.dumps(cost_summary, indent=2))

    # 3. breakdowns
    def breakdown(key):
        g = df.groupby(key).agg(n=("id", "size"), final_acc=("final_correct", "mean"),
                                said_text_enough=("text_enough", "mean"), text_only_acc=("text_only_correct", "mean"),
                                reveal_image=("img", "mean"), reveal_audio=("aud", "mean"),
                                mean_reveals=("n_reveals", "mean"), mean_cost=("cost", "mean"))
        return g
    by_audio = breakdown("audio_type").reindex(AUDIO_TYPES)
    by_task = breakdown("task_type").sort_values("n", ascending=False)
    by_audio.to_csv(out / "by_audio_type.csv")
    by_task.to_csv(out / "by_task_type.csv")
    first = pd.crosstab(df.audio_type, df.first_reveal).reindex(AUDIO_TYPES)
    first.to_csv(out / "first_reveal_by_audio_type.csv")
    print("\nBy audio type:\n" + by_audio.round(2).to_string())
    print("\nBy task type:\n" + by_task.round(2).to_string())
    print("\nFirst reveal by audio type:\n" + first.to_string())

    # 4. Fleiss' kappa on shared items
    shared = df[df.shared]
    full_cov = shared.groupby("id").annotator.nunique()
    ids_all = full_cov[full_cov == len(cfg["analysis_b"]["annotators"])].index
    kappa = {"n_items": int(len(ids_all))}
    if len(ids_all) >= 2:
        s = shared[shared.id.isin(ids_all)]
        for name, col in [("text_enough", "text_enough"), ("final_answer", "final")]:
            vals = s.pivot(index="id", columns="annotator",
                           values=col if col != "final" else "final")
            if col == "final":
                vals = vals.map(lambda f: f["answer_idx"])
            table, _ = aggregate_raters(vals.astype(int).values)
            kappa[name] = float(fleiss_kappa(table, method="fleiss"))
        kappa["final_answer_all_agree_rate"] = float(
            s.assign(a=s.final.map(lambda f: f["answer_idx"])).groupby("id").a.nunique().eq(1).mean())
    print("\nFleiss' kappa (shared items):", json.dumps(kappa))

    # 5. human vs model text-sufficient labels; write human labels for Analysis C
    human = df.groupby("id").text_enough.mean().ge(0.5).astype(int)
    repo_path(out / "human_text_sufficient.json").write_text(json.dumps({
        "definition": "stage-0 'is text enough?' judgment; majority vote (ties -> yes) on shared items",
        "n": int(len(human)), "positive_rate": float(human.mean()),
        "labels": {str(i): int(v) for i, v in human.items()},
    }, indent=2))
    model_path = repo_path("results/text_sufficient_labels.json")
    vs_model = None
    if model_path.exists():
        model = {int(k): v for k, v in json.loads(model_path.read_text())["labels"].items()}
        common = [i for i in human.index if i in model]
        if common:
            h = np.array([human[i] for i in common])
            m = np.array([model[i] for i in common])
            vs_model = {"n": len(common), "agreement": float((h == m).mean()),
                        "cohen_kappa": float(cohen_kappa_score(h, m)) if len(set(h)) > 1 or len(set(m)) > 1 else None,
                        "confusion[human][model]": confusion_matrix(h, m, labels=[0, 1]).tolist(),
                        "human_yes_rate": float(h.mean()), "model_yes_rate": float(m.mean())}
            print("\nHuman 'text enough' vs model text-sufficient:", json.dumps(vs_model))

    # Figures
    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    pat = pd.crosstab(df.audio_type, df.pattern, normalize="index").reindex(index=AUDIO_TYPES, columns=REVEAL_PATTERNS, fill_value=0)
    left = np.zeros(len(AUDIO_TYPES))
    for k, p in enumerate(REVEAL_PATTERNS):
        ax.barh(range(len(AUDIO_TYPES)), pat[p] * 100, left=left, color=[INK_2, PALETTE[0], PALETTE[1], PALETTE[6]][k]
                if k else "#b9b8b2", height=0.6, label=p, edgecolor="white", linewidth=0.6)
        left += pat[p].values * 100
    ax.set_yticks(range(len(AUDIO_TYPES)), AUDIO_TYPES)
    ax.invert_yaxis()
    ax.set_xlabel("Items (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.3), handlelength=1)
    save_fig(fig, out, "reveal_patterns")

    summ_path = repo_path("results/analysis_a/tables/config_costs.csv")
    if summ_path.exists():
        summ = pd.read_csv(summ_path, index_col=0)
        fig, ax = plt.subplots(figsize=(COL_W, 2.0))
        for c in CONFIG_ORDER:
            ax.scatter(summ.loc[c, "mean_cost"], summ.loc[c, "acc"] * 100, s=22, color=CONFIG_COLORS[c],
                       edgecolors="white", linewidths=0.6, label=f"Model: {CONFIG_LABELS[c]}")
        ax.scatter(cost_summary["mean_cost_vs_full"], cost_summary["final_acc"] * 100, marker="D", s=30,
                   color=INK, label="Humans (staged reveal)")
        ax.set_xlabel("Mean cost (fraction of full-input tokens)")
        ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=5.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25), columnspacing=0.8)
        save_fig(fig, out, "human_vs_model_cost")

    (out / "summary.json").write_text(json.dumps({
        "n_annotations": int(len(df)), "sufficiency_prediction": suff, "cost": cost_summary,
        "fleiss_kappa": kappa, "human_vs_model": vs_model,
    }, indent=2, default=float))
    print(f"\nwrote tables, figures, summary.json and human_text_sufficient.json to {out}")


if __name__ == "__main__":
    main()
