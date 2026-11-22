"""Idea 3 baseline: marginal utility of image vs. audio tokens under budgets (test split, no training).

Grid: image keep-ratio r_I in {0, .1, .25, .5, 1} x audio keep-ratio r_A in {0, .25, .5, 1}. The image is
downscaled (~r_I of the visual tokens); the audio is time-stretched (~r_A of the audio tokens, same content).
Corners come from the Idea-1 run: (1,1)=full, (1,0)=image only, (0,1)=audio only, (0,0)=question only.
Accuracy is averaged over three option orders.

    python -m src.midterm.idea3_budget
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import (AUDIO_TYPE_COLORS, COL_W, INK, INK_2, PALETTE, load_config, read_jsonl, repo_path,
                        save_fig, set_plot_style)
from src.data import AUDIO_TYPES, load_omnibench, text_records

R_IMG = [0.0, 0.1, 0.25, 0.5, 1.0]
R_AUD = [0.0, 0.25, 0.5, 1.0]
CORNERS = {"full": (1.0, 1.0), "img_only": (1.0, 0.0), "aud_only": (0.0, 1.0), "q_only": (0.0, 0.0)}


def load(cfg, test_ids):
    bud = pd.DataFrame(read_jsonl(repo_path(cfg["midterm"]["budget_out"])))
    base = pd.DataFrame(read_jsonl(repo_path(cfg["analysis_a"]["out"])))
    base = base[base.config.isin(CORNERS) & base.id.isin(test_ids)].copy()
    base["img_ratio"] = base.config.map(lambda c: CORNERS[c][0])
    base["aud_ratio"] = base.config.map(lambda c: CORNERS[c][1])
    df = pd.concat([bud, base], ignore_index=True).drop_duplicates(["id", "config", "perm"], keep="last")
    df = df[df.id.isin(test_ids)]
    df["mm_tok"] = df.n_image_tok + df.n_audio_tok
    df["act_mem_mb"] = df.peak_mem_mb - df.baseline_mem_mb
    n_cells = len(R_IMG) * len(R_AUD) * 3
    counts = df.groupby("id").size()
    complete = counts[counts == n_cells].index
    return df[df.id.isin(complete)].copy(), len(complete)


def grid(df, value, agg="mean"):
    return df.groupby(["img_ratio", "aud_ratio"])[value].agg(agg).unstack().reindex(index=R_IMG, columns=R_AUD)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_plot_style()
    out = repo_path(cfg["midterm"]["out_dir"]) / "idea3"
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(repo_path(cfg["midterm"]["split"]).read_text())
    df, n = load(cfg, split["test"])
    print(f"complete test items: {n}/{len(split['test'])}  "
          f"({df.groupby('id').audio_type.first().value_counts().to_dict()})")
    if n == 0:
        return

    # per-item, per-cell accuracy averaged over orders
    cell = df.groupby(["id", "img_ratio", "aud_ratio"]).agg(
        acc=("correct", "mean"), img_tok=("n_image_tok", "mean"), aud_tok=("n_audio_tok", "mean"),
        mm_tok=("mm_tok", "mean"), tok=("n_total_tok", "mean"), lat=("latency_ms", "mean"),
        mem=("act_mem_mb", "mean"), audio_type=("audio_type", "first")).reset_index()

    summ = cell.groupby(["img_ratio", "aud_ratio"]).agg(
        acc=("acc", "mean"), acc_sd=("acc", "std"), img_tok=("img_tok", "mean"), aud_tok=("aud_tok", "mean"),
        mm_tok=("mm_tok", "mean"), tok=("tok", "mean"), latency_ms=("lat", "mean"), act_mem_mb=("mem", "mean"))
    summ["acc_ci"] = 1.96 * summ.acc_sd / np.sqrt(n)
    for at in AUDIO_TYPES:
        summ[f"acc_{at.split()[0]}"] = cell[cell.audio_type == at].groupby(["img_ratio", "aud_ratio"]).acc.mean()
    summ = summ.reset_index()
    summ["img_share"] = summ.img_tok / summ.mm_tok.replace(0, np.nan)
    summ["policy"] = np.select(
        [(summ.img_ratio == summ.aud_ratio), (summ.img_ratio > summ.aud_ratio), (summ.img_ratio < summ.aud_ratio)],
        ["uniform", "image-heavy", "audio-heavy"], default="")
    summ.to_csv(out / "grid_summary.csv", index=False)
    print("\nAccuracy grid (rows: image keep-ratio, cols: audio keep-ratio):")
    acc_grid = summ.pivot(index="img_ratio", columns="aud_ratio", values="acc").reindex(index=R_IMG, columns=R_AUD)
    print((acc_grid * 100).round(1).to_string())
    print("\nMean multimodal tokens (image + audio):")
    print(summ.pivot(index="img_ratio", columns="aud_ratio", values="mm_tok").reindex(index=R_IMG, columns=R_AUD)
          .round(0).to_string())

    # marginal utility: accuracy change per 100 extra tokens of one modality, holding the other fixed
    mu = []
    for at in ["all"] + AUDIO_TYPES:
        c = cell if at == "all" else cell[cell.audio_type == at]
        if c.empty:
            continue
        g = c.groupby(["img_ratio", "aud_ratio"]).agg(acc=("acc", "mean"), img=("img_tok", "mean"),
                                                      aud=("aud_tok", "mean"))
        di, da = [], []
        for ra in R_AUD:
            for a, b in zip(R_IMG[:-1], R_IMG[1:]):
                d_tok = g.loc[(b, ra), "img"] - g.loc[(a, ra), "img"]
                if d_tok > 0:
                    di.append((g.loc[(b, ra), "acc"] - g.loc[(a, ra), "acc"]) / d_tok * 100)
        for ri in R_IMG:
            for a, b in zip(R_AUD[:-1], R_AUD[1:]):
                d_tok = g.loc[(ri, b), "aud"] - g.loc[(ri, a), "aud"]
                if d_tok > 0:
                    da.append((g.loc[(ri, b), "acc"] - g.loc[(ri, a), "acc"]) / d_tok * 100)
        # first tokens vs. last tokens: utility of going 0 -> smallest vs. half -> full
        first_img = (g.loc[(0.1, 0.0), "acc"] - g.loc[(0.0, 0.0), "acc"]) / max(g.loc[(0.1, 0.0), "img"], 1) * 100
        last_img = (g.loc[(1.0, 0.0), "acc"] - g.loc[(0.5, 0.0), "acc"]) / max(
            g.loc[(1.0, 0.0), "img"] - g.loc[(0.5, 0.0), "img"], 1) * 100
        first_aud = (g.loc[(0.0, 0.25), "acc"] - g.loc[(0.0, 0.0), "acc"]) / max(g.loc[(0.0, 0.25), "aud"], 1) * 100
        last_aud = (g.loc[(0.0, 1.0), "acc"] - g.loc[(0.0, 0.5), "acc"]) / max(
            g.loc[(0.0, 1.0), "aud"] - g.loc[(0.0, 0.5), "aud"], 1) * 100
        mu.append({"audio_type": at, "img_pts_per_100tok_mean": np.mean(di), "aud_pts_per_100tok_mean": np.mean(da),
                   "img_first_tokens": first_img * 100, "img_last_tokens": last_img * 100,
                   "aud_first_tokens": first_aud * 100, "aud_last_tokens": last_aud * 100})
    mu = pd.DataFrame(mu)
    for col in ["img_pts_per_100tok_mean", "aud_pts_per_100tok_mean"]:
        mu[col] = mu[col] * 100  # accuracy points (%) per 100 tokens
    mu.to_csv(out / "marginal_utility.csv", index=False)
    print("\nMarginal utility (accuracy points per 100 tokens):")
    print(mu.round(2).to_string(index=False))

    # matched budgets: best allocation within a token budget vs. the uniform policy within that budget
    budgets = [100, 200, 350, 500, 750, 1000]
    mb = []
    for B in budgets:
        ok = summ[summ.mm_tok <= B]
        best = ok.loc[ok.acc.idxmax()]
        uni = ok[ok.policy == "uniform"]
        u = uni.loc[uni.acc.idxmax()] if len(uni) else None
        mb.append({"budget": B, "best_alloc": f"({best.img_ratio:g},{best.aud_ratio:g})", "best_acc": best.acc,
                   "best_policy": best.policy, "best_mm_tok": best.mm_tok,
                   "uniform_alloc": f"({u.img_ratio:g},{u.aud_ratio:g})" if u is not None else None,
                   "uniform_acc": u.acc if u is not None else np.nan})
    mb = pd.DataFrame(mb)
    mb.to_csv(out / "matched_budgets.csv", index=False)
    print("\nBest allocation within each multimodal-token budget (descriptive, on test):")
    print(mb.round(3).to_string(index=False))

    # failure analysis: items solved at full budget but lost under compression
    full = cell[(cell.img_ratio == 1) & (cell.aud_ratio == 1)].set_index("id")
    records, _ = text_records(load_omnibench(cfg))
    by_id = {r["index"]: r for r in records}
    flips = []
    for (ri, ra) in [(0.25, 0.25), (0.1, 1.0), (1.0, 0.25), (0.5, 0.5)]:
        comp = cell[(cell.img_ratio == ri) & (cell.aud_ratio == ra)].set_index("id")
        lost = (full.acc >= 2 / 3) & (comp.acc <= 1 / 3)
        gained = (full.acc <= 1 / 3) & (comp.acc >= 2 / 3)
        for i in full.index[lost]:
            flips.append({"id": i, "alloc": f"({ri:g},{ra:g})", "kind": "lost", "audio_type": full.at[i, "audio_type"],
                          "acc_full": full.at[i, "acc"], "acc_comp": comp.at[i, "acc"],
                          "question": by_id[i]["question"]})
        print(f"({ri:g},{ra:g}): lost {int(lost.sum())} / gained {int(gained.sum())} items vs. full "
              f"(lost by type: {full[lost].audio_type.value_counts().to_dict()})")
    pd.DataFrame(flips).to_csv(out / "failure_cases_compression.csv", index=False)

    # audio speed-up on speech: does time-stretching destroy intelligibility?
    sp = cell[cell.img_ratio == 0].groupby(["audio_type", "aud_ratio"]).acc.mean().unstack()
    sp.to_csv(out / "audio_speedup_by_type.csv")
    print("\nAudio only, by speed-up (keep-ratio 1 = normal, .5 = 2x, .25 = 4x):\n" + (sp * 100).round(1).to_string())

    # ---------------- figures
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 2.9), sharex=True, sharey=True)
    vmax = float(np.nanmax(summ[[f"acc_{a.split()[0]}" for a in AUDIO_TYPES] + ["acc"]].values))
    for ax, (title, col) in zip(axes.flat, [("All", "acc")] + [(a, f"acc_{a.split()[0]}") for a in AUDIO_TYPES]):
        G = summ.pivot(index="img_ratio", columns="aud_ratio", values=col).reindex(index=R_IMG, columns=R_AUD)
        im = ax.imshow(G.values * 100, cmap="Blues", vmin=0, vmax=vmax * 100, origin="lower", aspect="auto")
        for (r_, c_), v in np.ndenumerate(G.values):
            ax.text(c_, r_, f"{v * 100:.0f}", ha="center", va="center", fontsize=5,
                    color="white" if v > vmax * 0.6 else INK)
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_xticks(range(len(R_AUD)), [f"{r:g}" for r in R_AUD])
        ax.set_yticks(range(len(R_IMG)), [f"{r:g}" for r in R_IMG])
        ax.grid(False)
    for ax in axes[1]:
        ax.set_xlabel("Audio keep-ratio", fontsize=6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Image keep-ratio", fontsize=6)
    fig.tight_layout(pad=0.3)
    save_fig(fig, out, "budget_heatmaps")

    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    cmap = plt.get_cmap("coolwarm")
    for _, r in summ.iterrows():
        share = 0.5 if np.isnan(r.img_share) else r.img_share
        ax.scatter(r.mm_tok, r.acc * 100, s=18, color=cmap(share), edgecolors="white", linewidths=0.5, zorder=3)
    for name, sel, col, ls in [("Uniform", summ.img_ratio == summ.aud_ratio, INK, "-"),
                               ("Image only", summ.aud_ratio == 0, PALETTE[7], "--"),
                               ("Audio only", summ.img_ratio == 0, PALETTE[0], "--")]:
        s = summ[sel].sort_values("mm_tok")
        ax.plot(s.mm_tok, s.acc * 100, color=col, ls=ls, lw=1.0, label=name, zorder=2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("Image share of tokens", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_xlabel("Multimodal tokens (image + audio)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(loc="lower right", fontsize=5.5)
    save_fig(fig, out, "budget_frontier")

    (out / "summary.json").write_text(json.dumps({"n_test_items": int(n), "grid": summ.round(4).to_dict(orient="records"),
                                                  "marginal_utility": mu.round(4).to_dict(orient="records"),
                                                  "matched_budgets": mb.round(4).to_dict(orient="records")},
                                                 indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
