"""Idea 3 analysis: marginal utility of image and audio tokens, by audio type.

    python -m src.idea3.analyze

Reads results/analysis_a/raw.jsonl (dev corners + surrogate configs) and, when present,
results/idea3/raw_{dev,test}.jsonl. Scores are averaged over option permutations; s = log p(gold).

Outputs (results/idea3/):
  tables/cells.csv          acc / log p / tokens per split x audio type x cell
  tables/shapley.csv        per-item 2x2 Shapley values and interaction (image, audio), by audio type
  tables/shapley_task.csv   same, by task type
  tables/policies.csv       budget policies tuned on dev, scored on test         (needs both grids)
  tables/truncation.csv     leading vs centered audio window, audio only          (needs dev grid)
  failures_*.csv            items for qualitative analysis
  shapley.pdf, grid_heatmap.pdf, marginal_curves.pdf, summary.json
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import (AUDIO_TYPE_COLORS, COL_W, INK_2, PAGE_W, PALETTE, load_config, read_jsonl, repo_path,
                        save_fig, set_plot_style)
from src.data import AUDIO_TYPES, load_omnibench, text_records
from src.idea3.cells import A_CONFIG_CELLS, CORNERS, SURROGATE_CELLS, grid_cells, parse_cell

EPS = 1e-6
TEXT_COLS = ["index", "question", "options", "answer", "audio content", "image content"]


def load(cfg):
    """All records; Idea 3 runs take precedence over Analysis A records for the same dev cell.

    Also returns the pred agreement between the two runs on shared (id, cell, perm) keys."""
    a = pd.DataFrame(read_jsonl(repo_path(cfg["analysis_a"]["out"])))  # gitignored; may be absent
    if len(a):
        a = a.assign(cell=a.config.map(A_CONFIG_CELLS), split="dev")
    frames = [a]
    for split in ("dev", "test"):
        f = repo_path(f"{cfg['idea3']['out_dir']}/raw_{split}.jsonl")
        if f.exists():
            frames.append(pd.DataFrame(read_jsonl(f)))
    if not any(len(f) for f in frames):
        raise SystemExit("no records: run src.idea3.run first (or add results/analysis_a/raw.jsonl)")
    agreement = None
    if len(a) and len(frames) > 1:
        keys = ["split", "id", "cell", "perm"]
        m = a.merge(pd.concat(frames[1:]), on=keys, suffixes=("_a", "_i3"))
        if len(m):
            agreement = {"n": len(m), "pred_agree": float((m.pred_idx_a == m.pred_idx_i3).mean()),
                         "acc_a": float(m.correct_a.mean()), "acc_idea3": float(m.correct_i3.mean()),
                         "devices": sorted(set(m.device_a) | set(m.device_i3)),
                         "dtypes": sorted(set(m.dtype_a) | set(m.dtype_i3))}
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["split", "id", "cell", "perm"], keep="last")
    df["logp"] = np.log(np.clip([p[k] for p, k in zip(df.probs, df.answer_idx)], EPS, 1))
    # always-"D" under the displayed order of this permutation
    df["always_d"] = [order[3] == k for order, k in zip(df.perm_order, df.answer_idx)]
    return df, agreement


def per_item(df):
    """(split, id, cell) -> permutation-averaged scores; cost relative to the item's full input."""
    g = df.groupby(["split", "id", "cell"])
    it = g.agg(acc=("correct", "mean"), robust=("correct", "all"), logp=("logp", "mean"),
               tokens=("n_total_tok", "mean"), latency_ms=("latency_ms", "mean"), n_perms=("perm", "nunique"),
               audio_type=("audio_type", "first"), task_type=("task_type", "first"),
               always_d=("always_d", "mean")).reset_index()
    full = it[it.cell == "v768_a1"].set_index(["split", "id"]).tokens
    it["cost"] = it.tokens / pd.MultiIndex.from_frame(it[["split", "id"]]).map(full)
    return it


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell_table(it):
    rows = []
    for (split, at, cell), g in pd.concat([it, it.assign(audio_type="all")]).groupby(
            ["split", "audio_type", "cell"]):
        rows.append({"split": split, "audio_type": at, "cell": cell, "n": len(g), "acc": g.acc.mean(),
                     "logp": g.logp.mean(), "tokens": g.tokens.mean(), "cost": g.cost.mean(),
                     "latency_ms": g.latency_ms.mean(), "always_d": g.always_d.mean()})
    return pd.DataFrame(rows)


def shapley(it, metric):
    """Per item: phi_img, phi_aud (2x2 Shapley over raw image / raw audio) and the interaction term."""
    w = it[it.cell.isin(CORNERS)].pivot_table(index=["split", "id"], columns="cell", values=metric)
    w = w.dropna()
    va, v, a, q = w["v768_a1"], w["v768_a0"], w["v0_a1"], w["v0_a0"]
    out = pd.DataFrame({"phi_img": 0.5 * ((va - a) + (v - q)), "phi_aud": 0.5 * ((va - v) + (a - q)),
                        "interaction": va - v - a + q, "img_given_aud": va - a, "aud_given_img": va - v,
                        "total": va - q})
    meta = it.drop_duplicates(["split", "id"]).set_index(["split", "id"])[["audio_type", "task_type"]]
    return out.join(meta).reset_index().assign(metric=metric)


def summarize_shapley(sh, by):
    rows = []
    for metric, gm in sh.groupby("metric"):
        for key, g in pd.concat([gm, gm.assign(**{by: "all"})]).groupby(by):
            for term in ["phi_img", "phi_aud", "interaction", "img_given_aud", "aud_given_img"]:
                lo, hi = boot_ci(g[term])
                rows.append({"metric": metric, by: key, "term": term, "n": len(g), "mean": g[term].mean(),
                             "ci_lo": lo, "ci_hi": hi, "frac_negative": (g[term] < 0).mean()})
    return pd.DataFrame(rows)


def plot_shapley(summ, out_dir):
    s = summ[(summ.metric == "logp") & summ.term.isin(["phi_img", "phi_aud", "interaction"])]
    terms = {"phi_img": "Image φ", "phi_aud": "Audio φ", "interaction": "Interaction"}
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    groups = AUDIO_TYPES + ["all"]
    w = 0.26
    for k, (t, lab) in enumerate(terms.items()):
        d = s[s.term == t].set_index("audio_type").reindex(groups)
        x = np.arange(len(groups)) + (k - 1) * w
        ax.bar(x, d["mean"], w * 0.92, color=PALETTE[k], label=lab)
        ax.errorbar(x, d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], fmt="none", ecolor=INK_2,
                    elinewidth=0.6, capsize=1.5)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.set_xticks(range(len(groups)), [g.capitalize() for g in groups])
    ax.set_ylabel("Δ log p(gold)")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.18))
    return save_fig(fig, out_dir, "shapley")


def plot_grid(cells, split, out_dir, icfg):
    bud, fr = icfg["image_budgets"], icfg["audio_fracs"]
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.9), sharey=True)
    for ax, at in zip(axes, AUDIO_TYPES):
        c = cells[(cells.split == split) & (cells.audio_type == at)].set_index("cell")
        m = np.array([[c.acc.get(f"v{b}_a{f:g}", np.nan) * 100 for f in fr] for b in bud])
        im = ax.imshow(m, origin="lower", cmap="Blues", vmin=15, vmax=75, aspect="auto")
        for i in range(len(bud)):
            for j in range(len(fr)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if m[i, j] > 55 else "black")
        ax.set_xticks(range(len(fr)), [f"{f:g}" for f in fr])
        ax.set_yticks(range(len(bud)), bud)
        ax.set_title(at.capitalize())
        ax.set_xlabel("Audio fraction kept")
        ax.grid(False)
    axes[0].set_ylabel("Image tokens (cap)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy (%)")
    return save_fig(fig, out_dir, f"grid_heatmap_{split}")


def plot_marginal(cells, split, out_dir, icfg):
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 1.9), sharey=True)
    sweeps = {"Image sweep, full audio": [f"v{b}_a1" for b in icfg["image_budgets"]],
              "Image sweep, no audio": [f"v{b}_a0" for b in icfg["image_budgets"]],
              "Audio sweep, full image": [f"v768_a{f:g}" for f in icfg["audio_fracs"]],
              "Audio sweep, no image": [f"v0_a{f:g}" for f in icfg["audio_fracs"]]}
    styles = [(PALETTE[0], "-"), (PALETTE[0], "--"), (PALETTE[1], "-"), (PALETTE[1], "--")]
    for ax, at in zip(axes, AUDIO_TYPES):
        c = cells[(cells.split == split) & (cells.audio_type == at)].set_index("cell")
        for (lab, cl), (col, ls) in zip(sweeps.items(), styles):
            cl = [x for x in cl if x in c.index]
            ax.plot(c.tokens[cl], c.acc[cl] * 100, ls, color=col, marker="o", ms=2.5, label=lab)
        for s in SURROGATE_CELLS:
            if s in c.index:
                ax.plot(c.tokens[s], c.acc[s] * 100, "D", color=PALETTE[2], ms=3)
        ax.set_title(at.capitalize())
        ax.set_xlabel("Mean input tokens")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="upper left", fontsize=5.5)
    return save_fig(fig, out_dir, f"marginal_curves_{split}")


def policies(it, betas):
    """Pick the most accurate cell with mean dev cost <= beta (globally or per audio type); score on test."""
    rows = []
    dev, test = it[it.split == "dev"], it[it.split == "test"]
    shared = sorted(set(dev.cell) & set(test.cell))
    raw_grid = [c for c in shared if c not in SURROGATE_CELLS]
    for cand_name, cand in [("raw grid", raw_grid), ("grid + surrogates", shared)]:
        for beta in betas:
            for pol in ["global", "per audio type"]:
                groups = [("all", AUDIO_TYPES)] if pol == "global" else [(t, [t]) for t in AUDIO_TYPES]
                picks, parts = {}, []
                for name, types in groups:
                    d = dev[dev.audio_type.isin(types) & dev.cell.isin(cand)].groupby("cell")[["acc", "cost"]].mean()
                    ok = d[d.cost <= beta + 1e-9]
                    if ok.empty:
                        ok = d.nsmallest(1, "cost")
                    best = ok.sort_values(["acc", "cost"], ascending=[False, True]).index[0]
                    picks[name] = best
                    parts.append(test[test.audio_type.isin(types) & (test.cell == best)])
                t = pd.concat(parts)
                rows.append({"candidates": cand_name, "beta": beta, "policy": pol, "test_acc": t.acc.mean(),
                             "test_cost": t.cost.mean(), "test_tokens": t.tokens.mean(),
                             "test_logp": t.logp.mean(), "picks": json.dumps(picks)})
    full = test[test.cell == "v768_a1"]
    orc = test[test.cell.isin(shared)].sort_values("cost").groupby("id")
    solved = orc.apply(lambda g: g[g.robust].head(1), include_groups=False)
    rows.append({"candidates": "-", "beta": 1.0, "policy": "full input", "test_acc": full.acc.mean(),
                 "test_cost": 1.0, "test_tokens": full.tokens.mean(), "test_logp": full.logp.mean()})
    rows.append({"candidates": "-", "beta": np.nan, "policy": "always D", "test_acc": full.always_d.mean()})
    rows.append({"candidates": "grid + surrogates", "beta": np.nan, "policy": "oracle (robust)",
                 "test_acc": len(solved) / test.id.nunique(), "test_cost": solved.cost.mean()})
    return pd.DataFrame(rows)


def truncation(it):
    d = it[(it.split == "dev") & it.cell.isin(["v0_a0.25", "v0_a0.25c", "v0_a0.5", "v0_a0.5c", "v0_a1"])]
    t = d.groupby(["audio_type", "cell"])[["acc", "logp"]].mean().reset_index()
    return pd.concat([t, d.groupby("cell")[["acc", "logp"]].mean().reset_index().assign(audio_type="all")])


def failures(it, sh, text, out_dir):
    """Items where adding a modality lowers log p(gold) most (negative marginal utility)."""
    s = sh[sh.metric == "logp"].set_index(["split", "id"])
    probs = it[it.cell.isin(CORNERS + SURROGATE_CELLS)].pivot_table(index=["split", "id"], columns="cell",
                                                                      values="acc")
    probs.columns = [f"acc_{c}" for c in probs.columns]
    base = s.join(probs).reset_index().merge(text, left_on="id", right_on="index", how="left")
    cols = ["split", "id", "audio_type", "task_type", "img_given_aud", "aud_given_img", "interaction",
            *probs.columns, "question", "options", "answer", "image content", "audio content"]
    for term in ["img_given_aud", "aud_given_img"]:
        base.sort_values(term)[cols].head(25).to_csv(out_dir / f"failures_{term}.csv", index=False)
    # strong negative interaction: each modality helps alone, together they do worse
    base.sort_values("interaction")[cols].head(25).to_csv(out_dir / "failures_interaction.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    icfg = cfg["idea3"]
    out_dir = repo_path(icfg["out_dir"])
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    set_plot_style()

    df, agreement = load(cfg)
    it = per_item(df)
    cells = cell_table(it)
    cells.to_csv(out_dir / "tables" / "cells.csv", index=False)

    sh = pd.concat([shapley(it, "logp"), shapley(it, "acc")])
    sh.to_csv(out_dir / "tables" / "shapley_items.csv", index=False)
    summ = summarize_shapley(sh, "audio_type")
    summ.to_csv(out_dir / "tables" / "shapley.csv", index=False)
    summarize_shapley(sh, "task_type").to_csv(out_dir / "tables" / "shapley_task.csv", index=False)
    plot_shapley(summ[summ.n > 0], out_dir)

    records, _ = text_records(load_omnibench(cfg))
    failures(it, sh, pd.DataFrame(records)[TEXT_COLS], out_dir)

    grid = set(grid_cells(icfg["image_budgets"], icfg["audio_fracs"]))
    summary = {"n_items": it.groupby("split").id.nunique().to_dict(), "agreement_with_analysis_a": agreement,
               "shapley_logp": summ[summ.metric == "logp"].set_index(["audio_type", "term"])["mean"]
               .unstack().round(3).to_dict(orient="index"),
               "shapley_acc": summ[summ.metric == "acc"].set_index(["audio_type", "term"])["mean"]
               .unstack().round(3).to_dict(orient="index")}
    for split in ("dev", "test"):
        have = set(it[it.split == split].cell)
        if grid <= have:
            plot_grid(cells, split, out_dir, icfg)
            plot_marginal(cells, split, out_dir, icfg)
    if {"v0_a0.25c", "v0_a0.5c"} <= set(it[it.split == "dev"].cell):
        truncation(it).to_csv(out_dir / "tables" / "truncation.csv", index=False)
    if grid <= set(it[it.split == "dev"].cell) and grid <= set(it[it.split == "test"].cell):
        pol = policies(it, [0.25, 0.4, 0.6, 0.8, 1.0])
        pol.to_csv(out_dir / "tables" / "policies.csv", index=False)
        summary["policies"] = pol.round(4).to_dict(orient="records")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
