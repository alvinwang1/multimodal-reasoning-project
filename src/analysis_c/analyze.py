"""Empath categories most associated with the text-sufficient label (Analysis C).

For each text view and Empath category: point-biserial correlation with the label and a
Mann-Whitney U test, Benjamini-Hochberg corrected across categories.

    python -m src.analysis_c.analyze
    python -m src.analysis_c.analyze --labels human
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pointbiserialr
from statsmodels.stats.multitest import multipletests

from src.analysis_c.features import VIEW_LABELS, VIEWS
from src.analysis_c.labels import FILE_SUFFIX, LABEL_TAGS, load_labels
from src.common import BINARY_COLORS, COL_W, load_config, repo_path, save_fig, set_plot_style


def associations(X, y, cats, min_docs=3):
    rows = []
    for j, c in enumerate(cats):
        x = X[:, j]
        if (x > 0).sum() < min_docs or x.std() == 0:
            continue
        r, _ = pointbiserialr(y, x)
        _, p = mannwhitneyu(x[y == 1], x[y == 0], alternative="two-sided")
        rows.append({"category": c, "r_pb": r, "p": p, "docs": int((x > 0).sum()),
                     "mean_pos": x[y == 1].mean(), "mean_neg": x[y == 0].mean()})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_bh"] = multipletests(df["p"], method="fdr_bh")[1]
        df = df.sort_values("r_pb", key=np.abs, ascending=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", choices=["model", "human"], default="model")
    ap.add_argument("--top", type=int, default=8, help="categories per direction in the plot")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg["analysis_c"]
    set_plot_style()
    tag, suffix = LABEL_TAGS[args.labels], FILE_SUFFIX[args.labels]
    out_dir = repo_path(ccfg["out_dir"])

    F = dict(np.load(repo_path(ccfg["features"])))
    labels = load_labels(args.labels)
    if labels is None:
        raise SystemExit(f"no {args.labels} labels found")
    keep = np.array([int(i) in labels for i in F["ids"]])
    y = np.array([labels[int(i)] for i in F["ids"][keep]])
    cats = list(F["empath_cats"])

    all_rows = []
    for v in VIEWS:
        df = associations(F[f"empath_{v}"][keep], y, cats)
        df.insert(0, "view", v)
        all_rows.append(df)
        print(f"\n{VIEW_LABELS[v]}{tag}: top categories by |r_pb| (n={len(y)}, pos rate {y.mean():.2f})")
        print(df.head(10).round(3).to_string(index=False))
    full = pd.concat(all_rows)
    full.to_csv(out_dir / f"empath_associations{suffix}.csv", index=False)

    # One bar chart per view: top +/- categories, * = BH-adjusted q < 0.05.
    for v in VIEWS:
        df = full[full.view == v].sort_values("r_pb")
        if df.empty:
            continue
        sel = pd.concat([df.head(args.top), df.tail(args.top)]).drop_duplicates("category")
        fig, ax = plt.subplots(figsize=(COL_W, 0.13 * len(sel) + 0.5))
        colors = [BINARY_COLORS[1] if r > 0 else BINARY_COLORS[0] for r in sel.r_pb]
        ax.barh(range(len(sel)), sel.r_pb, color=colors, height=0.7)
        ax.set_yticks(range(len(sel)))
        ax.set_yticklabels([f"{c}{'*' if q < 0.05 else ''}" for c, q in zip(sel.category, sel.q_bh)])
        ax.axvline(0, color="#52514e", lw=0.6)
        ax.set_xlabel("Point-biserial r with text-sufficient label")
        ax.set_title(f"Empath categories ({VIEW_LABELS[v]}){tag}")
        ax.grid(axis="y", visible=False)
        save_fig(fig, out_dir, f"empath_top_{v}{suffix}")
    print(f"\n* = BH-adjusted q < 0.05. Wrote {out_dir / f'empath_associations{suffix}.csv'}")


if __name__ == "__main__":
    main()
