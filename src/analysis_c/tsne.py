"""t-SNE of the SBERT embeddings, colored by task type, audio type and text-sufficiency.

    python -m src.analysis_c.tsne                   # model labels from Analysis A (if available)
    python -m src.analysis_c.tsne --labels human    # human labels from Analysis B

Writes one full-width grid (views x colorings) plus a single-column panel per view,
and a CSV of cosine silhouette scores (how well each grouping separates in embedding space).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from src.analysis_c.features import VIEW_LABELS, VIEWS
from src.analysis_c.labels import FILE_SUFFIX, LABEL_TAGS, load_labels
from src.common import (AUDIO_TYPE_COLORS, BINARY_COLORS, COL_W, PAGE_W, PALETTE, load_config,
                        repo_path, save_fig, set_plot_style)

TASK_SHORT = {
    "Action and Activity": "Action", "Plot Inference": "Plot", "Story Description": "Story",
    "Object Identification and Description": "Object ID",
    "Contextual and Environmental Questions": "Context/Env", "Identity and Relationship": "Identity",
    "Text and Symbols": "Text/Symbol", "Count and Quantity": "Count",
}


def colorings(F, labels, tag):
    task = np.array([TASK_SHORT.get(t, t) for t in F["task_type"]])
    task_order = [t for t in TASK_SHORT.values() if t in set(task)]
    out = [
        ("task_type", "Task type", task, {t: PALETTE[k] for k, t in enumerate(task_order)}),
        ("audio_type", "Audio type", F["audio_type"], AUDIO_TYPE_COLORS),
    ]
    if labels is not None:
        y = np.array([labels.get(int(i), -1) for i in F["ids"]])
        names = np.where(y == 1, "text-sufficient", np.where(y == 0, "needs raw input", "n/a"))
        out.append(("text_sufficient", "Text-sufficient" + tag, names,
                    {"needs raw input": BINARY_COLORS[0], "text-sufficient": BINARY_COLORS[1]}))
    return out


def scatter(ax, xy, groups, cmap, title, legend=True):
    for g, c in cmap.items():
        m = groups == g
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=14, c=c, edgecolors="white", linewidths=0.4,
                       label=f"{g} ({m.sum()})", zorder=3 if c != BINARY_COLORS[0] else 2)
    ax.set_title(title)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2, borderaxespad=0.2,
                  markerscale=1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels", choices=["model", "human"], default="model")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg["analysis_c"]
    set_plot_style()
    F = dict(np.load(repo_path(ccfg["features"])))
    labels = load_labels(args.labels)
    tag, suffix = LABEL_TAGS[args.labels], FILE_SUFFIX[args.labels]
    if labels is None:
        print(f"no {args.labels} labels yet; plotting task/audio type only")
    out_dir = ccfg["out_dir"]

    cols = colorings(F, labels, tag)
    proj = {}
    for v in VIEWS:
        tsne = TSNE(n_components=2, perplexity=ccfg["tsne_perplexity"], metric="cosine", init="pca",
                    random_state=cfg["seed"])
        proj[v] = tsne.fit_transform(F[f"emb_{v}"])

    # Full-width grid: rows = views, cols = colorings.
    fig, axes = plt.subplots(len(VIEWS), len(cols), figsize=(PAGE_W, 1.75 * len(VIEWS)), squeeze=False)
    for r, v in enumerate(VIEWS):
        for c, (_, title, groups, cmap) in enumerate(cols):
            scatter(axes[r, c], proj[v], groups, cmap, f"{VIEW_LABELS[v]}\nby {title.lower()}",
                    legend=(r == 0))
    fig.tight_layout(w_pad=6)
    save_fig(fig, out_dir, f"tsne_grid{suffix}")

    # Single-column panels: one per view, colored by the most informative grouping available.
    key, title, groups, cmap = cols[-1]
    for v in VIEWS:
        fig, ax = plt.subplots(figsize=(COL_W * 0.62, COL_W * 0.55))
        scatter(ax, proj[v], groups, cmap, f"{VIEW_LABELS[v]}: {title}")
        save_fig(fig, out_dir, f"tsne_{v}_{key}{suffix}")

    # Two-panel single-column figure: transcript view by audio type and by the last grouping.
    if len(cols) >= 3:
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.55))
        for ax, (_, title, groups, cmap) in zip(axes, [cols[1], cols[-1]]):
            scatter(ax, proj["transcript"], groups, cmap, f"Transcript by {title.lower()}", legend=False)
            handles = [plt.Line2D([], [], marker="o", ls="", color=c, markersize=3.5, label=g)
                       for g, c in cmap.items() if (groups == g).any()]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(handles),
                      fontsize=5, handletextpad=0.1, columnspacing=0.6)
        fig.tight_layout(w_pad=0.5)
        save_fig(fig, out_dir, f"tsne_transcript_pair{suffix}")

    rows = []
    for v in VIEWS:
        for key, title, groups, _ in cols:
            m = groups != "n/a"
            if len(set(groups[m])) > 1:
                rows.append({"view": v, "grouping": key + suffix,
                             "silhouette_cosine": silhouette_score(F[f"emb_{v}"][m], groups[m], metric="cosine")})
    sil = pd.DataFrame(rows)
    sil.to_csv(repo_path(out_dir) / f"tsne_silhouette{suffix}.csv", index=False)
    print(sil.pivot(index="view", columns="grouping", values="silhouette_cosine").round(3).to_string())
    print(f"wrote t-SNE figures to {repo_path(out_dir)}")


if __name__ == "__main__":
    main()
