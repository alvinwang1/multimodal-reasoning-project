"""Overview figure for the midterm report: input item -> representation choices -> frozen model.

    python -m src.midterm.overview_figure      # -> report_midterm/figures/overview.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.common import INK, INK_2, PALETTE, PAGE_W, save_fig, set_plot_style


def box(ax, x, y, w, h, text, fc, ec=None, fs=6.2, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=fc, ec=ec or fc, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color=INK_2, lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7, lw=lw, color=color))


def main():
    set_plot_style()
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    light = {"q": "#ecebe7", "v": "#dbe8f8", "a": "#fbe3d8", "t": "#d9f1e7", "m": "#f3f2ee"}
    H = 0.12

    # ---- input item
    ax.text(0.085, 0.97, "OmniBench item $x$", ha="center", va="top", fontsize=6.8, fontweight="bold")
    box(ax, 0.01, 0.74, 0.15, H, "question $q$ + 4 options $O$\n(~130 tokens)", light["q"], fs=5.8)
    box(ax, 0.01, 0.53, 0.15, H, "image $v$\n(~700 tokens)", light["v"], fs=5.8)
    box(ax, 0.01, 0.32, 0.15, H, "audio $a$: speech / sound /\nmusic (~250 tokens)", light["a"], fs=5.8)

    # ---- representation menu
    ax.text(0.445, 0.97, "Representation $r=(r_v, r_a)$ per modality", ha="center", va="top",
            fontsize=6.8, fontweight="bold")
    W, xs = 0.10, [0.22, 0.335, 0.45, 0.565]
    heads = ["raw", "compressed", "text surrogate", "omitted"]
    cells = {"v": ["raw image", "64-token image", "caption", "\u2014"],
             "a": ["raw audio", "first half of clip", "transcript /\ndescription", "\u2014"]}
    for x, h in zip(xs, heads):
        ax.text(x + W / 2, 0.82, h, ha="center", va="center", fontsize=5.8, color=INK_2)
    for key, y in (("v", 0.53), ("a", 0.32)):
        for x, txt in zip(xs, cells[key]):
            fc = "#ffffff" if txt == "\u2014" else light[key]
            box(ax, x, y, W, H, txt, fc, ec="#c9c7c0", fs=5.5)
        arrow(ax, 0.163, y + H / 2, 0.215, y + H / 2)
    # idea tags under their columns
    ax.text(xs[1] + W / 2, 0.27, "Idea 3:\nbudgets", ha="center", va="top", fontsize=5.6,
            color=PALETTE[1], fontweight="bold")
    ax.text(xs[2] + W / 2, 0.27, "Idea 4:\ngen. text", ha="center", va="top",
            fontsize=5.6, color=PALETTE[2], fontweight="bold")

    # ---- router
    ax.plot([0.68, 0.68], [0.32 + H / 2, 0.53 + H / 2], color=INK_2, lw=0.9)
    ax.plot([0.668, 0.68], [0.32 + H / 2, 0.32 + H / 2], color=INK_2, lw=0.9)
    ax.plot([0.668, 0.68], [0.53 + H / 2, 0.53 + H / 2], color=INK_2, lw=0.9)
    box(ax, 0.70, 0.42, 0.115, 0.25, "Idea 1:\nrouter $\\rho(x)$\npicks $r$", "#e7e3f6", ec=PALETTE[6],
        fs=5.9, bold=True, tc=PALETTE[6])
    arrow(ax, 0.68, 0.545, 0.70, 0.545)

    # ---- model + objective
    box(ax, 0.84, 0.42, 0.155, 0.25, "frozen\nQwen2.5-Omni-3B\n$p_\\theta(\\cdot\\mid q, O_\\pi, x_r)$", "#eeeeee",
        ec="#bdbbb4", fs=5.9)
    arrow(ax, 0.815, 0.545, 0.84, 0.545)
    ax.text(0.9175, 0.38, "answer, averaged over\n3 option orders\n$\\max_\\rho\\ \\mathrm{acc}-\\lambda\\,\\mathrm{cost}$",
            ha="center", va="top", fontsize=5.7, color=INK)

    # ---- challenges
    ax.text(0.01, 0.06, "Challenges:  (C1) the useful modality changes per item.   (C2) modalities interact, so "
            "budgets cannot be set one at a time.   (C3) cheap representations can silently drop the deciding "
            "evidence, and producing them costs compute.", ha="left", va="center", fontsize=5.6, color=INK_2)
    save_fig(fig, "report_midterm/figures", "overview")


if __name__ == "__main__":
    main()
