"""Shared helpers: config loading, seeding, paths."""
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path=None):
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def repo_path(rel):
    """Resolve a config path relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- plotting
# Categorical palette in a fixed order (CVD-validated ordering); assign by entity, never cycle.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W, PAGE_W = 3.25, 6.75  # ICML single column / full width, inches

CONFIG_ORDER = ["full", "img_trans", "cap_aud", "cap_trans", "img_only", "aud_only", "q_only"]
CONFIG_LABELS = {"full": "Image+Audio", "img_trans": "Image+Transcript", "cap_aud": "Caption+Audio",
                 "cap_trans": "Caption+Transcript", "img_only": "Image only", "aud_only": "Audio only",
                 "q_only": "Question only"}
CONFIG_COLORS = dict(zip(CONFIG_ORDER, PALETTE))
AUDIO_TYPE_COLORS = {"speech": PALETTE[0], "sound event": PALETTE[1], "music": PALETTE[2]}
BINARY_COLORS = {0: "#b9b8b2", 1: PALETTE[0]}  # recessive gray vs. highlighted


def set_plot_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "legend.title_fontsize": 6.5, "legend.frameon": False,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_fig(fig, out_dir, name):
    """Save as vector PDF (for LaTeX) and 300-dpi PNG."""
    import matplotlib.pyplot as plt
    out_dir = repo_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    return out_dir / f"{name}.pdf"
