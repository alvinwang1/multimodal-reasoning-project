"""Budget cells for Idea 3: (image representation, audio representation) pairs.

A cell name encodes both budgets:
    v<tokens>   raw image capped at <tokens> visual tokens (v0 = no image), or vcap = caption
    a<frac>     leading <frac> of the audio clip (a0 = no audio), a<frac>c = centered window,
                a<frac>e = loudest window (highest signal energy), or atrans = transcript/description
e.g. v256_a0.5, v0_a1, vcap_atrans. Analysis A configs map onto cells via A_CONFIG_CELLS.
"""
import re

import numpy as np

# Analysis A config -> cell (max_pixels there was 768 * 28 * 28, i.e. the same as v768)
A_CONFIG_CELLS = {
    "full": "v768_a1", "img_only": "v768_a0", "aud_only": "v0_a1", "q_only": "v0_a0",
    "img_trans": "v768_atrans", "cap_aud": "vcap_a1", "cap_trans": "vcap_atrans",
}
SURROGATE_CELLS = ["v768_atrans", "vcap_a1", "vcap_atrans"]
CORNERS = ["v768_a1", "v768_a0", "v0_a1", "v0_a0"]
TOKENS_PER_PATCH = 28 * 28  # one visual token per 28x28 pixels after Qwen's 2x2 patch merge

_CELL = re.compile(r"^v(\d+|cap)_a(trans|[\d.]+)([ce]?)$")
WINDOWS = {"": "lead", "c": "center", "e": "energy"}
HOP = 160  # candidate window starts every 10 ms at 16 kHz


def fmt_frac(f):
    return f"{f:g}"


def cell_name(img, aud, window="lead"):
    a = aud if isinstance(aud, str) else fmt_frac(aud)
    return f"v{img}_a{a}{ {v: k for k, v in WINDOWS.items()}[window] }"


def parse_cell(name):
    """-> dict(img=int|'cap', aud=float|'trans', window='lead'|'center'|'energy', center=bool)."""
    m = _CELL.match(name)
    if not m:
        raise ValueError(f"bad cell name {name!r}")
    img, aud, flag = m.groups()
    return {"img": img if img == "cap" else int(img),
            "aud": aud if aud == "trans" else float(aud),
            "window": WINDOWS[flag], "center": flag == "c"}


def grid_cells(image_budgets, audio_fracs):
    return [cell_name(b, f) for b in image_budgets for f in audio_fracs]


def truncate_audio(wav, frac, sr, min_sec, window="lead"):
    """Keep `frac` of the clip (at least min_sec, at most all of it) as one contiguous window:
    the leading segment, the centered window, or the window with the most signal energy."""
    if frac >= 1:
        return wav
    n = min(len(wav), max(int(round(frac * len(wav))), int(min_sec * sr)))
    if window is True or window == "center":  # True: older callers passed center=True
        start = (len(wav) - n) // 2
    elif window == "energy":
        c = np.concatenate([[0.0], np.cumsum(wav.astype(np.float64) ** 2)])
        starts = np.arange(0, len(wav) - n + 1, HOP)
        start = int(starts[np.argmax(c[starts + n] - c[starts])])
    else:
        start = 0
    return wav[start:start + n]
