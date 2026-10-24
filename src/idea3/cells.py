"""Budget cells for Idea 3: (image representation, audio representation) pairs.

A cell name encodes both budgets:
    v<tokens>   raw image capped at <tokens> visual tokens (v0 = no image), or vcap = caption
    a<frac>     leading <frac> of the audio clip (a0 = no audio), a<frac>c = centered window,
                or atrans = transcript/description
e.g. v256_a0.5, v0_a1, vcap_atrans. Analysis A configs map onto cells via A_CONFIG_CELLS.
"""
import re

# Analysis A config -> cell (max_pixels there was 768 * 28 * 28, i.e. the same as v768)
A_CONFIG_CELLS = {
    "full": "v768_a1", "img_only": "v768_a0", "aud_only": "v0_a1", "q_only": "v0_a0",
    "img_trans": "v768_atrans", "cap_aud": "vcap_a1", "cap_trans": "vcap_atrans",
}
SURROGATE_CELLS = ["v768_atrans", "vcap_a1", "vcap_atrans"]
CORNERS = ["v768_a1", "v768_a0", "v0_a1", "v0_a0"]
TOKENS_PER_PATCH = 28 * 28  # one visual token per 28x28 pixels after Qwen's 2x2 patch merge

_CELL = re.compile(r"^v(\d+|cap)_a(trans|[\d.]+)(c?)$")


def fmt_frac(f):
    return f"{f:g}"


def cell_name(img, aud, center=False):
    a = aud if isinstance(aud, str) else fmt_frac(aud)
    return f"v{img}_a{a}{'c' if center else ''}"


def parse_cell(name):
    """-> dict(img=int|'cap', aud=float|'trans', center=bool)."""
    m = _CELL.match(name)
    if not m:
        raise ValueError(f"bad cell name {name!r}")
    img, aud, center = m.groups()
    return {"img": img if img == "cap" else int(img),
            "aud": aud if aud == "trans" else float(aud),
            "center": bool(center)}


def grid_cells(image_budgets, audio_fracs):
    return [cell_name(b, f) for b in image_budgets for f in audio_fracs]


def truncate_audio(wav, frac, sr, min_sec, center=False):
    """Keep `frac` of the clip (at least min_sec, at most all of it): leading segment or centered window."""
    if frac >= 1:
        return wav
    n = min(len(wav), max(int(round(frac * len(wav))), int(min_sec * sr)))
    start = (len(wav) - n) // 2 if center else 0
    return wav[start:start + n]
