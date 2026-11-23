"""Input configurations and option permutations for Analysis A.

Every config contains the question and the four options. Raw modalities are passed
as media; text surrogates (caption / transcript) are inserted as text in their place.
"""
import re

import numpy as np

from src.data import LETTERS

# config name -> (image representation, audio representation); None = omitted
CONFIGS = {
    "full":      ("raw",     "raw"),
    "img_trans": ("raw",     "text"),
    "cap_aud":   ("text",    "raw"),
    "cap_trans": ("text",    "text"),
    "img_only":  ("raw",     None),
    "aud_only":  (None,      "raw"),
    "q_only":    (None,      None),
}

# Midterm configs. Generated surrogates come from src/midterm/surrogates.py:
#   gen      = Qwen2.5-Omni's own image caption, gen_desc = its own audio description, asr = Whisper transcript
GEN_CONFIGS = {
    "gen_cap_desc": ("gen", "gen_desc"),
    "gen_cap_asr":  ("gen", "asr"),
    "img_gendesc":  ("raw", "gen_desc"),
    "img_asr":      ("raw", "asr"),
    "gencap_aud":   ("gen", "raw"),
    # Final-report Idea 4 (src/final/idea4_generate.py):
    #   route    = Whisper transcript if the clip is classified as speech, else AST audio tags
    #   route_fb = route, but raw audio when the Whisper output looks broken (empty / filler / loop)
    #   qdesc    = Qwen audio description conditioned on the question (not the options)
    #   pipe     = full pipeline: speech -> Whisper (raw audio if broken); other audio -> AST tags + qdesc
    #   masked   = the human audio description with every option word masked (answer-leakage test)
    "img_route":    ("raw", "route"),
    "img_route_fb": ("raw", "route_fb"),
    "img_qdesc":    ("raw", "qdesc"),
    "img_pipe":     ("raw", "pipe"),
    "gencap_pipe":  ("gen", "pipe"),
    "img_masked":   ("raw", "masked"),
    "cap_masked":   ("text", "masked"),
    # Routing branches scored on every item, so the speech threshold can be swept afterwards (notebook §9c):
    #   tags  = AST audio tags only (what route gives non-speech clips); tagsq = tags + qdesc (what pipe gives them)
    "img_tags":     ("raw", "tags"),
    "img_tagsq":    ("raw", "tagsq"),
    "gencap_tagsq": ("gen", "tagsq"),
}
FALLBACK_REPS = {"route_fb", "pipe"}  # these become raw audio for items whose ASR output was flagged as broken
# Final-report Idea 4 generation-length ablation: <config>_t<n> (e.g. img_qdesc_t80) is <config> with every Qwen-generated
# text capped at n new tokens (fields <field>_t<n>, written by src/final/idea4_generate.py). Only these reps are generated.
GEN_LEN_REPS = {"gen", "gen_desc", "qdesc", "pipe", "tagsq"}
BASE_IMAGE_PIXELS = 602112  # 768 * 28 * 28, the full-resolution cap used everywhere


def config_spec(config):
    """(image repr, audio repr, image keep-ratio, audio keep-ratio) for any config name.

    Budget configs are named bud_i{r_img}_a{r_aud}: the image is resized to r_img of the pixel cap
    (~r_img of the visual tokens) and the audio is time-stretched by 1/r_aud (~r_aud of the audio
    tokens, same content played faster); a ratio of 0 drops that modality.
    Generation-length configs are named <generated-text config>_t<n>, e.g. img_qdesc_t80 (adds gen_len=n).
    """
    if config in CONFIGS:
        img, aud = CONFIGS[config]
        return {"img": img, "aud": aud, "img_ratio": 1.0, "aud_ratio": 1.0}
    if config in GEN_CONFIGS:
        img, aud = GEN_CONFIGS[config]
        return {"img": img, "aud": aud, "img_ratio": 1.0, "aud_ratio": 1.0}
    m = re.fullmatch(r"bud_i([0-9.]+)_a([0-9.]+)", config)
    if m:
        ri, ra = float(m.group(1)), float(m.group(2))
        return {"img": "raw" if ri > 0 else None, "aud": "raw" if ra > 0 else None,
                "img_ratio": ri, "aud_ratio": ra}
    m = re.fullmatch(r"(\w+)_t(\d+)", config)
    if m and m.group(1) in GEN_CONFIGS:
        img, aud = GEN_CONFIGS[m.group(1)]
        if img not in GEN_LEN_REPS and aud not in GEN_LEN_REPS:
            raise ValueError(f"{config!r}: {m.group(1)} uses no Qwen-generated text, so a length cap changes nothing")
        if int(m.group(2)) < 1:
            raise ValueError(f"{config!r}: the generation length must be at least 1 token")
        return {"img": img, "aud": aud, "img_ratio": 1.0, "aud_ratio": 1.0, "gen_len": int(m.group(2))}
    raise ValueError(f"unknown config {config!r}")


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech."
)
INSTRUCTION = "Answer with the option's letter from the given choices directly."
CAPTION_PREFIX = "Image description: "
TRANSCRIPT_PREFIX = "Audio transcript/description: "


def permutation(example_idx, perm_id, seed, n=4, prior=()):
    """perm[j] = original option index shown at letter j. perm 0 is the identity.

    Seeded per (seed, example, perm) and re-drawn until it differs from all earlier perms,
    so every extra permutation actually moves the options.
    """
    if perm_id == 0:
        return list(range(n))
    rng = np.random.default_rng([seed, int(example_idx), perm_id])
    seen = {tuple(p) for p in prior} | {tuple(range(n))}
    while True:
        p = rng.permutation(n).tolist()
        if tuple(p) not in seen:
            return p


def audio_rep(ex, spec):
    """The audio representation actually used for this item (fallback configs switch to raw audio)."""
    if spec["aud"] in FALLBACK_REPS and ex.get("asr_failed"):
        return "raw"
    return spec["aud"]


def surrogate_fields(config):
    """Fields this config reads from a --surrogates file (keep in sync with surrogate_texts / audio_rep)."""
    spec = config_spec(config)
    t = f"_t{spec['gen_len']}" if "gen_len" in spec else ""
    img = {"gen": ["gen_caption" + t]}
    aud = {"gen_desc": ["gen_audio_desc" + t], "asr": ["asr"], "route": ["route_text"],
           "route_fb": ["route_text", "asr_failed"], "qdesc": ["q_audio_desc" + t],
           "pipe": ["pipe_text" + t, "asr_failed"], "masked": ["masked_audio_content"], "tags": ["tag_text"],
           "tagsq": ["tagsq_text" + t]}
    return img.get(spec["img"], []) + aud.get(spec["aud"], [])


def surrogate_texts(ex, config):
    spec = config_spec(config)
    t = f"_t{spec['gen_len']}" if "gen_len" in spec else ""  # length-ablation configs read the capped texts
    img_text = {"text": lambda: ex["image content"], "gen": lambda: ex["gen_caption" + t]}
    aud_text = {"text": lambda: ex["audio content"], "gen_desc": lambda: ex["gen_audio_desc" + t],
                "asr": lambda: ex["asr"], "route": lambda: ex["route_text"], "route_fb": lambda: ex["route_text"],
                "qdesc": lambda: ex["q_audio_desc" + t], "pipe": lambda: ex["pipe_text" + t],
                "masked": lambda: ex["masked_audio_content"], "tags": lambda: ex["tag_text"],
                "tagsq": lambda: ex["tagsq_text" + t]}
    parts = []
    if spec["img"] in img_text:
        parts.append(CAPTION_PREFIX + img_text[spec["img"]]().strip())
    aud = audio_rep(ex, spec)
    if aud in aud_text:
        parts.append(TRANSCRIPT_PREFIX + aud_text[aud]().strip())
    return parts


def stretched_audio(ex, ratio):
    """Audio sped up by 1/ratio with pitch preserved (phase vocoder); cached per example."""
    if ratio >= 1:
        return ex["audio"]
    cache = ex.setdefault("_stretch", {})
    if ratio not in cache:
        import librosa
        cache[ratio] = librosa.effects.time_stretch(ex["audio"], rate=1.0 / ratio).astype(np.float32)
    return cache[ratio]


def build_messages(ex, config, perm):
    """Chat messages for one (example, config, permutation)."""
    spec = config_spec(config)
    content = []
    if spec["img"] == "raw":
        content.append({"type": "image", "image": ex["image"],
                        "max_pixels": int(spec["img_ratio"] * BASE_IMAGE_PIXELS)})
    if audio_rep(ex, spec) == "raw":
        content.append({"type": "audio", "audio": stretched_audio(ex, spec["aud_ratio"])})
    lines = surrogate_texts(ex, config)
    lines.append(f"Question: {ex['question'].strip()}")
    lines.append("Options:")
    lines += [f"{LETTERS[j]}. {ex['options'][perm[j]]}" for j in range(len(perm))]
    lines.append(INSTRUCTION)
    content.append({"type": "text", "text": "\n".join(lines)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
