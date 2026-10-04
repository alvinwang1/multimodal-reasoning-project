"""Input configurations and option permutations for Analysis A.

Every config contains the question and the four options. Raw modalities are passed
as media; text surrogates (caption / transcript) are inserted as text in their place.
"""
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


def surrogate_texts(ex, config):
    img, aud = CONFIGS[config]
    parts = []
    if img == "text":
        parts.append(CAPTION_PREFIX + ex["image content"].strip())
    if aud == "text":
        parts.append(TRANSCRIPT_PREFIX + ex["audio content"].strip())
    return parts


def build_messages(ex, config, perm):
    """Chat messages for one (example, config, permutation)."""
    img, aud = CONFIGS[config]
    content = []
    if img == "raw":
        content.append({"type": "image", "image": ex["image"]})
    if aud == "raw":
        content.append({"type": "audio", "audio": ex["audio"]})
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
