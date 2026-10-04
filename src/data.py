"""Load OmniBench and draw the shared, audio-type-balanced subset.

    python -m src.data                 # print schema + one example, build data/subset_ids.json
    python -m src.data --n 90 --force  # rebuild with a different size
"""
import argparse
import io
import json
from collections import Counter

import numpy as np

from src.common import load_config, repo_path, seed_everything

TEXT_FIELDS = {"task type", "audio type", "question", "audio content", "image content",
               "audio_path", "image_path", "index", "answer", "options"}
MEDIA_FIELDS = {"audio", "image"}
AUDIO_TYPES = ["speech", "sound event", "music"]
LETTERS = ["A", "B", "C", "D"]


def load_omnibench(cfg):
    """Return the HF dataset with media left as raw bytes (we decode ourselves)."""
    from datasets import Audio, Image, load_dataset

    ds = load_dataset(cfg["dataset"]["hf_id"], split=cfg["dataset"]["split"])
    missing = (TEXT_FIELDS | MEDIA_FIELDS) - set(ds.column_names)
    assert not missing, f"OmniBench schema drift, missing fields: {missing}"
    ds = ds.cast_column("audio", Audio(decode=False)).cast_column("image", Image(decode=False))
    return ds


def text_records(ds):
    """All rows without media, plus answer_idx. Items whose answer is ambiguous are dropped."""
    rows = ds.remove_columns(list(MEDIA_FIELDS)).to_list()
    out, dropped = [], []
    for pos, r in enumerate(rows):
        if len(set(r["options"])) != len(r["options"]) or r["options"].count(r["answer"]) != 1:
            dropped.append(r["index"])
            continue
        r["answer_idx"] = r["options"].index(r["answer"])
        r["_pos"] = pos
        out.append(r)
    return out, dropped


def build_subset(records, n, seed):
    """Fixed-seed subset, balanced across audio types. Short categories are taken whole."""
    rng = np.random.default_rng(seed)
    by_type = {t: sorted(r["index"] for r in records if r["audio type"] == t) for t in AUDIO_TYPES}
    share = {t: n // len(AUDIO_TYPES) + (i < n % len(AUDIO_TYPES)) for i, t in enumerate(AUDIO_TYPES)}
    notes, ids = [], {}
    for t in AUDIO_TYPES:
        if len(by_type[t]) < share[t]:
            notes.append(f"'{t}' has only {len(by_type[t])} items (< {share[t]}); took all of them")
            ids[t] = by_type[t]
        else:
            ids[t] = sorted(rng.choice(by_type[t], size=share[t], replace=False).tolist())
    return ids, notes


def load_subset_ids(cfg):
    with open(repo_path(cfg["subset"]["path"])) as f:
        return json.load(f)["ids"]


def decode_audio(audio_field, sr):
    import librosa
    wav, _ = librosa.load(io.BytesIO(audio_field["bytes"]), sr=sr, mono=True)
    return wav.astype(np.float32)


def decode_image(image_field):
    from PIL import Image
    return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")


class OmniBenchSubset:
    """Index-addressable access to decoded examples (image as PIL, audio as 16 kHz float32)."""

    def __init__(self, cfg, ids=None):
        self.cfg = cfg
        self.ds = load_omnibench(cfg)
        records, _ = text_records(self.ds)
        self.records = {r["index"]: r for r in records}
        self.ids = ids if ids is not None else load_subset_ids(cfg)

    def text(self, idx):
        return self.records[idx]

    def __getitem__(self, idx):
        r = dict(self.records[idx])
        row = self.ds[r["_pos"]]
        r["image"] = decode_image(row["image"])
        r["audio"] = decode_audio(row["audio"], self.cfg["dataset"]["audio_sr"])
        return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=None, help="subset size (default from config)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing subset file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    seed_everything(seed)
    n = args.n or cfg["subset"]["n"]

    ds = load_omnibench(cfg)
    print(ds)
    print("\nFeatures:")
    for k, v in ds.features.items():
        print(f"  {k!r}: {v}")

    records, dropped = text_records(ds)
    ex = records[0]
    print("\nExample record:")
    for k in ["index", "task type", "audio type", "question", "options", "answer", "answer_idx"]:
        print(f"  {k:12s}: {ex[k]}")
    print(f"  audio content: {ex['audio content'][:200]}")
    print(f"  image content: {ex['image content'][:200]}...")
    print(f"\n{len(records)} usable records; dropped {len(dropped)} with ambiguous answers: {dropped}")
    print("audio types:", dict(Counter(r["audio type"] for r in records)))

    out = repo_path(cfg["subset"]["path"])
    if out.exists() and not args.force:
        print(f"\n{out} exists; not overwriting (use --force).")
        return
    ids_by_type, notes = build_subset(records, n, seed)
    for note in notes:
        print("NOTE:", note)
    ids = sorted(i for v in ids_by_type.values() for i in v)
    by_idx = {r["index"]: r for r in records}
    payload = {
        "seed": seed,
        "n": len(ids),
        "per_audio_type": {t: len(v) for t, v in ids_by_type.items()},
        "per_task_type": dict(Counter(by_idx[i]["task type"] for i in ids).most_common()),
        "dropped_ambiguous": dropped,
        "notes": notes,
        "ids": ids,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(ids)} ids to {out}")
    print("per audio type:", payload["per_audio_type"])
    print("per task type: ", payload["per_task_type"])


if __name__ == "__main__":
    main()
