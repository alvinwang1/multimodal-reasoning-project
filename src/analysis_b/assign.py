"""Seeded item assignment for the staged-reveal annotation (Analysis B).

10 items shared by all annotators + 10 unique per annotator, drawn from the shared subset
and balanced across audio types. Also exports each assigned item's image/audio to
data/cache/analysis_b_media/ so the annotation tool does not need to decode the dataset.

    python -m src.analysis_b.assign            # writes results/analysis_b/assignments.json
    python -m src.analysis_b.assign --force    # overwrite (only before annotation starts!)
"""
import argparse
import json
from collections import Counter

import numpy as np

from src.common import load_config, repo_path
from src.data import AUDIO_TYPES, load_omnibench, load_subset_ids, text_records


def interleave(pools):
    """One continuous speech -> sound event -> music rotation over the (shuffled) pools."""
    out, pools = [], {t: list(v) for t, v in pools.items()}
    while any(pools.values()):
        for t in AUDIO_TYPES:
            if pools[t]:
                out.append(pools[t].pop())
    return out


def make_assignments(records_by_id, ids, annotators, n_shared, n_unique, seed):
    rng = np.random.default_rng(seed)
    pools = {t: [] for t in AUDIO_TYPES}
    for i in ids:
        pools[records_by_id[i]["audio type"]].append(i)
    for t in AUDIO_TYPES:
        pools[t] = [int(x) for x in rng.permutation(sorted(pools[t]))]

    need = n_shared + n_unique * len(annotators)
    if need > len(ids):
        raise ValueError(f"need {need} distinct items but the subset has {len(ids)}")

    order = interleave(pools)
    shared, rest = order[:n_shared], order[n_shared:]
    unique = {a: [] for a in annotators}
    for k in range(n_unique * len(annotators)):  # deal like cards so each person gets a balanced mix
        unique[annotators[k % len(annotators)]].append(rest[k])

    per = {}
    for a in annotators:
        per[a] = [int(x) for x in rng.permutation(shared + unique[a])]  # shared items mixed in
    return shared, unique, per


def export_media(cfg, ids):
    ds = load_omnibench(cfg)
    records, _ = text_records(ds)
    pos = {r["index"]: r["_pos"] for r in records}
    by_id = {r["index"]: r for r in records}
    out = repo_path(cfg["analysis_b"]["media_dir"])
    out.mkdir(parents=True, exist_ok=True)
    items = {}
    for i in ids:
        row = ds[pos[i]]
        files = {}
        for field in ("image", "audio"):
            ext = (row[field].get("path") or "").rsplit(".", 1)[-1].lower() or ("jpg" if field == "image" else "mp3")
            files[field] = f"{i}_{field}.{ext}"
            (out / files[field]).write_bytes(row[field]["bytes"])
        r = by_id[i]
        # No gold answer here: this file is what annotators' tool reads.
        items[str(i)] = {"question": r["question"], "options": r["options"], "caption": r["image content"],
                         "transcript": r["audio content"], "audio_type": r["audio type"],
                         "task_type": r["task type"], **files}
    (out / "items.json").write_text(json.dumps(items, indent=2))
    print(f"exported media + items.json for {len(ids)} items to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--media-only", action="store_true", help="re-export media for existing assignments")
    args = ap.parse_args()

    cfg = load_config(args.config)
    bcfg = cfg["analysis_b"]
    path = repo_path(bcfg["assignments"])

    if args.media_only or (path.exists() and not args.force):
        if not args.media_only:
            print(f"{path} exists; not overwriting (use --force). Re-exporting media.")
        assign = json.loads(path.read_text())
        export_media(cfg, sorted({i for v in assign["per_annotator"].values() for i in v}))
        return

    ds = load_omnibench(cfg)
    records, _ = text_records(ds)
    by_id = {r["index"]: r for r in records}
    ids = load_subset_ids(cfg)
    shared, unique, per = make_assignments(by_id, ids, bcfg["annotators"], bcfg["n_shared"],
                                           bcfg["n_unique"], cfg["seed"])
    payload = {
        "seed": cfg["seed"],
        "shared": shared,
        "unique": unique,
        "per_annotator": per,
        "audio_type_counts": {
            "shared": dict(Counter(by_id[i]["audio type"] for i in shared)),
            **{a: dict(Counter(by_id[i]["audio type"] for i in per[a])) for a in per},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    for k, v in payload["audio_type_counts"].items():
        print(f"  {k:8s} {v}")
    export_media(cfg, sorted({i for v in per.values() for i in v}))


if __name__ == "__main__":
    main()
