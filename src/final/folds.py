"""Cross-validation folds for the final report: every usable OmniBench item, stratified by audio type.

Every learned method (Idea 1 router, Idea 3 allocator) trains on 4 folds and predicts the 5th, so each
item gets exactly one out-of-fold prediction. The file also records which earlier sets each item was in,
so results can be broken out for the midterm test items and for items no one has inspected yet.

    python -m src.final.folds            # -> data/final_folds.json (refuses to overwrite)
"""
import argparse
import json
from collections import Counter

import numpy as np

from src.common import load_config, repo_path
from src.data import AUDIO_TYPES, load_omnibench, load_subset_ids, text_records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fcfg = cfg["final"]
    out = repo_path(fcfg["folds"])
    if out.exists() and not args.force:
        print(f"{out} exists; the folds are frozen (use --force only if the whole team agrees)")
        return

    records, dropped = text_records(load_omnibench(cfg))
    by_id = {r["index"]: r for r in records}
    k = fcfg["n_folds"]
    rng = np.random.default_rng(fcfg["fold_seed"])

    # stratify by audio type: shuffle each type, then deal its items round-robin across the folds
    fold_of = {}
    for t in AUDIO_TYPES:
        ids = sorted(i for i, r in by_id.items() if r["audio type"] == t)
        start = int(rng.integers(k))  # rotate so the leftover items don't always land in fold 0
        for j, i in enumerate(rng.permutation(ids).tolist()):
            fold_of[i] = (start + j) % k

    split = json.loads(repo_path(cfg["midterm"]["split"]).read_text())
    groups = {
        "a2_subset": sorted(load_subset_ids(cfg)),
        "midterm_train": sorted(split["train"]),
        "midterm_test": sorted(split["test"]),
        "idea3_test": sorted(json.loads(repo_path(cfg["idea3"]["test_subset"]).read_text())["ids"]),
    }
    seen = set().union(*groups.values())
    groups["fresh"] = sorted(i for i in by_id if i not in seen)  # never scored or inspected before the final

    folds = [sorted(i for i, f in fold_of.items() if f == j) for j in range(k)]
    payload = {
        "seed": fcfg["fold_seed"],
        "n_folds": k,
        "stratified_by": "audio type",
        "n": len(fold_of),
        "dropped_ambiguous": dropped,
        "per_fold_audio_type": [dict(Counter(by_id[i]["audio type"] for i in f)) for f in folds],
        "group_sizes": {g: len(v) for g, v in groups.items()},
        "fresh_audio_type": dict(Counter(by_id[i]["audio type"] for i in groups["fresh"])),
        "folds": folds,
        "groups": groups,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {len(fold_of)} items in {k} folds -> {out}")
    for j, c in enumerate(payload["per_fold_audio_type"]):
        print(f"  fold {j}: {len(folds[j])} items {c}")
    print("groups:", payload["group_sizes"], "| fresh by type:", payload["fresh_audio_type"])


if __name__ == "__main__":
    main()
