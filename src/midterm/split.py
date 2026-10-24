"""Train/test split for the midterm baselines.

Train = the 90-item Assignment-2 subset + 30 new items per audio type (60/type, 180 total).
Test  = 40 new items per audio type (120 total), never looked at during Assignment 2.

    python -m src.midterm.split
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
    mcfg = cfg["midterm"]
    out = repo_path(mcfg["split"])
    if out.exists() and not args.force:
        print(f"{out} exists (use --force)")
        return

    records, _ = text_records(load_omnibench(cfg))
    by_id = {r["index"]: r for r in records}
    old = set(load_subset_ids(cfg))
    rng = np.random.default_rng(cfg["seed"] + 1)  # different stream from the Assignment-2 subset
    train, test = sorted(old), []
    for t in AUDIO_TYPES:
        pool = sorted(i for i, r in by_id.items() if r["audio type"] == t and i not in old)
        n_old = sum(by_id[i]["audio type"] == t for i in old)
        n_new_train = mcfg["n_train_per_type"] - n_old
        pick = rng.choice(pool, size=n_new_train + mcfg["n_test_per_type"], replace=False).tolist()
        train += pick[:n_new_train]
        test += pick[n_new_train:]
    train, test = sorted(int(i) for i in train), sorted(int(i) for i in test)
    assert not set(train) & set(test)
    payload = {"seed": cfg["seed"] + 1, "train": train, "test": test,
               "counts": {"train": dict(Counter(by_id[i]["audio type"] for i in train)),
                          "test": dict(Counter(by_id[i]["audio type"] for i in test))}}
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["counts"]), f"-> {out}")


if __name__ == "__main__":
    main()
