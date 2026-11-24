"""CPU check before the GPU runs: every item in the final folds loads (image and audio decoding) and builds the prompt of
every base config and Idea 3 cell the runs use (incl. the image resizing at 64 and 768 tokens); for the Idea 4 subset
the option masking runs too. An item that fails here would otherwise stop a GPU section at that item on every resume.

    python -m src.final.preflight            # all 1,141 items, a few minutes on CPU
    python -m src.final.preflight --limit 20
"""
import argparse
import json
import time

from src.analysis_a.prompts import CONFIGS, build_messages
from src.common import load_config, repo_path
from src.data import OmniBenchSubset
from src.final.run_all import OTHER_CELLS, PROBE_CELLS, SELECT_CELLS
from src.idea3.run import cell_inputs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    from qwen_omni_utils import fetch_image  # noqa: E402  (heavy imports only when it actually runs)
    from src.final.idea4_generate import mask_options

    ids = sorted(i for f in json.loads(repo_path(cfg["final"]["folds"]).read_text())["folds"] for i in f)
    ids = ids[:args.limit] if args.limit else ids
    idea4 = set(json.loads(repo_path(cfg["final"]["idea4_ids"]).read_text())["ids"])
    data = OmniBenchSubset(cfg)
    min_px = cfg["model"]["min_pixels"]
    bad, t0 = [], time.time()
    for n, i in enumerate(ids, 1):
        try:
            ex = data[i]
            assert len(ex["options"]) == 4, f"{len(ex['options'])} options"
            assert len(ex["audio"]) > 0, "empty audio"
            for c in CONFIGS:
                build_messages(ex, c, [0, 1, 2, 3])
            for cell in PROBE_CELLS + OTHER_CELLS + SELECT_CELLS:
                ex_c, config, _, _ = cell_inputs(ex, cell, cfg)
                build_messages(ex_c, config, [0, 1, 2, 3])
            for tokens in (64, 768):
                fetch_image({"image": ex["image"], "min_pixels": min_px, "max_pixels": tokens * 28 * 28})
            if i in idea4:
                mask_options(ex["audio content"], ex["options"])
        except Exception as e:  # report every problem, not only the first
            bad.append((i, repr(e)))
        if n % 100 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} items checked, {len(bad)} problem(s), {time.time() - t0:.0f} s", flush=True)
    for i, e in bad:
        print(f"  id {i}: {e}")
    if bad:
        raise SystemExit(f"{len(bad)} item(s) would crash the GPU runs; send the list above to the team before starting")
    print("preflight ok: every item loads and builds its prompts")


if __name__ == "__main__":
    main()
