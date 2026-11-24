"""Export OmniBench's text fields for every item in the final folds (CPU; needs the dataset download).

    python -m src.final.export_text            # -> results/final/omnibench_text.json
    On Modal: modal run scripts/modal_run.py --sections text   (the dataset is already on the Modal Volume)

One record per item id: the question, the options, the answer and its index, the human-written "audio content"
(transcript or description) and "image content" (caption), the task type and the audio type. Analyses that need the
question text (Idea 4 translation quality, the transcript-loss coding sheet, failure cases) read this file, so they run
on a laptop without the dataset download. No media is exported.
"""
import argparse
import json

from src.common import load_config, repo_path
from src.data import load_omnibench, text_records

OUT = "results/final/omnibench_text.json"
FIELDS = ["question", "options", "answer", "answer_idx", "audio content", "image content", "task type", "audio type"]


def export(cfg, records, dropped, out=OUT):
    ids = sorted(i for f in json.loads(repo_path(cfg["final"]["folds"]).read_text())["folds"] for i in f)
    by_id = {r["index"]: r for r in records}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} items of the folds are not in the dataset (e.g. {missing[0]})")
    data = {str(i): {k: by_id[i][k] for k in FIELDS} for i in ids}
    path = repo_path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"{len(data)} items -> {path} ({path.stat().st_size / 2**20:.1f} MB); "
          f"items dropped as ambiguous (not in the folds): {dropped}", flush=True)
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    cfg = load_config(args.config)
    records, dropped = text_records(load_omnibench(cfg))
    export(cfg, records, dropped, args.out)


if __name__ == "__main__":
    main()
