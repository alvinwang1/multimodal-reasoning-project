"""Text-sufficiency labels for Analysis C.

model: results/text_sufficient_labels.json from Analysis A (caption+transcript correct
       under every option permutation).
human: results/analysis_b/human_text_sufficient.json from Analysis B (annotators' stage-0
       "is text enough?" judgment; majority vote on shared items).
"""
import json

from src.common import repo_path

LABEL_FILES = {
    "model": "results/text_sufficient_labels.json",
    "human": "results/analysis_b/human_text_sufficient.json",
}
LABEL_TAGS = {"model": "", "human": " [human labels]"}
FILE_SUFFIX = {"model": "", "human": "_humanlabels"}


def load_labels(source="model"):
    """Return {example id: 0/1}, or None if the label file does not exist yet."""
    path = repo_path(LABEL_FILES[source])
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {int(k): int(v) for k, v in data["labels"].items()}
