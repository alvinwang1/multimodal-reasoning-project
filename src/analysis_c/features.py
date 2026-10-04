"""Language features for Analysis C: Sentence-BERT embeddings + Empath categories.

Three text views per subset example:
  question     the question text
  q_cap_trans  question + image caption + audio transcript
  transcript   the audio transcript alone

    python -m src.analysis_c.features                 # -> results/analysis_c/features.npz
    python -m src.analysis_c.features --device mps
"""
import argparse

import numpy as np

from src.common import load_config, repo_path, seed_everything
from src.data import load_omnibench, load_subset_ids, text_records

VIEWS = ["question", "q_cap_trans", "transcript"]
VIEW_LABELS = {"question": "Question", "q_cap_trans": "Question+Caption+Transcript",
               "transcript": "Transcript"}


def view_texts(r):
    q, cap, tr = r["question"].strip(), r["image content"].strip(), r["audio content"].strip()
    return {
        "question": q,
        "q_cap_trans": f"{q}\nImage description: {cap}\nAudio transcript/description: {tr}",
        "transcript": tr,
    }


def empath_matrix(lexicon, texts):
    cats = sorted(lexicon.cats)
    rows = []
    for t in texts:
        if not t.split():
            rows.append(np.zeros(len(cats)))
            continue
        d = lexicon.analyze(t, categories=cats, normalize=True)
        rows.append(np.array([d[c] for c in cats], dtype=float))
    return np.vstack(rows), cats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu", help="cpu (default; leaves the GPU to Analysis A) / mps / cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ccfg = cfg["analysis_c"]
    seed_everything(cfg["seed"])

    from empath import Empath
    from sentence_transformers import SentenceTransformer

    records, _ = text_records(load_omnibench(cfg))
    by_id = {r["index"]: r for r in records}
    ids = load_subset_ids(cfg)
    texts = {v: [view_texts(by_id[i])[v] for i in ids] for v in VIEWS}

    sbert = SentenceTransformer(ccfg["sbert_model"], device=args.device)
    lexicon = Empath()
    out = {
        "ids": np.array(ids),
        "audio_type": np.array([by_id[i]["audio type"] for i in ids]),
        "task_type": np.array([by_id[i]["task type"] for i in ids]),
    }
    for v in VIEWS:
        out[f"emb_{v}"] = sbert.encode(texts[v], batch_size=16, normalize_embeddings=True,
                                       show_progress_bar=False).astype(np.float32)
        out[f"empath_{v}"], cats = empath_matrix(lexicon, texts[v])
        out[f"nwords_{v}"] = np.array([len(t.split()) for t in texts[v]])
        print(f"{v:12s} emb {out[f'emb_{v}'].shape}  empath {out[f'empath_{v}'].shape}  "
              f"words median {int(np.median(out[f'nwords_{v}']))}  "
              f"empath nonzero cats/text {np.mean((out[f'empath_{v}'] > 0).sum(1)):.1f}")
    out["empath_cats"] = np.array(cats)

    path = repo_path(ccfg["features"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
