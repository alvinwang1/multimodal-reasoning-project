"""Shared score bank for the final report: one row per (item, representation), averaged over option orders.

Sources. All final scoring runs use the Colab L4 (bf16), so every idea shares one GPU type:
  base      7 base configs, L4         results/final/bank_base_l4.jsonl  (colab/run_final.ipynb)
  idea3     Idea 3 budget cells, L4    results/idea3/raw_final.jsonl     (python -m src.idea3.run --split final)
  idea4     Idea 4 generated text, L4  results/final/raw_idea4_l4.jsonl  (src.final.idea4_generate, then run --split idea4)
            answering cost only: add the generation time from results/final/idea4_surrogates.json (*_ms fields)
  base_mac  7 base configs, M2 Max     results/analysis_a/raw.jsonl      (midterm runs; for comparison only)
Idea 3 representations are prefixed "i3:" (e.g. "i3:v64_a0.5") and Idea 4 ones "i4:" (e.g. "i4:img_pipe").
Idea 4 generation-length ablation reps end in _t<tokens> (e.g. "i4:img_qdesc_t80"); the unsuffixed Idea 4 reps use the
full 256-token generation, i.e. they are the 256 point of that ablation.
The Idea 3 grid corners and text cells are the same inputs as the base configs, so they are not re-run:
use src.idea3.cells.A_CONFIG_CELLS to map between the two names.

Columns: id, rep, source, audio_type, task_type, acc (mean correctness over orders), logp (mean log p(gold)),
max_prob, tokens, cost (tokens / the same item's full-input tokens), n_perms, and time in seconds:
model_s (forward pass), preproc_s (resize, audio cut, features, tokenize), decode_s (reading the media files),
latency_s = model_s + preproc_s (midterm definition), e2e_s = latency_s + decode_s (end to end).
Only (item, rep) pairs with all option orders present are kept.

    from src.final.bank import load_bank, matrix, load_folds
    bank = load_bank()                      # long table
    acc = matrix(bank, "acc")               # items x reps
    folds = load_folds()                    # list of 5 id lists

    python -m src.final.bank                # coverage report
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.common import load_config, read_jsonl, repo_path

EPS = 1e-6
SOURCES = {
    "base": dict(path=lambda cfg: cfg["final"]["base_out"], key="config", prefix="", full="full"),
    "idea3": dict(path=lambda cfg: f"{cfg['idea3']['out_dir']}/raw_final.jsonl", key="cell", prefix="i3:",
                  full="full"),  # cost relative to the base full input (the corner cells are not re-run)
    "idea4": dict(path=lambda cfg: cfg["final"]["idea4_out"], key="config", prefix="i4:", full="full"),
    "base_mac": dict(path=lambda cfg: cfg["analysis_a"]["out"], key="config", prefix="mac:", full="mac:full"),
}


def _load_source(cfg, name, n_perms):
    spec = SOURCES[name]
    df = pd.DataFrame(read_jsonl(repo_path(spec["path"](cfg))))
    if df.empty:
        return df
    df = df.drop_duplicates(["id", spec["key"], "perm"], keep="last")
    df["rep"] = spec["prefix"] + df[spec["key"]].astype(str)
    df["logp"] = np.log(np.maximum([p[a] for p, a in zip(df.probs, df.answer_idx)], EPS))
    df["model_s"] = df.latency_ms / 1000
    df["preproc_s"] = df.preproc_ms / 1000
    df["decode_s"] = df["decode_ms"] / 1000 if "decode_ms" in df else np.nan  # absent in midterm runs
    df["latency_s"] = df.model_s + df.preproc_s  # model + processor (the midterm definition)
    df["e2e_s"] = df.latency_s + df.decode_s     # + decoding the media files
    g = df.groupby(["id", "rep"])
    out = g.agg(audio_type=("audio_type", "first"), task_type=("task_type", "first"), acc=("correct", "mean"),
                logp=("logp", "mean"), max_prob=("max_prob", "mean"), tokens=("n_total_tok", "mean"),
                latency_s=("latency_s", "mean"), model_s=("model_s", "mean"), preproc_s=("preproc_s", "mean"),
                decode_s=("decode_s", "mean"), e2e_s=("e2e_s", "mean"),
                n_perms=("perm", "nunique")).reset_index()
    out = out[out.n_perms == n_perms]
    out["source"] = name
    return out


def load_bank(cfg=None, sources=("base", "idea3", "idea4"), n_perms=3):
    """Long table over the given sources (add "base_mac" to compare with the midterm Mac runs)."""
    cfg = cfg or load_config()
    frames = [_load_source(cfg, s, n_perms) for s in sources]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    bank = pd.concat(frames, ignore_index=True)
    # cost = tokens / the same item's full-input tokens; NaN until that full input has been scored
    bank["cost"] = np.nan
    for name, spec in SOURCES.items():
        rows = bank.source == name
        full_tok = bank[bank.rep == spec["full"]].set_index("id").tokens
        bank.loc[rows, "cost"] = bank.loc[rows, "tokens"] / bank.loc[rows, "id"].map(full_tok)
    return bank


def matrix(bank, value, reps=None, complete=True):
    """Items x representations table of `value`; with complete=True, only items that have every rep."""
    m = bank.pivot(index="id", columns="rep", values=value)
    if reps is not None:
        m = m[list(reps)]
    return m.dropna() if complete else m


def load_folds(cfg=None):
    cfg = cfg or load_config()
    return json.loads(repo_path(cfg["final"]["folds"]).read_text())["folds"]


def fold_of(cfg=None):
    return {i: k for k, f in enumerate(load_folds(cfg)) for i in f}


def load_groups(cfg=None):
    """Named item sets: a2_subset, midterm_train, midterm_test, idea3_test, fresh (never inspected)."""
    cfg = cfg or load_config()
    return json.loads(repo_path(cfg["final"]["folds"]).read_text())["groups"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    all_ids = {i for f in load_folds(cfg) for i in f}
    bank = load_bank(cfg)
    if bank.empty:
        print("bank is empty")
        return
    print(f"{len(all_ids)} items in the folds\n")
    cov = bank.groupby(["source", "rep"]).agg(items=("id", "nunique"), acc=("acc", "mean"),
                                              cost=("cost", "mean")).reset_index()
    cov["coverage"] = cov["items"] / len(all_ids)
    print(cov.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
