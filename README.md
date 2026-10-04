# Efficient Multimodal Reasoning: Assignment 2 data analyses

CMU 11-777 (Fall 2026) team project: Nikkita Batra, Arjun Shah, Alvin Wang, Lindy Zhang.

Three analyses on [OmniBench](https://huggingface.co/datasets/m-a-p/OmniBench) with
[Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B), all on a shared, fixed-seed
90-example subset (30 speech / 30 sound event / 30 music):

| | Analysis | Code | Outputs |
|---|---|---|---|
| A | Raw modalities vs. text surrogates (7 input configs, option permutations, oracle routing, cascade) | `src/analysis_a/` | `results/analysis_a/`, `results/text_sufficient_labels.json` |
| B | Staged-reveal human annotation (Gradio tool + analysis) | `src/analysis_b/` | `results/analysis_b/` |
| C | Language modality: SBERT + Empath, t-SNE, predicting text-sufficiency | `src/analysis_c/` | `results/analysis_c/` |

All settings (seed, subset size, model, image pixel budget, paths) are in `configs/default.yaml`.
Every script takes `--config` and `--help`. Paths are relative to the repo root.

## Setup

Tested on macOS with an Apple M2 Max (PyTorch MPS backend), Python 3.12. CUDA also works; the device
is picked automatically (cuda > mps > cpu).

```bash
brew install ffmpeg                      # audio decoding
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data                       # prints the schema; writes data/subset_ids.json (committed)
```

The first run downloads OmniBench (~1.3 GB) and Qwen2.5-Omni-3B (~12 GB) into `~/.cache/huggingface`.
Item 1091 is excluded from sampling because it has duplicate answer options.

## Analysis A: raw modalities vs. text surrogates

Configs (question + 4 options are always present; captions/transcripts are inserted as text in place of
the raw modality): `full` (image+audio), `img_trans`, `cap_aud`, `cap_trans`, `img_only`, `aud_only`, `q_only`.
`full` and `cap_trans` are also run under 2 extra fixed-seed option permutations (`--n-perms`).

The model is scored with a single forward pass: softmax over the next-token logits of the four letter tokens
`A`–`D` (single tokens, ids 32–35). Each JSONL row also logs token counts by modality, forward latency
(batch size 1, warmup, device sync) and peak memory. On MPS, torch has no peak-memory counter, so a
background thread polls allocated memory during the forward pass; treat the number as approximate.

```bash
# smoke test: 5 examples x all configs, also checks greedy generation against the logit argmax
python -m src.analysis_a.run --limit 5 --out results/analysis_a/smoke.jsonl --check-generate
python -m src.analysis_a.inspect_results results/analysis_a/smoke.jsonl --rows

# full run (990 forward passes, ~30 min on an M2 Max); resumable, so re-run the same command after an interruption
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 \
  python -m src.analysis_a.run 2>&1 | tee results/analysis_a/run.log

python -m src.analysis_a.analyze          # figures, tables, summary.json, text_sufficient_labels.json
```

Outputs (PDF + PNG): `acc_by_config_audio`, `acc_vs_tokens` (Pareto frontier), `oracle_routing`,
`utility_sweep`, `cascade`; tables in `results/analysis_a/tables/`; `modality_costs.json` (per-example
token costs of each input, used by B).

## Analysis B: staged-reveal human annotation

Each annotator gets 20 items (10 shared by all four + 10 unique; 50 distinct items), seeded and balanced by audio type.
Stage 0 shows the question, options, caption and transcript. The annotator then judges whether text is enough
(plus confidence and an optional answer), then chooses **reveal image / reveal audio / answer now**.
Every step is logged.

```bash
python -m src.analysis_b.assign           # writes results/analysis_b/assignments.json and exports media
python -m src.analysis_b.app              # open http://127.0.0.1:7860, pick your name
python -m src.analysis_b.app --share      # temporary public link so teammates can annotate remotely
python -m src.analysis_b.analyze          # after annotation (needs Analysis A's analyze output)
```

Results are appended to `results/analysis_b/<name>.jsonl` after every item. Restarting resumes at the
first unfinished item; an item interrupted midway restarts from stage 0. The gold answer is never shown.
`analyze` reports sufficiency-prediction accuracy, reveals and token cost per correct answer, breakdowns by
audio and task type, Fleiss' kappa on shared items, and agreement with the model's text-sufficient labels.
It also writes `human_text_sufficient.json` for Analysis C.

## Analysis C: language modality

```bash
python -m src.analysis_c.features         # SBERT (all-mpnet-base-v2) + Empath on 3 text views -> features.npz
python -m src.analysis_c.tsne             # t-SNE by task type / audio type / text-sufficient label
python -m src.analysis_c.analyze          # Empath categories associated with text-sufficiency
python -m src.analysis_c.predict          # AUROC (stratified 5-fold): SBERT LogReg, Empath LogReg,
                                          # Omni max-prob, majority; ROC curves; learning curve
# same with human labels from Analysis B (outputs suffixed _humanlabels, titles marked [human labels]):
python -m src.analysis_c.tsne --labels human
python -m src.analysis_c.analyze --labels human
python -m src.analysis_c.predict --labels human
```

Text views: `question`, `q_cap_trans` (question + caption + transcript), `transcript`.

## What is (not) committed

Committed: code, configs, `data/subset_ids.json`, `results/analysis_b/assignments.json`, annotation JSONLs,
figures and small tables. Ignored (see `.gitignore`): the dataset and model caches, exported annotation media
(`data/cache/`), raw model outputs (`results/analysis_a/raw*.jsonl`, `smoke*.jsonl`), feature arrays (`*.npz`).
