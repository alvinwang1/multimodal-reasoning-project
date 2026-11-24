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

## Midterm baselines (Ideas 1, 3, 4)

Split: `python -m src.midterm.split` → `data/midterm_split.json` (180 train incl. the Assignment-2 subset /
120 test, balanced by audio type). Same frozen Qwen2.5-Omni-3B and scoring as Analysis A; accuracy is
averaged over 3 option orders. Full plan and rationale: [plan.md](plan.md).

```bash
# Model runs (resumable). Idea 1 = 7 base configs on all 300 items; Idea 3 = token-budget grid on test.
caffeinate -i ./results/midterm/run_chain.sh
# Idea 4: generate surrogates (Whisper ASR + Qwen captions / audio descriptions), then score them
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.midterm.surrogates
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 python -m src.analysis_a.run \
  --split test --configs gen_cap_desc gen_cap_asr img_gendesc img_asr gencap_aud --perm-configs all \
  --surrogates results/midterm/generated_surrogates.json --out results/midterm/raw_surrogate.jsonl
# Analyses -> results/midterm/idea{1,3,4}/
python -m src.midterm.idea1_router      # learned router vs. fixed configs, cascades, oracle
python -m src.midterm.idea3_budget      # image/audio token-budget grid, marginal utility
python -m src.midterm.idea4_surrogates  # gold vs. generated surrogates, incl. generation cost
```

Config names understood by the runner: the 7 base configs, `gen_cap_desc`, `gen_cap_asr`, `img_gendesc`,
`img_asr`, `gencap_aud` (generated surrogates), and `bud_i{r_img}_a{r_aud}` (image downscaled to `r_img` of
the pixel cap; audio time-stretched to `r_aud` of its tokens; 0 drops the modality).

## Final report runs on Modal (Ideas 1, 3, 4)

Every model pass for the final report runs on one NVIDIA L4 (bf16) on [Modal](https://modal.com) (serverless GPUs,
billed per second; about $10 for the whole run, about 8 h). `scripts/modal_run.py` runs `src/final/run_all.py` in an
L4 container, with the code uploaded from your checkout and the results on the Modal Volume `mmml-final`.

```bash
pip install modal && modal setup                         # once: install the client and log in
modal run scripts/modal_run.py --sections preflight      # CPU check: every item loads and builds its prompts
modal run --detach scripts/modal_run.py                  # everything; keeps going if you close the laptop
modal run scripts/modal_run.py --progress                # how far each section has got
modal volume get mmml-final results/final/final_results.tar.gz .   # when it has finished
tar xzf final_results.tar.gz                             # in the repo root: the files land in results/
```

`src/final/run_all.py` runs the same sections and config lists as `colab/run_final.ipynb` (the Colab fallback):
the base bank (6), Idea 3 probe cells (7), Idea 4 generation and scoring (8, 9, 9b length ablation, 9c threshold
sweep), and the rest of the Idea 3 grid (10, 11). It checks that the GPU is an L4, logs every section to
`results/final/logs/`, stops at the first failure, and resumes where it stopped when run again. The last step writes
`results/final/final_results.tar.gz` (the 4 result files and the logs); load them with `src.final.bank.load_bank()`.

Follow-ups for the report:

```bash
python -m src.final.stats                                # CPU: 95% CIs and paired tests -> results/final/stats/
modal run scripts/modal_run.py --sections router         # Idea 1 router study (CPU on Modal, ~10 min)
modal volume get --force mmml-final results/final/idea1_results.tar.gz . && tar xzf idea1_results.tar.gz
modal run scripts/modal_run.py --sections text           # OmniBench's text fields -> results/final/omnibench_text.json
python -m src.final.idea3_allocator                      # Idea 3 allocator + analyses -> results/final/idea3/ (~2 min)
python -m src.final.idea4_analysis                       # Idea 4 analyses + coding sheet -> results/final/idea4/
modal run scripts/modal_run.py --sections protocol --limit 2          # protocol check, 2 items first (~5 min)
modal run --detach scripts/modal_run.py --sections protocol          # all items (~1 h): no image cap, written
                                                                     # answers, original vs shuffled option order
```

`src.final.idea1_router` trains the Idea 1 routers (pointwise and pairwise, several feature sets and menus) on the
score bank with the 5 frozen folds, against fixed inputs, random mixing (the hull), cascades and the oracle, and writes
`results/final/idea1/report.md` with every table and figure. It needs SBERT question embeddings, which it computes
first (sentence-transformers and the OmniBench download, both on the Modal Volume); locally, `--no-question` skips them.
`src.final.stats` bootstraps over items (an item's accuracy is the mean over its option orders) and runs sign-flip
tests for the comparisons the report relies on. `src.final.protocol_check` asks why the full-input accuracy is below
Qwen's published OmniBench score: it writes `results/final/protocol_check.jsonl` and a summary,
`results/final/protocol_check.md` (download both with `modal volume get`, as printed at the end of the run).

## What is (not) committed

Committed: code, configs, `data/subset_ids.json`, `results/analysis_b/assignments.json`, annotation JSONLs,
figures and small tables. Ignored (see `.gitignore`): the dataset and model caches, exported annotation media
(`data/cache/`), raw model outputs (`results/analysis_a/raw*.jsonl`, `smoke*.jsonl`), feature arrays (`*.npz`).
