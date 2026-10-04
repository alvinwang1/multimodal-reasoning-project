# Assignment 2 — Data & Modality Analyses: Plan

Deadline: **Sun Oct 4, 2026, 8pm ET**. Team of 4, so we need **3 analyses**:

| | Analysis | Type (per assignment) | Feeds research idea |
|---|---|---|---|
| A | Raw modalities vs. text surrogates (Qwen2.5-Omni-3B, 7 configs) | Idea-specific: data analysis | RI4, RI1 (oracle routing), RI3 (cost) |
| B | Staged-reveal human annotation | Idea-specific: human simulation | RI1 (routing policy), RI4 |
| C | Language modality: SBERT + Empath, predict text-sufficiency | Modality-specific: language | RI1 (cheap router signals) |

Work through this file top to bottom and check boxes as you go. **No full model runs until the smoke test (Step 3) has been reviewed.**

---

## 0. Findings from inspecting the repo and environment

**Repo**
- This folder has `11_777_Preproposal.pdf`, `omnibench_analysis.ipynb` and `omnibench_analysis.pdf`. There is no code package yet.
- ⚠️ **Git:** this folder isn't its own repo. `git rev-parse --show-toplevel` returns `/Users/ameet` (your home directory), and the remote is `github.com/Arjun-254/Synapse_LP`.
- **Team repo:** `https://github.com/alvinwang1/multimodal-reasoning-project` (currently empty). Claude writes the code; **Arjun runs all git/GitHub commands**. Claude supplies the exact commands at commit time (Step 6).

**Dataset schema** (`m-a-p/OmniBench`, confirmed in the notebook; `data.py` will re-print it)
- One `train` split with 1,142 rows. HF parquet is about 1.26 GB and is already cached in `~/.cache/huggingface/hub`.
- Fields: `index`, `task type`, `audio type`, `question`, `options` (list of 4 strings), `answer` (**the answer string, not a letter**; it must be mapped to an index), `audio content` (human transcript), `image content` (VLM caption), `audio_path`, `image_path`, `audio`, `image`.
- Audio types: speech 771, sound event 265, music 106. Every category is large enough for a balanced subset.
- 8 task types, with a long tail: Identity 32, Text/Symbols 25, Count 15. Task-type breakdowns on a small subset will be noisy, so we report them as descriptive only.
- **1 item has duplicate options.** We exclude it from the subset, because the answer-to-letter mapping is ambiguous for that item.
- Earlier finding (preproposal): the question-only lexical heuristic scores 26.1% against a 25% chance rate, so no text shortcut.

**Hardware: run everything locally on the M2 Max (no CUDA)**
- Local machine: Apple M2 Max, 64 GB unified memory, no NVIDIA GPU, torch not yet installed.
- Local backend options for Qwen2.5-Omni-3B (researched 2026-10-03):

| Backend | Qwen2.5-Omni-3B support | Full 4-letter logits | Token counts by modality | Notes |
|---|---|---|---|---|
| **PyTorch + HF transformers on MPS** ✅ chosen | Official weights, bf16/fp16 | Yes (direct forward pass) | Yes (count `<\|IMAGE\|>`/`<\|AUDIO\|>` ids) | Same model and numerics family as the CUDA reference; code is device-agnostic. About 7 GB of weights in 64 GB. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops. |
| MLX: `mlx-vlm` 0.7.4 | ❌ No `qwen2_5_omni` (it has `qwen3_omni_moe`, which is a different, 30B model) | Generation API; logprobs via server | Not directly | Would mean switching models away from the proposal. |
| MLX: community `mlx-lm-omni` (`giangndm/qwen2.5-omni-3b-mlx-8bit`) | Unofficial port, 4/8-bit only | Possible (raw MLX forward) | Possible | One-person project, quantized; fidelity to the original model is unverified. |
| llama.cpp (`ggml-org/Qwen2.5-Omni-3B-GGUF`, Metal) | Yes, but audio input is labeled "highly experimental" | Only top-k logprobs via the server | Total only | Fastest option, but quantized and experimental audio make it a weak basis for a report. |

- **Decision:** HF transformers on MPS as the primary backend. Fallback if MPS is too slow or breaks on an op: `mlx-lm-omni` 8-bit, with a sanity check that its predictions match MPS on the smoke examples.
- **Precision:** try bf16 on MPS (torch ≥ 2.3). Fall back to fp16, then fp32 (about 14 GB, still fits) if outputs are NaN or garbage. Batch size is 1.
- **Latency on MPS:** `torch.mps.synchronize()` before t0 and after the forward pass, plus warmup passes. The MPS shader cache makes the first passes slow.
- **Peak memory on MPS:** torch.mps has no `reset_peak_memory_stats` / `max_memory_allocated`. Instead, a background thread polls `torch.mps.current_allocated_memory()` every ~2 ms during the forward pass, and we record the max minus the pre-forward baseline (= activation peak) along with `driver_allocated_memory()`. This is documented as an approximation in the report.
- The runner keeps a CUDA code path (exact peak stats), so numbers can be re-collected on a GPU later if needed. All records log `device`.
- **Feasibility gate (Step 3a):** load the thinker on MPS, run 1 example × 7 configs, and time it. If a full-config forward takes longer than ~10 s, reconsider the subset size or the fallback backend.

---

## 1. Decisions needed before / while coding

1. **Subset size. Recommendation: 90 examples (30/30/30), not ~50.**
   Analysis B needs 10 shared + 4×20 unique = **90 distinct items**, and the spec says B draws from the same subset. A 50-item subset can't supply that. 90 also helps Analysis C: 5-fold CV on 50 items with a possibly rare positive class is very fragile.
   Cost: 7 configs + 2 configs × 2 extra permutations = 11 forward passes per example, so about 1,000 passes total. It's a prefill-only forward (no generation), so on the M2 Max via MPS expect roughly 1–5 s/pass, about 0.5–1.5 h total. Confirm at the feasibility gate. `--n` stays a CLI flag.
2. ~~Which CUDA machine?~~ **Resolved:** run locally on the M2 Max with PyTorch MPS (see Hardware).
3. ~~Git~~ **Resolved:** the target is the empty team repo. Claude runs no git commands, and the commands are handed to Arjun at the end. They must not commit into the home-directory repo.
4. **Image resolution:** Qwen2.5-Omni uses dynamic resolution, so image token count depends on image size. Recommendation: fix `max_pixels` (e.g. 768×28×28) so image cost is controlled and documented. Alternatively, leave the default and report the variance.
5. **TA sign-off:** the assignment says to check analyses with the primary TA first. Send them the table at the top of this file.

---

## 2. File structure

```
Project/                         (team repo root)
├── plan.md
├── README.md                    # one section per analysis: how to run
├── requirements.txt             # pinned
├── .gitignore                   # data/cache, results/**/raw, *.jsonl outputs, model weights
├── configs/
│   └── default.yaml             # paths, model id, seed, subset size, max_pixels, etc.
├── data/
│   └── subset_ids.json          # committed (small) — the shared subset
├── src/
│   ├── __init__.py
│   ├── common.py                # load config, seeding, path helpers, plot style (3.25in, PDF+PNG)
│   ├── data.py                  # load OmniBench, print schema, draw balanced subset
│   ├── analysis_a/
│   │   ├── __init__.py
│   │   ├── prompts.py           # build the 7 configs' chat messages; option permutation
│   │   ├── model.py             # load Qwen2.5-Omni thinker, letter-token ids, scoring fn
│   │   ├── run.py               # resumable runner → results/analysis_a/raw.jsonl
│   │   └── analyze.py           # tables + plots + text_sufficient_labels.json
│   ├── analysis_b/
│   │   ├── __init__.py
│   │   ├── assign.py            # seeded 10 shared + 10 unique per annotator (changed from 20 to cut annotation time)
│   │   ├── app.py               # Gradio staged-reveal tool
│   │   └── analyze.py
│   └── analysis_c/
│       ├── __init__.py
│       ├── features.py          # SBERT embeddings + Empath features → results/analysis_c/*.npz
│       ├── tsne.py
│       ├── predict.py           # 4 predictors, stratified 5-fold, ROC, learning curve
│       └── analyze.py           # Empath category association
└── results/
    ├── text_sufficient_labels.json
    ├── analysis_a/  (raw.jsonl ignored; figures + tables committed)
    ├── analysis_b/  (<name>.jsonl, assignments.json, figures)
    └── analysis_c/
```

---

## 3. Steps

### Step 1 — Setup
- [x] Add `.gitignore` covering: `data/cache/`, `results/analysis_a/raw*.jsonl`, `*.npz`, `*.safetensors`, `.DS_Store`, `__pycache__/`, `.venv/`.
- [x] `requirements.txt`: check the Qwen2.5-Omni-3B model card for the minimum transformers version (expected ≥ 4.52, which adds `Qwen2_5OmniForConditionalGeneration` / `Qwen2_5OmniThinkerForConditionalGeneration` / `Qwen2_5OmniProcessor`) and pin `qwen-omni-utils`. Also pin: torch, datasets, soundfile/librosa, accelerate, gradio, sentence-transformers, empath, scikit-learn, statsmodels (Fleiss' κ), matplotlib, pandas, pyyaml.
- [x] `configs/default.yaml` + `src/common.py`: seed everything (`random`, `numpy`, `torch`, `PYTHONHASHSEED`). Paths are relative to the repo root.

### Step 2 — `src/data.py`
- [x] Load `m-a-p/OmniBench` and print the schema plus one example (no assumed field names: assert the expected set, as in the notebook).
- [x] Drop the duplicate-option item. Map `answer` string → `answer_idx`, and assert it's found exactly once.
- [x] Balanced subset with seed 42 across `audio type`, n split evenly. If a category is short, take all of it and log that. Save `data/subset_ids.json` with `{seed, n, per_type_counts, ids:[index,...]}`.
- [x] Helper `get_example(idx)` → dict with PIL image, 16 kHz mono waveform, text fields, `answer_idx`.

### Step 3 — Analysis A runner + smoke test  ✅ done 2026-10-03, awaiting review
- [x] **3a Feasibility gate:** venv + install, then load `Qwen/Qwen2.5-Omni-3B` thinker on MPS and run 1 example × 7 configs. Report dtype, which ops fell back, s/pass and memory. Decide go / fallback.
- [x] **Model loading:** use `Qwen2_5OmniThinkerForConditionalGeneration` (text-only and smaller, with no talker/token2wav loaded). If the full model class is used instead, call `disable_talker()` and pass `return_audio=False`. dtype from config (bf16 → fp16 → fp32 fallback), `attn_implementation="sdpa"`, device auto: cuda > mps > cpu.
- [x] **Prompts** (one template for all configs; question + 4 options are always present):
  - full: `[image][audio] Question … A. … B. … C. … D. … Answer with the option's letter from the given choices directly.`
  - surrogates are inserted as text: `Image description: <caption>` / `Audio transcript: <transcript>` take the slot of the raw modality.
  - configs: `full, img+trans, cap+aud, cap+trans, img_only, aud_only, q_only`.
- [x] **Scoring:** one forward pass, no generation. Take the logits at the last position (immediately after `<|im_start|>assistant\n`), then softmax over the 4 letter token ids only → `probs[A..D]`. `pred = argmax`.
- [x] **Letter-token check:** print the tokenization of `"A"`, `" A"`, `"A."` etc. Assert that each of A–D is a single token. Also check, on the smoke examples, that greedy `generate(max_new_tokens=3)` starts with the same letter as the argmax. Log the agreement rate.
- [x] **Token counts by modality:** after the processor, count `input_ids == image_token_id` → `n_image_tok` and `== audio_token_id` → `n_audio_tok`; the rest is `n_text_tok` (template + question + options + any caption/transcript). Also log `n_surrogate_tok` separately, for Analysis B step costs.
- [x] **Timing:** 3 warmup passes at startup. Per example: device-specific `synchronize` → t0 → forward → `synchronize` → t1. Peak memory comes from `torch.cuda` peak stats on CUDA, or from the polling thread on MPS (see Hardware). Latency covers the model forward only. Processor time is logged separately as `preproc_ms`.
- [x] **Permutations:** `--n-perms 3`. Perm 0 is the original order; perms 1..k use `np.random.default_rng(seed + p)` for each example. They're applied to configs `full` and `cap+trans` only (configurable). Probabilities are mapped back to the original option order before logging.
- [x] **JSONL record:** `{id, config, perm, perm_order, audio_type, task_type, probs(orig order), pred_idx, answer_idx, correct, n_text_tok, n_image_tok, n_audio_tok, n_surrogate_tok, n_total_tok, latency_ms, preproc_ms, peak_mem_mb, device, model, timestamp}`.
- [x] **Resumable:** on start, read existing JSONL, build a set of `(id, config, perm)`, and skip those. Append and flush after every record. `tqdm` plus a periodic log line with throughput and ETA.
- [x] **Smoke test:** `python -m src.analysis_a.run --limit 5 --out results/analysis_a/smoke.jsonl` with all configs and perms. Show the user: a per-row table (config, pred, gold, probs, token counts, latency, memory), the letter-token check and generate-vs-argmax agreement.

**Smoke results (2026-10-03):** M2 Max, MPS, bf16. All thinker weights loaded (0 missing). Letters A–D are single tokens (32–35). The top vocab token was a letter in 100% of passes, and greedy generation matched the argmax letter in 100%. Forward time: about 3 s with an image, 0.3–1.6 s without. Peak memory about 9.1–9.5 GB, of which about 9.05 GB is weights. Full run estimate: 990 passes at about 2 s, roughly 35 min (without `--check-generate`).

### Step 4 — Full run (user launches) ∥ build B and C  ✅ code done; full run in progress (launched 20:44)
- [x] User launches `python -m src.analysis_a.run` locally (plug the laptop in and use `caffeinate -i`, since it's resumable if interrupted). Analysis C feature extraction can run at the same time on the CPU.
- [x] **B: `assign.py`.** Seed 42 over the subset: 10 shared, then 20 unique per annotator (nikkita, arjun, alvin, lindy) → `results/analysis_b/assignments.json`. Balance across audio types where possible. Shuffle the order per annotator.
- [x] **B: `app.py` (Gradio).** Run with `--annotator <name>`, served locally.
  - Stage 0: question, options, caption, transcript. Inputs: *Is text enough?* (yes/no), confidence 1–5, optional answer.
  - Then a loop over the actions **Reveal image / Reveal audio / Answer now**. Every action is appended to `steps` with a timestamp.
  - Answer screen: final answer, confidence 1–5, free-text note on which input decided it.
  - Each completed item is appended to `results/analysis_b/<name>.jsonl` and flushed immediately. On launch, completed ids are skipped (resumable). The gold answer is never shown.
  - Record: `{id, annotator, stage0:{text_enough, conf, answer}, steps:[{action, t}], final:{answer_idx, conf, note}, revealed:{image, audio}, duration_s}`.
- [x] **C: `features.py`.** Three text views per subset item: `q` = question + options, `q+cap+trans`, `trans`. Embed each with `sentence-transformers/all-mpnet-base-v2` (normalized). Compute Empath features (normalized category counts, all ~194 categories) on the same views. Cache to `.npz`. Runs on the Mac's CPU/MPS.
- [x] **C: `tsne.py`.** For each view, t-SNE (perplexity ≈ 15 given n≈90, seed fixed) colored by task type, audio type and text-sufficient label (the last one once A finishes).

### Step 5 — Analysis scripts  ✅ written and tested on partial/synthetic data; rerun on the final results
**A: `analyze.py`** → `results/analysis_a/*.pdf|png` + `tables/*.csv|tex`
- [x] Accuracy by config × audio type as a grouped bar chart, plus a table with n and counts. Perm-0 rows are used for configs without permutations. For `full` and `cap+trans`, report both perm-0 and mean-over-perms.
- [x] Accuracy vs. mean total input tokens, one point per config, with the Pareto frontier drawn as a step line.
- [x] **Oracle routing:** cost(config, ex) = n_total_tok(config, ex) / n_total_tok(full, ex). For each example, take the cheapest correct config, falling back to `full` (counted wrong) if none is correct. Plot a histogram of the chosen config. Report oracle accuracy and mean cost against full.
- [x] **Text-sufficient label:** `cap+trans` correct under **all** perms → `results/text_sufficient_labels.json` (`{id: 0/1}`, plus metadata on n_perms and the base rate).
- [x] **Utility sweep:** U(config) = acc − λ·mean_cost for λ ∈ [0, 1.5]. Plot the band of the argmax config along λ and the U curves.
- [x] **Cascade:** start with `cap+trans` (perm 0). If max prob < τ, escalate to `full` (cost = cap+trans cost + full cost, since both were run). Sweep τ ∈ [0.25, 1] and plot the accuracy–cost curve with fixed configs and the oracle as points.
- [x] Mean token counts per modality → `results/analysis_a/modality_costs.json` (used by B).

**B: `analyze.py`**
- [x] Sufficiency-prediction accuracy: stage-0 "text enough?" against ground truth. The ground truth for each item is whether that annotator's stage-0 answer was correct. Also compare against the model label (below).
- [x] Steps and cost per correct answer. Step costs come from `modality_costs.json`: stage 0 = text + surrogate tokens, +image tokens if the image is revealed, +audio tokens if the audio is revealed.
- [x] Breakdowns by audio type and task type. Distribution of first reveal (image first vs. audio first) by audio type.
- [x] Fleiss' κ on the 10 shared items for (a) text-enough yes/no and (b) final answer.
- [x] Human text-enough vs. model `text_sufficient` label on the overlapping items: confusion matrix + Cohen's κ.

**C: `predict.py` / `analyze.py`**
- [x] Empath: point-biserial correlation, or a Mann-Whitney test with FDR, between each category and the text-sufficient label. Show the top ±10 categories in a bar chart.
- [x] Predictors of text-sufficient, using stratified 5-fold CV (seed 42) with AUROC mean ± std:
  (a) LogReg on SBERT embeddings (one per view, C tuned with an inner CV or fixed with a note)
  (b) LogReg on Empath features (standardized)
  (c) max option prob under `cap+trans` perm 0, used directly as the score (no training)
  (d) majority class (AUROC = 0.5, reported for reference)
- [x] ROC curves: the mean ROC across folds with a std band, one panel.
- [x] Learning curve for (a): for each outer fold, subsample the training split to {10, 20, …, 100}% stratified, then evaluate on the fold's test split. Plot mean ± std.
- [x] `--labels {model,human}`. `human` uses B's majority stage-0 "text enough?" label. Every output filename and plot title is suffixed `[human labels]`.
- [x] ⚠️ If the positive class has fewer than 5 examples, stratified 5-fold breaks. Detect this and fall back to repeated stratified k with smaller k, with a printed warning.

### Step 6 — README + report hand-off
- [x] README section per analysis: install, exact commands, outputs.
- [x] Plot style: figure width 3.25 in, font size 7–8 pt, PDF (vector) + PNG at 300 dpi, colorblind-safe palette, consistent config colors across all figures.
- [ ] Claude gives Arjun the git commands to run: a standalone `git init` in this folder (so the home-directory repo is bypassed), add the `alvinwang1/multimodal-reasoning-project` remote, commit the code, subset ids, figures and small tables, then push. Do not commit the raw JSONL, dataset or weights. Claude does not run these.

---

### Step 7 — Remaining
- [x] A finishes (990 records, 31 min, 21:15), then rerun `analysis_a.analyze` → `analysis_c.tsne` / `analyze` / `predict` on the real labels
- [x] Team annotates in the B tool (20 items each: 10 shared + 10 unique), then `analysis_b.analyze`, then C with `--labels human`
- [x] Option-permutation rerun for all 7 configs (900 passes, 24 min); A re-analyzed with permutation-averaged accuracy
- [x] Report draft: `report/main.tex` (all numbers filled), `report/figures/`, `report/references_additions.bib`
- [ ] Team: revise draft, complete AI-usage + contributions appendix, compile in Overleaf, check page count
- [ ] Git commands handed to Arjun

## 4. Risks / notes
- **Cost metric is token-based.** Caption+transcript can sometimes have *more* text tokens than an image has image tokens; this is a real finding, not a bug. We report latency alongside tokens so cost is not judged on tokens alone.
- **Audio-only & question-only configs** check for priors and shortcuts. If `q_only` is far above 25%, flag it in the discussion.
- **Thinker vs. full model:** the logits must come from the same model the team will use later. Confirm that the thinker's outputs match the full model on the smoke examples.
- **Transcripts for sound/music:** `audio content` for non-speech audio may be a description rather than a transcript. Check this in the smoke test, and note it in the report because it changes what "text surrogate" means per audio type.
- **Small n:** at about 30 items per audio type, a ±1-item change moves accuracy by 3.3 points. Show 95% Wilson CIs on the bar charts.

---

## 5. Report mapping (ICML, 5 pages + refs + AI-disclosure appendix)
- Intro (0.5 page): why these three analyses, tied to RI1/RI3/RI4.
- A, B, C: each with its own setup → results → **discussion** subsection (required by the assignment).
- Discussion (1 page): how the oracle gap, cascade, human reveal policy and text-sufficiency predictability update RI1 (router targets and signals), RI4 (when surrogates suffice) and RI3 (which modality to spend on).
- Appendix: AI/LLM usage disclosure + per-teammate contributions.
