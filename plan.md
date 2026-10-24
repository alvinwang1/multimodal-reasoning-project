# Midterm Report: Plan

**Deadline: Sunday Oct 25, 2026, 8pm ET.** Team of 4: **3 baselines** (one per research idea), **10 pages** + references; optional appendix (≤5 pages, not graded) + AI-usage/contributions page.
(The Assignment 2 plan is now `plan_assignment2.md`.)

Each research idea needs: an initial implementation built on recent methods, results on a **held-out test set**, **failure-case analysis** (most important), and an updated idea for the final report.

---

## 1. Decisions (made Oct 24)

| Decision | Choice | Why |
|---|---|---|
| Ideas with baselines | **Idea 1** (adaptive routing), **Idea 3** (modality token budgets), **Idea 4** (generated text surrogates) | All inference-only and reuse the existing pipeline. Idea 2 (LoRA audio adaptation) needs training on the Mac and won't finish in time. |
| Data | **300 OmniBench questions**: 180 train / 120 test, balanced 60/60/60 and 40/40/40 by audio type | Train = the 90 Assignment-2 items + 90 new ones. Test = 120 **new** items never inspected before. |
| Model | Qwen2.5-Omni-3B (thinker, bf16, Apple M2 Max), same scoring as Assignment 2 | Comparability with Assignment 2. |
| Evaluation | Accuracy **averaged over 3 option orders** (position bias, see Assignment 2), plus tokens, latency and activation memory | The answer-position bias makes single-order accuracy misleading. |

---

## 2. What's done so far

- [x] **Train/test split.** [src/midterm/split.py](src/midterm/split.py) → [data/midterm_split.json](data/midterm_split.json). Seeded and balanced; train and test don't overlap.
- [x] **Runner extended** ([src/analysis_a/](src/analysis_a/)):
  - `--split train|test|all` runs on the midterm split.
  - **Budget configs** `bud_i{r}_a{r}`. The image is resized to a fraction `r` of the pixel cap (≈ `r` × visual tokens). The audio is **time-stretched** (sped up, pitch preserved) to ≈ `r` × audio tokens, so it keeps the same content in fewer tokens. A ratio of 0 drops the modality. See `config_spec` and `stretched_audio` in [prompts.py](src/analysis_a/prompts.py).
  - **Generated-surrogate configs** (`gen_cap_desc`, `gen_cap_asr`, `img_gendesc`, `img_asr`, `gencap_aud`), loaded with `--surrogates <json>`.
- [x] **Smoke test on one test question.** Budgets scale as intended: image tokens 750 → 357 → 180 → 72, audio 750 → 375 → 187, latency 4.6 s → 0.8 s.
- [x] **Overnight run script.** [results/midterm/run_chain.sh](results/midterm/run_chain.sh) runs Idea 1, then Idea 3.

---

## 3. Commands you run (in order)

All from the project folder, laptop plugged in, lid open. Every step is **resumable**: if it stops, run the same command again.

```bash
cd "/Users/ameet/Desktop/Fall 2026/MMML/Project"
```

### Step A: Idea 1 + Idea 3 model runs (start now, ~4 h)
```bash
caffeinate -i ./results/midterm/run_chain.sh
```
- **Idea 1:** 7 base configs × 3 orders on the 210 new questions → appends to `results/analysis_a/raw.jsonl`. It starts with `1890 done, 4410 to run` and takes ~2 h.
- **Idea 3:** 16 budget combinations × 3 orders on the 120 test questions → `results/midterm/raw_budget.jsonl`. It starts with `0 done, 5760 to run` and takes ~2 h.
- Logs: `results/midterm/run_idea1.log`, `run_idea3.log`. It's finished when you see `=== Idea 3 done`.

### Step B: Idea 4, generate the text substitutes (after Step A, ~45 min)
*(script being written now: `src/midterm/surrogates.py`)*
```bash
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m src.midterm.surrogates
```
For each of the 120 test questions this produces a Whisper transcript, a Qwen-written image caption and a Qwen-written audio description, plus how long each took to generate → `results/midterm/generated_surrogates.json`. The first run downloads Whisper (~1.6 GB).

### Step C: Idea 4, score the generated substitutes (~30 min)
```bash
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1 .venv/bin/python -m src.analysis_a.run \
  --split test --configs gen_cap_desc gen_cap_asr img_gendesc img_asr gencap_aud --perm-configs all \
  --surrogates results/midterm/generated_surrogates.json --out results/midterm/raw_surrogate.jsonl
```

### Step D: analysis + figures (minutes; I can run these)
*(scripts being written now)*
```bash
.venv/bin/python -m src.midterm.idea1_router
.venv/bin/python -m src.midterm.idea3_budget
.venv/bin/python -m src.midterm.idea4_surrogates
```

---

## 4. The three baselines

### Idea 1: Adaptive representation routing
- **Question:** can a cheap router pick, per question, which input configuration to use, at a better accuracy–cost trade-off than any single fixed configuration?
- **Baseline:** a **learned router**. One logistic-regression model per configuration predicts P(correct | features) from signals available after a cheap caption+transcript pass:
  - the model's 4 option probabilities, max probability, margin and entropy
  - the question's SBERT embedding (PCA-reduced)
  - surrogate lengths
  - audio type

  Policy: choose `argmax_c P̂(correct_c | x) − λ·cost(c)`, where cost includes the probe pass. It is trained on the 180 train items, with labels = correctness averaged over orders, and evaluated on the 120 test items. This follows OmniSelect-style relevance routing, but chooses among representation types rather than pruning levels.
- **Compared against:** each fixed config, random routing at matched cost, the confidence cascade (τ tuned on train), and the permutation-robust oracle.
- **Metrics:** accuracy vs. mean cost curve (sweeping λ), and the share of the oracle gap recovered.
- **Failure analysis:** which items the router sends to the wrong config, by audio and task type; whether its errors are low-confidence items; confusion between predicted and oracle configs; qualitative examples.

### Idea 3: Marginal utility of modality tokens under budgets
- **Question:** at a fixed total multimodal token budget, does splitting it unevenly between image and audio beat uniform compression, and does the best split depend on audio type?
- **Baseline:** a resolution- and time-based token budget grid, image keep-ratio ∈ {0, 0.1, 0.25, 0.5, 1} × audio keep-ratio ∈ {0, 0.25, 0.5, 1}. Uniform compression is the diagonal; image-heavy and audio-heavy splits are the off-diagonals at matched total tokens. This is the standard dynamic-resolution baseline used by Qwen-VL models and the uniform baseline in OmniSIFT and OmniSelect.
- **Metrics:** accuracy vs. total tokens per allocation, the best allocation per budget and audio type, and a marginal-utility surface ∂acc/∂tokens per modality.
- **Failure analysis:** where compression breaks (e.g. small text in images, fast speech at 4× speed), items correct at full budget but wrong when compressed, and examples.

### Idea 4: Raw modalities vs. *generated* text surrogates
- **Question:** OmniBench's captions and transcripts are human-written (Assignment 2 caveat). Do *model-generated* substitutes keep the accuracy advantage once their generation cost is counted?
- **Baseline:** describe-then-answer:
  - **Whisper large-v3-turbo** speech transcripts (ASR)
  - Qwen2.5-Omni's own image captions and audio descriptions

  These are scored in 5 configs against the gold-surrogate and raw configs from Idea 1.
- **Metrics:** accuracy (averaged over orders), answer-time tokens, and **total cost including generation** (latency and generated tokens).
- **Failure analysis:** ASR on non-speech audio (hallucinated or empty transcripts), captions missing question-relevant details, and gold-vs-generated gaps per audio type, with paired examples.

---

## 5. Remaining work checklist

- [ ] **You:** Step A tonight (~4 h)
- [ ] **Claude:** `src/midterm/surrogates.py` (Idea 4 generation), smoke-tested on 2 items
- [ ] **Claude:** analysis scripts `idea1_router.py`, `idea3_budget.py`, `idea4_surrogates.py`, tested on partial results
- [ ] **You:** Steps B and C after Step A (~1.25 h)
- [ ] Run Step D, then inspect failure cases (qualitative examples per idea)
- [ ] **Report** (new LaTeX, 10 pages):

| Section | Pages | Source |
|---|---|---|
| Abstract | ¶ | — |
| Introduction (+ overview figure) | 0.5–1 | Assignment 2 intro |
| Related work (12–15 citations, grouped by idea) | ~0.67 | proposal related work |
| Problem statement (math: inputs, representations, cost, objective) | 0.25 | new |
| Experimental methodology (data, split, metrics, option-order averaging) | 0.5 | Assignment 2 setup |
| Per idea × 3: implementation (~1 p), results + failure analysis (~0.75–1 p), updated idea (~0.5 p) | ~6.5–7.5 | new |
| Appendix: implementation details, extra results; AI-usage + contributions page | — | — |

- [ ] **Team:** each person reviews and rewrites the sections they own (keep this consistent with the contributions page)
- [ ] Commit and push (Claude provides the git commands, Arjun runs them). The uncommitted Assignment 2 fixes get included.

---

## 6. Notes and risks
- **Timing:** Steps A–C total ~5.5 h of model runs. Starting by ~3pm Saturday, results land Saturday night, leaving Sunday for analysis and writing.
- **Test set size:** 40 items per audio type means ±~8 points per cell. Report confidence intervals and avoid over-reading small per-type differences.
- **Time-stretching audio** is a crude compression (4× speech may be unintelligible). That's itself a failure case to discuss, and it motivates learned token selection for the final report.
- **Router features use the audio type,** which a deployed system would need to infer. Report an ablation without it.
- **AI usage:** Claude Code is building most of the code again. The AI disclosure must say so, and teammates should lead the analysis and writing of their idea.
