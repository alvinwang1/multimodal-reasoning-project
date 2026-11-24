# Final Report + Poster: Plan

**Report due: Sunday Dec 6, 2026, 8pm ET.** Team of 4, so the report covers **3 research ideas** (1, 3, 4) in **10 pages** plus references. An optional appendix of up to 5 pages is allowed but not graded, plus the AI-disclosure and contributions page.

**Poster:** 30″×40″, one idea, 2–3 key results, a 30-second pitch plus a 4-minute talk. All four of us present, in rotation. *Session date: TBD (fill in).*

Earlier plans: `plan_midterm.md`, `plan_assignment2.md`.

---

## Where we are (as of Sun Nov 22)

> For a full explanation of how each part was done (data, pipeline, every idea's implementation, results, weaknesses, likely TA comments, file map), see **[midterm_explained.md](midterm_explained.md)**.

**Done so far:**
- **Assignment 2 (Oct 4).** Data and modality analyses on 90 OmniBench items with Qwen2.5-Omni-3B.
  - Strong answer-position bias: 49% of answers are "D", and the model prefers "B". Because of this, all later results average over 3 option orders.
  - Caption+transcript matched the full raw input while using 39% of its tokens.
- **Midterm report (submitted Oct 25).** One baseline per idea, evaluated on held-out items. The report is in `report_midterm/main.tex`.
  - **Idea 1 (routing).** A per-item oracle reaches 69.2% at cost 0.33. The best fixed input is image+transcript (54.2% at cost 0.78); the full input gets 45.3%. Logistic-regression routers score 44–46% and stay below the fixed-input hull, except for up to +5.9 points around cost 0.2. Why it fails:
    - its features predict item difficulty, not which modality helps;
    - it is overconfident (it predicts 0.67 but is right 43% of the time);
    - its labels are noisy;
    - it has only 180 training items.
  - **Idea 3 (budgets).** About 64 visual tokens recover most of the image's value (47.0% vs 50.0% for the full image, not significant). Cutting the audio is safe, but speeding it up breaks speech: accuracy falls 37.5 → 21.7 → 12.5. Image and audio interact strongly on sound events (+1.28), where the image alone causes confident errors. Per-audio-type lookup tables overfit (37.4% vs 47.0% for a single global cell).
  - **Idea 4 (generated surrogates).** Generated surrogates lose 7.5 points to the reference ones, mostly on music (−21.7), and cost about 25× more time than the raw route. Whisper returns empty, filler or looping text on 52 of 80 non-speech clips. The human audio descriptions often name the answer.
- **Compute used so far:**
  - Idea 1: 6,300 passes (M2 Max);
  - Idea 3: the 10,800-pass budget grid (V100) and the 5,760-pass time-stretch ablation (M2 Max);
  - Idea 4: 4,320 passes (L4), plus a 1,800-pass replication (M2 Max).
- **Code fixes made along the way:** a PyAV fallback for one bad audio file, and an end-of-turn stop token for generation.
- **Repo (Nov 22).** All midterm work is committed. `midterm-report`, `idea3` and `idea4` are merged into `main` and pushed. Everyone now starts from `main`.
- **Final plan drafted** (this file). Estimated effort: about 95–140 person-hours in total, roughly 25–35 per person.

**Pending:**
- **TA feedback on the midterm** arrives Mon Nov 23. Add it to §6, then adjust §3.
- The team needs to agree on the data protocol (§1).
- Then start the score bank (§2).

---

**What the final report must add over the midterm.** Each idea needs:
- a **new approach** beyond the midterm baseline, compared directly against that baseline;
- stated **research questions**, with **ablations** of each new component;
- a **hyperparameter sweep**;
- **statistical tests**;
- **success and failure cases**;
- a **model description and learning** section with the loss function and, where possible, its gradient.

---

## 0. Before anything else (this weekend)

- [x] **Commit the midterm work and merge the branches** (done Nov 22).
  - The midterm work is committed on `midterm-report`.
  - `midterm-report`, `idea4` and `idea3` are merged into `main` and pushed. The only conflict was in `.gitignore`, resolved by keeping both sides.
  - From here on, everyone branches off `main` and commits often: graders use the repo history to judge how evenly work was shared.
- [ ] **Midterm feedback:** the TA gives it on **Mon Nov 23**. Paste it into §6 below; the final report must address it.
- [x] **Data protocol in §1 agreed** (Nov 22).
  - Folds are frozen in `data/final_folds.json`.
  - The shared loader is `src/final/bank.py`.
  - Both runners now take `--split final`.
- [ ] Teammates: `git pull` on `main`.

---

## 1. Data protocol (agreed Nov 22)

**Status:**
- **Folds.** `python -m src.final.folds` wrote `data/final_folds.json`: 1,141 items in 5 folds, each with about 154 speech, 53 sound event and 21 music items (seed 2026).
  - The folds are **frozen**: the script refuses to overwrite them.
  - The file also lists named groups: `a2_subset`, `midterm_train`, `midterm_test`, `idea3_test`, and `fresh`.
  - `fresh` is the 787 items nobody has scored or inspected: 643 speech, 142 sound event, 2 music.
- **Shared loader.** `src/final/bank.py`. `load_bank()` returns one row per item and representation, with:
  - `acc`, `logp`, `cost` and `latency_s`, averaged over the 3 option orders;
  - the source: `base` (L4), `idea3` (L4), or `base_mac` (the midterm Mac runs, for comparison).
  - Helpers: `matrix()`, `load_folds()`, `fold_of()`, `load_groups()`.
  - Checked: it reproduces the midterm Idea 1 table exactly on the 120 test items.

**The constraint.** OmniBench has 1,141 usable items: 771 speech, 265 sound event and 105 music. We have already used 300 of them (100 per audio type), and only **5 music items** have never been touched. A new balanced held-out test set is therefore not possible. The midterm test sets are also small: 120 items for Ideas 1 and 4, and a different 90 items for Idea 3. That makes our confidence intervals wide.

**Recommended protocol:**
1. **Score all 1,141 items** with one shared set of configurations: the "score bank" in §2. This gives 3.8× more data for the learned policies (Ideas 1 and 3) and puts every idea on the same items.
2. **Fixed folds for the learned methods.** Use stratified 5-fold cross-validation over all 1,141 items, with folds saved once to `data/final_folds.json`. Every learned method trains on 4 folds and predicts the held-out fold. We report results pooled over the held-out folds, so every item gets exactly one out-of-sample prediction. Hyperparameters are tuned inside the training folds, never on the held-out fold.
3. **Two reporting views:**
   - **micro** accuracy over all items;
   - **macro** accuracy, the mean of the three per-audio-type accuracies. This matches the midterm's balanced sets and stops speech, which is 68% of the data, from dominating.
4. **A direct comparison with the midterm on its own test set.** Retrain the midterm baseline and the new method on all items *except* the midterm test items, then score both on that test set (the 120 items for Ideas 1 and 4, the 90 items for Idea 3). This gives a like-for-like table: "midterm number → new number".
5. **Keep the midterm scoring rules unchanged:** accuracy averaged over 3 option orders, `log p(gold)` for utility, cost as input tokens relative to the full input, and wall time. Use paired bootstrap confidence intervals and Wilcoxon or sign-flip tests.

---

## 2. Shared score bank (run first): all on one L4 (Modal)

**Compute changed on Nov 22.** PSC is offline, so we now have Arjun's **M2 Max** and his **Colab Pro** credits: an L4, and maybe an A100.

**And again on Nov 23:** Colab could not allocate an L4 (it fell back to a T4), and AWS had no L4 capacity either, so the runs moved to an **L4 on Modal** (serverless GPUs, billed per second). The Colab notebook stays as a fallback.

**Decision: every final scoring run uses one NVIDIA L4 (bf16), on Modal, from one command: `modal run --detach scripts/modal_run.py`, which runs `python -m src.final.run_all` in an L4 container (all 3 ideas; the same sections and config lists as `colab/run_final.ipynb`).**
- The L4 is **6.6× faster** than the Mac: 5.1 s vs 33.6 s per item for the 7 base configs × 3 orders.
- It agrees with the Mac on **97.9%** of predictions (2,520 shared passes; accuracies within 1.4 points).
- Idea 4 already ran on an L4 at the midterm.
- With everything on one GPU type, cost and latency are comparable across Ideas 1, 3 and 4. This removes the midterm's mixed-hardware limitation.

**What each machine is for:**
- **Mac:** development, analysis and small checks. The midterm Mac runs stay in the loader as the `base_mac` source, for comparison only.
- **A100:** only as a fallback, e.g. for the optional FastV comparison, because it uses compute units much faster.

| Job | Section | Passes | L4 time (est.) |
|---|---|---|---|
| 7 base configs × 3 orders × 1,141 items (rerunning the 300 midterm items too, for one GPU type) | 6 | 23,961 | ~1.6 h |
| Idea 3 probe cells `(64,½)`, `(64,0)`, `(0,½)`, which are also Idea 1 features | 7 | 10,269 | ~0.7 h |
| Idea 3: the other 9 grid cells | 10 | 30,807 | ~2 h |
| Idea 3: audio-selection cells, centered (`c`) and loudest window (`e`) | 11 | 20,538 | ~1 h |
| Idea 4: generate the text (Whisper, AST tags, Qwen generic and question-conditioned descriptions) for 242 items, then score 11 configs | 8–9 | 7,986 | ~1.5 h |
| Idea 4: generation-length ablation, 6 configs at 80 and 120 tokens (cut from the 256-token generation) | 9b | 8,712 | ~0.5 h |
| Idea 4: speech-threshold sweep inputs, the 4 routing branches on every item | 9c | 2,904 | ~0.2 h |
| **Total** | | | **≈ 7 L4-hours** |

**Grid corners and text cells are not re-run.** The Idea 3 corners (`v768_a1`, `v768_a0`, `v0_a1`, `v0_a0`) and text cells (`v768_atrans`, `vcap_a1`, `vcap_atrans`) are exactly the same inputs as the base configs. Map between the names with `src/idea3/cells.py::A_CONFIG_CELLS`.

**How to run (Modal):** the commands are in the README, "Final report runs on Modal".
- `pip install modal` and `modal setup` once; then `modal run scripts/modal_run.py --sections preflight` (CPU check) and `modal run --detach scripts/modal_run.py` (everything). With `--detach` it keeps running after the laptop is closed.
- It uploads the local `src/`, `configs/` and `data/`, refuses to start unless the GPU is an L4, logs every section to `results/final/logs/`, and records the commit and GPU of every run in `results/final/logs/runs.log`.
- Results, logs and the model/dataset downloads are on the Modal Volume `mmml-final`. It stops at the first failed section; every section is resumable: run the same command again.
- The last step writes `results/final/final_results.tar.gz` (the 4 result files and the logs). Download it with `modal volume get mmml-final results/final/final_results.tar.gz .`, unpack it in the repo root, then call `load_bank()`.

**Colab fallback (`colab/run_final.ipynb`):**
- Open it in Colab, set Runtime → L4, and run the cells top to bottom.
- Outputs go straight to **`MyDrive/mmml_final/`** through symlinks, so a disconnect loses nothing. Every step is resumable: if the session dies, reconnect and rerun from the top.
- **Colab Pro has no background execution,** so keep the tab open.
- **Disconnect safety:**
  - every log is copied to `MyDrive/mmml_final/logs/`;
  - timestamped copies of every results file go to `backups/` at the start of each session and every 15 min (the last 3 are kept);
  - a line cut off by a disconnect is skipped on resume and that pass is re-run (`src/common.py`);
  - optionally, `CACHE_ON_DRIVE = True` keeps the model and dataset downloads on Drive (about 15 GB).
- To analyse locally, download the jsonl files from Drive into the matching repo paths:
  - `results/final/bank_base_l4.jsonl`
  - `results/idea3/raw_final.jsonl`

  Then call `load_bank()`.

**Cost.** On Modal an L4 is about $0.80/h plus CPU and memory, about $10 for the whole run, which the Starter plan's $30 of monthly credits covers. On the Colab fallback, Colab Pro gives about 100 units a month. An L4 costs roughly 2 units/h and an A100 roughly 10+ units/h; check the exact rates in Colab's Resources panel. About 12 L4-hours is therefore about 25 units, leaving plenty of margin for reruns.

**Who launches what.**
- **Lindy runs it on Modal on behalf of the team** (the same sections as the notebook).
- The last step writes `results/final/final_results.tar.gz`; Lindy shares it with the team, and everyone unpacks it in the repo root.
- Alvin and Nikkita review and commit any changes to their sections (§10–11 and §8–9) *before* the run starts. Modal runs the code in Lindy's checkout at launch, and `results/final/logs/runs.log` records its commit.

**`results/final/run_bank_base.sh` (the Mac script) is now only a fallback,** in case Colab is unavailable.

---

## 3. Research ideas: new approach, ablations, sweeps

### 3.1 Idea 1: Adaptive representation routing (owners: **Arjun + Lindy**)

**Midterm baseline:** pointwise logistic regression for each representation, choosing `argmax p̂_r − λC_r`. It never beat the fixed-representation hull except at cost ≈ 0.2. The diagnosis was that it learned item difficulty, not which modality helps; it was miscalibrated; its labels were noisy; and 180 training items were too few.

**New approach: a pairwise ranking router with disagreement features.**
- **Model.** A score `s(x,r) = w_rᵀ φ(x) + b_r`, linear per representation, with a small MLP as a variant. It is trained with the cost-sensitive pairwise logistic loss from the midterm's "updated idea":
  - For each pair (r, r′) where r beats r′ on item i, the loss is w_{i,r,r′} · log(1 + exp[s(x_i,r′) − s(x_i,r)]).
  - The pair weight w is the gap in `log p(gold)`; pairs with no gap are dropped.
  - Gradient for the report: ∂ℓ/∂s(x,r) = −w · σ(s(x,r′) − s(x,r)), with the opposite sign for r′.
- **Policy.** `argmax_r [s(x,r) − λ C_r(x)]`. The score `s` is ordinal, not a probability, so λ is set per fold by matching a cost target.
- **New features:**
  - **disagreement** between the cheap image-only and audio-only passes: KL(p_v‖p_a), whether their top answers agree, and each one's maximum probability;
  - the probe's distribution, **temperature-scaled**, with the temperature fitted on the training folds.
- **Bigger menu.** Add the budget cells from Idea 3, e.g. `(64,1)` and `(64,½)`, so the router can pick a small image instead of either no image or the full image.

**Research questions:**
- RQ1.1: Does pairwise training beat pointwise training at matched cost?
- RQ1.2: Do disagreement features help, beyond the probe and the question embedding?
- RQ1.3: Does adding budget cells to the menu move the accuracy–cost frontier?
- RQ1.4: How much of the midterm failure was simply too little data? Answer this with a learning curve over training size, from 180 items to all four training folds.

**Ablations:**
- pointwise vs pairwise loss;
- ± disagreement features;
- ± temperature scaling;
- 7-representation menu vs 7 + budget cells;
- probe vs no probe.

**Sweeps:** λ (the full accuracy–cost curve), L2 strength, training-set size, and the pair-weighting scheme (gap vs uniform).

**Metrics:**
- the midterm's headline: **gain over the hull at matched cost**;
- **area under the accuracy–cost curve** between cost 0.2 and 0.8;
- **oracle gap closed**;
- per-item **Spearman correlation** between predicted and true ranking.

Report all of them with bootstrap confidence intervals over items.

**Optional extra if time allows:** a contextual-bandit version (ε-greedy or LinUCB), which only sees the outcome of the representation it picked. Only do this if the main results are finished by Nov 30.

### 3.2 Idea 3: Marginal utility of modality budgets (owner: **Alvin**)

> **Implemented Nov 23 (first draft, for Alvin to review):** `src/idea3/cells.py` now supports `a<frac>e`, the loudest contiguous window (maximum signal energy, 10 ms steps), next to `a<frac>c` (centered) and plain `a<frac>` (the first part of the clip). Cutting the audio now counts as preprocessing time.

**Midterm baseline:** a lookup table with one budget cell per audio type, chosen on dev. It overfit (37.4% vs 47.0% for a single global cell). Other midterm findings:
- the image saturates at about 64 tokens;
- cutting the audio is safe, but speeding it up breaks speech;
- image and audio interact strongly, and on sound events the image alone produces confident errors.

**New approach: an item-level allocator that accounts for interactions, plus content-aware audio selection.**
- **(A) Item-level allocator.**
  - A first pass at `b₀ = (64, ½)` gives features z(x): the option distribution, KL(p_v‖p_a) and KL(p_va‖p_a) from the cheap cells, and the audio duration.
  - A regressor g(z, b) predicts the gain Δs = s(x;b) − s(x;b₀) for each candidate cell b. Use ridge regression or gradient-boosted trees, cross-fitted inside the training folds.
  - The policy is `π(x) = argmax_b [g(z(x),b) − λ c(x;b)]`, and the first-pass cost is always counted.
  - Loss: squared error on Δs, with an L2 penalty.
- **(B) Choosing which audio to keep.** The midterm showed that what matters is *which* part of the clip is kept, not compressing all of it. Compare three ways of choosing it:
  - the first half of the clip (the midterm method);
  - a centred window;
  - an **energy/VAD-selected window**: keep the highest-energy segments up to the budget (librosa RMS, cheap).

  Measure the effect at a ¼ and ½ audio budget.

**Research questions:**
- RQ3.1: Does item-level allocation beat the per-type table and the single global cell at matched cost?
- RQ3.2: Do the interaction (KL) features matter? Ablate them.
- RQ3.3: Does content-aware audio selection beat keeping the first part of the clip at the same budget?

**Ablations:** ± KL features; ± first pass (use only question features); the regressor type; each audio-selection method.

**Sweeps:** λ (accuracy–cost curve), the first-pass budget b₀, and the regularization strength.

**Optional:** compare visual-token pruning in the style of FastV (dropping visual tokens by attention, inside the model) against simple resizing at 64 and 256 tokens.

**Overlap with Idea 1 (must be explicit in the report).** Idea 1 chooses the **type of representation**: raw, text or omitted. Idea 3 chooses **how many tokens each modality gets, and which ones**. They share the cheap-probe features from the bank. The Discussion can then report "Idea 1 + Idea 3 together" as one combined menu.

### 3.3 Idea 4: Generated text surrogates (owner: **Nikkita**)

> **Implemented Nov 23 (first draft, for Nikkita to review and extend):** `src/final/idea4_generate.py` and the new configs in `src/analysis_a/prompts.py`.
> - **Changes from the plan below:**
>   - the tagger is **AST** (Audio Spectrogram Transformer, fine-tuned on AudioSet) instead of CLAP;
>   - speech vs non-speech uses AST's P(Speech) ≥ 0.5 (the threshold can be swept afterwards);
>   - the subset is **242 items**: the 120 midterm test items plus 60 fresh speech, 60 fresh sound and the 2 fresh music items.
> - **Configs:** `img_route`, `img_route_fb`, `img_qdesc`, `img_pipe`, `gencap_pipe`, `img_masked`, `cap_masked`, plus the regenerated midterm baselines.
> - **Masking is lexical:** it masks every option word (and its singular form). It cannot catch paraphrases such as "fuse burning" for "flare stick"; state this as a limitation.

**Midterm baseline:** surrogates generated without regard to the question: a Qwen caption, a Qwen audio description and Whisper ASR. These lost 7.5 points to the reference surrogates, mostly on music (−21.7). Whisper hallucinated on 52 of 80 non-speech clips. Generation cost about 25× the raw route. The human audio descriptions often name the answer.

**New approach: surrogates matched to the audio type and the question, with a fallback when transcription fails.**
- **(A) Route each clip to the right surrogate.**
  - An audio-type classifier decides between speech and non-speech, so the audio-type label isn't needed at test time. Use CLAP zero-shot or Qwen's audio-only pass.
  - Speech goes to Whisper.
  - Non-speech goes to an audio tagger or captioner, e.g. CLAP zero-shot tags over AudioSet labels plus a short description.
  - An **ASR-failure detector** falls back to raw audio when the ASR output is empty, a known filler phrase, a repeating loop, or has low average log-probability.
- **(B) Question-conditioned generation.** Pass the question to the generator ("Describe the aspects of this audio relevant to: q"), but **not** the options, so the answer can't leak.
- **(C) Amortised cost.** `C = C_ans + C_gen / k` over the k questions asked about the same input. Report the **break-even k**, the point where generated text becomes cheaper than the raw route.

**Research questions:**
- RQ4.1: Does audio-type routing plus the fallback close the gap to the reference surrogates?
- RQ4.2: Does conditioning on the question help, and does it increase leakage?
- RQ4.3: How much of the reference surrogates' advantage comes from naming the answer? Test this by **masking the answer entity** in the human descriptions.

**Ablations:** ± routing, ± fallback, ± question conditioning, and which tagger or captioner is used.

**Sweeps:** the generation limit (80/120/256 tokens) against accuracy and cost, and the fallback detector's threshold. The 80- and 120-token texts are the first 80/120 tokens of the 256-token generation, which greedy decoding makes identical to a lower limit; the time each limit takes is logged (notebook §8, scored in §9b). The speech threshold (AST P(Speech), 0.5 in the main runs) and the fallback detector are swept afterwards from the bank: notebook §9c scores every item on each routing branch, and `img_asr` and `full` are scored on every item.

**Measuring translation quality:**
- how often the answer is mentioned in the surrogate;
- token-F1 against the human text.

Relate both to accuracy.

**Scope:** the full Idea 4 pipeline costs about 9 s per item. Use a 300–400-item evaluation subset: the 120 midterm test items plus new speech and sound items. Don't try to cover all 1,141.

---

## 4. Report (reuse `report_midterm/` → `report_final/`)

- **Structure:**
  - Abstract, Introduction (with a **new, clean overview figure**: the midterm's was dropped), Related Work, Problem Statement, Experimental Methodology (shared protocol from §1–2);
  - then, for each idea, **Proposed Approach** (model and learning), **Experimental Setup** (research questions, ablations, sweeps), **Results and Discussion** (one subsection per research question, plus failure and success cases);
  - then Conclusion and Future Directions.
- **Page budget (10 pages):**
  - Introduction + Related Work + Problem Statement + Methodology: about 2.5 pages;
  - each idea: about 2.3 pages;
  - Conclusion: 0.3 pages.
  - The midterm-baseline details move to a single sentence plus a citation of our midterm. Extra results go in the ungraded appendix.
- **Related Work** needs about 12–15 citations, grouped by idea. New citations:
  - learning-to-rank: RankNet (Burges et al., 2005);
  - contextual bandits: LinUCB (Li et al., 2010);
  - audio tagging: CLAP (Elizalde et al., 2023 / LAION-CLAP);
  - VAD and energy-based audio selection;
  - query-aware captioning.
- **Must carry over** the midterm fixes:
  - the captions were made by InternVL-2-76B, and only the audio text is human-written;
  - averaging over option orders;
  - the answer-leakage finding.
- **AI disclosure and contributions page:** each person writes their own bullet, accurately. Avoid vague verbs like "compile".

## 5. Poster

- **Recommendation: pick the idea by Wed Nov 25, based on whichever result is clearest.**
  - *After the TA feedback:* lean towards **Idea 1 + the cheap-first-pass combined system**, which is what the TA liked most, if the router beats the hull.
  - Current leaning: **Idea 4**, because it is self-explanatory to outsiders ("text surrogates look cheap but aren't; human captions leak the answer").
  - The alternative is **Idea 3**, if the item-level allocator clearly beats the table.
- **Layout:** 3 columns, with the middle one widest.
  - **Left:** motivation (an image is about 700 tokens and audio about 250) and an OmniBench example item.
  - **Middle:** one large method figure plus 1–2 equations.
  - **Right:** 2–3 plots, each with a one-line takeaway.
- **Fonts:** body text at least 24 pt, headers about 36 pt, never below 18 pt.
- **Talk:** write the 30-second pitch and the 4-minute script, and run them aloud as a team at least once before the session.

---

## 6. Midterm feedback (TA, received Mon Nov 23) and how we address it

**What they liked:**
- the accuracy-vs-compute analysis showing that more image or audio tokens are not always better;
- adaptive routing to the cheapest useful input format ("intuitive and promising");
- a cheap first pass, with extra compute only for difficult questions ("a practical direction").

**Implication:** make the **cheap-first-pass → adaptive escalation** story the backbone of the final report. Ideas 1 and 3 both already use a cheap first pass; the combined experiment in item 4 below ties them together.

**Suggestions and questions, with our response:**

| # | TA suggestion | What we do | Where / owner |
|---|---|---|---|
| 1 | *Compare routing with fixed formats/budgets, random or confidence routing, and an oracle best-format selector.* | The router baselines become: every **fixed format** (7) and every **fixed budget cell** (Idea 3); **random routing**, both per item from the menu (uniform) and random mixing at matched cost (the hull); **confidence routing** (the midterm cascade, plus a cascade on the cheap first pass); and the **oracle**. One table and one accuracy-cost plot contain all of them. *New: the explicit random router.* | §3.1 Idea 1 (Arjun, Lindy) |
| 2 | *Report end-to-end latency, preprocessing and router overhead, not only token counts and accuracy.* | The runners now also log **`decode_ms`** (reading the image and audio files). The Idea 3 audio cut is now inside `preproc_ms`. `bank.py` exposes `model_s`, `preproc_s`, `decode_s` and `e2e_s`. **Router overhead** = the cheap passes (measured on the L4) + feature extraction (SBERT, KL) + router inference, timed on CPU and added per item. Every accuracy-cost curve is shown twice, against **tokens** and against **end-to-end seconds**, plus a **latency breakdown** table: decode / preprocess / cheap pass / router / final pass. | All ideas; the bank and code are done; tables in each section |
| 3 | *Test speech, music, environmental sound and truly cross-modal questions; identify information lost in transcripts.* | **(a)** Every result is broken down by speech / music / sound event (the "environmental sound" category). **(b)** A **"truly cross-modal" subset**, defined from the bank: items the full input answers (≥ ⅔ of orders) while image only and audio only both fail (≤ ⅓). All ideas report on it. **(c)** **Information lost in transcripts:** compare raw audio vs transcript per item (full vs image+transcript; audio only vs caption+transcript). Hand-code the items where raw audio wins into categories: tone or emotion, speaker identity, non-speech sound, instrument or timbre, background events, timing. Also code the reverse case, where the transcript wins (answer leakage). Combine with Idea 4's masking test. | (a, b) all ideas; (c) Idea 4 (Nikkita) + Lindy helps with the coding |
| 4 | *Evaluate when the cheap pass fails, and whether combining routing with adaptive compute yields actual savings.* | **(a) Cheap-pass failure analysis:** how often the first pass (the caption+transcript probe, and the (64, ½) cell) is **confidently wrong**, by audio type, task type and the cross-modal subset; reliability diagrams before and after temperature scaling. **(b) Combined system:** cheap first pass → the Idea 1 router chooses among formats **and** Idea 3 budget cells → final pass. Compared with the full input at matched accuracy, in **wall-clock seconds including all overhead**. This is the "actual savings" number. | (a) Ideas 1 and 3; (b) a new Discussion subsection "Combined system" (Arjun, Lindy + Alvin) |

**Changes to §3:**
- **Idea 1:** add the random router. Report latency and overhead. Report results on the cross-modal subset. Lead the combined-system experiment.
- **Idea 3:** add the cheap-pass failure analysis for the (64, ½) first pass. Report end-to-end latency for each budget cell, since cutting the audio and resizing the image also cost time.
- **Idea 4:** add the "information lost in transcripts" coding. Report latency end to end (generation time is already included).

**Extra work:** about 6–8 person-hours, mostly the transcript-loss coding and the combined-system run, which reuses the existing bank (no new GPU runs). It fits into Tue–Wed in §7.

## 7. Timeline: finish in ONE week (Mon Nov 23 – Sun Nov 29)

**Goal:** a submittable report by **Sun Nov 29**. That leaves Nov 30 – Dec 6 as a buffer for the poster session and for polishing.

**What makes one week possible:**
- **The GPU work is small.** About 10–12 L4-hours in total, and the first ~2.5 h can start **today**, without waiting for the TA feedback.
- **The learned methods are cheap.** Idea 1's router and Idea 3's allocator run on CPU in minutes once the score bank exists, so ablations and sweeps are fast.
- **The TA feedback changes the writing and priorities,** not the data runs. Don't block the runs on it.

**The cost:** about 75–90 person-hours, i.e. **~20 h each in 7 days**, including Thanksgiving. The reduced scope below is now the **plan**, not the fallback.

| Day | Everyone | Idea 1 (Arjun, Lindy) | Idea 3 (Alvin) | Idea 4 (Nikkita) |
|---|---|---|---|---|
| **Mon 11/23** | Commit and push; **start the Colab notebook** (smoke test → base bank → probe cells, ~2.5 L4-h); TA feedback into §6 | Code the pairwise router + disagreement features, tested on the midterm 300 (Mac data) | Implement the audio-selection cells (centered + energy window) | Set up CLAP tags + the ASR-failure detector, tested on the 120 midterm items |
| **Tue 11/24** | Notebook: the other 9 grid cells (~2 h) + audio-selection cells (~1 h) | Full 5-fold runs on the bank: pairwise vs pointwise, ± disagreement, ± calibration, learning curve, λ sweep; **random router**; latency/overhead accounting (§6) | Allocator (ridge) on the bank; ± KL features; λ sweep | Generation runs: routed and question-conditioned pipelines (~4 L4-h); masking test (scoring only) |
| **Wed 11/25** | **Results freeze tonight.** Choose the poster idea. Cross-modal subset results for all ideas (§6.3b) | Figures + failure cases; **combined system** with Alvin (§6.4b) | Figures + failure cases; audio-selection comparison; **cheap-pass failure analysis** (§6.4a) | Analysis, figures, break-even k; **information lost in transcripts** coding, with Lindy (§6.3c) |
| **Thu 11/26** (Thanksgiving) | Light: each person drafts their idea section (Approach, Setup, Results) | Idea 1 section | Idea 3 section | Idea 4 section |
| **Fri 11/27** | Intro, Related Work, Problem Statement, Methodology, Conclusion; overview figure | Shared sections + overview figure | Review Ideas 1/4 | Review Ideas 1/3 |
| **Sat 11/28** | **Full draft**, page-fit to 10 pages; AI disclosure + contributions; start the poster | | | |
| **Sun 11/29** | Final pass; **report ready to submit**; poster draft; pitch script | | | |

**Scope for this week (dropped unless a day frees up):**
- **Idea 1:** no bandit and no MLP. Keep pairwise vs pointwise, disagreement features, calibration, the budget-cell menu, the learning curve and the λ sweep.
- **Idea 3:** no FastV comparison and no boosted trees (ridge only). Keep the allocator ± KL features and the audio-selection comparison.
- **Idea 4:** CLAP tags only (no second captioner); the generation-length sweep 80/120/256, cut from one generation (no extra generation time); the subset is the 120 midterm test items plus up to 120 new speech and sound items. Keep routing ± fallback, question conditioning, the masking test and break-even k.

Every idea must still deliver **new approach vs midterm baseline + ≥2 ablations + 1 sweep + failure cases**.

**Biggest risks and their mitigations:**
- **Idea 4's new models** (CLAP setup). If CLAP isn't working by Tue noon, use Qwen's own question-conditioned description for non-speech audio and keep the fallback.
- **Colab disconnects.** Every step is resumable; keep the tab open.
- **Page limit.** Each idea section gets about 2.3 pages. Put the extras in the appendix.
