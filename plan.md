# Final Report + Poster: Plan

**Report due: Sunday Dec 6, 2026, 8pm ET.** Team of 4, so the report covers **3 research ideas** (1, 3, 4) in **10 pages** plus references. An optional appendix of up to 5 pages is allowed but not graded, plus the AI-disclosure and contributions page.

**Poster:** 30″×40″, one idea, 2–3 key results, a 30-second pitch plus a 4-minute talk. All four of us present, in rotation. *Session date: TBD (fill in).*

Earlier plans: `plan_midterm.md`, `plan_assignment2.md`.

**What the final report must add over the midterm.** Each idea needs:
- a **new approach** beyond the midterm baseline, compared directly against that baseline;
- stated **research questions**, with **ablations** of each new component;
- a **hyperparameter sweep**;
- **statistical tests**;
- **success and failure cases**;
- a **model description and learning** section with the loss function and, where possible, its gradient.

---

## 0. Before anything else (this weekend)

- [ ] **Commit the midterm work and merge the branches.** None of Arjun and Lindy's midterm work is committed yet. `origin/idea3` (Alvin) and `origin/idea4` (Nikkita) are not merged into `main`. Arjun runs:
  ```bash
  git checkout -b midterm-report
  git add .gitignore README.md plan.md plan_midterm.md src/midterm src/midterm_arjun results/midterm results/midterm_arjun \
          report_midterm/main.tex report_midterm/references_midterm.bib report_midterm/figures
  git commit -m "Midterm: Idea 1 router, Idea 3 time-stretch ablation, Idea 4 replication, report"
  git push -u origin midterm-report
  ```
  Then open PRs, or merge `midterm-report`, `origin/idea3` and `origin/idea4` into `main`. From here on, everyone branches off `main`. Commit often: graders use the repo history to judge how evenly work was shared.
- [ ] **Collect the midterm feedback** from the TAs and paste it into §6 below. The final report must address it.
- [ ] **Agree on the data protocol in §1** before anyone runs anything new.

---

## 1. Data protocol (the main shared decision)

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

## 2. Shared score bank (run first)

Every item is run under every configuration below, in all 3 option orders. The rows already store `probs`, so `log p(gold)` comes for free.

| Configs | Who / where | Passes | Time |
|---|---|---|---|
| 7 base configs: full, img_trans, cap_aud, cap_trans, img_only, aud_only, q_only | Arjun/Lindy, M2 Max | 841 new items × 21 = 17.7k | **≈ 8 h** (overnight) |
| Cheap probe cells for Ideas 1 and 3: image 64 tok + audio ½ `(64,½)`, image-only 64 `(64,0)`, audio-only ½ `(0,½)` | Alvin, PSC V100 | 1,141 × 9 = 10.3k | ≈ 3 h |
| Idea 3 grid, the remaining `(b_v,b_a)` cells and the new audio-selection cells (§3.2) | Alvin, PSC V100 | ~1,141 × 16 × 3 | ~1 night |
| Generated surrogates and their scoring (Idea 4) | Nikkita, L4 | see §3.3 | ~1 night |

- **Hardware.** Run each config on a single machine type. Mac and V100 predictions agree on 97.5% of passes, which is fine but must be stated in the report. Never mix machines within one configuration.
- **Output.** Everything goes into one bank file, `results/final/bank_*.jsonl`, with a loader `src/final/bank.py`. The loader returns a matrix of accuracy, `log p(gold)` and cost for every item × configuration.

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

**Sweeps:** the generation limit (40/80/160 tokens) against accuracy and cost, and the fallback detector's threshold.

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

- **Recommendation: pick the idea by Nov 30, based on whichever result is clearest.**
  - Current leaning: **Idea 4**, because it is self-explanatory to outsiders ("text surrogates look cheap but aren't; human captions leak the answer").
  - The alternative is **Idea 3**, if the item-level allocator clearly beats the table.
- **Layout:** 3 columns, with the middle one widest.
  - **Left:** motivation (an image is about 700 tokens and audio about 250) and an OmniBench example item.
  - **Middle:** one large method figure plus 1–2 equations.
  - **Right:** 2–3 plots, each with a one-line takeaway.
- **Fonts:** body text at least 24 pt, headers about 36 pt, never below 18 pt.
- **Talk:** write the 30-second pitch and the 4-minute script, and run them aloud as a team at least once before the session.

---

## 6. Midterm feedback to address
*(paste TA comments here)*

---

## 7. Timeline (15 days; Thanksgiving is Thu Nov 26)

| Dates | Everyone | Idea 1 (Arjun, Lindy) | Idea 3 (Alvin) | Idea 4 (Nikkita) |
|---|---|---|---|---|
| **Sat 11/21 – Sun 11/22** | Merge branches; agree on §1; freeze `final_folds.json` | Start the base-config bank on the Mac (overnight) | Probe cells + grid on PSC | Pick the tagger/captioner; prototype the fallback detector on the 120 midterm items |
| **Mon 11/23 – Wed 11/25** | Bank loader `src/final/bank.py` | Pairwise router + features; first results on the midterm 300 | Allocator + audio-selection cells | Question-conditioned generation; masking experiment |
| **Thu 11/26 – Sun 11/29** | (Thanksgiving: light) | Full cross-validation runs, ablations, sweeps | Full cross-validation runs, ablations, sweeps | Full runs on the evaluation subset |
| **Mon 11/30** | **Results freeze for the poster**; choose the poster idea | Figures | Figures | Figures |
| **Tue 12/1 – Thu 12/3** | Poster + presentation practice | Write the Idea 1 section | Write the Idea 3 section | Write the Idea 4 section |
| **Fri 12/4 – Sat 12/5** | Intro, Related Work, Methodology, Conclusion, overview figure; full draft; page check | | | |
| **Sun 12/6** | Final pass, disclosure page, submit by 8 pm | | | |

**Fallbacks if we run behind:**
- **Idea 1:** drop the bandit and the MLP.
- **Idea 3:** drop the FastV comparison.
- **Idea 4:** drop the size sweep and use only CLAP tags.

Every idea must still finish **new approach vs midterm baseline + ≥2 ablations + 1 sweep**.
