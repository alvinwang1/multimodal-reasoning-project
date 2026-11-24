# Idea 1: learned routing between input representations

1141 items, 5 frozen folds: every decision is out of fold, and every choice (L2 strength, lambda, cascade threshold, temperature, PCA) is made inside the training folds. Accuracy is the mean over the 3 option orders; tokens are input tokens relative to the full input, including every cheap pass a method runs; ms/item is end to end on the L4 (reading the media, all passes, question embedding, router). *vs hull*: accuracy minus random mixing of the method's own fixed options at the same mean token cost (paired over items); *oracle gap closed*: that gain as a share of the oracle's (same options, same cost). Macro = mean of the 3 audio types; cross-modal = the 83 items the full input answers while image only and audio only both fail. 95% CIs: 10,000 bootstrap resamples over items.

Cheap passes (always paid by the routers that use them; answering with one costs nothing extra): probe = v64_a0.5: 0.28 tokens, 168 ms; disagree = v64_a0 + v0_a0.5: 0.40 tokens, 207 ms; midterm = C+T: 0.35 tokens, 64 ms.

## Headline (operating point chosen for accuracy)

| method | accuracy | macro | cross-modal | tokens | ms/item | vs hull at same cost (pts) | oracle gap closed |
|---|---|---|---|---|---|---|---|
| I+A | 42.3 [39.8, 44.8] | 44.6 | 80.7 | 1.00 | 435 | - | - |
| I+T | 46.6 [44.1, 49.1] | 50.8 | 70.3 | 0.82 | 347 | - | - |
| v256_a1 | 42.8 [40.3, 45.3] | 44.6 | 66.3 | 0.57 | 268 | - | - |
| C+T | 41.0 [38.5, 43.4] | 45.8 | 53.8 | 0.35 | 64 | - | - |
| v64_a0.5 | 36.6 [34.3, 39.1] | 40.3 | 47.4 | 0.28 | 203 | - | - |
| Q | 24.6 [22.8, 26.4] | 30.5 | 13.7 | 0.12 | 40 | - | - |
| Uniform random router (base7+cells) | 37.6 [35.8, 39.4] | 41.1 | 47.2 | 0.52 | 227 | -5.3 [-6.6, -4.1] | -19.6 |
| Cascade C+T -> I+T (midterm) | 46.4 [43.9, 48.9] | 50.6 | 69.1 | 1.12 | 391 | -0.2 [-0.6, +0.2] | -2.3 |
| Cascade v64_a0.5 -> I+A | 42.2 [39.7, 44.7] | 44.6 | 80.7 | 1.28 | 602 | -0.1 [-0.3, +0.0] | -1.5 |
| Cascade v64_a0.5 -> v256_a1 | 42.8 [40.3, 45.3] | 44.6 | 66.3 | 0.84 | 436 | +0.0 [+0.0, +0.0] | +0.0 |
| Midterm router | 45.5 [43.0, 48.1] | 49.5 | 69.1 | 1.07 | 388 | -1.0 [-1.9, -0.1] | -4.7 |
| New router | 42.9 [40.5, 45.4] | 47.5 | 57.8 | 1.14 | 617 | -3.6 [-5.6, -1.7] | -14.8 |
| New router, no cheap passes | 46.5 [44.0, 49.1] | 50.7 | 70.7 | 0.82 | 367 | -0.0 [-0.2, +0.1] | -0.0 |
| New router, no human text | 40.4 [38.1, 42.9] | 42.9 | 55.8 | 1.03 | 586 | -2.4 [-4.1, -0.6] | -10.8 |
| Oracle (base7+cells; sees the answers) | 70.2 [68.0, 72.3] | 73.0 | 90.0 | 0.31 | 138 | +31.6 [+30.0, +33.2] | +100.0 |

## At a target cost (lambda or threshold chosen in the training folds to reach it)

A router or cascade that runs cheap passes cannot cost less than them, so below that floor its rows sit at the floor (see *tokens*).

| target tokens | method | accuracy | macro | cross-modal | tokens | ms/item | vs hull at same cost (pts) | oracle gap closed |
|---|---|---|---|---|---|---|---|---|
| 0.35 | Cascade C+T -> I+T (midterm) | 41.0 [38.5, 43.4] | 45.8 | 53.8 | 0.35 | 66 | -0.0 [-0.2, +0.1] | -1.7 |
| 0.35 | Cascade v64_a0.5 -> I+A | 37.6 [35.2, 40.1] | 41.1 | 53.0 | 0.35 | 232 | +0.4 [-0.2, +1.0] | +5.3 |
| 0.35 | Cascade v64_a0.5 -> v256_a1 | 38.2 [35.7, 40.7] | 41.0 | 51.4 | 0.35 | 232 | -0.0 [-0.8, +0.8] | -0.2 |
| 0.35 | Midterm router | 41.0 [38.5, 43.4] | 45.8 | 53.8 | 0.35 | 86 | -0.0 [-0.0, -0.0] | -0.0 |
| 0.35 | New router | 35.6 [33.5, 37.8] | 40.0 | 43.8 | 0.68 | 432 | -9.2 [-11.5, -6.9] | -35.0 |
| 0.35 | New router, no cheap passes | 39.3 [36.9, 41.7] | 45.1 | 61.8 | 0.35 | 210 | -1.7 [-4.0, +0.6] | -5.9 |
| 0.35 | New router, no human text | 36.0 [33.8, 38.2] | 40.1 | 43.8 | 0.68 | 432 | -6.8 [-9.1, -4.5] | -31.1 |
| 0.35 | Oracle | 70.2 [68.0, 72.3] | 73.0 | 90.0 | 0.31 | 138 | +31.6 [+30.0, +33.2] | +100.0 |
| 0.5 | Cascade C+T -> I+T (midterm) | 43.3 [40.8, 45.9] | 48.1 | 61.4 | 0.50 | 127 | +0.6 [-0.3, +1.5] | +5.2 |
| 0.5 | Cascade v64_a0.5 -> I+A | 38.3 [35.9, 40.8] | 41.8 | 58.6 | 0.50 | 291 | -0.1 [-1.0, +0.8] | -0.8 |
| 0.5 | Cascade v64_a0.5 -> v256_a1 | 39.0 [36.6, 41.5] | 41.9 | 54.2 | 0.50 | 294 | -2.4 [-3.4, -1.3] | -35.3 |
| 0.5 | Midterm router | 42.6 [40.1, 45.0] | 47.7 | 59.4 | 0.47 | 137 | +0.1 [-0.8, +1.1] | +0.5 |
| 0.5 | New router | 35.6 [33.5, 37.8] | 40.0 | 43.8 | 0.68 | 432 | -9.2 [-11.5, -6.9] | -35.0 |
| 0.5 | New router, no cheap passes | 43.2 [40.7, 45.7] | 48.6 | 69.1 | 0.50 | 278 | +0.5 [-1.4, +2.3] | +1.7 |
| 0.5 | New router, no human text | 36.0 [33.8, 38.2] | 40.1 | 43.8 | 0.68 | 432 | -6.8 [-9.1, -4.5] | -31.1 |
| 0.5 | Oracle | 70.2 [68.0, 72.3] | 73.0 | 90.0 | 0.31 | 138 | +31.6 [+30.0, +33.2] | +100.0 |
| 0.7 | Cascade C+T -> I+T (midterm) | 45.5 [42.9, 48.1] | 49.7 | 63.9 | 0.70 | 213 | +0.4 [-0.7, +1.5] | +4.3 |
| 0.7 | Cascade v64_a0.5 -> I+A | 39.0 [36.6, 41.6] | 42.6 | 64.3 | 0.70 | 371 | -0.9 [-1.9, +0.1] | -10.9 |
| 0.7 | Cascade v64_a0.5 -> v256_a1 | 40.8 [38.3, 43.3] | 43.3 | 61.4 | 0.70 | 376 | -2.0 [-2.9, -1.2] | -38.7 |
| 0.7 | Midterm router | 44.4 [41.9, 47.0] | 48.8 | 67.1 | 0.70 | 233 | -0.7 [-1.9, +0.5] | -3.0 |
| 0.7 | New router | 36.9 [34.7, 39.1] | 42.2 | 45.4 | 0.70 | 439 | -8.2 [-10.5, -5.9] | -31.5 |
| 0.7 | New router, no cheap passes | 46.2 [43.6, 48.8] | 50.5 | 71.5 | 0.74 | 346 | +0.6 [-0.3, +1.5] | +2.5 |
| 0.7 | New router, no human text | 36.2 [34.0, 38.4] | 40.0 | 45.8 | 0.69 | 441 | -6.6 [-8.9, -4.4] | -30.3 |
| 0.7 | Oracle | 70.2 [68.0, 72.3] | 73.0 | 90.0 | 0.31 | 138 | +31.6 [+30.0, +33.2] | +100.0 |

## Research questions (A - B, cost-adjusted, paired over items)

Each side is its accuracy minus random mixing of all 25 fixed representations at that side's mean token cost, so a cheaper router is not penalized for spending less. p: sign-flip test, Holm-adjusted within each operating point.

|  | comparison | point | A: acc / tokens | B: acc / tokens | A - B (pts) | p_holm |
|---|---|---|---|---|---|---|
| RQ1.1 | pairwise vs pointwise training | accuracy-tuned | 42.9 / 1.14 | 46.1 / 1.38 | -3.1 [-5.0, -1.3] | 0.010 |
| RQ1.1 | pairwise vs pointwise training | cost 0.5 | 35.6 / 0.68 | 36.5 / 0.68 | -0.8 [-2.3, +0.6] | 0.974 |
| RQ1.2 | adding the disagreement passes | accuracy-tuned | 42.9 / 1.14 | 43.9 / 0.92 | -0.9 [-2.5, +0.5] | 0.974 |
| RQ1.2 | adding the disagreement passes | cost 0.5 | 35.6 / 0.68 | 38.1 / 0.48 | -4.7 [-6.8, -2.6] | 0.001 |
| RQ1.2 | the cheap first pass vs no pass | accuracy-tuned | 43.9 / 0.92 | 46.5 / 0.82 | -2.7 [-4.2, -1.3] | 0.003 |
| RQ1.2 | the cheap first pass vs no pass | cost 0.5 | 38.1 / 0.48 | 43.2 / 0.50 | -4.9 [-6.7, -3.2] | 0.001 |
| RQ1.3 | adding Idea 3 budget cells to the menu | accuracy-tuned | 42.9 / 1.14 | 42.7 / 1.18 | +0.3 [-0.8, +1.4] | 1.000 |
| RQ1.3 | adding Idea 3 budget cells to the menu | cost 0.5 | 35.6 / 0.68 | 36.1 / 0.68 | -0.5 [-1.0, -0.1] | 0.170 |
| ablation | temperature scaling of the passes | accuracy-tuned | 42.9 / 1.14 | 42.4 / 1.14 | +0.5 [-0.5, +1.5] | 1.000 |
| ablation | temperature scaling of the passes | cost 0.5 | 35.6 / 0.68 | 35.8 / 0.68 | -0.1 [-0.9, +0.6] | 1.000 |
| ablation | the audio-type feature | accuracy-tuned | 42.9 / 1.14 | 41.5 / 1.13 | +1.4 [+0.5, +2.3] | 0.017 |
| ablation | the audio-type feature | cost 0.5 | 35.6 / 0.68 | 35.6 / 0.68 | +0.0 [-0.4, +0.4] | 1.000 |
| ablation | the question embedding | accuracy-tuned | 42.9 / 1.14 | 43.6 / 1.16 | -0.7 [-2.1, +0.7] | 1.000 |
| ablation | the question embedding | cost 0.5 | 35.6 / 0.68 | 35.7 / 0.68 | -0.1 [-1.0, +0.8] | 1.000 |
| midterm | new router vs the midterm router | accuracy-tuned | 42.9 / 1.14 | 45.5 / 1.07 | -2.6 [-4.6, -0.6] | 0.053 |
| midterm | new router vs the midterm router | cost 0.5 | 35.6 / 0.68 | 42.6 / 0.47 | -9.3 [-11.8, -6.9] | 0.001 |
| midterm | no human-written text in the menu | accuracy-tuned | 40.4 / 1.03 | 42.9 / 1.14 | -2.5 [-3.9, -1.2] | 0.003 |
| midterm | no human-written text in the menu | cost 0.5 | 36.0 / 0.68 | 35.6 / 0.68 | +0.4 [-0.0, +0.8] | 0.565 |

## Router variants

*vs hull @ tau*: at the operating point aimed at mean token cost tau (realized cost in brackets: a router that runs cheap passes cannot go below their cost). *Added area*: mean accuracy (pts) the router's operating points add to the fixed-option frontier over costs 0.2-0.8 (0 = never better than random mixing). *Spearman*: per item, router scores vs log p(gold) over the menu.

### Model x features x menu

| router | acc (tokens), accuracy-tuned | vs hull, accuracy-tuned | vs hull @ 0.35 (tokens) | vs hull @ 0.5 (tokens) | vs hull @ 0.7 (tokens) | added area | oracle gap closed | Spearman | L2 (folds) |
|---|---|---|---|---|---|---|---|---|---|
| pointwise · base7 · question+meta+audio type+midterm | 45.5 (1.07) | -1.0 [-1.9, -0.1] | -0.0 (0.35) | +0.1 (0.47) | -0.7 (0.70) | +0.2 | -4.7 | 0.211 | 0.1x5 |
| pairwise · base7+cells · question+meta+audio type+probe+disagree | 42.9 (1.14) | -3.6 [-5.6, -1.7] | -9.2 (0.68) | -9.2 (0.68) | -8.2 (0.70) | +0.0 | -14.8 | 0.221 | 1x5 |
| pairwise · base7+cells · question+meta+audio type | 46.5 (0.82) | -0.0 [-0.2, +0.1] | -1.7 (0.35) | +0.5 (0.50) | +0.6 (0.74) | +0.7 | -0.0 | 0.190 | 10x5 |
| pairwise · raw · question+meta+audio type+probe+disagree | 40.4 (1.03) | -2.4 [-4.1, -0.6] | -6.8 (0.68) | -6.8 (0.68) | -6.6 (0.69) | +0.0 | -10.8 | 0.227 | 1x5 |
| pointwise · base7 · question+meta+audio type | 45.4 (0.79) | -0.8 [-1.6, +0.1] | -0.0 (0.35) | +0.1 (0.49) | -0.7 (0.71) | +0.7 | -3.4 | 0.217 | 0.1x5 |
| pointwise · base7 · question+meta+audio type+probe | 45.8 (1.07) | -0.7 [-1.5, +0.1] | -3.0 (0.33) | -3.9 (0.49) | -3.4 (0.71) | +0.0 | -3.2 | 0.204 | 0.1x5 |
| pointwise · base7 · question+meta+audio type+probe+disagree | 45.9 (1.44) | -0.7 [-1.8, +0.3] | -8.4 (0.68) | -8.4 (0.68) | -7.9 (0.70) | +0.0 | -3.0 | 0.225 | 0.1x5 |
| pairwise · base7 · question+meta+audio type | 46.0 (0.79) | -0.2 [-0.9, +0.4] | -4.3 (0.33) | -1.8 (0.53) | -0.2 (0.73) | +0.1 | -1.1 | 0.199 | 1x3, 10x2 |
| pairwise · base7 · question+meta+audio type+probe | 44.6 (0.93) | -2.0 [-3.3, -0.7] | -4.9 (0.34) | -4.6 (0.48) | -4.0 (0.69) | +0.0 | -8.8 | 0.201 | 1x5 |
| pairwise · base7 · question+meta+audio type+probe+disagree | 42.7 (1.18) | -3.9 [-5.7, -2.1] | -8.7 (0.68) | -8.7 (0.68) | -8.4 (0.69) | +0.0 | -16.6 | 0.227 | 1x5 |
| pointwise · base7+cells · question+meta+audio type | 45.9 (0.75) | +0.2 [-0.8, +1.2] | -1.4 (0.35) | +0.4 (0.51) | +0.1 (0.73) | +0.5 | +0.8 | 0.196 | 0.1x5 |
| pointwise · base7+cells · question+meta+audio type+probe | 46.0 (1.03) | -0.5 [-1.5, +0.4] | -2.6 (0.33) | -3.5 (0.47) | -3.6 (0.73) | +0.0 | -2.2 | 0.198 | 0.1x4, 1x1 |
| pointwise · base7+cells · question+meta+audio type+probe+disagree | 46.1 (1.38) | -0.5 [-1.7, +0.8] | -8.4 (0.68) | -8.4 (0.68) | -7.9 (0.69) | +0.0 | -2.0 | 0.222 | 0.1x5 |
| pairwise · base7+cells · question+meta+audio type+probe | 43.9 (0.92) | -2.7 [-4.1, -1.3] | -5.2 (0.34) | -4.5 (0.48) | -3.7 (0.70) | +0.0 | -11.4 | 0.186 | 1x4, 10x1 |
| pointwise · raw · question+meta+audio type | 42.2 (0.65) | -0.6 [-1.5, +0.3] | -0.3 (0.35) | -1.0 (0.51) | -0.6 (0.67) | +0.3 | -2.8 | 0.175 | 0.1x4, 1x1 |
| pointwise · raw · question+meta+audio type+probe | 41.9 (0.91) | -0.9 [-1.9, +0.0] | -1.6 (0.33) | -3.2 (0.47) | -2.1 (0.71) | +0.0 | -4.5 | 0.177 | 0.1x4, 1x1 |
| pointwise · raw · question+meta+audio type+probe+disagree | 40.9 (1.23) | -1.9 [-3.2, -0.7] | -6.3 (0.68) | -6.3 (0.68) | -6.4 (0.69) | +0.0 | -8.8 | 0.204 | 0.1x5 |
| pairwise · raw · question+meta+audio type | 41.4 (0.51) | -0.8 [-1.6, -0.0] | -0.2 (0.34) | +0.4 (0.51) | -0.1 (0.55) | +0.2 | -3.7 | 0.176 | 10x5 |
| pairwise · raw · question+meta+audio type+probe | 41.0 (0.76) | -1.8 [-3.2, -0.3] | -4.3 (0.34) | -5.1 (0.48) | -2.5 (0.70) | +0.0 | -8.5 | 0.188 | 1x5 |
| pairwise · base7 · question+meta+audio type+midterm | 43.8 (1.04) | -2.7 [-3.9, -1.6] | -0.4 (0.35) | -5.0 (0.48) | -4.5 (0.69) | +0.0 | -12.6 | 0.208 | 1x5 |

### Ablations (new router and its pointwise twin)

| router | acc (tokens), accuracy-tuned | vs hull, accuracy-tuned | vs hull @ 0.35 (tokens) | vs hull @ 0.5 (tokens) | vs hull @ 0.7 (tokens) | added area | oracle gap closed | Spearman | L2 (folds) |
|---|---|---|---|---|---|---|---|---|---|
| pairwise · base7+cells · question+meta+audio type+probe+disagree · no temperature | 42.4 (1.14) | -4.1 [-6.1, -2.2] | -9.0 (0.68) | -9.0 (0.68) | -8.7 (0.69) | +0.0 | -16.8 | 0.220 | 1x5 |
| pairwise · base7+cells · question+meta+probe+disagree | 41.5 (1.13) | -5.0 [-7.0, -3.1] | -9.2 (0.68) | -9.2 (0.68) | -8.4 (0.69) | +0.0 | -20.5 | 0.214 | 1x5 |
| pairwise · base7+cells · meta+audio type+probe+disagree | 43.6 (1.16) | -3.0 [-4.8, -1.1] | -9.1 (0.68) | -9.1 (0.68) | -8.6 (0.70) | +0.0 | -12.0 | 0.227 | 1x4, 0.1x1 |
| pointwise · base7+cells · question+meta+audio type+probe+disagree · no temperature | 45.2 (1.38) | -1.3 [-2.6, -0.1] | -8.5 (0.68) | -8.5 (0.68) | -8.4 (0.69) | +0.0 | -5.5 | 0.215 | 0.1x5 |
| pointwise · base7+cells · question+meta+probe+disagree | 45.3 (1.40) | -1.3 [-2.5, -0.1] | -8.5 (0.68) | -8.5 (0.68) | -7.9 (0.69) | +0.0 | -5.1 | 0.218 | 0.1x5 |
| pointwise · base7+cells · meta+audio type+probe+disagree | 45.9 (1.40) | -0.7 [-1.6, +0.2] | -7.7 (0.68) | -7.7 (0.68) | -6.8 (0.69) | +0.0 | -2.9 | 0.218 | 0.1x4, 0.001x1 |
| pairwise · base7+cells · question+meta | 46.5 (0.82) | -0.0 [-0.2, +0.0] | -1.5 (0.35) | +0.1 (0.49) | +0.2 (0.77) | +0.5 | -0.1 | 0.186 | 10x5 |
| pairwise · base7+cells · meta+audio type | 46.6 (0.73) | +1.1 [+0.1, +2.2] | -0.5 (0.32) | +0.4 (0.51) | +1.2 (0.70) | +0.9 | +4.5 | 0.194 | 1x5 |
| pointwise · base7+cells · question+meta | 45.4 (0.76) | -0.4 [-1.4, +0.5] | -0.9 (0.34) | -0.2 (0.50) | -0.2 (0.72) | +0.4 | -1.8 | 0.193 | 0.1x5 |
| pointwise · base7+cells · meta+audio type | 45.8 (0.76) | -0.0 [-0.7, +0.7] | -0.3 (0.34) | -0.7 (0.51) | +0.3 (0.72) | +0.7 | -0.0 | 0.199 | 0.1x5 |

### Sweeps: L2 strength and pair weighting

| router | acc (tokens), accuracy-tuned | vs hull, accuracy-tuned | vs hull @ 0.35 (tokens) | vs hull @ 0.5 (tokens) | vs hull @ 0.7 (tokens) | added area | oracle gap closed | Spearman | L2 (folds) |
|---|---|---|---|---|---|---|---|---|---|
| pairwise · base7+cells · question+meta+audio type+probe+disagree · l2=0.1 | 41.8 (1.11) | -4.7 [-6.8, -2.6] | -8.9 (0.68) | -8.9 (0.68) | -8.3 (0.69) | +0.0 | -19.3 | 0.210 | 0.1x5 |
| pairwise · base7+cells · question+meta+audio type+probe+disagree · l2=0.3 | 42.2 (1.10) | -4.4 [-6.4, -2.3] | -8.6 (0.68) | -8.6 (0.68) | -8.3 (0.69) | +0.0 | -17.7 | 0.213 | 0.3x5 |
| pairwise · base7+cells · question+meta+audio type+probe+disagree · l2=1 | 42.9 (1.14) | -3.6 [-5.6, -1.7] | -9.2 (0.68) | -9.2 (0.68) | -8.2 (0.70) | +0.0 | -14.8 | 0.221 | 1x5 |
| pairwise · base7+cells · question+meta+audio type+probe+disagree · l2=3 | 44.0 (1.21) | -2.6 [-4.4, -0.8] | -9.2 (0.68) | -9.2 (0.68) | -8.4 (0.70) | +0.0 | -10.5 | 0.228 | 3x5 |
| pairwise · base7+cells · question+meta+audio type+probe+disagree · l2=10 | 45.0 (1.35) | -1.6 [-2.9, -0.4] | -9.9 (0.68) | -9.9 (0.68) | -9.2 (0.70) | +0.0 | -6.5 | 0.230 | 10x5 |
| pairwise · base7+cells · question+meta+audio type+probe+disagree · l2=30 | 46.2 (1.46) | -0.4 [-1.0, +0.3] | -9.7 (0.68) | -9.7 (0.68) | -9.8 (0.69) | +0.0 | -1.5 | 0.222 | 30x5 |
| pointwise · base7+cells · question+meta+audio type+probe+disagree · l2=0.001 | 43.8 (1.27) | -2.7 [-4.4, -1.1] | -8.6 (0.68) | -8.6 (0.68) | -8.3 (0.69) | +0.0 | -11.2 | 0.217 | 0.001x5 |
| pointwise · base7+cells · question+meta+audio type+probe+disagree · l2=0.01 | 44.2 (1.32) | -2.4 [-4.0, -0.8] | -8.8 (0.68) | -8.8 (0.68) | -8.6 (0.69) | +0.0 | -9.8 | 0.218 | 0.01x5 |
| pointwise · base7+cells · question+meta+audio type+probe+disagree · l2=0.03 | 44.8 (1.34) | -1.8 [-3.3, -0.4] | -8.7 (0.68) | -8.7 (0.68) | -8.4 (0.69) | +0.0 | -7.4 | 0.220 | 0.03x5 |
| pointwise · base7+cells · question+meta+audio type+probe+disagree · l2=0.1 | 46.1 (1.38) | -0.5 [-1.7, +0.8] | -8.4 (0.68) | -8.4 (0.68) | -7.9 (0.69) | +0.0 | -2.0 | 0.222 | 0.1x5 |
| pointwise · base7+cells · question+meta+audio type+probe+disagree · l2=0.3 | 46.3 (1.45) | -0.3 [-1.1, +0.5] | -8.0 (0.68) | -8.0 (0.68) | -7.7 (0.69) | +0.0 | -1.2 | 0.222 | 0.3x5 |
| pointwise · base7+cells · question+meta+audio type+probe+disagree · l2=1 | 46.5 (1.49) | -0.1 [-0.3, +0.0] | -8.3 (0.68) | -8.3 (0.68) | -7.8 (0.69) | +0.0 | -0.4 | 0.216 | 1x5 |
| pairwise · base7+cells · question+meta+audio type+probe+disagree · weights=acc | 46.2 (1.43) | -0.4 [-1.3, +0.4] | -7.6 (0.68) | -7.6 (0.68) | -7.3 (0.70) | +0.0 | -1.7 | 0.233 | 1x5 |
| pairwise · base7+cells · question+meta+audio type+probe+disagree · weights=uniform | 45.6 (1.37) | -0.9 [-2.2, +0.4] | -8.1 (0.68) | -8.1 (0.68) | -7.3 (0.70) | +0.0 | -3.8 | 0.240 | 1x5 |

## Learning curve (RQ1.4): training items per fold

Random subsamples of the training folds (3 per size; ± = SD over them); the largest size is all of them (one run).

| router | model | training items | runs | accuracy (accuracy-tuned) | vs hull, accuracy-tuned | added area |
|---|---|---|---|---|---|---|
| New router | pairwise | 180 | 3 | 43.5 ± 0.9 | -3.1 ± 0.9 | +0.0 |
| New router | pairwise | 360 | 3 | 43.2 ± 0.3 | -3.3 ± 0.3 | +0.0 |
| New router | pairwise | 540 | 3 | 42.2 ± 0.9 | -4.3 ± 0.9 | +0.0 |
| New router | pairwise | 720 | 3 | 42.8 ± 0.7 | -3.8 ± 0.7 | +0.0 |
| New router | pairwise | 912 | 1 | 42.9 ± - | -3.6 ± - | +0.0 |
| New router | pointwise | 180 | 3 | 45.1 ± 0.3 | -1.5 ± 0.3 | +0.0 |
| New router | pointwise | 360 | 3 | 45.8 ± 0.1 | -0.8 ± 0.1 | +0.0 |
| New router | pointwise | 540 | 3 | 45.0 ± 0.6 | -1.6 ± 0.6 | +0.0 |
| New router | pointwise | 720 | 3 | 45.3 ± 0.3 | -1.2 ± 0.3 | +0.0 |
| New router | pointwise | 912 | 1 | 46.1 ± - | -0.5 ± - | +0.0 |
| New router, no cheap passes | pairwise | 180 | 3 | 45.9 ± 0.4 | +0.6 ± 0.1 | +0.6 |
| New router, no cheap passes | pairwise | 360 | 3 | 45.9 ± 1.0 | +0.1 ± 0.6 | +0.3 |
| New router, no cheap passes | pairwise | 540 | 3 | 46.4 ± 0.3 | +0.2 ± 0.3 | +0.5 |
| New router, no cheap passes | pairwise | 720 | 3 | 46.4 ± 0.2 | +0.0 ± 0.2 | +0.7 |
| New router, no cheap passes | pairwise | 912 | 1 | 46.5 ± - | -0.0 ± - | +0.7 |
| New router, no cheap passes | pointwise | 180 | 3 | 45.1 ± 1.2 | +0.2 ± 0.7 | +1.0 |
| New router, no cheap passes | pointwise | 360 | 3 | 46.0 ± 0.5 | +0.4 ± 0.3 | +0.8 |
| New router, no cheap passes | pointwise | 540 | 3 | 45.5 ± 0.2 | +0.0 ± 0.2 | +0.6 |
| New router, no cheap passes | pointwise | 720 | 3 | 45.9 ± 0.2 | +0.1 ± 0.1 | +0.6 |
| New router, no cheap passes | pointwise | 912 | 1 | 45.9 ± - | +0.2 ± - | +0.5 |

## By audio type (accuracy-tuned operating point)

| method | speech | sound event | music | macro | cross-modal |
|---|---|---|---|---|---|
| I+A (fixed) | 41.8 [38.7, 44.8] | 39.6 [34.5, 44.8] | 52.4 [43.8, 61.0] | 44.6 | 80.7 [77.1, 84.3] |
| I+T (fixed) | 45.4 [42.4, 48.4] | 43.0 [37.7, 48.3] | 63.8 [54.9, 72.1] | 50.8 | 70.3 [63.1, 77.1] |
| Cascade C+T -> I+T (midterm) | 45.3 [42.2, 48.3] | 42.8 [37.5, 48.2] | 63.8 [54.9, 72.1] | 50.6 | 69.1 [61.8, 76.3] |
| Cascade v64_a0.5 -> I+A | 41.7 [38.6, 44.7] | 39.6 [34.5, 44.8] | 52.4 [43.8, 61.0] | 44.6 | 80.7 [77.1, 84.3] |
| Cascade v64_a0.5 -> v256_a1 | 41.9 [38.9, 44.9] | 43.0 [37.9, 48.3] | 48.9 [40.3, 57.5] | 44.6 | 66.3 [59.0, 73.5] |
| Midterm router | 45.1 [42.0, 48.1] | 39.6 [34.5, 44.8] | 63.8 [55.2, 72.1] | 49.5 | 69.1 [61.8, 76.3] |
| New router | 40.9 [38.0, 43.8] | 42.5 [37.7, 47.3] | 59.0 [50.5, 67.3] | 47.5 | 57.8 [49.8, 65.9] |
| New router, no cheap passes | 45.4 [42.4, 48.4] | 42.8 [37.5, 48.1] | 63.8 [54.9, 72.1] | 50.7 | 70.7 [63.5, 77.5] |
| New router, no human text | 39.0 [36.1, 41.9] | 41.8 [36.9, 46.5] | 47.9 [40.0, 55.9] | 42.9 | 55.8 [47.0, 64.3] |
| Oracle | 67.7 [65.0, 70.4] | 74.7 [70.4, 78.9] | 76.5 [69.8, 82.9] | 73.0 | 90.0 [86.7, 93.2] |

items: speech 771, sound event 265, music 105, cross-modal 83

## Where the time goes (ms per item, L4, accuracy-tuned operating point)

| method | read media | cheap passes | question embedding | router | final pass | total |
|---|---|---|---|---|---|---|
| I+A (fixed) | 35.0 | 0.0 | 0.0 | 0.0 | 399.8 | 434.8 |
| Cascade C+T -> I+T (midterm) | 33.2 | 64.0 | 0.0 | 0.0 | 293.9 | 391.2 |
| Cascade v64_a0.5 -> I+A | 35.0 | 168.3 | 0.0 | 0.0 | 398.8 | 602.0 |
| Cascade v64_a0.5 -> v256_a1 | 35.0 | 168.3 | 0.0 | 0.0 | 232.6 | 435.8 |
| Midterm router | 30.5 | 64.0 | 21.6 | 0.0 | 271.7 | 387.9 |
| New router | 35.0 | 375.5 | 21.6 | 0.0 | 185.1 | 617.1 |
| New router, no cheap passes | 35.0 | 0.0 | 21.6 | 0.0 | 310.5 | 367.1 |
| New router, no human text | 35.0 | 375.5 | 21.6 | 0.0 | 153.8 | 585.8 |

Question embedding: SBERT on CPU, batch size 1 (21.6 ms per question).

## Where the routers send items (share of items)

**New router, accuracy-tuned**

| representation | all | speech | sound event | music |
|---|---|---|---|---|
| I+A | 3.3 | 4.5 | 1.1 | 0.0 |
| I+T | 34.4 | 38.8 | 15.8 | 48.6 |
| C+A | 1.0 | 1.4 | 0.0 | 0.0 |
| C+T | 4.4 | 1.7 | 0.0 | 35.2 |
| A | 1.8 | 0.6 | 6.0 | 0.0 |
| Q | 1.2 | 0.8 | 3.0 | 0.0 |
| v256_a1 | 11.6 | 14.5 | 7.5 | 0.0 |
| v64_a1 | 18.3 | 14.0 | 35.5 | 6.7 |
| v64_a0.5 | 6.5 | 5.8 | 7.2 | 9.5 |
| v64_a0 | 1.0 | 1.4 | 0.0 | 0.0 |
| v0_a0.5 | 16.6 | 16.3 | 23.8 | 0.0 |

**New router, cost 0.5**

| representation | all | speech | sound event | music |
|---|---|---|---|---|
| Q | 0.1 | 0.0 | 0.4 | 0.0 |
| v64_a0.5 | 44.1 | 40.3 | 48.3 | 61.0 |
| v64_a0 | 7.6 | 9.1 | 2.6 | 9.5 |
| v0_a0.5 | 48.2 | 50.6 | 48.7 | 29.5 |

**New router, no human text, accuracy-tuned**

| representation | all | speech | sound event | music |
|---|---|---|---|---|
| I+A | 9.2 | 11.2 | 1.5 | 14.3 |
| v768_a0.5 | 0.8 | 0.9 | 0.0 | 1.9 |
| v256_a1 | 24.8 | 31.1 | 12.1 | 10.5 |
| v256_a0.5 | 0.9 | 1.2 | 0.0 | 1.0 |
| v64_a1 | 28.5 | 23.0 | 45.3 | 26.7 |
| v64_a0.5 | 9.4 | 7.4 | 7.9 | 27.6 |
| I | 0.3 | 0.3 | 0.0 | 1.0 |
| A | 2.2 | 0.9 | 5.7 | 2.9 |
| Q | 2.0 | 1.4 | 3.0 | 3.8 |
| v64_a0 | 1.1 | 1.2 | 0.0 | 2.9 |
| v0_a0.5 | 20.9 | 21.5 | 24.5 | 7.6 |

**New router, no human text, cost 0.5**

| representation | all | speech | sound event | music |
|---|---|---|---|---|
| v64_a0.5 | 46.1 | 41.4 | 50.6 | 69.5 |
| Q | 0.1 | 0.0 | 0.4 | 0.0 |
| v64_a0 | 6.8 | 8.6 | 1.9 | 6.7 |
| v0_a0.5 | 47.0 | 50.1 | 47.2 | 23.8 |

## The cheap passes: calibration and confident errors (original option order, one pass)

Temperature fitted on the other folds. Confidently wrong = top probability >= 0.8 and wrong.

| pass | accuracy | mean max prob | ECE | ECE scaled | temperature | confidently wrong (p>=0.8) | confidently wrong, scaled | confidently wrong: speech | confidently wrong: sound event | confidently wrong: music |
|---|---|---|---|---|---|---|---|---|---|---|
| v64_a0.5 | 35.1 | 68.8 | 33.7 | 1.3 | 5.31 | 20.3 | 0.0 | 21.8 | 21.5 | 6.7 |
| v64_a0 | 29.4 | 70.8 | 41.4 | 3.7 | 13.54 | 28.0 | 0.0 | 27.4 | 38.9 | 4.8 |
| v0_a0.5 | 26.7 | 61.1 | 34.4 | 2.1 | 11.98 | 16.5 | 0.0 | 14.9 | 23.8 | 9.5 |
| C+T | 38.7 | 71.7 | 33.1 | 2.6 | 4.07 | 21.9 | 0.0 | 21.7 | 26.8 | 11.4 |

## On the midterm test set (120 items; routers trained on the other 1,021)

| method | accuracy | macro | cross-modal | tokens | ms/item | vs hull at same cost (pts) | oracle gap closed |
|---|---|---|---|---|---|---|---|
| I+A (fixed) | 46.4 [38.9, 54.2] | 46.4 | 85.7 | 1.00 | 455 | - | - |
| I+T (fixed) | 53.3 [45.8, 61.4] | 53.3 | 81.0 | 0.78 | 354 | - | - |
| Midterm router | 52.5 [44.4, 60.6] | 52.5 | 85.7 | 0.98 | 362 | -0.8 [-4.2, +2.2] | -5.3 |
| New router | 45.0 [37.5, 52.5] | 45.0 | 57.1 | 1.10 | 595 | -8.3 [-14.2, -2.5] | -46.9 |
| New router, no cheap passes | 53.1 [45.6, 60.8] | 53.1 | 81.0 | 0.73 | 364 | +0.6 [-0.8, +1.8] | +3.4 |
| New router, no human text | 41.4 [34.4, 48.6] | 41.4 | 42.9 | 1.00 | 579 | -5.0 [-10.0, -0.0] | -27.3 |
| Oracle (base7+cells) | 71.1 [64.7, 77.5] | 71.1 | 85.7 | 0.29 | 129 | +30.0 [+24.9, +35.1] | +100.0 |

New - midterm router, cost-adjusted: -7.5 [-13.6, -1.4] pts (sign-flip p = 0.019).

## Files

CSVs: `headline.csv`, `matched_cost.csv`, `comparisons.csv`, `variants.csv`, `learning_curve.csv`, `by_audio_type.csv`, `latency.csv`, `routing_shares.csv`, `calibration.csv`, `midterm_split.csv`, `curves.csv`, `choices.csv`. Figures: `router_curves.pdf/png`, `learning_curve.pdf/png`.
