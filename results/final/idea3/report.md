# Idea 3: item-level budget allocation (1141 items)

Every choice (ridge strength, lambda, the baselines' cells, temperatures) is made inside the training folds; every decision is out of fold. Accuracy is the mean over the 3 option orders; *tokens*: input tokens relative to the full input, including the cheap passes; *ms/item*: end to end on the L4. *vs hull*: accuracy minus random mixing of the 16 budget cells at the same mean token cost (paired over items). 95% CIs: 10,000 bootstrap resamples over items. The first pass b0 = v64_a0.5 costs 0.28 tokens; the two KL passes (v64_a0, v0_a0.5) 0.40 more. The baselines are the midterm's: the most accurate cell in the training folds within the budget, for all items or per audio type (which needs the audio type at test time).

## Headline (operating point chosen for accuracy)

| method | accuracy | macro | cross-modal | tokens | ms/item | vs hull at same cost (pts) | oracle gap closed |
|---|---|---|---|---|---|---|---|
| v768_a1 (fixed) | 42.3 [39.8, 44.8] | 44.6 | 80.7 | 1.00 | 435 | - | - |
| v256_a1 (fixed) | 42.8 [40.3, 45.3] | 44.6 | 66.3 | 0.57 | 268 | - | - |
| v64_a1 (fixed) | 40.5 [38.1, 43.0] | 43.5 | 61.0 | 0.38 | 218 | - | - |
| v64_a0.5 (fixed) | 36.6 [34.3, 39.1] | 40.3 | 47.4 | 0.28 | 203 | - | - |
| Best single cell | 42.0 [39.6, 44.5] | 44.2 | 71.1 | 0.65 | 301 | -0.8 [-1.5, -0.1] | -3.4 |
| Best cell per audio type | 42.1 [39.6, 44.6] | 44.8 | 71.9 | 0.64 | 301 | -0.7 [-1.5, +0.1] | -2.9 |
| Allocator | 42.2 [39.7, 44.6] | 44.3 | 60.6 | 1.19 | 627 | -0.6 [-1.9, +0.6] | -2.8 |
| Allocator, no KL passes | 42.3 [39.9, 44.8] | 44.7 | 70.3 | 0.91 | 462 | -0.5 [-1.1, +0.2] | -2.0 |
| Allocator, no first pass (question + meta) | 42.6 [40.1, 45.2] | 45.0 | 69.5 | 0.65 | 326 | -0.2 [-1.0, +0.6] | -0.8 |
| Allocator, no first pass (meta only) | 42.6 [40.2, 45.2] | 44.8 | 69.5 | 0.61 | 287 | -0.2 [-0.7, +0.4] | -0.8 |
| Allocator, boosted trees | 38.8 [36.5, 41.2] | 41.7 | 52.2 | 1.17 | 619 | -4.0 [-5.8, -2.2] | -17.2 |
| Allocator, log p target | 36.9 [34.7, 39.1] | 39.9 | 46.2 | 0.90 | 513 | -5.9 [-7.9, -3.9] | -25.5 |
| Allocator, + audio type | 41.3 [38.8, 43.7] | 44.0 | 62.2 | 1.24 | 647 | -1.5 [-2.9, -0.2] | -6.6 |
| Oracle (sees the answers) | 66.1 [63.9, 68.3] | 67.1 | 86.3 | 0.24 | 131 | +31.3 [+29.3, +33.2] | +100.0 |

## At a token budget (chosen in the training folds)

The allocator cannot cost less than its cheap passes; the baselines take the best cell within the budget, so they can come in under it.

| budget | method | accuracy | macro | cross-modal | tokens | ms/item | vs hull at same cost (pts) | oracle gap closed |
|---|---|---|---|---|---|---|---|---|
| 0.35 | Best single cell | 36.6 [34.3, 39.1] | 40.3 | 47.4 | 0.28 | 203 | +0.0 [+0.0, +0.0] | +0.0 |
| 0.35 | Best cell per audio type | 37.4 [35.0, 39.8] | 41.2 | 48.2 | 0.29 | 202 | +0.3 [-0.5, +1.0] | +0.9 |
| 0.35 | Allocator | 37.6 [35.3, 40.0] | 41.2 | 42.2 | 0.68 | 411 | -5.2 [-7.2, -3.2] | -22.3 |
| 0.35 | Allocator, no KL passes | 37.8 [35.4, 40.2] | 41.1 | 50.6 | 0.33 | 232 | -0.8 [-1.6, -0.0] | -3.0 |
| 0.35 | Allocator, no first pass (question + meta) | 38.5 [36.1, 41.0] | 40.2 | 58.2 | 0.34 | 228 | -0.5 [-1.6, +0.6] | -1.9 |
| 0.5 | Best single cell | 40.5 [38.1, 43.0] | 43.5 | 61.0 | 0.38 | 218 | +0.0 [+0.0, +0.0] | +0.0 |
| 0.5 | Best cell per audio type | 40.3 [37.9, 42.8] | 42.9 | 60.2 | 0.37 | 213 | +0.4 [-0.2, +0.9] | +1.4 |
| 0.5 | Allocator | 37.6 [35.3, 40.0] | 41.2 | 42.2 | 0.68 | 411 | -5.2 [-7.2, -3.2] | -22.3 |
| 0.5 | Allocator, no KL passes | 39.4 [37.0, 41.8] | 42.2 | 54.6 | 0.47 | 295 | -2.3 [-3.4, -1.1] | -9.2 |
| 0.5 | Allocator, no first pass (question + meta) | 41.0 [38.6, 43.5] | 43.2 | 65.1 | 0.51 | 276 | -1.1 [-1.9, -0.3] | -4.6 |
| 0.7 | Best single cell | 42.8 [40.3, 45.3] | 44.6 | 66.3 | 0.57 | 268 | +0.0 [+0.0, +0.0] | +0.0 |
| 0.7 | Best cell per audio type | 42.3 [39.8, 44.8] | 43.3 | 66.3 | 0.53 | 257 | -0.1 [-0.7, +0.4] | -0.6 |
| 0.7 | Allocator | 37.8 [35.6, 40.2] | 41.4 | 42.6 | 0.69 | 420 | -5.0 [-7.0, -3.0] | -21.3 |
| 0.7 | Allocator, no KL passes | 41.1 [38.7, 43.6] | 43.1 | 61.4 | 0.72 | 395 | -1.7 [-2.7, -0.6] | -7.2 |
| 0.7 | Allocator, no first pass (question + meta) | 42.7 [40.3, 45.3] | 45.4 | 69.1 | 0.66 | 331 | -0.1 [-0.9, +0.7] | -0.3 |

## Research questions (A - B, each side minus the hull at its own cost; paired; Holm per point)

|  | comparison | point | A: acc / tokens | B: acc / tokens | A - B (pts) | p_holm |
|---|---|---|---|---|---|---|
| RQ3.1 | allocator vs the best single cell | accuracy-tuned | 42.2 / 1.19 | 42.0 / 0.65 | +0.1 [-1.1, +1.4] | 1.000 |
| RQ3.1 | allocator vs the best single cell | budget 0.5 | 37.6 / 0.68 | 40.5 / 0.38 | -5.2 [-7.2, -3.2] | 0.001 |
| RQ3.1 | allocator vs the best cell per audio type | accuracy-tuned | 42.2 / 1.19 | 42.1 / 0.64 | +0.0 [-1.2, +1.3] | 1.000 |
| RQ3.1 | allocator vs the best cell per audio type | budget 0.5 | 37.6 / 0.68 | 40.3 / 0.37 | -5.6 [-7.7, -3.4] | 0.001 |
| RQ3.1 | per audio type vs single cell (the midterm comparison) | accuracy-tuned | 42.1 / 0.64 | 42.0 / 0.65 | +0.1 [-0.4, +0.7] | 1.000 |
| RQ3.1 | per audio type vs single cell (the midterm comparison) | budget 0.5 | 40.3 / 0.37 | 40.5 / 0.38 | +0.4 [-0.2, +0.9] | 0.207 |
| RQ3.2 | adding the KL passes and features | accuracy-tuned | 42.2 / 1.19 | 42.3 / 0.91 | -0.2 [-1.4, +1.1] | 1.000 |
| RQ3.2 | adding the KL passes and features | budget 0.5 | 37.6 / 0.68 | 39.4 / 0.47 | -2.9 [-4.8, -1.1] | 0.010 |
| first pass | a first pass vs question + meta features only | accuracy-tuned | 42.3 / 0.91 | 42.6 / 0.65 | -0.3 [-1.1, +0.5] | 1.000 |
| first pass | a first pass vs question + meta features only | budget 0.5 | 39.4 / 0.47 | 41.0 / 0.51 | -1.2 [-2.5, +0.2] | 0.187 |
| regressor | boosted trees vs ridge | accuracy-tuned | 38.8 / 1.17 | 42.2 / 1.19 | -3.4 [-5.1, -1.7] | 0.002 |
| regressor | boosted trees vs ridge | budget 0.5 | 36.3 / 0.68 | 37.6 / 0.68 | -1.3 [-2.6, -0.1] | 0.146 |
| target | predicting the log p gain vs the accuracy gain | accuracy-tuned | 36.9 / 0.90 | 42.2 / 1.19 | -5.3 [-7.1, -3.4] | 0.001 |
| target | predicting the log p gain vs the accuracy gain | budget 0.5 | 35.1 / 0.68 | 37.6 / 0.68 | -2.5 [-4.0, -1.0] | 0.008 |

## Allocator variants

*Added area*: accuracy (pts) the allocator's operating points add to the cell frontier over costs 0.2-0.8. *Spearman*: per item, predicted vs actual gain over the cells.

| allocator | acc (tokens), accuracy-tuned | vs hull, accuracy-tuned | vs hull @ 0.35 (tokens) | vs hull @ 0.5 (tokens) | vs hull @ 0.7 (tokens) | added area | oracle gap closed | Spearman | alpha (folds) |
|---|---|---|---|---|---|---|---|---|---|
| ridge · first pass v64_a0.5+KL+meta | 42.2 (1.19) | -0.6 [-1.9, +0.6] | -5.2 (0.68) | -5.2 (0.68) | -5.0 (0.69) | +0.0 | -2.8 | 0.195 | 0.1x4, 1x1 |
| ridge · first pass v64_a0.5+meta | 42.3 (0.91) | -0.5 [-1.1, +0.2] | -0.8 (0.33) | -2.3 (0.47) | -1.7 (0.72) | +0.0 | -2.0 | 0.166 | 1000x5 |
| ridge · no first pass+question+meta | 42.6 (0.65) | -0.2 [-1.0, +0.6] | -0.5 (0.34) | -1.1 (0.51) | -0.1 (0.66) | +0.3 | -0.8 | 0.164 | 1000x5 |
| ridge · no first pass+meta | 42.6 (0.61) | -0.2 [-0.7, +0.4] | +0.6 (0.34) | +0.2 (0.53) | -0.1 (0.63) | +0.3 | -0.8 | 0.166 | 1000x5 |
| gbt · first pass v64_a0.5+KL+meta | 38.8 (1.17) | -4.0 [-5.8, -2.2] | -6.5 (0.68) | -6.5 (0.68) | -6.5 (0.70) | +0.0 | -17.2 | 0.138 | - |
| ridge · first pass v64_a0.5+KL+meta · target log p | 36.9 (0.90) | -5.9 [-7.9, -3.9] | -7.7 (0.68) | -7.7 (0.68) | -8.0 (0.70) | +0.0 | -25.5 | 0.205 | 1x3, 0.1x2 |
| ridge · first pass v64_a0.5+KL+meta+audio type | 41.3 (1.24) | -1.5 [-2.9, -0.2] | -5.2 (0.68) | -5.2 (0.68) | -4.9 (0.69) | +0.0 | -6.6 | 0.197 | 0.1x4, 1x1 |

### Sweeps: the first-pass budget b0 (no KL passes) and the ridge strength

| allocator | acc (tokens), accuracy-tuned | vs hull, accuracy-tuned | vs hull @ 0.35 (tokens) | vs hull @ 0.5 (tokens) | vs hull @ 0.7 (tokens) | added area | oracle gap closed | Spearman | alpha (folds) |
|---|---|---|---|---|---|---|---|---|---|
| ridge · first pass v64_a0.25+meta | 42.2 (0.87) | -0.6 [-1.3, +0.1] | -2.5 (0.34) | -2.7 (0.49) | -1.3 (0.71) | +0.0 | -2.5 | 0.166 | 1000x4, 100x1 |
| ridge · first pass v0_a0.5+meta | 42.7 (0.82) | -0.1 [-0.6, +0.4] | -3.3 (0.35) | -3.3 (0.47) | -0.7 (0.72) | +0.0 | -0.5 | 0.170 | 1000x5 |
| ridge · first pass v256_a0.5+meta | 42.4 (1.09) | -0.4 [-0.9, +0.2] | -3.0 (0.46) | -3.0 (0.47) | -2.7 (0.71) | +0.0 | -1.5 | 0.166 | 1000x5 |
| ridge · first pass v64_a0.5+KL+meta · alpha=0.1 | 42.2 (1.20) | -0.6 [-1.8, +0.6] | -5.2 (0.68) | -5.2 (0.68) | -5.0 (0.69) | +0.0 | -2.5 | 0.196 | 0.1x5 |
| ridge · first pass v64_a0.5+KL+meta · alpha=1 | 42.1 (1.22) | -0.7 [-2.0, +0.5] | -5.6 (0.68) | -5.6 (0.68) | -5.2 (0.69) | +0.0 | -3.1 | 0.192 | 1x5 |
| ridge · first pass v64_a0.5+KL+meta · alpha=10 | 42.0 (1.17) | -0.8 [-2.0, +0.4] | -5.1 (0.68) | -5.1 (0.68) | -4.8 (0.69) | +0.0 | -3.4 | 0.192 | 10x5 |
| ridge · first pass v64_a0.5+KL+meta · alpha=100 | 41.7 (1.22) | -1.1 [-2.2, -0.0] | -6.0 (0.68) | -6.0 (0.68) | -5.6 (0.70) | +0.0 | -4.6 | 0.187 | 100x5 |
| ridge · first pass v64_a0.5+KL+meta · alpha=1000 | 42.1 (1.28) | -0.7 [-1.5, +0.0] | -5.9 (0.68) | -5.9 (0.68) | -5.3 (0.69) | +0.0 | -3.0 | 0.182 | 1000x5 |

## The cheap passes: when are they confidently wrong? (plan §6.4a)

One pass in the original option order. *Confidently wrong*: top probability >= 0.8 and wrong. *Scaled*: temperature fitted on the other folds; the fitted temperatures are large (the passes are overconfident), so after scaling no answer reaches 0.8. By task type: cheap_pass_failures.csv.

| pass | group | items | accuracy | mean max prob | ECE | ECE scaled | confidently wrong | confidently wrong, scaled |
|---|---|---|---|---|---|---|---|---|
| v64_a0.5 | all | 1141 | 35.1 | 68.8 | 33.7 | 1.3 | 20.3 | 0.0 |
| v64_a0.5 | speech | 771 | 32.3 | 68.2 | 36.0 | 2.6 | 21.8 | 0.0 |
| v64_a0.5 | sound event | 265 | 35.5 | 70.9 | 35.5 | 3.5 | 21.5 | 0.0 |
| v64_a0.5 | music | 105 | 54.3 | 67.2 | 14.2 | 20.2 | 6.7 | 0.0 |
| v64_a0.5 | cross-modal | 83 | 42.2 | 63.9 | 21.8 | 9.2 | 10.8 | 0.0 |
| v64_a0 | all | 1141 | 29.4 | 70.8 | 41.4 | 3.7 | 28.0 | 0.0 |
| v64_a0 | speech | 771 | 27.4 | 70.1 | 42.7 | 1.9 | 27.4 | 0.0 |
| v64_a0 | sound event | 265 | 25.3 | 74.4 | 49.1 | 7.7 | 38.9 | 0.0 |
| v64_a0 | music | 105 | 54.3 | 66.7 | 14.3 | 25.8 | 4.8 | 0.0 |
| v64_a0 | cross-modal | 83 | 13.3 | 68.3 | 55.1 | 15.7 | 31.3 | 0.0 |
| v0_a0.5 | all | 1141 | 26.7 | 61.1 | 34.4 | 2.1 | 16.5 | 0.0 |
| v0_a0.5 | speech | 771 | 23.3 | 58.6 | 35.4 | 4.6 | 14.9 | 0.0 |
| v0_a0.5 | sound event | 265 | 24.9 | 67.0 | 42.1 | 4.0 | 23.8 | 0.0 |
| v0_a0.5 | music | 105 | 56.2 | 65.0 | 17.2 | 27.5 | 9.5 | 0.0 |
| v0_a0.5 | cross-modal | 83 | 13.3 | 63.9 | 50.7 | 15.2 | 27.7 | 0.0 |
| v64_a0.25 | all | 1141 | 32.9 | 68.2 | 35.4 | 1.4 | 20.2 | 0.0 |
| v64_a0.25 | speech | 771 | 29.8 | 67.7 | 38.0 | 2.7 | 20.8 | 0.0 |
| v64_a0.25 | sound event | 265 | 32.8 | 70.5 | 37.7 | 2.1 | 23.8 | 0.0 |
| v64_a0.25 | music | 105 | 55.2 | 66.1 | 12.2 | 23.3 | 7.6 | 0.0 |
| v64_a0.25 | cross-modal | 83 | 36.1 | 62.0 | 27.0 | 10.5 | 15.7 | 0.0 |
| v256_a0.5 | all | 1141 | 38.2 | 70.5 | 32.3 | 3.6 | 23.3 | 0.0 |
| v256_a0.5 | speech | 771 | 36.2 | 69.8 | 34.2 | 4.3 | 23.9 | 0.0 |
| v256_a0.5 | sound event | 265 | 37.4 | 73.5 | 36.1 | 4.3 | 27.2 | 0.0 |
| v256_a0.5 | music | 105 | 55.2 | 67.8 | 17.1 | 19.1 | 9.5 | 0.0 |
| v256_a0.5 | cross-modal | 83 | 38.6 | 64.6 | 27.0 | 11.7 | 16.9 | 0.0 |

**v64_a0.5 by task type**

| group | items | accuracy | confidently wrong |
|---|---|---|---|
| task: Object Identification and Description | 211 | 30.3 | 28.0 |
| task: Action and Activity | 250 | 33.2 | 23.2 |
| task: Identity and Relationship | 32 | 31.2 | 18.8 |
| task: Contextual and Environmental Questions | 141 | 56.0 | 17.7 |
| task: Story Description | 230 | 34.3 | 17.4 |
| task: Plot Inference | 237 | 33.3 | 16.5 |
| task: Count and Quantity | 15 | 20.0 | 13.3 |
| task: Text and Symbols | 25 | 12.0 | 12.0 |

## Time per representation (L4, mean per item)

| representation | accuracy | tokens | forward_ms | preprocess_ms | read_media_ms | end_to_end_ms |
|---|---|---|---|---|---|---|
| v0_a0 | 24.6 | 0.12 | 39 | 1 | 35 | 76 |
| v0_a0.25c | 29.2 | 0.17 | 59 | 53 | 35 | 147 |
| v0_a0.25 | 27.9 | 0.17 | 59 | 51 | 35 | 145 |
| v0_a0.25e | 28.8 | 0.17 | 59 | 53 | 35 | 148 |
| v64_a0 | 29.5 | 0.17 | 76 | 11 | 35 | 122 |
| v0_a0.5c | 30.5 | 0.22 | 66 | 53 | 35 | 153 |
| v0_a0.5 | 30.3 | 0.22 | 66 | 55 | 35 | 156 |
| v0_a0.5e | 30.5 | 0.22 | 66 | 53 | 35 | 154 |
| v64_a0.25e | 35.1 | 0.23 | 98 | 63 | 35 | 196 |
| v64_a0.25 | 34.4 | 0.23 | 98 | 60 | 35 | 192 |
| v64_a0.5 | 36.6 | 0.28 | 105 | 63 | 35 | 204 |
| v64_a0.5e | 37.8 | 0.28 | 105 | 63 | 35 | 203 |
| v0_a1 | 32.0 | 0.32 | 83 | 48 | 35 | 166 |
| cap_trans | 41.0 | 0.35 | 62 | 2 | 35 | 99 |
| v256_a0 | 32.7 | 0.36 | 115 | 14 | 35 | 164 |
| v64_a1 | 40.5 | 0.38 | 123 | 61 | 35 | 218 |
| v256_a0.25 | 37.2 | 0.41 | 137 | 63 | 35 | 234 |
| v256_a0.5 | 38.5 | 0.46 | 146 | 63 | 35 | 244 |
| cap_aud | 37.4 | 0.52 | 108 | 48 | 35 | 191 |
| v256_a1 | 42.8 | 0.57 | 169 | 64 | 35 | 267 |
| v768_a0 | 32.2 | 0.79 | 286 | 20 | 35 | 342 |
| img_trans | 46.6 | 0.82 | 292 | 20 | 35 | 347 |
| v768_a0.25 | 35.8 | 0.85 | 303 | 69 | 35 | 407 |
| v768_a0.5 | 37.8 | 0.90 | 312 | 70 | 35 | 416 |
| v768_a1 | 42.3 | 1.00 | 334 | 66 | 35 | 435 |

## Which part of the audio to keep (RQ3.3)

| window | budget cell | difference (pts) | speech | sound event | music | p_holm |
|---|---|---|---|---|---|---|
| centered vs leading | v0_a0.25 | +1.3 [-0.2, +2.9] | +2.2 | -1.1 | +1.3 | 0.524 |
| loudest vs leading | v0_a0.25 | +0.9 [-0.4, +2.2] | +1.0 | +0.4 | +1.6 | 0.807 |
| centered vs leading | v0_a0.5 | +0.2 [-1.3, +1.7] | +0.2 | +0.6 | -1.0 | 1.000 |
| loudest vs leading | v0_a0.5 | +0.2 [-1.1, +1.5] | +0.7 | +0.1 | -3.2 | 1.000 |
| loudest vs leading | v64_a0.25 | +0.7 [-0.4, +1.9] | +0.9 | +0.8 | -0.6 | 0.807 |
| loudest vs leading | v64_a0.5 | +1.2 [-0.0, +2.4] | +1.2 | +1.3 | +1.0 | 0.368 |

## The midterm test set (90 items; trained on the other 1051)

*midterm*: results/idea3/summary.json (dev 90 / test 90, Mac); *this run*: the L4 bank, trained on every other item. The allocator cannot go below its cheap passes.

| budget | policy | midterm acc | midterm tokens | this run | tokens |
|---|---|---|---|---|---|
| 0.25 | global | 33.0 | 0.23 | 40.4 [31.9, 49.3] | 0.23 |
| 0.25 | per audio type | 31.9 | 0.19 | 34.1 [26.3, 41.9] | 0.21 |
| 0.25 | allocator | - | - | 43.3 [34.4, 52.2] | 0.69 |
| 0.4 | global | 47.0 | 0.40 | 47.8 [39.3, 56.3] | 0.40 |
| 0.4 | per audio type | 37.4 | 0.29 | 45.9 [37.0, 54.8] | 0.32 |
| 0.4 | allocator | - | - | 43.3 [34.4, 52.2] | 0.69 |
| 0.6 | global | 47.4 | 0.58 | 48.9 [39.6, 57.8] | 0.58 |
| 0.6 | per audio type | 46.3 | 0.38 | 47.4 [38.5, 56.3] | 0.45 |
| 0.6 | allocator | - | - | 43.3 [34.4, 52.2] | 0.69 |
| 0.8 | global | 47.4 | 0.58 | 48.9 [39.6, 57.8] | 0.58 |
| 0.8 | per audio type | 47.4 | 0.49 | 47.4 [38.5, 56.3] | 0.45 |
| 0.8 | allocator | - | - | 45.9 [37.4, 54.8] | 0.83 |

## Files

CSVs: `headline.csv`, `matched_cost.csv`, `comparisons.csv`, `variants.csv`, `cheap_pass_failures.csv`, `time_per_cell.csv`, `audio_selection.csv`, `midterm_test.csv`, `curves.csv`. Figures: `allocator_curves.pdf/png`, `reliability.pdf/png`.
