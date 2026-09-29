# PRGF results (Qwen3-8B, RULER-4K)

PRGF = Partitioned Restore with Global Fusion: RestoreKV's 8 restore tokens split into 7 local slots
(each reads the kept cache + the evicted part of its own context region) and 1 global slot (reads the
full context + all local slots). Same KVzip evictor, same total KV budget, one restore pass.

## Protocol

- Benchmark: RULER-4K (`simonjegou/ruler`, `4096`), the KVPress leaderboard dataset; 13 tasks.
- Metric: kvpress RULER string match (answer contained in the generation; fraction of answers for
  multi-answer tasks), averaged per task, then over the 13 tasks (0-100).
- Pipeline: kvpress `kv-press-text-generation` (pinned commit), greedy decoding, task `max_new_tokens`,
  bf16, SDPA attention, identical for every method.
- Splits (seeded, per task): **dev** = 40 samples/task (520), used for all development decisions;
  **test:50** = 50 samples/task (650) from the remaining rows, never used during development.
- Confidence intervals: paired, task-stratified bootstrap (5000 resamples) of the score difference on
  the same samples.
- Compression ratio (cr) = fraction of KV pairs removed; budget = 1 - cr (restore slots are paid from it).

## Dev (520 samples)

| Method | cr=0.95 (5% budget) | cr=0.90 (10% budget) |
|---|---|---|
| KVzip | 37.36 | 86.97 |
| RestoreKV (official checkpoint) | 72.28 | 90.90 |
| RestoreKV + same extra training (control) | 69.64 | 91.35 |
| PRGF zero-shot (official ckpt, PRGF mask, no training) | 70.12 | 90.76 |
| **PRGF v1** (2000 steps from the official ckpt) | **76.50** | **91.76** |

PRGF v1 - baseline (95% CI):

| | cr=0.95 | cr=0.90 |
|---|---|---|
| vs RestoreKV official | +4.23 [+2.20, +6.25] | +0.86 [-0.14, +1.97] |
| vs control (same data/steps/LR, original mask) | +6.87 [+4.67, +8.98] | +0.41 [-0.37, +1.18] |

At 5% budget the gain comes from the PRGF mask, not from the extra training: the identically trained
control is *below* the official checkpoint (-2.64 [-5.07, -0.36]). At 10% budget all restore variants
are within noise.

PRGF v2 (local slots also read the global slot in the last third of the layers) did not improve over
v1 on dev:20 at cr=0.95 (74.4-75.9 across checkpoints vs 75.4 for v1); v1 is the retained version.

## Held-out test (650 samples, PRGF v1 only)

| cr | budget | PRGF v1 |
|---|---|---|
| 0.95 | 5% (20x) | 75.63 |
| 0.9375 | 6.25% (16x, KVPress benchmark setting) | 84.48 |

The 5% test score (75.63) matches dev (76.50): no sign of overfitting to the development split.

### Against the published KVPress leaderboard (cr = 0.9375, 16x)

The leaderboard ships per-sample predictions for all 6500 RULER-4K rows, so PRGF is compared on the
exact same 650 test rows (paired bootstrap, 95% CI):

| Method | Leaderboard (6500) | Same 650 test rows | PRGF v1 - method |
|---|---|---|---|
| RestoreKV (KVzip scoring) | 81.93 | 83.20 | +1.28 [-0.71, +3.26] |
| RestoreKV+ (KVzip+ scoring, #1) | 86.38 | 86.33 | -1.85 [-3.95, +0.25] |
| **PRGF v1 (KVzip scoring)** | - | **84.48** | |

The like-for-like baseline is RestoreKV (same KVzip scorer). Leaderboard runs used FlashAttention-2 and
the older `qwen3-8b_restorekv.pt` checkpoint; ours use SDPA and the PEFT checkpoint.
The PRGF gain grows as the budget shrinks: +0.9 (10%, dev), +1.3 (6.25%, test vs leaderboard),
+4.2 (5%, dev, significant). PRGF on KVzip+ (init from RestoreKV+) is the natural next step.

## Cost / throughput (A100, per 4K context)

prefill ~340 ms; compression (KVzip scoring + restore pass) 1.12 s RestoreKV vs 1.18 s PRGF;
decoding identical (~75 ms/token, dominated by kvpress' fake-eviction key search).
Training: ~0.36 it/s on one A100-80GB with cached KVzip scores (2000 steps ~ 1.5 h).

## Diagnostics at 16x (why PRGF loses points)

Error analysis on the held-out rows plus counterfactual runs (all at cr = 0.9375):

- Three tasks carry most of the loss vs full cache: cwe (-51), niah_multivalue (-26), niah_single_2 (-20).
- The restore slots carry the needle tasks: on single_2 + multivalue, KVzip alone scores 44.0 and PRGF with its
  slots dropped at decode 40.8, vs 77.2 for RestoreKV on the same rows.
- Needles in the second half of the context are much harder for every method, including KVzip without restore
  (multivalue 46 -> 29, single_2 89 -> 36). Eviction itself is uniform across regions and KVzip chunks (kept
  5.7-6.5 % everywhere, equal chunk scores), and needle-value tokens are kept at the same ~6 % rate as filler,
  before or after the chunk boundary, for solved and failed samples alike. KVzip does not favour needles at 16x;
  answers rely on residual fragments plus the restore slots.
- cwe: max-attention reconstruction scoring splits attention over repeated words, so the frequent words (the
  answer) are evicted first; PRGF recovers 4.6/10 words, with the wrong ones essentially random.
- Ruled out: region partitioning by evicted mass (eviction is uniform, so it equals the current partition) and
  a chunk-dependent eviction bias.

### Partitioned reconstruction distillation (pilot)

Each step adds one evicted-region span per local slot ("Repeat the part ... starting with: <prefix>"), distilled
from the full cache; the error on region j is attributable to slot R_j because only it reads that region's
evicted tokens. 500 steps from PRGF v1 (`recon_weight=1`), dev single_2 + multivalue at cr = 0.9375:

| | multivalue early | multivalue late | single_2 early | single_2 late | average |
|---|---|---|---|---|---|
| PRGF v1 | 91.8 | 57.5 | 100 | 94.4 | 85.31 |
| + reconstruction (500 steps) | 94.5 | 59.8 | 100 | 94.4 | 86.56 |

+1.25 [-0.62, +3.12] (paired bootstrap), 5 samples better / 2 worse: right direction, not yet significant.

## Round 3: reconstruction distillation + chained slots (16x)

Dev (520 samples), cr = 0.9375:

| | RestoreKV | RestoreKV+ | PRGF v1 | v3 (recon) step 1000 | v3 final | v4 (2 slots/region, 2 globals, recon) step 1000 | v4 final |
|---|---|---|---|---|---|---|---|
| average | 81.26 | 84.46 | 83.40 | 84.56 | 84.38 | **84.61** | 83.98 |

Both runs peak around step 1000 and then lose QA accuracy. Held-out test (650 samples), paired vs the leaderboard
predictions on the same rows:

| | RestoreKV | RestoreKV+ | PRGF v1 | v3 step 1000 | **v4 step 1000** |
|---|---|---|---|---|---|
| test:50 | 83.20 | 86.33 | 84.48 | 84.78 | **85.12** |

v4 step 1000 - RestoreKV = +1.92 [+0.25, +3.73]; - RestoreKV+ = -1.21 [-3.21, +0.76] (the gap is cwe: 47 vs 80,
an effect of the KVzip+ scorer); - PRGF v1 = +0.64 [-0.86, +2.21].

## Transport write (conserving merge of evicted KV), zero-shot on v4 step 1000

`prgf/transport.py`: scope-restricted entropic assignment of evicted KV to the 16 slots (cost = expected logit
error under calibrated G = E[qq^T/d] + value error), conserving weighted merge, 2 rounds, slots attend with
+ log(mass). dev:20 (260 samples), cr = 0.9375:

| | average | cwe | fwe | multikey_1 | multivalue | single_2 | qa_1 | qa_2 |
|---|---|---|---|---|---|---|---|---|
| v4 step 1000 (learned slots) | **83.24** | 48.0 | 81.7 | 95.0 | 68.8 | 95.0 | 75.0 | 40.0 |
| + transport write (+ log m) | 65.62 | 18.0 | 70.0 | 60.0 | 48.8 | 60.0 | 50.0 | 25.0 |
| + transport write, no mass bias | 50.94 | 23.0 | 46.7 | 40.0 | 25.0 | 45.0 | 60.0 | 20.0 |

Removing the mass term makes it worse, so the loss is not the log-mass weighting: replacing the learned slot KV by
averages of ~240 evicted tokens per slot and head removes the information the slots carry (the failure mode
anticipated for averaging). Fine-tuning only changes the assignment (through the PRGF initialisation), not the
merged representation, so the pure-merge write was not trained further.
