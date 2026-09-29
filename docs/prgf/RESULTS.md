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
Baselines were not run on the test split (budget), so test numbers are not a paired comparison.
Note: the paper's 86.4 at 16x is RestoreKV on **KVzip+** scoring; PRGF here uses plain KVzip.

## Cost / throughput (A100, per 4K context)

prefill ~340 ms; compression (KVzip scoring + restore pass) 1.12 s RestoreKV vs 1.18 s PRGF;
decoding identical (~75 ms/token, dominated by kvpress' fake-eviction key search).
Training: ~0.36 it/s on one A100-80GB with cached KVzip scores (2000 steps ~ 1.5 h).
