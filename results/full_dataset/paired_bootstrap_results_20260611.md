# Paired Bootstrap Results — exp-01 through exp-05

**Date:** 2026-06-11  
**Script:** `scripts/paired_bootstrap.py`  
**Settings:** 10,000 resamples, seed 0  
**Metric:** 3-class macro F1 (good / neutral / poor)  
**OOV handling:** predictions outside {good, neutral, poor} (e.g. `mixed`, `error`) count as wrong — they add to the true class's false negatives but never contribute to any class's precision.

---

## Per-System Macro F1

| Experiment | Design | Prediction col | OOV preds | Macro F1 | 95% CI |
|:---|:---|:---|---:|---:|:---|
| exp-01 | Single-call baseline | `llm_sentiment` | 0 | 0.6214 | [0.6138, 0.6291] |
| exp-02 | Sequential self-review, critic blind to text | `llm_sentiment` | 3 | 0.6298 | [0.6223, 0.6373] |
| exp-03 | Sequential self-review, critic sees text | `llm_sentiment` | 2 | 0.6291 | [0.6217, 0.6366] |
| exp-04 | Independent critic (two LLMs in parallel) | `final_sentiment` | 2 | 0.6317 | [0.6242, 0.6392] |
| exp-05 | Self-consistency voting (5 parallel calls) | `final_sentiment` | 0 | 0.6295 | [0.6221, 0.6371] |

---

## Pairwise Comparisons

### exp-01 (single-call baseline) vs multi-call designs

| Pair | F1 (A) | F1 (B) | Diff (B−A) | 95% CI on diff | P(B ≤ A) | Significant? |
|:---|---:|---:|---:|:---|---:|:---|
| exp-01 vs exp-02 | 0.6214 | 0.6298 | +0.0084 | [+0.0049, +0.0120] | 0.000 | **Yes** |
| exp-01 vs exp-03 | 0.6214 | 0.6291 | +0.0077 | [+0.0041, +0.0113] | 0.000 | **Yes** |
| exp-01 vs exp-04 | 0.6214 | 0.6317 | +0.0103 | [+0.0069, +0.0138] | 0.000 | **Yes** |
| exp-01 vs exp-05 | 0.6214 | 0.6295 | +0.0081 | [+0.0046, +0.0116] | 0.000 | **Yes** |

### Among multi-call designs (exp-02 through exp-05)

| Pair | F1 (A) | F1 (B) | Diff (B−A) | 95% CI on diff | P(B ≤ A) | Significant? |
|:---|---:|---:|---:|:---|---:|:---|
| exp-02 vs exp-03 | 0.6298 | 0.6291 | −0.0006 | [−0.0032, +0.0018] | 0.694 | No |
| exp-02 vs exp-04 | 0.6298 | 0.6317 | +0.0019 | [−0.0002, +0.0041] | 0.039 | No |
| exp-02 vs exp-05 | 0.6298 | 0.6295 | −0.0002 | [−0.0023, +0.0018] | 0.585 | No |
| exp-03 vs exp-04 | 0.6291 | 0.6317 | +0.0026 | [+0.0002, +0.0050] | 0.016 | **Yes** |
| exp-03 vs exp-05 | 0.6291 | 0.6295 | +0.0004 | [−0.0020, +0.0029] | 0.370 | No |
| exp-04 vs exp-05 | 0.6317 | 0.6295 | −0.0022 | [−0.0041, −0.0003] | 0.988 | **Yes** |

Significance criterion: 95% CI on (B − A) excludes zero.

---

## Interpretation

**Result 1 — Multi-call architectures outperform single-call.** Every multi-call design (exp-02–05) is significantly better than the single-call baseline (exp-01), with p ≈ 0 and 95% CIs entirely above zero across all four comparisons. The gain is consistent at +0.008–0.010 macro F1.

**Result 2 — Best multi-call design: exp-04 (independent critic).** exp-04 achieves the highest macro F1 (0.632) and is the only multi-call design with statistically significant improvements over other multi-call architectures (exp-03 and exp-05). Its key advantage is eliminating anchoring bias: two LLMs independently analyse the same review in parallel from scratch, with no knowledge of each other's output. If they agree, that sentiment is accepted immediately. If they disagree, a third LLM arbitrates using both analyses and the original review text. This design delivers the largest single gain over the baseline (+0.0103, 95% CI [+0.0069, +0.0138]) while remaining efficient — 2 API calls per review on agreement, 3 only when arbitration is needed.
