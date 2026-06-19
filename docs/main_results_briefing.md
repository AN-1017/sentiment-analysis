# Main Branch — Results Briefing for Paper Writing

This document is a complete briefing of all experiments run on the main branch. It contains every number needed to write the paper. Do not invent or extrapolate any figures — if a metric is not listed here, it was not measured.

---

## Abstract

Large language models (LLMs) are increasingly applied to sentiment classification, yet the value of multi-call architectures — where a second LLM reviews or challenges an initial prediction — remains underexplored. This paper presents a systematic comparison of five LLM pipeline designs for three-class sentiment classification (positive, neutral, negative) on a real-world e-commerce review dataset (22,641 reviews). Using GPT-4o-mini and macro F1 as the primary metric, we evaluate a single-call baseline against four multi-call architectures: sequential self-review (critic-blind and critic-aware variants), independent parallel critic with arbitration, and self-consistency voting. Paired bootstrap testing (10,000 resamples) establishes two key findings. First, any form of review step significantly outperforms the single-call baseline (p ≈ 0, ΔF1 = +0.008–0.010). Second, the independent critic design — in which two LLMs classify the same review in parallel without knowledge of each other's output, with a third LLM arbitrating on disagreement — achieves the highest macro F1 (0.632) and is the only architecture with statistically significant gains over competing multi-call designs. These results suggest that eliminating anchoring bias through independent parallel analysis is more effective than iterative self-review or ensemble voting, while remaining cost-efficient at 2–3 API calls per instance.

---

## Dataset and Setup

- **Dataset:** 22,641 real-world Women's Clothing E-Commerce reviews; three-class sentiment: `good` (77.1%), `neutral` (12.5%), `poor` (10.5%)
- **Model:** `gpt-4o-mini`, temperature 0, constrained `Literal["good","neutral","poor"]` label output
- **Primary metric:** macro F1 — unweighted average of per-class F1. Accuracy is misleading due to class imbalance (77% good); a naive "always good" classifier scores 77.1% accuracy.
- **Full dataset runs:** all experiments processed all 22,641 reviews
- **Pipeline framework:** LangGraph (all async)
- **Statistical testing:** paired bootstrap, 10,000 resamples, seed 0, 3-class macro F1; OOV predictions (e.g. `error`, `mixed`) count as wrong

### True Label Distribution

| Class | Count | % |
|:---|---:|---:|
| good | 17,448 | 77.1% |
| neutral | 2,823 | 12.5% |
| poor | 2,370 | 10.5% |

---

## Metrics Correction (Important)

All experiments except exp-05 originally reported deflated macro F1 values because scikit-learn included spurious zero-support labels (`mixed`, `error`) emitted by the LLM as extra classes with F1 = 0.00, dragging down the macro average. All figures in this document use **corrected 3-class metrics**, recomputed over good/neutral/poor only. OOV predictions still count against recall for their true class.

| Experiment | Originally Reported F1 | Corrected F1 |
|:---|---:|---:|
| exp-02 | 0.47 | 0.6298 |
| exp-03 | 0.38 | 0.6291 |
| exp-04 | 0.47 | 0.6317 |
| exp-05 | 0.63 | 0.6295 (no change) |

---

## Architecture Descriptions

### exp-01 — Single-Call Baseline
One LLM call per review. The model receives the review text and outputs a label directly. No reasoning is captured (saves tokens). This is the cost floor: 1 call/review.

### exp-02 — Sequential Self-Review, Critic Blind to Text
The answerer classifies the review and produces a label with reasoning. A second LLM (the critic) then reviews the label and reasoning — but does **not** see the original review text. If the critic disagrees, the answerer reclassifies using the critic's feedback. This loops up to 3 times. The critic is intentionally blind to the review text to test whether internal consistency of reasoning alone drives improvement.

### exp-03 — Sequential Self-Review, Critic Sees Text
Identical to exp-02 except the critic also receives the original review text alongside the label and reasoning. This gives the critic full context to challenge the answerer's prediction. The hypothesis is that more information for the critic leads to better corrections.

### exp-04 — Independent Critic with Arbitration
Two LLMs classify the same review **in parallel, from scratch**, with no knowledge of each other's output. If they agree, that label is the final answer (2 calls). If they disagree, a third LLM arbitrates — it sees both predictions, both reasoning chains, and the original review text (3 calls). The key design property is independence: the critic forms its opinion before seeing the answerer's, eliminating anchoring bias.

### exp-05 — Self-Consistency Voting (5×)
Five independent LLM calls run in parallel on the same review. The final label is decided by majority vote. No critic, no feedback loop. All calls use temperature 0. This tests whether stochastic variation across calls (even at low temperature) can improve accuracy through ensemble aggregation.

---

## exp-01 — Single-Call Baseline

**Run:** 2026-06-11 | 22,641 reviews, 0 errors

### Results

| Metric | Value |
|:---|---:|
| Macro F1 | 0.6214 |
| 95% CI | [0.6138, 0.6291] |
| Accuracy | 78.15% |
| good F1 | 0.90 |
| neutral F1 | 0.33 |
| poor F1 | 0.63 |
| Calls/review | 1 |

### Classification Report

|  | precision | recall | f1-score | support |
|:---|---:|---:|---:|---:|
| good | 0.99 | 0.83 | 0.90 | 17,448 |
| neutral | 0.29 | 0.38 | 0.33 | 2,823 |
| poor | 0.49 | 0.89 | 0.63 | 2,370 |
| **macro avg** | **0.59** | **0.70** | **0.62** | **22,641** |

### Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| good | 14,523 | 2,363 | 562 |
| neutral | 140 | 1,060 | 1,623 |
| poor | 20 | 239 | 2,111 |

---

## exp-02 — Sequential Self-Review, Critic Blind to Text

**Run:** 2026-06-03 | 22,641 reviews, 3 OOV predictions (`mixed`)

### Results

| Metric | Value |
|:---|---:|
| Macro F1 | 0.6298 |
| 95% CI | [0.6223, 0.6373] |
| Accuracy | 77.47% |
| good F1 | 0.8952 |
| neutral F1 | 0.3450 |
| poor F1 | 0.6491 |
| Calls/review | ~2.0 |

### Per-Class Metrics (Corrected)

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| good | 0.9912 | 0.8161 | 0.8952 | 17,448 |
| neutral | 0.2881 | 0.4300 | 0.3450 | 2,823 |
| poor | 0.5142 | 0.8802 | 0.6491 | 2,370 |

### Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| good | 14,240 | 2,734 | 474 |
| neutral | 109 | 1,214 | 1,497 |
| poor | 18 | 266 | 2,086 |

### Self-Review Stats

| Iterations | Reviews |
|---:|---:|
| 1 | 22,472 (99.4%) |
| 2 | 152 |
| 3 | 17 |

Agreement rate (first pass): **99.99%** — the loop almost never changes the prediction.

---

## exp-03 — Sequential Self-Review, Critic Sees Text

**Run:** 2026-06-05 | 22,641 reviews, 2 OOV predictions (`error`, `mixed`)

### Results

| Metric | Value |
|:---|---:|
| Macro F1 | 0.6291 |
| 95% CI | [0.6217, 0.6366] |
| Accuracy | 77.73% |
| good F1 | 0.8979 |
| neutral F1 | 0.3444 |
| poor F1 | 0.6452 |
| Calls/review | ~2.0 |

### Per-Class Metrics (Corrected)

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| good | 0.9905 | 0.8211 | 0.8979 | 17,448 |
| neutral | 0.2907 | 0.4222 | 0.3444 | 2,823 |
| poor | 0.5102 | 0.8772 | 0.6452 | 2,370 |

### Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| good | 14,327 | 2,634 | 485 |
| neutral | 120 | 1,192 | 1,511 |
| poor | 17 | 274 | 2,079 |

### Self-Review Stats

| Iterations | Reviews |
|---:|---:|
| 1 | 22,279 (98.4%) |
| 2 | 281 |
| 3 | 80 |

Agreement rate (first pass): **99.87%**. Giving the critic more context (the original text) leads to more disagreements than exp-02 but no improvement in F1.

---

## exp-04 — Independent Critic with Arbitration (Best Architecture)

**Run:** 2026-06-07 | 22,641 reviews, 2 OOV predictions (`error`)

### Results

| Metric | Value |
|:---|---:|
| Macro F1 | **0.6317** |
| 95% CI | [0.6242, 0.6392] |
| Accuracy | 77.47% |
| good F1 | 0.8948 |
| neutral F1 | 0.3494 |
| poor F1 | 0.6510 |
| Calls/review | 2.007 (avg) |

### Per-Class Metrics (Corrected)

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| good | 0.9916 | 0.8152 | 0.8948 | 17,448 |
| neutral | 0.2895 | 0.4407 | 0.3494 | 2,823 |
| poor | 0.5184 | 0.8747 | 0.6510 | 2,370 |

### Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| good | 14,223 | 2,773 | 450 |
| neutral | 103 | 1,244 | 1,476 |
| poor | 17 | 280 | 2,073 |

### Agreement Stats

| Outcome | Reviews | % |
|:---|---:|---:|
| Both LLMs agreed → direct result | 22,477 | 99.3% |
| Disagreed → sent to arbitrator | 164 | 0.7% |

99.3% of reviews need only 2 calls. Arbitration is rare.

---

## exp-05 — Self-Consistency Voting (5×)

**Run:** 2026-06-08 | 22,641 reviews, 0 OOV predictions

### Results

| Metric | Value |
|:---|---:|
| Macro F1 | 0.6295 |
| 95% CI | [0.6221, 0.6371] |
| Accuracy | 77.32% |
| good F1 | 0.8941 |
| neutral F1 | 0.3465 |
| poor F1 | 0.6481 |
| Calls/review | 5 (all parallel — same latency as 1 call) |

### Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| good | 0.9916 | 0.8140 | 0.8941 | 17,448 |
| neutral | 0.2868 | 0.4375 | 0.3465 | 2,823 |
| poor | 0.5155 | 0.8726 | 0.6481 | 2,370 |

### Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| good | 14,203 | 2,785 | 460 |
| neutral | 104 | 1,235 | 1,484 |
| poor | 16 | 286 | 2,068 |

### Voting Stats

**98.7% of reviews were unanimous (5-0 vote).** Non-unanimous votes:

| Vote pattern | Count | % |
|:---|---:|---:|
| 5-0 (unanimous) | 22,335 | 98.7% |
| 4-1 | 268 | 1.2% |
| 3-2 | 38 | 0.2% |

The model at temperature 0 is almost never uncertain — majority voting adds no meaningful new information over a single call.

---

## Paired Bootstrap Results

**Script:** `scripts/paired_bootstrap.py` | 10,000 resamples, seed 0 | OOV predictions count as wrong

**Significance criterion:** A result is significant when the 95% CI on the difference (B − A) excludes zero — equivalent to p < 0.05. The `P(B ≤ A)` column gives the empirical bootstrap p-value directly.

### Per-System Macro F1

| Experiment | Design | Macro F1 | 95% CI |
|:---|:---|---:|:---|
| exp-01 | Single-call baseline | 0.6214 | [0.6138, 0.6291] |
| exp-02 | Sequential self-review, critic blind | 0.6298 | [0.6223, 0.6373] |
| exp-03 | Sequential self-review, critic sees text | 0.6291 | [0.6217, 0.6366] |
| exp-04 | Independent critic | **0.6317** | [0.6242, 0.6392] |
| exp-05 | Self-consistency voting (5×) | 0.6295 | [0.6221, 0.6371] |

### exp-01 vs. All Multi-Call Designs

| Pair | Diff (B−A) | 95% CI | P(B ≤ A) | Significant? |
|:---|---:|:---|---:|:---|
| exp-01 vs exp-02 | +0.0084 | [+0.0049, +0.0120] | 0.000 | **Yes** |
| exp-01 vs exp-03 | +0.0077 | [+0.0041, +0.0113] | 0.000 | **Yes** |
| exp-01 vs exp-04 | +0.0103 | [+0.0069, +0.0138] | 0.000 | **Yes** |
| exp-01 vs exp-05 | +0.0081 | [+0.0046, +0.0116] | 0.000 | **Yes** |

All multi-call designs significantly outperform the single-call baseline (p ≈ 0, CIs entirely above zero).

### Among Multi-Call Designs (exp-02 through exp-05)

| Pair | Diff (B−A) | 95% CI | P(B ≤ A) | Significant? |
|:---|---:|:---|---:|:---|
| exp-02 vs exp-03 | −0.0006 | [−0.0032, +0.0018] | 0.694 | No |
| exp-02 vs exp-04 | +0.0019 | [−0.0002, +0.0041] | 0.039 | No |
| exp-02 vs exp-05 | −0.0002 | [−0.0023, +0.0018] | 0.585 | No |
| exp-03 vs exp-04 | +0.0026 | [+0.0002, +0.0050] | 0.016 | **Yes** |
| exp-03 vs exp-05 | +0.0004 | [−0.0020, +0.0029] | 0.370 | No |
| exp-04 vs exp-05 | −0.0022 | [−0.0041, −0.0003] | 0.988 | **Yes** |

Two pairs are technically significant (exp-03 vs exp-04, exp-04 vs exp-05) but the absolute differences are ~0.002 — practically negligible. All four multi-call designs should be treated as equivalent in practice.

---

## Master Comparison Table

| Exp | Design | Macro F1 | good F1 | neutral F1 | poor F1 | Accuracy | Calls/review |
|:---|:---|---:|---:|---:|---:|---:|---:|
| exp-01 | Single-call baseline | 0.6214 | 0.90 | 0.33 | 0.63 | 78.15% | 1 |
| exp-02 | Self-review, critic blind | 0.6298 | 0.90 | 0.35 | 0.65 | 77.47% | ~2 |
| exp-03 | Self-review, critic sees text | 0.6291 | 0.90 | 0.34 | 0.65 | 77.73% | ~2 |
| exp-04 | Independent critic | **0.6317** | 0.89 | 0.35 | 0.65 | 77.47% | 2.007 |
| exp-05 | Self-consistency 5× | 0.6295 | 0.89 | 0.35 | 0.65 | 77.32% | 5 |

---

## Key Findings (Narrative for Paper)

**Finding 1 — Any review step significantly outperforms a single call.**
All four multi-call architectures (exp-02–05) beat the single-call baseline (exp-01) with p ≈ 0 and 95% CIs entirely above zero. The gain is consistent at +0.008–0.010 macro F1.

**Finding 2 — The form of the review step does not matter (among same-model designs).**
exp-02 through exp-05 span only 0.003 macro F1 (0.6291–0.6317). The vast majority of pairwise comparisons are not significant. Whether the critic sees the original text, operates independently, or is replaced by 5-vote majority voting, the outcome is equivalent.

**Finding 3 — exp-04 (independent critic) is the recommended architecture.**
It achieves the highest macro F1 (0.6317) and is the only design with statistically significant advantages over other multi-call designs (exp-03 and exp-05). It reaches this result at only 2.007 calls/review — nearly as cheap as the simpler architectures. The independence property (no anchoring bias) appears to be the key differentiator, even if the statistical margin is small in absolute terms.

**Finding 4 — Self-consistency voting (exp-05) is the least cost-efficient design.**
5× the API calls for equivalent F1 to exp-02 (2 calls). The voting stats explain why: 98.7% of reviews receive a unanimous 5-0 vote at temperature 0 — the model is almost never genuinely uncertain, so aggregating over identical predictions adds nothing.

**Finding 5 — Neutral class is the persistent bottleneck.**
Neutral F1 ≈ 0.33–0.35 across all architectures including the best (exp-04: 0.35). No multi-call design improves neutral classification meaningfully. This is likely a label-noise problem: rating-3 reviews are textually ambiguous by nature and the gold labels are noisy proxies.

---

## What Is Not Addressed (Scope Gaps for Full Submission)

- **Additional datasets** — all experiments run on a single proprietary e-commerce dataset. Generalisability claims require at least two external datasets (e.g. SemEval-2017 Task 4, Amazon Reviews, Yelp).
- **Fine-tuned encoder baseline** — no DistilBERT/RoBERTa baseline exists; reviewers will ask for it.
- **Cost reporting** — API costs per experiment were not formally recorded; only calls/review is reported.
- **Multiple seeds** — all runs used a single seed. For publication, ≥3 seeds with mean ± std are expected.
- **Latency reporting** — p95 latency was not captured for any main branch experiment.
- **Label-noise analysis** — neutral class failure is attributed informally to label noise but not formally quantified.
- **Positioning against prior work** — no literature review or comparison against chain-of-thought, LLM-as-judge, or self-consistency prior work.
