# Metrics Review — Full Dataset Run (2026-06-03)

**Input:** `full_dataset.csv` (22,641 reviews)  
**Script:** `sentiment_analysis_async.py` (concurrency=20)  
**Overall Accuracy:** 77.47% — misleading due to class imbalance (see below)  
**Macro F1:** 0.47 — the honest performance metric for this dataset

---

## Class Imbalance & Why Accuracy Is Misleading

The dataset is heavily skewed toward the `good` class:

| Class | Count | % |
|:---|---:|---:|
| good | 17,448 | 77.1% |
| neutral | 2,823 | 12.5% |
| poor | 2,370 | 10.5% |

A naive classifier that predicts **"good" for every review** would achieve **77.1% accuracy** — nearly identical to the model's 77.47%. The model barely beats a dumb baseline on accuracy alone.

### Weighted vs Macro Averages — Explained

**Macro average** — treats all classes equally regardless of size. Add up each class's metric and divide by the number of classes. Every class counts the same whether it has 100 or 10,000 samples.

**Weighted average** — weights each class's metric by how many samples it has. Bigger classes pull the average more. With 77% of data in `good`, the weighted avg is dominated by how well the model does on `good`.

**Concrete example from this dataset:**

- `good` F1 = 0.90, `neutral` F1 = 0.35, `poor` F1 = 0.65

**Macro F1** = (0.90 + 0.35 + 0.65) / 3 = **0.47** — equal weight to all three classes

**Weighted F1** = (0.90 × 17,448 + 0.35 × 2,823 + 0.65 × 2,370) / 22,641 = **0.80** — `good`'s strong score pulls it up

**Which to use:** For imbalanced datasets like this one, macro F1 is more honest — it forces the model to do well on minority classes (`neutral`, `poor`), not just the dominant one. Weighted F1 can hide poor performance on minority classes behind a large majority class.

| Metric | Weighted Avg | Macro Avg |
|:---|---:|---:|
| Precision | 0.85 | 0.45 |
| Recall | 0.77 | 0.53 |
| F1 | 0.80 | 0.47 |

- **Weighted avg** is inflated by the dominant `good` class — makes the model look better than it is.
- **Macro avg** treats all classes equally and is the honest number. **Macro F1 of 0.47** is the real performance story.

### Per-Class Precision & Recall

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| good | 0.99 | 0.82 | 0.90 | 17,448 |
| neutral | 0.29 | 0.43 | 0.35 | 2,823 |
| poor | 0.51 | 0.88 | 0.65 | 2,370 |

- **`good`:** High precision (99%) means nearly every "good" prediction is correct, but low recall (82%) means 3,208 truly good reviews were misclassified — mostly pushed into `neutral`.
- **`neutral`:** Low precision (29%) means most "neutral" predictions are wrong; low recall (43%) means the model also misses most true neutral reviews. Worst-performing class by far.
- **`poor`:** High recall (88%) means the model catches most poor reviews, but precision (51%) is modest — it over-labels some neutral reviews as poor.

**Primary metric to optimise going forward: Macro F1 (currently 0.47).**

---

## What's Working Well

- **`good` class is strong** — 99% precision, 82% recall, 0.90 F1. The model is very confident and mostly correct when predicting positive reviews.
- **`poor` class is reasonable** — 88% recall means it catches most truly poor reviews, though precision (51%) is lower, so it over-predicts poor on some neutral reviews.
- **0 errors** across the entire run.

---

## Where It Struggles

- **`neutral` is the weak spot** — 29% precision and 43% recall, F1 of 0.35. The model finds neutral sentiment ambiguous, which is expected — neutral reviews sit between positive and negative and are harder to distinguish by reasoning alone.
- **`mixed` label appeared 3 times** — the LLM invented a 4th label not in the ground truth (`good`/`neutral`/`poor`). Minor prompt adherence issue.
- **`good` is under-predicted** — 3,208 good reviews were misclassified (mostly as neutral), pulling recall down to 82%.

---

## Self-Review Loop Analysis

- **99.99% first-pass agreement** — the reviewer almost always agrees with the initial prediction.
- **Average iterations: 1.01** — the retry loop is essentially doing nothing useful.
- The loop adds latency and doubles API calls with no meaningful quality improvement. Consider removing it to halve costs and runtime.

---

## Recommendations

1. **Remove the self-review loop** — 99.99% agreement rate means it provides no value. Removing it cuts API calls in half and roughly doubles speed.
2. **Improve `neutral` classification** — add few-shot examples to the prompt that clearly distinguish neutral from good/poor, or use a confidence threshold approach.
3. **Fix `mixed` label leakage** — constrain the prompt more explicitly (e.g., "respond with exactly one of: good, neutral, poor") to prevent the model from inventing new labels.
