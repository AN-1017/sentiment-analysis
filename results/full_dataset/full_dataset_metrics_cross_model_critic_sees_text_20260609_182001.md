# Sentiment Analysis Metrics

**Run:** 2026-06-09 18:20:03  
**Input:** `/Users/adityanarvekar/Downloads/full_dataset.csv`  
**Total reviews:** 22641

## Overall Accuracy

`0.4632` (46.32%)

## Classification Report

```
              precision    recall  f1-score   support

       error       0.00      0.00      0.00         0
        good       0.99      0.49      0.65     17448
     neutral       0.29      0.26      0.28      2823
        poor       0.53      0.54      0.53      2370

    accuracy                           0.46     22641
   macro avg       0.45      0.32      0.37     22641
weighted avg       0.85      0.46      0.59     22641
```

## Confusion Matrix

| True \ Predicted | error | good | neutral | poor |
|:---|---:|---:|---:|---:|
| **error** | 0 | 0 | 0 | 0 |
| **good** | 7100 | 8472 | 1606 | 270 |
| **neutral** | 1133 | 72 | 733 | 885 |
| **poor** | 917 | 11 | 159 | 1283 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| error | 9150 | 40.4% |
| good | 8555 | 37.8% |
| neutral | 2498 | 11.0% |
| poor | 2438 | 10.8% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 17448 | 77.1% |
| neutral | 2823 | 12.5% |
| poor | 2370 | 10.5% |

## LLM Self-Review Stats

- **Agreement rate (first pass):** 0.5959 (59.59%)
- **Average iterations per review:** 0.60

### Iteration count distribution

| Iterations | Reviews |
|---:|---:|
| 0 | 9150 |
| 1 | 13447 |
| 2 | 43 |
| 3 | 1 |

## Corrected 3-Class Metrics (added 2026-06-11)

> **Correction:** The macro average above (0.37) includes a zero-support `error` class. Metrics below are recomputed over the three true classes only. The 9,150 `error` predictions (40.4% of the dataset) still count against recall, so this run remains a clear failure even after correction.

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| good | 0.9903 | 0.4856 | 0.6516 | 17448 |
| neutral | 0.2934 | 0.2597 | 0.2755 | 2823 |
| poor | 0.5263 | 0.5414 | 0.5337 | 2370 |

**Corrected Macro F1: 0.4869** (previously reported: 0.37) — still well below the exp-02 baseline (0.6298)