# Sentiment Analysis Metrics

**Run:** 2026-06-03 22:03:09  
**Input:** `/Users/adityanarvekar/Downloads/full_dataset.csv`  
**Total reviews:** 22641

## Overall Accuracy

`0.7747` (77.47%)

## Classification Report

```
              precision    recall  f1-score   support

        good       0.99      0.82      0.90     17448
       mixed       0.00      0.00      0.00         0
     neutral       0.29      0.43      0.35      2823
        poor       0.51      0.88      0.65      2370

    accuracy                           0.77     22641
   macro avg       0.45      0.53      0.47     22641
weighted avg       0.85      0.77      0.80     22641
```

## Confusion Matrix

| True \ Predicted | good | mixed | neutral | poor |
|:---|---:|---:|---:|---:|
| **good** | 14240 | 0 | 2734 | 474 |
| **mixed** | 0 | 0 | 0 | 0 |
| **neutral** | 109 | 3 | 1214 | 1497 |
| **poor** | 18 | 0 | 266 | 2086 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 14367 | 63.5% |
| neutral | 4214 | 18.6% |
| poor | 4057 | 17.9% |
| mixed | 3 | 0.0% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 17448 | 77.1% |
| neutral | 2823 | 12.5% |
| poor | 2370 | 10.5% |

## LLM Self-Review Stats

- **Agreement rate (first pass):** 0.9999 (99.99%)
- **Average iterations per review:** 1.01

### Iteration count distribution

| Iterations | Reviews |
|---:|---:|
| 1 | 22472 |
| 2 | 152 |
| 3 | 17 |

## Corrected 3-Class Metrics (added 2026-06-11)

> **Correction:** The macro average above (0.47) is deflated by a spurious zero-support `mixed` label emitted by the LLM, which sklearn counts as a 4th class with F1 = 0.00. Metrics below are recomputed from the confusion matrix over the three true classes only; out-of-label predictions still count against recall.

| Class | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| good | 0.9912 | 0.8161 | 0.8952 | 17448 |
| neutral | 0.2881 | 0.4300 | 0.3450 | 2823 |
| poor | 0.5142 | 0.8802 | 0.6491 | 2370 |

**Corrected Macro F1: 0.6298** (previously reported: 0.47)