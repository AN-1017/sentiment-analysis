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