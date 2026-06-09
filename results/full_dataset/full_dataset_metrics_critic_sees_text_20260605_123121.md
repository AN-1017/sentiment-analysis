# Sentiment Analysis Metrics

**Run:** 2026-06-05 12:31:23  
**Input:** `/Users/adityanarvekar/Downloads/full_dataset.csv`  
**Total reviews:** 22641

## Overall Accuracy

`0.7773` (77.73%)

## Classification Report

```
              precision    recall  f1-score   support

       error       0.00      0.00      0.00         0
        good       0.99      0.82      0.90     17448
       mixed       0.00      0.00      0.00         0
     neutral       0.29      0.42      0.34      2823
        poor       0.51      0.88      0.65      2370

    accuracy                           0.78     22641
   macro avg       0.36      0.42      0.38     22641
weighted avg       0.85      0.78      0.80     22641
```

## Confusion Matrix

| True \ Predicted | error | good | mixed | neutral | poor |
|:---|---:|---:|---:|---:|---:|
| **error** | 0 | 0 | 0 | 0 | 0 |
| **good** | 1 | 14327 | 1 | 2634 | 485 |
| **mixed** | 0 | 0 | 0 | 0 | 0 |
| **neutral** | 0 | 120 | 0 | 1192 | 1511 |
| **poor** | 0 | 17 | 0 | 274 | 2079 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 14464 | 63.9% |
| neutral | 4100 | 18.1% |
| poor | 4075 | 18.0% |
| error | 1 | 0.0% |
| mixed | 1 | 0.0% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 17448 | 77.1% |
| neutral | 2823 | 12.5% |
| poor | 2370 | 10.5% |

## LLM Self-Review Stats

- **Agreement rate (first pass):** 0.9987 (99.87%)
- **Average iterations per review:** 1.02

### Iteration count distribution

| Iterations | Reviews |
|---:|---:|
| 0 | 1 |
| 1 | 22279 |
| 2 | 281 |
| 3 | 80 |