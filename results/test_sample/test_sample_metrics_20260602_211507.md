# Sentiment Analysis Metrics

**Run:** 2026-06-02 21:15:07  
**Input:** `test_sample.csv`  
**Total reviews:** 5

## Overall Accuracy

`0.8000` (80.00%)

## Classification Report

```
              precision    recall  f1-score   support

        good       1.00      1.00      1.00         4
     neutral       0.00      0.00      0.00         1
        poor       0.00      0.00      0.00         0

    accuracy                           0.80         5
   macro avg       0.33      0.33      0.33         5
weighted avg       0.80      0.80      0.80         5
```

## Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| **good** | 4 | 0 | 0 |
| **neutral** | 0 | 0 | 1 |
| **poor** | 0 | 0 | 0 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 4 | 80.0% |
| poor | 1 | 20.0% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 4 | 80.0% |
| neutral | 1 | 20.0% |

## LLM Self-Review Stats

- **Agreement rate (first pass):** 1.0000 (100.00%)
- **Average iterations per review:** 1.00

### Iteration count distribution

| Iterations | Reviews |
|---:|---:|
| 1 | 5 |