# Sentiment Analysis Metrics — Single-Call Reasoning Baseline (exp-01b)

**Run:** 2026-10-16 20:42:27  
**Input:** `data/sample_200.csv`  
**Total reviews:** 200

## Overall Accuracy

`0.7200` (72.00%)

## Classification Report

```
              precision    recall  f1-score   support

        good       1.00      0.74      0.85       151
     neutral       0.23      0.42      0.30        26
        poor       0.51      0.91      0.66        23

    accuracy                           0.72       200
   macro avg       0.58      0.69      0.60       200
weighted avg       0.84      0.72      0.76       200
```

## Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| **good** | 112 | 34 | 5 |
| **neutral** | 0 | 11 | 15 |
| **poor** | 0 | 2 | 21 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 112 | 56.0% |
| neutral | 47 | 23.5% |
| poor | 41 | 20.5% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 151 | 75.5% |
| neutral | 26 | 13.0% |
| poor | 23 | 11.5% |