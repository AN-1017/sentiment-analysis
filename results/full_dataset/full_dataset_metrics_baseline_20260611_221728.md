# Sentiment Analysis Metrics — Single-Call Baseline (exp-01)

**Run:** 2026-06-11 22:17:29  
**Input:** `data/full_dataset.csv`  
**Total reviews:** 22641

## Overall Accuracy

`0.7815` (78.15%)

## Classification Report

```
              precision    recall  f1-score   support

        good       0.99      0.83      0.90     17448
     neutral       0.29      0.38      0.33      2823
        poor       0.49      0.89      0.63      2370

    accuracy                           0.78     22641
   macro avg       0.59      0.70      0.62     22641
weighted avg       0.85      0.78      0.80     22641
```

## Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| **good** | 14523 | 2363 | 562 |
| **neutral** | 140 | 1060 | 1623 |
| **poor** | 20 | 239 | 2111 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 14683 | 64.9% |
| poor | 4296 | 19.0% |
| neutral | 3662 | 16.2% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 17448 | 77.1% |
| neutral | 2823 | 12.5% |
| poor | 2370 | 10.5% |