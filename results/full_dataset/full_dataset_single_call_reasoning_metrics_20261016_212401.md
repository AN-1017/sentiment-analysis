# Sentiment Analysis Metrics — Single-Call Reasoning Baseline (exp-01b)

**Run:** 2026-10-16 21:24:01  
**Input:** `data/full_dataset.csv`  
**Total reviews:** 22641

## Overall Accuracy

`0.7724` (77.24%)

## Classification Report

```
              precision    recall  f1-score   support

        good       0.99      0.81      0.89     17448
     neutral       0.29      0.44      0.35      2823
        poor       0.52      0.87      0.65      2370

    accuracy                           0.77     22641
   macro avg       0.60      0.71      0.63     22641
weighted avg       0.85      0.77      0.80     22641
```

## Confusion Matrix

| True \ Predicted | good | neutral | poor |
|:---|---:|---:|---:|
| **good** | 14178 | 2815 | 455 |
| **neutral** | 100 | 1246 | 1477 |
| **poor** | 16 | 291 | 2063 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 14294 | 63.1% |
| neutral | 4352 | 19.2% |
| poor | 3995 | 17.6% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 17448 | 77.1% |
| neutral | 2823 | 12.5% |
| poor | 2370 | 10.5% |