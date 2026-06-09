# Sentiment Analysis Metrics — Independent Critic (exp-04)

**Run:** 2026-06-07 12:35:53  
**Input:** `/Users/adityanarvekar/Downloads/full_dataset.csv`  
**Total reviews:** 22641

## Design

Two LLMs analyze each review independently in parallel. If they agree, that
sentiment is used directly. If they disagree, a third LLM arbitrates.

## Overall Accuracy

`0.7747` (77.47%)

## Classification Report

```
              precision    recall  f1-score   support

       error       0.00      0.00      0.00         0
        good       0.99      0.82      0.89     17448
     neutral       0.29      0.44      0.35      2823
        poor       0.52      0.87      0.65      2370

    accuracy                           0.77     22641
   macro avg       0.45      0.53      0.47     22641
weighted avg       0.85      0.77      0.80     22641
```

## Confusion Matrix

| True \ Predicted | error | good | neutral | poor |
|:---|---:|---:|---:|---:|
| **error** | 0 | 0 | 0 | 0 |
| **good** | 2 | 14223 | 2773 | 450 |
| **neutral** | 0 | 103 | 1244 | 1476 |
| **poor** | 0 | 17 | 280 | 2073 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 14343 | 63.3% |
| neutral | 4297 | 19.0% |
| poor | 3999 | 17.7% |
| error | 2 | 0.0% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 17448 | 77.1% |
| neutral | 2823 | 12.5% |
| poor | 2370 | 10.5% |

## Agreement Stats

- **Both LLMs agreed:** 22477 (99.3%) — went straight to END
- **LLMs disagreed:** 164 (0.7%) — sent to arbitrator
- **Arbitrated:** 164