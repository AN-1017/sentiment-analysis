# Sentiment Analysis Metrics — Self-Consistency Voting (exp-05)

**Run:** 2026-06-07 15:41:13  
**Input:** `test_sample.csv`  
**Total reviews:** 5

## Design

Five independent LLM calls analyse each review in parallel. Final sentiment
is determined by majority vote (ties broken by `Counter.most_common`).

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

## Voting Stats

| Vote pattern | Count | % |
|:---|---:|---:|
| Majority vote (5 LLMs): good=5 → good | 4 | 80.0% |
| Majority vote (5 LLMs): poor=5 → poor | 1 | 20.0% |