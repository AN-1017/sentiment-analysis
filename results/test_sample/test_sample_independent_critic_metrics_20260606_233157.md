# Sentiment Analysis Metrics — Independent Critic (exp-04)

**Run:** 2026-06-06 23:31:57  
**Input:** `test_sample.csv`  
**Total reviews:** 5

## Design

Two LLMs analyze each review independently in parallel. If they agree, that
sentiment is used directly. If they disagree, a third LLM arbitrates.

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

## Agreement Stats

- **Both LLMs agreed:** 5 (100.0%) — went straight to END
- **LLMs disagreed:** 0 (0.0%) — sent to arbitrator
- **Arbitrated:** 0