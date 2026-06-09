# Sentiment Analysis Metrics — Self-Consistency Voting (exp-05)

**Run:** 2026-06-08 15:23:59  
**Input:** `/Users/adityanarvekar/Downloads/full_dataset.csv`  
**Total reviews:** 22641

## Design

Five independent LLM calls analyse each review in parallel. Final sentiment
is determined by majority vote (ties broken by `Counter.most_common`).

## Overall Accuracy

`0.7732` (77.32%)

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
| **good** | 14203 | 2785 | 460 |
| **neutral** | 104 | 1235 | 1484 |
| **poor** | 16 | 286 | 2068 |

## Predicted Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 14323 | 63.3% |
| neutral | 4306 | 19.0% |
| poor | 4012 | 17.7% |

## True Sentiment Distribution

| Sentiment | Count | % |
|:---|---:|---:|
| good | 17448 | 77.1% |
| neutral | 2823 | 12.5% |
| poor | 2370 | 10.5% |

## Voting Stats

| Vote pattern | Count | % |
|:---|---:|---:|
| Majority vote (5 LLMs): good=5 → good | 14253 | 63.0% |
| Majority vote (5 LLMs): neutral=5 → neutral | 4165 | 18.4% |
| Majority vote (5 LLMs): poor=5 → poor | 3917 | 17.3% |
| Majority vote (5 LLMs): poor=4, neutral=1 → poor | 51 | 0.2% |
| Majority vote (5 LLMs): neutral=4, poor=1 → neutral | 48 | 0.2% |
| Majority vote (5 LLMs): good=4, neutral=1 → good | 43 | 0.2% |
| Majority vote (5 LLMs): poor=3, neutral=2 → poor | 42 | 0.2% |
| Majority vote (5 LLMs): neutral=3, poor=2 → neutral | 40 | 0.2% |
| Majority vote (5 LLMs): neutral=3, good=2 → neutral | 28 | 0.1% |
| Majority vote (5 LLMs): good=3, neutral=2 → good | 27 | 0.1% |
| Majority vote (5 LLMs): neutral=4, good=1 → neutral | 25 | 0.1% |
| Majority vote (5 LLMs): poor=3, good=2 → poor | 1 | 0.0% |
| Majority vote (5 LLMs): poor=4, good=1 → poor | 1 | 0.0% |