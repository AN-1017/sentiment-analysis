# Exp-01b — single-call reasoning baseline

Dataset: `data\full_dataset.csv`  
Rows: 22641  
Model: `gpt-4o-mini`  
Temperature: `0`  
Max tokens: `1024`  
API calls attempted: 22641
Invalid/error predictions: 0  
Macro F1 (3 classes): 0.630116
Accuracy: 0.772802

## Classification report

```
              precision    recall  f1-score   support

        good       0.99      0.81      0.89     17448
     neutral       0.29      0.44      0.35      2823
        poor       0.52      0.87      0.65      2370

    accuracy                           0.77     22641
   macro avg       0.60      0.71      0.63     22641
weighted avg       0.85      0.77      0.80     22641

```

## Confusion matrix

Order: ['good', 'neutral', 'poor']

```
[[14187  2816   445]
 [   98  1247  1478]
 [   16   291  2063]]
```
