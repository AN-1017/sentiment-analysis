# Revised Statistical Analysis — Exp-01b

## Experimental Setup

Dataset: Women's Clothing E-Commerce Reviews  
Sample size: 22,641 reviews  
Model: GPT-4o-mini  
Temperature: 0  
Primary evaluation metric: Macro F1  
Statistical method: Paired bootstrap resampling  
Bootstrap resamples: 10,000  
Random seed: 0  

## Experiment Performance

| Experiment | Macro F1 |
|---|---:|
| Exp-01: Single-call label-only baseline | 0.6214 |
| Exp-01b: Single-call reasoning baseline | 0.6301 |
| Exp-04: Independent critic with arbitration | 0.6317 |

## Pairwise Bootstrap Results

| Comparison | Difference (B − A) | 95% CI | Significance |
|---|---:|---|---|
| Exp-01 vs Exp-01b | +0.0087 | [+0.0052, +0.0123] | Significant |
| Exp-01b vs Exp-04 | +0.0016 | [-0.0005, +0.0037] | Not significant |

All comparisons use 10,000 paired bootstrap resamples with random seed 0.

### Interpretation

The single-call reasoning baseline achieves a higher macro F1 than the original label-only baseline, with an observed improvement of 0.0087 and a 95% paired bootstrap confidence interval excluding zero.

The independent-critic architecture achieves a numerically higher macro F1 than the reasoning baseline, but the observed improvement is only 0.0016, with a 95% confidence interval spanning zero.

These results suggest that the original improvement over the label-only baseline was largely associated with the introduction of reasoning instructions. The evidence does not establish that the independent-critic architecture provides additional performance benefits beyond the single-call reasoning baseline.

### Reporting Limitations

These results are preliminary. The existing bootstrap script reports one-sided empirical tail probabilities. Two-sided testing and multiple-comparison correction have not yet been completed.

Results should not be described as establishing statistical equivalence between the reasoning baseline and the independent-critic architecture.