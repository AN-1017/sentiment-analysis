# exp-01b analysis — planned comparisons

N = 22641 reviews · 10000 paired bootstrap resamples · seed 0 · equivalence margin ±0.005

OOV predictions count as wrong and never as a predicted class.

## 1. Per-system results

| System | OOV | Macro F1 | 95% CI | good F1 | neutral F1 | poor F1 | Accuracy |
|:---|---:|---:|:---|---:|---:|---:|---:|
| exp-01 | 0 | 0.6214 | [0.6138, 0.6291] | 0.904 | 0.327 | 0.633 | 78.15% |
| exp-01b | 0 | 0.6296 | [0.6222, 0.6370] | 0.893 | 0.347 | 0.648 | 77.24% |
| exp-02 | 3 | 0.6298 | [0.6223, 0.6373] | 0.895 | 0.345 | 0.649 | 77.47% |
| exp-03 | 2 | 0.6291 | [0.6217, 0.6366] | 0.898 | 0.344 | 0.645 | 77.73% |
| exp-04 | 2 | 0.6317 | [0.6242, 0.6392] | 0.895 | 0.349 | 0.651 | 77.47% |
| exp-05 | 0 | 0.6295 | [0.6221, 0.6371] | 0.894 | 0.346 | 0.648 | 77.32% |
| exp-04 call 1 (no review) | 2 | 0.6306 | [0.6231, 0.6380] | 0.894 | 0.348 | 0.650 | 77.35% |
| exp-04 call 2 (no review) | 2 | 0.6301 | [0.6227, 0.6376] | 0.894 | 0.347 | 0.650 | 77.31% |

## 2. Primary planned comparisons (Holm over these 5)

- exp-01 → exp-01b isolates the **prompt format** (label-only vs reasoning prompt).
- exp-01b → exp-02…05 isolates the **review / arbitration / voting step**.

| Comparison (A → B) | Diff (B−A) | 95% CI | p (two-sided) | Holm p | Significant after Holm? | Equivalent within ±0.005? |
|:---|---:|:---|---:|---:|:---|:---|
| exp-01 → exp-01b | +0.0082 | [+0.0047, +0.0117] | 0.0002 | 0.0010 | **Yes** | No |
| exp-01b → exp-02 | +0.0002 | [-0.0021, +0.0024] | 0.8689 | 1.0000 | No | Yes |
| exp-01b → exp-03 | -0.0005 | [-0.0031, +0.0020] | 0.7195 | 1.0000 | No | Yes |
| exp-01b → exp-04 | +0.0021 | [+0.0000, +0.0042] | 0.0496 | 0.1984 | No | Yes |
| exp-01b → exp-05 | -0.0001 | [-0.0021, +0.0019] | 0.9489 | 1.0000 | No | Yes |

## 3. The paper's original pairwise tests (Holm over these 10)

| Comparison (A → B) | Diff (B−A) | 95% CI | p (two-sided) | Holm p | Significant after Holm? | Equivalent within ±0.005? |
|:---|---:|:---|---:|---:|:---|:---|
| exp-01 → exp-02 | +0.0084 | [+0.0049, +0.0120] | 0.0002 | 0.0020 | **Yes** | No |
| exp-01 → exp-03 | +0.0077 | [+0.0041, +0.0113] | 0.0002 | 0.0020 | **Yes** | No |
| exp-01 → exp-04 | +0.0103 | [+0.0069, +0.0138] | 0.0002 | 0.0020 | **Yes** | No |
| exp-01 → exp-05 | +0.0081 | [+0.0046, +0.0116] | 0.0002 | 0.0020 | **Yes** | No |
| exp-02 → exp-03 | -0.0006 | [-0.0032, +0.0018] | 0.6129 | 1.0000 | No | Yes |
| exp-02 → exp-04 | +0.0019 | [-0.0002, +0.0041] | 0.0772 | 0.3088 | No | Yes |
| exp-02 → exp-05 | -0.0002 | [-0.0023, +0.0018] | 0.8311 | 1.0000 | No | Yes |
| exp-03 → exp-04 | +0.0026 | [+0.0002, +0.0050] | 0.0322 | 0.1610 | No | Yes |
| exp-03 → exp-05 | +0.0004 | [-0.0020, +0.0029] | 0.7405 | 1.0000 | No | Yes |
| exp-04 → exp-05 | -0.0022 | [-0.0041, -0.0003] | 0.0240 | 0.1440 | No | Yes |

## 4. Run-to-run noise (same prompt, same model, temperature 0)

| Run A | Run B | Rows with different labels | Agreement |
|:---|:---|---:|---:|
| exp-01b | exp-04 call 1 (no review) | 228 | 98.99% |
| exp-01b | exp-04 call 2 (no review) | 248 | 98.90% |
| exp-04 call 1 (no review) | exp-04 call 2 (no review) | 164 | 99.28% |

Macro F1 of the three identical-prompt single calls: 0.6296, 0.6306, 0.6301 (range 0.0009).

## 5. Label changes

- **exp-01 → exp-01b:** 785 rows differ; 434 got worse, 227 got better.
- **exp-01b → exp-02:** 319 rows differ; 102 got worse, 155 got better.
- **exp-01b → exp-04:** 250 rows differ; 75 got worse, 128 got better.
- **exp-04 arbitration:** 164 disagreements between the two calls; vs call 1, arbitration fixed 45 and broke 17.
- **exp-04 full pipeline vs its own call 1:** +0.0012, 95% CI [-0.0000, +0.0024].

## 6. exp-01b confusion matrix (rows = true, columns = predicted)

| True \ Pred | good | neutral | poor | OOV |
|:---|---:|---:|---:|---:|
| **good** | 14178 | 2815 | 455 | 0 |
| **neutral** | 100 | 1246 | 1477 | 0 |
| **poor** | 16 | 291 | 2063 | 0 |

Predicted distribution: good 14294 (63.1%), neutral 4352 (19.2%), poor 3995 (17.6%)

