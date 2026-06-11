# Sentiment Analysis Pipeline

LangGraph-based sentiment analysis pipeline classifying Women's Clothing E-Commerce reviews as **good**, **neutral**, or **poor** using GPT-4o-mini. Six experiment designs have been implemented and evaluated on 22,641 reviews.

## Repository Structure

```
├── scripts/          — pipeline scripts (one per experiment)
├── data/             — input CSVs (full dataset, test sample, original reviews)
├── results/
│   ├── full_dataset/ — full-run outputs (results CSV + metrics MD)
│   └── test_sample/  — test-run outputs
├── logs/             — run logs
├── notebooks/        — exploratory Jupyter notebooks
├── docs/             — experiment planning documents
├── README.md
└── requirements.txt
```

## Experiments

All experiments use GPT-4o-mini as the answerer unless noted. Primary metric is **Macro F1** — accuracy (~77%) is near-identical across experiments due to class imbalance (77% of reviews are `good`).

> **Metrics note:** All macro F1 values are computed over the three true classes only (good / neutral / poor). Out-of-vocabulary predictions (`mixed`, `error`) count as wrong but are not included as extra classes, avoiding sklearn macro-average deflation.

| Exp | Design | Script | Macro F1 | Accuracy | Notes |
|:---|:---|:---|---:|---:|:---|
| exp-01 | Single-call baseline | `sentiment_analysis_single_call_async.py` | 0.621 | — | True baseline; no review step |
| exp-02 | Sequential self-review, critic blind to text | `sentiment_analysis_async.py` | 0.630 | 77.47% | +0.008 over exp-01 (significant) |
| exp-03 | Sequential self-review, critic sees text | `sentiment_analysis_critic_sees_text_async.py` | 0.629 | 77.73% | Equivalent to exp-02 |
| exp-04 | Independent critic (two LLMs in parallel) | `sentiment_analysis_independent_critic_async.py` | 0.632 | 77.47% | Equivalent to exp-02 |
| exp-05 | Self-consistency voting (5 parallel calls) | `sentiment_analysis_self_consistency_async.py` | 0.630 | 77.32% | Equivalent to exp-02 at 5× the cost |
| exp-06 | Cross-model critic (GPT-4o critic, sees text) | `sentiment_analysis_cross_model_critic_sees_text_async.py` | 0.487 | 46.32% | ~40% error outputs — GPT-4o critic too strict |

## Statistical Findings

Paired bootstrap test (10,000 resamples) on 22,641 reviews. Full results: `results/full_dataset/paired_bootstrap_results_20260611.md`.

**Any review step significantly beats no review step.** All multi-call designs (exp-02–05) are significantly better than the single-call baseline (exp-01), with 95% CIs entirely above zero and p ≈ 0.

| Comparison | Diff (B − A) | 95% CI | Significant? |
|:---|---:|:---|:---|
| exp-01 vs exp-02 | +0.0084 | [+0.0049, +0.0120] | Yes |
| exp-01 vs exp-03 | +0.0077 | [+0.0041, +0.0113] | Yes |
| exp-01 vs exp-04 | +0.0103 | [+0.0069, +0.0138] | Yes |
| exp-01 vs exp-05 | +0.0081 | [+0.0046, +0.0116] | Yes |

**Among multi-call designs, differences are small but some are statistically significant.**

| Comparison | Diff (B − A) | 95% CI | Significant? |
|:---|---:|:---|:---|
| exp-02 vs exp-03 | −0.0006 | [−0.0032, +0.0018] | No |
| exp-02 vs exp-04 | +0.0019 | [−0.0002, +0.0041] | No |
| exp-02 vs exp-05 | −0.0002 | [−0.0023, +0.0018] | No |
| exp-03 vs exp-04 | +0.0026 | [+0.0002, +0.0050] | Yes |
| exp-03 vs exp-05 | +0.0004 | [−0.0020, +0.0029] | No |
| exp-04 vs exp-05 | −0.0022 | [−0.0041, −0.0003] | Yes |

**Result 1 — Multi-call architectures outperform single-call.** Every multi-call design (exp-02–05) significantly beats the single-call baseline (exp-01), with p ≈ 0 and 95% CIs entirely above zero across all four comparisons. Adding any form of review step yields a consistent +0.008–0.010 macro F1 gain.

**Result 2 — Best multi-call design: exp-04 (independent critic).** exp-04 achieves the highest macro F1 (0.632) and is the only multi-call design with statistically significant improvements over other multi-call architectures (exp-03 and exp-05). Its key advantage is eliminating anchoring bias: two LLMs independently analyse the same review in parallel from scratch, with no knowledge of each other's output. If they agree, that sentiment is accepted immediately. If they disagree, a third LLM arbitrates using both analyses and the original review text. This design delivers the largest single gain over the baseline (+0.0103, 95% CI [+0.0069, +0.0138]) while remaining efficient — 2 API calls per review on agreement, 3 only when arbitration is needed.

## Scripts

| Script | Experiment | Async | Checkpoint |
|:---|:---|:---:|:---:|
| `sentiment_analysis_single_call_async.py` | exp-01 | Yes | No |
| `sentiment_analysis.py` | exp-02 (sequential) | No | No |
| `sentiment_analysis_async.py` | exp-02 (async) | Yes | No |
| `sentiment_analysis_critic_sees_text_async.py` | exp-03 | Yes | No |
| `sentiment_analysis_async_nvidia.py` | NVIDIA Nemotron via OpenRouter | Yes | Yes |
| `sentiment_analysis_independent_critic_async.py` | exp-04 | Yes | Yes |
| `sentiment_analysis_self_consistency_async.py` | exp-05 | Yes | Yes |
| `sentiment_analysis_cross_model_critic_sees_text_async.py` | exp-06 | Yes | Yes |

## Usage

All scripts accept the same arguments. Run from the `scripts/` directory or provide the full path.

```bash
python scripts/<script_name>.py <csv_path> [--env-file <path>] [--concurrency <n>]
```

| Argument | Description | Default |
|:---|:---|:---|
| `csv_path` | Path to the input CSV file | — |
| `--env-file` | Path to a `.env` file containing API keys | auto-discover |
| `--concurrency` | Number of reviews to process in parallel | `20` |

### Running in the background (large datasets)

```bash
nohup python3 -u scripts/sentiment_analysis_async.py data/full_dataset.csv --env-file .env > logs/run_log.txt 2>&1 &
```

The `-u` flag disables Python's output buffering so progress is written to the log immediately.

## Input Format

The input CSV must contain:

| Column | Description |
|:---|:---|
| `Review_text` | The review text to analyse |
| `Sentiment` | Ground-truth label (`good`, `neutral`, or `poor`) |

Any additional columns are preserved in the output.

## Outputs

Two files are written per run, timestamped:

### `*_results_<timestamp>.csv`

All original input columns plus graph output columns (varies by experiment — includes `llm_sentiment` or `final_sentiment`, `reasoning`, `agreement`, `review_count`).

### `*_metrics_<timestamp>.md`

- Overall accuracy and classification report (precision, recall, F1 per class)
- Confusion matrix
- Predicted vs. true sentiment distributions
- Experiment-specific stats (agreement rate, vote breakdown, etc.)

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/adity-narvekar/sentiment-analysis.git
cd sentiment-analysis
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up API keys**

Place keys in a `.env` file and pass via `--env-file`:

```
OPENAI_API_KEY=sk-...
```

```bash
python scripts/sentiment_analysis_async.py data/full_dataset.csv --env-file .env
```

## Dataset

[Women's Clothing E-Commerce Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) (23,486 reviews; 22,641 after dropping nulls). Ground-truth labels derived from the `Rating` column:

| Rating | Sentiment |
|:---:|:---|
| 1–2 | poor |
| 3 | neutral |
| 4–5 | good |

**Class distribution:** good 77.1% · neutral 12.5% · poor 10.5% — heavily imbalanced. Use Macro F1 as the primary evaluation metric, not accuracy.
