# Sentiment Analysis Pipeline

LangGraph-based sentiment analysis pipeline that classifies product reviews as **good**, **neutral**, or **poor** using a two-node self-review loop powered by GPT-4o-mini.

## How it works

The pipeline runs each review through a LangGraph graph with two nodes:

1. **`answer_node`** — GPT-4o-mini predicts the sentiment and provides reasoning. On retry iterations it incorporates the reviewer's prior feedback.
2. **`review_node`** — A second LLM call independently judges whether the prediction is correct.
3. **`should_continue`** — If the reviewer disagrees, the loop retries (up to 3 times). Once agreement is reached or the limit is hit, the graph exits.

```
START → answer_node → review_node ─── agree or limit reached ──→ END
                          │
                          └── disagrees, count < 3 ──→ answer_node
```

## Usage

```bash
python sentiment_analysis.py <csv_path> [--env-file <path>]
```

**Arguments:**

| Argument | Description |
|:---|:---|
| `csv_path` | Path to the input CSV file |
| `--env-file` | Path to a `.env` file containing API keys (default: auto-discover from current directory) |

**Example:**

```bash
python sentiment_analysis.py "Womens Clothing E-Commerce Reviews.csv" --env-file /path/to/.env
```

## Input format

The input CSV must contain these two columns:

| Column | Description |
|:---|:---|
| `Review_text` | The review text to analyse |
| `Sentiment` | Ground-truth label (`good`, `neutral`, or `poor`) used for evaluation |

Any additional columns in the CSV are preserved in the output.

## Outputs

Two files are written alongside the input CSV, both suffixed with a timestamp:

### `*_results_<timestamp>.csv`

All original input columns plus the following graph output columns:

| Column | Description |
|:---|:---|
| `llm_sentiment` | Predicted sentiment (`good`, `neutral`, `poor`) |
| `reasoning` | LLM reasoning for the final prediction |
| `agreement` | Whether the reviewer agreed (`True`/`False`) |
| `review_reasoning` | Reviewer's reasoning for agreement or disagreement |
| `review_count` | Number of answer-review iterations taken |

### `*_metrics_<timestamp>.md`

- Overall accuracy
- Full classification report (precision, recall, F1 per class)
- Confusion matrix
- Predicted vs. true sentiment distributions
- LLM self-review stats: agreement rate and iteration count distribution

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/adity-narvekar/sentiment-analysis.git
cd sentiment-analysis
```

**2. Create and activate a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up your OpenAI API key**

Either export it as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

Or place it in a `.env` file and pass it via `--env-file`:

```
OPENAI_API_KEY=sk-...
```

```bash
python sentiment_analysis.py data.csv --env-file .env
```

## Dataset

The notebook and example run use the [Women's Clothing E-Commerce Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) dataset (23,486 reviews). Ground-truth sentiment labels were derived from the `Rating` column:

| Rating | Sentiment |
|:---:|:---|
| 1–2 | poor |
| 3 | neutral |
| 4–5 | good |
