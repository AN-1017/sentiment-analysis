#!/usr/bin/env python3
"""LangGraph sentiment analysis pipeline — single-call reasoning baseline (exp-01b).

One LLM call per review; no self-review loop, no critic, no voting.

Purpose: a control for exp-02..exp-05. exp-01 asks for the label only, whereas
every multi-call design asks its answerer for a label PLUS reasoning. exp-01b
uses exactly the same prompt and output schema as the multi-call designs'
first answer (see sentiment_analysis_async.py, generate_answer, review_count == 0)
but makes only that one call. Comparing exp-01b against exp-02..exp-05
therefore isolates the effect of the review / arbitration / voting step;
comparing exp-01 against exp-01b isolates the effect of the prompt format.

Usage:
    python scripts/sentiment_analysis_single_call_reasoning_async.py <csv_path> \
        [--env-file <path>] [--concurrency <n>]

The input CSV must contain:
    - Review_text   : the review text to analyse
    - Sentiment     : the ground-truth label (good / neutral / poor)

Outputs (written alongside the input CSV):
    - <base>_single_call_reasoning_results_<timestamp>.csv
          all input columns + llm_sentiment, reasoning
    - <base>_single_call_reasoning_metrics_<timestamp>.md
          classification report and supporting metrics

The results CSV uses the column name `llm_sentiment`, so it can be passed
directly to scripts/paired_bootstrap.py alongside the other experiments.
"""

import argparse
import asyncio
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ── Pydantic state definitions ─────────────────────────────────────────────────
# NOTE: OutputState is deliberately identical (field names, order, descriptions)
# to the OutputState in sentiment_analysis_async.py (exp-02), so that the model
# sees the same output schema. Do not reorder or reword these fields.

class OutputState(BaseModel):
    sentiment: str = Field(description="Predicted sentiment: good, neutral, or poor.")
    reasoning: str = Field(description="Reasoning behind the sentiment prediction.")


class OverallState(BaseModel):
    review_text: str = Field(description="The review text to analyse.")
    sentiment: str = Field(default="", description="Predicted sentiment.")
    reasoning: str = Field(default="", description="Reasoning for the sentiment.")


# ── Graph builder ──────────────────────────────────────────────────────────────

def _build_graph(llm: ChatOpenAI):
    llm_answer = llm.with_structured_output(OutputState)

    def generate_answer(state: OverallState) -> dict:
        # Prompt text is identical to the first-iteration answerer prompt in
        # exp-02 / exp-03 and to answerer 1 / 2 in exp-04. Do not reword.
        prompt = (
            "Analyze the following review and determine its sentiment "
            "(good, neutral, or poor). Provide reasoning for your analysis.\n\n"
            f'Review: "{state.review_text}"'
        )
        response = llm_answer.invoke(prompt)
        return {"sentiment": response.sentiment, "reasoning": response.reasoning}

    builder = StateGraph(OverallState)
    builder.add_node("answer_node", generate_answer)
    builder.add_edge(START, "answer_node")
    builder.add_edge("answer_node", END)
    return builder.compile()


# ── Async concurrent runner ────────────────────────────────────────────────────

async def _process_one(graph, inp: dict, i: int, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            result = await graph.ainvoke(inp)
            return i, result, None
        except Exception as e:
            return i, None, e


async def _run_concurrent(graph, inputs: list, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_process_one(graph, inp, i, semaphore) for i, inp in enumerate(inputs)]

    results = [None] * len(inputs)
    errors = 0
    completed = 0

    for coro in asyncio.as_completed(tasks):
        i, result, error = await coro
        completed += 1

        if error is not None:
            print(f"  [row {i}] ERROR: {type(error).__name__}: {str(error)[:120]} — skipping", flush=True)
            results[i] = {
                "review_text": inputs[i]["review_text"],
                "sentiment": "error",
                "reasoning": str(error)[:200],
            }
            errors += 1
        else:
            results[i] = result

        if completed % 500 == 0:
            print(f"  {completed}/{len(inputs)} processed ({errors} errors so far)...", flush=True)

    return results, errors


# ── Metrics markdown builder ───────────────────────────────────────────────────

def _build_metrics_md(df: pd.DataFrame, csv_path: str) -> str:
    y_true = df["Sentiment"]
    y_pred = df["llm_sentiment"]

    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    report_str = classification_report(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    lines = [
        "# Sentiment Analysis Metrics — Single-Call Reasoning Baseline (exp-01b)",
        "",
        f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Input:** `{csv_path}`  ",
        f"**Total reviews:** {len(df)}",
        "",
        "## Overall Accuracy",
        "",
        f"`{accuracy:.4f}` ({accuracy * 100:.2f}%)",
        "",
        "## Classification Report",
        "",
        f"```\n{report_str}```",
        "",
        "## Confusion Matrix",
        "",
        "| True \\ Predicted | " + " | ".join(labels) + " |",
        "|:---|" + "---:|" * len(labels),
    ]
    for i, label in enumerate(labels):
        row = (
            f"| **{label}** | "
            + " | ".join(str(cm[i][j]) for j in range(len(labels)))
            + " |"
        )
        lines.append(row)

    lines += [
        "",
        "## Predicted Sentiment Distribution",
        "",
        "| Sentiment | Count | % |",
        "|:---|---:|---:|",
    ]
    for sent, count in df["llm_sentiment"].value_counts().items():
        lines.append(f"| {sent} | {count} | {count / len(df) * 100:.1f}% |")

    lines += [
        "",
        "## True Sentiment Distribution",
        "",
        "| Sentiment | Count | % |",
        "|:---|---:|---:|",
    ]
    for sent, count in df["Sentiment"].value_counts().items():
        lines.append(f"| {sent} | {count} | {count / len(df) * 100:.1f}% |")

    return "\n".join(lines)


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run(csv_path: str, env_file: Optional[str] = None, concurrency: int = 20) -> pd.DataFrame:
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)

    df = pd.read_csv(csv_path)

    missing = {"Review_text", "Sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    # Same model settings as every other experiment.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)
    graph = _build_graph(llm)

    print(f"Processing {len(df)} reviews with concurrency={concurrency}...", flush=True)
    inputs = [{"review_text": str(text)} for text in df["Review_text"]]

    results_list, errors = asyncio.run(_run_concurrent(graph, inputs, concurrency))

    print(f"Done. {len(results_list)} processed, {errors} errors.", flush=True)

    graph_df = (
        pd.DataFrame(results_list)
        .rename(columns={"sentiment": "llm_sentiment"})
        .drop(columns=["review_text"])
    )

    final_df = pd.concat(
        [df.reset_index(drop=True), graph_df.reset_index(drop=True)], axis=1
    )

    base = os.path.splitext(csv_path)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{base}_single_call_reasoning_results_{timestamp}.csv"
    metrics_path = f"{base}_single_call_reasoning_metrics_{timestamp}.md"

    final_df.to_csv(results_path, index=False)
    print(f"Results saved  → {results_path}", flush=True)

    md = _build_metrics_md(final_df, csv_path)
    with open(metrics_path, "w") as f:
        f.write(md)
    print(f"Metrics saved  → {metrics_path}", flush=True)

    return final_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run single-call reasoning LangGraph sentiment analysis baseline (exp-01b)."
    )
    parser.add_argument(
        "csv_path",
        help="Path to input CSV with 'Review_text' and 'Sentiment' columns.",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file for API keys (default: auto-discover from cwd).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Number of reviews to process concurrently (default: 20).",
    )
    args = parser.parse_args()
    run(args.csv_path, env_file=args.env_file, concurrency=args.concurrency)
