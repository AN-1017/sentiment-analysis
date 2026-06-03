#!/usr/bin/env python3
"""
LangGraph sentiment analysis pipeline.

Usage:
    python sentiment_analysis.py <csv_path> [--env-file <path>]

The input CSV must contain:
    - Review_text   : the review text to analyse
    - Sentiment     : the ground-truth label (good / neutral / poor)

Outputs (written alongside the input CSV):
    - <base>_results_<timestamp>.csv   : all input columns + all graph output columns
    - <base>_metrics_<timestamp>.md    : classification report and supporting metrics
"""

import argparse
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

class OutputState(BaseModel):
    sentiment: str = Field(description="Predicted sentiment: good, neutral, or poor.")
    reasoning: str = Field(description="Reasoning behind the sentiment prediction.")


class ReviewState(BaseModel):
    agreement: bool = Field(description="True if the reviewer agrees with the sentiment.")
    review_reasoning: str = Field(description="Reviewer reasoning for agreement or disagreement.")


class OverallState(BaseModel):
    review_text: str = Field(description="The review text to analyse.")
    sentiment: str = Field(default="", description="Predicted sentiment.")
    reasoning: str = Field(default="", description="Reasoning for the sentiment.")
    agreement: bool = Field(default=False, description="Whether the reviewer agreed.")
    review_reasoning: str = Field(default="", description="Reviewer reasoning.")
    review_count: int = Field(default=0, description="Number of answer-review cycles completed.")


# ── Graph builder ──────────────────────────────────────────────────────────────

def _build_graph(llm: ChatOpenAI):
    llm_answer = llm.with_structured_output(OutputState)
    llm_review = llm.with_structured_output(ReviewState)

    def generate_answer(state: OverallState) -> dict:
        if state.review_count == 0:
            prompt = (
                "Analyze the following review and determine its sentiment "
                "(good, neutral, or poor). Provide reasoning for your analysis.\n\n"
                f'Review: "{state.review_text}"'
            )
        else:
            prompt = (
                "Analyze the following review and determine its sentiment "
                "(good, neutral, or poor). Provide reasoning for your analysis.\n\n"
                f'Review: "{state.review_text}"\n\n'
                f"Previous Reviewer's Agreement: {state.agreement}\n"
                f'Previous Reviewer\'s Reasoning: "{state.review_reasoning}"'
            )
        response = llm_answer.invoke(prompt)
        return {"sentiment": response.sentiment, "reasoning": response.reasoning}

    def review_sentiment(state: OverallState) -> dict:
        prompt = (
            "Review the following sentiment analysis and determine if you agree "
            "with the sentiment based on the reasoning provided. "
            "Provide your reasoning for agreement or disagreement.\n\n"
            f"Sentiment: {state.sentiment}\n"
            f"Reasoning: {state.reasoning}"
        )
        response = llm_review.invoke(prompt)
        return {
            "agreement": response.agreement,
            "review_reasoning": response.review_reasoning,
            "review_count": state.review_count + 1,
        }

    def should_continue(state: OverallState) -> str:
        if not state.agreement and state.review_count < 3:
            return "answer_node"
        return END

    builder = StateGraph(OverallState)
    builder.add_node("answer_node", generate_answer)
    builder.add_node("review_node", review_sentiment)
    builder.add_edge(START, "answer_node")
    builder.add_edge("answer_node", "review_node")
    builder.add_conditional_edges("review_node", should_continue, ["answer_node", END])
    return builder.compile()


# ── Metrics markdown builder ───────────────────────────────────────────────────

def _build_metrics_md(df: pd.DataFrame, csv_path: str) -> str:
    y_true = df["Sentiment"]
    y_pred = df["llm_sentiment"]

    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    report_str = classification_report(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    lines = [
        "# Sentiment Analysis Metrics",
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

    agreement_rate = df["agreement"].mean()
    avg_iters = df["review_count"].mean()
    review_count_dist = df["review_count"].value_counts().sort_index()

    lines += [
        "",
        "## LLM Self-Review Stats",
        "",
        f"- **Agreement rate (first pass):** {agreement_rate:.4f} ({agreement_rate * 100:.2f}%)",
        f"- **Average iterations per review:** {avg_iters:.2f}",
        "",
        "### Iteration count distribution",
        "",
        "| Iterations | Reviews |",
        "|---:|---:|",
    ]
    for iters, count in review_count_dist.items():
        lines.append(f"| {iters} | {count} |")

    return "\n".join(lines)


# ── Pipeline entry point ───────────────────────────────────────────────────────

def run(csv_path: str, env_file: Optional[str] = None) -> pd.DataFrame:
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)

    df = pd.read_csv(csv_path)

    missing = {"Review_text", "Sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)
    graph = _build_graph(llm)

    print(f"Processing {len(df)} reviews...")
    inputs = [{"review_text": str(text)} for text in df["Review_text"]]

    results = []
    errors = 0
    for i, inp in enumerate(inputs):
        try:
            result = graph.invoke(inp)
            results.append(result)
        except Exception as e:
            print(f"  [row {i}] ERROR: {type(e).__name__}: {str(e)[:120]} — skipping")
            results.append({
                "review_text": inp["review_text"],
                "sentiment": "error",
                "reasoning": str(e)[:200],
                "agreement": False,
                "review_reasoning": "",
                "review_count": 0,
            })
            errors += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(inputs)} processed ({errors} errors so far)...")

    print(f"Done. {len(results)} processed, {errors} errors.")

    graph_df = (
        pd.DataFrame(results)
        .rename(columns={"sentiment": "llm_sentiment"})
        .drop(columns=["review_text"])
    )

    final_df = pd.concat(
        [df.reset_index(drop=True), graph_df.reset_index(drop=True)], axis=1
    )

    base = os.path.splitext(csv_path)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{base}_results_{timestamp}.csv"
    metrics_path = f"{base}_metrics_{timestamp}.md"

    final_df.to_csv(results_path, index=False)
    print(f"Results saved  → {results_path}")

    md = _build_metrics_md(final_df, csv_path)
    with open(metrics_path, "w") as f:
        f.write(md)
    print(f"Metrics saved  → {metrics_path}")

    return final_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LangGraph sentiment analysis pipeline on a CSV file."
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
    args = parser.parse_args()
    run(args.csv_path, env_file=args.env_file)
