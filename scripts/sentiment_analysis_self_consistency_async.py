#!/usr/bin/env python3
"""
LangGraph sentiment analysis — exp-05 self-consistency voting, async concurrent.

Five independent LLM calls analyse each review in parallel. The final sentiment
is determined by majority vote across the five results.

Usage:
    python sentiment_analysis_self_consistency_async.py <csv_path> [--env-file <path>] [--concurrency <n>]

The input CSV must contain:
    - Review_text   : the review text to analyse
    - Sentiment     : the ground-truth label (good / neutral / poor)

Outputs (written alongside the input CSV):
    - <base>_self_consistency_results_<timestamp>.csv
    - <base>_self_consistency_metrics_<timestamp>.md
"""

import argparse
import asyncio
import operator
import os
from collections import Counter
from datetime import datetime
from typing import Annotated, List, Optional

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


class OverallState(BaseModel):
    review_text: str = Field(description="The review text to analyse.")
    sentiment: Annotated[List[str], operator.add] = Field(
        default=[], description="Accumulated sentiment votes from each LLM call."
    )
    reasoning: Annotated[List[str], operator.add] = Field(
        default=[], description="Accumulated reasoning from each LLM call."
    )
    final_sentiment: str = Field(default="", description="Majority-vote final sentiment.")
    final_reasoning: str = Field(default="", description="Summary of the vote result.")


# ── Graph builder ──────────────────────────────────────────────────────────────

def _build_graph(llm: ChatOpenAI):
    llm_answer = llm.with_structured_output(OutputState)

    def generate_answer(state: OverallState) -> dict:
        prompt = (
            "Analyze the following review and determine its sentiment "
            "(good, neutral, or poor). Provide reasoning for your analysis.\n\n"
            f'Review: "{state.review_text}"'
        )
        response = llm_answer.invoke(prompt)
        return {"sentiment": [response.sentiment], "reasoning": [response.reasoning]}

    def final_sentiment_node(state: OverallState) -> dict:
        counts = Counter(state.sentiment)
        final = counts.most_common(1)[0][0]
        total = len(state.sentiment)
        votes_str = ", ".join(f"{s}={c}" for s, c in counts.most_common())
        return {
            "final_sentiment": final,
            "final_reasoning": f"Majority vote ({total} LLMs): {votes_str} → {final}",
        }

    builder = StateGraph(OverallState)
    for i in range(1, 6):
        builder.add_node(f"answer_node_{i}", generate_answer)
        builder.add_edge(START, f"answer_node_{i}")
        builder.add_edge(f"answer_node_{i}", "final_sentiment_node")
    builder.add_node("final_sentiment_node", final_sentiment_node)
    builder.add_edge("final_sentiment_node", END)

    return builder.compile()


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _checkpoint_path(csv_path: str) -> str:
    return os.path.splitext(csv_path)[0] + "_self_consistency_checkpoint.csv"


def _load_checkpoint(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        done = {int(row["_idx"]): row.drop("_idx").to_dict() for _, row in df.iterrows()}
        print(f"Checkpoint found: {len(done)} rows already done — resuming.", flush=True)
        return done
    except Exception as e:
        print(f"Warning: checkpoint unreadable ({e}), starting fresh.", flush=True)
        return {}


def _save_checkpoint(path: str, done: dict) -> None:
    if not done:
        return
    rows = [{"_idx": idx, **result} for idx, result in sorted(done.items())]
    pd.DataFrame(rows).to_csv(path, index=False)


# ── Async concurrent runner ────────────────────────────────────────────────────

async def _process_one(graph, inp: dict, i: int, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            result = await graph.ainvoke(inp)
            return i, result, None
        except Exception as e:
            return i, None, e


async def _run_concurrent(graph, inputs: list, concurrency: int, done: dict, checkpoint_path: str):
    total = len(inputs)
    remaining = [(i, inp) for i, inp in enumerate(inputs) if i not in done]

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_process_one(graph, inp, i, semaphore) for i, inp in remaining]

    errors = sum(1 for r in done.values() if r.get("final_sentiment") == "error")
    completed = len(done)

    for coro in asyncio.as_completed(tasks):
        i, result, error = await coro
        completed += 1

        if error is not None:
            print(f"  [row {i}] ERROR: {type(error).__name__}: {str(error)[:120]} — skipping", flush=True)
            done[i] = {
                "review_text": inputs[i]["review_text"],
                "sentiment": "[]",
                "reasoning": "[]",
                "final_sentiment": "error",
                "final_reasoning": str(error)[:200],
            }
            errors += 1
        else:
            done[i] = result

        if completed % 500 == 0:
            print(f"  {completed}/{total} processed ({errors} errors so far)...", flush=True)
            _save_checkpoint(checkpoint_path, done)

    _save_checkpoint(checkpoint_path, done)
    return [done[i] for i in range(total)], errors


# ── Metrics markdown builder ───────────────────────────────────────────────────

def _build_metrics_md(df: pd.DataFrame, csv_path: str) -> str:
    y_true = df["Sentiment"]
    y_pred = df["final_sentiment"]

    labels = sorted(set(y_true.tolist() + y_pred.tolist()))
    report_str = classification_report(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    lines = [
        "# Sentiment Analysis Metrics — Self-Consistency Voting (exp-05)",
        "",
        f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Input:** `{csv_path}`  ",
        f"**Total reviews:** {len(df)}",
        "",
        "## Design",
        "",
        "Five independent LLM calls analyse each review in parallel. Final sentiment",
        "is determined by majority vote (ties broken by `Counter.most_common`).",
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
    for sent, count in df["final_sentiment"].value_counts().items():
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

    lines += [
        "",
        "## Voting Stats",
        "",
        "| Vote pattern | Count | % |",
        "|:---|---:|---:|",
    ]
    vote_patterns = df["final_reasoning"].value_counts()
    for pattern, count in vote_patterns.items():
        lines.append(f"| {pattern} | {count} | {count / len(df) * 100:.1f}% |")

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

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)
    graph = _build_graph(llm)

    chk_path = _checkpoint_path(csv_path)
    done = _load_checkpoint(chk_path)

    print(f"Processing {len(df)} reviews with concurrency={concurrency}...", flush=True)
    inputs = [{"review_text": str(text)} for text in df["Review_text"]]

    results_list, errors = asyncio.run(_run_concurrent(graph, inputs, concurrency, done, chk_path))

    print(f"Done. {len(results_list)} processed, {errors} errors.", flush=True)

    graph_df = (
        pd.DataFrame(results_list)
        .drop(columns=["review_text"])
    )

    final_df = pd.concat(
        [df.reset_index(drop=True), graph_df.reset_index(drop=True)], axis=1
    )

    base = os.path.splitext(csv_path)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{base}_self_consistency_results_{timestamp}.csv"
    metrics_path = f"{base}_self_consistency_metrics_{timestamp}.md"

    final_df.to_csv(results_path, index=False)
    print(f"Results saved  → {results_path}", flush=True)

    md = _build_metrics_md(final_df, csv_path)
    with open(metrics_path, "w") as f:
        f.write(md)
    print(f"Metrics saved  → {metrics_path}", flush=True)

    if os.path.exists(chk_path):
        os.remove(chk_path)
        print(f"Checkpoint removed → {chk_path}", flush=True)

    return final_df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run exp-05 self-consistency voting sentiment pipeline (async concurrent)."
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
