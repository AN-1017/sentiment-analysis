"""Exp-01b: single-call reasoning baseline for the sentiment study.

One GPT-4o-mini inference per input review. The instruction and structured
response match the initial answerer call in exp-02/exp-04, but no critique,
arbitration, or voting occurs.

Run from the repository root:
 python scripts/sentiment_analysis_single_call_reasoning_async.py data/test_sample.csv
 python scripts/sentiment_analysis_single_call_reasoning_async.py data/full_dataset.csv
"""

import argparse
import asyncio
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CLASSES = ["good", "neutral", "poor"]


class OutputState(BaseModel):
    # Keep these fields identical to the initial answerer in exp-02/exp-04.
    sentiment: str = Field(description="Predicted sentiment: good, neutral, or poor.")
    reasoning: str = Field(description="Reasoning behind the sentiment prediction.")


class OverallState(BaseModel):
    review_text: str = Field(description="The review text to analyse.")
    sentiment: str = Field(default="", description="Predicted sentiment.")
    reasoning: str = Field(default="", description="Reasoning for the sentiment.")


def build_graph(llm: ChatOpenAI):
    answerer = llm.with_structured_output(OutputState)

    def generate_answer(state: OverallState) -> dict:
        # Intentionally exactly the same initial prompt as exp-02 and exp-04.
        prompt = (
            "Analyze the following review and determine its sentiment "
            "(good, neutral, or poor). Provide reasoning for your analysis.\n\n"
            f'Review: "{state.review_text}"'
        )
        result = answerer.invoke(prompt)
        return {"sentiment": result.sentiment, "reasoning": result.reasoning}

    builder = StateGraph(OverallState)
    builder.add_node("answer_node", generate_answer)
    builder.add_edge(START, "answer_node")
    builder.add_edge("answer_node", END)
    return builder.compile()


async def process_one(graph, review_text: str, index: int, semaphore: asyncio.Semaphore):
    async with semaphore:
        try:
            result = await graph.ainvoke({"review_text": review_text})
            return index, result.get("sentiment", "error"), result.get("reasoning", ""), ""
        except Exception as exc:
            return index, "error", "", f"{type(exc).__name__}: {str(exc)[:250]}"


async def run_async(graph, reviews: list[str], concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [process_one(graph, text, i, semaphore) for i, text in enumerate(reviews)]
    results = [None] * len(reviews)
    for completed, future in enumerate(asyncio.as_completed(tasks), start=1):
        i, sentiment, reasoning, error = await future
        results[i] = {"llm_sentiment": sentiment, "reasoning": reasoning, "error_detail": error}
        if completed % 500 == 0 or completed == len(reviews):
            n_errors = sum(1 for r in results if r is not None and r["error_detail"])
            print(f"Processed {completed}/{len(reviews)} (API errors: {n_errors})", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Single-call reasoning baseline exp-01b")
    parser.add_argument("csv_path")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None, help="Local smoke test only; omit for final full run")
    args = parser.parse_args()
    if args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--concurrency and --limit must be positive")

    load_dotenv(args.env_file, override=True) if args.env_file else load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY not found. Configure a local .env file (do not commit it).")

    source = Path(args.csv_path)
    df = pd.read_csv(source)
    missing = {"Review_text", "Sentiment"} - set(df.columns)
    if missing:
        parser.error(f"Missing input columns: {sorted(missing)}")
    if args.limit is not None:
        df = df.iloc[:args.limit].copy()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)
    graph = build_graph(llm)
    records = asyncio.run(run_async(graph, df["Review_text"].astype(str).tolist(), args.concurrency))
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save under results/, not under the input data directory.
    subset = "smoke" if args.limit is not None or "test" in source.stem.lower() else "full_dataset"
    destination = Path("results") / ("test_sample" if subset != "full_dataset" else "full_dataset")
    destination.mkdir(parents=True, exist_ok=True)
    csv_output = destination / f"exp01b_{subset}_results_{timestamp}.csv"
    metrics_output = destination / f"exp01b_{subset}_metrics_{timestamp}.md"
    out.to_csv(csv_output, index=False)

    actual = out["Sentiment"].astype(str).str.lower().str.strip()
    predicted = out["llm_sentiment"].astype(str).str.lower().str.strip()
    invalid = (~predicted.isin(CLASSES)).sum()
    macro = f1_score(actual, predicted, labels=CLASSES, average="macro", zero_division=0)
    acc = accuracy_score(actual, predicted)
    report = classification_report(actual, predicted, labels=CLASSES, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=CLASSES)
    metrics = (
        "# Exp-01b — single-call reasoning baseline\n\n"
        f"Dataset: `{source}`  \nRows: {len(out)}  \nModel: `gpt-4o-mini`  "
        f"\nTemperature: `0`  \nMax tokens: `1024`  \nAPI calls attempted: {len(out)}"
        f"\nInvalid/error predictions: {invalid}  \nMacro F1 (3 classes): {macro:.6f}"
        f"\nAccuracy: {acc:.6f}\n\n## Classification report\n\n```\n{report}\n```\n"
        f"\n## Confusion matrix\n\nOrder: {CLASSES}\n\n```\n{matrix}\n```\n"
    )
    metrics_output.write_text(metrics, encoding="utf-8")
    print(f"Macro F1: {macro:.6f} | Accuracy: {acc:.6f} | Invalid: {invalid}")
    print(f"Predictions: {csv_output}\nMetrics: {metrics_output}")
    if invalid:
        print("WARNING: Invalid/error predictions detected; inspect error_detail and resolve before final comparison.")


if __name__ == "__main__":
    main()
