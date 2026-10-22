
#!/usr/bin/env python3
"""Error-transition and inference-call analysis for Exp-01b versus Exp-04.

Run from the sentiment-analysis repository root:
    python scripts/revision_error_analysis.py

Uses saved full-dataset predictions; makes NO API requests.
"""
import argparse
from pathlib import Path
import pandas as pd

CLASSES = ("good", "neutral", "poor")
DEFAULT_A = "results/full_dataset/exp01b_full_dataset_results_20261018_220442.csv"
DEFAULT_B = "results/full_dataset/full_dataset_independent_critic_results_20260607_123551.csv"


def normalized(s):
    return s.fillna("<missing>").astype(str).str.strip().str.lower()


def scores(truth, pred, mask=None):
    if mask is not None:
        truth, pred = truth[mask], pred[mask]
    values = {}
    for c in CLASSES:
        tp = int(((truth == c) & (pred == c)).sum())
        fp = int(((truth != c) & (pred == c)).sum())
        fn = int(((truth == c) & (pred != c)).sum())
        values[c] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp01b", default=DEFAULT_A)
    parser.add_argument("--exp04", default=DEFAULT_B)
    parser.add_argument("--output-dir", default="results/full_dataset/revision_error_analysis")
    args = parser.parse_args()
    a, b = pd.read_csv(args.exp01b), pd.read_csv(args.exp04)
    required_a = {"Review_text", "Sentiment", "llm_sentiment"}
    required_b = {"Review_text", "Sentiment", "final_sentiment", "sentiment_1", "sentiment_2"}
    if required_a - set(a.columns) or required_b - set(b.columns):
        raise ValueError(f"Missing columns: exp01b={sorted(required_a-set(a.columns))}; exp04={sorted(required_b-set(b.columns))}")
    if len(a) != len(b):
        raise ValueError(f"Row-count mismatch ({len(a)} vs {len(b)}).")
    if not a["Review_text"].fillna("<missing>").astype(str).equals(b["Review_text"].fillna("<missing>").astype(str)):
        raise ValueError("Review texts do not align exactly; stopping.")
    truth_a, truth_b = normalized(a["Sentiment"]), normalized(b["Sentiment"])
    if not truth_a.equals(truth_b):
        raise ValueError("True labels do not align exactly; stopping.")
    if not truth_a.isin(CLASSES).all():
        raise ValueError("Unknown gold label found.")

    truth = truth_a
    p_a, p_b = normalized(a["llm_sentiment"]), normalized(b["final_sentiment"])
    first, second = normalized(b["sentiment_1"]), normalized(b["sentiment_2"])
    ca, cb = p_a.eq(truth), p_b.eq(truth)
    improved = ~ca & cb
    worsened = ca & ~cb
    both_correct = ca & cb
    both_wrong = ~ca & ~cb
    changed = p_a.ne(p_b)
    disagreed = first.ne(second)
    n = len(a)

    stats = {
        "total_reviews": n,
        "both_correct": int(both_correct.sum()),
        "exp01b_wrong_exp04_correct": int(improved.sum()),
        "exp01b_correct_exp04_wrong": int(worsened.sum()),
        "both_wrong": int(both_wrong.sum()),
        "net_additional_correct_exp04": int(improved.sum() - worsened.sum()),
        "changed_predictions": int(changed.sum()),
        "changed_predictions_pct": 100 * float(changed.mean()),
        "exp01b_invalid": int((~p_a.isin(CLASSES)).sum()),
        "exp04_invalid": int((~p_b.isin(CLASSES)).sum()),
        "exp04_first_second_disagreements": int(disagreed.sum()),
        "exp04_first_second_agreements": int((~disagreed).sum()),
        "exp04_first_answer_matches_exp01b": int(first.eq(p_a).sum()),
        "exp01b_total_api_calls_expected": n,
        "exp04_total_api_calls_expected": 2*n + int(disagreed.sum()),
        "extra_api_calls_exp04_expected": n + int(disagreed.sum()),
        "exp04_calls_per_review_expected": 2 + float(disagreed.mean()),
    }
    # Call totals derive from designed routing, not measured API usage/retries.
    if stats["net_additional_correct_exp04"] > 0:
        stats["extra_calls_per_net_additional_correct"] = stats["extra_api_calls_exp04_expected"] / stats["net_additional_correct_exp04"]
    else:
        stats["extra_calls_per_net_additional_correct"] = "not applicable (no positive net gain)"

    score_a, score_b = scores(truth, p_a), scores(truth, p_b)
    class_rows = []
    for label in CLASSES:
        m = truth.eq(label)
        class_rows.append({
            "true_sentiment": label,
            "support": int(m.sum()),
            "exp01b_correct": int((ca & m).sum()),
            "exp04_correct": int((cb & m).sum()),
            "exp01b_wrong_exp04_correct": int((improved & m).sum()),
            "exp01b_correct_exp04_wrong": int((worsened & m).sum()),
            "net_correct_gain_exp04": int((improved & m).sum() - (worsened & m).sum()),
            "exp01b_class_f1": score_a[label],
            "exp04_class_f1": score_b[label],
            "class_f1_delta_exp04_minus_exp01b": score_b[label]-score_a[label],
        })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(class_rows).to_csv(output / "class_comparison.csv", index=False)
    pd.DataFrame([{"metric": k, "value": v} for k,v in stats.items()]).to_csv(output / "summary.csv", index=False)
    # Save aggregate confusion transitions only; avoid publishing raw review text by default.
    transitions = pd.DataFrame({"truth": truth, "exp01b": p_a, "exp04": p_b}).groupby(
        ["truth", "exp01b", "exp04"], dropna=False
    ).size().reset_index(name="count")
    transitions.to_csv(output / "prediction_transitions_aggregate.csv", index=False)

    print(f"Aligned reviews: {n:,}. No API requests made.\n")
    for k in ["both_correct", "exp01b_wrong_exp04_correct", "exp01b_correct_exp04_wrong", "both_wrong", "net_additional_correct_exp04", "changed_predictions", "exp04_first_second_disagreements", "extra_api_calls_exp04_expected", "extra_calls_per_net_additional_correct"]:
        print(f"{k}: {stats[k]}")
    print("\nPer-true-class results:")
    print(pd.DataFrame(class_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved analysis in: {output}")
    print("NOTE: API call counts assume 1 call per node, no retries. They are not measured token or dollar costs.")

if __name__ == "__main__":
    main()
