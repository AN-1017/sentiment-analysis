#!/usr/bin/env python3
"""Paired bootstrap comparison of two experiment results CSVs.

Computes 3-class macro-F1 (good/neutral/poor) for each system, a 95%
bootstrap CI per system, and a paired bootstrap test on the difference.
Out-of-vocabulary predictions (e.g. 'mixed', 'error') are kept as wrong
predictions: they count against the true class's recall but are never a
prediction class — consistent with the corrected metrics of 2026-06-11.

Usage:
    python scripts/paired_bootstrap.py <results_A.csv> <results_B.csv> \
        [--pred-col-a final_sentiment] [--pred-col-b final_sentiment] \
        [--truth-col Sentiment] [--n-boot 10000] [--seed 0]

Both CSVs must come from runs over the same input file in the same row
order (true for all runs derived from full_dataset.csv).
"""

import argparse
import sys

import numpy as np
import pandas as pd

CLASSES = ["good", "neutral", "poor"]
PRED_COL_CANDIDATES = ["final_sentiment", "llm_sentiment", "predicted_sentiment"]


def detect_pred_col(df: pd.DataFrame, override: str | None, path: str) -> str:
    if override:
        if override not in df.columns:
            sys.exit(f"Column '{override}' not in {path}. Columns: {list(df.columns)}")
        return override
    for c in PRED_COL_CANDIDATES:
        if c in df.columns:
            return c
    sys.exit(f"No prediction column found in {path}. Columns: {list(df.columns)}\n"
             f"Pass --pred-col-a/--pred-col-b explicitly.")


def encode(series: pd.Series) -> np.ndarray:
    """good/neutral/poor -> 0/1/2; anything else -> 3 (counts as wrong)."""
    s = series.astype(str).str.strip().str.lower()
    out = np.full(len(s), 3, dtype=np.int8)
    for i, c in enumerate(CLASSES):
        out[s == c] = i
    return out


def macro_f1(t: np.ndarray, p: np.ndarray) -> float:
    f1s = []
    for c in range(3):
        tp = np.sum((p == c) & (t == c))
        fp = np.sum((p == c) & (t != c))
        fn = np.sum((p != c) & (t == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f1s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_a", help="Results CSV for system A (e.g. the baseline)")
    ap.add_argument("csv_b", help="Results CSV for system B (the challenger)")
    ap.add_argument("--pred-col-a", default=None)
    ap.add_argument("--pred-col-b", default=None)
    ap.add_argument("--truth-col", default="Sentiment")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)
    if len(df_a) != len(df_b):
        sys.exit(f"Row count mismatch: {len(df_a)} vs {len(df_b)} — "
                 f"runs must cover the same input file.")

    col_a = detect_pred_col(df_a, args.pred_col_a, args.csv_a)
    col_b = detect_pred_col(df_b, args.pred_col_b, args.csv_b)
    for df, path in ((df_a, args.csv_a), (df_b, args.csv_b)):
        if args.truth_col not in df.columns:
            sys.exit(f"Truth column '{args.truth_col}' not in {path}.")

    t_a = encode(df_a[args.truth_col])
    t_b = encode(df_b[args.truth_col])
    if not np.array_equal(t_a, t_b):
        sys.exit("Ground-truth columns differ between the two files — "
                 "rows are not aligned. Sort both runs identically first.")
    t = t_a
    pa = encode(df_a[col_a])
    pb = encode(df_b[col_b])
    n = len(t)

    oov_a, oov_b = int(np.sum(pa == 3)), int(np.sum(pb == 3))
    f1_a, f1_b = macro_f1(t, pa), macro_f1(t, pb)

    print(f"N = {n} reviews")
    print(f"System A ({col_a}, {args.csv_a}): macro-F1 = {f1_a:.4f}"
          f"{f'  [{oov_a} out-of-vocab preds]' if oov_a else ''}")
    print(f"System B ({col_b}, {args.csv_b}): macro-F1 = {f1_b:.4f}"
          f"{f'  [{oov_b} out-of-vocab preds]' if oov_b else ''}")

    rng = np.random.default_rng(args.seed)
    s_a = np.empty(args.n_boot)
    s_b = np.empty(args.n_boot)
    for i in range(args.n_boot):
        idx = rng.integers(0, n, n)          # same resample for both = paired
        ti = t[idx]
        s_a[i] = macro_f1(ti, pa[idx])
        s_b[i] = macro_f1(ti, pb[idx])
    diffs = s_b - s_a

    ci = lambda x: np.percentile(x, [2.5, 97.5])
    lo_a, hi_a = ci(s_a)
    lo_b, hi_b = ci(s_b)
    lo_d, hi_d = ci(diffs)
    p_one_sided = float(np.mean(diffs <= 0))
    significant = lo_d > 0 or hi_d < 0

    print(f"\nBootstrap ({args.n_boot} resamples, seed {args.seed}):")
    print(f"  A: {f1_a:.4f}  95% CI [{lo_a:.4f}, {hi_a:.4f}]")
    print(f"  B: {f1_b:.4f}  95% CI [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"\nPaired difference (B - A): {f1_b - f1_a:+.4f}  "
          f"95% CI [{lo_d:+.4f}, {hi_d:+.4f}]")
    print(f"P(B <= A) = {p_one_sided:.4f}")
    print(f"Conclusion: {'SIGNIFICANT at the 95% level' if significant else 'NOT significant — treat the systems as equivalent'}")


if __name__ == "__main__":
    main()
