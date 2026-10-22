#!/usr/bin/env python3
"""Paired statistical analysis for the six sentiment experiments.

Compares exp-01b against exp-01/02/03/04/05 on the same 22,641 reviews.
Uses paired bootstrap percentile confidence intervals and paired permutation
(randomization) tests for two-sided p-values, adjusted by Holm's procedure.

No LLM calls. Run from repository root:
    python scripts/revision_statistics.py
Optional: --bootstrap 10000 --permutations 10000 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "full_dataset"
EXPERIMENTS = {
    "exp-01": ("full_dataset_results_baseline_20260611_221728.csv", "llm_sentiment"),
    "exp-01b": ("exp01b_full_dataset_results_20261018_220442.csv", "llm_sentiment"),
    "exp-02": ("full_dataset_results_20260603_220307.csv", "llm_sentiment"),
    "exp-03": ("full_dataset_results_critic_sees_text_20260605_123121.csv", "llm_sentiment"),
    "exp-04": ("full_dataset_independent_critic_results_20260607_123551.csv", "final_sentiment"),
    "exp-05": ("full_dataset_self_consistency_results_20260608_152353.csv", "final_sentiment"),
}
LABELS = ("good", "neutral", "poor")
VALID = {label: n for n, label in enumerate(LABELS)}

def encode(col: pd.Series) -> np.ndarray:
    return col.fillna("").astype(str).str.strip().str.lower().map(VALID).fillna(3).to_numpy(dtype=np.int8)


def f1_from_confusion(conf: np.ndarray) -> np.ndarray:
    """Return macro F1 for confusion matrices [..., true 3, predicted 4]."""
    tp = np.stack([conf[..., j, j] for j in range(3)], axis=-1)
    supports = conf.sum(axis=-1)
    predicted = conf.sum(axis=-2)[..., :3]
    denom = supports + predicted
    return np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0).mean(axis=-1)


def paired_contingency(truth: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bincount(truth * 16 + a * 4 + b, minlength=64).reshape(4, 4, 4)[:3]


def score_pair_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Counts shape [..., true 3, predA 4, predB 4]."""
    ca = counts.sum(axis=-1)
    cb = counts.sum(axis=-2)
    return f1_from_confusion(ca), f1_from_confusion(cb)


def bootstrap_ci(counts: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    probs = counts.reshape(-1) / counts.sum()
    lows = []
    for _ in range(0, n_boot, 500):
        k = min(500, n_boot - len(lows))
        samples = rng.multinomial(counts.sum(), probs, size=k).reshape(k, 3, 4, 4)
        fa, fb = score_pair_counts(samples)
        lows.extend((fb - fa).tolist())
    return tuple(float(x) for x in np.quantile(lows, [0.025, 0.975]))


def permutation_p(counts: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Paired label swaps under the exchangeability null, two-sided p-value."""
    obs_a, obs_b = score_pair_counts(counts)
    observed = abs(float(obs_b - obs_a))
    exceed = 0
    flat = counts.reshape(-1)
    for start in range(0, n_perm, 500):
        k = min(500, n_perm - start)
        swapped = rng.binomial(flat, 0.5, size=(k, 48)).reshape(k, 3, 4, 4)
        others = counts[None] - swapped
        # Non-swapped keep A=i,B=j; swapped become A=j,B=i.
        permuted = others + swapped.swapaxes(-1, -2)
        fa, fb = score_pair_counts(permuted)
        exceed += int(np.count_nonzero(np.abs(fb - fa) >= observed - 1e-12))
    return (exceed + 1) / (n_perm + 1)


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(m, dtype=float)
    largest = 0.0
    for rank, idx in enumerate(order):
        largest = max(largest, min(1.0, (m-rank) * pvals[idx]))
        adjusted[idx] = largest
    return adjusted.tolist()


def read_all():
    frames = {}
    for name, (filename, prediction_col) in EXPERIMENTS.items():
        path = BASE / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
        df = pd.read_csv(path)
        needed = {"Review_text", "Sentiment", prediction_col}
        if not needed.issubset(df.columns):
            raise ValueError(f"{name}: missing columns {needed - set(df.columns)}")
        frames[name] = df
    reference = frames["exp-01b"]
    for name, df in frames.items():
        if len(df) != len(reference):
            raise ValueError(f"{name}: row-count mismatch: {len(df)} vs {len(reference)}")
        for col in ("Review_text", "Sentiment"):
            if not df[col].fillna("").astype(str).equals(reference[col].fillna("").astype(str)):
                raise ValueError(f"{name}: {col} mismatch at same row positions; review alignment failed")
    truth = encode(reference["Sentiment"])
    if np.any(truth == 3):
        raise ValueError("Unexpected reference truth labels")
    predictions = {name: encode(df[col]) for name, (filename, col) in EXPERIMENTS.items() for df in [frames[name]]}
    return truth, predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.bootstrap < 100 or args.permutations < 100:
        ap.error("Use at least 100 bootstrap/permutation samples")
    truth, preds = read_all()
    print(f"Aligned all six experiments: {len(truth):,} reviews, texts and true labels identical.\n")
    print("Macro F1:")
    for name, values in preds.items():
        conf = np.bincount(truth*4 + values, minlength=16).reshape(4,4)[:3]
        print(f"  {name:<7}: {float(f1_from_confusion(conf)):.6f} ({int(np.sum(values==3))} invalid predictions)")
    baseline = preds["exp-01b"]
    rows = []
    for i, name in enumerate(k for k in EXPERIMENTS if k != "exp-01b"):
        counts = paired_contingency(truth, baseline, preds[name])
        a, b = score_pair_counts(counts)
        rng_ci = np.random.default_rng(args.seed + 100 * i)
        rng_p = np.random.default_rng(args.seed + 100 * i + 1)
        lo, hi = bootstrap_ci(counts, args.bootstrap, rng_ci)
        p = permutation_p(counts, args.permutations, rng_p)
        rows.append({"comparison":f"exp-01b vs {name}", "difference_B_minus_A":float(b-a), "CI_95_low":lo, "CI_95_high":hi,"p_two_sided":p})
    adjusted = holm_adjust([r["p_two_sided"] for r in rows])
    for row, adj in zip(rows, adjusted):
        row["p_holm"] = adj
        row["significant_holm_0_05"] = adj < 0.05
    out = pd.DataFrame(rows)
    dest = BASE / "revision_exp01b_statistical_comparisons.csv"
    out.to_csv(dest, index=False, float_format="%.8f")
    print("\nPaired tests: 95% percentile bootstrap CIs; 2-sided permutation p-values; Holm correction:")
    print(out.to_string(index=False, formatters={k:lambda v:f"{v:+.6f}" for k in ["difference_B_minus_A", "CI_95_low", "CI_95_high"]}))
    print(f"\nSaved: {dest}")
    print("Note: Lack of statistical significance is not proof of equivalence.")

if __name__ == "__main__":
    main()
