#!/usr/bin/env python3
"""Planned comparisons including exp-01b (single-call reasoning baseline).

Reads the existing full-dataset results CSVs plus the exp-01b results CSV and
reports, using one shared paired bootstrap (same resampled rows for every
system):

  1. Per-system macro F1 with 95% CI, per-class F1, accuracy, OOV count
  2. Primary planned comparisons (prompt effect, architecture effect)
  3. The paper's original 10 pairwise tests, with exact two-sided p-values
     and Holm-adjusted p-values
  4. An equivalence check against a +/- margin (TOST at alpha = 0.05, i.e. the
     90% CI must lie inside the margin)
  5. Run-to-run noise: agreement between repeated calls of the SAME prompt
     (exp-01b vs exp-04's two independent first-pass calls)
  6. Label-flip analysis for exp-04 arbitration

Usage:
    python scripts/analyze_exp01b_comparisons.py <exp01b_results.csv> \
        [--results-dir results/full_dataset] [--n-boot 10000] [--seed 0] \
        [--margin 0.005] [--out results/full_dataset/exp01b_analysis.md]

The exp-01b CSV needs only the columns `Sentiment` and `llm_sentiment`.
OOV predictions (anything outside good/neutral/poor) count as wrong and never
as a predicted class, matching scripts/paired_bootstrap.py.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

CLASSES = ["good", "neutral", "poor"]


def encode(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    out = np.full(len(s), 3, dtype=np.int8)
    for i, c in enumerate(CLASSES):
        out[s == c] = i
    return out


def cm_from(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.bincount(t.astype(np.int64) * 4 + p, minlength=12).reshape(3, 4)


def f1s_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm[:, :3]).astype(float)
    fp = cm[:, :3].sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.where(tp > 0, 2 * tp / np.where(denom == 0, 1, denom), 0.0)


def macro_f1(t, p) -> float:
    return float(f1s_from_cm(cm_from(t, p)).mean())


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def two_sided_p(d: np.ndarray) -> float:
    b = len(d)
    lo = (np.sum(d <= 0) + 1) / (b + 1)
    hi = (np.sum(d >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def find_one(results_dir: str, pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not hits:
        sys.exit(f"No file matching {pattern} in {results_dir}")
    return hits[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp01b_csv")
    ap.add_argument("--results-dir", default="results/full_dataset")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = args.results_dir
    f01 = find_one(rd, "full_dataset_results_baseline_*.csv")
    f02 = find_one(rd, "full_dataset_results_2*.csv")
    f03 = find_one(rd, "full_dataset_results_critic_sees_text_*.csv")
    f04 = find_one(rd, "full_dataset_independent_critic_results_*.csv")
    f05 = find_one(rd, "full_dataset_self_consistency_results_*.csv")

    e01, e02, e03 = pd.read_csv(f01), pd.read_csv(f02), pd.read_csv(f03)
    e04, e05 = pd.read_csv(f04), pd.read_csv(f05)
    e01b = pd.read_csv(args.exp01b_csv)

    n = len(e01)
    for name, df in (("exp-02", e02), ("exp-03", e03), ("exp-04", e04),
                     ("exp-05", e05), ("exp-01b", e01b)):
        if len(df) != n:
            sys.exit(f"Row count mismatch: {name} has {len(df)}, exp-01 has {n}")
    t = encode(e01["Sentiment"])
    for name, df in (("exp-02", e02), ("exp-03", e03), ("exp-04", e04),
                     ("exp-05", e05), ("exp-01b", e01b)):
        if not np.array_equal(t, encode(df["Sentiment"])):
            sys.exit(f"Ground truth rows are not aligned for {name}")

    preds = {
        "exp-01": encode(e01["llm_sentiment"]),
        "exp-01b": encode(e01b["llm_sentiment"]),
        "exp-02": encode(e02["llm_sentiment"]),
        "exp-03": encode(e03["llm_sentiment"]),
        "exp-04": encode(e04["final_sentiment"]),
        "exp-05": encode(e05["final_sentiment"]),
        "exp-04 call 1 (no review)": encode(e04["sentiment_1"]),
        "exp-04 call 2 (no review)": encode(e04["sentiment_2"]),
    }
    names = list(preds)

    # ── shared paired bootstrap ────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.n_boot) for k in names}
    for i in range(args.n_boot):
        idx = rng.integers(0, n, n)
        ti = t[idx]
        for k in names:
            boot[k][i] = macro_f1(ti, preds[k][idx])

    lines = []
    w = lines.append
    w("# exp-01b analysis — planned comparisons")
    w("")
    w(f"N = {n} reviews · {args.n_boot} paired bootstrap resamples · seed {args.seed} · "
      f"equivalence margin ±{args.margin}")
    w("")
    w("OOV predictions count as wrong and never as a predicted class.")
    w("")

    # 1. per-system table
    w("## 1. Per-system results")
    w("")
    w("| System | OOV | Macro F1 | 95% CI | good F1 | neutral F1 | poor F1 | Accuracy |")
    w("|:---|---:|---:|:---|---:|---:|---:|---:|")
    for k in names:
        p = preds[k]
        cm = cm_from(t, p)
        f = f1s_from_cm(cm)
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        acc = np.trace(cm[:, :3]) / n
        w(f"| {k} | {int((p == 3).sum())} | {f.mean():.4f} | [{lo:.4f}, {hi:.4f}] | "
          f"{f[0]:.3f} | {f[1]:.3f} | {f[2]:.3f} | {acc * 100:.2f}% |")
    w("")

    def compare(a, b):
        d = boot[b] - boot[a]
        point = macro_f1(t, preds[b]) - macro_f1(t, preds[a])
        lo, hi = np.percentile(d, [2.5, 97.5])
        lo90, hi90 = np.percentile(d, [5, 95])
        return dict(a=a, b=b, diff=point, lo=lo, hi=hi, p=two_sided_p(d),
                    equiv=(lo90 > -args.margin and hi90 < args.margin))

    def table(rows, adj_label):
        ps = np.array([r["p"] for r in rows])
        adj = holm(ps)
        w(f"| Comparison (A → B) | Diff (B−A) | 95% CI | p (two-sided) | {adj_label} | "
          f"Significant after Holm? | Equivalent within ±{args.margin}? |")
        w("|:---|---:|:---|---:|---:|:---|:---|")
        for r, ap_ in zip(rows, adj):
            w(f"| {r['a']} → {r['b']} | {r['diff']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
              f"{r['p']:.4f} | {ap_:.4f} | {'**Yes**' if ap_ < 0.05 else 'No'} | "
              f"{'Yes' if r['equiv'] else 'No'} |")
        w("")

    # 2. primary planned comparisons
    w("## 2. Primary planned comparisons (Holm over these 5)")
    w("")
    w("- exp-01 → exp-01b isolates the **prompt format** (label-only vs reasoning prompt).")
    w("- exp-01b → exp-02…05 isolates the **review / arbitration / voting step**.")
    w("")
    primary = [compare("exp-01", "exp-01b")] + [
        compare("exp-01b", k) for k in ("exp-02", "exp-03", "exp-04", "exp-05")]
    table(primary, "Holm p")

    # 3. the paper's original 10 tests
    w("## 3. The paper's original pairwise tests (Holm over these 10)")
    w("")
    base = ["exp-01", "exp-02", "exp-03", "exp-04", "exp-05"]
    orig = []
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            orig.append(compare(base[i], base[j]))
    table(orig, "Holm p")

    # 4. run-to-run noise
    w("## 4. Run-to-run noise (same prompt, same model, temperature 0)")
    w("")
    same = ["exp-01b", "exp-04 call 1 (no review)", "exp-04 call 2 (no review)"]
    w("| Run A | Run B | Rows with different labels | Agreement |")
    w("|:---|:---|---:|---:|")
    for i in range(len(same)):
        for j in range(i + 1, len(same)):
            diff = int(np.sum(preds[same[i]] != preds[same[j]]))
            w(f"| {same[i]} | {same[j]} | {diff} | {(1 - diff / n) * 100:.2f}% |")
    f1s = [macro_f1(t, preds[k]) for k in same]
    w("")
    w(f"Macro F1 of the three identical-prompt single calls: "
      f"{', '.join(f'{x:.4f}' for x in f1s)} (range {max(f1s) - min(f1s):.4f}).")
    w("")

    # 5. flips
    w("## 5. Label changes")
    w("")
    for a, b in (("exp-01", "exp-01b"), ("exp-01b", "exp-02"), ("exp-01b", "exp-04")):
        pa, pb = preds[a], preds[b]
        changed = pa != pb
        w(f"- **{a} → {b}:** {int(changed.sum())} rows differ; "
          f"{int(np.sum(changed & (pa == t) & (pb != t)))} got worse, "
          f"{int(np.sum(changed & (pa != t) & (pb == t)))} got better.")
    p1, p2, pf = (preds["exp-04 call 1 (no review)"], preds["exp-04 call 2 (no review)"],
                  preds["exp-04"])
    dis = p1 != p2
    w(f"- **exp-04 arbitration:** {int(dis.sum())} disagreements between the two calls; "
      f"vs call 1, arbitration fixed {int(np.sum(dis & (p1 != t) & (pf == t)))} and "
      f"broke {int(np.sum(dis & (p1 == t) & (pf != t)))}.")
    d_arb = boot["exp-04"] - boot["exp-04 call 1 (no review)"]
    lo, hi = np.percentile(d_arb, [2.5, 97.5])
    w(f"- **exp-04 full pipeline vs its own call 1:** "
      f"{macro_f1(t, pf) - macro_f1(t, p1):+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}].")
    w("")

    # 6. exp-01b confusion
    cm = cm_from(t, preds["exp-01b"])
    w("## 6. exp-01b confusion matrix (rows = true, columns = predicted)")
    w("")
    w("| True \\ Pred | good | neutral | poor | OOV |")
    w("|:---|---:|---:|---:|---:|")
    for i, c in enumerate(CLASSES):
        w(f"| **{c}** | " + " | ".join(str(int(x)) for x in cm[i]) + " |")
    w("")
    pd_ = pd.Series(preds["exp-01b"]).map({0: "good", 1: "neutral", 2: "poor", 3: "oov"})
    w("Predicted distribution: " + ", ".join(
        f"{k} {v} ({v / n * 100:.1f}%)" for k, v in pd_.value_counts().items()))
    w("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
