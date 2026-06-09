# Option C — Calibration & Selective Prediction on the Ambiguous Middle

**Research goal:** Turn the model's known failure mode — confident and wrong on `neutral` — into two results: (1) a calibration study showing GPT-4o-mini is systematically overconfident on minority/ambiguous sentiment, and (2) a selective-prediction system that abstains on low-confidence cases and recovers macro-F1 at the cost of coverage.

**Why this works:** In the current run the model has 99% precision on `good` but pushes 1,497 of 2,823 true-`neutral` reviews into `poor` (neutral F1 = 0.35). The question is whether the model *knows* when it is on shaky ground. If it does not, that is a miscalibration finding; if we can measure confidence at all, we can abstain on the low-confidence tail and report how much accuracy that buys back.

**Dataset:** Same as every other experiment — `full_dataset.csv` (22,641 reviews). Class-imbalanced (77% good / 12.5% neutral / 10.5% poor), so **macro-F1 is the primary metric** and per-class calibration is essential. See the dataset-sizing note below before running.

### Dataset sizing — natural distribution, with optional stratified subsample

Calibration is **distribution-dependent**: ECE and the risk–coverage curve must be measured against the model's real base rates to be deployment-honest. Two rules follow:

1. **Do not class-balance the primary evaluation.** Rebalancing to 33/33/33 reports calibration on a distribution the model never sees in production and breaks comparability with Options A/B. The natural distribution is the headline evaluation.
2. **Do not shrink to "a few" samples.** Calibration estimates are statistically hungry — ECE bins, then per-class splits (Step 6), then the answered subset at high abstention (Step 9) each subdivide the data. Small samples make every Option C metric high-variance and unreliable.

**Default: run the full 22,641 rows at natural proportions.** The set already contains ~2,823 `neutral` and ~2,370 `poor` reviews — thousands per minority class, ample for clean per-class calibration without any balancing.

**If cost forces a cut** (Step 2's vote-share signal is N calls/review), use a **proportional stratified subsample**, not a balanced one:
- Draw with `sklearn.model_selection.train_test_split(..., stratify=df['Sentiment'])` (or `df.groupby('Sentiment').sample(frac=...)`) so the 77/12.5/10.5 class ratio is preserved exactly.
- Fixed random seed; save once as `eval_subsample.csv` and reuse across all c-branches for comparability.
- Size so minorities stay well-populated: **target ≥5,000 rows** (≈600 neutral, ≈530 poor) as a floor; never drop below a few hundred per minority class.
- This is ~22% of the cost while keeping calibration statistically sound.

**Balanced set = optional secondary view only.** A 33/33/33 sample may be built *in addition* purely to make per-class reliability diagrams easier to read, but it must never replace the natural-distribution numbers and should still hold hundreds of samples per class. Report it as a supplementary analysis, clearly labeled.

**Suggested branches:**
- `exp-c1-confidence-elicitation` — Steps 1–3
- `exp-c2-calibration-analysis` — Steps 4–8
- `exp-c3-selective-prediction` — Steps 9–12

Hand any branch's steps to Claude Code together with the relevant context to generate runnable code.

---

## exp-c1-confidence-elicitation

### Step 1 — Lock the dataset and define "correct"
Use `full_dataset.csv` (22,641 reviews) at natural proportions, or the proportional `eval_subsample.csv` if cost-cutting (see Dataset sizing note above — never class-balance the primary set, never go below a few hundred per minority class). For every prediction define `is_correct = (llm_sentiment == Sentiment)`. The existing results file has predictions but **no confidence values**, so it cannot be reused directly — predictions are regenerated with confidence in Step 2.
**Output:** confirmed eval file + the correctness rule.

### Step 2 — Re-run the classifier capturing confidence signals
Modify the single-call classifier (the `exp-01-baseline-single-call` answerer) to emit, per review, the label **and** a confidence number. Implement all three signal sources so they can be compared:
- **`conf_logprob`** — log-probability of the predicted label token (request `logprobs` from the API). Note: structured output may hide token logprobs, so elicit the label as a plain constrained token (`good`/`neutral`/`poor`) to read its probability.
- **`conf_verbal`** — verbalized confidence: ask the model for a 0–100 self-reported score alongside the label.
- **`conf_vote`** — self-consistency vote share: N samples at temperature 0.6, fraction voting for the winning label (reuse `exp-05-self-consistency-voting` if already run).

Use the constrained label set (`good`/`neutral`/`poor`, Literal type) per the project convention.
**Output:** results CSV with `label`, `is_correct`, `conf_logprob`, `conf_verbal`, `conf_vote`.

### Step 3 — Sanity-check the signals
Confirm each confidence column is populated, lies in [0,1] (rescale verbalized 0–100 → 0–1), and varies across reviews (non-constant). Drop/repair any rows with missing confidence.
**Output:** a clean confidence table ready for analysis.

---

## exp-c2-calibration-analysis

### Step 4 — Build reliability diagrams
For each confidence signal, bin predictions into ~10 equal-width confidence bins and plot mean confidence (x) vs. actual accuracy (y) per bin. The diagonal is perfect calibration; points below it indicate overconfidence.
**Output:** three reliability plots (one per signal).

### Step 5 — Compute calibration metrics
For each signal compute **Expected Calibration Error (ECE)** and **Brier score**, overall.
**Output:** a table ranking the three confidence signals by ECE.

### Step 6 — Break calibration out per class
Recompute ECE separately for `good`, `neutral`, and `poor` predictions.
**Output:** the key finding table — expected pattern: `good` well-calibrated, `neutral` badly overconfident.

### Step 7 — Pick the best-calibrated signal
Select the confidence column with the lowest ECE; use it for all downstream steps.
**Output:** one chosen confidence signal.

### Step 8 — (Optional) Apply post-hoc calibration
Split the data (e.g. 70/30, fixed seed). Fit **temperature scaling** (or isotonic regression / Platt scaling) on the train split, apply to the test split, recompute ECE.
**Output:** before/after ECE quantifying the correction. Skip if Step 5 already shows good calibration.

---

## exp-c3-selective-prediction

### Step 9 — Build the risk–coverage curve
Sweep a confidence threshold τ from low to high. At each τ, "answer" only predictions with confidence ≥ τ and "abstain" on the rest. Record **coverage** (fraction answered) and **macro-F1 / accuracy on the answered subset**.
**Output:** a table of (τ, coverage, macro-F1) and the plotted risk–coverage curve.

### Step 10 — Summarize selective-prediction quality
From Step 9, report **macro-F1 at fixed coverage points** (e.g. 100%, 90%, 80%) and compute **AURC** (area under the risk–coverage curve) as a single headline number.
**Output:** the "macro-F1 rises from 0.47 → X at Y% coverage" claim.

### Step 11 — Analyze what gets abstained
Take the reviews below the chosen τ and report their class composition.
**Output:** evidence that abstentions concentrate on `neutral` (and, if Option B is done, on rating-text-divergent / `mixed` cases).

### Step 12 — Assemble results
Collect the per-class ECE table (Step 6), the reliability diagrams (Step 4), the risk–coverage figure (Step 9), and the F1-at-coverage numbers (Step 10) into one results section.
**Output:** the four claims below.

---

## Claims the paper can make (in rough order of strength)

1. GPT-4o-mini is **systematically overconfident on `neutral` sentiment** while well-calibrated on the dominant `good` class (per-class ECE, Step 6).
2. Among confidence signals, **vote-share / log-probs calibrate better than verbalized confidence** — a useful practitioner finding (Steps 5–7).
3. A simple **abstention threshold recovers X points of macro-F1 at Y% coverage**, giving a deployable precision/coverage knob (Steps 9–10).
4. The **abstained set coincides with the genuinely ambiguous middle**, linking calibration failure to the neutral/mixed problem (Step 11).

## Composition with A and B

- **Reuses Option A:** the best confidence signal (`conf_vote`) comes free from `exp-05-self-consistency-voting`.
- **Robustness check via Option B:** calibration is measured against the labels, so if labels are noisy, some apparent miscalibration is really label noise. Re-running Step 6 against Option B's text-faithful labels is a clean robustness check.

## Caveat to state up front
Calibration and abstention are evaluated against the star-derived labels. If those labels are noisy (Option B), part of the measured miscalibration and part of the abstention benefit reflect label noise rather than model uncertainty. Frame results accordingly and cross-check with Option B's corrected labels where possible.

---

## Appendix — Calibration metrics, worked examples

The one idea behind every metric here: **does the model's stated confidence match how often it is actually right?** If the model says "90% confident" on 100 reviews, a calibrated model gets ~90 of them right. If it gets only 60 right, it is **overconfident** — the hypothesized failure mode on `neutral`.

### A1. Reliability diagram (Step 4)
Group predictions into confidence bins; per bin, compare average confidence to actual accuracy. Example with 100 predictions in 3 bins:

| Bin (confidence) | # preds | Avg confidence | Actual accuracy | Gap |
|:---|---:|---:|---:|---:|
| 0.5–0.7 | 20 | 0.60 | 0.60 | 0.00 |
| 0.7–0.9 | 30 | 0.80 | 0.60 | 0.20 |
| 0.9–1.0 | 50 | 0.95 | 0.70 | 0.25 |

Plot confidence (x) vs accuracy (y). Perfect calibration is the diagonal (confidence = accuracy). Bins **below** the diagonal are overconfident — here the model claims 0.95 but is right only 70% of the time.

### A2. ECE — Expected Calibration Error (Step 5)
A single number: the average confidence–accuracy gap, weighted by bin size. Using the table above:

```
ECE = (20/100)·|0.60−0.60| + (30/100)·|0.80−0.60| + (50/100)·|0.95−0.70|
    = 0.2·0  +  0.3·0.20  +  0.5·0.25
    = 0 + 0.06 + 0.125
    = 0.185
```

**ECE = 0.185** → confidence is off from real accuracy by ~18.5 points on average. **Lower is better; 0 = perfect.** Computed per class in Step 6 — expected pattern: `good` ECE ≈ low (well-calibrated), `neutral` ECE ≈ high (overconfident).

### A3. Brier score (Step 5)
Mean squared error between predicted probability and outcome (1 = correct, 0 = wrong). Punishes confident mistakes hardest. Four examples:

| Review | Confidence | Correct? (outcome) | (conf − outcome)² |
|:---|---:|:---:|---:|
| A | 0.90 | yes (1) | 0.01 |
| B | 0.90 | **no (0)** | **0.81** |
| C | 0.60 | yes (1) | 0.16 |
| D | 0.55 | no (0) | 0.30 |

```
Brier = (0.01 + 0.81 + 0.16 + 0.30) / 4 = 0.32
```

**Lower is better; 0 = perfect.** Review B (confident but wrong) contributes a huge 0.81 — the neutral overconfidence problem in one cell.

### A4. ECE vs Brier — why report both
ECE measures *only* confidence-vs-accuracy match, so a lazy model that outputs "0.77" on everything (matching the 77% base rate) can post a low ECE while being useless. Brier adds **sharpness** — it rewards confidence that is both calibrated *and* decisive. Reporting both closes that loophole.

### A5. AURC (Step 10) — not a calibration metric
AURC scores the *selective-prediction* curve (accuracy as low-confidence cases are abstained), not calibration itself. It rides on top of A1–A4: good calibration is what makes the abstention threshold meaningful. Lower AURC = better risk–coverage trade-off.
