# Option A — Self-Review Architecture Ablation

**Research goal:** Determine whether *any* multi-call LLM scheme (sequential self-review or self-consistency voting) earns its extra cost over a single classification call on sentiment classification, and under what conditions a self-review critic adds real signal rather than rubber-stamping the answer.

**Core finding to test:** In the current pipeline the critic agrees with the answerer 99.99% of the time (avg 1.01 iterations) and provides no measurable accuracy gain. The leading hypothesis is that the critic is **blind to the original review text** — it only sees the predicted label and the answerer's reasoning, so it judges internal coherence, not correctness. These experiments isolate that variable and compare against simpler and stronger alternatives.

Each experiment below is a self-contained study. The level-2 header on each is the intended **git branch name** (kebab-case, copy-paste ready). Hand any single section to Claude Code together with the "Shared harness" section to generate a runnable script.

---

## Shared harness (applies to every experiment)

Every branch must reuse the **same** evaluation protocol so the master comparison table is valid. Build this once (suggested branch `exp-00-shared-harness`) and have the other branches import it.

**Evaluation set.** All experiments run on the **same dataset the current experiment used: `full_dataset.csv` (22,641 reviews)** — the identical file, no resampling, so results are directly comparable to the existing run and to each other. Note the set is class-imbalanced (77% good / 12.5% neutral / 10.5% poor); this is why **macro-F1 is the primary metric** (it weights the minority `neutral`/`poor` classes equally rather than letting the dominant `good` class inflate the score). Budget accordingly: at ~22.6k reviews × multiple calls each, several of these experiments are non-trivial in cost and runtime — see each experiment's call profile.

**Model defaults.** Answerer = `gpt-4o-mini`, `max_tokens=1024`. Temperature = `0` for all single-label/deterministic experiments; `0.6` only for the self-consistency experiment (noted there).

**Constrained label set.** Fix the `mixed`-label leakage seen in the full run: structured output must allow **exactly** `good | neutral | poor` (use a Pydantic `Literal["good","neutral","poor"]` field, not a free string). Reject/retry any out-of-vocabulary label.

**Metrics captured identically by all experiments** (write to `<branch>_metrics_<timestamp>.md` and append one row to a shared `results_master.csv`):

- Overall accuracy, **macro-F1 (primary metric)**, weighted-F1
- Per-class precision / recall / F1 (good, neutral, poor)
- Confusion matrix
- **Cost profile:** total LLM calls, calls per review, total input+output tokens, USD cost
- **Latency profile:** mean and p95 wall-clock per review
- Where applicable: **flip rate** (how often the final label differs from the first answerer label) and **cost per corrected error** (net errors fixed ÷ extra calls spent)

**Reference numbers from the current full-dataset run** (directly comparable — same file): accuracy 77.47%, macro-F1 0.47, neutral F1 0.35. This run corresponds to the `exp-02-review-critic-blind-text` architecture and can serve as its result row.

---

## exp-01-baseline-single-call

**Title:** Single-call baseline (no review, no voting)

**Research question:** What does one straight classification call achieve? This is the floor every other experiment is measured against — the whole ablation is meaningless without it.

**Architecture:** A single `answer` call per review. No `review_node`, no loop, no voting. One LLM call per review.

**Implementation:**
- Strip the graph down to the answerer only (or call the LLM directly with structured output).
- Prompt: classify the review as good / neutral / poor with brief reasoning. Use the constrained Literal label.
- Temperature 0.

**Calls/cost:** 1 call/review. This is the cheapest possible configuration and the cost denominator for "cost per corrected error" in later experiments.

**Outputs:** Standard harness metrics. Record this as the reference row in `results_master.csv`.

---

## exp-02-review-critic-blind-text

**Title:** Sequential self-review, critic blind to review text (current architecture = control)

**Research question:** Does the existing answer→review→retry loop beat the single-call baseline? Reproduce the rubber-stamp result as the controlled baseline of the loop family.

**Architecture:** The current `sentiment_analysis_async.py` design. `answer_node` predicts; `review_node` sees **only** the predicted label + answerer reasoning (NOT the review text) and decides agree/disagree; disagreement retries the answerer up to 3×.

**Implementation:**
- Port the existing two-node LangGraph as-is, plus the constrained-label fix and the flip-rate instrumentation from the shared harness.
- Keep the critic prompt text-blind exactly as in the current code (`review_sentiment` receives sentiment + reasoning only).

**Calls/cost:** ~2 calls/review (more on retries; expect ~1.01 avg iterations → ~2.02 calls).

**Expected result / hypothesis:** ≈ baseline macro-F1, ~99% agreement, near-zero flips → confirms the loop adds cost with no gain when the critic can't see the evidence.

---

## exp-03-review-critic-sees-text

**Title:** Sequential self-review, critic given the review text

**Research question:** If the critic can see the original review text, does it stop rubber-stamping and start correcting real errors?

**Architecture:** Identical to exp-02 but `review_node` receives the **original review text** in addition to the predicted label and reasoning.

**Implementation:**
- Add `review_text` to the critic prompt: "Here is the original review: '…'. The model predicted <label> because <reasoning>. Independently judge whether <label> is correct for this review."
- Everything else (retry logic, limit of 3, params) identical to exp-02 so the only changed variable is critic visibility of the text.

**Calls/cost:** ~2 calls/review (+retries). Critic input tokens rise (it now ingests the review text).

**Expected result / hypothesis:** Higher disagreement and flip rate than exp-02; net macro-F1 should improve if flips are corrective. This is the experiment that confirms or refutes the blindness hypothesis — the paper's central mechanism.

---

## exp-04-review-critic-independent-label

**Title:** Independent-critic review (critic labels before seeing the answer)

**Research question:** Does removing answer-anchoring — having the critic form its own opinion *first* — beat a critic that reacts to an existing answer (exp-03)?

**Architecture:** Two independent classifications of the same review by separate calls. The critic classifies the **review text from scratch** before being shown the answerer's label. Reconcile:
- If the two labels **agree** → accept, stop.
- If they **disagree** → either (a) trigger an arbitration call that sees both labels + both reasonings and picks the final label, or (b) retry the answerer with the disagreement surfaced (up to 3×). Pick one reconciliation strategy and state it in the script; (a) arbitration is recommended for a cleaner cost profile.

**Implementation:**
- `answer_node`: classify review → label_A, reasoning_A.
- `independent_node`: classify the **same review text** independently → label_B, reasoning_B (does NOT see label_A).
- `reconcile_node`: if label_A == label_B, done; else arbitration call sees review text + both opinions → final label.

**Calls/cost:** 2 calls/review when they agree; 3 when they disagree (arbitration). Report the agreement rate — it's a measure of intrinsic model disagreement on ambiguous (esp. neutral) reviews.

**Expected result / hypothesis:** Disagreement concentrates on neutral; arbitration recovers some neutral errors. Tests whether "independence" is what matters, beyond merely showing the critic the text.

---

## exp-05-self-consistency-voting

**Title:** Self-consistency voting (N independent samples, majority vote, no critic)

**Research question:** Does sampling the answerer N times at temperature > 0 and taking the majority vote beat both the single call and the self-review loop? And does the vote split give a usable confidence signal?

**Architecture:** No critic at all. Run the answerer **N independent times** on each review at temperature > 0; final label = majority vote. The vote distribution is recorded as a confidence estimate.

**Implementation:**
- N = 5 primary; also run N = 3 for the cost/accuracy trade-off curve.
- **Temperature = 0.6** (must be > 0 or all samples are identical and voting is pointless).
- Fire the N calls **concurrently** (no data dependency) so wall-clock ≈ one call.
- Record per-review vote split, e.g. `{good:4, neutral:1}`; define a confidence score (e.g. winning fraction) for later use in calibration/abstention work.
- Tie-breaking rule: state it explicitly (e.g. fall back to the lowest-temperature / first sample, or to a deterministic priority).

**Calls/cost:** N calls/review (5, or 3). More tokens than the loop, but fully parallel → lowest added latency. This is NOT the cheap option — it is the option that spends comparable calls more usefully (independence + free confidence signal). Report cost honestly against the single-call baseline.

**Expected result / hypothesis:** Macro-F1 ≥ self-review loop; vote-split entropy is high precisely on neutral reviews, yielding a confidence signal the sequential loop cannot produce.

---

## exp-06-cross-model-review

**Title:** Cross-model review (different model as critic)

**Research question:** Is the rubber-stamp effect caused by self-reinforcement (a model agreeing with itself)? Does using a *different* model as the critic break the agreement bias and add corrective signal?

**Architecture:** Sequential review like exp-03 (critic sees the review text), but the **critic is a different model** from the answerer.

**Implementation:**
- Answerer = `gpt-4o-mini`. Critic = a different model (e.g. a different OpenAI tier such as `gpt-4o`, or a non-OpenAI model via its API — make the critic model a config parameter).
- Critic sees review text + predicted label + reasoning, judges agree/disagree, retry up to 3× (mirror exp-03 so the only changed variable is critic model identity).
- Keep the answerer and all params identical to exp-03 for a clean comparison.

**Calls/cost:** ~2 calls/review (+retries). Critic calls may cost more per token if a larger model is used — report separately.

**Expected result / hypothesis:** Lower agreement rate and higher flip rate than same-model review; tests whether *model diversity* (not just text visibility) is the lever. A strong cross-model gain would be a notable result on its own.

---

## Master comparison (build after all branches run)

Aggregate every branch's row from `results_master.csv` into one table and one figure for the paper:

| Experiment | Calls/review | Macro-F1 | Neutral F1 | Flip rate | Cost/corrected error | p95 latency |
|:---|---:|---:|---:|---:|---:|---:|
| exp-01-baseline-single-call | 1 | — | — | n/a | n/a | — |
| exp-02-review-critic-blind-text | ~2 | — | — | — | — | — |
| exp-03-review-critic-sees-text | ~2 | — | — | — | — | — |
| exp-04-review-critic-independent-label | 2–3 | — | — | — | — | — |
| exp-05-self-consistency-voting (N=5) | 5 | — | — | n/a | — | — |
| exp-06-cross-model-review | ~2 | — | — | — | — | — |

**Headline the paper can claim, depending on outcome:**
- If exp-02 ≈ exp-01 → naive same-text-blind self-review is wasted compute (negative result).
- If exp-03/04 > exp-02 → self-review only works when the critic sees the evidence and/or labels independently (conditional result — the mechanism).
- If exp-05 ≥ the review family at comparable budget → simple voting dominates sequential critique and yields a free confidence signal (positive result + bridge to abstention work).

**Statistical rigor:** run each experiment with ≥3 seeds (or bootstrap the eval set) and report mean ± std on macro-F1, plus a significance test (e.g. paired bootstrap) for each pairwise comparison vs. the baseline. Reviewers at a peer-reviewed venue will expect this.
