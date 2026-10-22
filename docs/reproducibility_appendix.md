# Supplementary Appendix — Prompt Templates and Experimental Configuration

**Status:** Draft supplement for the revised sentiment-classification paper. Templates for Exp-01 and Exp-02–05 are transcribed from `adity-narvekar/sentiment-analysis` (main branch, inspected 18 October 2026). Exp-01b is transcribed from the team's local revision script `sentiment_analysis_single_call_reasoning_async.py`. Replace any model-version assertions with verified API run metadata before publication. No API calls were made to prepare this appendix.

## A. Common setup

- Data: `data/full_dataset.csv`, 22,641 reviews; required input columns `Review_text` and `Sentiment`.
- Gold labels: `good` (ratings 4–5), `neutral` (rating 3), `poor` (ratings 1–2).
- Answerer: `ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)` for Exp-01–05 and Exp-01b.
- Structured outputs: Pydantic response models supplied through `with_structured_output(...)`. The original `sentiment` field is a `str`, **not** a strict enum; out-of-vocabulary values could therefore occur.
- Each template below replaces `{review_text}` or other braces with the per-review values. Quotation marks, capitalization and wording reflect the code. The term *reasoning* here means a **generated rationale field**, not guaranteed access to a model's private chain of thought.
- Experiments all operate on the same rows. Revised comparisons additionally verified equal review text and ground truth across six prediction files.

## B. Exact prompts

### Exp-01: Single-call, label-only

```
Analyze the following review and classify its sentiment as good, neutral, or poor. Respond with the sentiment label only.

Review: "{review_text}"
```

Output schema: `sentiment: str`. One call; no reviewer or arbitration.

### Exp-01b: Single-call, reasoning-enabled (new control)

```
Analyze the following review and determine its sentiment (good, neutral, or poor). Provide reasoning for your analysis.

Review: "{review_text}"
```

Output schema: `sentiment: str`, `reasoning: str`. One call; no reviewer or arbitration. The initial answerer prompt and field descriptions match Exp-02 and Exp-04.

### Exp-02: Sequential critic blind to source text

**Initial answerer:** Exactly the Exp-01b prompt.

**Critic:**

```
Review the following sentiment analysis and determine if you agree with the sentiment based on the reasoning provided. Provide your reasoning for agreement or disagreement.

Sentiment: {sentiment}
Reasoning: {reasoning}
```

**Answerer after disagreement:**

```
Analyze the following review and determine its sentiment (good, neutral, or poor). Provide reasoning for your analysis.

Review: "{review_text}"

Previous Reviewer's Agreement: {agreement}
Previous Reviewer's Reasoning: "{review_reasoning}"
```

Answerer schema: `sentiment: str`, `reasoning: str`. Critic schema: `agreement: bool`, `review_reasoning: str`. The graph cycles `answer -> review -> answer` on disagreement, stopping after at most three review cycles. Critic does not receive the original review text.

### Exp-03: Sequential critic with access to source text

**Initial answerer and answerer-after-disagreement:** Exactly as in Exp-02.

**Critic:**

```
Study the following sentiment analysis and determine if you agree with the sentiment based on the reasoning provided. Provide your reasoning for agreement or disagreement.

Review: {review_text}
Sentiment: {sentiment}
Reasoning: {reasoning}
```

Answerer and critic schemas and maximum review-cycle count match Exp-02. **Important design note:** Exp-02 and Exp-03 differ in both original-text visibility and the critic's leading verb (`Review` vs `Study`). Describe the treatment accurately rather than asserting the prompt changed in source visibility alone.

### Exp-04: Two independent classifiers with conditional arbitration

**First classifier:** Exactly the Exp-01b prompt.

**Second classifier:** Exactly the Exp-01b prompt, issued independently (the two outputs are not shared before classification).

**Arbitrator (only on disagreement):**

```
Two independent sentiment analyses of the same review disagree. Review both analyses and the original review, then determine the correct final sentiment (good, neutral, or poor). Provide your reasoning.

Review: "{review_text}"

Analysis 1 — Sentiment: {sentiment_1}
Analysis 1 — Reasoning: {reasoning_1}

Analysis 2 — Sentiment: {sentiment_2}
Analysis 2 — Reasoning: {reasoning_2}
```

Answerer schema: `sentiment: str`, `reasoning: str`. Arbitrator schema: `final_sentiment: str`, `final_reasoning: str`. Accept an agreed label after two classifier calls; otherwise make one additional arbitration call. The saved full-dataset analysis reports 164 first/second disagreements and therefore estimates 45,446 nominal calls (2×22,641 + 164).

### Exp-05: Five-call self-consistency voting

**Each of the five classifier calls:** Exactly the Exp-01b prompt.

Outputs are aggregated via `Counter(state.sentiment).most_common(1)[0][0]`. Voting is performed in Python, with **no additional LLM prompt**. All five calls use temperature 0. The implementation specifies no independent tie-breaker beyond `Counter.most_common` order; ties are not possible with five valid votes distributed across two classes, but are possible with three classes (e.g., 2–2–1).

## C. Output fields and evaluation

| Experiment | Final prediction column | Calls per review (excluding possible API retries) |
|---|---|---|
| Exp-01 | `llm_sentiment` | 1 |
| Exp-01b | `llm_sentiment` | 1 |
| Exp-02 | `llm_sentiment` | 2 or more (extra cycles on disagreement) |
| Exp-03 | `llm_sentiment` | 2 or more (extra cycles on disagreement) |
| Exp-04 | `final_sentiment` | 2 or 3 |
| Exp-05 | `final_sentiment` | 5 |

Evaluation uses macro F1 across **only** `good`, `neutral`, and `poor`; invalid predictions count as wrong rather than creating additional averaged classes. The revised statistical script uses paired resampling and two-sided paired permutation tests with Holm adjustment for the five prespecified comparisons against Exp-01b. Include the precise script version, random seed, number of resamples/randomizations, and data files with the submission.

## D. Reporting cautions / outstanding metadata

1. The model string `gpt-4o-mini` does not identify a pinned model snapshot. Record API model/version metadata if recoverable; otherwise state the limitation and original/revision run dates explicitly.
2. Exp-01b was run in October 2026, whereas the original experiment outputs date to June 2026. Matching model aliases and parameters cannot guarantee the same underlying deployed model across those dates; acknowledge this possible time-related confound.
3. Prompt differences are not the only implementation differences; original scripts vary in output schemas, orchestration, error handling and checkpointing. Preserve exact code and package versions.
4. API-call counts are node-call estimates, not measured tokens, elapsed time or monetary cost.
5. The original code requests rationales. Avoid describing them as access to private/internal chain of thought.
6. Do not infer statistical equivalence from non-significant differences. The revised confidence intervals bound the possible observed advantage under the chosen sampling model.
7. Document the deterministic voting configuration (temperature 0) as a limitation; classic self-consistency often uses diversified samples.

## E. Source file map

- `scripts/sentiment_analysis_single_call_async.py` (Exp-01)
- `scripts/sentiment_analysis_single_call_reasoning_async.py` (Exp-01b; new, local)
- `scripts/sentiment_analysis_async.py` (Exp-02)
- `scripts/sentiment_analysis_critic_sees_text_async.py` (Exp-03)
- `scripts/sentiment_analysis_independent_critic_async.py` (Exp-04)
- `scripts/sentiment_analysis_self_consistency_async.py` (Exp-05)
- `scripts/revision_statistics.py` (revised comparisons; local)
- `scripts/revision_error_analysis.py` (error transitions; local)

**Repository:** https://github.com/adity-narvekar/sentiment-analysis
