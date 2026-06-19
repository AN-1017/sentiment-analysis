# Option F — Sentiment as a Latent Trajectory (LLM Emissions + HMM)

**Research goal:** Model a review as a *sequence* of per-segment sentiments rather than a single label. Hidden states are the true per-segment sentiments; the LLM is the **emission model** (a noisy sensor that reads each segment); a Hidden Markov Model supplies the **transitions** that capture how sentiment flows within a review (e.g. a `good→poor` flip). The bet: explicitly representing mixed "good-but-bad" reviews fixes the `neutral` class that flat classifiers collapse.

**Why this works:** A 3-star review like "love the fabric, but it shrank after one wash" is textually two-sided — segments `[good, poor]`. A whole-review classifier sees the blend and predicts `neutral`/`poor` inconsistently (current neutral F1 = 0.35). A trajectory model represents the `good→poor` transition explicitly and can map "contains both strong good and strong poor" to `neutral`/`mixed` in a principled way.

**Dataset:** Same family as the other experiments — `full_dataset.csv` (22,641 reviews). **Important:** this experiment makes *per-segment* LLM calls (≈3–5× the number of reviews), so the proportional `eval_subsample.csv` from Option C is the sensible default for cost. Use the identical frozen file across all f-branches for comparability. Macro-F1 is the primary metric.

**Suggested branches:**
- `exp-f1-segmentation-emissions` — Steps 1–3 (build the observation sequences)
- `exp-f2-hmm-fit-decode` — Steps 4–10 (define, fit, decode the HMM)
- `exp-f3-eval-ablations` — Steps 11–14 (evaluate and stress-test)

---

## The model in one paragraph

Each review is a hidden Markov chain. **Hidden states** `S = {good, neutral, poor}` are the true per-segment sentiments. **Transitions** `A[i,j] = P(state_{t+1}=j | state_t=i)` capture within-review sentiment flow. The **LLM is the emission model**: it reads each segment and outputs a sentiment distribution that acts as a noisy observation of the hidden state. The HMM denoises the sequence (forward–backward / Viterbi); the decoded state path is aggregated into one review label, and the sequence posteriors provide a per-review confidence signal.

---

## exp-f1-segmentation-emissions

### Step 1 — Pick the data
Use `full_dataset.csv` at natural proportions, or (recommended for cost) the proportional `eval_subsample.csv` from Option C — per-segment calls multiply LLM spend. Freeze one file and reuse it across all f-branches.
**Output:** confirmed eval file.

### Step 2 — Segment each review
Split every review into an ordered sequence of units `o_1…o_T`. Start with sentence tokenization (spaCy or nltk `sent_tokenize`); for finer trajectories, additionally split on contrastive conjunctions ("but, however, although, though"). Preserve order.
**Output:** `review_id → [segment_1, …, segment_T]`.

### Step 3 — Get LLM emissions per segment
Run the classifier on **each segment** to produce a probability distribution over {good, neutral, poor}. Reuse a confidence signal from Option C (log-probs or vote-share preferred over verbalized). Keep the constrained label set.
**Output:** per review, an ordered list of 3-dim probability vectors — the observation sequence the HMM consumes.

---

## exp-f2-hmm-fit-decode

### Step 4 — Define the HMM
States `S = {good, neutral, poor}`. Specify initial distribution π (3-dim), transition matrix A (3×3), and emission model B. Choose the observation representation:
- **Hard:** `o_t = argmax` of the LLM vector; emission is a confusion matrix `B[j,k] = P(LLM reads k | true state j)`.
- **Soft:** use the LLM probability vectors directly as emission likelihoods (divide by the state prior so they act as `P(segment | state)` up to a constant).

Decide hard vs soft here and record it.
**Output:** an initialized HMM specification.

### Step 5 — Anchor emissions to keep states interpretable
Initialize B near-identity (state `good` mostly emits `good`, etc.) so latent states map to named sentiments. This prevents the **label-switching problem**, where EM finds states that don't correspond to real sentiments.
**Output:** anchored emission init.

### Step 6 — Estimate parameters
Recommended: **fix/anchor the emission matrix; learn transitions A and initial π via Baum-Welch (EM)** on all observation sequences. Use `hmmlearn` (categorical/multinomial) or `pomegranate` (custom emissions) — both run in log-space (forward/backward via log-sum-exp), which keeps long sequences numerically stable. Fixed seed, several random restarts on the transition init; keep the highest log-likelihood fit.
**Output:** fitted A, π (and B if learned).

### Step 7 — Verify state semantics
After fitting, inspect the emission/confusion rows to confirm each latent state still maps to its intended sentiment. If a state drifted, re-anchor and refit. Do not proceed until states are interpretable.
**Output:** validated, interpretable HMM.

### Step 8 — Decode each review
For every review compute: **Viterbi** → most likely hidden-state path + path log-probability; **forward–backward** → per-segment posteriors `γ_t(j)` (per-segment confidence); **forward** → total sequence log-likelihood.
**Output:** per-review state path, posteriors, and log-likelihood.

### Step 9 — Aggregate the path into one review label
Define an explicit, tunable rule mapping the decoded state sequence → one review label:
- **Majority** — most frequent state in the Viterbi path.
- **Final-state** — the last segment's state (captures "ended badly").
- **Mixed-aware** — if the path contains both high-posterior `good` and high-posterior `poor` segments, map to `neutral` (or a new `mixed` label). Most likely to fix the weak class.

Pick one as primary; the others become ablations (Step 12).
**Output:** one predicted label per review.

### Step 10 — Derive review-level confidence
Compute a per-review confidence from the sequence — e.g. mean posterior of the chosen label across segments, or normalized Viterbi path log-prob. This is a new confidence signal that drops directly into Option C's calibration analysis.
**Output:** confidence column per review.

---

## exp-f3-eval-ablations

### Step 11 — Evaluate against ground truth
Score review-level predictions vs star labels with the shared harness: macro-F1 (primary), per-class P/R/F1, confusion matrix. Compare against two baselines on the **same reviews**: `exp-01-baseline-single-call` (flat whole-review classifier) and the current review-loop run.
**Output:** comparison table; headline question — does trajectory modeling raise `neutral` F1?

### Step 12 — Ablations
Vary one knob at a time, all else fixed:
- (a) transitions **on vs off** (off = independent per-segment → isolates the value of the Markov structure vs plain aggregation);
- (b) **hard vs soft** emissions;
- (c) segmentation **granularity** (sentence vs clause);
- (d) **aggregation rule** (Step 9 variants);
- (e) with vs without the `mixed` mapping.
**Output:** ablation table attributing the gain to specific components.

### Step 13 — Qualitative analysis
Pull example reviews where the flat baseline errs but the trajectory model is right (the "love it *but*…" cases) and show the decoded path. Inspect the **learned transition matrix**: do `good→poor` transitions correlate with true `neutral`/3-star labels?
**Output:** worked examples + transition-matrix figure.

### Step 14 — Assemble results and claims
**Output:** the claims below.

---

## Claims the paper can make (in rough order of strength)

1. Modeling **within-review sentiment trajectories improves macro-F1**, concentrated on the `neutral`/mixed class.
2. The gain comes from the **Markov transitions, not just segmentation** (from ablation 12a) — or, if not, the honest alternative headline: "segmentation + aggregation helps; the Markov part does not."
3. **HMM posteriors calibrate better** than flat LLM log-probs as a confidence signal (links to Option C).
4. **`good→poor` transitions are an interpretable signal** for genuinely mixed reviews that star ratings mislabel (links to Option B).

## Composition with other options

- **Reuses Option C:** per-segment emissions reuse Option C's confidence signal; HMM posteriors feed back into Option C's calibration comparison.
- **Reuses Option C's subsample:** runs on the same proportional `eval_subsample.csv` for cost and comparability.
- **Links to Option B:** the mixed-review / `good→poor` story is direct evidence for the rating-vs-text label-noise argument.

## Risks and caveats to state up front

- **Cost:** per-segment LLM calls multiply spend (~3–5× reviews). Start on the subsample.
- **Label switching:** EM can find states that don't map to sentiments — mitigated by anchoring emissions (Step 5); verify in Step 7.
- **No segment-level ground truth:** emissions can't be directly validated; evaluation is review-level only, relying on the anchored confusion matrix.
- **Possible null result:** if transitions-off (ablation 12a) matches transitions-on, the Markov structure adds nothing — still publishable, but reframe the headline accordingly.
