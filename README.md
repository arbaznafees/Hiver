# Hiver Open Challenge — AI Suggested-Reply Generator + Evaluation System

Given an incoming customer support email, this system (1) generates a
suggested reply grounded in a dataset of past resolved emails, and (2)
scores how good that generated reply actually is, per-response and overall.

## Quick start

```bash
pip install -r requirements.txt

# Optional but recommended: enables real LLM generation + LLM-as-judge scoring.
export ANTHROPIC_API_KEY=sk-ant-...

# Regenerate the dataset (already committed, but reproducible):
python data/generate_dataset.py

# Run the full pipeline: generate + evaluate all 48 emails
python main.py run

# Or a smaller sample
python main.py run --n 10

# Or a single ad-hoc email
python main.py single "Hi, my order still hasn't arrived after 10 days, please help"
```

Without `ANTHROPIC_API_KEY` set, the generator falls back to a clearly-labeled
offline stub reply and the evaluator skips the LLM-judge signal — the whole
pipeline still runs end-to-end and produces `results.json`, just with a
weaker (non-LLM) generated reply and a reweighted composite score. This was
a deliberate choice so the submission is runnable by anyone grading it,
with or without a key.

## 1. The dataset (`data/`)

48 synthetic (incoming_email, ideal_reply) pairs across 8 categories:
refund requests, shipping delays, billing questions, angry complaints,
feature requests, cancellations, technical issues, and positive feedback.

**Why synthetic, not a public corpus:** the standard public option (Enron)
is internal corporate correspondence, not customer-support traffic — wrong
domain for what a Hiver-style tool actually does. Hand-authoring guarantees
every "ideal_reply" is genuinely a *good* reply (specific, on-policy, on-tone),
which matters a lot once that reply becomes the reference for scoring —
garbage references would make every downstream metric meaningless.

**Why it's representative, not just plausible:** each category uses
templated generation with randomized slot-fillers (names, order IDs,
products, dollar amounts, day counts) so examples in the same category
vary in specifics while sharing the underlying support pattern, and the
category mix (complaints, logistics, billing, positive feedback, etc.)
mirrors what actually lands in a shared support inbox rather than only
the easy cases.

**Honest limitation:** synthetic data can't capture the full messiness of
real customer language (typos, multi-issue emails, non-English mixed in).
Documented, not hidden — a real deployment would replace this with a
sample of anonymized historical tickets.

## 2. The generator (`generator/`)

**Approach: retrieval-augmented few-shot prompting (RAG), not fine-tuning,
not a bare zero-shot prompt.**

- *Why not fine-tuning:* 48 examples is far too small to fine-tune without
  overfitting to exact phrasing, and a training job doesn't fit inside a
  timed build where you need to iterate.
- *Why not zero-shot prompting alone:* a generic LLM prompt invents
  plausible-sounding but ungrounded policy ("we'll refund in 24 hours") that
  doesn't match what the company would actually promise. Showing the model
  2-3 real past resolutions for similar issues grounds tone *and* policy.
- *Retrieval mechanism:* TF-IDF cosine similarity over past incoming
  emails (`generator/reply_generator.py`). Chosen over an embeddings API to
  keep the pipeline dependency-light and runnable with zero extra network
  calls beyond the one generation call.
  **Trade-off:** TF-IDF is lexical, not semantic — it can miss a
  paraphrased-but-relevant past email. Swapping in embedding-based
  retrieval (e.g. pgvector) behind the same `retrieve()` interface would
  be the natural upgrade for production.
- The retrieved examples + the new email go into a single few-shot prompt
  (`SYSTEM_PROMPT` + `build_prompt` in `reply_generator.py`) sent to
  Claude (`claude-sonnet-4-6`).

## 3. The evaluation system (`eval/`) — the core of this challenge

**What "accurate" means here:** exact match against the reference reply is
the wrong bar — two replies can use completely different words and both
resolve the issue well. So instead of one score, four independent,
interpretable signals are measured and only combined at the end:

| Signal | What it catches | Why it's needed |
|---|---|---|
| `semantic_similarity` | Wildly off-topic replies | Cheap sanity floor (TF-IDF cosine vs. reference) |
| `key_fact_coverage` | Reply that's fluent but never mentions the actual order ID / specific issue | The most common real failure mode of suggested-reply tools — sounds good, says nothing specific |
| `structural_quality` | Missing greeting/sign-off, leftover template placeholders, refusals, absurd length | Cheap rule-based checks correlating with "would I actually send this" |
| `judge_score` | Wrong tone for the situation, answered the wrong question, incomplete resolution | The only signal that reasons like a human reviewer would — LLM graded against a fixed 4-axis rubric (`eval/judge_prompt.txt`): correctness, completeness, tone, conciseness |

**Composite score** = weighted average, weights shown per response in
`weights_used` so it's never a black box:
- With judge available: `judge_score 0.50, key_fact_coverage 0.25, semantic_similarity 0.15, structural_quality 0.10`
- Without judge (offline mode): re-normalized across the remaining three signals.

Judge gets the highest weight deliberately — it's the signal that can
actually tell "answered a different question" apart from "same words as
reference," which the cheaper signals structurally cannot.

**How the metric was validated (not just asserted):** three deliberately
injected bad replies were run through the evaluator to confirm the score
actually drops for real failure types, not just rewards generic fluency:

```
reference reply used as-is:      composite ≈ 64%  (high — it's genuinely the answer)
off-topic reply (wrong subject): composite ≈  9%  (correctly tanks)
truncated non-answer ("thanks"): composite ≈ 14%  (correctly tanks)
```

This is a directional sanity check, not a full calibration study — with
more time, the next step would be having 2-3 people independently rate a
held-out sample 1-5 and checking Spearman correlation with the composite
score, to confirm the automatic signals track human judgment beyond these
three hand-picked cases.

**Reporting:** `python main.py run` writes `results.json` containing
every per-response signal breakdown, the judge's rationale, and the
composite, plus an `overall` block with mean/median/stdev/min/max
composite and a mean-by-category breakdown (useful for spotting which
issue types the generator handles worst).

## AI tool usage disclosure

Built with Claude (Anthropic) as a pair-programming assistant for scaffolding
the dataset templates, the RAG generator, and the evaluator — including this
README. All design decisions (RAG over fine-tuning, the four-signal scoring
approach, the weighting, and the validation method) were made and reviewed
by the author; the assistant was used for drafting speed under the time limit.

## What I'd do with more time

- Embedding-based retrieval instead of TF-IDF.
- Real historical ticket data instead of synthetic, with PII scrubbing.
- Proper human-rating correlation study for the judge weight, rather than
  a 3-case spot check.
- A confidence/abstain signal — flag low-scoring generations for human
  review instead of auto-sending, which is closer to how this would
  actually ship inside a product like Hiver.
