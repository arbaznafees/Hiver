"""
Accuracy/evaluation system for generated support replies.

Core idea: "accurate" for a free-text reply is not exact-match, and it's
not a single number either -- a reply can be correct but curt, or warm
but wrong. So we score four independent, interpretable signals and only
combine them at the end, keeping every sub-score visible per response.

Signals:
  1. semantic_similarity  (0-1, automatic, cheap)
     TF-IDF cosine similarity between generated reply and the reference
     reply that was actually sent. Catches wildly off-topic replies.
     Deliberately NOT the primary signal: two replies can solve the same
     problem in very different words and both be "correct" -- this axis
     alone would unfairly punish good paraphrasing. It exists as a cheap
     sanity floor, not the final word.

  2. key_fact_coverage  (0-1, automatic)
     Extracts salient tokens from the incoming email that a good reply
     should acknowledge -- order IDs (regex), and top TF-IDF terms from
     the email itself (proxy for "the specific thing they're upset/asking
     about") -- and checks what fraction appear in the generated reply.
     This catches the most common real failure mode of suggested-reply
     tools: a fluent, on-brand reply that never actually mentions the
     customer's order number or specific complaint.

  3. structural_quality  (0-1, automatic, rule-based)
     Cheap heuristics that correlate with "would I actually send this":
     has a greeting, has a sign-off, isn't absurdly short or long
     relative to the reference, doesn't contain leftover template
     placeholders (e.g. "[insert", "{{"), doesn't contain an obvious
     LLM refusal ("I can't help with that").

  4. judge_score  (0-1, LLM-as-judge, the most expensive but most
     informative signal)
     Prompts an LLM with a fixed rubric (eval/judge_prompt.txt) to score
     correctness, completeness, tone, and conciseness against the actual
     incoming email -- this is the only signal that can catch "answered
     the wrong question" or "wrong tone for an angry customer" the way a
     human reviewer would. Skipped (reported as null) when no API key is
     configured, since it requires a real LLM call.

Composite score = weighted average of whichever signals are available,
renormalized so missing signals (e.g. no judge in offline mode) don't
silently drag the score down -- we report explicitly which signals were
used per response.

Weights (composite, when judge is available):
    judge_score            0.5   <- most informative, human-aligned
    key_fact_coverage       0.25   <- catches the #1 real failure mode
    semantic_similarity     0.15
    structural_quality      0.10

Validation approach (see README for full discussion): we spot-checked
the judge + composite score against 3 deliberately-injected bad replies
(off-topic, wrong tone, and a truncated non-answer) to confirm the score
actually drops for each failure type rather than just rewarding fluency.
"""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from generator.llm_client import complete

JUDGE_PROMPT_TEMPLATE = (Path(__file__).parent / "judge_prompt.txt").read_text()

WEIGHTS_WITH_JUDGE = {
    "judge_score": 0.50,
    "key_fact_coverage": 0.25,
    "semantic_similarity": 0.15,
    "structural_quality": 0.10,
}
WEIGHTS_NO_JUDGE = {
    "key_fact_coverage": 0.50,
    "semantic_similarity": 0.30,
    "structural_quality": 0.20,
}

ORDER_ID_RE = re.compile(r"#HV-\d{5}")
PLACEHOLDER_RE = re.compile(r"\[insert|\{\{|TODO|<[a-z_]+>", re.IGNORECASE)
REFUSAL_MARKERS = ["i can't help with that", "as an ai", "i cannot assist"]


def semantic_similarity(generated_reply: str, ideal_reply: str) -> float:
    vec = TfidfVectorizer(stop_words="english").fit([generated_reply, ideal_reply])
    matrix = vec.transform([generated_reply, ideal_reply])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def _top_terms(text: str, n: int = 6):
    try:
        vec = TfidfVectorizer(stop_words="english")
        matrix = vec.fit_transform([text])
        scores = matrix.toarray()[0]
        terms = vec.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
        return [t for t, s in ranked[:n] if s > 0]
    except ValueError:
        return []


def key_fact_coverage(incoming_email: str, generated_reply: str) -> float:
    facts = set(ORDER_ID_RE.findall(incoming_email))
    facts.update(_top_terms(incoming_email, n=6))
    if not facts:
        return 1.0  # nothing specific to check for
    reply_lower = generated_reply.lower()
    hit = sum(1 for f in facts if f.lower() in reply_lower)
    return hit / len(facts)


def structural_quality(generated_reply: str, ideal_reply: str) -> float:
    checks = []
    text = generated_reply.strip()
    lower = text.lower()

    has_greeting = bool(re.match(r"^(hi|hello|hey|dear)\b", lower))
    checks.append(has_greeting)

    has_signoff = any(s in lower for s in ["best,", "regards,", "thanks,", "thank you,"])
    checks.append(has_signoff)

    ratio = len(text) / max(len(ideal_reply), 1)
    reasonable_length = 0.25 <= ratio <= 3.0
    checks.append(reasonable_length)

    no_placeholders = not PLACEHOLDER_RE.search(text)
    checks.append(no_placeholders)

    no_refusal = not any(m in lower for m in REFUSAL_MARKERS)
    checks.append(no_refusal)

    return sum(checks) / len(checks)


def judge_score(incoming_email: str, ideal_reply: str, generated_reply: str):
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        incoming_email=incoming_email,
        ideal_reply=ideal_reply,
        generated_reply=generated_reply,
    )
    raw = complete(prompt, max_tokens=200)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").split("\n", 1)[1] if "\n" in cleaned else cleaned
        parsed = json.loads(cleaned)
        axes = ["correctness", "completeness", "tone", "conciseness"]
        avg = sum(parsed[a] for a in axes) / (5 * len(axes))
        return {
            "available": True,
            "raw_scores": {a: parsed[a] for a in axes},
            "rationale": parsed.get("rationale", ""),
            "normalized": avg,
        }
    except Exception as e:
        return {"available": False, "error": str(e), "raw_output": raw}


def evaluate_response(incoming_email: str, ideal_reply: str, generated_reply: str,
                       use_judge: bool = True) -> dict:
    sig = {
        "semantic_similarity": round(semantic_similarity(generated_reply, ideal_reply), 4),
        "key_fact_coverage": round(key_fact_coverage(incoming_email, generated_reply), 4),
        "structural_quality": round(structural_quality(generated_reply, ideal_reply), 4),
    }

    judge = judge_score(incoming_email, ideal_reply, generated_reply) if use_judge else {"available": False}
    if judge.get("available"):
        sig["judge_score"] = round(judge["normalized"], 4)
        weights = WEIGHTS_WITH_JUDGE
    else:
        weights = WEIGHTS_NO_JUDGE

    composite = sum(sig[k] * w for k, w in weights.items() if k in sig)
    composite = composite / sum(w for k, w in weights.items() if k in sig)

    return {
        "signals": sig,
        "judge_detail": judge,
        "weights_used": weights,
        "composite_score": round(composite, 4),
        "composite_score_pct": round(composite * 100, 1),
    }
