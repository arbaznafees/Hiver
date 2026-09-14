"""
Thin wrapper around the Gemini API (Google GenAI SDK).

If GEMINI_API_KEY is not set, we fall back to a deterministic offline
"stub" model so the whole pipeline (generation + evaluation) still runs
end-to-end for grading without requiring a key -- clearly labeled as a
stub, never silently pretending to be a real LLM.

Rate limiting: the free tier for gemini-3.5-flash-lite allows 15
requests/minute. Each evaluated email makes up to 2 calls (one to
generate the reply, one for the LLM-judge score), so a full 48-email
run is ~96 calls -- well over the burst limit if fired back-to-back.
We self-pace with a minimum gap between calls, and auto-retry on 429
(RESOURCE_EXHAUSTED) using the server's own suggested retry delay when
it's present in the error message, so a long run just gets slower
instead of failing partway through.
"""
import os
import random
import re
import time

MODEL = "gemini-3.5-flash-lite"

# Free tier: 15 req/min -> stay safely under that with a floor gap.
MIN_SECONDS_BETWEEN_CALLS = 4.5
MAX_RETRIES = 4

_last_call_ts = 0.0


def _stub_complete(prompt: str) -> str:
    """
    Deterministic offline fallback. Not a real generative model --
    just enough to keep the pipeline runnable without an API key.
    """
    opener = random.choice(["Hi,", "Hello,", "Hi there,"])
    return (
        f"{opener}\n\nThanks for reaching out. I've looked into your message "
        f"and here's what I can do to help resolve this for you. I'll follow "
        f"up with the specific next steps shortly.\n\n"
        f"[STUB REPLY -- set GEMINI_API_KEY for a real generated response]\n\n"
        f"Best,\nSupport Team"
    )


def _wait_for_rate_limit():
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_ts = time.monotonic()


def _extract_retry_delay(error_msg: str, default: float = 20.0) -> float:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+)", error_msg)
    if match:
        return float(match.group(1)) + 2.0  # small buffer
    return default


def complete(prompt: str, max_tokens: int = 400, system: str = None) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _stub_complete(prompt)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        system_instruction=system if system else None,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        _wait_for_rate_limit()
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            if is_rate_limit and attempt < MAX_RETRIES:
                delay = _extract_retry_delay(msg)
                print(f"[RATE LIMIT] Waiting {delay:.0f}s before retry "
                      f"{attempt}/{MAX_RETRIES}...", flush=True)
                time.sleep(delay)
                continue
            # Never let an API hiccup crash the whole eval run -- surface
            # it in-band (results.json) AND to the console.
            import sys
            print(f"[WARNING] Gemini call failed after {attempt} attempt(s): {e}",
                  file=sys.stderr)
            return f"[LLM_CALL_FAILED: {e}]"
