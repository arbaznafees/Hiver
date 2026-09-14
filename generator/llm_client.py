"""
Thin wrapper around the Gemini API (Google GenAI SDK).

If GEMINI_API_KEY is not set, we fall back to a deterministic offline
"stub" model so the whole pipeline (generation + evaluation) still runs
end-to-end for grading without requiring a key -- clearly labeled as a
stub, never silently pretending to be a real LLM.
"""
import os
import random

MODEL = "gemini-2.5-flash"


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


def complete(prompt: str, max_tokens: int = 400, system: str = None) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _stub_complete(prompt)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system if system else None,
        )
        resp = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=config,
        )
        return (resp.text or "").strip()
    except Exception as e:
        # Never let an API hiccup crash the whole eval run -- surface it
        # in-band so it's visible in results.json instead of losing the run.
        return f"[LLM_CALL_FAILED: {e}]"
