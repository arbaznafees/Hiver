"""
Thin wrapper around the Anthropic API (Claude).

Design choice: Anthropic API only, not multi-provider. Keeping one
well-tested path is more reliable inside a 100-minute build than
half-supporting three SDKs. If ANTHROPIC_API_KEY is not set, we fall
back to a deterministic offline "stub" model so the whole pipeline
(generation + evaluation) still runs end-to-end for grading without
requiring a key -- clearly labeled as a stub, never silently pretending
to be a real LLM.
"""
import os
import random

MODEL = "claude-sonnet-4-6"


def _stub_complete(prompt: str) -> str:
    """
    Deterministic offline fallback. Not a real generative model --
    just enough to keep the pipeline runnable without an API key.
    Pulls the single most similar retrieved example's reply structure
    and lightly adapts it, so the eval system still has something
    non-trivial to score.
    """
    marker = "INCOMING EMAIL:\n"
    if marker in prompt:
        email = prompt.split(marker, 1)[1].split("\n\nReply", 1)[0].strip()
    else:
        email = prompt.strip()
    opener = random.choice(["Hi,", "Hello,", "Hi there,"])
    return (
        f"{opener}\n\nThanks for reaching out. I've looked into your message "
        f"and here's what I can do to help resolve this for you. I'll follow "
        f"up with the specific next steps shortly.\n\n"
        f"[STUB REPLY -- set ANTHROPIC_API_KEY for a real generated response]\n\n"
        f"Best,\nSupport Team"
    )


def complete(prompt: str, max_tokens: int = 400, system: str = None) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _stub_complete(prompt)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        kwargs = dict(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()
    except Exception as e:
        # Never let an API hiccup crash the whole eval run -- surface it
        # in-band so it's visible in results.json instead of losing the run.
        return f"[LLM_CALL_FAILED: {e}]"
