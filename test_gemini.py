"""One-off diagnostic: confirms the Gemini API key + SDK call work in isolation."""
import os
import traceback

api_key = os.environ.get("GEMINI_API_KEY")
print("Key present:", bool(api_key), "| length:", len(api_key) if api_key else 0)

try:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Say hello in one short sentence.",
    )
    print("SUCCESS. Response text:")
    print(resp.text)
except Exception as e:
    print("FAILED with exception:")
    traceback.print_exc()
