"""
Suggested-reply generator.

Approach: retrieval-augmented few-shot prompting (RAG), not fine-tuning.

Why RAG over fine-tuning:
- 48 examples is nowhere near enough to fine-tune an LLM without
  overfitting to phrasing; fine-tuning also costs a training job you
  can't iterate on inside 100 minutes.
- Why RAG over pure prompting with no examples: grounding the model in
  2-3 *actually similar past resolutions* (same category of issue,
  real company tone/policy) measurably improves consistency and factual
  correctness (e.g. "we refund within 5-7 business days") versus asking
  a general-purpose LLM to invent a policy from nothing.
- Retrieval uses TF-IDF cosine similarity, not embeddings API, so the
  whole pipeline runs with zero extra network calls or model downloads
  beyond the one LLM call for generation itself -- keeps it fast and
  dependency-light for a timed build.

Trade-off acknowledged in README: TF-IDF retrieval is lexical, not
semantic, so it can miss paraphrased-but-similar emails. For a
production system, swapping in an embedding-based retriever (e.g.
pgvector, which the author has used before) would be a straightforward
upgrade behind the same retrieve() interface.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from generator.llm_client import complete

DATASET_PATH = Path(__file__).parent.parent / "data" / "emails_dataset.json"

SYSTEM_PROMPT = (
    "You are a customer support agent for an e-commerce company. Write "
    "replies that are warm, specific, and action-oriented -- acknowledge "
    "the issue, state exactly what you're doing about it, and avoid "
    "generic corporate filler. Match the tone and structure of the "
    "example replies you're shown. Keep replies under 150 words."
)


class ReplyGenerator:
    def __init__(self, dataset_path: Path = DATASET_PATH):
        self.data = json.loads(Path(dataset_path).read_text())
        self.corpus = [row["incoming_email"] for row in self.data]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def retrieve(self, email: str, k: int = 3, exclude_id: int = None):
        """Return the k most similar past (incoming_email, ideal_reply) pairs."""
        query_vec = self.vectorizer.transform([email])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        results = []
        for i in ranked:
            row = self.data[i]
            if exclude_id is not None and row["id"] == exclude_id:
                continue
            results.append((row, float(sims[i])))
            if len(results) == k:
                break
        return results

    def build_prompt(self, email: str, examples) -> str:
        blocks = []
        for row, score in examples:
            blocks.append(
                f"Example incoming email:\n{row['incoming_email']}\n\n"
                f"Example reply that was sent:\n{row['ideal_reply']}"
            )
        examples_block = "\n\n---\n\n".join(blocks)
        return (
            f"Here are examples of past incoming emails and the replies our "
            f"team actually sent:\n\n{examples_block}\n\n"
            f"---\n\nNow write a reply for this new incoming email, following "
            f"the same tone and level of specificity.\n\n"
            f"INCOMING EMAIL:\n{email}\n\nReply:"
        )

    def generate(self, email: str, k: int = 3, exclude_id: int = None) -> dict:
        examples = self.retrieve(email, k=k, exclude_id=exclude_id)
        prompt = self.build_prompt(email, examples)
        reply = complete(prompt, system=SYSTEM_PROMPT)
        return {
            "generated_reply": reply,
            "retrieved_ids": [row["id"] for row, _ in examples],
            "retrieval_scores": [round(s, 4) for _, s in examples],
        }


if __name__ == "__main__":
    gen = ReplyGenerator()
    sample_email = (
        "Hi, I ordered a laptop sleeve last week (#HV-55555) and it still "
        "hasn't arrived. Tracking shows no movement in 5 days. What's going on?"
    )
    result = gen.generate(sample_email)
    print(json.dumps(result, indent=2))
