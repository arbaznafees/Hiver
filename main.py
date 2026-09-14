"""
End-to-end runner.

Usage:
  python main.py run [--n N] [--no-judge]
      Generate + evaluate replies for N sample emails (default: all 48).
      For each email, we exclude its own pair from retrieval so the
      generator isn't shown the exact reference reply it's being
      scored against.

  python main.py single "some incoming email text"
      Generate + evaluate a single ad-hoc email (no reference reply,
      so only signals that don't need a reference are meaningful --
      judge score and structural_quality still run).

Output: results.json (per-response) + a printed summary table.
"""
import argparse
import json
import statistics
from pathlib import Path

from generator.reply_generator import ReplyGenerator
from eval.evaluator import evaluate_response

DATA_PATH = Path(__file__).parent / "data" / "emails_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def run_full(n: int = None, use_judge: bool = True):
    dataset = json.loads(DATA_PATH.read_text())
    if n:
        dataset = dataset[:n]

    gen = ReplyGenerator()
    results = []

    for row in dataset:
        gen_out = gen.generate(row["incoming_email"], k=3, exclude_id=row["id"])
        eval_out = evaluate_response(
            incoming_email=row["incoming_email"],
            ideal_reply=row["ideal_reply"],
            generated_reply=gen_out["generated_reply"],
            use_judge=use_judge,
        )
        results.append({
            "id": row["id"],
            "category": row["category"],
            "incoming_email": row["incoming_email"],
            "ideal_reply": row["ideal_reply"],
            "generated_reply": gen_out["generated_reply"],
            "retrieved_ids": gen_out["retrieved_ids"],
            **eval_out,
        })
        print(f"[{row['id']:>3}] {row['category']:<20} composite={eval_out['composite_score_pct']:>5}%")

    scores = [r["composite_score"] for r in results]
    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["composite_score"])

    overall = {
        "n_responses": len(results),
        "mean_composite": round(statistics.mean(scores), 4) if scores else None,
        "median_composite": round(statistics.median(scores), 4) if scores else None,
        "stdev_composite": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
        "min_composite": round(min(scores), 4) if scores else None,
        "max_composite": round(max(scores), 4) if scores else None,
        "mean_by_category": {
            cat: round(statistics.mean(v), 4) for cat, v in by_category.items()
        },
        "judge_used": use_judge,
    }

    output = {"overall": overall, "per_response": results}
    RESULTS_PATH.write_text(json.dumps(output, indent=2))

    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")


def run_single(email_text: str, use_judge: bool = True):
    gen = ReplyGenerator()
    gen_out = gen.generate(email_text, k=3)
    # No ground-truth reply for an ad-hoc email; use the top retrieved
    # example's reply as a loose reference for the similarity signals.
    reference = None
    if gen_out["retrieved_ids"]:
        dataset = json.loads(DATA_PATH.read_text())
        by_id = {row["id"]: row for row in dataset}
        reference = by_id[gen_out["retrieved_ids"][0]]["ideal_reply"]

    eval_out = evaluate_response(
        incoming_email=email_text,
        ideal_reply=reference or "",
        generated_reply=gen_out["generated_reply"],
        use_judge=use_judge,
    )
    print(json.dumps({"generated_reply": gen_out["generated_reply"], **eval_out}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--n", type=int, default=None)
    p_run.add_argument("--no-judge", action="store_true")

    p_single = sub.add_parser("single")
    p_single.add_argument("email_text")
    p_single.add_argument("--no-judge", action="store_true")

    args = parser.parse_args()
    if args.command == "run":
        run_full(n=args.n, use_judge=not args.no_judge)
    elif args.command == "single":
        run_single(args.email_text, use_judge=not args.no_judge)
