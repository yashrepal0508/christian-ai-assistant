"""
Quick eval runner — prints each test case, the moderation result, and (optionally) the LLM response.
Run: python eval/run_eval.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.moderation import pre_filter, classify_image_prompt
from core.bible_rag import retrieve_relevant_verses

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")


def run():
    with open(EVAL_PATH) as f:
        cases = json.load(f)

    print(f"\n{'='*70}")
    print(f"  CHRISTIAN AI ASSISTANT — EVALUATION ({len(cases)} cases)")
    print(f"{'='*70}\n")

    passed = 0
    for case in cases:
        print(f"[{case['id']}] {case['category'].upper()}")
        print(f"  Prompt: {case['prompt'][:80]}...")

        if case["category"] == "image":
            result = classify_image_prompt(case["prompt"])
            blocked = not result["safe"]
            label = "BLOCKED" if blocked else "ALLOWED"
        else:
            result = pre_filter(case["prompt"])
            blocked = result["blocked"]
            label = "BLOCKED" if blocked else "ALLOWED"

        expected_block = case.get("adversarial", False) and (
            "block" in case["expected_behavior"].lower()
            or "decline" in case["expected_behavior"].lower()
            or "hard block" in case["expected_behavior"].lower()
        )

        if expected_block == blocked:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"

        print(f"  Moderation: {label}  →  {status}")
        if blocked:
            reason = result.get("reason", "")
            print(f"  Reason: {reason[:80]}")

        # Show RAG context for non-adversarial cases
        if not blocked and case["category"] not in ("image",):
            context = retrieve_relevant_verses(case["prompt"], n_results=2)
            if context:
                first_line = context.split("\n")[0]
                print(f"  RAG hit: {first_line[:80]}")

        print()

    print(f"{'='*70}")
    print(f"  Results: {passed}/{len(cases)} passed moderation checks")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run()
