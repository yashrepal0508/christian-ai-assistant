"""
Two-stage moderation layer.

Stage 1 — Pre-filter (rule-based, fast):
    Catches obvious adversarial patterns before hitting the LLM.

Stage 2 — LLM-based classification:
    For borderline inputs, asks the LLM itself to classify intent.
    This is intentionally lightweight to stay within free-tier limits.
"""

import re
from huggingface_hub import InferenceClient
import os

# Patterns that are blocked immediately without LLM call
HARD_BLOCK_PATTERNS = [
    r"rewrite.{0,30}(bible|scripture|verse).{0,30}(support|justify|promote)",
    r"(bible|jesus|god|christianity).{0,30}(evil|satanic|hate|racist|sexist)",
    r"make.{0,30}(bible|jesus|god).{0,30}(say|support|endorse).{0,30}(violence|hate|racism)",
    r"generate.{0,30}(porn|sexual|nude|naked).{0,30}(christian|jesus|mary|biblical)",
    r"(kill|harm|hurt).{0,30}(christian|muslim|jew|religion)",
    r"create.{0,30}(cult|extremist|radicali[zs])",
]

SOFT_WARN_PATTERNS = [
    r"prove.{0,20}(bible|christianity).{0,20}wrong",
    r"(atheist|satanist).{0,20}perspective",
    r"contradict.{0,20}(bible|scripture|god)",
]


def pre_filter(text: str) -> dict:
    """
    Fast rule-based check. Returns {'blocked': bool, 'reason': str, 'warn': bool}.
    """
    lower = text.lower()

    for pattern in HARD_BLOCK_PATTERNS:
        if re.search(pattern, lower):
            return {
                "blocked": True,
                "warn": False,
                "reason": "This request asks me to misrepresent, alter, or weaponize scripture or produce harmful content. I'm not able to help with that.",
            }

    for pattern in SOFT_WARN_PATTERNS:
        if re.search(pattern, lower):
            return {
                "blocked": False,
                "warn": True,
                "reason": "This touches on challenging theological territory. I'll engage thoughtfully.",
            }

    return {"blocked": False, "warn": False, "reason": ""}


def classify_image_prompt(prompt: str) -> dict:
    """
    Validate an image generation prompt for Christian appropriateness.
    Returns {'safe': bool, 'reason': str, 'refined_prompt': str}.
    """
    lower = prompt.lower()
    blocked_terms = [
        "nude", "naked", "sexual", "violent", "blood", "gore",
        "demon", "satanic", "occult", "666", "antichrist",
    ]
    for term in blocked_terms:
        if term in lower:
            return {
                "safe": False,
                "reason": f"Image prompt contains inappropriate content ('{term}'). Please describe a reverent Christian scene.",
                "refined_prompt": "",
            }

    refined = (
        f"{prompt.strip()}, Christian art style, reverent, beautiful, "
        "appropriate for all ages, religious, sacred, high quality"
    )
    return {"safe": True, "reason": "", "refined_prompt": refined}


def check_response_safety(response: str) -> dict:
    """
    Post-generation check on LLM output.
    Flags if the response contains fabricated verse markers or problematic content.
    """
    issues = []

    # Check for [fabricated] or similar markers an honest model might add
    if re.search(r"\[note:?.{0,40}fabricat|hallucinate|not.{0,10}real verse\]", response, re.I):
        issues.append("Response contains a flag about a fabricated verse.")

    # Check for hateful language
    hate_terms = ["infidel", "heretic should die", "kill all", "destroy all"]
    for term in hate_terms:
        if term in response.lower():
            issues.append(f"Response contains potentially harmful language: '{term}'")

    return {"safe": len(issues) == 0, "issues": issues}
