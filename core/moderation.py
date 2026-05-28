"""
Two-stage moderation layer.

Stage 1 — Pre-filter (rule-based, fast):
    Catches obvious adversarial patterns before hitting the LLM.

Stage 2 — LLM-based classification:
    For borderline inputs, asks the LLM itself to classify intent.
    This is intentionally lightweight to stay within free-tier limits.
"""

import re
import os

# Patterns that are blocked immediately without LLM call
HARD_BLOCK_PATTERNS = [
    # Scripture rewriting — covers "to say", "to support", "to justify", "to promote", "to mean"
    r"rewrite.{0,40}(bible|scripture|verse|john|matthew|genesis|psalm).{0,40}(say|support|justify|promote|mean|show|prove)",
    # Hate directed at religion/religious figures
    r"(bible|jesus|god|christianity).{0,30}(evil|satanic|hate|racist|sexist)",
    # Making religious figures endorse harm
    r"make.{0,30}(bible|jesus|god|scripture).{0,30}(say|support|endorse).{0,30}(violence|hate|racism|harm)",
    # NSFW + religious figures — order-independent: checks both directions
    r"(porn|sexual|nude|naked|offensive).{0,50}(christian|jesus|mary|biblical|god|christ)",
    r"(christian|jesus|mary|biblical|god|christ).{0,50}(porn|sexual|nude|naked|offensive).{0,30}(pose|image|picture|photo)",
    # Physical harm to religious groups
    r"(kill|harm|hurt|attack|murder).{0,30}(christian|muslim|jew|hindu|religion|religious)",
    # Extremism creation
    r"create.{0,30}(cult|extremist|radicali[zs])",
    # Race superiority via scripture
    r"(race|racial|ethnicity).{0,40}(superior|inferior|better|worse|dominant).{0,40}(bible|scripture|god|christian)",
    r"(bible|scripture|god|christian).{0,40}(race|racial|ethnicity).{0,40}(superior|inferior|better|worse)",
    r"(prove|show|use|justify).{0,30}(bible|scripture|verse).{0,30}(race|racial).{0,30}(superior|inferior)",
    r"one race.{0,30}(superior|inferior|better|chosen|cursed)",
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
        # Explicit content
        "nude", "naked", "sexual", "porn", "erotic",
        # Violence & gore
        "violent", "violence", "blood", "gore", "kill", "murder", "weapon",
        "gun", "rifle", "pistol", "machine gun", "sword", "knife", "bomb",
        # Occult / anti-Christian
        "demon", "satanic", "satan", "occult", "666", "antichrist", "lucifer",
        # Offensive framing
        "offensive", "inappropriate", "shocking",
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
