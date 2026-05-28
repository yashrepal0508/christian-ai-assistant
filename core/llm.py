"""
LLM module using the HF free Inference API directly via requests.

Why not InferenceClient.chat_completion?
  Newer huggingface_hub versions route chat_completion through paid third-party
  providers (Novita, Together, etc.) instead of the free HF Inference API,
  causing 404 errors on the free tier. Calling api-inference.huggingface.co
  directly bypasses this routing entirely.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

SYSTEM_PROMPT_TEMPLATE = """You are a Christian AI assistant grounded in Biblical truth. You serve Christians of all denominations with accuracy, grace, and humility.

Your core principles:
1. Ground every theological claim in actual Bible scripture — only cite verses you are certain exist verbatim.
2. If you are unsure of the exact wording of a verse, say "I believe this is from [Book] but let me be careful here" rather than guessing.
3. Acknowledge denominational differences honestly (Catholic includes Deuterocanonical books; Orthodox has additional traditions; Protestant relies on 66 canonical books).
4. Decline gracefully but firmly any request to rewrite, alter, or misrepresent scripture.
5. Handle difficult theological questions (evil, suffering, contradictions) with intellectual honesty and pastoral care.
6. Never produce content that is heretical, hateful, or disrespectful toward any person or faith.
7. Maintain a warm, conversational, and reverent tone.

User's denomination context: {denomination}

Retrieved Bible passages for this query (use these to ground your response):
{context}

Important: If the user cites a Bible verse that does not match scripture, gently correct them with the actual text."""


def _build_mistral_prompt(messages: list[dict], denomination: str, context: str) -> str:
    """
    Format conversation as Mistral instruction template:
    <s>[INST] system + first_user [/INST] assistant </s>[INST] user [/INST] ...
    """
    system = SYSTEM_PROMPT_TEMPLATE.format(
        denomination=denomination,
        context=context if context else "No specific passages retrieved for this query.",
    )

    prompt = ""
    first_user = True
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "user":
            if first_user:
                prompt += f"<s>[INST] {system}\n\n{msg['content']} [/INST]"
                first_user = False
            else:
                prompt += f"<s>[INST] {msg['content']} [/INST]"
        elif msg["role"] == "assistant":
            prompt += f" {msg['content']} </s>"
        i += 1

    return prompt


def get_llm_response(messages: list[dict], denomination: str, context: str) -> str:
    token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    prompt = _build_mistral_prompt(messages, denomination, context)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.5,
            "return_full_text": False,
            "do_sample": True,
        },
    }

    for attempt in range(3):
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

        # Model is loading (cold start) — wait and retry
        if resp.status_code == 503:
            wait = resp.json().get("estimated_time", 20)
            time.sleep(min(wait, 30))
            continue

        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()

        return "I was unable to generate a response. Please try again."

    return "The model is still loading. Please wait a moment and try again."
