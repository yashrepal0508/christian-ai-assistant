"""
LLM module using Groq's free API (Llama-3.1-8B).

Why Groq instead of HF Inference API?
  HF's free inference endpoint may be blocked on corporate networks.
  Groq provides a free OpenAI-compatible API with faster inference
  and no routing through paid third-party providers.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "llama-3.1-8b-instant"  # free tier, fast

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


def build_prompt(messages: list[dict], denomination: str, context: str) -> list[dict]:
    system_content = SYSTEM_PROMPT_TEMPLATE.format(
        denomination=denomination,
        context=context if context else "No specific passages retrieved for this query.",
    )
    return [{"role": "system", "content": system_content}] + messages


def get_llm_response(messages: list[dict], denomination: str, context: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    full_messages = build_prompt(messages, denomination, context)

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=full_messages,
        max_tokens=700,
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()
