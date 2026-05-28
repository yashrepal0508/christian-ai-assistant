# Architecture Note — Christian AI Assistant

## Stack (fully free / open-source)

| Component | Technology | Why |
|---|---|---|
| LLM | Mistral-7B-Instruct-v0.3 (HF Serverless API) | Strong instruction following, no license gate, free tier |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers, local) | Fast, zero cost, good semantic similarity |
| Vector DB | ChromaDB (in-memory) | Zero setup, sufficient for demo scale |
| Image Gen | SDXL-base-1.0 (HF Serverless API) | Free tier, high quality |
| Verse Verification | bible-api.com (REST, no auth) | Ground truth for exact citation checking |
| UI | Streamlit | Fastest path to a working chat demo |

---

## Key Engineering Decisions

### 1. RAG for Hallucination Prevention
Every user query triggers a semantic search over ~50 embedded KJV Bible verses. The top-4 matches are injected into the LLM system prompt as grounding context. This constrains the model to reason from actual scripture rather than generating plausible-sounding but fabricated verses.

### 2. Two-Stage Moderation
**Pre-filter (rule-based):** Fast regex pattern matching catches hard adversarial cases (scripture rewriting, hate speech, inappropriate image requests) _before_ any API call. Zero latency, zero cost.

**Post-filter (structural):** Response is scanned for hallucination markers and harmful language after generation. If flagged, the response is suppressed and the user is redirected.

### 3. Verse Verification Layer
Any Bible reference cited in the LLM response (e.g., "John 3:16") is extracted via regex and cross-checked against `bible-api.com` in real time. Verified citations show ✓ with the actual KJV text; unverifiable ones are flagged — making hallucinated references visible to the user.

### 4. Denomination Context Injection
The user's denomination is passed into the system prompt template. This lets the model appropriately:
- Reference Deuterocanonical books for Catholic users
- Note sola scriptura for Protestants
- Acknowledge iconographic and liturgical tradition for Orthodox users

### 5. Image Safety — Prompt Refinement + Blocklist
Image prompts go through a blocklist check (nudity, violence, occult terms) before any generation call. Safe prompts are automatically refined by appending "Christian art style, reverent, appropriate for all ages" to steer SDXL output.

---

## Data Flow

```
User Input
    │
    ▼
[Pre-filter Moderation] ──blocked──► Decline message
    │ safe
    ▼
[Bible RAG] — semantic search → top-4 relevant verses
    │
    ▼
[LLM] — Mistral-7B with system prompt (denomination + RAG context + history)
    │
    ▼
[Post-filter Check] ──flagged──► Suppress + redirect
    │ safe
    ▼
[Verse Verification] — extract citations → verify against bible-api.com
    │
    ▼
Display response + grounding context + verification badges
```

---

## Tricky Scenario Handling

| Scenario | Handling Strategy |
|---|---|
| Fake Bible verse (e.g., "Hezekiah 4:12") | Verse verification catches unknown book; model instructed to flag non-existent books |
| Misquoted verse injected by user | RAG context shows real verse; system prompt instructs correction |
| "Rewrite scripture to support X" | Hard-blocked by pre-filter regex pattern |
| Hateful religious content | Hard-blocked by pre-filter |
| Contradictory theological prompts | LLM handles with denomination context + acknowledged tension |
| Adversarial image prompts | Blocklist check before any HF API call |
| Hallucinated historical claims | System prompt instructs epistemic humility; model trained to hedge uncertainty |
| Denomination conflicts (e.g., purgatory) | Denomination context in prompt + instruction to present multiple views fairly |

---

## Evaluation Dataset

Located in `eval/eval_dataset.json` — 14 test cases covering:
- Hallucination (fake books, wrong dates, misquoted verses)
- Adversarial (scripture rewriting, hate, NSFW image prompts)
- Difficult theology (problem of evil, slavery in the Bible)
- Denomination splits (purgatory, Marian prayer)
- Normal use (anxiety, love chapter)

---

## Limitations & Future Work
- ChromaDB is in-memory only — a persistent store (e.g., Chroma persistent client) would survive restarts
- Only ~50 verses embedded; a full Bible embedding would improve RAG recall
- HF free tier has rate limits — production would use a dedicated inference endpoint
- Post-filter is structural; an LLM-based classifier would catch subtler issues
