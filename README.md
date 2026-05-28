# Christian AI Assistant

A Christianity-focused AI assistant built with HuggingFace free models, Bible RAG, and multi-layer safety.

## Setup

```bash
# 1. Clone and enter directory
cd christian-ai-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your HuggingFace token
cp .env.example .env
# Edit .env and paste your HF token: HF_TOKEN=hf_xxxx

# 4. Run the app
streamlit run app.py

# 5. (Optional) Run the evaluation suite
python eval/run_eval.py
```

## Features

- **Chat interface** with conversation memory
- **Scripture-aware responses** grounded in Bible RAG (ChromaDB + sentence-transformers)
- **Live verse verification** via bible-api.com — hallucinated citations are flagged
- **Christian image generation** via Stable Diffusion XL (HF free tier)
- **Denomination-aware** — Catholic / Protestant / Orthodox / etc.
- **Two-stage moderation** — rule-based pre-filter + structural post-filter
- **Evaluation dataset** — 14 cases covering hallucination, adversarial, and theological edge cases

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design notes and data flow.
