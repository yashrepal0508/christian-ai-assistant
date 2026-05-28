"""
Bible RAG module.

Two-layer strategy:
  1. Semantic retrieval from embedded KJV verses (ChromaDB + sentence-transformers)
     to ground every LLM response with relevant scripture.
  2. Exact verse verification via bible-api.com (free, no auth) to catch
     hallucinated or misquoted citations.
"""

import json
import os
import re
import requests
import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "bible_kjv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_chroma_client = None
_collection = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.Client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        _collection = _chroma_client.get_collection(
            name=COLLECTION_NAME, embedding_function=ef
        )
    except Exception:
        _collection = _chroma_client.create_collection(
            name=COLLECTION_NAME, embedding_function=ef
        )
        _load_bible_data(_collection)

    return _collection


def _load_bible_data(collection):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "bible_sample.json")
    with open(data_path, "r") as f:
        verses = json.load(f)

    documents, ids, metadatas = [], [], []
    for i, v in enumerate(verses):
        ref = f"{v['book']} {v['chapter']}:{v['verse']}"
        documents.append(f"{ref} — {v['text']}")
        ids.append(f"v{i}")
        metadatas.append({"book": v["book"], "chapter": v["chapter"], "verse": v["verse"], "ref": ref})

    collection.add(documents=documents, ids=ids, metadatas=metadatas)


def retrieve_relevant_verses(query: str, n_results: int = 4) -> str:
    """Return top-N relevant Bible passages as formatted string for LLM context."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = results["documents"][0]
    if not docs:
        return ""
    return "\n".join(f"• {d}" for d in docs)


def verify_verse(reference: str) -> dict:
    """
    Verify a Bible verse citation against bible-api.com.
    Returns {'found': bool, 'text': str, 'reference': str}.
    """
    try:
        url = f"https://bible-api.com/{requests.utils.quote(reference)}?translation=kjv"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "found": True,
                "text": data.get("text", "").strip(),
                "reference": data.get("reference", reference),
            }
    except Exception:
        pass
    return {"found": False, "text": "", "reference": reference}


def extract_and_verify_citations(text: str) -> list[dict]:
    """
    Find any Bible references in the text (e.g., John 3:16) and verify them.
    Returns list of verification results for display.
    """
    pattern = r"\b([1-3]?\s?[A-Za-z]+)\s(\d+):(\d+(?:-\d+)?)\b"
    matches = re.findall(pattern, text)
    results = []
    seen = set()
    for book, chapter, verse in matches:
        ref = f"{book.strip()} {chapter}:{verse}"
        if ref not in seen:
            seen.add(ref)
            result = verify_verse(ref)
            results.append(result)
    return results
