"""
rag_pipeline.py  —  Lightweight version for Render free tier
-------------------------------------------------------------
- NO local embedding model (saves ~400MB RAM)
- Uses TF-IDF style keyword retrieval instead of FAISS
- Groq LLM for generation (free, fast)
- Total memory: ~80MB  (fits Render free 512MB easily)
"""

import os
import re
import json
import math
from collections import Counter
from typing import List, Dict

from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DATASET_FILES = {
    "bike": "bike_issues.txt",
    "car":  "car_issues.txt",
    "suv":  "suv_issues.txt",
}

_document_cache: Dict[str, List[Dict]] = {}


def load_documents(vehicle_type: str) -> List[Dict]:
    """Load and parse dataset for given vehicle type. Cached after first load."""
    key = vehicle_type.lower()
    if key in _document_cache:
        return _document_cache[key]

    filename = DATASET_FILES.get(key)
    if not filename:
        raise ValueError(f"Unknown vehicle type: {vehicle_type}")

    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]

    records = []
    for block in blocks:
        record = {}
        for line in block.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                record[k.strip().lower()] = v.strip()
        if record:
            record["chunk"] = (
                f"Problem: {record.get('problem', '')}\n"
                f"Cause: {record.get('cause', '')}\n"
                f"Solution: {record.get('solution', '')}\n"
                f"Severity: {record.get('severity', '')}\n"
                f"Cost: {record.get('cost', '')}"
            )
            records.append(record)

    _document_cache[key] = records
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Lightweight keyword retrieval (TF-IDF style, no heavy dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def bm25_score(query_tokens: List[str], doc_tokens: List[str],
               doc_freq: Dict[str, int], num_docs: int,
               k1: float = 1.5, b: float = 0.75,
               avg_doc_len: float = 50.0) -> float:
    """Simple BM25 ranking score."""
    score = 0.0
    doc_len = len(doc_tokens)
    doc_token_counts = Counter(doc_tokens)

    for token in query_tokens:
        if token not in doc_token_counts:
            continue
        tf = doc_token_counts[token]
        df = doc_freq.get(token, 1)
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += idf * tf_norm

    return score


def retrieve_top_k(query: str, vehicle_type: str, k: int = 4) -> List[Dict]:
    """Retrieve top-k records using BM25 keyword matching."""
    records = load_documents(vehicle_type)
    query_tokens = tokenize(query)

    # Tokenize all documents
    doc_tokens_list = [tokenize(r["chunk"]) for r in records]
    num_docs = len(doc_tokens_list)
    avg_doc_len = sum(len(d) for d in doc_tokens_list) / max(num_docs, 1)

    # Build document frequency table
    doc_freq: Dict[str, int] = {}
    for doc_tokens in doc_tokens_list:
        for token in set(doc_tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    # Score each document
    scored = []
    for i, (record, doc_tokens) in enumerate(zip(records, doc_tokens_list)):
        score = bm25_score(query_tokens, doc_tokens, doc_freq, num_docs, avg_doc_len=avg_doc_len)
        scored.append((score, i, record))

    # Sort by score descending, return top-k
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, _, record in scored[:k]:
        rec = dict(record)
        rec["similarity_score"] = round(score, 4)
        results.append(rec)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Groq LLM generation
# ─────────────────────────────────────────────────────────────────────────────

_groq_client = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


SYSTEM_PROMPT = """You are an expert vehicle mechanic and diagnostician with 20+ years of experience \
across bikes, cars, and SUVs in India. You diagnose issues based on retrieved knowledge and give \
practical, structured advice. Always respond in valid JSON with exactly these keys:
- problem_summary (string)
- possible_causes (list of strings)
- severity (one of: "Low", "Moderate", "High")
- suggested_actions (list of strings)
- estimated_cost_inr (string, e.g. "INR 500 - INR 3000")
- additional_notes (string)

Be concise, accurate, and safety-conscious. Costs are in Indian Rupees (INR).
Return ONLY the JSON object. No markdown fences, no extra text."""


def generate_diagnosis(query: str, vehicle_type: str, context_records: List[Dict]) -> Dict:
    """Send retrieved context + query to Groq and get structured JSON diagnosis."""
    context_text = "\n\n---\n\n".join(
        f"[Record {i+1}]\n{rec['chunk']}"
        for i, rec in enumerate(context_records)
    )

    user_message = f"""Vehicle type: {vehicle_type.upper()}
User complaint: {query}

Relevant knowledge retrieved from vehicle health database:
{context_text}

Diagnose this {vehicle_type} issue based on the above records.
Return ONLY a valid JSON object."""

    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    raw_text = response.choices[0].message.content.strip()
    raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
    raw_text = re.sub(r"\n?```$", "", raw_text)

    return json.loads(raw_text)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_pipeline(query: str, vehicle_type: str) -> Dict:
    """
    Lightweight RAG pipeline:
      1. BM25 keyword retrieval (no heavy ML model needed)
      2. Groq LLM generation (free API)
      3. Return structured diagnosis
    """
    top_records = retrieve_top_k(query, vehicle_type, k=4)

    if not top_records:
        return {
            "problem_summary": "Could not find relevant information.",
            "possible_causes": [],
            "severity": "Unknown",
            "suggested_actions": ["Please consult a certified mechanic."],
            "estimated_cost_inr": "N/A",
            "additional_notes": "",
        }

    diagnosis = generate_diagnosis(query, vehicle_type, top_records)
    diagnosis["vehicle_type"]            = vehicle_type.capitalize()
    diagnosis["query"]                   = query
    diagnosis["retrieved_records_count"] = len(top_records)

    return diagnosis
