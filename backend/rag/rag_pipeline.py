"""
rag_pipeline.py
---------------
RAG pipeline for AI Vehicle Health Assistant.
Uses sentence-transformers for embeddings, FAISS for vector storage,
and Groq (free) for LLM generation.
"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading & chunking
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DATASET_FILES = {
    "bike": "bike_issues.txt",
    "car":  "car_issues.txt",
    "suv":  "suv_issues.txt",
}


def load_documents(vehicle_type: str) -> List[Dict]:
    """Load and parse the flat-file dataset for the given vehicle type."""
    filename = DATASET_FILES.get(vehicle_type.lower())
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
                key, _, value = line.partition(":")
                record[key.strip().lower()] = value.strip()
        if record:
            record["chunk"] = (
                f"Problem: {record.get('problem', '')}\n"
                f"Cause: {record.get('cause', '')}\n"
                f"Solution: {record.get('solution', '')}\n"
                f"Severity: {record.get('severity', '')}\n"
                f"Cost: {record.get('cost', '')}"
            )
            records.append(record)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Embedding model
# ─────────────────────────────────────────────────────────────────────────────

_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FAISS index (cached per vehicle type)
# ─────────────────────────────────────────────────────────────────────────────

_indexes: Dict = {}


def build_index(vehicle_type: str):
    key = vehicle_type.lower()
    if key in _indexes:
        return _indexes[key]

    records = load_documents(key)
    chunks  = [r["chunk"] for r in records]
    vectors = embed_texts(chunks)

    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    _indexes[key] = (index, records)
    return index, records


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_top_k(query: str, vehicle_type: str, k: int = 4) -> List[Dict]:
    index, records = build_index(vehicle_type)
    q_vec = embed_texts([query])
    scores, indices = index.search(q_vec, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        rec = dict(records[idx])
        rec["similarity_score"] = float(scores[0][rank])
        results.append(rec)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Groq LLM generation (FREE)
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
across bikes, cars, and SUVs. You diagnose issues based on retrieved knowledge and give practical, \
structured advice. Always respond in valid JSON with exactly these keys:
- problem_summary (string)
- possible_causes (list of strings)
- severity (one of: "Low", "Moderate", "High")
- suggested_actions (list of strings)
- estimated_cost_inr (string, e.g. "INR 500 - INR 3000")
- additional_notes (string, optional tips)

Be concise, accurate, and safety-conscious. Costs are in Indian Rupees (INR).
Return ONLY the JSON object. No markdown, no extra text, no explanation."""


def generate_diagnosis(query: str, vehicle_type: str, context_records: List[Dict]) -> Dict:
    """Call Groq LLM with retrieved context to produce structured diagnosis."""
    context_text = "\n\n---\n\n".join(
        f"[Record {i+1}]\n{rec['chunk']}"
        for i, rec in enumerate(context_records)
    )

    user_message = f"""Vehicle type: {vehicle_type.upper()}
User complaint: {query}

Relevant knowledge from the vehicle health database:
{context_text}

Based on the above context and your expertise, diagnose this {vehicle_type} issue.
Return ONLY a valid JSON object (no markdown, no extra text)."""

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

    # Strip accidental markdown fences if present
    raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
    raw_text = re.sub(r"\n?```$", "", raw_text)

    diagnosis = json.loads(raw_text)
    return diagnosis


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Public entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_rag_pipeline(query: str, vehicle_type: str) -> Dict:
    """
    Full RAG pipeline:
      1. Retrieve relevant records from FAISS vector store.
      2. Feed context + query to Groq LLM (free).
      3. Return structured diagnosis dict.
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
