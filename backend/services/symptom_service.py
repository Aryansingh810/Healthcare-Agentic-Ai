from __future__ import annotations

import json
import os
import re

from langchain_groq import ChatGroq

from vector_db import similarity_search


def _get_llm() -> ChatGroq | None:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(api_key=key, model=model, temperature=0.3)


def analyze_symptoms(symptoms: str, user_id: int) -> dict:
    symptoms = (symptoms or "").strip()
    if not symptoms:
        return {"error": "Symptoms text is required", "confidence": 0.0}

    docs = similarity_search(symptoms, k=5)
    context = "\n".join(f"- {d.page_content}" for d in docs) or "General primary care context."

    llm = _get_llm()
    if llm is None:
        return _mock_response(symptoms, context)

    prompt = f"""You are a clinical information assistant (not a doctor). Given patient-reported symptoms, respond with JSON only:
{{
  "suggested_specialists": ["..."],
  "recommendations": ["short bullet points"],
  "disclaimer": "Seek emergency care if red flags; this is not a diagnosis."
}}

Symptoms: {symptoms}

Relevant reference snippets (may be partial):
{context}

Use conservative, evidence-aligned language. Output valid JSON only."""

    raw = llm.invoke(prompt)
    text = getattr(raw, "content", str(raw))
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return _mock_response(symptoms, context)
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return _mock_response(symptoms, context)
    data["retrieval_context_used"] = bool(docs)
    data["confidence"] = 0.78 if docs else 0.55
    data["user_id"] = user_id
    return data


def _mock_response(symptoms: str, context: str) -> dict:
    return {
        "suggested_specialists": ["Primary care / Internal medicine", "Specialty based on persistence of symptoms"],
        "recommendations": [
            "Document onset, duration, and severity of symptoms.",
            "Seek urgent care if chest pain, stroke symptoms, or severe shortness of breath.",
            "Follow up with a clinician for evaluation — demo mode without GROQ_API_KEY.",
        ],
        "disclaimer": "Educational only; not a diagnosis. Configure GROQ_API_KEY for AI-enriched output.",
        "retrieval_context_used": bool(context and len(context) > 20),
        "confidence": 0.45,
    }
