from __future__ import annotations

import os
import traceback
from typing import Any, Dict

from langchain_groq import ChatGroq
from langchain_core.tools import tool

from vector_db import similarity_search


# -----------------------------
# LLM INITIALIZATION (SAFE)
# -----------------------------
def _get_llm():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing in environment variables")

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.2,
    )


# -----------------------------
# TOOLS
# -----------------------------
@tool
def vector_search_tool(query: str) -> str:
    """Search medical/vector database."""
    try:
        docs = similarity_search(query, k=4)
        if not docs:
            return "No relevant documents found."

        return "\n".join(
            f"- {d.page_content}" for d in docs
        )
    except Exception as e:
        return f"Vector search failed: {str(e)}"


@tool
def medical_knowledge_retriever(query: str) -> str:
    """Retrieve medical knowledge safely."""
    try:
        docs = similarity_search(query, k=3, filter_meta={"type": "medical"})

        if not docs:
            docs = similarity_search(query, k=3)

        if not docs:
            return "No medical knowledge retrieved."

        return "\n".join(f"- {d.page_content}" for d in docs)

    except Exception as e:
        return f"Medical retrieval error: {str(e)}"


@tool
def task_validator(subtasks: str) -> str:
    """Basic safety validator."""
    try:
        text = subtasks.lower().strip()

        if len(text) < 10:
            return "INVALID: Subtasks too short"

        issues = []

        if not any(word in text for word in ["doctor", "physician", "consult", "hospital"]):
            issues.append("Add recommendation for medical consultation")

        return "VALID" if not issues else "NEEDS_REVIEW: " + "; ".join(issues)

    except Exception as e:
        return f"Validator error: {str(e)}"


# -----------------------------
# SAFE LLM CALL
# -----------------------------
def _safe_llm_call(prompt: str) -> str:
    try:
        llm = _get_llm()
        res = llm.invoke(prompt)
        return getattr(res, "content", str(res))

    except Exception as e:
        return f"LLM fallback response due to error: {str(e)}"


# -----------------------------
# MAIN PLANNER (USED BY API)
# -----------------------------
def run_planner_agent(goal: str) -> Dict[str, Any]:
    try:
        goal = (goal or "").strip()

        if not goal:
            return {
                "goal": "",
                "steps": [],
                "final_plan": "",
                "error": "Empty goal provided",
                "confidence": 0.0,
            }

        # -------------------------
        # STEP 1: SAFE CONTEXT BUILD
        # -------------------------
        context_prompt = f"""
You are a medical planning assistant.

User goal:
{goal}

Generate a short clinical understanding summary.
Do NOT give treatment advice.
"""

        summary = _safe_llm_call(context_prompt)

        # -------------------------
        # STEP 2: STRUCTURED PLAN
        # -------------------------
        plan_prompt = f"""
Goal: {goal}

Medical context:
{summary}

Create a safe, step-by-step plan:
- simple steps
- non-prescriptive
- include doctor consultation advice

End with disclaimer: "This is not medical advice."
"""

        final_plan = _safe_llm_call(plan_prompt)

        # -------------------------
        # STEP 3: RESPONSE FORMAT
        # -------------------------
        return {
            "goal": goal,
            "steps": [
                {"step": 1, "action": "Understand user goal"},
                {"step": 2, "action": "Retrieve medical context"},
                {"step": 3, "action": "Generate safe plan"},
                {"step": 4, "action": "Add medical disclaimer"},
            ],
            "reasoning_summary": summary,
            "final_plan": final_plan,
            "confidence": 0.85,
        }

    except Exception as e:
        print("❌ CRITICAL ERROR IN PLANNER:")
        print(traceback.format_exc())

        return {
            "goal": goal,
            "steps": [],
            "final_plan": "System error occurred while generating plan.",
            "error": str(e),
            "confidence": 0.0,
        }


# -----------------------------
# OPTIONAL MOCK MODE (SAFE TESTING)
# -----------------------------
def _mock_planner(goal: str) -> Dict[str, Any]:
    return {
        "goal": goal,
        "steps": [
            {"step": 1, "action": "Mock analysis"},
            {"step": 2, "action": "Mock retrieval"},
            {"step": 3, "action": "Mock plan generation"},
        ],
        "final_plan": "This is a mock plan. Configure GROQ_API_KEY for real output.",
        "confidence": 0.5,
    }