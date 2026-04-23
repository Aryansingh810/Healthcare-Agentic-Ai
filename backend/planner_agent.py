from __future__ import annotations

import os
from typing import Any

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from vector_db import similarity_search


class PlanStepModel(BaseModel):
    step: int
    action: str
    status: str


class PlannerJSON(BaseModel):
    goal: str = ""
    steps: list[PlanStepModel] = Field(default_factory=list)
    final_plan: str = ""


def _get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(api_key=api_key, model=model, temperature=0.2)


@tool
def vector_search_tool(query: str) -> str:
    """Search vector database for relevant healthcare documents."""
    docs = similarity_search(query, k=4)
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        parts.append(f"[{i}] ({meta.get('type', 'doc')}) {d.page_content}")
    return "\n".join(parts)


@tool
def medical_knowledge_retriever(query: str) -> str:
    """Retrieve medical knowledge related to symptoms or condition."""
    docs = similarity_search(query, k=3, filter_meta={"type": "medical"})
    if not docs:
        docs = similarity_search(query, k=3)
    if not docs:
        return "No medical knowledge retrieved."
    return "\n".join(f"- {d.page_content}" for d in docs)


@tool
def task_validator(subtasks: str) -> str:
    """Validate generated medical subtasks for safety and completeness."""
    t = subtasks.strip().lower()
    if len(t) < 10:
        return "INVALID: subtasks too short."
    issues = []
    if "consult" not in t and "physician" not in t and "doctor" not in t:
        issues.append("Consider explicit in-person consultation.")
    status = "VALID" if not issues else "NEEDS_REVIEW: " + "; ".join(issues)
    return status


def _build_agent_executor() -> AgentExecutor:
    llm = _get_llm()
    tools = [vector_search_tool, medical_knowledge_retriever, task_validator]
    system = """You are a healthcare planning assistant agent. Use tools to gather evidence.
Follow this reasoning order:
1) Understand the medical context (use medical_knowledge_retriever / vector_search_tool).
2) Retrieve supporting data from the vector database.
3) Propose concrete sub-tasks as plain text.
4) Call task_validator with a summary of your sub-tasks.
5) Summarize a safe, stepwise execution plan (non-prescriptive; remind user to seek professional care).

After tools, respond with a concise final summary of the plan in plain English (the app will structure JSON separately)."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=8, handle_parsing_errors=True)


def _structured_plan(goal: str, agent_summary: str) -> PlannerJSON:
    llm = _get_llm()
    prompt = f"""Goal: {goal}

Agent research summary:
{agent_summary}

Produce JSON matching the schema with exactly 5 steps numbered 1-5:
1 Understand the medical context
2 Retrieve relevant data from the vector database  
3 Generate sub-tasks
4 Validate steps  
5 Produce a final execution plan

Each step must have action and status (e.g. pending/done). final_plan must be a clear, safe narrative plan.
Include disclaimer that this is not medical advice."""
    try:
        structured = llm.with_structured_output(PlannerJSON)
        return structured.invoke(prompt)
    except Exception:
        raw = llm.invoke(
            prompt
            + "\n\nRespond with ONLY valid JSON matching keys: goal, steps (array of step, action, status), final_plan."
        )
        content = getattr(raw, "content", str(raw))
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return PlannerJSON.model_validate_json(content[start:end])
        raise


def run_planner_agent(goal: str) -> dict[str, Any]:
    goal = (goal or "").strip()
    if not goal:
        return {"goal": "", "steps": [], "final_plan": "", "error": "Empty goal"}

    try:
        executor = _build_agent_executor()
        agent_out = executor.invoke({"input": f"Healthcare planning goal: {goal}"})
        summary = str(agent_out.get("output", ""))
        plan = _structured_plan(goal, summary)
        data = plan.model_dump()
        data["reasoning_summary"] = summary
        data["confidence"] = 0.85
        return data
    except RuntimeError as e:
        if "GROQ_API_KEY" in str(e):
            return _mock_planner(goal)
        raise
    except Exception as e:
        return {
            "goal": goal,
            "steps": [],
            "final_plan": "",
            "error": str(e),
            "confidence": 0.0,
        }


def _mock_planner(goal: str) -> dict[str, Any]:
    steps = [
        {"step": 1, "action": "Understand the medical context for the stated goal", "status": "done"},
        {"step": 2, "action": "Retrieve relevant data from vector database (demo mode)", "status": "done"},
        {"step": 3, "action": "Generate sub-tasks and education points", "status": "done"},
        {"step": 4, "action": "Validate steps with task validator (mock)", "status": "done"},
        {"step": 5, "action": "Produce final execution plan outline", "status": "done"},
    ]
    return {
        "goal": goal,
        "steps": steps,
        "final_plan": (
            f"[Demo mode — set GROQ_API_KEY] Outline for: {goal}. "
            "Review evidence-based guidelines, confirm diagnoses with a clinician, "
            "personalize therapy, and schedule follow-up. This is not medical advice."
        ),
        "reasoning_summary": "Mock planner: configure Groq API for full agent + LLM output.",
        "confidence": 0.5,
    }


