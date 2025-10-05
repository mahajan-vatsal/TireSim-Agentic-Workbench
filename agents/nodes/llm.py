# taw/agents/nodes/llm.py
from __future__ import annotations
import os, json
from typing import Dict, Any, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # allow import without openai installed

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# -------- Planner --------
def plan_with_llm(objective: str, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Ask the LLM to emit a JSON plan (list of steps with tool + inputs).
    Falls back to None on any error so caller can use fixed plan.
    """
    client = _client()
    if client is None:
        return None

    system = (
        "You are a planning assistant for a tire FEM agent. "
        "Output ONLY valid JSON (no markdown) with this shape: "
        '{"steps":[{"name": "...","tool":"rag.retrieve","inputs":{"q": "...","k": 3}},'
        '{"name":"fill","tool":"templates.fill","inputs":{"name":"vertical_stiffness.j2","params":{...}}},'
        '{"name":"run","tool":"fem.run","inputs":{}},'
        '{"name":"plot","tool":"plots.plot","inputs":{}},'
        '{"name":"summarize","tool":"summarize","inputs":{}},'
        '{"name":"validate","tool":"validate","inputs":{}}]} '
        "Use the provided objective & params. Do not invent extra tools."
    )
    user = f"Objective: {objective}\nParams JSON: {json.dumps(params)}"

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_object"},  # structured JSON if model supports it
        )
        txt = resp.choices[0].message.content.strip()
        data = json.loads(txt)
        steps = data.get("steps")
        if isinstance(steps, list) and steps:
            return steps
        return None
    except Exception:
        return None

# -------- Summarizer --------
def summarize_with_citations(objective: str, citations: List[str], metrics: Dict[str, Any]) -> Optional[str]:
    """
    Short, factual summary that references only provided metrics/citations.
    """
    client = _client()
    if client is None:
        return None

    system = (
        "You are an engineering reporting assistant. Write a short, factual summary (4–6 lines). "
        "Only use the given objective, metrics, and the cited document IDs. Do not invent details."
    )
    user = (
        f"Objective: {objective}\n"
        f"Citations: {citations}\n"
        f"Metrics: {json.dumps(metrics, ensure_ascii=False)}\n\n"
        "Write the summary. Avoid claims not supported by metrics/citations."
    )

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None
