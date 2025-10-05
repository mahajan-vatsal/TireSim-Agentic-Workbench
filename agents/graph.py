# taw/agents/graph.py
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd, numpy as np
from langgraph.graph import StateGraph, END

from agents.models import TaskSpec, AgentPlan, PlanStep, RunResult, RunArtifact
from agents.mcp_tools import get_tools
from validation.run_card import validate_and_write_run_card
from agents.nodes.llm import plan_with_llm, summarize_with_citations

class State(dict): ...

def _ensure_defaults(state: State):
    state.setdefault("db_path", "taw.db")
    state.setdefault("artifacts_dir", "artifacts")

def _ensure_spec(state: State) -> TaskSpec:
    if isinstance(state.get("spec"), TaskSpec):
        return state["spec"]
    prompt = state.get("prompt") or state.get("objective") or "Vertical load sweep"
    objective = state.get("objective") or prompt
    params = state.get("params") or {
        "tire_size": "315/80 R22.5",
        "rim": "9.00x22.5",
        "inflation_bar": 8.0,
        "mesh_mm": 8.0,
        "load_sweep": [0, 10000, 20000, 30000, 40000, 50000],
    }
    spec = TaskSpec(prompt=prompt, objective=objective, params=params)
    state["spec"] = spec
    return spec

def _estimate_stiffness(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return 0.0
    x, y = df["deflection_m"].values, df["load_N"].values
    a, _ = np.polyfit(x, y, deg=1)
    return float(max(a, 0.0))

def planner(state: State) -> State:
    _ensure_defaults(state)
    spec = _ensure_spec(state)
    tools = get_tools(db_path=state["db_path"], artifacts_dir=state["artifacts_dir"])

    # make sure task exists
    if "task_id" not in state:
        state["task_id"] = tools.ds_new_task(prompt=spec.prompt, objective=spec.objective, params=spec.params)

    # record tooling mode so we can see it in Studio
    state["tooling_mode"] = getattr(tools, "mode", "unknown")
    tools.ds_log_run(state["task_id"], "tooling", "tooling.mode",
                     {"db_path": state["db_path"], "artifacts_dir": state["artifacts_dir"]},
                     {"mode": state["tooling_mode"]})

    # LLM plan with fallback
    llm_steps = plan_with_llm(spec.objective, spec.params)
    if isinstance(llm_steps, list) and llm_steps:
        steps = [PlanStep(name=s.get("name","step"), tool=s.get("tool",""), inputs=s.get("inputs",{})) for s in llm_steps]
        plan = AgentPlan(steps=steps)
        tools.ds_log_run(state["task_id"], "plan", "planner.llm",
                         {"objective": spec.objective}, {"steps": [s.dict() for s in plan.steps]})
    else:
        plan = AgentPlan(steps=[
            PlanStep(name="retrieve", tool="rag.retrieve", inputs={"q": spec.objective, "k": 3}),
            PlanStep(name="fill", tool="templates.fill", inputs={"name": "vertical_stiffness.j2", "params": spec.params}),
            PlanStep(name="run", tool="fem.run", inputs={}),
            PlanStep(name="plot", tool="plots.plot", inputs={}),
            PlanStep(name="summarize", tool="summarize", inputs={}),
            PlanStep(name="validate", tool="validate", inputs={}),
        ])
        tools.ds_log_run(state["task_id"], "plan", "planner.fixed",
                         {"objective": spec.objective}, {"steps": [s.dict() for s in plan.steps]})
    state["plan"] = plan
    return state

def executor(state: State) -> State:
    _ensure_defaults(state)
    spec: TaskSpec = _ensure_spec(state)
    tools = get_tools(db_path=state["db_path"], artifacts_dir=state["artifacts_dir"])
    if "task_id" not in state:
        state["task_id"] = tools.ds_new_task(prompt=spec.prompt, objective=spec.objective, params=spec.params)
    tid = state["task_id"]

    # 1) RAG
    hits = tools.rag_retrieve(spec.objective, k=3, method="hybrid")

# Normalize just in case (safety belt)
    import json as _json
    if isinstance(hits, str):
        try:
            hits = _json.loads(hits)
        except Exception:
            hits = []
    if isinstance(hits, dict):
        hits = hits.get("items") or hits.get("data") or [hits]
    hits = [h for h in hits if isinstance(h, dict)]

    citations = [h.get("source_id", "") for h in hits if "source_id" in h]
    tools.ds_log_run(tid, "retrieve", "rag.retrieve", {"q": spec.objective, "k": 3}, {"citations": citations})

    # 2) Fill template
    deck = tools.templates_fill("vertical_stiffness.j2", spec.params)
    tools.ds_log_run(tid, "fill", "templates.fill", {"template": "vertical_stiffness.j2", "params": spec.params}, {"deck_len": len(deck)})

    # 3) FEM (mock)
    run_out = tools.fem_run(deck)

    # Defensive normalization
    def _first_existing(path: str | None) -> str | None:
        from pathlib import Path
        return path if path and Path(path).exists() else None

    csv_path = _first_existing(run_out.get("csv"))
    log_path = _first_existing(run_out.get("log"))
    deck_path = _first_existing(run_out.get("deck"))

# If missing, try to discover the most recent artifacts in artifacts_dir
    from pathlib import Path
    artdir = Path(state["artifacts_dir"])
    if not csv_path:
        cands = sorted(artdir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        csv_path = cands[0].as_posix() if cands else None
    if not log_path:
        cands = sorted(artdir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        log_path = cands[0].as_posix() if cands else None
    if not deck_path:
        cands = sorted(list(artdir.glob("*.inp")) + list(artdir.glob("*.txt")), key=lambda p: p.stat().st_mtime, reverse=True)
        deck_path = cands[0].as_posix() if cands else None

    if not (csv_path and log_path and deck_path):
    # Log + fail early with a helpful message
        tools.ds_log_run(tid, "run", "fem.run_mock", {"note": "fem outputs missing", "raw": run_out}, {"ok": False}, status="error")
        raise RuntimeError(f"fem.run did not yield expected artifacts (got {run_out}).")

    tools.ds_log_run(tid, "run", "fem.run_mock", {"deck_path": deck_path}, {"csv": csv_path, "log": log_path})

    # 4) Plot
    png_path = tools.plots_plot_curve(csv_path, filename="curve.png")["png"]
    tools.ds_log_run(tid, "plot", "plots.plot_curve", {"csv": csv_path}, {"png": png_path})

    # 5) Metrics
    k_est = _estimate_stiffness(csv_path)
    metrics = {"stiffness_est_N_per_m": k_est}
    tools.ds_log_run(tid, "summarize", "metrics.estimate", {"csv": csv_path}, {"metrics": metrics})

    # 5b) LLM summary
    summary = summarize_with_citations(spec.objective, citations, metrics)

    # 6) Validate + Run Card
    artifacts = [
        {"path": deck_path, "kind": "deck"},
        {"path": log_path, "kind": "log"},
        {"path": csv_path, "kind": "csv"},
        {"path": png_path, "kind": "plot"},
    ]
    from validation.run_card import validate_and_write_run_card
    validation, run_card_path = validate_and_write_run_card(
        task_id=tid,
        artifacts_dir=state["artifacts_dir"],
        prompt=spec.prompt,
        objective=spec.objective,
        params=spec.params,
        citations=citations,
        metrics=metrics,
        artifacts=artifacts,
        summary=summary
    )
    tools.ds_log_run(tid, "validate", "validation.run_card",
                     {"csv": csv_path, "citations": citations},
                     {"status": validation.status, "run_card": run_card_path, "fail_reasons": validation.fail_reasons})

    state["result"] = RunResult(
        metrics=metrics,
        artifacts=[RunArtifact(**a) for a in artifacts],
        citations=citations,
        task_id=tid,
        validation={
            "status": validation.status,
            "loads_strictly_increasing": validation.loads_strictly_increasing,
            "deflection_nonnegative": validation.deflection_nonnegative,
            "has_citation": validation.has_citation,
            "num_points": validation.num_points,
        },
        run_card_path=run_card_path,
        summary=summary,
        tooling_mode=state.get("tooling_mode", "unknown"),
    )
    return state

def build_graph():
    g = StateGraph(State)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", END)
    return g.compile()

graph = build_graph()
