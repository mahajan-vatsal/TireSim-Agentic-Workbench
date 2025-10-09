# taw/agents/graph.py
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd, numpy as np
from pathlib import Path

from langgraph.graph import StateGraph, END

from agents.models import TaskSpec, AgentPlan, PlanStep, RunResult, RunArtifact
from agents.mcp_tools import get_tools
from validation.run_card import validate_and_write_run_card
from agents.nodes.llm import plan_with_llm, summarize_with_citations

# -----------------------
# State & helpers
# -----------------------
class State(dict):
    """
    Shared state passed between nodes.
    Keys that appear after the pipeline:
      - task_id: int
      - citations: List[str]
      - deck_text: str
      - fem_csv / fem_log / fem_deck: str
      - plot_png: str
      - metrics: Dict[str, float]
      - validation: Dict[str, Any]
      - run_card_path: str
      - summary: str
      - result: RunResult
    """
    ...

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

def _first_existing(p: str | None) -> str | None:
    return p if p and Path(p).exists() else None

# -----------------------
# Node 1: DecisionAgent
# -----------------------
def decision_node(state: State) -> State:
    """
    - Ensure defaults/spec
    - Open datastore task (task_id)
    - Record tooling mode
    - (Optional) RAG retrieve citations (kept minimal & fast)
    """
    _ensure_defaults(state)
    spec = _ensure_spec(state)
    tools = get_tools(db_path=state["db_path"], artifacts_dir=state["artifacts_dir"])

    # Create task if needed
    if "task_id" not in state:
        state["task_id"] = tools.ds_new_task(
            prompt=spec.prompt, objective=spec.objective, params=spec.params
        )
    tid = state["task_id"]

    # Log tooling mode for visibility in Studio
    state["tooling_mode"] = getattr(tools, "mode", "unknown")
    tools.ds_log_run(
        tid, "tooling", "tooling.mode",
        {"db_path": state["db_path"], "artifacts_dir": state["artifacts_dir"]},
        {"mode": state["tooling_mode"]}
    )

    # Very light RAG (optional; safe-guarded)
    try:
        hits = tools.rag_retrieve(spec.objective, k=3, method="hybrid_v2")
        if isinstance(hits, dict):
            hits = hits.get("items") or hits.get("data") or [hits]
        hits = [h for h in hits if isinstance(h, dict)]
        citations = [h.get("source_id", "") for h in hits if "source_id" in h]
    except Exception:
        citations = []
    state["citations"] = citations
    tools.ds_log_run(tid, "retrieve", "rag.retrieve",
                     {"q": spec.objective, "k": 3}, {"citations": citations})
    return state

# -----------------------
# Node 2: TemplateAgent
# -----------------------
def template_node(state: State) -> State:
    """
    - Fill the Jinja deck template
    """
    _ensure_defaults(state)
    spec = _ensure_spec(state)
    tools = get_tools(db_path=state["db_path"], artifacts_dir=state["artifacts_dir"])
    tid = state["task_id"]

    template_name = "vertical_stiffness.j2"  # keep explicit
    deck_text = tools.templates_fill(template_name, spec.params)
    state["deck_text"] = deck_text

    tools.ds_log_run(
        tid, "fill", "templates.fill",
        {"template": template_name, "params": spec.params},
        {"deck_len": len(deck_text)}
    )
    return state

# -----------------------
# Node 3: FEMAgent
# -----------------------
def fem_node(state: State) -> State:
    """
    - Run mock FEM
    - Normalize artifact paths (csv/log/deck)
    """
    _ensure_defaults(state)
    tools = get_tools(db_path=state["db_path"], artifacts_dir=state["artifacts_dir"])
    tid = state["task_id"]

    run_out = tools.fem_run(state["deck_text"])

    csv_path = _first_existing(run_out.get("csv"))
    log_path = _first_existing(run_out.get("log"))
    deck_path = _first_existing(run_out.get("deck"))

    # Fallback discovery in artifacts dir (defensive)
    artdir = Path(state["artifacts_dir"])
    if not csv_path:
        cands = sorted(artdir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        csv_path = cands[0].as_posix() if cands else None
    if not log_path:
        cands = sorted(artdir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        log_path = cands[0].as_posix() if cands else None
    if not deck_path:
        cands = sorted(list(artdir.glob("*.inp")) + list(artdir.glob("*.txt")),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        deck_path = cands[0].as_posix() if cands else None

    if not (csv_path and log_path and deck_path):
        tools.ds_log_run(
            tid, "run", "fem.run_mock",
            {"note": "fem outputs missing", "raw": run_out}, {"ok": False}, status="error"
        )
        raise RuntimeError(f"fem.run did not yield expected artifacts (got {run_out}).")

    state["fem_csv"] = csv_path
    state["fem_log"] = log_path
    state["fem_deck"] = deck_path

    tools.ds_log_run(
        tid, "run", "fem.run_mock",
        {"deck_path": deck_path},
        {"csv": csv_path, "log": log_path}
    )
    return state

# -----------------------
# Node 4: PostProcessAgent
# -----------------------
def postprocess_node(state: State) -> State:
    """
    - Plot curve
    - Compute metrics
    - LLM summary
    - Validate + write Run Card
    - Build RunResult
    """
    _ensure_defaults(state)
    spec: TaskSpec = _ensure_spec(state)
    tools = get_tools(db_path=state["db_path"], artifacts_dir=state["artifacts_dir"])
    tid = state["task_id"]

    # Plot
    png_path = tools.plots_plot_curve(state["fem_csv"], filename="curve.png")["png"]
    state["plot_png"] = png_path
    tools.ds_log_run(tid, "plot", "plots.plot_curve", {"csv": state["fem_csv"]}, {"png": png_path})

    # Metrics
    k_est = _estimate_stiffness(state["fem_csv"])
    metrics = {"stiffness_est_N_per_m": k_est}
    state["metrics"] = metrics
    tools.ds_log_run(tid, "summarize", "metrics.estimate", {"csv": state["fem_csv"]}, {"metrics": metrics})

    # Summary with citations (lightweight LLM)
    citations = state.get("citations", [])
    summary = summarize_with_citations(spec.objective, citations, metrics)
    state["summary"] = summary

    # Validate + Run Card
    artifacts = [
        {"path": state["fem_deck"], "kind": "deck"},
        {"path": state["fem_log"],  "kind": "log"},
        {"path": state["fem_csv"],  "kind": "csv"},
        {"path": state["plot_png"], "kind": "plot"},
    ]
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
    state["run_card_path"] = run_card_path

    tools.ds_log_run(
        tid, "validate", "validation.run_card",
        {"csv": state["fem_csv"], "citations": citations},
        {"status": validation.status, "run_card": run_card_path, "fail_reasons": validation.fail_reasons}
    )

    # Final result object (what /plan-run expects)
    result = RunResult(
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
    state["result"] = result
    return state

# -----------------------
# Build the 4-node graph
# -----------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("DecisionAgent", decision_node)
    g.add_node("TemplateAgent", template_node)
    g.add_node("FEMAgent", fem_node)
    g.add_node("PostProcessAgent", postprocess_node)

    g.set_entry_point("DecisionAgent")
    g.add_edge("DecisionAgent", "TemplateAgent")
    g.add_edge("TemplateAgent", "FEMAgent")
    g.add_edge("FEMAgent", "PostProcessAgent")
    g.add_edge("PostProcessAgent", END)
    return g.compile()

graph = build_graph()
