# taw/agents/graph.py
from __future__ import annotations
from typing import Dict, Any, List
import json
import pandas as pd, numpy as np
from pathlib import Path

from langgraph.graph import StateGraph, END

from agents.models import TaskSpec, RunResult, RunArtifact
from agents.mcp_tools import get_tools
from agents.nodes.llm import summarize_with_citations
from validation.run_card import validate_and_write_run_card

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

def _ensure_defaults(state: State) -> Dict[str, str]:
    return {
        "db_path": state.get("db_path", "taw.db"),
        "artifacts_dir": state.get("artifacts_dir", "artifacts"),
    }

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
    return TaskSpec(prompt=prompt, objective=objective, params=params)

def _estimate_stiffness(csv_path: str | None) -> float:
    if not csv_path or not Path(csv_path).exists():
        return 0.0
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return 0.0
    x, y = df["deflection_m"].values, df["load_N"].values
    a, _ = np.polyfit(x, y, deg=1)
    return float(max(a, 0.0))

def _first_existing(p: str | None) -> str | None:
    return p if p and Path(p).exists() else None

def _to_dict(obj: Any) -> Dict[str, Any]:
    """Normalize tool return shapes to dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        for k in ("value", "data", "result"):
            v = obj.get(k)
            if isinstance(v, dict):
                return v
        return obj
    if isinstance(obj, str):
        try:
            dec = json.loads(obj)
            return dec if isinstance(dec, dict) else {}
        except Exception:
            if obj.endswith(".png"):
                return {"png": obj}
            return {}
    if isinstance(obj, list) and obj:
        it = obj[0]
        if isinstance(it, dict):
            return it
        if isinstance(it, str):
            try:
                dec = json.loads(it)
                return dec if isinstance(dec, dict) else {}
            except Exception:
                if it.endswith(".png"):
                    return {"png": it}
                return {}
    return {}

def _latest_by_suffix(dir_path: str, suffix: str, prefer_name: str | None = None) -> str | None:
    p = Path(dir_path)
    if not p.exists():
        return None
    files = list(p.glob(f"*{suffix}"))
    if not files:
        return None
    if prefer_name:
        preferred = [f for f in files if f.name == prefer_name]
        if preferred:
            return preferred[0].as_posix()
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0].as_posix()

# -----------------------
# Node 1: DecisionAgent
# -----------------------
def decision_node(state: State) -> Dict[str, Any]:
    defaults = _ensure_defaults(state)
    spec = _ensure_spec(state)
    tools = get_tools(db_path=defaults["db_path"], artifacts_dir=defaults["artifacts_dir"])

    task_id = state.get("task_id") or tools.ds_new_task(
        prompt=spec.prompt, objective=spec.objective, params=spec.params
    )

    tooling_mode = getattr(tools, "mode", "unknown")
    tools.ds_log_run(
        task_id, "tooling", "tooling.mode",
        {"db_path": defaults["db_path"], "artifacts_dir": defaults["artifacts_dir"]},
        {"mode": tooling_mode}
    )

    citations: List[str] = []
    try:
        hits = tools.rag_retrieve(spec.objective, k=3, method="hybrid_v2")
        if isinstance(hits, dict):
            hits = hits.get("items") or hits.get("data") or [hits]
        hits = [h for h in hits if isinstance(h, dict)]
        citations = [h.get("source_id", "") for h in hits if "source_id" in h]
    except Exception:
        citations = []
    tools.ds_log_run(task_id, "retrieve", "rag.retrieve",
                     {"q": spec.objective, "k": 3}, {"citations": citations})

    return {
        "db_path": defaults["db_path"],
        "artifacts_dir": defaults["artifacts_dir"],
        "spec": spec,
        "task_id": task_id,
        "tooling_mode": tooling_mode,
        "citations": citations,
    }

# -----------------------
# Node 2: TemplateAgent
# -----------------------
def template_node(state: State) -> Dict[str, Any]:
    defaults = _ensure_defaults(state)
    spec: TaskSpec = _ensure_spec(state)
    tools = get_tools(db_path=defaults["db_path"], artifacts_dir=defaults["artifacts_dir"])
    task_id = state.get("task_id") or tools.ds_new_task(
        prompt=spec.prompt, objective=spec.objective, params=spec.params
    )

    template_name = "vertical_stiffness.j2"
    deck_text = tools.templates_fill(template_name, spec.params)

    tools.ds_log_run(
        task_id, "fill", "templates.fill",
        {"template": template_name, "params": spec.params},
        {"deck_len": len(deck_text)}
    )
    return {
        "task_id": task_id,
        "deck_text": deck_text,
    }

# -----------------------
# Node 3: FEMAgent
# -----------------------
def fem_node(state: State) -> Dict[str, Any]:
    defaults = _ensure_defaults(state)
    tools = get_tools(db_path=defaults["db_path"], artifacts_dir=defaults["artifacts_dir"])
    task_id = state.get("task_id") or 0
    deck_text = state.get("deck_text") or ""

    run_out = tools.fem_run(deck_text)

    csv_path = _first_existing(run_out.get("csv"))
    log_path = _first_existing(run_out.get("log"))
    deck_path = _first_existing(run_out.get("deck"))

    artdir = Path(defaults["artifacts_dir"])
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
            task_id, "run", "fem.run_mock",
            {"note": "fem outputs missing", "raw": run_out}, {"ok": False}, status="error"
        )
        raise RuntimeError(f"fem.run did not yield expected artifacts (got {run_out}).")

    tools.ds_log_run(
        task_id, "run", "fem.run_mock",
        {"deck_path": deck_path},
        {"csv": csv_path, "log": log_path}
    )
    return {
        "task_id": task_id,
        "fem_csv": csv_path,
        "fem_log": log_path,
        "fem_deck": deck_path,
    }

# -----------------------
# Node 4: PostProcessAgent
# -----------------------
def postprocess_node(state: State) -> Dict[str, Any]:
    defaults = _ensure_defaults(state)
    spec: TaskSpec = _ensure_spec(state)
    tools = get_tools(db_path=defaults["db_path"], artifacts_dir=defaults["artifacts_dir"])

    task_id = state.get("task_id") or 0
    fem_csv = state.get("fem_csv")
    fem_log = state.get("fem_log")
    fem_deck = state.get("fem_deck")
    citations = state.get("citations", [])

    # If fem_csv is missing (e.g., node ran out of order in Studio), try to discover it again.
    if not fem_csv or not Path(fem_csv).exists():
        fem_csv = _latest_by_suffix(defaults["artifacts_dir"], ".csv")
    if not fem_csv or not Path(fem_csv).exists():
        tools.ds_log_run(
            task_id, "plot", "plots.plot_curve",
            {"note": "CSV missing before plot/metrics"}, {"ok": False}, status="error"
        )
        raise RuntimeError("No FEM CSV found to post-process. Ensure FEMAgent ran and produced a CSV in artifacts/.")

    # Plot (robust to varying return shapes)
    raw_plot = tools.plots_plot_curve(fem_csv, filename="curve.png")
    plot_dict = _to_dict(raw_plot)
    png_path = plot_dict.get("png")
    if not png_path or not Path(png_path).exists():
        png_path = _latest_by_suffix(defaults["artifacts_dir"], ".png", prefer_name="curve.png") \
                   or _latest_by_suffix(defaults["artifacts_dir"], ".png")
    if not png_path or not Path(png_path).exists():
        tools.ds_log_run(
            task_id, "plot", "plots.plot_curve",
            {"csv": fem_csv, "note": "png missing", "raw": raw_plot},
            {"ok": False}, status="error"
        )
        raise RuntimeError(f"plots.plot_curve did not yield a PNG (got {raw_plot}).")

    tools.ds_log_run(task_id, "plot", "plots.plot_curve", {"csv": fem_csv}, {"png": png_path})

    # Metrics (now safe)
    k_est = _estimate_stiffness(fem_csv)
    metrics = {"stiffness_est_N_per_m": k_est}
    tools.ds_log_run(task_id, "summarize", "metrics.estimate", {"csv": fem_csv}, {"metrics": metrics})

    # Summary with citations (LLM)
    summary = summarize_with_citations(spec.objective, citations, metrics)

    # Validate + Run Card
    artifacts = [
        {"path": fem_deck, "kind": "deck"} if fem_deck else None,
        {"path": fem_log,  "kind": "log"}  if fem_log  else None,
        {"path": fem_csv,  "kind": "csv"},
        {"path": png_path, "kind": "plot"},
    ]
    artifacts = [a for a in artifacts if a and a.get("path")]

    validation, run_card_path = validate_and_write_run_card(
        task_id=task_id,
        artifacts_dir=defaults["artifacts_dir"],
        prompt=spec.prompt,
        objective=spec.objective,
        params=spec.params,
        citations=citations,
        metrics=metrics,
        artifacts=artifacts,
        summary=summary
    )

    tools.ds_log_run(
        task_id, "validate", "validation.run_card",
        {"csv": fem_csv, "citations": citations},
        {"status": validation.status, "run_card": run_card_path, "fail_reasons": validation.fail_reasons}
    )

    result = RunResult(
        metrics=metrics,
        artifacts=[RunArtifact(**a) for a in artifacts],
        citations=citations,
        task_id=task_id,
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

    return {
        "task_id": task_id,
        "plot_png": png_path,
        "metrics": metrics,
        "run_card_path": run_card_path,
        "summary": summary,
        "result": result,
    }

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
