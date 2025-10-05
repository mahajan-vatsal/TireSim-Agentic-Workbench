# taw/agents/graph.py
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from langgraph.graph import StateGraph, END

from agents.models import TaskSpec, AgentPlan, PlanStep, RunResult, RunArtifact
from tools.datastore import Store
from tools import templates, fem_mock, plots
from rag.retriever import retrieve as rag_retrieve
from validation.run_card import validate_and_write_run_card
from agents.nodes.llm import plan_with_llm, summarize_with_citations

# ---- State type ----
class State(dict):
    """
    JSON-serializable state for LangGraph Studio.
    Keys:
      - spec: TaskSpec
      - task_id: int
      - db_path: str
      - artifacts_dir: str
      - result: RunResult
    """

def _ensure_defaults(state: State):
    state.setdefault("db_path", "taw.db")
    state.setdefault("artifacts_dir", "artifacts")

def _get_store(state: State) -> Store:
    _ensure_defaults(state)
    store = Store(db_path=state["db_path"], artifacts_dir=state["artifacts_dir"])
    store.init("datastore/schema.sql")
    store.ensure_artifacts_dir()
    return store

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

def _ensure_task_id(state: State, store: Store):
    if "task_id" not in state:
        spec: TaskSpec = state["spec"]
        state["task_id"] = store.new_task(prompt=spec.prompt, objective=spec.objective, params=spec.params)

def _estimate_stiffness(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        return 0.0
    x = df["deflection_m"].values
    y = df["load_N"].values
    a, _ = np.polyfit(x, y, deg=1)
    return float(max(a, 0.0))

# ---- Nodes ----
def planner(state: State) -> State:
    _ensure_defaults(state)
    spec = _ensure_spec(state)
    store = _get_store(state)
    _ensure_task_id(state, store)
    tid = state["task_id"]

    # Try LLM plan first; fall back to fixed plan on failure
    llm_steps = plan_with_llm(spec.objective, spec.params)
    if isinstance(llm_steps, list) and llm_steps:
        steps = []
        for s in llm_steps:
            steps.append(PlanStep(name=s.get("name","step"),
                                  tool=s.get("tool",""),
                                  inputs=s.get("inputs",{})))
        plan = AgentPlan(steps=steps)
        store.log_run(tid, "plan", "planner.llm", {"objective": spec.objective}, {"steps": [s.dict() for s in plan.steps]})
    else:
        plan = AgentPlan(steps=[
            PlanStep(name="retrieve", tool="rag.retrieve", inputs={"q": spec.objective, "k": 3}),
            PlanStep(name="fill", tool="templates.fill", inputs={"name": "vertical_stiffness.j2", "params": spec.params}),
            PlanStep(name="run", tool="fem.run", inputs={}),
            PlanStep(name="plot", tool="plots.plot", inputs={}),
            PlanStep(name="summarize", tool="summarize", inputs={}),
            PlanStep(name="validate", tool="validate", inputs={}),
        ])
        store.log_run(tid, "plan", "planner.fixed", {"objective": spec.objective}, {"steps": [s.dict() for s in plan.steps]})

    state["plan"] = plan
    return state

def executor(state: State) -> State:
    store = _get_store(state)
    spec: TaskSpec = _ensure_spec(state)
    _ensure_task_id(state, store)
    tid: int = state["task_id"]

    # 1) Retrieve (RAG)
    hits = rag_retrieve(spec.objective, k=3, method="hybrid")
    citations = [h.source_id for h in hits]
    store.log_run(tid, "retrieve", "rag.retrieve", {"q": spec.objective, "k": 3}, {"citations": citations})

    # 2) Fill template
    deck = templates.fill_template("vertical_stiffness.j2", spec.params)
    store.log_run(tid, "fill", "templates.fill", {"template": "vertical_stiffness.j2", "params": spec.params}, {"deck_len": len(deck)})

    # 3) Run FEM (mock)
    csv_path, log_path, deck_path = fem_mock.run_case(deck, output_dir=store.artifacts_dir)
    store.log_run(tid, "run", "fem.run_mock", {"deck_path": deck_path}, {"csv": csv_path, "log": log_path})

    # 4) Plot
    png_path = plots.plot_curve(csv_path, output_dir=store.artifacts_dir, filename="curve.png")
    store.log_run(tid, "plot", "plots.plot_curve", {"csv": csv_path}, {"png": png_path})

    # 5) Metrics
    k_est = _estimate_stiffness(csv_path)
    metrics = {"stiffness_est_N_per_m": k_est}
    store.log_run(tid, "summarize", "metrics.estimate", {"csv": csv_path}, {"metrics": metrics})

    # 5b) LLM Summary (optional)
    summary = summarize_with_citations(spec.objective, citations, metrics)
    if summary:
        store.log_run(tid, "summarize", "llm.summary", {"citations": citations, "metrics": metrics}, {"length": len(summary)})
    else:
        summary = None

    # 6) Validate + Run Card
    artifacts = [
        {"path": deck_path, "kind": "deck"},
        {"path": log_path, "kind": "log"},
        {"path": csv_path, "kind": "csv"},
        {"path": png_path, "kind": "plot"},
    ]
    validation, run_card_path = validate_and_write_run_card(
        task_id=tid,
        artifacts_dir=store.artifacts_dir,
        prompt=spec.prompt,
        objective=spec.objective,
        params=spec.params,
        citations=citations,
        metrics=metrics,
        artifacts=artifacts,
        summary=summary
    )
    store.log_run(tid, "validate", "validation.run_card",
                  {"csv": csv_path, "citations": citations},
                  {"status": validation.status, "run_card": run_card_path, "fail_reasons": validation.fail_reasons})

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
        summary=summary
    )
    state["result"] = result
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
