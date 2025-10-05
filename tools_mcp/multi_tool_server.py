# tools_mcp/multi_tool_server.py
from __future__ import annotations
from typing import Any
import os

from mcp.server.fastmcp import FastMCP

# Reuse your existing local modules
from tools import templates as templates_local
from tools import fem_mock as fem_local
from tools import plots as plots_local
from tools.datastore import Store
from rag.retriever import retrieve as rag_local
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _store(db_path: str, artifacts_dir: str) -> Store:
    s = Store(db_path=db_path, artifacts_dir=artifacts_dir)
    s.init("datastore/schema.sql")
    s.ensure_artifacts_dir()
    return s


# One multi-tool MCP server
mcp = FastMCP(
    "taw-tools",
    instructions="TireSim Agent tools: templates, FEM mock, plots, datastore, RAG."
)

# ---------- Health ----------
@mcp.tool(name="tools.health")
def health() -> dict[str, str]:
    """Return basic server health info."""
    return {"status": "ok", "cwd": os.getcwd()}

# ---------- Templates ----------
@mcp.tool(name="templates.list_templates")
def templates_list() -> list[str]:
    """List available input-deck templates."""
    return templates_local.list_templates()

@mcp.tool(name="templates.fill_template")
def templates_fill(name: str, params: dict[str, Any]) -> str:
    """Fill a Jinja template into a deck text."""
    return templates_local.fill_template(name, params)

# ---------- FEM (mock) ----------
@mcp.tool(name="fem.run")
def fem_run(deck_text: str, output_dir: str | None = None) -> dict[str, str]:
    """Run mock FEM and return artifact paths."""
    out_dir = output_dir or "artifacts"
    csv_path, log_path, deck_path = fem_local.run_case(deck_text, output_dir=out_dir)
    return {"csv": csv_path, "log": log_path, "deck": deck_path}

# ---------- Plots ----------
@mcp.tool(name="plots.plot_curve")
def plots_plot_curve(csv_path: str, output_dir: str | None = None, filename: str = "curve.png") -> dict[str, str]:
    """Generate a curve plot PNG from a CSV path."""
    out_dir = output_dir or "artifacts"
    png = plots_local.plot_curve(csv_path, output_dir=out_dir, filename=filename)
    return {"png": png}

# ---------- RAG ----------
@mcp.tool(name="rag.retrieve")
def rag_retrieve(q: str, k: int = 3, method: str = "hybrid_v2") -> list[dict[str, Any]]:
    """Retrieve top-k docs with citations."""
    hits = rag_local(q, k=k, method=method)
    return [{"text": h.text, "source_id": h.source_id, "score": float(h.score), "title": h.title} for h in hits]

# ---------- Datastore ----------
@mcp.tool(name="datastore.new_task")
def ds_new_task(db_path: str, artifacts_dir: str, prompt: str, objective: str, params: dict[str, Any]) -> dict[str, int]:
    """Create a new task and return its ID."""
    s = _store(db_path, artifacts_dir)
    task_id = s.new_task(prompt, objective, params)
    return {"task_id": int(task_id)}

@mcp.tool(name="datastore.log_run")
def ds_log_run(
    db_path: str,
    artifacts_dir: str,
    task_id: int,
    step: str,
    tool: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    status: str = "ok",
) -> dict[str, int]:
    """Log a run step and return run_id."""
    s = _store(db_path, artifacts_dir)
    run_id = s.log_run(task_id, step, tool, inputs, outputs, status=status)
    return {"run_id": int(run_id)}

@mcp.tool(name="datastore.write_metric")
def ds_write_metric(db_path: str, artifacts_dir: str, task_id: int, name: str, value: float) -> dict[str, int]:
    """Write a numeric metric and return metric_id."""
    s = _store(db_path, artifacts_dir)
    metric_id = s.write_metric(task_id, name, value)
    return {"metric_id": int(metric_id)}

if __name__ == "__main__":
    # Default transport is stdio; perfect for Studio launching a child process.
    mcp.run()
