# taw/agents/mcp_tools.py
from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List

# -------------------------------
# Local modules (fallback paths)
# -------------------------------
from tools import templates as templates_local
from tools import fem_mock as fem_local
from tools import plots as plots_local
from tools.datastore import Store
from rag.retriever import retrieve as rag_local


# -------------------------------
# Shared helpers
# -------------------------------
def _store(db_path: str, artifacts_dir: str) -> Store:
    """Create/init a Store instance (idempotent)."""
    s = Store(db_path=db_path, artifacts_dir=artifacts_dir)
    s.init("datastore/schema.sql")
    s.ensure_artifacts_dir()
    return s

def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Normalize different result shapes into a Dict[str, Any].
    Accepts: dict, str(JSON), list[dict] (first), dict with 'data'/'value'.
    """
    if obj is None:
        return {}
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return {}
    if isinstance(obj, dict):
        if "value" in obj and isinstance(obj["value"], dict):
            return obj["value"]
        if "data" in obj and isinstance(obj["data"], dict):
            return obj["data"]
        return obj
    if isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, dict):
            return first
        if isinstance(first, str):
            try:
                dec = json.loads(first)
                return dec if isinstance(dec, dict) else {}
            except Exception:
                return {}
    return {}

def _to_list_of_dicts(obj: Any) -> List[Dict[str, Any]]:
    """
    Normalize various result shapes into List[Dict].
    Accepts: list[dict], dict, list[str JSON], str JSON, dict with 'items'/'data'.
    """
    if obj is None:
        return []
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return []
    if isinstance(obj, dict):
        if "items" in obj and isinstance(obj["items"], list):
            obj = obj["items"]
        elif "data" in obj and isinstance(obj["data"], list):
            obj = obj["data"]
        else:
            obj = [obj]
    if not isinstance(obj, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in obj:
        if isinstance(it, str):
            try:
                it = json.loads(it)
            except Exception:
                continue
        if isinstance(it, dict):
            out.append(it)
    return out

def _latest_by_suffix(dir_path: str, suffix: str) -> str | None:
    p = Path(dir_path)
    if not p.exists():
        return None
    files = sorted(p.glob(f"*{suffix}"), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0].as_posix() if files else None

def _ensure_fem_paths(d: Dict[str, Any], artifacts_dir: str) -> Dict[str, Any]:
    """
    Ensure the FEM result dict has 'csv', 'log', 'deck'.
    If missing, try common wrappers; else discover most-recent files in artifacts/.
    """
    # unwrap common containers
    for k in ("result", "value", "data"):
        if k in d and isinstance(d[k], dict):
            d = d[k]

    need = {"csv", "log", "deck"}
    if need.issubset(d.keys()):
        return d

    # fallback: find latest artifacts by suffix
    csv = d.get("csv") or _latest_by_suffix(artifacts_dir, ".csv")
    log = d.get("log") or _latest_by_suffix(artifacts_dir, ".log")
    # deck could be .inp/.txt; try both (prefer .inp)
    deck = d.get("deck") or _latest_by_suffix(artifacts_dir, ".inp") or _latest_by_suffix(artifacts_dir, ".txt")

    out = {}
    if csv: out["csv"] = csv
    if log: out["log"] = log
    if deck: out["deck"] = deck
    return out


# ===============================
# Local fallback implementation
# ===============================
class LocalTools:
    """
    Directly calls your local Python tools (no MCP).
    Useful when USE_MCP is off or the MCP SDK/server is unavailable.
    """
    def __init__(self, db_path: str, artifacts_dir: str):
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        self.mode = "local"

    # ----- templates -----
    def templates_list(self) -> List[str]:
        return templates_local.list_templates()

    def templates_fill(self, name: str, params: Dict[str, Any]) -> str:
        return templates_local.fill_template(name, params)

    # ----- FEM (mock) -----
    def fem_run(self, deck_text: str) -> Dict[str, str]:
        csv_path, log_path, deck_path = fem_local.run_case(deck_text, output_dir=self.artifacts_dir)
        return {"csv": csv_path, "log": log_path, "deck": deck_path}

    # ----- RAG -----
    def rag_retrieve(self, q: str, k: int = 3, method: str = "hybrid") -> List[Dict[str, Any]]:
        hits = rag_local(q, k=k, method=method)
        return [{"text": h.text, "source_id": h.source_id, "score": float(h.score), "title": h.title} for h in hits]

    # ----- plots -----
    def plots_plot_curve(self, csv_path: str, filename: str = "curve.png") -> Dict[str, str]:
        png = plots_local.plot_curve(csv_path, output_dir=self.artifacts_dir, filename=filename)
        return {"png": png}

    # ----- datastore -----
    def ds_new_task(self, prompt: str, objective: str, params: Dict[str, Any]) -> int:
        s = _store(self.db_path, self.artifacts_dir)
        return s.new_task(prompt, objective, params)

    def ds_log_run(
        self,
        task_id: int,
        step: str,
        tool: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        status: str = "ok",
    ) -> int:
        s = _store(self.db_path, self.artifacts_dir)
        return s.log_run(task_id, step, tool, inputs, outputs, status=status)

    def ds_write_metric(self, task_id: int, name: str, value: float) -> int:
        s = _store(self.db_path, self.artifacts_dir)
        return s.write_metric(task_id, name, value)


# ===============================
# MCP-backed implementation
# ===============================
class MCPTools:
    """
    Uses the current MCP client API over stdio, with per-call sessions
    to avoid cross-loop shutdown issues in Studio.
    """
    def __init__(self, db_path: str, artifacts_dir: str, server_cmd: str | None = None):
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir
        self.server_cmd = server_cmd or os.getenv("MCP_SERVER_CMD", "python -m tools_mcp.multi_tool_server")
        self.mode = "mcp (stdio, per-call)"

    async def _acall(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Connect → call → close in a single async context.
        Robustly parse results without relying on SDK-specific content classes.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parts = shlex.split(self.server_cmd)
        if len(parts) == 1:
            args = [sys.executable, "-m", "tools_mcp.multi_tool_server"]
        else:
            args = parts

        # Inject defaults for datastore tools
        if tool_name.startswith("datastore."):
            arguments = {"db_path": self.db_path, "artifacts_dir": self.artifacts_dir, **arguments}

        async with AsyncExitStack() as stack:
            server_params = StdioServerParameters(command=args[0], args=args[1:], env=None)
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()

            result = await session.call_tool(tool_name, arguments)

            # 1) Prefer structured content (dict/list)
            sc = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
            if sc is not None:
                return sc

            # 2) Fallback: scan result.content items by duck typing
            content = getattr(result, "content", None)
            if isinstance(content, list) and content:
                # Objects with .value attribute (JSON-like)
                for item in content:
                    val = getattr(item, "value", None)
                    if val is not None:
                        return val
                # Objects with .text attribute (string payload)
                for item in content:
                    txt = getattr(item, "text", None)
                    if txt is not None:
                        try:
                            return json.loads(txt)
                        except Exception:
                            return txt
                # Dict-like fallbacks
                for item in content:
                    if isinstance(item, dict):
                        if "value" in item:
                            return item["value"]
                        if "data" in item:
                            return item["data"]
                        if "text" in item:
                            txt = item["text"]
                            try:
                                return json.loads(txt)
                            except Exception:
                                return txt

            # 3) Give up — empty object
            return {}

    # ---- sync wrappers for LangGraph nodes ----
    def templates_list(self) -> List[str]:
        return asyncio.run(self._acall("templates.list_templates", {}))

    def templates_fill(self, name: str, params: Dict[str, Any]) -> str:
        return asyncio.run(self._acall("templates.fill_template", {"name": name, "params": params}))

    def fem_run(self, deck_text: str) -> Dict[str, str]:
        raw = asyncio.run(self._acall("fem.run", {"deck_text": deck_text, "output_dir": self.artifacts_dir}))
        d = _to_dict(raw)
        d = _ensure_fem_paths(d, self.artifacts_dir)
        return d

    def rag_retrieve(self, q: str, k: int = 3, method: str = "hybrid") -> List[Dict[str, Any]]:
        raw = asyncio.run(self._acall("rag.retrieve", {"q": q, "k": k, "method": method}))
        return _to_list_of_dicts(raw)

    def plots_plot_curve(self, csv_path: str, filename: str = "curve.png") -> Dict[str, str]:
        raw = asyncio.run(self._acall("plots.plot_curve", {"csv_path": csv_path, "output_dir": self.artifacts_dir, "filename": filename}))
        return _to_dict(raw)

    def ds_new_task(self, prompt: str, objective: str, params: Dict[str, Any]) -> int:
        data = asyncio.run(self._acall("datastore.new_task", {"prompt": prompt, "objective": objective, "params": params}))
        data = _to_dict(data)
        return int(data.get("task_id", 0))

    def ds_log_run(
        self,
        task_id: int,
        step: str,
        tool: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        status: str = "ok",
    ) -> int:
        data = asyncio.run(self._acall("datastore.log_run", {
            "task_id": task_id,
            "step": step,
            "tool": tool,
            "inputs": inputs,
            "outputs": outputs,
            "status": status,
        }))
        data = _to_dict(data)
        return int(data.get("run_id", 0))

    def ds_write_metric(self, task_id: int, name: str, value: float) -> int:
        data = asyncio.run(self._acall("datastore.write_metric", {"task_id": task_id, "name": name, "value": float(value)}))
        data = _to_dict(data)
        return int(data.get("metric_id", 0))


# ===============================
# Factory
# ===============================
def get_tools(db_path: str, artifacts_dir: str):
    """
    If USE_MCP=1 (or 'true'/'yes') and the MCP SDK is installed,
    return MCPTools; otherwise fall back to LocalTools.
    """
    use_mcp = os.getenv("USE_MCP", "0").strip().lower() in {"1", "true", "yes"}
    if use_mcp:
        try:
            import mcp  # noqa: F401 (probe install)
            return MCPTools(db_path=db_path, artifacts_dir=artifacts_dir)
        except Exception:
            # Fall through to local on any import/initialization issue
            pass
    return LocalTools(db_path=db_path, artifacts_dir=artifacts_dir)
