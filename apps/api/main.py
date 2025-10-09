# taw/apps/api/main.py
from __future__ import annotations

import os, json, sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Reuse your compiled LangGraph and existing components
from agents.graph import graph
from agents.models import TaskSpec
from rag.embeddings import get_backend, build_faiss, INDEX_DIR, BASE_DIR  # for /ingest
from agents.mcp_tools import get_tools

# --------- Config (env-overridable) ----------
DB_PATH = os.getenv("DB_PATH", "taw.db")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
RETRIEVAL_METHOD_DEFAULT = os.getenv("RETRIEVAL_METHOD", "hybrid_v2")
USE_MCP = os.getenv("USE_MCP", "1")  # graph uses this to route via MCP tools

Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

# --------- FastAPI app & CORS ----------
app = FastAPI(title="TireSim Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static serving of artifacts (images, logs, CSV, Run Cards)
app.mount(
    "/artifacts",
    StaticFiles(directory=Path(ARTIFACTS_DIR).resolve().as_posix()),
    name="artifacts",
)

# --------- Pydantic models ----------
class PlanRunRequest(BaseModel):
    objective: str = Field(..., description="High-level task objective")
    prompt: Optional[str] = Field(None, description="User prompt; defaults to objective")
    params: Dict[str, Any] = Field(default_factory=lambda: {
        "tire_size": "315/80 R22.5",
        "rim": "9.00x22.5",
        "inflation_bar": 8.0,
        "mesh_mm": 8.0,
        "load_sweep": [0, 10000, 20000, 30000, 40000, 50000],
    })
    retrieval_method: str = Field(default=RETRIEVAL_METHOD_DEFAULT)

class PlanRunResponse(BaseModel):
    task_id: int
    citations: List[str]
    metrics: Dict[str, float]
    validation: Dict[str, Any]
    run_card_path: Optional[str] = None
    run_card_url: Optional[str] = None
    artifact_urls: Dict[str, str] = Field(default_factory=dict)
    summary: Optional[str] = None

class IngestRequest(BaseModel):
    backend: str = Field("openai", description="'openai' | 'local' | 'auto'")
    max_chars: int = 800
    overlap: int = 120

class IngestResponse(BaseModel):
    backend: str
    count: int
    dim: int
    index_dir: str

# --------- Helpers ----------
def _rel_artifact_url(abs_path: str) -> Optional[str]:
    """
    Convert an absolute (or relative) filesystem path to a /artifacts URL if it lives under ARTIFACTS_DIR.
    """
    try:
        p = Path(abs_path).resolve()
        base = Path(ARTIFACTS_DIR).resolve()
        rel = p.relative_to(base)
        return f"/artifacts/{rel.as_posix()}"
    except Exception:
        return None
    
def _plan_run_procedural(state: dict) -> dict:
    """
    Lightweight replacement for graph.invoke(state).
    Calls: ds_new_task -> templates_fill -> fem_run -> plots_plot_curve -> write_metric(s)
    and returns a dict with the same shape your API expects.
    """
    tools = get_tools(db_path=state["db_path"], artifacts_dir=state["artifacts_dir"])

    # 1) task row
    task_id = tools.ds_new_task(
        prompt=state["prompt"], objective=state["objective"], params=state["params"]
    )

    # 2) deck from template (pick a sensible default template name you already have)
    template_name = "vertical_stiffness.j2"  # adjust to your actual template key
    deck_text = tools.templates_fill(template_name, state["params"])

    tools.ds_log_run(task_id, step="template", tool="templates.fill", 
                     inputs={"template": template_name, "params": state["params"]},
                     outputs={"deck_len": len(deck_text)}, status="ok")

    # 3) FEM (mock) run
    fem_out = tools.fem_run(deck_text)
    tools.ds_log_run(task_id, step="run", tool="fem.run", 
                     inputs={"deck_preview": deck_text[:200]},
                     outputs=fem_out, status="ok")

    # 4) Plot
    plot_out = tools.plots_plot_curve(fem_out["csv"], filename=f"curve_task_{task_id}.png")
    tools.ds_log_run(task_id, step="plot", tool="plots.plot_curve", 
                     inputs={"csv": fem_out["csv"]},
                     outputs=plot_out, status="ok")

    # 5) (Optional) simple validation + metrics
    #    You can compute something from the CSV later; for now stub a metric.
    tools.ds_write_metric(task_id, "status_ok", 1.0)

    # 6) Build response compatible with PlanRunResponse
    artifacts = {
        "csv": fem_out.get("csv"),
        "log": fem_out.get("log"),
        "deck": fem_out.get("deck"),
        "plot": plot_out.get("png"),
    }
    # Optional: write a run-card file if your UI expects it
    run_card_path = os.path.join(state["artifacts_dir"], f"run_card_{task_id}.md")
    with open(run_card_path, "w", encoding="utf-8") as f:
        f.write(f"# Run Card {task_id}\n\n- Objective: {state['objective']}\n- Template: {template_name}\n")

    return {
        "result": type("R", (), {  # quick shim to mimic your previous .result object
            "task_id": task_id,
            "citations": [],
            "metrics": {"status_ok": 1.0},
            "validation": {},
            "run_card_path": run_card_path,
            "artifacts": [{"kind": k, "path": v} for k, v in artifacts.items() if v],
            "summary": f"Completed: {state['objective']}",
        })()
    }

def _fetch_task_from_db(task_id: int) -> Dict[str, Any]:
    """
    Read task, runs, metrics from SQLite. Also try to infer artifact paths from run outputs.
    """
    dbp = Path(DB_PATH)
    if not dbp.exists():
        raise HTTPException(status_code=404, detail="DB not found")

    conn = sqlite3.connect(dbp.as_posix())
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()

        cur.execute("SELECT * FROM tasks WHERE id=?", (task_id,))
        trow = cur.fetchone()
        if not trow:
            raise HTTPException(status_code=404, detail="Task not found")

        # Parse task JSON
        tdata = dict(trow)
        for k in ("params",):
            if tdata.get(k) and isinstance(tdata[k], str):
                try:
                    tdata[k] = json.loads(tdata[k])
                except Exception:
                    pass

        # Runs
        cur.execute("SELECT * FROM runs WHERE task_id=? ORDER BY id ASC", (task_id,))
        runs = [dict(r) for r in cur.fetchall()]
        for r in runs:
            for k in ("inputs", "outputs"):
                if r.get(k) and isinstance(r[k], str):
                    try:
                        r[k] = json.loads(r[k])
                    except Exception:
                        pass

        # Metrics
        cur.execute("SELECT name, value FROM metrics WHERE task_id=? ORDER BY id ASC", (task_id,))
        metrics = {row["name"]: row["value"] for row in cur.fetchall()}

        # Infer artifacts from run steps
        artifacts: Dict[str, str] = {}
        run_card_guess = Path(ARTIFACTS_DIR) / f"run_card_{task_id}.md"
        if run_card_guess.exists():
            artifacts["run_card"] = run_card_guess.as_posix()

        # Try to pull from run outputs
        for r in reversed(runs):
            out = r.get("outputs", {})
            # FEM outputs
            if r.get("step") == "run":
                for key in ("csv", "log"):
                    if out.get(key):
                        artifacts[key] = out[key]
            # Plot output
            if r.get("step") == "plot" and out.get("png"):
                artifacts["png"] = out["png"]
            # Validate -> run card path
            if r.get("step") == "validate" and out.get("run_card"):
                artifacts["run_card"] = out["run_card"]
                break

        return {
            "task": tdata,
            "runs": runs,
            "metrics": metrics,
            "artifacts": artifacts,
        }
    finally:
        conn.close()

def _ingest_corpus(backend_choice: str = "openai", max_chars: int = 800, overlap: int = 120) -> IngestResponse:
    """
    Ingest markdown corpus into FAISS index (replicates scripts/ingest_corpus.py logic).
    """
    import re
    CORPUS_DIR = (BASE_DIR / "rag" / "corpus").resolve()

    def slugify(text: str) -> str:
        t = re.sub(r"[^\w\s-]", "", text.lower()).strip()
        t = re.sub(r"\s+", "-", t)
        t = re.sub(r"-+", "-", t)
        return t

    def read_markdown_with_anchors(path: Path):
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        sections = []
        current_anchor = slugify(title)
        buf: List[str] = []
        for line in lines:
            if line.strip().startswith("#"):
                if buf:
                    text = "\n".join(buf).strip()
                    if text:
                        sections.append((current_anchor, text))
                hdr = line.lstrip("# ").strip()
                current_anchor = slugify(hdr)
                buf = [line]
            else:
                buf.append(line)
        if buf:
            text = "\n".join(buf).strip()
            if text:
                sections.append((current_anchor, text))
        return title, sections

    def chunk_text(text: str) -> List[str]:
        chunks = []
        i = 0
        while i < len(text):
            end = min(len(text), i + max_chars)
            chunk = text[i:end]
            chunks.append(chunk)
            if end == len(text):
                break
            i = max(0, end - overlap)
        return chunks

    backend = get_backend(backend_choice)
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for p in sorted(CORPUS_DIR.glob("*.md")):
        title, sections = read_markdown_with_anchors(p)
        for anchor, sect_text in sections:
            for idx, ch in enumerate(chunk_text(sect_text)):
                metas.append({
                    "doc_id": p.name,
                    "title": title,
                    "section": anchor,
                    "chunk_id": f"{p.name}::{anchor}::{idx}",
                    "source_id": f"{p.name}#{anchor}",
                    "path": p.as_posix(),
                    "text": ch,
                })
                texts.append(ch)

    vi = build_faiss(texts, metas, backend, normalize=True)
    vi.save(INDEX_DIR)
    return IngestResponse(
        backend=getattr(backend, "name", "unknown"),
        count=len(texts),
        dim=vi.dim,
        index_dir=INDEX_DIR.as_posix()
    )

# --------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "use_mcp": USE_MCP, "db": DB_PATH, "artifacts": ARTIFACTS_DIR}

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        return _ingest_corpus(req.backend, req.max_chars, req.overlap)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

USE_GRAPH = os.getenv("USE_GRAPH", "0").lower() in {"1","true","yes","on"}

@app.post("/plan-run", response_model=PlanRunResponse)
def plan_run(req: PlanRunRequest):
    prompt = req.prompt or req.objective
    state = {
        "prompt": prompt,
        "objective": req.objective,
        "params": req.params,
        "db_path": DB_PATH,
        "artifacts_dir": ARTIFACTS_DIR,
        "retrieval_method": req.retrieval_method,
    }

    if USE_GRAPH:
        # original behavior
        out = graph.invoke(state)
    else:
        # graph bypass
        out = _plan_run_procedural(state)

    res = out.get("result")
    if not res:
        raise HTTPException(status_code=500, detail="Pipeline returned no result")

    # Build URLs as you already do
    artifact_urls = {}
    for a in getattr(res, "artifacts", []):
        url = _rel_artifact_url(a["path"])
        if url:
            if a["kind"] in ("plot", "csv", "log", "deck"):
                artifact_urls[a["kind"]] = url
    run_card_url = _rel_artifact_url(getattr(res, "run_card_path", None)) if getattr(res, "run_card_path", None) else None

    return PlanRunResponse(
        task_id=res.task_id,
        citations=list(getattr(res, "citations", []) or []),
        metrics=dict(getattr(res, "metrics", {}) or {}),
        validation=dict(getattr(res, "validation", {}) or {}),
        run_card_path=getattr(res, "run_card_path", None),
        run_card_url=run_card_url,
        artifact_urls=artifact_urls,
        summary=getattr(res, "summary", None),
    )

@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    data = _fetch_task_from_db(task_id)

    # Convert artifact file paths to URLs
    urls = {}
    for k, p in data.get("artifacts", {}).items():
        u = _rel_artifact_url(p)
        if u:
            urls[k] = u

    # Attach Run Card content if available
    run_card_text = None
    rc_path = data.get("artifacts", {}).get("run_card")
    if rc_path and Path(rc_path).exists():
        run_card_text = Path(rc_path).read_text(encoding="utf-8")

    return {
        "task": data["task"],
        "runs": data["runs"],
        "metrics": data["metrics"],
        "artifacts": data["artifacts"],
        "artifact_urls": urls,
        "run_card": run_card_text,
    }
