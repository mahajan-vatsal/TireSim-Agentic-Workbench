# taw/apps/cli/__main__.py
from __future__ import annotations
import typer
from agents.graph import graph
from tools.datastore import Store

app = typer.Typer(help="TireSim Agentic Workbench CLI")

@app.command()
def init():
    store = Store()
    store.init("datastore/schema.sql")
    store.ensure_artifacts_dir()
    typer.echo("✅ Initialized DB and artifacts/.")

@app.command("plan-run")
def plan_run(task: str,
             tire_size: str = "315/80 R22.5",
             rim: str = "9.00x22.5",
             inflation_bar: float = 8.0,
             mesh_mm: float = 8.0):
    params = {
        "tire_size": tire_size,
        "rim": rim,
        "inflation_bar": inflation_bar,
        "mesh_mm": mesh_mm,
        "load_sweep": [0, 10000, 20000, 30000, 40000, 50000],
    }
    input_state = {
        "prompt": task,
        "objective": task,
        "params": params,
        "db_path": "taw.db",
        "artifacts_dir": "artifacts",
    }
    out = graph.invoke(input_state)
    res = out.get("result")
    if not res:
        typer.echo("⚠️ No result returned.")
        raise typer.Exit(code=1)

    typer.echo(f"✅ Done. Task ID: {res.task_id}")
    typer.echo(f"   Citations: {', '.join(res.citations) or '-'}")
    for k, v in res.metrics.items():
        typer.echo(f"   Metric {k}: {v:.3f}")

    if res.validation:
        typer.echo(f"   Validation: {res.validation.get('status','UNKNOWN')}")
    if res.run_card_path:
        typer.echo(f"   Run Card: {res.run_card_path}")
        typer.echo("   Open the file to check PASS/FAIL and details.")

if __name__ == "__main__":
    app()
