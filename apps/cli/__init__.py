# apps/cli/__init__.py
from __future__ import annotations
import typer
from datastore.store import Store

def _build_app() -> typer.Typer:
    app = typer.Typer(help="TireSim Agentic Workbench - CLI")

    @app.command("init")
    def init_cmd(
        db_path: str = typer.Option("taw.db", "--db-path", help="SQLite DB path"),
        schema_path: str = typer.Option("datastore/schema.sql", "--schema-path", help="SQL schema path"),
    ):
        """Initialize the SQLite database using the provided schema."""
        store = Store(db_path=db_path)
        store.init(schema_path=schema_path)
        typer.echo(f"Initialized DB at '{db_path}' using schema '{schema_path}'.")

    return app

# The Typer group we will invoke from __main__
app = _build_app()
