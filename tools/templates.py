# tools/templates.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from jinja2 import Template

# templates/ directory is sibling of this file's parent (repo/templates)
TEMPL_DIR = (Path(__file__).resolve().parents[1] / "templates").resolve()

def list_templates() -> List[str]:
    """Return all available .j2 templates by filename."""
    return sorted([p.name for p in TEMPL_DIR.glob("*.j2")])

def fill_template(name: str, params: Dict) -> str:
    """
    Render the Jinja2 template with the given params and return deck text.
    Raises FileNotFoundError if missing.
    """
    path = TEMPL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    text = path.read_text(encoding="utf-8")
    tmpl = Template(text)
    deck_text = tmpl.render(**params)
    return deck_text
