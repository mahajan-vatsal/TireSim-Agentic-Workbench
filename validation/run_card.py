# taw/validation/run_card.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import datetime as dt
import pandas as pd

@dataclass
class ValidationResult:
    loads_strictly_increasing: bool
    deflection_nonnegative: bool
    has_citation: bool
    num_points: int
    min_load: float
    max_load: float
    min_deflection: float
    max_deflection: float
    status: str               # "PASS" | "FAIL"
    fail_reasons: List[str]

def _strictly_increasing(values) -> bool:
    if len(values) < 2: 
        return False
    return all(values[i] < values[i+1] for i in range(len(values)-1))

def run_validation(csv_path: str, citations: List[str]) -> ValidationResult:
    df = pd.read_csv(csv_path)
    loads = df["load_N"].tolist()
    defs  = df["deflection_m"].tolist()

    reasons: List[str] = []

    inc = _strictly_increasing(loads)
    if not inc:
        reasons.append("Loads are not strictly increasing.")

    nonneg = all(d >= 0 for d in defs)
    if not nonneg:
        reasons.append("Deflection contains negative values.")

    has_cit = len(citations) > 0
    if not has_cit:
        reasons.append("No citations present.")

    status = "PASS" if (inc and nonneg and has_cit) else "FAIL"
    return ValidationResult(
        loads_strictly_increasing=inc,
        deflection_nonnegative=nonneg,
        has_citation=has_cit,
        num_points=len(df),
        min_load=float(min(loads) if loads else 0.0),
        max_load=float(max(loads) if loads else 0.0),
        min_deflection=float(min(defs) if defs else 0.0),
        max_deflection=float(max(defs) if defs else 0.0),
        status=status,
        fail_reasons=reasons
    )

def write_run_card(*,
                   task_id: int,
                   artifacts_dir: str,
                   timestamp: str,
                   prompt: str,
                   objective: str,
                   params: Dict[str, Any],
                   citations: List[str],
                   metrics: Dict[str, float],
                   artifacts: List[Dict[str, Any]],
                   validation: ValidationResult) -> str:
    """
    Create Markdown run card and return its path.
    """
    outdir = Path(artifacts_dir); outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"run_card_{task_id}.md"

    def _fmt_metrics(m: Dict[str, float]) -> List[str]:
        lines = []
        for k, v in m.items():
            try:
                vv = f"{v:.6g}"
            except Exception:
                vv = str(v)
            lines.append(f"- **{k}**: {vv}")
        return lines

    def _fmt_artifacts(a: List[Dict[str, Any]]) -> List[str]:
        return [f"- **{x.get('kind','file')}**: `{x.get('path','')}`" for x in a]

    md = []
    md.append("# Run Card")
    md.append("")
    md.append(f"- **Task ID**: {task_id}")
    md.append(f"- **Timestamp**: {timestamp}")
    md.append("")
    md.append("## Request")
    md.append(f"- **Prompt**: {prompt}")
    md.append(f"- **Objective**: {objective}")
    md.append("- **Params:**")
    for k, v in params.items():
        md.append(f"  - {k}: {v}")
    md.append("")
    md.append("## Citations")
    if citations:
        for c in citations:
            md.append(f"- {c}")
    else:
        md.append("- (none)")
    md.append("")
    md.append("## Metrics")
    mm = _fmt_metrics(metrics)
    md.extend(mm if mm else ["- (none)"])
    md.append("")
    md.append("## Artifacts")
    aa = _fmt_artifacts(artifacts)
    md.extend(aa if aa else ["- (none)"])
    md.append("")
    md.append("## Validation")
    md.append(f"- **Loads strictly increasing**: {validation.loads_strictly_increasing}")
    md.append(f"- **Deflection nonnegative**: {validation.deflection_nonnegative}")
    md.append(f"- **Has citation**: {validation.has_citation}")
    md.append(f"- **Samples**: {validation.num_points}")
    md.append(f"- **Load range (N)**: {validation.min_load} → {validation.max_load}")
    md.append(f"- **Deflection range (m)**: {validation.min_deflection:.6g} → {validation.max_deflection:.6g}")
    md.append("")
    md.append(f"### Overall Status: **{validation.status}**")
    if validation.fail_reasons:
        md.append("**Reasons:**")
        for r in validation.fail_reasons:
            md.append(f"- {r}")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")
    return str(out_path)

def validate_and_write_run_card(*,
                                task_id: int,
                                artifacts_dir: str,
                                prompt: str,
                                objective: str,
                                params: Dict[str, Any],
                                citations: List[str],
                                metrics: Dict[str, float],
                                artifacts: List[Dict[str, Any]]) -> Tuple[ValidationResult, str]:
    """
    High-level helper: run validation on CSV, then write run card.
    Expects a CSV artifact with kind=='csv'.
    """
    csv_path = None
    for a in artifacts:
        if a.get("kind") == "csv":
            csv_path = a.get("path")
            break
    if not csv_path:
        raise FileNotFoundError("No CSV artifact found for validation.")

    validation = run_validation(csv_path, citations)
    run_card_path = write_run_card(
        task_id=task_id,
        artifacts_dir=artifacts_dir,
        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
        prompt=prompt,
        objective=objective,
        params=params,
        citations=citations,
        metrics=metrics,
        artifacts=artifacts,
        validation=validation
    )
    return validation, run_card_path
