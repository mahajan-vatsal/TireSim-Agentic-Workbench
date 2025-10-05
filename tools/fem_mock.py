# tools/fem_mock.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import re, hashlib
import numpy as np

def _seed_from_text(text: str) -> int:
    """Deterministic seed from deck text (stable across machines)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)

def _parse_load_sweep(deck_text: str) -> List[float]:
    """
    Parse load_sweep: [a, b, c] from the deck text. If missing, return a default sweep.
    """
    m = re.search(r"load_sweep:\s*\[([^\]]+)\]", deck_text, flags=re.IGNORECASE)
    if not m:
        # default: 0..50k in 21 steps
        return list(np.linspace(0, 50_000, 21))
    raw = m.group(1)
    vals = []
    for part in raw.split(","):
        try:
            vals.append(float(part.strip()))
        except ValueError:
            pass
    if not vals:
        return list(np.linspace(0, 50_000, 21))
    return vals

def run_case(deck_text: str, output_dir: str = "artifacts") -> Tuple[str, str, str]:
    """
    Mock FEM run:
      - Saves the incoming deck to deck.txt
      - Generates result.csv with columns: load_N, deflection_m
      - Writes solver.log with a brief summary
    Returns (csv_path, log_path, deck_path)
    """
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # Save deck text for provenance
    deck_path = outdir / "deck.txt"
    deck_path.write_text(deck_text, encoding="utf-8")

    # Deterministic seed based on deck content
    seed = _seed_from_text(deck_text)
    rng = np.random.default_rng(seed)

    loads = np.array(_parse_load_sweep(deck_text), dtype=float)
    loads = np.clip(loads, a_min=0, a_max=None)

    # Simple stiffness model (N/m) with tiny seed-based jitter
    stiffness = 250_000.0 + rng.normal(0, 5_000.0)  # around 250 kN/m
    # deflection = load / stiffness + small noise
    deflection = loads / stiffness + rng.normal(0, 5e-4, size=loads.shape)

    # Ensure non-negative deflection & monotonic load
    deflection = np.maximum(deflection, 0.0)
    loads = np.sort(loads)

    # Write CSV
    csv_path = outdir / "result.csv"
    lines = ["load_N,deflection_m"] + [f"{L:.1f},{d:.6f}" for L, d in zip(loads, deflection)]
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    # Write log
    log_path = outdir / "solver.log"
    log = [
        "Mock FEM run OK",
        f"seed={seed}",
        f"stiffness_estimate_N_per_m≈{stiffness:.1f}",
        f"samples={len(loads)}",
    ]
    log_path.write_text("\n".join(log), encoding="utf-8")

    return str(csv_path), str(log_path), str(deck_path)
