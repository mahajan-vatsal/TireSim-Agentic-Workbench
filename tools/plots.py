# tools/plots.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

# use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_curve(csv_path: str, output_dir: str = "artifacts", filename: str = "curve.png") -> str:
    """
    Read result.csv and save a load–deflection plot to curve.png.
    Returns the PNG path.
    """
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots()
    ax.plot(df["load_N"], df["deflection_m"])
    ax.set_xlabel("Load (N)")
    ax.set_ylabel("Deflection (m)")
    ax.set_title("Vertical Stiffness Sweep (Mock)")
    out = outdir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)
