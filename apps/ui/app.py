# taw/apps/ui/app.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
ARTIFACTS_BASE = f"{API_URL.rstrip('/')}/artifacts"

st.set_page_config(page_title="TireSim Agentic Workbench", layout="wide")

st.title("🛞 TireSim Agentic Workbench — Demo UI")

with st.sidebar:
    st.markdown("### Settings")
    api_url = st.text_input("API URL", API_URL)
    backend = st.selectbox("Embedding backend for ingestion", ["openai", "local", "auto"], index=0)
    st.caption("Run ingestion after uploading docs. For local backend you need sentence-transformers + torch set up.")

tabs = st.tabs(["📚 Corpus", "⚙️ Run", "📄 View Task"])

# ---------------- Tab 1: Corpus ----------------
with tabs[0]:
    st.subheader("Upload corpus files (.md)")
    uploaded = st.file_uploader("Drop Markdown files", type=["md"], accept_multiple_files=True)
    corpus_dir = Path(__file__).resolve().parents[2] / "rag" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if uploaded:
        for f in uploaded:
            out = corpus_dir / f.name
            out.write_bytes(f.read())
        st.success(f"Saved {len(uploaded)} file(s) to {corpus_dir}")

    st.markdown("---")
    colA, colB = st.columns([1,2])
    with colA:
        if st.button("Build / Rebuild vector index"):
            try:
                resp = requests.post(f"{api_url}/ingest", json={"backend": backend})
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Index built with backend={data['backend']} • dim={data['dim']} • count={data['count']}")
                else:
                    st.error(f"Ingest failed: {resp.status_code} {resp.text}")
            except Exception as e:
                st.error(f"Error calling API: {e}")
    with colB:
        st.info("The API reads Markdown files from `taw/rag/corpus/`, chunks/anchors them, and writes FAISS to `taw/rag/index/`.")

# ---------------- Tab 2: Run ----------------
with tabs[1]:
    st.subheader("Plan & Run a task")
    default_obj = "Vertical load sweep for 315/80 R22.5 at 8 bar"
    objective = st.text_input("Objective", value=default_obj)
    tire_size = st.text_input("tire_size", value="315/80 R22.5")
    rim = st.text_input("rim", value="9.00x22.5")
    inflation_bar = st.number_input("inflation_bar", value=8.0, step=0.5)
    mesh_mm = st.number_input("mesh_mm", value=8.0, step=0.5)
    load_sweep = st.text_input("load_sweep (comma separated N)", value="0,10000,20000,30000,40000,50000")

    if st.button("Run"):
        try:
            params = {
                "tire_size": tire_size,
                "rim": rim,
                "inflation_bar": float(inflation_bar),
                "mesh_mm": float(mesh_mm),
                "load_sweep": [int(x.strip()) for x in load_sweep.split(",") if x.strip()],
            }
            payload = {"objective": objective, "params": params}
            resp = requests.post(f"{api_url}/plan-run", json=payload, timeout=120)
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} {resp.text}")
            else:
                data = resp.json()
                st.session_state["last_task_id"] = data["task_id"]
                st.success(f"Run complete. Task ID: {data['task_id']}")
                st.json(data)

                # Show plot if available
                png_url = data.get("artifact_urls", {}).get("plot")
                if png_url:
                    st.image(f"{api_url}{png_url}", caption="Load–deflection curve")

                # Show Run Card
                if data.get("run_card_url"):
                    st.markdown("### Run Card")
                    try:
                        rc_text = requests.get(f"{api_url}{data['run_card_url']}").text
                        st.markdown(rc_text)
                    except Exception:
                        st.info(f"Run Card: {api_url}{data['run_card_url']}")
        except Exception as e:
            st.error(f"Client error: {e}")

# ---------------- Tab 3: View Task ----------------
with tabs[2]:
    st.subheader("Fetch a past task")
    task_id = st.number_input("Task ID", min_value=1, step=1, value=int(st.session_state.get("last_task_id", 1)))
    if st.button("Get task"):
        try:
            resp = requests.get(f"{api_url}/tasks/{task_id}")
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} {resp.text}")
            else:
                data = resp.json()
                st.write("### Task")
                st.json(data.get("task", {}))
                st.write("### Metrics")
                st.json(data.get("metrics", {}))
                st.write("### Runs")
                st.json(data.get("runs", []))

                rc_text = data.get("run_card")
                if rc_text:
                    st.markdown("### Run Card")
                    st.markdown(rc_text)

                png_url = data.get("artifact_urls", {}).get("png")
                if png_url:
                    st.image(f"{api_url}{png_url}", caption="Load–deflection curve")
        except Exception as e:
            st.error(f"Client error: {e}")
