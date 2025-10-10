# TireSim Agentic Workbench

This project implements a self‑contained workbench for automating tire finite‑element simulations with agentic behaviour and retrieval‑augmented generation (RAG). It plans and executes a vertical stiffness sweep, logs every step to a SQLite datastore, retrieves relevant background information from a user‑provided corpus and produces a structured run card summarising the results.

---

## ✨ Features at a glance

1. **Retrieval‑augmented planning** – When a user provides an objective, the agent optionally fetches background material from a local Markdown corpus. The retriever supports keyword, BM25, dense and hybrid methods to provide contextual citations for the simulation. A large‑language model (LLM) then plans a series of steps to achieve the objective.
2. **Template‑driven deck generation** – Input decks are defined as Jinja templates. The built‑in vertical_stiffness.j2 template takes parameters such as tire size, rim, inflation pressure, mesh size and load sweep and produces a deterministic FEM deck. Users can add additional templates for other analyses by dropping new .j2 files into the templates/ folder.
3. **Mock FEM solver** – To keep the demo self‑contained, a deterministic mock solver generates load‑deflection curves based on the deck text. It outputs a CSV, a log and a copy of the deck, which are stored as artifacts.
4. **Automated plotting and metrics** – A plotting tool uses Matplotlib to create a PNG of the load–deflection curve. Post‑processing then computes vertical stiffness (slope) and other metrics. An LLM produces a concise summary with inline citations
5. **Validation and run card generation** – The run_card.py module checks that loads are strictly increasing, deflections are non‑negative and at least one citation exists. It writes a Run Card summarising the prompt, objective, parameters, citations, metrics, artifacts and validation status in Markdown.
6. **Datastore** – Simulation tasks and runs are logged to a SQLite database. Each run stores its parameters, metrics, artifacts and validation status for later retrieval.

---


## 📦 Installation
1. **Clone the repo** (or download the ZIP if using internal connectors):
```bash
git clone https://github.com/mahajan-vatsal/TireSim-Agentic-Workbench.git
```
2. **Set up Python** – This project requires Python ≥ 3.12. Create a virtual environment and install dependencies:
```bash
cd TireSim-Agentic-Workbench
python3 -m nenv env
source env/bin/activate (for macOS)
source env\bin\activate (for Windows)
pip install -r requirements.txt
```
3. **API keys** – Create a **.env** file in the project root with the following keys:
```bash
# LLM Provider
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=your_model_name
OPENAI_EMBED_MODEL=your_embeddingmodel_name
#Langgraph for defining the Workflow
LANGCHAIN_PROJECT=TireSim Agentic Workbench
export LANGCHAIN_API_KEY=your_langchain_key_here
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=TireSim Agentic Workbench
```
4. **Run the workflow** - Use the LangGraph API (installed via langgraph-api):
```bash
langgraph dev
```

## 📊 LangGraph Flow
Below is the actual LangGraph pipeline used in this project:

<img width="651" height="721" alt="TireSim workflow" src="https://github.com/user-attachments/assets/daa24949-d53c-4194-80ca-15f6c80cf750" />

## 🎯 Output
The figure below is a typical output from a vertical stiffness sweep. It shows the simulated relationship between vertical load and tyre deflection. The slope of the line (≈250 kN/m) corresponds to the estimated vertical stiffness reported in the run card.

<img width="876" height="678" alt="curve" src="https://github.com/user-attachments/assets/f6942faa-15c9-40aa-a88b-4ca63ace65d6" />

Below is the Run Card image summarising the objective, parameters, citations, metrics, artifacts and validation status in Markdown.
<img width="651" height="797" alt="run card" src="https://github.com/user-attachments/assets/e207129f-c71a-404a-b3c1-76db077f6ad7" />







