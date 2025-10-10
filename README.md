# AI G‑Code Generation

Turn a photograph of a business card into a ready‑to‑engrave laser job. This project brings together a vision model, a language model, image processing and G‑code generation so you can rebuild and customise physical cards with ease. Snap a picture, watch the AI extract the contact details and layout, edit the design if you feel like it, and end up with G‑code and previews for your laser engraver.

---

## 💡 Why this exists

When you look at a business card you don’t just see text – you see logos, icons, a QR code or an NFC chip, and the way everything is arranged. Recreating that by hand is fiddly and time‑consuming. The idea here is to automate the tedious parts: use a vision model to read the card and locate its elements, write out an SVG that matches the layout, convert it to a black‑and‑white image and finally to G‑code for a laser cutter. You can interact with the process at the point where it matters – tell the model which parts to move, delete or replace – and preview your changes before committing.

---

## ✨ Features at a glance

1. **Vision‑language extraction** – A Qwen2.5‑VL model recognises names, titles, phone numbers, emails, addresses, websites, QR codes, NFC chips, company logos and other icons from a card image. It returns structured JSON so you can re‑use the data.
2. **Layout analysis** – The same model also produces bounding boxes for text, logos and icons. Those coordinates are converted into millimetres on an 85×54 mm card so that the geometry is preserved.
3. **Automatic SVG generation** – From the extracted data the code assembles an SVG business card. It embeds the original or replacement logos, draws a QR code if one was detected, positions text, and includes optional NFC chip templates from the **assets/nfc_templates** folder.
4. **Interactive editing** – The pipeline pauses to let you refine the design. You can either accept the generated SVG or ask for changes. Instructions like “move 3 to x=20,y=30” or “replace with ‘Senior Developer’” are parsed and applied to the SVG. An LLM (Mistral 7B via OpenRouter) can also convert natural language instructions into these edit commands.
5. **Rasterisation and binarisation** – The final SVG is rendered to a PNG and then binarised into a black‑and‑white image ready for engraving.
6. **G‑code generation & preview** – The black‑and‑white image is scanned line by line to produce G‑code that controls laser power on dark pixels and travels quickly over white pixels. A preview tool plots the laser path on a canvas so you can check it before engraving.


---

## 🛠 How it works

1. **Extract information** — The **ocr_agent** uses an advanced vision model (via Fireworks/OpenAI) to read the card and return structured JSON fields (name, title, contact details, conference info, etc.).
2. **Detect layout elements** —The **visual_analysis_agent** calls a vision language model (Qwen2.5‑VL) to detect bounding boxes for all visual items (text, logos, QR code, NFC chip) and enriches them with sizes in millimetres.
3. **Generate SVG design** — The **svg_agent** assembles an SVG from the detected text blocks, logos, icons and optional user overrides. It can embed QR codes and NFC icons and flips the Y‑axis to match millimetre coordinates.
4. **Preview and edit** — Users can preview the card and optionally modify it. The **svg_preview_agent** launches a zoomable Tkinter window; the **svg_mapper_agent** maps semantic elements and gives them IDs; the **llm_svg_agent** uses a language model to turn free‑form instructions into edit commands; the **svg_editor_agent** applies those commands (move, delete, replace) to the SVG.
5. **Rasterize and binarize** — The **rasterization** module converts the SVG into a high‑resolution PNG and then into a black‑and‑white image, ready for engraving.
6. **Generate G‑code** — The **gcode_agent** reads the binarized image and produces a scanline G‑code program, including zig‑zag motion, laser on/off commands and proper feedrates
7. **Preview G‑code** – The **gcode_preview_agent** parses G‑code, scales it to fit a canvas and draws the toolpath so you can visualise the engraving before running it.

The entire sequence is orchestrated via a LangGraph graph in **graph/main_graph.py**. Users can choose to edit the SVG or proceed directly to rasterization and G‑code generation.


---

## 📦 Installation
1. **Clone the repo** (or download the ZIP if using internal connectors):
```bash
git clone https://github.com/mahajan-vatsal/AI-GCode-Generation.git
```
2. **Set up Python** – This project requires Python ≥ 3.12. Create a virtual environment and install dependencies:
```bash
cd AI-GCode-Generation
python3 -m nenv env
source env/bin/activate (for macOS)
source env\bin\activate (for Windows)
pip install -r requirements.txt
```
3. **API keys** – Create a **.env** file in the project root with the following keys:
```bash
# Fireworks (OpenAI) for OCR and layout detection
FIREWORKS_API_KEY=your_fireworks_key_here
# OpenRouter (Mistral) for generating SVG edit commands
OPENROUTER_API_KEY=your_openrouter_key_here
#Langgraph for defining the Workflow
LANGCHAIN_PROJECT=AI-GCode-Generator
export LANGCHAIN_API_KEY=your_langchain_key_here
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=AI-GCode-Generator
```
4. **Run the workflow** - Use the LangGraph API (installed via langgraph-api):
```bash
langgraph dev
```
5. **Output** - After completion you will find:
- **output.svg / output_edited.svg** – the generated vector business card.
- **output_edited.png** and **output_edited_bw.png** – rasterized versions used for engraving.
- **output_edited.gcode** – the final G‑code file ready for your CNC or laser engraver.

## 📦 Workflow with Docker
1. **Clone the repo** (or download the ZIP if using internal connectors):
```bash
git clone https://github.com/mahajan-vatsal/AI-GCode-Generation.git
```

2. **Build the Docker Image**:
```bash
cd AI-GCode-Generation
docker-compose build
```

3. **Launch the Container**:
```bash
docker-compose up
```

4. **Access the System Locally**:
```bash
Web UI: http://localhost:8501
Backend API: http://localhost:8080
```
---

## 📦 Triggering the Web UI

After cloning the repository, set up a virtual environment, install the required dependencies, and configure the necessary API keys.
1. **Start the Backend API Server**:
```bash
uvicorn server.server_api:app --host 0.0.0.0 --port 8080
```

2. **Frontend Web UI**: 
```bash
streamlit run server/streamlit.py --server.address 0.0.0.0 --server.port 8501
```

3. **API Documentation with FastAPI**:
```bash
http://localhost:8080/docs
```
---

## 📊 LangGraph Flow
Below is the actual LangGraph pipeline used in this project:

<img width="500" height="500" alt="Langgraph flow" src="https://github.com/user-attachments/assets/c3f595e6-398b-45b5-be69-f8b26365d99d" />

---

## Web UI 

<img width="1440" height="900" alt="WebUI" src="https://github.com/user-attachments/assets/c0596157-5213-4502-8164-eb700253bf7b" />

---
## API Documentation with FastAPI

<img width="1440" height="895" alt="API" src="https://github.com/user-attachments/assets/c887b3d5-c4a5-4e0a-9ac0-29be0422186f" />







