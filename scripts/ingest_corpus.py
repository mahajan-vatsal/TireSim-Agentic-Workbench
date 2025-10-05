# taw/scripts/ingest_corpus.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple

from rag.embeddings import get_backend, build_faiss, INDEX_DIR, VectorIndex, BASE_DIR

CORPUS_DIR = (BASE_DIR / "rag" / "corpus").resolve()

def slugify(text: str) -> str:
    t = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    t = re.sub(r"\s+", "-", t)
    t = re.sub(r"-+", "-", t)
    return t

def read_markdown_with_anchors(path: Path) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Returns (title, sections) where sections = [(anchor, section_text)]
    Split by headings (# ...).
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    title = lines[0].lstrip("# ").strip() if lines else path.stem

    sections: List[Tuple[str, str]] = []
    current_anchor = slugify(title)
    current_buffer: List[str] = []
    for line in lines:
        if line.strip().startswith("#"):
            if current_buffer:
                text = "\n".join(current_buffer).strip()
                if text:
                    sections.append((current_anchor, text))
            hdr = line.lstrip("# ").strip()
            current_anchor = slugify(hdr)
            current_buffer = [line]
        else:
            current_buffer.append(line)
    if current_buffer:
        text = "\n".join(current_buffer).strip()
        if text:
            sections.append((current_anchor, text))
    return title, sections

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto","openai","local"], default="auto",
                    help="Embedding backend. 'auto' uses OpenAI if OPENAI_API_KEY is set, else local MiniLM.")
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    args = ap.parse_args()

    backend = get_backend(args.backend)
    print(f"[ingest] Using backend: {backend.__class__.__name__} ({backend.name}, dim={backend.dim})")

    all_texts: List[str] = []
    all_meta: List[Dict] = []

    for p in sorted(CORPUS_DIR.glob("*.md")):
        title, sections = read_markdown_with_anchors(p)
        for anchor, sect_text in sections:
            chunks = chunk_text(sect_text, args.max_chars, args.overlap)
            for idx, ch in enumerate(chunks):
                source_id = f"{p.name}#{anchor}"
                meta = {
                    "doc_id": p.name,
                    "title": title,
                    "section": anchor,
                    "chunk_id": f"{p.name}::{anchor}::{idx}",
                    "source_id": source_id,
                    "path": p.as_posix(),
                    "text": ch,
                }
                all_texts.append(ch)
                all_meta.append(meta)

    vi = build_faiss(all_texts, all_meta, backend, normalize=True)
    vi.save(INDEX_DIR)

    (INDEX_DIR / "chunks.json").write_text(json.dumps({
        "backend": vi.backend_name, "dim": vi.dim, "count": len(all_texts)
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ingest] Saved index to {INDEX_DIR} (faiss.index, meta.json)")

if __name__ == "__main__":
    main()
