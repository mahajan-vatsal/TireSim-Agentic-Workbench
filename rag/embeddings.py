# taw/rag/embeddings.py
from __future__ import annotations
import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try FAISS
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# Optional OpenAI backend
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# Optional local sentence-transformers
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = (BASE_DIR / "rag" / "index").resolve()
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

# ---------------- Backends ----------------
class EmbeddingBackend:
    name: str
    dim: int
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

class OpenAIBackend(EmbeddingBackend):
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI backend requested but missing package or OPENAI_API_KEY.")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # text-embedding-3-small: 1536 dims
        self.dim = 1536 if "small" in self.model else 3072
        self.name = f"openai:{self.model}"

    def embed(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return arr

class LocalSTBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not _HAS_ST:
            raise RuntimeError("sentence-transformers not installed.")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = f"local:{model_name}"

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
        return np.array(emb, dtype=np.float32)

def get_backend(prefer: str = "auto") -> EmbeddingBackend:
    """prefer: 'openai' | 'local' | 'auto'."""
    if prefer == "openai" or (prefer == "auto" and os.getenv("OPENAI_API_KEY")):
        try:
            return OpenAIBackend()
        except Exception:
            pass
    return LocalSTBackend()

# ---------------- Vector Index (FAISS) ----------------
@dataclass
class VectorIndex:
    index: Any
    normalize: bool
    dim: int
    backend_name: str
    meta: List[Dict[str, Any]]   # one dict per chunk (includes text + anchors)

    def search(self, query_vecs: np.ndarray, top_k: int = 10) -> List[List[Tuple[int, float]]]:
        qv = query_vecs.astype(np.float32)
        if self.normalize:
            qv = _normalize_rows(qv)
        sims, idxs = self.index.search(qv, top_k)
        out: List[List[Tuple[int, float]]] = []
        for row_idx in range(idxs.shape[0]):
            out.append([(int(idxs[row_idx, j]), float(sims[row_idx, j])) for j in range(idxs.shape[1]) if idxs[row_idx, j] != -1])
        return out

    def save(self, dir_path: str | Path):
        if not _HAS_FAISS:
            raise RuntimeError("faiss-cpu not installed.")
        dirp = Path(dir_path); dirp.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, (dirp / "faiss.index").as_posix())
        # store meta; keep chunk text trimmed to keep file small
        meta_trimmed = []
        for m in self.meta:
            mt = dict(m)
            t = mt.get("text", "")
            if t and len(t) > 700:
                mt["text"] = t[:700]
            meta_trimmed.append(mt)
        (dirp / "meta.json").write_text(json.dumps({
            "normalize": self.normalize,
            "dim": self.dim,
            "backend_name": self.backend_name,
            "meta": meta_trimmed,
        }, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(dir_path: str | Path) -> Optional["VectorIndex"]:
        if not _HAS_FAISS:
            return None
        dirp = Path(dir_path)
        idx_path = dirp / "faiss.index"
        meta_path = dirp / "meta.json"
        if not idx_path.exists() or not meta_path.exists():
            return None
        index = faiss.read_index(idx_path.as_posix())
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        vi = VectorIndex(index=index,
                         normalize=bool(data.get("normalize", True)),
                         dim=int(data.get("dim", 0)),
                         backend_name=str(data.get("backend_name", "unknown")),
                         meta=list(data.get("meta", [])))
        return vi

def build_faiss(texts: List[str], metadatas: List[Dict[str, Any]], backend: EmbeddingBackend,
                normalize: bool = True) -> VectorIndex:
    if not _HAS_FAISS:
        raise RuntimeError("faiss-cpu not installed.")
    vecs = backend.embed(texts).astype(np.float32)
    if normalize:
        vecs = _normalize_rows(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine via dot on unit vecs
    index.add(vecs)
    return VectorIndex(index=index, normalize=normalize, dim=vecs.shape[1], backend_name=backend.name, meta=metadatas)
