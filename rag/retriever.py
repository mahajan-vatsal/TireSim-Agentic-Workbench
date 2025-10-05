# taw/rag/retriever.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os
import math
import numpy as np

from .indexer import load_corpus, keyword_scores, BM25Index, TfidfIndex
from .embeddings import VectorIndex, INDEX_DIR, get_backend

@dataclass
class RetrievedChunk:
    text: str
    source_id: str  # e.g., "Boundary_Conditions.md#boundary-conditions-for-vertical-stiffness"
    score: float
    title: str

# ----- load vector index -----
def _load_vector_index() -> Optional[VectorIndex]:
    try:
        return VectorIndex.load(INDEX_DIR)
    except Exception:
        return None

def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    rng = (mx - mn) or 1.0
    return {k: (v - mn) / rng for k, v in scores.items()}

def _dense_scores(vi: VectorIndex, query: str, top_k: int = 50) -> Dict[str, float]:
    backend = get_backend("auto")
    qv = backend.embed([query]).astype(np.float32)
    hits = vi.search(qv, top_k=top_k)[0]  # [(idx, sim)]
    out: Dict[str, float] = {}
    for idx, sim in hits:
        meta = vi.meta[idx]
        out[meta["chunk_id"]] = float(sim)
    return out

def _materialize_docs(doc_ids: List[str]) -> List[RetrievedChunk]:
    docs = load_corpus()
    by_id = {d.doc_id: d for d in docs}
    out: List[RetrievedChunk] = []
    for did in doc_ids:
        d = by_id.get(did)
        if not d: 
            continue
        snippet = "\n".join(d.text.strip().splitlines()[:6])[:800]
        out.append(RetrievedChunk(text=snippet, source_id=f"{d.doc_id}#top", score=0.0, title=d.title))
    return out

def _materialize_chunks(vi: VectorIndex, chunk_ids: List[str]) -> List[RetrievedChunk]:
    out: List[RetrievedChunk] = []
    meta_by_id = {m["chunk_id"]: m for m in vi.meta}
    for cid in chunk_ids:
        m = meta_by_id.get(cid)
        if not m: 
            continue
        text = m.get("text", "")
        title = m.get("title", m.get("doc_id",""))
        out.append(RetrievedChunk(text=text[:800], source_id=m["source_id"], score=0.0, title=title))
    return out

def retrieve(query: str, k: int = 3, method: str = "hybrid") -> List[RetrievedChunk]:
    """
    Modes:
      - "keyword" | "bm25" | "tfidf"
      - "dense"           (FAISS only)
      - "hybrid"          (BM25 + TF-IDF + keywords; doc-level)
      - "hybrid_v2"       (BM25 doc-level + dense chunk-level, optional CrossEncoder rerank)
    """
    docs = load_corpus()
    bm25 = BM25Index(docs)
    tfidf = TfidfIndex(docs)
    vi = _load_vector_index()

    # lexical scores (doc-level)
    bm = bm25.score(query)
    tf = tfidf.score(query)
    kw = keyword_scores(query, docs)

    if method == "keyword":
        ranked = sorted(kw.items(), key=lambda x: x[1], reverse=True)
        res = _materialize_docs([d for d, _ in ranked[:k]])
        for r in res:
            r.score = float(kw.get(r.source_id.split("#")[0], 0.0))
        return res

    if method == "bm25":
        ranked = sorted(bm.items(), key=lambda x: x[1], reverse=True)
        res = _materialize_docs([d for d, _ in ranked[:k]])
        for r in res:
            r.score = float(bm.get(r.source_id.split("#")[0], 0.0))
        return res

    if method == "tfidf":
        ranked = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        res = _materialize_docs([d for d, _ in ranked[:k]])
        for r in res:
            r.score = float(tf.get(r.source_id.split("#")[0], 0.0))
        return res

    if method == "dense":
        if not vi:
            # fall back to bm25
            return retrieve(query, k=k, method="bm25")
        ds = _dense_scores(vi, query, top_k=max(50, k))
        ranked = sorted(ds.items(), key=lambda x: x[1], reverse=True)[:k]
        out: List[RetrievedChunk] = []
        meta_by_id = {m["chunk_id"]: m for m in vi.meta}
        for cid, sc in ranked:
            m = meta_by_id.get(cid)
            if not m:
                continue
            text = m.get("text", "")
            out.append(RetrievedChunk(text=text[:800], source_id=m["source_id"], score=float(sc), title=m.get("title", "")))
        return out

    if method == "hybrid":
        weights = {"bm25": 0.6, "tfidf": 0.3, "keyword": 0.1}
        acc: Dict[str, float] = {d.doc_id: 0.0 for d in docs}
        for name, raw in [("bm25", bm), ("tfidf", tf), ("keyword", kw)]:
            mm = _minmax(raw)
            w = weights[name]
            for kdoc, v in mm.items():
                acc[kdoc] += w * v
        ranked_ids = [doc for doc, _ in sorted(acc.items(), key=lambda x: x[1], reverse=True)]
        res = _materialize_docs(ranked_ids[:k])
        for r in res:
            r.score = float(acc.get(r.source_id.split("#")[0], 0.0))
        return res

    # ---------- hybrid_v2: BM25 doc + dense chunks (+ optional rerank) ----------
    if method == "hybrid_v2":
        if not vi:
            # degrade gracefully
            return retrieve(query, k=k, method="hybrid")
        bm_norm = _minmax(bm)
        top_bm_docs = [doc for doc, _ in sorted(bm.items(), key=lambda x: x[1], reverse=True)[:20]]
        ds = _dense_scores(vi, query, top_k=100)  # chunk-level
        # candidate pool: doc tops + dense chunks
        cand_scores: Dict[str, float] = {}
        # doc tops as synthetic keys "<doc>#top"
        for did in top_bm_docs:
            cand_scores[f"{did}#top"] = bm_norm.get(did, 0.0)
        # dense chunk IDs as-is
        # normalize dense sims
        ds_norm = _minmax(ds)
        for cid, sc in ds_norm.items():
            cand_scores[cid] = sc

        def blended(key: str) -> float:
            wb, wd = 0.45, 0.55  # weights: BM25 (doc) vs dense (chunk)
            if key.endswith("#top"):
                b = cand_scores.get(key, 0.0)
                d = 0.0
            else:
                # chunk: use dense-only for now
                b = 0.0
                d = cand_scores.get(key, 0.0)
            return wb * b + wd * d

        ranked_keys = sorted(cand_scores.keys(), key=lambda kk: blended(kk), reverse=True)[: max(50, k)]

        # Optional CrossEncoder rerank
        use_rerank = os.getenv("USE_RERANKER", "0").strip().lower() in {"1","true","yes"}
        if use_rerank:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
                ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                # Build (query, text) pairs
                texts: List[str] = []
                for key in ranked_keys:
                    if key.endswith("#top"):
                        # doc preview
                        doc_id = key.split("#")[0]
                        d = next((x for x in docs if x.doc_id == doc_id), None)
                        preview = "\n".join((d.text if d else "").splitlines()[:6])[:800]
                        texts.append(preview)
                    else:
                        meta = next((m for m in vi.meta if m["chunk_id"] == key), None)
                        texts.append(meta.get("text","")[:800] if meta else "")
                pairs = [(query, t) for t in texts]
                rr_scores = ce.predict(pairs)
                # Replace blended ranking with rerank scores (desc)
                ranked_keys = [rk for _, rk in sorted(zip(rr_scores, ranked_keys), key=lambda x: x[0], reverse=True)]
            except Exception:
                pass

        # materialize top-k
        out: List[RetrievedChunk] = []
        meta_by_id = {m["chunk_id"]: m for m in vi.meta}
        for key in ranked_keys[:k]:
            if key.endswith("#top"):
                doc_id = key.split("#")[0]
                d = next((x for x in docs if x.doc_id == doc_id), None)
                snippet = "\n".join((d.text if d else "").splitlines()[:6])[:800]
                out.append(RetrievedChunk(text=snippet, source_id=f"{doc_id}#top", score=blended(key), title=(d.title if d else doc_id)))
            else:
                m = meta_by_id.get(key)
                if not m:
                    continue
                text = m.get("text","")[:800]
                out.append(RetrievedChunk(text=text, source_id=m["source_id"], score=blended(key), title=m.get("title","")))
        return out

    # fallback
    return retrieve(query, k=k, method="hybrid")
