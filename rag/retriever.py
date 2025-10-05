
# rag/retriever.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Literal
from pathlib import Path

from .indexer import load_corpus, keyword_scores, BM25Index, TfidfIndex

@dataclass
class RetrievedChunk:
    text: str
    source_id: str
    score: float
    title: str

class Retriever:
    def __init__(self, corpus_dir: Path | None = None):
        self.docs = load_corpus(corpus_dir or Path(__file__).resolve().parent / "corpus")
        # Build indexes
        self.bm25 = BM25Index(self.docs)
        self.tfidf = TfidfIndex(self.docs)
        # no precompute needed for keyword_scores

    def _combine_scores(self, methods: List[str], query: str) -> Dict[str, float]:
        # each method contributes; weight them (BM25 strongest)
        weights = {"bm25": 0.6, "tfidf": 0.3, "keyword": 0.1}
        acc: Dict[str, float] = {d.doc_id: 0.0 for d in self.docs}

        for m in methods:
            if m == "bm25":
                s = self.bm25.score(query)
            elif m == "tfidf":
                s = self.tfidf.score(query)
            elif m == "keyword":
                s = keyword_scores(query, self.docs)
            else:
                continue
            w = weights.get(m, 0.0)
            # normalize per method to [0,1] before weighting
            vals = list(s.values())
            if vals:
                mn, mx = min(vals), max(vals)
                rng = (mx - mn) or 1.0
                for k, v in s.items():
                    acc[k] += w * ((v - mn) / rng)
        return acc

    def retrieve(self, query: str, k: int = 3, method: Literal["keyword","bm25","tfidf","hybrid"]="hybrid") -> List[RetrievedChunk]:
        if method == "keyword":
            scores = keyword_scores(query, self.docs)
        elif method == "bm25":
            scores = self.bm25.score(query)
        elif method == "tfidf":
            scores = self.tfidf.score(query)
        else:
            scores = self._combine_scores(["bm25","tfidf","keyword"], query)

        ranked = sorted(self.docs, key=lambda d: scores.get(d.doc_id, 0.0), reverse=True)
        out: List[RetrievedChunk] = []
        for d in ranked[:k]:
            # short preview text for display
            preview = d.text.strip().splitlines()
            snippet = "\n".join(preview[:5])[:800]
            out.append(RetrievedChunk(
                text=snippet,
                source_id=d.doc_id,
                score=float(scores.get(d.doc_id, 0.0)),
                title=d.title
            ))
        return out

# convenience function mirroring earlier plan
def retrieve(query: str, k: int = 3, method: str = "hybrid") -> List[RetrievedChunk]:
    r = Retriever()
    return r.retrieve(query, k=k, method=method if method in {"keyword","bm25","tfidf","hybrid"} else "hybrid")
