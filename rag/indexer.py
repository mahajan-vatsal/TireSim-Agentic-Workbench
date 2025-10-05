# rag/indexer.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import math
import re
import numpy as np

CORPUS_DIR = (Path(__file__).resolve().parent / "corpus").resolve()

_token_re = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    return _token_re.findall(text.lower())

@dataclass
class CorpusDoc:
    doc_id: str          # filename
    title: str
    text: str
    tokens: List[str]
    length: int

def load_corpus(corpus_dir: Path = CORPUS_DIR) -> List[CorpusDoc]:
    docs: List[CorpusDoc] = []
    for p in sorted(corpus_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text.strip() else p.stem
        toks = tokenize(text)
        docs.append(CorpusDoc(
            doc_id=p.name,
            title=title,
            text=text,
            tokens=toks,
            length=len(toks)
        ))
    return docs

# --------------------------
# Keyword scoring (MVP)
# --------------------------
def keyword_scores(query: str, docs: List[CorpusDoc]) -> Dict[str, float]:
    q_terms = tokenize(query)
    scores: Dict[str, float] = {}
    for d in docs:
        score = sum(d.text.lower().count(t) for t in q_terms)
        if score > 0:
            scores[d.doc_id] = float(score)
        else:
            scores.setdefault(d.doc_id, 0.0)
    return scores

# --------------------------
# BM25 implementation
# --------------------------
class BM25Index:
    def __init__(self, docs: List[CorpusDoc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.avgdl = sum(d.length for d in docs) / max(1, self.N)
        # Build DF and per-doc term frequencies
        self.df: Dict[str, int] = {}
        self.tf: Dict[str, Dict[str, int]] = {}  # doc_id -> {term: count}
        for d in docs:
            tf_d: Dict[str, int] = {}
            for t in d.tokens:
                tf_d[t] = tf_d.get(t, 0) + 1
            self.tf[d.doc_id] = tf_d
            for t in tf_d.keys():
                self.df[t] = self.df.get(t, 0) + 1
        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for t, df in self.df.items():
            # classic BM25 idf
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> Dict[str, float]:
        q_terms = tokenize(query)
        scores: Dict[str, float] = {}
        for d in self.docs:
            s = 0.0
            tf_d = self.tf[d.doc_id]
            for t in q_terms:
                if t not in tf_d:
                    continue
                idf = self.idf.get(t, 0.0)
                tf = tf_d[t]
                denom = tf + self.k1 * (1 - self.b + self.b * (d.length / (self.avgdl or 1.0)))
                s += idf * (tf * (self.k1 + 1)) / (denom or 1.0)
            scores[d.doc_id] = s
        return scores

# --------------------------
# TF-IDF (cosine) — light-weight
# --------------------------
class TfidfIndex:
    def __init__(self, docs: List[CorpusDoc]):
        self.docs = docs
        self.N = len(docs)
        # vocab & df
        df: Dict[str, int] = {}
        for d in docs:
            seen = set(d.tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
        self.idf = np.zeros(len(self.vocab), dtype=float)
        for t, i in self.vocab.items():
            self.idf[i] = math.log((self.N + 1) / (df[t] + 1)) + 1.0
        # doc vectors
        self.doc_vecs = np.zeros((self.N, len(self.vocab)), dtype=float)
        for idx, d in enumerate(docs):
            tf: Dict[int, float] = {}
            for t in d.tokens:
                i = self.vocab.get(t)
                if i is not None:
                    tf[i] = tf.get(i, 0.0) + 1.0
            if tf:
                vec = np.zeros(len(self.vocab), dtype=float)
                for i, v in tf.items():
                    vec[i] = (v) * self.idf[i]
                # L2 normalize
                norm = np.linalg.norm(vec) or 1.0
                self.doc_vecs[idx, :] = vec / norm

    def _query_vec(self, query: str) -> np.ndarray:
        tf: Dict[int, float] = {}
        for t in tokenize(query):
            i = self.vocab.get(t)
            if i is not None:
                tf[i] = tf.get(i, 0.0) + 1.0
        if not tf:
            return np.zeros(len(self.vocab), dtype=float)
        vec = np.zeros(len(self.vocab), dtype=float)
        for i, v in tf.items():
            vec[i] = (v) * self.idf[i]
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm

    def score(self, query: str) -> Dict[str, float]:
        qv = self._query_vec(query)
        sims = self.doc_vecs @ qv
        # map back
        scores: Dict[str, float] = {}
        for idx, d in enumerate(self.docs):
            scores[d.doc_id] = float(sims[idx])
        return scores
