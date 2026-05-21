"""
BM25 keyword search for hybrid retrieval (BM25 + dense → RRF fusion).

Builds a BM25 index from all indexed chunks and provides
keyword-aware search that complements dense vector retrieval.
"""
from __future__ import annotations

import re
import numpy as np
from collections.abc import Sequence


class BM25Retriever:
    """
    Scikit-learn-free BM25 implementation for lightweight keyword search.

    Uses the Okapi BM25 formula with configurable k1 and b parameters.
    No external dependencies beyond numpy.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []          # tokenized documents
        self._metadata: list[dict] = []               # per-document metadata
        self._doc_len: np.ndarray | None = None
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._built = False

    # ── tokenization ──────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenizer: lowercase + split on non-alphanumeric chars."""
        return re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())

    # ── index building ────────────────────────────────────────

    def index(self, documents: list[str], metadatas: list[dict] | None = None):
        """
        Build BM25 index from a list of document strings.

        documents: list of chunk texts
        metadatas: optional list of metadata dicts (paper_name, page, etc.)
        """
        self._corpus = [self._tokenize(doc) for doc in documents]
        self._metadata = metadatas or [{}] * len(documents)
        self._doc_len = np.array([len(d) for d in self._corpus], dtype=np.float32)
        self._avgdl = float(np.mean(self._doc_len)) if len(self._corpus) > 0 else 1.0

        # Compute IDF for each term
        N = len(self._corpus)
        df: dict[str, int] = {}
        for doc in self._corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        self._idf = {
            term: max(0, np.log((N - freq + 0.5) / (freq + 0.5) + 1))
            for term, freq in df.items()
        }
        self._built = True

    # ── search ────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """
        Search for query, returning list of (doc_index, bm25_score).
        Sorted by score descending.
        """
        if not self._built:
            return []

        query_tokens = self._tokenize(query)
        scores = np.zeros(len(self._corpus), dtype=np.float32)

        for term in query_tokens:
            idf = self._idf.get(term, 0)
            if idf == 0:
                continue

            # For each document, compute BM25 score for this term
            for i, doc_tokens in enumerate(self._corpus):
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue
                doc_len = float(self._doc_len[i])
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                scores[i] += idf * numerator / denominator

        # Get top-k indices
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    @property
    def doc_count(self) -> int:
        return len(self._corpus)


# ── RRF (Reciprocal Rank Fusion) ──────────────────────────────

def rrf_fuse(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    """
    Fuse dense and BM25 result lists using Reciprocal Rank Fusion.

    Each result dict must have a unique 'chunk_id' (or we key by text content).
    Returns merged list sorted by RRF score descending.

    k: RRF constant (default 60, standard value)
    top_k: optional limit on output count
    """
    scores: dict[str, float] = {}
    results_map: dict[str, dict] = {}

    def _key(r: dict) -> str:
        return f"{r.get('paper_name', '')}::{r.get('chunk_id', hash(r['text']))}"

    for rank, r in enumerate(dense_results):
        key = _key(r)
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        results_map[key] = r

    for rank, r in enumerate(bm25_results):
        key = _key(r)
        scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
        if key not in results_map:
            results_map[key] = r

    # Sort by RRF score descending
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    merged = [results_map[key] for key in sorted_keys]
    # Carry forward the best score from either source
    for r in merged:
        key = _key(r)
        r["rrf_score"] = round(scores[key], 6)

    if top_k:
        merged = merged[:top_k]

    return merged


# ── Convenience: build BM25 from Chroma collection ────────────

def build_bm25_from_collection(collection) -> BM25Retriever:
    """
    Build a BM25 index from all chunks in a Chroma collection.
    Returns a ready-to-search BM25Retriever instance.
    """
    data = collection.get()
    bm25 = BM25Retriever()
    if data["documents"]:
        bm25.index(
            documents=data["documents"],
            metadatas=data["metadatas"],
        )
    return bm25
