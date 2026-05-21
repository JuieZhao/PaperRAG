"""
Retrieval with paper-level deduplication, optional BM25 hybrid search,
cross-encoder reranking, and metadata filtering.
"""
import os
import numpy as np
from .embedder import embed_query
from .bm25_retriever import BM25Retriever, rrf_fuse, build_bm25_from_collection

_cross_encoder = None
_bm25_index: BM25Retriever | None = None
_bm25_collection_hash: int | None = None  # track when to rebuild


def _get_cross_encoder():
    """Lazy-load cross-encoder model for reranking (first call downloads ~120MB)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _get_bm25(collection, force_rebuild: bool = False) -> BM25Retriever:
    """Lazy-load BM25 index, rebuilding if collection changed."""
    global _bm25_index, _bm25_collection_hash
    current_hash = hash(str(collection.get()["ids"]) if collection.count() > 0 else "")
    if _bm25_index is None or force_rebuild or _bm25_collection_hash != current_hash:
        _bm25_index = build_bm25_from_collection(collection)
        _bm25_collection_hash = current_hash
    return _bm25_index


def set_bm25_param(k1: float = 1.5, b: float = 0.75):
    """Configure BM25 parameters globally. Call before retrieval."""
    global _bm25_index
    _bm25_index = BM25Retriever(k1=k1, b=b)


def rerank(query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    """
    Re-rank candidates with a cross-encoder for better precision.
    Returns candidates sorted by cross-encoder score (descending).
    """
    if len(candidates) <= 1:
        return candidates

    model = _get_cross_encoder()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    for c, score in zip(candidates, scores):
        c["score"] = round(float(score), 4)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if top_k:
        candidates = candidates[:top_k]

    return candidates


def retrieve(
    collection,
    query: str,
    model_id: str | None = None,
    top_k: int = 5,
    max_per_paper: int = 2,
    hybrid: bool = True,
    bm25_weight: float = 0.5,
    filters: dict | None = None,
):
    """
    Retrieve with dedup + optional BM25 hybrid search + metadata filtering.

    Parameters:
        hybrid: if True, fuse dense + BM25 via RRF (default: True)
        bm25_weight: relative weight of BM25 in RRF (0-1, for future use)
        filters: Chroma where clause dict for metadata filtering
                 e.g. {"author": "Smith"} or {"year": {"$gte": "2020"}}

    Returns: [{'text': ..., 'paper_name': ..., 'chunk_id': ..., 'score': ..., 'page': ...}]
    """
    fetch_k = max(top_k * 3, 30)

    # ── Dense retrieval ──
    query_vec = embed_query(query, model_id)
    chroma_kwargs = {"query_embeddings": [query_vec], "n_results": fetch_k}
    if filters:
        chroma_kwargs["where"] = filters
    results = collection.query(**chroma_kwargs)

    dense_retrieved = []
    seen_counts: dict[str, int] = {}

    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            paper = meta.get("paper_name", "Unknown")
            if seen_counts.get(paper, 0) >= max_per_paper:
                continue
            seen_counts[paper] = seen_counts.get(paper, 0) + 1
            dense_retrieved.append({
                "text": doc,
                "paper_name": paper,
                "chunk_id": meta.get("chunk_id", -1),
                "page": meta.get("page", "?"),
                "score": round(1 - dist, 4) if dist else 0,
            })

    # ── BM25 retrieval (hybrid mode) ──
    if hybrid:
        bm25 = _get_bm25(collection)
        bm25_hits = bm25.search(query, top_k=fetch_k)
        data = collection.get()
        bm25_retrieved = []
        seen_bm25: dict[str, int] = {}

        for idx, score in bm25_hits:
            paper = "Unknown"
            if data["metadatas"] and idx < len(data["metadatas"]):
                meta = data["metadatas"][idx]
                paper = meta.get("paper_name", "Unknown")
                if filters:
                    # Apply same filters to BM25 results
                    mismatch = False
                    for key, val in filters.items():
                        if isinstance(val, dict):
                            if "$gte" in val and str(meta.get(key, "")) < str(val["$gte"]):
                                mismatch = True
                        elif meta.get(key, "") != val:
                            mismatch = True
                    if mismatch:
                        continue
            if seen_bm25.get(paper, 0) >= max_per_paper:
                continue
            seen_bm25[paper] = seen_bm25.get(paper, 0) + 1
            bm25_retrieved.append({
                "text": data["documents"][idx] if data["documents"] and idx < len(data["documents"]) else "",
                "paper_name": paper,
                "chunk_id": data["metadatas"][idx].get("chunk_id", idx) if data["metadatas"] and idx < len(data["metadatas"]) else idx,
                "page": data["metadatas"][idx].get("page", "?") if data["metadatas"] and idx < len(data["metadatas"]) else "?",
                "score": round(float(score), 4),
            })

        # ── RRF fusion ──
        retrieved = rrf_fuse(dense_retrieved, bm25_retrieved, top_k=fetch_k)
    else:
        retrieved = dense_retrieved

    # Re-apply per-paper limit post-fusion
    final = []
    seen_final: dict[str, int] = {}
    for r in retrieved:
        paper = r["paper_name"]
        if seen_final.get(paper, 0) >= max_per_paper:
            continue
        seen_final[paper] = seen_final.get(paper, 0) + 1
        final.append(r)
        if len(final) >= top_k * 3:
            break

    return final


def search_papers(
    collection,
    query: str,
    model_id: str | None = None,
    top_k: int = 5,
    hybrid: bool = True,
    filters: dict | None = None,
) -> list[dict]:
    """
    Higher-level search: retrieve → rerank → return top_k.
    This is the recommended entry point for most use cases.
    """
    raw = retrieve(
        collection, query,
        model_id=model_id,
        top_k=top_k,
        max_per_paper=2,
        hybrid=hybrid,
        filters=filters,
    )
    return rerank(query, raw, top_k=top_k)
