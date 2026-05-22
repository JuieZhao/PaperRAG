"""
Retrieval with source-level deduplication, optional BM25 hybrid search,
cross-encoder reranking, and metadata filtering.

Lightweight by default — reranker and embedding model are both opt-in.
Set RETRIEVAL_MODE=bm25 for zero-model (BM25-only) retrieval.
Set ENABLE_RERANK=true to activate cross-encoder reranking.
"""
import os
import numpy as np
from .embedder import embed_query
from .bm25_retriever import BM25Retriever, rrf_fuse, build_bm25_from_collection

_cross_encoder = None
_bm25_index: BM25Retriever | None = None
_bm25_collection_hash: int | None = None  # track when to rebuild


def _get_retrieval_mode() -> str:
    """RETRIEVAL_MODE: 'hybrid' (default), 'bm25', or 'dense'."""
    mode = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
    if mode not in ("hybrid", "bm25", "dense"):
        return "hybrid"
    return mode


def _get_cross_encoder():
    """Lazy-load cross-encoder model for reranking (first call downloads ~120MB)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install -r requirements-optional.txt"
            )
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
    max_per_source: int = 2,
    hybrid: bool = True,
    bm25_weight: float = 0.5,
    filters: dict | None = None,
    **kwargs,
):
    """
    Retrieve with dedup + optional BM25 hybrid search + metadata filtering.

    Parameters:
        max_per_source: max chunks returned per source document (default: 2)
        hybrid: if True, fuse dense + BM25 via RRF (default: True)
                Ignored when RETRIEVAL_MODE env var is set.
        bm25_weight: relative weight of BM25 in RRF (0-1, for future use)
        filters: Chroma where clause dict for metadata filtering
                 e.g. {"author": "Smith"} or {"year": {"$gte": "2020"}}

    Returns: [{'text': ..., 'source_name': ..., 'chunk_id': ..., 'score': ..., 'page': ...}]
    """
    # Backward compat: accept old max_per_paper kwarg
    if "max_per_paper" in kwargs:
        max_per_source = kwargs.pop("max_per_paper")
    mode = _get_retrieval_mode()
    fetch_k = max(top_k * 3, 30)

    # ── BM25-only mode (zero embedding model) ──
    if mode == "bm25":
        return _bm25_retrieve(collection, query, fetch_k, max_per_source, filters)

    # ── Dense-only mode ──
    if mode == "dense":
        return _dense_retrieve(collection, query, model_id, fetch_k, max_per_source, filters)

    # ── Hybrid mode (default) ──
    if not hybrid:
        return _dense_retrieve(collection, query, model_id, fetch_k, max_per_source, filters)

    dense_hits = _dense_retrieve(collection, query, model_id, fetch_k, max_per_source, filters)
    bm25_hits = _bm25_retrieve(collection, query, fetch_k, max_per_source, filters)

    # ── RRF fusion ──
    merged = rrf_fuse(dense_hits, bm25_hits, top_k=fetch_k)

    # Re-apply per-source limit post-fusion
    final = []
    seen_final: dict[str, int] = {}
    for r in merged:
        src = r["source_name"]
        if seen_final.get(src, 0) >= max_per_source:
            continue
        seen_final[src] = seen_final.get(src, 0) + 1
        final.append(r)
        if len(final) >= top_k * 3:
            break

    return final


def _dense_retrieve(
    collection,
    query: str,
    model_id: str | None,
    fetch_k: int,
    max_per_source: int,
    filters: dict | None,
) -> list[dict]:
    """Dense (embedding) retrieval. Uses sentence-transformers model."""
    query_vec = embed_query(query, model_id)
    chroma_kwargs = {"query_embeddings": [query_vec], "n_results": fetch_k}
    if filters:
        chroma_kwargs["where"] = filters
    results = collection.query(**chroma_kwargs)

    retrieved = []
    seen_counts: dict[str, int] = {}

    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            src = meta.get("source_name", "Unknown")
            if seen_counts.get(src, 0) >= max_per_source:
                continue
            seen_counts[src] = seen_counts.get(src, 0) + 1
            retrieved.append({
                "text": doc,
                "source_name": src,
                "chunk_id": meta.get("chunk_id", -1),
                "page": meta.get("page", "?"),
                "score": round(1 - dist, 4) if dist else 0,
            })

    return retrieved


def _bm25_retrieve(
    collection,
    query: str,
    fetch_k: int,
    max_per_source: int,
    filters: dict | None,
) -> list[dict]:
    """BM25 keyword retrieval. Zero models — pure numpy."""
    bm25 = _get_bm25(collection)
    bm25_hits = bm25.search(query, top_k=fetch_k)
    data = collection.get()

    retrieved = []
    seen_counts: dict[str, int] = {}

    for idx, score in bm25_hits:
        src = "Unknown"
        if data["metadatas"] and idx < len(data["metadatas"]):
            meta = data["metadatas"][idx]
            src = meta.get("source_name", "Unknown")
            if filters:
                mismatch = False
                for key, val in filters.items():
                    if isinstance(val, dict):
                        if "$gte" in val and str(meta.get(key, "")) < str(val["$gte"]):
                            mismatch = True
                    elif meta.get(key, "") != val:
                        mismatch = True
                if mismatch:
                    continue
        if seen_counts.get(src, 0) >= max_per_source:
            continue
        seen_counts[src] = seen_counts.get(src, 0) + 1
        retrieved.append({
            "text": data["documents"][idx] if data["documents"] and idx < len(data["documents"]) else "",
            "source_name": src,
            "chunk_id": data["metadatas"][idx].get("chunk_id", idx) if data["metadatas"] and idx < len(data["metadatas"]) else idx,
            "page": data["metadatas"][idx].get("page", "?") if data["metadatas"] and idx < len(data["metadatas"]) else "?",
            "score": round(float(score), 4),
        })

    return retrieved


def search_documents(
    collection,
    query: str,
    model_id: str | None = None,
    top_k: int = 5,
    hybrid: bool = True,
    filters: dict | None = None,
) -> list[dict]:
    """
    Higher-level search: retrieve → [rerank] → return top_k.

    Reranking is opt-in — set ENABLE_RERANK=true to activate.
    Without it, returns raw retrieval results trimmed to top_k.
    """
    raw = retrieve(
        collection, query,
        model_id=model_id,
        top_k=top_k,
        max_per_source=2,
        hybrid=hybrid,
        filters=filters,
    )

    # Reranking is opt-in to keep the default stack lightweight
    if os.getenv("ENABLE_RERANK", "").lower() in ("true", "1", "yes"):
        return rerank(query, raw, top_k=top_k)

    # Without reranker: just return top_k by retrieval score
    return raw[:top_k]


# ── Backward-compatible alias ──
search_papers = search_documents
