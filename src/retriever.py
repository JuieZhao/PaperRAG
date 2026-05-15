"""
Retrieval with paper-level deduplication + optional cross-encoder reranking
"""
import os
from .embedder import embed_query

_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load cross-encoder model for reranking (first call downloads ~120MB)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


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
):
    """
    Retrieve with dedup: max max_per_paper chunks per paper.
    This prevents one large document from dominating results.

    Returns: [{'text': ..., 'paper_name': ..., 'chunk_id': ..., 'score': ...}]
    """
    # Fetch more candidates than needed for dedup headroom
    fetch_k = max(top_k * 3, 20)
    query_vec = embed_query(query, model_id)
    results = collection.query(query_embeddings=[query_vec], n_results=fetch_k)

    retrieved = []
    seen_counts = {}  # paper_name -> count

    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            paper = meta.get("paper_name", "Unknown")

            # Skip if this paper already hit the per-paper limit
            if seen_counts.get(paper, 0) >= max_per_paper:
                continue

            seen_counts[paper] = seen_counts.get(paper, 0) + 1
            retrieved.append({
                "text": doc,
                "paper_name": paper,
                "chunk_id": meta.get("chunk_id", -1),
                "page": meta.get("page", "?"),
                "score": round(1 - dist, 4) if dist else 0,
            })

            if len(retrieved) >= top_k:
                break

    return retrieved
