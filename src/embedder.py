"""
Text embedding with model selection and caching.
"""
from sentence_transformers import SentenceTransformer

# Registry — ranked by quality/cost
MODELS = {
    "chinese": "BAAI/bge-small-zh-v1.5",        # Chinese, 512 dims, 95MB — cached locally
    "english": "BAAI/bge-base-en-v1.5",           # English, 768 dims, 440MB
    "fast": "all-MiniLM-L6-v2",                    # English, 384 dims, 80MB
    "multilingual": "BAAI/bge-m3",                 # 100+ languages, 1024 dims, 2.2GB
}

_model = None
_model_id = None


def get_embedder(model_id: str | None = None):
    """Lazy-load embedding model (global singleton)."""
    global _model, _model_id
    mid = model_id or MODELS["chinese"]
    if _model is None or _model_id != mid:
        print(f"  Loading embedding model: {mid} ...")
        _model = SentenceTransformer(mid)
        _model_id = mid
    return _model


def embed_texts(texts: list[str], model_id: str | None = None) -> list[list[float]]:
    """Batch text -> vectors."""
    model = get_embedder(model_id)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str, model_id: str | None = None) -> list[float]:
    """Single query -> vector."""
    return embed_texts([query], model_id)[0]


def get_model_info(model_id: str | None = None) -> dict:
    """Return model metadata."""
    mid = model_id or MODELS["chinese"]
    model = get_embedder(mid)
    return {
        "model_id": mid,
        "dimensions": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
    }
