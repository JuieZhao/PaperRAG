"""
Text embedding with model selection, GPU detection, and batch processing.

Optional dependency: sentence-transformers.
Install with: pip install -r requirements-optional.txt
"""
import os

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None  # type: ignore


# Registry — ranked by quality/cost
MODELS = {
    "chinese": "BAAI/bge-small-zh-v1.5",        # Chinese, 512 dims, 95MB
    "english": "BAAI/bge-base-en-v1.5",           # English, 768 dims, 440MB
    "fast": "all-MiniLM-L6-v2",                    # English, 384 dims, 80MB
    "multilingual": "BAAI/bge-m3",                 # 100+ languages, 1024 dims, 2.2GB
}

_model = None
_model_id = None
_device = None


def _detect_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_device() -> str:
    """Return the device being used for embeddings. Lazily detected on first call."""
    global _device
    if _device is None:
        _device = os.getenv("EMBEDDING_DEVICE") or _detect_device()
    return _device


def _default_model() -> str:
    """Resolve default model: env var → registry key → bare model id."""
    env_val = os.getenv("EMBEDDING_MODEL", "chinese")
    return MODELS.get(env_val, env_val)


def get_embedder(model_id: str | None = None):
    """Lazy-load embedding model (global singleton, auto device)."""
    global _model, _model_id
    if not _HAS_SENTENCE_TRANSFORMERS:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install with: pip install -r requirements-optional.txt\n"
            "Or use RETRIEVAL_MODE=bm25 for zero-model retrieval."
        )
    mid = model_id or _default_model()
    device = get_device()
    if _model is None or _model_id != mid:
        print(f"  Loading embedding model: {mid} on {device} ...")
        _model = SentenceTransformer(mid, device=device)
        _model_id = mid
    return _model


def embed_texts(
    texts: list[str],
    model_id: str | None = None,
    batch_size: int = 32,
) -> list[list[float]]:
    """
    Batch text → vectors with configurable batch size.
    Larger batch_size = faster on GPU, but more memory.
    CPU: 8-16 recommended. GPU: 32-128 depending on VRAM.
    """
    model = get_embedder(model_id)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    return embeddings.tolist()


def embed_query(query: str, model_id: str | None = None) -> list[float]:
    """Single query → vector."""
    return embed_texts([query], model_id, batch_size=1)[0]


def get_model_info(model_id: str | None = None) -> dict:
    """Return model metadata including device info."""
    mid = model_id or _default_model()
    model = get_embedder(mid)
    return {
        "model_id": mid,
        "dimensions": model.get_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "device": get_device(),
    }
