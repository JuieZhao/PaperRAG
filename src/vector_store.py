"""
Chroma vector store with extended metadata support
(author, year, file_hash, table_count).
"""
import os
import chromadb
from chromadb.config import Settings


def get_collection(persist_dir: str = "./chroma_db", collection_name: str = "minirag_docs"):
    """Get or create Chroma collection."""
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=collection_name)


def add_chunks(
    collection,
    chunks: list[dict],
    embeddings: list[list[float]],
    source_meta: dict | None = None,
    source_meta_map: dict[str, dict] | None = None,
):
    """
    Add chunks + embeddings to Chroma with extended metadata.

    chunks: [{'source_name': ..., 'chunk_id': ..., 'text': ..., 'page': ...,
              'is_table': bool (optional)}]
    source_meta: optional fallback metadata dict:
                {'file_hash': ..., 'author': ..., 'year': ..., 'title': ...}
    source_meta_map: optional mapping keyed by source_name for multi-file batches.
    """
    ids = [f"{c['source_name']}_chunk_{c['chunk_id']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {
            "source_name": c["source_name"],
            "chunk_id": c["chunk_id"],
            "page": c.get("page", "?"),
            "is_table": c.get("is_table", False),
        }
        # Merge source-level metadata into each chunk
        chunk_meta = (source_meta_map or {}).get(c["source_name"], source_meta or {})
        if chunk_meta:
            meta["author"] = chunk_meta.get("author", "")
            meta["year"] = chunk_meta.get("year", "")
            meta["file_hash"] = chunk_meta.get("file_hash", "")
            meta["source_title"] = chunk_meta.get("title", "")
        metadatas.append(meta)

    # Batch insert
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def get_chunk_count(collection) -> int:
    """Return total indexed chunk count."""
    return collection.count()


def get_source_names(collection) -> list[str]:
    """Return sorted list of all indexed source names (deduplicated)."""
    all_data = collection.get()
    sources_set = set()
    if all_data["metadatas"]:
        for m in all_data["metadatas"]:
            sources_set.add(m.get("source_name", "?"))
    return sorted(sources_set)


def get_source_meta_map(collection) -> dict[str, dict]:
    """
    Return {source_name: {author, year, file_hash, source_title}} for all sources.
    """
    all_data = collection.get()
    meta_map: dict[str, dict] = {}
    if all_data["metadatas"]:
        for m in all_data["metadatas"]:
            name = m.get("source_name", "?")
            if name not in meta_map:
                meta_map[name] = {
                    "author": m.get("author", ""),
                    "year": m.get("year", ""),
                    "file_hash": m.get("file_hash", ""),
                    "source_title": m.get("source_title", ""),
                }
    return meta_map


def get_file_hashes(collection) -> set[str]:
    """Return set of all indexed file SHA256 hashes."""
    all_data = collection.get()
    hashes = set()
    if all_data["metadatas"]:
        for m in all_data["metadatas"]:
            h = m.get("file_hash", "")
            if h:
                hashes.add(h)
    return hashes


def delete_source(collection, source_name: str) -> int:
    """Delete all chunks for a source. Returns count of deleted chunks."""
    result = collection.get(where={"source_name": source_name})
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def get_filter_options(collection) -> dict:
    """
    Return available filter values for UI dropdowns.
    {'authors': [...], 'years': [...], 'source_names': [...]}
    """
    all_data = collection.get()
    authors = set()
    years = set()
    source_names = set()
    if all_data["metadatas"]:
        for m in all_data["metadatas"]:
            if m.get("author"):
                authors.add(m["author"])
            if m.get("year"):
                years.add(m["year"])
            source_names.add(m.get("source_name", "?"))
    return {
        "authors": sorted(authors),
        "years": sorted(years, reverse=True),
        "source_names": sorted(source_names),
    }


# ── Backward-compatible aliases ──
get_paper_names = get_source_names
get_paper_meta_map = get_source_meta_map
delete_paper = delete_source
