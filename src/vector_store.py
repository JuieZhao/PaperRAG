"""
Chroma vector store with extended metadata support
(author, year, file_hash, table_count).
"""
import os
import chromadb
from chromadb.config import Settings


def get_collection(persist_dir: str = "./chroma_db", collection_name: str = "paperrag_docs"):
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
    paper_meta: dict | None = None,
):
    """
    Add chunks + embeddings to Chroma with extended metadata.

    chunks: [{'paper_name': ..., 'chunk_id': ..., 'text': ..., 'page': ...,
              'is_table': bool (optional)}]
    paper_meta: optional dict with per-paper metadata:
                {'file_hash': ..., 'author': ..., 'year': ..., 'title': ...}
    """
    ids = [f"{c['paper_name']}_chunk_{c['chunk_id']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {
            "paper_name": c["paper_name"],
            "chunk_id": c["chunk_id"],
            "page": c.get("page", "?"),
            "is_table": c.get("is_table", False),
        }
        # Merge paper-level metadata into each chunk
        if paper_meta:
            meta["author"] = paper_meta.get("author", "")
            meta["year"] = paper_meta.get("year", "")
            meta["file_hash"] = paper_meta.get("file_hash", "")
            meta["paper_title"] = paper_meta.get("title", "")
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


def get_paper_names(collection) -> list[str]:
    """Return sorted list of all indexed paper names (deduplicated)."""
    all_data = collection.get()
    papers_set = set()
    if all_data["metadatas"]:
        for m in all_data["metadatas"]:
            papers_set.add(m.get("paper_name", "?"))
    return sorted(papers_set)


def get_paper_meta_map(collection) -> dict[str, dict]:
    """
    Return {paper_name: {author, year, file_hash, paper_title}} for all papers.
    """
    all_data = collection.get()
    meta_map: dict[str, dict] = {}
    if all_data["metadatas"]:
        for m in all_data["metadatas"]:
            name = m.get("paper_name", "?")
            if name not in meta_map:
                meta_map[name] = {
                    "author": m.get("author", ""),
                    "year": m.get("year", ""),
                    "file_hash": m.get("file_hash", ""),
                    "paper_title": m.get("paper_title", ""),
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


def delete_paper(collection, paper_name: str) -> int:
    """Delete all chunks for a paper. Returns count of deleted chunks."""
    result = collection.get(where={"paper_name": paper_name})
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def get_filter_options(collection) -> dict:
    """
    Return available filter values for UI dropdowns.
    {'authors': [...], 'years': [...], 'paper_names': [...]}
    """
    all_data = collection.get()
    authors = set()
    years = set()
    paper_names = set()
    if all_data["metadatas"]:
        for m in all_data["metadatas"]:
            if m.get("author"):
                authors.add(m["author"])
            if m.get("year"):
                years.add(m["year"])
            paper_names.add(m.get("paper_name", "?"))
    return {
        "authors": sorted(authors),
        "years": sorted(years, reverse=True),
        "paper_names": sorted(paper_names),
    }
