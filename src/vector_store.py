"""
Chroma 向量库操作
"""
import os
import chromadb
from chromadb.config import Settings


def get_collection(persist_dir: str = "./chroma_db", collection_name: str = "paperrag_docs"):
    """获取或创建 Chroma collection"""
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=collection_name)


def add_chunks(collection, chunks: list[dict], embeddings: list[list[float]]):
    """
    将文本块 + 向量写入 Chroma
    chunks: [{'paper_name': ..., 'chunk_id': ..., 'text': ...}]
    """
    ids = [f"{c['paper_name']}_chunk_{c['chunk_id']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"paper_name": c["paper_name"], "chunk_id": c["chunk_id"]} for c in chunks]

    # 分批添加（避免一次太大）
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )


def get_chunk_count(collection) -> int:
    """返回已索引的文本块数"""
    return collection.count()


def get_paper_names(collection) -> list[str]:
    """返回所有已索引的论文名（去重）"""
    all_data = collection.get()
    papers_set = set()
    if all_data["metadatas"]:
        for m in all_data["metadatas"]:
            papers_set.add(m.get("paper_name", "?"))
    return sorted(papers_set)


def delete_paper(collection, paper_name: str) -> int:
    """
    删除指定论文的所有 chunks。
    返回删除的数量。
    """
    # Chroma 按 metadata 过滤获取 id，然后批量删除
    result = collection.get(where={"paper_name": paper_name})
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)
