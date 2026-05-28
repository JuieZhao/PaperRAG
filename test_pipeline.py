"""MiniRAG test — run the full pipeline with hybrid search on existing documents"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from src.loader import load_documents_from_dir, chunk_documents, compute_file_hash, load_document_meta
from src.embedder import embed_texts, get_model_info
from src.vector_store import get_collection, add_chunks, get_chunk_count
from src.retriever import search_documents
from src.generator import generate_answer

DOCUMENTS_DIR = "documents"
BORDER = "=" * 60


def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode("ascii"))


print(BORDER)
print("MiniRAG Pipeline Test (Hybrid Edition)")
print(BORDER)

# Show embedding model info
info = get_model_info()
safe_print(f"\nEmbedding: {info['model_id']} · {info['dimensions']}d · device={info['device']}")

# Step 1: Load
print("\n[1/5] Loading documents...")
papers = load_documents_from_dir(DOCUMENTS_DIR)
if not papers:
    print("ERROR: No supported documents found in documents/")
    print("Add PDF, DOCX, TXT, Markdown, or CSV files first.")
    sys.exit(1)
print(f"  {len(papers)} papers loaded")
for p in papers:
    safe_print(f"    {p['name']} ({len(p['text'])} chars)")

# Step 2: Chunk (with table extraction)
print("\n[2/5] Citation-aware chunking + table extraction...")
chunks = chunk_documents(papers, extract_tables=True)
table_count = sum(1 for c in chunks if c.get("is_table"))
print(f"  {len(chunks)} chunks ({table_count} tables)")
per_source = {}
for c in chunks:
    per_source[c["source_name"]] = per_source.get(c["source_name"], 0) + 1
for p, n in sorted(per_source.items()):
    safe_print(f"    {p[:60]}: {n} chunks")

# Step 3: Embed + Store
print("\n[3/5] Embedding + storing...")
collection = get_collection()
texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)

# Build per-document metadata map
source_metas = {}
for p in papers:
    path = p.get("path", os.path.join(DOCUMENTS_DIR, p["name"]))
    meta = load_document_meta(path)
    meta["file_hash"] = compute_file_hash(path)
    source_metas[p["name"]] = meta

add_chunks(collection, chunks, embeddings, source_meta_map=source_metas)
print(f"  {len(embeddings)} embeddings x {len(embeddings[0])} dims")
print(f"  DB total: {get_chunk_count(collection)} chunks")

# Step 4: Hybrid retrieval
print("\n[4/5] Testing hybrid retrieval (BM25 + Dense + RRF)...")
queries = [
    "global value chain disruptions and inflation",
    "AI impact on international trade",
    "GVC resilience and supply chain",
]
for q in queries:
    results = search_documents(collection, q, top_k=6, hybrid=True)
    papers_seen = set()
    safe_print(f"\n  Q: {q}")
    safe_print(f"  Got {len(results)} chunks from {len(set(r['source_name'] for r in results))} documents:")
    for r in results:
        safe_print(f"    [{r['source_name'][:50]}] score={r['score']}")

# Step 5: LLM
print("\n[5/5] Generating answer...")
test_query = "How do global value chain disruptions affect trade and what are the policy implications?"
results = search_documents(collection, test_query, top_k=6, hybrid=True)
answer = generate_answer(test_query, results)
print(f"\n  Q: {test_query}")
safe_print(f"  A:\n{answer[:800]}...")

print(f"\n{BORDER}")
print("All tests passed!")
print(BORDER)
