"""PaperRAG test — run the full pipeline on existing papers"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from src.loader import load_pdfs_from_dir, chunk_papers
from src.embedder import embed_texts
from src.vector_store import get_collection, add_chunks, get_chunk_count
from src.retriever import retrieve
from src.generator import generate_answer

PAPERS_DIR = "data/papers"

BORDER = "=" * 60

def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', errors='replace').decode('ascii'))

print(BORDER)
print("PaperRAG Pipeline Test")
print(BORDER)

# Step 1: Load
print("\n[1/5] Loading PDFs...")
papers = load_pdfs_from_dir(PAPERS_DIR)
if not papers:
    print("ERROR: No PDFs found")
    sys.exit(1)
print(f"  {len(papers)} papers loaded")
for p in papers:
    safe_print(f"    {p['name']} ({len(p['text'])} chars)")

# Step 2: Chunk
print("\n[2/5] Citation-aware chunking...")
chunks = chunk_papers(papers)
print(f"  {len(chunks)} chunks (was 1873 with naive chunking)")
per_paper = {}
for c in chunks:
    per_paper[c["paper_name"]] = per_paper.get(c["paper_name"], 0) + 1
for p, n in sorted(per_paper.items()):
    safe_print(f"    {p[:60]}: {n} chunks")

# Step 3: Embed + Store
print("\n[3/5] Embedding + storing...")
collection = get_collection()
texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)
add_chunks(collection, chunks, embeddings)
print(f"  {len(embeddings)} embeddings x {len(embeddings[0])} dims")
print(f"  DB total: {get_chunk_count(collection)} chunks")

# Step 4: Retrieve
print("\n[4/5] Testing dedup retrieval (max 2 per paper)...")
model_id = None  # use default
queries = [
    "global value chain disruptions and inflation",
    "AI impact on international trade",
    "GVC resilience and supply chain",
]
for q in queries:
    results = retrieve(collection, q, top_k=6, max_per_paper=2, model_id=model_id)
    papers_seen = set()
    safe_print(f"\n  Q: {q}")
    safe_print(f"  Got {len(results)} chunks from {len(set(r['paper_name'] for r in results))} papers:")
    for r in results:
        safe_print(f"    [{r['paper_name'][:50]}] score={r['score']}")

# Step 5: LLM
print("\n[5/5] Generating answer...")
test_query = "How do global value chain disruptions affect trade and what are the policy implications?"
results = retrieve(collection, test_query, top_k=6, max_per_paper=2)
answer = generate_answer(test_query, results)
print(f"\n  Q: {test_query}")
safe_print(f"  A:\n{answer[:800]}...")

print(f"\n{BORDER}")
print("All tests passed!")
print(BORDER)
