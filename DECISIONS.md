# PaperRAG Architecture Decisions

> Grill-with-docs: read this before writing code for PaperRAG.

## What we're building
A local-first RAG system for academic papers. Users upload PDFs, ask questions, get answers with citations.

## Core design decisions

1. **Chroma for vector store** (not Pinecone/Weaviate)
   - Reason: local-only, zero config, Python-native. If data >10k chunks, consider Milvus Lite.

2. **Citation-aware chunking** (not fixed-size character split)
   - Reason: academic papers have clear section structure. Splitting on headers preserves semantic coherence.

3. **Dedup retrieval** (max 2 chunks per paper)
   - Reason: prevents large PDFs from dominating results. Our 5MB WTO report swamped everything before this fix.

4. **Side-by-side UI** (answer + sources panel)
   - Reason: every answer must be verifiable. Generic RAG tools hide sources; we show them.

5. **BGE-small-zh embedding** (default, fallback to en when available)
   - Reason: our papers are English but we want Chinese Q&A support. Tradeoff: lower retrieval scores on pure English.
   - **May 2026 update:** Recommend `EMBEDDING_MODEL=english` for English-heavy libraries. GPU auto-detect added.

6. **Hybrid BM25 + Dense retrieval (NEW)**
   - Reason: pure dense retrieval misses technical terms and acronyms (GVC, LMDI, WTO).
   - Implementation: custom BM25 (numpy-only, no sklearn dep) + RRF fusion with dense results.
   - BM25 index is lazily rebuilt when collection changes.

7. **Content-based dedup (NEW)**
   - Reason: filename-based dedup was too weak — same PDF under different name would be indexed twice.
   - Implementation: SHA256 file hash stored in Chroma metadata, checked before indexing.

8. **Table extraction (NEW)**
   - Reason: economic papers contain critical data in tables. Previous pipeline lost all tabular info.
   - Implementation: PyMuPDF find_tables() → Markdown conversion → indexed as regular chunks (with is_table flag).

9. **Metadata filtering (NEW)**
   - Reason: users want to scope searches to specific authors, years, or papers.
   - Implementation: author/year/file_hash/paper_title stored in Chroma metadata, passed as `where` clause.

## Tech choices we debated

| Choice | Picked | Rejected | Why |
|---|---|---|---|
| UI | Streamlit + Gradio | Chainlit, FastAPI | Streamlit for prototyping, Gradio for chat-native UX |
| LLM | DeepSeek API | Ollama local | API is simpler to start; local mode is a future feature |
| PDF lib | PyMuPDF | pdfplumber, Unstructured | Fastest, best text + table extraction |
| Chunking | Custom section-split | LangChain text splitter | Preserves paper structure |
| DB | Chroma | FAISS, Milvus | Simplest persistent local store |
| BM25 | Custom numpy impl | rank_bm25, sklearn | Zero extra deps, fast enough for <10k docs |
| Reranker | ms-marco-MiniLM-L-6-v2 | BGE-reranker | Lightweight, good English performance |

## Known issues
- bge-small-zh has low recall on English papers (scores 0.08-0.40). **→ Set EMBEDDING_MODEL=english.**
- Chroma version upgrades break old databases. Need migration path.
- Table extraction requires PyMuPDF >= 1.23.0 (silently falls back).
- BM25 index rebuilds on collection change (lazy, but could be slow for very large DBs).
- Gradio streaming is less smooth than Streamlit's native streaming support.

## Project structure convention
- `src/` = core pipeline modules (independent, importable)
- `main.py` = Streamlit UI
- `gradio_app.py` = Gradio chat UI
- `test_pipeline.py` = end-to-end test
- `download_papers.py` = arXiv API + RSS + manual downloads
- `data/papers/` = user's PDFs
- `chroma_db/` = vector store (gitignored)
