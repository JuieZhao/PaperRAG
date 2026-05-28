# MiniRAG Architecture Decisions

> Grill-with-docs: read this before writing code for MiniRAG.

## What we're building
A lightweight, general-purpose RAG system for documents. Users upload files (PDFs, DOCX, text, spreadsheets), ask questions, get answers with citations. Zero-model baseline, scaling up only when needed.

## Core philosophy

**Lightweight-first.** We cannot and should not compete with large RAG projects on model-driven features. Every model added is a tax on startup time, memory, and complexity. Prefer zero-model solutions; only add models when there is clear, verifiable benefit.

Model gradient (least → most):
- `RETRIEVAL_MODE=bm25` → **zero models**, pure keyword search, instant startup. No extra deps.
- `RETRIEVAL_MODE=hybrid` → **one model** (embedding), balanced precision/recall. Needs `pip install -r requirements-optional.txt`
- `+ENABLE_RERANK=true` → **two models** (embedding + cross-encoder), highest precision.

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
   - Implementation: PyMuPDF find_tables() for PDFs and python-docx table parsing for DOCX → Markdown conversion → indexed as regular chunks (with is_table flag for extracted PDF tables).

9. **Metadata filtering (NEW)**
   - Reason: users want to scope searches to specific authors, years, or papers.
   - Implementation: author/year/file_hash/paper_title stored in Chroma metadata, passed as `where` clause.

## Tech choices we debated

| Choice | Picked | Rejected | Why |
|---|---|---|---|
| UI | Streamlit | Gradio, Chainlit, FastAPI | Keep one maintained local UI instead of two parallel frontends |
| LLM | DeepSeek API | Ollama local | API is simpler to start; local mode is a future feature |
| Parsing libs | PyMuPDF + python-docx | pdfplumber, Unstructured | Fast local PDF/DOCX parsing without a heavy document pipeline |
| Chunking | Custom section-split | LangChain text splitter | Preserves paper structure |
| DB | Chroma | FAISS, Milvus | Simplest persistent local store |
| BM25 | Custom numpy impl | rank_bm25, sklearn | Zero extra deps, fast enough for <10k docs |
| Reranker | ms-marco-MiniLM-L-6-v2 (opt-in) | BGE-reranker | Set ENABLE_RERANK=true; stays off by default for lightweight startup |

## Known issues
- bge-small-zh has low recall on English papers (scores 0.08-0.40). **→ Set EMBEDDING_MODEL=english.**
- **Chroma collection renamed from `paperrag_docs` → `minirag_docs`.** Old databases need re-indexing (or rename the collection manually).
- **Metadata key renamed from `paper_name` → `source_name`, `paper_title` → `source_title`.** Existing Chroma DBs will have old keys; re-index after upgrading.
- Chroma version upgrades break old databases. Need migration path.
- Table extraction requires PyMuPDF >= 1.23.0 (silently falls back).
- BM25 index rebuilds on collection change (lazy, but could be slow for very large DBs).
- DOCX page numbers are not available from python-docx, so citations show page `?`.

## Project structure convention
- `src/` = core pipeline modules (independent, importable)
- `main.py` = Streamlit UI
- `test_pipeline.py` = end-to-end test
- `documents/` = user's files (PDF, DOCX, TXT, Markdown, CSV)
- `data/` = local app data such as Q&A history
- `chroma_db/` = vector store (gitignored)
