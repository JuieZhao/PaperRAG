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

## Tech choices we debated

| Choice | Picked | Rejected | Why |
|---|---|---|---|
| UI | Streamlit | Gradio, Chainlit | Fastest to prototype, most Pythonic |
| LLM | DeepSeek API | Ollama local | API is simpler to start; local mode is a future feature |
| PDF lib | PyMuPDF | pdfplumber, Unstructured | Fastest, best text extraction |
| Chunking | Custom section-split | LangChain text splitter | Preserves paper structure |
| DB | Chroma | FAISS, Milvus | Simplest persistent local store |

## Known issues
- bge-small-zh has low recall on English papers (scores 0.08-0.40). Migrate to bge-base-en when available.
- Chroma version upgrades break old databases. Need migration path.
- No table/figure extraction from PDFs yet.

## Project structure convention
- `src/` = core pipeline modules (independent, importable)
- `main.py` = Streamlit UI only
- `test_pipeline.py` = end-to-end test
- `data/papers/` = user's PDFs
- `chroma_db/` = vector store (gitignored)
