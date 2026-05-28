# MiniRAG — Lightweight Document RAG

> 📄 [中文文档](README.zh-CN.md)

Upload documents, ask questions in natural language, get cited answers grounded in your own files. Minimal models, maximum control.

## Why MiniRAG?

- **Lightweight by design** — zero GPU required, single embedding model, no Docker, no heavy dependencies
- **Your documents, your knowledge** — answers are grounded in *your* uploaded files, not web search
- **Every answer is traceable** — citations link to specific documents and text chunks
- **You control the knowledge base** — add/remove files to curate what the AI knows
- **Hybrid search** — BM25 keyword + dense vector retrieval fused via RRF for better recall
- **Multi-format support** — PDF, DOCX, TXT, Markdown, CSV
- **Content dedup** — SHA256 hashing prevents duplicate indexing
- **Flexible chunking** — section-aware for reports, fixed-size for general docs
- **Metadata filters** — filter by source, author, year

## Features

- 📤 Upload documents → auto-parse, chunk, embed, index
- 🔀 Hybrid retrieval: BM25 + Dense vector → RRF fusion
- 🔍 Natural language QA → retrieve relevant chunks → LLM generates cited answers
- 🌐 Bilingual search (Chinese + English)
- 🎯 Cross-encoder reranking for better retrieval precision (optional)
- ⚡ Streaming responses (answers appear token by token)
- 🗑️ Batch document management (checkbox selection + bulk delete)
- 📋 Export answers as Markdown with source citations
- 🏷️ Metadata filtering by author / year / source
- 📊 PDF table extraction (converted to Markdown)
- 🔤 Auto GPU detection for embeddings

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run main.py
```

Open the URL, paste your DeepSeek API key in the sidebar, upload your files, and start asking questions.

You can also put files directly in `documents/` before indexing.

> 💡 Get a free API key at [platform.deepseek.com](https://platform.deepseek.com).
> The key stays in your browser session — never saved to disk.
>
> Create a `.env` file (see `.env.example`) for persistent config.

## Project Structure

```
MiniRAG/
├── main.py                  # Streamlit UI
├── requirements.txt
├── .env.example
├── documents/               # Files to upload/index
├── test_pipeline.py         # End-to-end pipeline test
├── src/
│   ├── loader.py            # Document parsing + table extraction + chunking
│   ├── embedder.py          # Text embedding (GPU auto-detect)
│   ├── vector_store.py      # Chroma vector DB + metadata
│   ├── retriever.py         # Hybrid retrieval (BM25 + Dense + RRF)
│   ├── bm25_retriever.py    # BM25 keyword search engine
│   └── generator.py         # LLM answer generation (streaming)
├── data/                    # Local app data, such as Q&A history
└── chroma_db/               # Vector store persistence (git-ignored)
```

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| UI | Streamlit | Single local frontend |
| Parsing | PyMuPDF + python-docx | PDF table extraction + DOCX paragraphs/tables |
| Embedding | BGE (multiple models) | GPU auto-detect, local inference |
| Keyword Search | BM25 (custom impl) | No external deps beyond numpy |
| Vector DB | Chroma | Persistent, zero-config |
| LLM | DeepSeek API | OpenAI-compatible SDK |
| Reranking | ms-marco-MiniLM-L-6-v2 | Cross-encoder for precision |
| Fusion | RRF | Reciprocal Rank Fusion |

## Configuration

Create a `.env` file:

```bash
DEEPSEEK_API_KEY=sk-xxx
EMBEDDING_MODEL=english     # Recommended for English documents
# EMBEDDING_DEVICE=cuda     # Auto-detected by default
# HTTP_PROXY=http://127.0.0.1:7890  # Behind firewall
```

## License

MIT
