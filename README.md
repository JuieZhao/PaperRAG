# PaperRAG — Ask Your Papers

> 📄 [中文文档](README.zh-CN.md)

Upload PDF papers, ask questions in natural language, get cited answers backed by your own literature.

## Why PaperRAG?

- **Your papers, your knowledge** — answers are grounded in *your* uploaded PDFs, not web search
- **Every answer is traceable** — citations link to specific papers and text chunks
- **You control the knowledge base** — add/remove PDFs to curate what the AI knows
- **Hybrid search (NEW)** — BM25 keyword + dense vector retrieval fused via RRF for better recall
- **Content dedup (NEW)** — SHA256 hashing prevents duplicate indexing
- **Table extraction (NEW)** — PDF tables extracted and searchable as Markdown
- **Metadata filters (NEW)** — filter by author, year, or specific paper

## Features

- 📤 Upload PDFs → auto-parse, chunk, embed, index
- 🔀 Hybrid retrieval: BM25 + Dense vector → RRF fusion
- 🔍 Natural language QA → retrieve relevant chunks → LLM generates cited answers
- 🌐 Bilingual search (Chinese + English)
- 🎯 Cross-encoder reranking for better retrieval precision
- ⚡ Streaming responses (answers appear token by token)
- 🗑️ Batch paper management (checkbox selection + bulk delete)
- 📋 Export answers as Markdown with source citations
- 🏷️ Metadata filtering by author / year / paper
- 📊 PDF table extraction (converted to Markdown)
- 🔤 Auto GPU detection for embeddings

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose your UI

**Streamlit (original):**
```bash
streamlit run main.py
```

**Gradio (NEW — chat-native):**
```bash
python gradio_app.py
```

Open the URL, paste your DeepSeek API key in the sidebar, upload your PDFs, and start asking questions.

> 💡 Get a free API key at [platform.deepseek.com](https://platform.deepseek.com).
> The key stays in your browser session — never saved to disk.
>
> Create a `.env` file (see `.env.example`) for persistent config.

### 3. (Optional) Download sample papers

```bash
# Manual URL list
python download_papers.py manual

# Search arXiv by keyword
python download_papers.py arxiv-search "global value chains" -c econ.GN -n 5

# Fetch latest from arXiv RSS feed
python download_papers.py arxiv-rss -c cs.AI -n 10

# Dry-run (search only, no download)
python download_papers.py arxiv-search "supply chain resilience" --dry-run
```

## Project Structure

```
PaperRAG/
├── main.py                  # Streamlit UI
├── gradio_app.py            # Gradio chat UI (NEW)
├── requirements.txt
├── .env.example
├── download_papers.py       # arXiv API + RSS + manual downloads
├── test_pipeline.py         # End-to-end pipeline test
├── src/
│   ├── loader.py            # PDF parsing + table extraction + chunking
│   ├── embedder.py          # Text embedding (GPU auto-detect)
│   ├── vector_store.py      # Chroma vector DB + metadata
│   ├── retriever.py         # Hybrid retrieval (BM25 + Dense + RRF)
│   ├── bm25_retriever.py    # BM25 keyword search engine (NEW)
│   └── generator.py         # LLM answer generation (streaming)
├── data/papers/             # Your PDF files (git-ignored)
└── chroma_db/               # Vector store persistence (git-ignored)
```

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| UI | Streamlit / Gradio | Two frontends included |
| PDF Parsing | PyMuPDF (fitz) | Citation-aware + table extraction |
| Embedding | BGE (multiple models) | GPU auto-detect, local inference |
| Keyword Search | BM25 (custom impl) | No external deps beyond numpy |
| Vector DB | Chroma | Persistent, zero-config |
| LLM | DeepSeek API | OpenAI-compatible SDK |
| Reranking | ms-marco-MiniLM-L-6-v2 | Cross-encoder for precision |
| Fusion | RRF | Reciprocal Rank Fusion |
| Paper Source | arXiv API + RSS | Automated paper ingestion |

## Configuration

Create a `.env` file:

```bash
DEEPSEEK_API_KEY=sk-xxx
EMBEDDING_MODEL=english     # Recommended for English papers
# EMBEDDING_DEVICE=cuda     # Auto-detected by default
# HTTP_PROXY=http://127.0.0.1:7890  # Behind firewall
```

## License

MIT
