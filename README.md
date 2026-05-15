# PaperRAG — Ask Your Papers

> 📄 [中文文档](README.zh-CN.md)

Upload PDF papers, ask questions in natural language, get cited answers backed by your own literature.

## Why PaperRAG?

- **Your papers, your knowledge** — answers are grounded in *your* uploaded PDFs, not web search
- **Every answer is traceable** — citations link to specific papers and text chunks
- **You control the knowledge base** — add/remove PDFs to curate what the AI knows

## Features

- 📤 Upload PDFs → auto-parse, chunk, embed, index
- 🔍 Natural language QA → retrieve relevant chunks → LLM generates cited answers
- 🌐 Bilingual search (Chinese + English)
- 🎯 Cross-encoder reranking for better retrieval precision
- ⚡ Streaming responses (answers appear token by token)
- 🗑️ Batch paper management (checkbox selection + bulk delete)
- 📋 Export answers as Markdown with source citations

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key

Copy `.env.example` to `.env` and fill in your DeepSeek API key:

```bash
cp .env.example .env
```

`.env`:
```
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 3. Run

```bash
streamlit run main.py
```

Open http://localhost:8501, upload your PDFs, and start asking questions.

## Project Structure

```
PaperRAG/
├── main.py              # Streamlit UI entry point
├── requirements.txt
├── .env.example
├── download_papers.py   # Utility: batch download papers from URLs
├── test_pipeline.py     # End-to-end pipeline test
├── src/
│   ├── loader.py        # PDF parsing + citation-aware chunking
│   ├── embedder.py      # Text embedding (BGE models)
│   ├── vector_store.py  # Chroma vector DB operations
│   ├── retriever.py     # Retrieval + cross-encoder reranking
│   └── generator.py     # LLM answer generation (streaming)
├── data/papers/         # Your PDF files (git-ignored)
└── chroma_db/           # Vector store persistence (git-ignored)
```

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| UI | Streamlit | `pip install streamlit` |
| PDF Parsing | PyMuPDF (fitz) | Citation-aware section splitting |
| Embedding | BGE (BAAI/bge-small-zh-v1.5) | Runs locally, no API needed |
| Vector DB | Chroma | Persistent, zero-config |
| LLM | DeepSeek API | OpenAI-compatible SDK |
| Reranking | ms-marco-MiniLM-L-6-v2 | Cross-encoder for precision |

## License

MIT
