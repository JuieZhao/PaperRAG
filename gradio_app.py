"""
PaperRAG — Gradio Chat UI
Dark academic theme, hybrid search, metadata filtering, streaming answers.
"""
from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from src.loader import load_pdf_text, load_pdf_meta, chunk_papers, compute_file_hash
from src.embedder import embed_texts, get_model_info
from src.vector_store import (
    get_collection, add_chunks, get_chunk_count, get_paper_names,
    delete_paper, get_file_hashes, get_filter_options,
)
from src.retriever import search_papers
from src.generator import generate_answer_stream

HISTORY_FILE = "data/qa_history.json"
COLLECTION = get_collection()


# ═══════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════

CUSTOM_CSS = """
:root {
  --bg: #0f1419;
  --bg2: #151c25;
  --card: #1a2230;
  --border: rgba(99,130,180,0.15);
  --accent: #5b8def;
  --text: #e2e6ed;
  --text2: #8899b4;
}

body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Noto Sans SC', 'Inter', sans-serif !important;
}
.gradio-container { max-width: 100% !important; }

/* Chat messages */
.message.user { background: var(--card) !important; border: 1px solid var(--border) !important; }
.message.bot { background: var(--bg2) !important; border: 1px solid var(--border) !important; }
.message { color: var(--text) !important; border-radius: 8px !important; padding: 16px !important; }

/* Buttons */
button.primary {
  background: var(--accent) !important; border: none !important; color: white !important;
  font-weight: 600 !important; border-radius: 6px !important;
}
button.secondary {
  background: transparent !important; border: 1px solid var(--border) !important;
  color: var(--text2) !important; border-radius: 6px !important;
}

/* Inputs */
input, textarea, select {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: 6px !important;
}

/* Sidebar */
#sidebar { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
"""


# ═══════════════════════════════════════════
#  Core handlers
# ═══════════════════════════════════════════

def get_stats() -> str:
    """Get current library stats."""
    chunk_count = get_chunk_count(COLLECTION)
    paper_count = len(get_paper_names(COLLECTION))
    try:
        info = get_model_info()
    except Exception:
        info = {"model_id": "?", "dimensions": "?", "device": "?"}
    return (
        f"📊 **Chunks:** {chunk_count:,}  |  📄 **Papers:** {paper_count}\n"
        f"🔤 Model: `{info['model_id'].split('/')[-1]}` · {info['dimensions']}d · `{info['device']}`"
    )


def upload_and_index(files: list[str], progress=gr.Progress()) -> str:
    """Upload PDFs, index with dedup and table extraction."""
    if not files:
        return "⚠️ No files selected."

    os.makedirs("data/papers", exist_ok=True)
    existing_names = set(get_paper_names(COLLECTION))
    existing_hashes = get_file_hashes(COLLECTION)

    new_files = []
    skipped = 0

    for tmp_path in progress.tqdm(files, desc="Checking files..."):
        name = os.path.basename(tmp_path)
        # Copy to papers dir
        dest = os.path.join("data/papers", name)
        with open(tmp_path, "rb") as src:
            content = src.read()
        with open(dest, "wb") as dst:
            dst.write(content)

        file_hash = compute_file_hash(dest)
        if name in existing_names or file_hash in existing_hashes:
            skipped += 1
            if name not in existing_names:
                os.remove(dest)
            continue
        new_files.append((name, dest, file_hash))

    if not new_files:
        return f"⏭️ All {len(files)} file(s) already indexed (or duplicates)."

    papers = []
    failed = []
    paper_metas = {}

    for name, path, file_hash in progress.tqdm(new_files, desc="Parsing PDFs..."):
        text = load_pdf_text(path)
        if text.strip():
            papers.append({"name": name, "text": text, "path": path})
            meta = load_pdf_meta(path)
            meta["file_hash"] = file_hash
            paper_metas[name] = meta
        else:
            failed.append(name)

    if not papers:
        return f"❌ No extractable text found in {len(new_files)} file(s)."

    progress(0.5, desc="Chunking + embedding...")
    chunks = chunk_papers(papers, extract_tables=True)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Store with metadata
    for c in chunks:
        if c["paper_name"] in paper_metas:
            add_chunks(COLLECTION, [c], [embeddings[chunks.index(c)]],
                       paper_meta=paper_metas[c["paper_name"]])

    table_count = sum(1 for c in chunks if c.get("is_table"))
    msg = f"✅ Indexed {len(papers)} paper(s) → {len(chunks)} chunks"
    if table_count:
        msg += f" (incl. {table_count} tables)"
    if skipped:
        msg += f" | Skipped {skipped} duplicate(s)"
    if failed:
        msg += f"\n⚠️ Failed: {', '.join(failed)} (scanned PDFs?)"
    return msg


def delete_selected_papers(paper_names: list[str]) -> str:
    """Delete selected papers by name."""
    if not paper_names:
        return "No papers selected."
    total = 0
    for name in paper_names:
        total += delete_paper(COLLECTION, name)
    return f"🗑️ Removed {total} chunks from {len(paper_names)} paper(s)."


def ask_question(
    query: str,
    history: list,
    api_key: str,
    model: str,
    temperature: float,
    top_k: int,
    hybrid: bool,
    author_filter: str,
    year_filter: str,
    paper_filter: str,
) -> tuple[list, str]:
    """Handle a question: retrieve, generate streaming answer, return chat history."""
    if not query.strip():
        return history, ""

    if not api_key:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "⚠️ Please set your DeepSeek API key in the sidebar."})
        return history, ""

    os.environ["DEEPSEEK_API_KEY"] = api_key

    # Build metadata filters
    filters = None
    conditions = []
    if author_filter and author_filter != "All":
        conditions.append({"author": author_filter})
    if year_filter and year_filter != "All":
        conditions.append({"year": year_filter})
    if paper_filter and paper_filter != "All":
        conditions.append({"paper_name": paper_filter})
    if len(conditions) == 1:
        filters = conditions[0]
    elif len(conditions) > 1:
        filters = {"$and": conditions}

    # Retrieve + rerank
    try:
        results = search_papers(COLLECTION, query, top_k=top_k, hybrid=hybrid, filters=filters)
    except Exception as e:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": f"⚠️ Retrieval error: {e}"})
        return history, ""

    if not results:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "No relevant passages found. Try rephrasing or uploading more papers."})
        return history, ""

    # Generate answer with streaming
    sources_text = "\n\n---\n**📚 Sources:**\n"
    seen = set()
    for r in results:
        if r["paper_name"] not in seen:
            table_tag = " 📊" if r.get("is_table") else ""
            sources_text += f"- *{r['paper_name']}* (p.{r.get('page', '?')}, {r['score']:.0%}){table_tag}\n"
            seen.add(r["paper_name"])

    full_answer = ""
    for token in generate_answer_stream(
        query, results,
        model=model,
        temperature=temperature,
    ):
        full_answer += token

    answer = full_answer + sources_text
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    # Save to persistent history
    _save_chat(query, answer)
    return history, ""


def _save_chat(query: str, answer: str):
    """Persist chat to JSON file."""
    os.makedirs("data", exist_ok=True)
    entry = {
        "query": query,
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
    }
    history_data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history_data = json.load(f)
        except Exception:
            pass
    history_data.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)


def refresh_filters() -> tuple:
    """Return updated filter dropdown choices."""
    opts = get_filter_options(COLLECTION)
    authors = ["All"] + opts.get("authors", [])
    years = ["All"] + opts.get("years", [])
    papers = ["All"] + opts.get("paper_names", [])
    return (
        gr.update(choices=authors, value="All"),
        gr.update(choices=years, value="All"),
        gr.update(choices=papers, value="All"),
        gr.update(value=get_stats()),
    )


# ═══════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════

def build_ui():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="PaperRAG — Ask Your Papers",
        fill_height=True,
    ) as demo:
        # ── Sidebar ──
        with gr.Column(scale=1, elem_id="sidebar", min_width=300):
            gr.Markdown(
                """<h1 style="font-family:'Source Serif 4',serif;font-size:1.6rem;color:#e2e6ed;margin-bottom:4px;">
                Paper<span style="color:#5b8def;">RAG</span>
                </h1>"""
            )
            stats_display = gr.Markdown(get_stats(), every=10)

            with gr.Accordion("🔑 API Key", open=True):
                api_key_input = gr.Textbox(
                    label="DeepSeek API Key",
                    type="password",
                    value=os.getenv("DEEPSEEK_API_KEY", ""),
                    placeholder="sk-xxx...xxxx",
                )

            with gr.Accordion("📤 Upload Papers", open=False):
                file_upload = gr.File(
                    label="Select PDF files",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                upload_btn = gr.Button("⚡ Index Papers", variant="primary")
                upload_status = gr.Markdown("")

                upload_btn.click(
                    upload_and_index,
                    inputs=[file_upload],
                    outputs=[upload_status],
                ).then(refresh_filters, outputs=[
                    gr.State(), gr.State(), gr.State(), stats_display,
                ])

            with gr.Accordion("📚 Library", open=False):
                paper_list = gr.CheckboxGroup(
                    label="Select papers to delete",
                    choices=get_paper_names(COLLECTION),
                )
                delete_btn = gr.Button("🗑️ Delete Selected", variant="stop", size="sm")
                delete_status = gr.Markdown("")
                refresh_btn = gr.Button("🔄 Refresh", size="sm")

                delete_btn.click(
                    delete_selected_papers,
                    inputs=[paper_list],
                    outputs=[delete_status],
                )
                refresh_btn.click(
                    lambda: gr.update(choices=get_paper_names(COLLECTION)),
                    outputs=[paper_list],
                )

            with gr.Accordion("🔎 Filters", open=False):
                filter_opts = get_filter_options(COLLECTION)
                author_dd = gr.Dropdown(
                    label="Author",
                    choices=["All"] + filter_opts.get("authors", []),
                    value="All",
                )
                year_dd = gr.Dropdown(
                    label="Year",
                    choices=["All"] + filter_opts.get("years", []),
                    value="All",
                )
                paper_dd = gr.Dropdown(
                    label="Paper",
                    choices=["All"] + filter_opts.get("paper_names", []),
                    value="All",
                )
                refresh_filters_btn = gr.Button("🔄 Refresh Filters", size="sm")
                refresh_filters_btn.click(
                    refresh_filters,
                    outputs=[author_dd, year_dd, paper_dd, stats_display],
                )

            with gr.Accordion("⚙️ Settings", open=False):
                model_dd = gr.Dropdown(
                    label="LLM Model",
                    choices=["deepseek-chat", "deepseek-reasoner"],
                    value="deepseek-chat",
                )
                temp_slider = gr.Slider(
                    label="Temperature",
                    minimum=0.0, maximum=1.0, value=0.3, step=0.1,
                )
                topk_dd = gr.Dropdown(
                    label="Results",
                    choices=[5, 8, 10, 15],
                    value=8,
                )
                hybrid_toggle = gr.Checkbox(
                    label="🔀 Hybrid search (BM25 + Dense)",
                    value=True,
                )

        # ── Main Chat Area ──
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Ask your papers",
                type="messages",
                height="70vh",
                placeholder="Ask a question about your papers... e.g. 'What is the effect of trade policy uncertainty on firm innovation?'",
            )
            query_input = gr.Textbox(
                label="Your question",
                placeholder="Type your question here...",
                scale=4,
            )
            send_btn = gr.Button("🔍 Ask", variant="primary", scale=1)

            # Chat interaction
            send_btn.click(
                ask_question,
                inputs=[
                    query_input,
                    chatbot,
                    api_key_input,
                    model_dd,
                    temp_slider,
                    topk_dd,
                    hybrid_toggle,
                    author_dd,
                    year_dd,
                    paper_dd,
                ],
                outputs=[chatbot, query_input],
            ).then(lambda: "", outputs=[query_input])

            query_input.submit(
                ask_question,
                inputs=[
                    query_input,
                    chatbot,
                    api_key_input,
                    model_dd,
                    temp_slider,
                    topk_dd,
                    hybrid_toggle,
                    author_dd,
                    year_dd,
                    paper_dd,
                ],
                outputs=[chatbot, query_input],
            ).then(lambda: "", outputs=[query_input])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
