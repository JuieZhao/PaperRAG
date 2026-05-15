"""
PaperRAG — Your Papers, Your Knowledge
Streamlit UI with side-by-side answer + enhanced sources + export + multi-turn
"""
import os
import json
import streamlit as st
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from src.loader import load_pdf_text, load_pdf_meta, chunk_papers
from src.embedder import embed_texts
from src.vector_store import get_collection, add_chunks, get_chunk_count, get_paper_names, delete_paper
from src.retriever import retrieve, rerank
from src.generator import generate_answer_stream

HISTORY_FILE = "data/qa_history.json"

st.set_page_config(page_title="PaperRAG", page_icon="📄", layout="wide")
st.title("📄 PaperRAG — Your Papers, Your Knowledge")


# ---- History persistence helpers ----
def _save_history():
    """Persist Q&A history and chat messages to JSON."""
    os.makedirs("data", exist_ok=True)
    payload = {
        "qa_history": st.session_state.history,
        "chat_messages": st.session_state.chat_messages,
    }
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_history():
    """Load persisted history from JSON. Returns (qa_history, chat_messages)."""
    if not os.path.exists(HISTORY_FILE):
        return [], []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("qa_history", []), data.get("chat_messages", [])
    except Exception:
        return [], []


# ---- Session state ----
if "history" not in st.session_state:
    qa, msgs = _load_history()
    st.session_state.history = qa  # [(question, answer, sources, timestamp)]
    st.session_state.chat_messages = msgs  # [{"role": ..., "content": ...}]

# ---- Sidebar ----
with st.sidebar:
    st.header("📁 Paper Library")

    # ---- API Key (UI-friendly, no .env file needed) ----
    with st.expander("🔑 API Key Setup", expanded=not os.getenv("DEEPSEEK_API_KEY")):
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            placeholder="sk-xxxxxxxxxxxxxxxx",
            help="Paste your DeepSeek API key here. Get one at platform.deepseek.com",
            label_visibility="collapsed",
        )
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        st.caption("Get a free key at [platform.deepseek.com](https://platform.deepseek.com)")
        if not api_key:
            st.warning("⚠️ API key required to ask questions")

    collection = get_collection()
    chunk_count = get_chunk_count(collection)
    st.metric("Indexed chunks", chunk_count)

    # Load paper names early (needed for dedup check during upload)
    if "paper_list" not in st.session_state or st.session_state.paper_list is None:
        st.session_state.paper_list = get_paper_names(collection)
    existing_papers = set(st.session_state.paper_list)

    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Index papers", type="primary"):
            with st.spinner("Parsing + embedding..."):
                os.makedirs("data/papers", exist_ok=True)

                # Check for already-indexed papers
                skipped = [uf.name for uf in uploaded_files if uf.name in existing_papers]
                new_files = [uf for uf in uploaded_files if uf.name not in existing_papers]

                if skipped:
                    st.info(f"⏭️ Already indexed, skipped: {', '.join(skipped)}")

                if not new_files:
                    st.warning("All uploaded papers are already indexed.")
                else:
                    papers = []
                    failed = []
                    for uf in new_files:
                        path = os.path.join("data/papers", uf.name)
                        with open(path, "wb") as f:
                            f.write(uf.getbuffer())
                        size_kb = os.path.getsize(path) / 1024
                        text = load_pdf_text(path)
                        if text.strip():
                            papers.append({"name": uf.name, "text": text})
                        else:
                            failed.append((uf.name, size_kb))

                    if not papers:
                        st.error("No valid PDFs found.")
                        if failed:
                            for name, kb in failed:
                                st.warning(f"❌ `{name}` ({kb:.0f} KB) — no extractable text (scanned PDF?)")
                            st.info("💡 Scanned/image-based PDFs need OCR. Try re-saving as text-based PDF or use an OCR tool first.")
                    else:
                        # Step 1: Chunk
                        progress = st.progress(0, "Chunking papers...")
                        chunks = chunk_papers(papers)
                        progress.progress(30, "Embedding text...")

                        # Step 2: Embed
                        texts = [c["text"] for c in chunks]
                        embeddings = embed_texts(texts)
                        progress.progress(80, "Storing to database...")

                        # Step 3: Store
                        add_chunks(collection, chunks, embeddings)
                        progress.progress(100, "Done!")

                        st.success(f"✅ Indexed {len(papers)} papers, {len(chunks)} chunks")
                        st.session_state.paper_list = None
                        st.rerun()

    st.divider()

    if chunk_count > 0:
        st.subheader("📚 Indexed papers")
        # Refresh paper list if chunk count changed (after index/delete)
        if st.session_state.paper_list is None or get_chunk_count(collection) != chunk_count:
            st.session_state.paper_list = get_paper_names(collection)
            st.session_state.paper_meta = {}  # reset metadata cache
        paper_names = st.session_state.paper_list

        # Lazy-load paper metadata (title, author, year) with caching
        if "paper_meta" not in st.session_state:
            st.session_state.paper_meta = {}
        for p in paper_names:
            if p not in st.session_state.paper_meta:
                path = os.path.join("data/papers", p)
                if os.path.exists(path):
                    st.session_state.paper_meta[p] = load_pdf_meta(path)
                else:
                    st.session_state.paper_meta[p] = {"title": "", "author": "", "year": ""}

        # Filter papers
        if len(paper_names) > 5:
            filter_text = st.text_input(
                "🔍 Filter papers",
                placeholder=f"Search among {len(paper_names)} papers...",
                label_visibility="collapsed",
            )
            if filter_text:
                paper_names = [p for p in paper_names if filter_text.lower() in p.lower()]

        if len(paper_names) > 10:
            with st.expander(f"📋 {len(paper_names)} papers (click to expand)", expanded=len(paper_names) <= 5):
                for p in paper_names:
                    meta = st.session_state.paper_meta.get(p, {})
                    title = meta.get("title") or p
                    author = meta.get("author", "")
                    year = meta.get("year", "")
                    line = f"• **{title}**"
                    if author:
                        line += f" — {author}"
                    if year:
                        line += f" ({year})"
                    st.caption(line)
        else:
            for p in paper_names:
                meta = st.session_state.paper_meta.get(p, {})
                title = meta.get("title") or p
                author = meta.get("author", "")
                year = meta.get("year", "")
                line = f"• **{title}**"
                if author:
                    line += f" — {author}"
                if year:
                    line += f" ({year})"
                st.caption(line)

        st.divider()

        # Batch delete papers (checkbox + bulk action)
        st.subheader("🗑️ Remove papers")
        selected = []
        for p in paper_names:
            if st.checkbox(p, key=f"del_{p}"):
                selected.append(p)

        if selected:
            label = f"🗑️ Delete {len(selected)} selected paper(s)"
            if st.button(label, type="primary"):
                total_removed = 0
                for p in selected:
                    n = delete_paper(collection, p)
                    total_removed += n
                st.success(f"Removed {total_removed} chunks from {len(selected)} paper(s)")
                st.session_state.paper_list = None
                st.rerun()

    st.divider()

    # ---- Model settings ----
    st.subheader("⚙️ Model Settings")
    model_choice = st.selectbox(
        "LLM Model",
        ["deepseek-chat", "deepseek-reasoner"],
        index=0,
        label_visibility="collapsed",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0, value=0.3, step=0.1,
        label_visibility="collapsed",
    )
    st.caption(f"Model: `{model_choice}` | Temp: `{temperature}`")

    st.divider()

    # Clear history button
    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []
        st.session_state.chat_messages = []
        _save_history()
        st.rerun()

    st.caption("Built with Chroma + BGE + DeepSeek")

# ---- Main area ----
tab1, tab2 = st.tabs(["🔍 Ask", "💬 History"])

with tab1:
    with st.form("search_form", clear_on_submit=False):
        col_q, col_k, col_btn = st.columns([4, 1, 1])
        with col_q:
            query = st.text_input(
                "Question",
                placeholder="Ask anything based on your uploaded papers...",
                label_visibility="collapsed",
            )
        with col_k:
            top_k = st.selectbox("Results", [5, 8, 10, 15], index=1, label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

    if query and submitted:
        left, right = st.columns([3, 2])

        with st.spinner("Retrieving + reranking..."):
            # Fetch extra candidates, then cross-encoder rerank
            results = retrieve(collection, query, top_k=top_k * 3, max_per_paper=3)
            results = rerank(query, results, top_k=top_k)

        if not results:
            st.warning("No relevant papers found. Upload some PDFs first.")
        else:
            # Stream answer with LaTeX support + conversation history
            with left:
                st.markdown("### 📝 Answer")
                placeholder = st.empty()
                full_text = ""
                for chunk in generate_answer_stream(
                    query, results,
                    model=model_choice,
                    history=st.session_state.chat_messages,
                    temperature=temperature,
                ):
                    full_text += chunk
                    # Normalize LaTeX delimiters: \(...\) → $...$, \[...\] → $$...$$
                    display = full_text.replace("\\(", "$").replace("\\)", "$")
                    display = display.replace("\\[", "$$\n").replace("\\]", "\n$$")
                    placeholder.markdown(display)
                answer = full_text

                # Copy + export
                copy_text = answer + "\n\n---\n### Sources\n"
                seen = set()
                for r in results:
                    if r["paper_name"] not in seen:
                        copy_text += f"- {r['paper_name']} (relevance: {r['score']:.0%})\n"
                        seen.add(r["paper_name"])
                st.download_button(
                    "📋 Export answer as Markdown",
                    data=copy_text,
                    file_name="paperrag_answer.md",
                    mime="text/markdown",
                )

            # Sources in right column
            with right:
                st.markdown("### 📚 Sources")
                by_paper = defaultdict(list)
                for r in results:
                    by_paper[r["paper_name"]].append(r)
                for paper_name, chunks_list in by_paper.items():
                    rel = max(c["score"] for c in chunks_list)
                    with st.expander(
                        f"📄 {paper_name} ({len(chunks_list)} chunks, best match {rel:.0%})",
                        expanded=False,
                    ):
                        for i, c in enumerate(chunks_list):
                            st.markdown(f"**Relevance:** `{c['score']:.2%}`")
                            st.text(c["text"])
                            if i < len(chunks_list) - 1:
                                st.divider()

            # Save to history + conversation
            timestamp = datetime.now().isoformat()
            st.session_state.history.append((query, answer, results, timestamp))
            st.session_state.chat_messages.append({"role": "user", "content": query})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            _save_history()

with tab2:
    if not st.session_state.history:
        st.caption("Your Q&A history will appear here. Use the sidebar to clear. History persists across restarts.")
    else:
        for q, a, sources, ts in reversed(st.session_state.history):
            label = f"❓ {q[:100]}..."
            with st.expander(label, expanded=False):
                st.caption(f"🕒 {ts[:19]}")
                st.markdown(f"**Q:** {q}")
                st.markdown(a)
                st.caption("—" * 20)
                seen = set()
                for s in sources:
                    if s["paper_name"] not in seen:
                        st.caption(f"📄 {s['paper_name']}")
                        seen.add(s["paper_name"])
