"""
PaperRAG — Your Papers, Your Knowledge
Streamlit UI with side-by-side answer + enhanced sources + export + multi-turn
"""
import os
import streamlit as st
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from src.loader import load_pdf_text, chunk_papers
from src.embedder import embed_texts
from src.vector_store import get_collection, add_chunks, get_chunk_count, get_paper_names, delete_paper
from src.retriever import retrieve, rerank
from src.generator import generate_answer_stream

st.set_page_config(page_title="PaperRAG", page_icon="📄", layout="wide")
st.title("📄 PaperRAG — Your Papers, Your Knowledge")

# ---- Session state for multi-turn ----
if "history" not in st.session_state:
    st.session_state.history = []  # [(question, answer, sources)]

# ---- Sidebar ----
with st.sidebar:
    st.header("📁 Paper Library")

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
                    for uf in new_files:
                        path = os.path.join("data/papers", uf.name)
                        with open(path, "wb") as f:
                            f.write(uf.getbuffer())
                        text = load_pdf_text(path)
                        if text.strip():
                            papers.append({"name": uf.name, "text": text})

                    if not papers:
                        st.error("No valid PDFs found.")
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
        paper_names = st.session_state.paper_list

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
                    st.caption(f"• {p}")
        else:
            for p in paper_names:
                st.caption(f"• {p}")

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

    # Clear history button
    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []
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
            # Stream answer with LaTeX support
            with left:
                st.markdown("### 📝 Answer")
                placeholder = st.empty()
                full_text = ""
                for chunk in generate_answer_stream(query, results):
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

            # Save to history
            st.session_state.history.append((query, answer, results))

with tab2:
    if not st.session_state.history:
        st.caption("Your Q&A history will appear here. Use the sidebar to clear.")
    else:
        for q, a, sources in reversed(st.session_state.history):
            with st.expander(f"❓ {q[:100]}...", expanded=False):
                st.markdown(f"**Q:** {q}")
                st.markdown(a)
                st.caption("—" * 20)
                seen = set()
                for s in sources:
                    if s["paper_name"] not in seen:
                        st.caption(f"📄 {s['paper_name']}")
                        seen.add(s["paper_name"])
