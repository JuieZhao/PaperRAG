"""
MiniRAG — 轻量级文档知识检索 · 北极甜虾 Design Edition
Streamlit UI: dark academic theme, hybrid search, metadata filtering
"""
import os
import json
import hashlib
import streamlit as st
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from src.loader import load_document_text, load_document_meta, chunk_documents, compute_file_hash
from src.embedder import embed_texts, get_model_info
from src.vector_store import (
    get_collection, add_chunks, get_chunk_count, get_source_names,
    delete_source, get_file_hashes, get_filter_options,
)
from src.retriever import search_documents

HISTORY_FILE = "data/qa_history.json"
DOCUMENTS_DIR = "documents"

st.set_page_config(page_title="MiniRAG", page_icon="📄", layout="wide")

# ═══════════════════════════════════════════
#  Custom CSS — Dark Academic Theme
# ═══════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg-primary: #0f1419;
    --bg-secondary: #151c25;
    --bg-card: #1a2230;
    --bg-hover: #1e2838;
    --border: rgba(99, 130, 180, 0.15);
    --accent: #5b8def;
    --accent-dim: rgba(91, 141, 239, 0.12);
    --text-primary: #e2e6ed;
    --text-secondary: #8899b4;
    --text-muted: #5a6e88;
    --green: #4ec9a8;
    --red: #f07178;
    --amber: #e5b567;
    --purple: #c792ea;
  }

  .stApp { background: var(--bg-primary); }
  .main .block-container { padding-top: 1.5rem; max-width: 100%; }

  h1, h2, h3, h4, h5, h6 {
    font-family: 'Source Serif 4', 'Noto Sans SC', serif !important;
    color: var(--text-primary) !important; font-weight: 700 !important;
  }
  p, li, label, div, span { font-family: 'Noto Sans SC', sans-serif; }
  code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important; border-radius: 4px !important;
  }

  [data-testid="stSidebar"] {
    background: var(--bg-secondary); border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

  .stTextInput > div > div > input,
  .stSelectbox > div > div,
  .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
    color: var(--text-primary) !important; font-size: 0.9rem;
  }
  .stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
  }

  .stButton > button {
    font-family: 'Noto Sans SC', sans-serif !important; font-weight: 600 !important;
    border-radius: 6px !important; transition: all 0.2s ease !important;
  }
  .stButton > button[kind="primary"] {
    background: var(--accent) !important; border: none !important; color: white !important;
  }
  .stButton > button[kind="primary"]:hover { filter: brightness(1.15); box-shadow: 0 0 16px var(--accent-dim); }

  [data-testid="stMetric"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 16px;
  }
  [data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important; letter-spacing: 0.08em !important; color: var(--text-muted) !important;
  }
  [data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important; font-weight: 700 !important; color: var(--accent) !important;
  }

  .paper-card {
    padding: 8px 12px; margin: 4px 0; border-radius: 6px;
    border: 1px solid var(--border); background: var(--bg-card);
    transition: all 0.15s ease; cursor: default;
  }
  .paper-card:hover { border-color: var(--accent); background: var(--bg-hover); }
  .paper-card .title { font-weight: 600; color: var(--text-primary); font-size: 0.8rem; }
  .paper-card .meta { color: var(--text-muted); font-size: 0.68rem; margin-top: 2px; }

  .source-block {
    padding: 12px 16px; margin: 8px 0; border-left: 3px solid var(--accent);
    background: var(--bg-card); border-radius: 0 6px 6px 0;
    font-size: 0.78rem; color: var(--text-secondary); line-height: 1.6;
  }
  .source-block .relevance { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--accent); margin-bottom: 6px; }
  .source-block .text { color: var(--text-primary); }

  .answer-container {
    padding: 20px 24px; background: var(--bg-card); border-radius: 8px;
    border: 1px solid var(--border); min-height: 200px; line-height: 1.8; font-size: 0.9rem;
  }

  .footer-text {
    font-size: 0.62rem; color: var(--text-muted); letter-spacing: 0.06em;
    text-align: center; padding: 16px 0 4px;
  }
  hr { border-color: var(--border) !important; }

  .streamlit-expanderHeader {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; font-size: 0.82rem !important; color: var(--text-secondary) !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-secondary) !important; border: 1px solid var(--border) !important;
    border-top: none !important; border-radius: 0 0 6px 6px !important; padding: 16px !important;
  }

  .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid var(--border); }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 6px 6px 0 0 !important;
    padding: 8px 20px !important; color: var(--text-muted) !important; font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: var(--bg-card) !important; color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }

  .stSpinner > div { border-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  History persistence
# ═══════════════════════════════════════════
def _save_history():
    os.makedirs("data", exist_ok=True)
    payload = {
        "qa_history": st.session_state.history,
        "chat_messages": st.session_state.chat_messages,
    }
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_history():
    if not os.path.exists(HISTORY_FILE):
        return [], []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("qa_history", []), data.get("chat_messages", [])
    except Exception:
        return [], []


# ═══════════════════════════════════════════
#  Session state
# ═══════════════════════════════════════════
if "history" not in st.session_state:
    qa, msgs = _load_history()
    st.session_state.history = qa
    st.session_state.chat_messages = msgs


# ═══════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
      <span style="font-family:'Source Serif 4',serif;font-size:2rem;font-weight:700;color:#e2e6ed;">
        Mini<span style="color:#5b8def;">RAG</span>
      </span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#5a6e88;letter-spacing:0.15em;">
        HYBRID EDITION
      </span>
    </div>
    <p style="color:#8899b4;font-size:0.82rem;margin-top:0;">
      Your documents, your knowledge. Hybrid BM25 + Dense retrieval.
    </p>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════
with st.sidebar:
    collection = get_collection()
    chunk_count = get_chunk_count(collection)

    # ── Stats ──
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-bottom:20px;">
      <div style="flex:1;background:#1a2230;border:1px solid rgba(99,130,180,.15);border-radius:8px;padding:12px 14px;text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#5b8def;">{chunk_count:,}</div>
        <div style="font-size:0.62rem;color:#5a6e88;letter-spacing:.06em;margin-top:2px;">CHUNKS</div>
      </div>
      <div style="flex:1;background:#1a2230;border:1px solid rgba(99,130,180,.15);border-radius:8px;padding:12px 14px;text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#4ec9a8;">
          {len(st.session_state.get('source_list') or get_source_names(collection)):,}
        </div>
        <div style="font-size:0.62rem;color:#5a6e88;letter-spacing:.06em;margin-top:2px;">DOCUMENTS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Embedding info ──
    try:
        info = get_model_info()
        st.caption(f"🔤 `{info['model_id'].split('/')[-1]}` · {info['dimensions']}d · {info['device']}")
    except Exception:
        pass

    # ── API Key ──
    with st.expander("🔑 API Key", expanded=not os.getenv("DEEPSEEK_API_KEY")):
        api_key = st.text_input(
            "DeepSeek API Key", type="password",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            placeholder="sk-xxx...xxxx", label_visibility="collapsed",
        )
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        st.caption("[platform.deepseek.com](https://platform.deepseek.com)")
        if not api_key:
            st.warning("API key required")

    st.divider()

    # ── Upload ──
    st.markdown("### 📤 Upload Documents")
    st.caption(f"Files are saved in `{DOCUMENTS_DIR}/`.")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Index Documents", type="primary", use_container_width=True):
            with st.spinner("Parsing + embedding..."):
                os.makedirs(DOCUMENTS_DIR, exist_ok=True)

                if "source_list" not in st.session_state or st.session_state.source_list is None:
                    st.session_state.source_list = get_source_names(collection)
                existing_names = set(st.session_state.source_list)
                existing_hashes = get_file_hashes(collection)

                skipped = []
                new_files = []

                for uf in uploaded_files:
                    path = os.path.join(DOCUMENTS_DIR, uf.name)
                    with open(path, "wb") as f:
                        f.write(uf.getbuffer())

                    # ── Content dedup: check hash ──
                    file_hash = compute_file_hash(path)
                    if uf.name in existing_names or file_hash in existing_hashes:
                        skipped.append(uf.name)
                        if uf.name not in existing_names:
                            os.remove(path)  # clean up duplicate under different name
                        continue
                    new_files.append((uf.name, path, file_hash))

                if skipped:
                    st.info(f"⏭️ Skipped {len(skipped)} duplicate(s): {', '.join(skipped[:5])}"
                            f"{'...' if len(skipped) > 5 else ''}")

                if not new_files:
                    st.warning("All files already indexed (or duplicates).")
                else:
                    papers = []
                    failed = []
                    source_metas = {}

                    for name, path, file_hash in new_files:
                        text = load_document_text(path)
                        if text.strip():
                            papers.append({"name": name, "text": text, "path": path})
                            meta = load_document_meta(path)
                            meta["file_hash"] = file_hash
                            source_metas[name] = meta
                        else:
                            failed.append(name)

                    if not papers:
                        st.error("No extractable text found.")
                        for name in failed:
                            st.warning(f"❌ `{name}` — no extractable text found")
                    else:
                        progress = st.progress(0, "Chunking...")
                        chunks = chunk_documents(papers, extract_tables=True)

                        progress.progress(25, "Embedding...")
                        texts = [c["text"] for c in chunks]
                        embeddings = embed_texts(texts)

                        progress.progress(75, "Storing...")
                        add_chunks(collection, chunks, embeddings, source_meta_map=source_metas)
                        progress.progress(100, "Done!")
                        table_count = sum(1 for c in chunks if c.get("is_table"))
                        msg = f"✅ {len(papers)} documents · {len(chunks)} chunks"
                        if table_count:
                            msg += f" · {table_count} tables"
                        st.success(msg)
                        st.session_state.source_list = None
                        st.rerun()

    # ── Document Library ──
    if chunk_count > 0:
        st.divider()
        st.markdown("### 📚 Library")

        if st.session_state.get("source_list") is None or get_chunk_count(collection) != chunk_count:
            st.session_state.source_list = get_source_names(collection)
            st.session_state.source_meta = {}

        source_names = st.session_state.source_list

        if "source_meta" not in st.session_state:
            st.session_state.source_meta = {}
        for p in source_names:
            if p not in st.session_state.source_meta:
                path = os.path.join(DOCUMENTS_DIR, p)
                if os.path.exists(path):
                    st.session_state.source_meta[p] = load_document_meta(path)
                else:
                    st.session_state.source_meta[p] = {"title": "", "author": "", "year": ""}

        if len(source_names) > 5:
            filter_text = st.text_input(
                "Filter", placeholder=f"Search {len(source_names)} documents...",
                label_visibility="collapsed",
            )
            if filter_text:
                source_names = [p for p in source_names if filter_text.lower() in p.lower()]

        for p in source_names[:20]:
            meta = st.session_state.source_meta.get(p, {})
            title = meta.get("title") or p.rsplit(".", 1)[0][:80]
            author = meta.get("author", "")
            year = meta.get("year", "")
            subtitle = ""
            if author:
                subtitle += author
            if year:
                subtitle += f" · {year}" if subtitle else year
            st.markdown(f"""
            <div class="paper-card">
              <div class="title">{title}</div>
              <div class="meta">{subtitle or '—'}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(source_names) > 20:
            st.caption(f"... and {len(source_names) - 20} more")

        # ── Delete papers ──
        st.divider()
        st.markdown("### 🗑️ Remove")
        selected = []
        for p in source_names:
            if st.checkbox(p, key=f"del_{p}"):
                selected.append(p)
        if selected:
            if st.button(f"Delete {len(selected)} document(s)", type="primary", use_container_width=True):
                total = 0
                for p in selected:
                    total += delete_source(collection, p)
                st.success(f"Removed {total} chunks")
                st.session_state.source_list = None
                st.rerun()

    # ── Settings ──
    st.divider()
    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "Model", ["deepseek-chat", "deepseek-reasoner"],
        index=0, label_visibility="collapsed",
    )
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
        label_visibility="collapsed",
    )

    # Hybrid search toggle
    use_hybrid = st.checkbox("🔀 Hybrid search (BM25 + Dense)", value=True,
                             help="Combine keyword matching with semantic search via RRF fusion")

    st.caption(f"`{model_choice}` · temp `{temperature}` · {'hybrid' if use_hybrid else 'dense only'}")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.chat_messages = []
        _save_history()
        st.rerun()

    st.markdown('<p class="footer-text">Chroma + BGE + DeepSeek + BM25</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  Main Area
# ═══════════════════════════════════════════
tab1, tab2 = st.tabs(["🔍 Ask", "📜 History"])

with tab1:
    # ── Metadata filter bar ──
    if chunk_count > 0:
        filter_opts = get_filter_options(collection)
        with st.expander("🔎 Filters", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                author_filter = st.selectbox(
                    "Author", ["All"] + filter_opts.get("authors", []),
                    key="filter_author",
                )
            with fc2:
                year_filter = st.selectbox(
                    "Year", ["All"] + filter_opts.get("years", []),
                    key="filter_year",
                )
            with fc3:
                source_filter = st.selectbox(
                    "Paper", ["All"] + filter_opts.get("source_names", []),
                    key="filter_paper",
                )

    with st.form("search_form", clear_on_submit=False):
        col_q, col_k, col_btn = st.columns([4, 1, 1])
        with col_q:
            query = st.text_input(
                "Your question",
                placeholder="e.g. What is the effect of trade policy uncertainty on firm innovation?",
                label_visibility="collapsed",
            )
        with col_k:
            top_k = st.selectbox("", [5, 8, 10, 15], index=1, label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("🔍 Ask", type="primary", use_container_width=True)

    if query and submitted:
        # ── Build metadata filters ──
        filters = None
        if chunk_count > 0:
            conditions = []
            if author_filter != "All":
                conditions.append({"author": author_filter})
            if year_filter != "All":
                conditions.append({"year": year_filter})
            if source_filter != "All":
                conditions.append({"source_name": source_filter})
            if len(conditions) == 1:
                filters = conditions[0]
            elif len(conditions) > 1:
                filters = {"$and": conditions}

        with st.spinner("Searching documents + generating answer..."):
            results = search_documents(
                collection, query,
                top_k=top_k,
                hybrid=use_hybrid,
                filters=filters,
            )

        if not results:
            st.warning("No relevant passages found. Try rephrasing, adjusting filters, or uploading more documents.")
        else:
            left, right = st.columns([3, 2])

            with left:
                st.markdown("### 📝 Answer")

                # Build context for generator
                from src.generator import generate_answer_stream
                answer_div = st.empty()
                full_text = ""
                for chunk in generate_answer_stream(
                    query, results,
                    model=model_choice,
                    history=st.session_state.chat_messages,
                    temperature=temperature,
                ):
                    full_text += chunk
                    display = full_text.replace(r"\(", "$").replace(r"\)", "$")
                    display = display.replace("\\[", "$$\n").replace("\\]", "\n$$")
                    answer_div.markdown(
                        f'<div class="answer-container">{display}</div>',
                        unsafe_allow_html=True,
                    )

                answer = full_text

                # Export
                copy_text = answer + "\n\n---\n### Sources\n"
                seen = set()
                for r in results:
                    if r["source_name"] not in seen:
                        copy_text += f"- {r['source_name']} (p.{r.get('page', '?')}, rel: {r['score']:.0%})\n"
                        seen.add(r["source_name"])
                st.download_button(
                    "📋 Export Markdown",
                    data=copy_text,
                    file_name="minirag_answer.md",
                    mime="text/markdown",
                )

            with right:
                st.markdown("### 📚 Sources")
                by_source = defaultdict(list)
                for r in results:
                    by_source[r["source_name"]].append(r)

                for source_name, chunks_list in by_source.items():
                    best_rel = max(c["score"] for c in chunks_list)
                    table_count = sum(1 for c in chunks_list if c.get("is_table"))
                    source_display = source_name.rsplit(".", 1)[0][:60]
                    label = f"📄 {source_display} ({len(chunks_list)} chunks, {best_rel:.0%})"
                    if table_count:
                        label += f" · {table_count} 📊"
                    with st.expander(label, expanded=False):
                        for c in chunks_list:
                            is_table = c.get("is_table", False)
                            tag = "📊 TABLE" if is_table else "▸"
                            st.markdown(f"""
                            <div class="source-block">
                              <div class="relevance">{tag} Relevance {c['score']:.0%} · Page {c.get('page', '?')}</div>
                              <div class="text">{c['text']}</div>
                            </div>
                            """, unsafe_allow_html=True)

            # Save history
            timestamp = datetime.now().isoformat()
            st.session_state.history.append((query, answer, results, timestamp))
            st.session_state.chat_messages.append({"role": "user", "content": query})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            _save_history()

with tab2:
    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;">
          <div style="font-size:3rem;margin-bottom:16px;">📜</div>
          <p style="color:#5a6e88;">Your Q&A history will appear here.</p>
          <p style="color:#5a6e88;font-size:0.78rem;">History persists across restarts.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        total = len(st.session_state.history)
        for i, (q, a, sources, ts) in enumerate(reversed(st.session_state.history)):
            actual_idx = total - 1 - i
            label = f"{q[:80]}..."
            with st.expander(label, expanded=False):
                st.caption(f"🕒 {ts[:19]}")
                st.markdown(f"**Q:** {q}")
                st.markdown(a)
                st.caption("—" * 20)
                seen = set()
                for s in sources:
                    name = s.get("source_name") or s.get("paper_name", "?")
                    if name not in seen:
                        st.caption(f"📄 {name}")
                        seen.add(name)
                if st.button("🗑️ Remove", key=f"del_hist_{ts}"):
                    st.session_state.history.pop(actual_idx)
                    msg_idx = actual_idx * 2
                    if msg_idx + 1 < len(st.session_state.chat_messages):
                        del st.session_state.chat_messages[msg_idx:msg_idx + 2]
                    _save_history()
                    st.rerun()
