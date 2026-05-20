"""
PaperRAG — 论文知识检索库 · 北极甜虾 Design Edition
Streamlit UI: dark academic theme, enhanced paper cards, polished UX
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

# ═══════════════════════════════════════════
#  Custom CSS — Dark Academic Theme
# ═══════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ── Root colors ── */
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

  /* ── Global overrides ── */
  .stApp {
    background: var(--bg-primary);
  }
  .main .block-container {
    padding-top: 1.5rem;
    max-width: 100%;
  }

  /* Typography */
  h1, h2, h3, h4, h5, h6 {
    font-family: 'Source Serif 4', 'Noto Sans SC', serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
  }
  p, li, label, div, span {
    font-family: 'Noto Sans SC', sans-serif;
  }
  code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
  }
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Noto Sans SC', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase;
  }

  /* ── Inputs ── */
  .stTextInput > div > div > input,
  .stSelectbox > div > div,
  .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem;
  }
  .stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'Noto Sans SC', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border: none !important;
    color: white !important;
  }
  .stButton > button[kind="primary"]:hover {
    filter: brightness(1.15);
    box-shadow: 0 0 16px var(--accent-dim);
  }
  .stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
  }

  /* ── Metrics ── */
  [data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
  }
  [data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
  }
  [data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--accent) !important;
  }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
  }
  .streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
    padding: 16px !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 20px !important;
    color: var(--text-muted) !important;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }

  /* ── Spinner ── */
  .stSpinner > div {
    border-color: var(--accent) !important;
  }

  /* ── Paper card in sidebar ── */
  .paper-card {
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    transition: all 0.15s ease;
    cursor: default;
  }
  .paper-card:hover {
    border-color: var(--accent);
    background: var(--bg-hover);
  }
  .paper-card .title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.8rem;
  }
  .paper-card .meta {
    color: var(--text-muted);
    font-size: 0.68rem;
    margin-top: 2px;
  }

  /* ── Source block ── */
  .source-block {
    padding: 12px 16px;
    margin: 8px 0;
    border-left: 3px solid var(--accent);
    background: var(--bg-card);
    border-radius: 0 6px 6px 0;
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.6;
  }
  .source-block .relevance {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    margin-bottom: 6px;
  }
  .source-block .text {
    color: var(--text-primary);
  }

  /* ── Answer area ── */
  .answer-container {
    padding: 20px 24px;
    background: var(--bg-card);
    border-radius: 8px;
    border: 1px solid var(--border);
    min-height: 200px;
    line-height: 1.8;
    font-size: 0.9rem;
  }

  /* ── Footer ── */
  .footer-text {
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    text-align: center;
    padding: 16px 0 4px;
  }

  /* Divider override */
  hr {
    border-color: var(--border) !important;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  History persistence helpers
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
        Paper<span style="color:#5b8def;">RAG</span>
      </span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#5a6e88;letter-spacing:0.15em;">
        ACADEMIC EDITION
      </span>
    </div>
    <p style="color:#8899b4;font-size:0.82rem;margin-top:0;">
      Your papers, your knowledge. Ask questions, get cited answers.
    </p>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════
with st.sidebar:
    # Repository badge
    collection = get_collection()
    chunk_count = get_chunk_count(collection)

    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-bottom:20px;">
      <div style="flex:1;background:#1a2230;border:1px solid rgba(99,130,180,.15);border-radius:8px;padding:12px 14px;text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#5b8def;">{chunk_count:,}</div>
        <div style="font-size:0.62rem;color:#5a6e88;letter-spacing:.06em;margin-top:2px;">CHUNKS</div>
      </div>
      <div style="flex:1;background:#1a2230;border:1px solid rgba(99,130,180,.15);border-radius:8px;padding:12px 14px;text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#4ec9a8;">{len(st.session_state.get('paper_list') or get_paper_names(collection)):,}</div>
        <div style="font-size:0.62rem;color:#5a6e88;letter-spacing:.06em;margin-top:2px;">PAPERS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key ──
    with st.expander("🔑 API Key", expanded=not os.getenv("DEEPSEEK_API_KEY")):
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            placeholder="sk-xxxxxxxxxxxxxxxx",
            label_visibility="collapsed",
        )
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
        st.caption("[platform.deepseek.com](https://platform.deepseek.com)")
        if not api_key:
            st.warning("API key required")

    st.divider()

    # ── Upload ──
    st.markdown("### 📤 Upload Papers")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Index Papers", type="primary", use_container_width=True):
            with st.spinner("Parsing + embedding..."):
                os.makedirs("data/papers", exist_ok=True)

                if "paper_list" not in st.session_state or st.session_state.paper_list is None:
                    st.session_state.paper_list = get_paper_names(collection)
                existing = set(st.session_state.paper_list)

                skipped = [uf.name for uf in uploaded_files if uf.name in existing]
                new_files = [uf for uf in uploaded_files if uf.name not in existing]

                if skipped:
                    st.info(f"⏭️ Skipped: {', '.join(skipped)}")

                if not new_files:
                    st.warning("All already indexed.")
                else:
                    papers = []
                    failed = []
                    for uf in new_files:
                        path = os.path.join("data/papers", uf.name)
                        with open(path, "wb") as f:
                            f.write(uf.getbuffer())
                        text = load_pdf_text(path)
                        if text.strip():
                            papers.append({"name": uf.name, "text": text})
                        else:
                            failed.append(uf.name)

                    if not papers:
                        st.error("No extractable text found.")
                        for name in failed:
                            st.warning(f"❌ `{name}` — scanned PDF?")
                    else:
                        progress = st.progress(0, "Chunking...")
                        chunks = chunk_papers(papers)
                        progress.progress(30, "Embedding...")
                        texts = [c["text"] for c in chunks]
                        embeddings = embed_texts(texts)
                        progress.progress(80, "Storing...")
                        add_chunks(collection, chunks, embeddings)
                        progress.progress(100, "Done!")
                        st.success(f"✅ {len(papers)} papers · {len(chunks)} chunks")
                        st.session_state.paper_list = None
                        st.rerun()

    # ── Paper Library ──
    if chunk_count > 0:
        st.divider()
        st.markdown("### 📚 Library")

        if st.session_state.get("paper_list") is None or get_chunk_count(collection) != chunk_count:
            st.session_state.paper_list = get_paper_names(collection)
            st.session_state.paper_meta = {}

        paper_names = st.session_state.paper_list

        if "paper_meta" not in st.session_state:
            st.session_state.paper_meta = {}
        for p in paper_names:
            if p not in st.session_state.paper_meta:
                path = os.path.join("data/papers", p)
                if os.path.exists(path):
                    st.session_state.paper_meta[p] = load_pdf_meta(path)
                else:
                    st.session_state.paper_meta[p] = {"title": "", "author": "", "year": ""}

        if len(paper_names) > 5:
            filter_text = st.text_input(
                "Filter", placeholder=f"Search {len(paper_names)} papers...",
                label_visibility="collapsed",
            )
            if filter_text:
                paper_names = [p for p in paper_names if filter_text.lower() in p.lower()]

        for p in paper_names[:20]:
            meta = st.session_state.paper_meta.get(p, {})
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

        if len(paper_names) > 20:
            st.caption(f"... and {len(paper_names) - 20} more")

        # ── Delete papers ──
        st.divider()
        st.markdown("### 🗑️ Remove")
        selected = []
        for p in paper_names:
            if st.checkbox(p, key=f"del_{p}"):
                selected.append(p)
        if selected:
            if st.button(f"Delete {len(selected)} paper(s)", type="primary", use_container_width=True):
                total = 0
                for p in selected:
                    total += delete_paper(collection, p)
                st.success(f"Removed {total} chunks")
                st.session_state.paper_list = None
                st.rerun()

    # ── Model Settings ──
    st.divider()
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Model",
        ["deepseek-chat", "deepseek-reasoner"],
        index=0,
        label_visibility="collapsed",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0, value=0.3, step=0.1,
        label_visibility="collapsed",
    )
    st.caption(f"`{model_choice}` · temp `{temperature}`")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.chat_messages = []
        _save_history()
        st.rerun()

    st.markdown('<p class="footer-text">Chroma + BGE + DeepSeek</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  Main Area
# ═══════════════════════════════════════════
tab1, tab2 = st.tabs(["🔍 Ask", "📜 History"])

with tab1:
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
        with st.spinner("Searching papers + generating answer..."):
            results = retrieve(collection, query, top_k=top_k * 3, max_per_paper=3)
            results = rerank(query, results, top_k=top_k)

        if not results:
            st.warning("No relevant passages found. Try rephrasing or upload more papers.")
        else:
            left, right = st.columns([3, 2])

            with left:
                st.markdown("### 📝 Answer")
                answer_div = st.empty()
                full_text = ""
                for chunk in generate_answer_stream(
                    query, results,
                    model=model_choice,
                    history=st.session_state.chat_messages,
                    temperature=temperature,
                ):
                    full_text += chunk
                    display = full_text.replace("\\(", "$").replace("\\)", "$")
                    display = display.replace("\\[", "$$\n").replace("\\]", "\n$$")
                    answer_div.markdown(f'<div class="answer-container">{display}</div>', unsafe_allow_html=True)

                answer = full_text

                # Export
                copy_text = answer + "\n\n---\n### Sources\n"
                seen = set()
                for r in results:
                    if r["paper_name"] not in seen:
                        copy_text += f"- {r['paper_name']} (p.{r.get('page', '?')}, rel: {r['score']:.0%})\n"
                        seen.add(r["paper_name"])
                st.download_button(
                    "📋 Export Markdown",
                    data=copy_text,
                    file_name="paperrag_answer.md",
                    mime="text/markdown",
                )

            with right:
                st.markdown("### 📚 Sources")
                by_paper = defaultdict(list)
                for r in results:
                    by_paper[r["paper_name"]].append(r)

                for paper_name, chunks_list in by_paper.items():
                    best_rel = max(c["score"] for c in chunks_list)
                    paper_display = paper_name.rsplit(".", 1)[0][:60]
                    with st.expander(
                        f"📄 {paper_display} ({len(chunks_list)} chunks, {best_rel:.0%})",
                        expanded=False,
                    ):
                        for i, c in enumerate(chunks_list):
                            st.markdown(f"""
                            <div class="source-block">
                              <div class="relevance">▸ Relevance {c['score']:.0%} · Page {c.get('page', '?')}</div>
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
                    if s["paper_name"] not in seen:
                        st.caption(f"📄 {s['paper_name']}")
                        seen.add(s["paper_name"])
                if st.button("🗑️ Remove", key=f"del_hist_{ts}"):
                    st.session_state.history.pop(actual_idx)
                    msg_idx = actual_idx * 2
                    if msg_idx + 1 < len(st.session_state.chat_messages):
                        del st.session_state.chat_messages[msg_idx:msg_idx + 2]
                    _save_history()
                    st.rerun()
