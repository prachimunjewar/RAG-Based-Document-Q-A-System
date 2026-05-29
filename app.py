import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import tempfile
import time

from utils.pdf_loader import load_document
from utils.chunker import chunk_text
from utils.vector_store import VectorStore
from utils.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 0 !important; }

/* ── HEADER ── */
.dm-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 1.4rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin: -1rem -1rem 0 -1rem;
}
.dm-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.5px;
}
.dm-logo span { color: #7c6dfa; }
.dm-badge {
    background: rgba(124,109,250,0.2);
    border: 1px solid rgba(124,109,250,0.4);
    color: #a99ffe;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.dm-nav {
    display: flex;
    gap: 1.5rem;
    align-items: center;
}
.dm-nav a {
    color: rgba(255,255,255,0.55);
    font-size: 0.85rem;
    text-decoration: none;
    font-weight: 500;
    transition: color .2s;
}
.dm-nav a:hover { color: #fff; }

/* ── HERO ── */
.dm-hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 60%, #1a1a2e 100%);
    padding: 3.5rem 2.5rem 2.5rem;
    text-align: center;
    margin: 0 -1rem;
    position: relative;
    overflow: hidden;
}
.dm-hero::before {
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(124,109,250,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.dm-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #fff;
    margin-bottom: 0.6rem;
    line-height: 1.15;
}
.dm-hero h1 span { color: #7c6dfa; }
.dm-hero p {
    color: rgba(255,255,255,0.6);
    font-size: 1rem;
    max-width: 520px;
    margin: 0 auto 1.6rem;
    line-height: 1.6;
}
.dm-pills {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.dm-pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.7);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 5px 14px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #13111f !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] * { color: #c9c4e8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #fff !important; }

.sidebar-section {
    background: rgba(124,109,250,0.08);
    border: 1px solid rgba(124,109,250,0.2);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7c6dfa !important;
    margin-bottom: 0.4rem;
}

/* ── STAT CARDS ── */
.stat-row {
    display: flex;
    gap: 10px;
    margin: 1rem 0;
}
.stat-card {
    flex: 1;
    background: rgba(124,109,250,0.1);
    border: 1px solid rgba(124,109,250,0.25);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    text-align: center;
}
.stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #a99ffe;
}
.stat-lbl {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── HOW IT WORKS ── */
.how-card {
    background: #16132b;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin: 1.5rem 0;
}
.how-step {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 1rem;
}
.how-step:last-child { margin-bottom: 0; }
.how-num {
    min-width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #7c6dfa, #5b4de0);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
}
.how-text strong { color: #c9c4e8; font-size: 0.88rem; }
.how-text p { color: rgba(255,255,255,0.45); font-size: 0.78rem; margin: 2px 0 0; }

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 0.4rem !important;
}

/* ── SUGGESTIONS ── */
.stButton > button {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all .2s !important;
}
div[data-testid="column"] .stButton > button {
    background: #1e1a35 !important;
    border: 1px solid rgba(124,109,250,0.35) !important;
    color: #a99ffe !important;
    width: 100% !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(124,109,250,0.2) !important;
    border-color: #7c6dfa !important;
    color: #fff !important;
}

/* ── PRIMARY BUTTON ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c6dfa, #5b4de0) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] textarea {
    background: #1e1a35 !important;
    border: 1px solid rgba(124,109,250,0.3) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #7c6dfa !important;
    box-shadow: 0 0 0 2px rgba(124,109,250,0.2) !important;
}

/* ── SLIDERS ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #7c6dfa !important;
}

/* ── FOOTER ── */
.dm-footer {
    background: #0a0814;
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 1.2rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 2rem -1rem -1rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.dm-footer-left {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.3);
}
.dm-footer-left span { color: #7c6dfa; font-weight: 600; }
.dm-footer-links {
    display: flex;
    gap: 1.2rem;
}
.dm-footer-links a {
    color: rgba(255,255,255,0.3);
    font-size: 0.75rem;
    text-decoration: none;
}
.dm-footer-links a:hover { color: #7c6dfa; }
.dm-footer-tag {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.2);
    font-family: monospace;
}

/* ── SUCCESS / INFO ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}

/* ── DIVIDER ── */
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-header">
  <div class="dm-logo">Doc<span>Mind</span></div>
  <div class="dm-nav">
    <span class="dm-badge">RAG · FAISS · Groq</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("<p style='font-size:0.78rem;color:rgba(255,255,255,0.4);margin-top:-8px'>RAG-powered document intelligence</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-label">📂 Upload document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf", "txt"], label_visibility="collapsed")

    st.divider()
    st.markdown('<div class="sidebar-label">⚙️ Retrieval settings</div>', unsafe_allow_html=True)
    chunk_size = st.slider("Chunk size (words)", 200, 800, 500, 50)
    overlap    = st.slider("Overlap (words)", 20, 200, 100, 10)
    top_k      = st.slider("Chunks to retrieve", 2, 8, 4)

    if uploaded:
        st.divider()
        if st.button("⚡ Process Document", type="primary", use_container_width=True):
            with st.spinner("Extracting text…"):
                suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                doc = load_document(tmp_path)
                os.unlink(tmp_path)

            with st.spinner(f"Chunking into {chunk_size}-word pieces…"):
                chunks = chunk_text(doc["full_text"], chunk_size, overlap)

            with st.spinner(f"Embedding {len(chunks)} chunks…"):
                t0 = time.time()
                vs = VectorStore()
                vs.build(chunks)
                elapsed = round(time.time() - t0, 1)

            st.session_state.pipeline = RAGPipeline(vs)
            st.session_state.chat     = []
            st.session_state.doc_info = {
                "name": doc["file_name"],
                "pages": doc["total_pages"],
                "chunks": len(chunks),
                "embed_time": elapsed,
            }
            st.success("✅ Document ready!")

    if st.session_state.doc_info:
        d = st.session_state.doc_info
        st.divider()
        st.markdown('<div class="sidebar-label">📊 Document stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="stat-val">{d['pages']}</div><div class="stat-lbl">Pages</div></div>
          <div class="stat-card"><div class="stat-val">{d['chunks']}</div><div class="stat-lbl">Chunks</div></div>
          <div class="stat-card"><div class="stat-val">{d['embed_time']}s</div><div class="stat-lbl">Indexed</div></div>
        </div>
        <p style='font-size:0.72rem;color:rgba(255,255,255,0.3);margin-top:4px'>📄 {d['name']}</p>
        """, unsafe_allow_html=True)

    if st.session_state.chat:
        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat = []
            if st.session_state.pipeline:
                st.session_state.pipeline.reset_memory()
            st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:0.7rem;color:rgba(255,255,255,0.2);line-height:1.7'>
      <b style='color:rgba(255,255,255,0.4)'>Stack</b><br>
      🔷 FAISS vector index<br>
      🟣 sentence-transformers<br>
      ⚡ Groq LLaMA 3 (free)<br>
      🔵 Streamlit UI
    </div>
    """, unsafe_allow_html=True)

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
if not st.session_state.pipeline:

    st.markdown("""
    <div class="dm-hero">
      <h1>Ask anything about<br><span>your documents</span></h1>
      <p>Upload a PDF or text file and get instant, accurate answers powered by semantic search and a free LLM — no data leaves your machine.</p>
      <div class="dm-pills">
        <span class="dm-pill">🔍 Semantic retrieval</span>
        <span class="dm-pill">🧩 FAISS vector index</span>
        <span class="dm-pill">💬 Multi-turn memory</span>
        <span class="dm-pill">⚡ Groq LLaMA 3</span>
        <span class="dm-pill">🔒 Runs locally</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("&nbsp;")

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown("#### How it works")
        st.markdown("""
        <div class="how-card">
          <div class="how-step">
            <div class="how-num">1</div>
            <div class="how-text"><strong>Upload your PDF</strong><p>Text is extracted page by page using pypdf</p></div>
          </div>
          <div class="how-step">
            <div class="how-num">2</div>
            <div class="how-text"><strong>Smart chunking</strong><p>Text split into overlapping windows — no lost context at boundaries</p></div>
          </div>
          <div class="how-step">
            <div class="how-num">3</div>
            <div class="how-text"><strong>Local embeddings</strong><p>all-MiniLM-L6-v2 runs on your machine — no API calls, free forever</p></div>
          </div>
          <div class="how-step">
            <div class="how-num">4</div>
            <div class="how-text"><strong>FAISS retrieval</strong><p>Sub-millisecond cosine search returns the most relevant chunks</p></div>
          </div>
          <div class="how-step">
            <div class="how-num">5</div>
            <div class="how-text"><strong>Groq LLaMA 3</strong><p>Retrieved context + your question → fast, grounded answer with memory</p></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Get started")
        st.info("👈 Upload a PDF or TXT file in the sidebar, then click **Process Document**.")
        st.markdown("#### Tips")
        st.markdown("""
- Use **text-based PDFs** (not scanned images)
- Larger chunk sizes = more context per answer
- Increase **overlap** for dense technical docs
- Try the **suggested questions** that appear after loading
        """)
        st.markdown("#### Free APIs used")
        st.markdown("""
| Component | Provider | Cost |
|---|---|---|
| Embeddings | HuggingFace | Free |
| LLM | Groq LLaMA 3 | Free tier |
| Vector search | FAISS (local) | Free |
        """)

else:
    st.markdown("&nbsp;")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.chat:
        d = st.session_state.doc_info
        st.markdown(f"#### 📄 {d['name']} is ready")
        st.markdown("**Try a suggested question:**")
        suggestions = [
            "Summarize this document",
            "What are the main topics covered?",
            "What conclusions are made?",
            "List the key findings",
        ]
        cols = st.columns(len(suggestions))
        for col, q in zip(cols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state._prefill = q
                st.rerun()
        st.markdown("---")

    prefill  = st.session_state.pop("_prefill", "")
    question = st.chat_input("Ask anything about your document…")
    if not question and prefill:
        question = prefill

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer…"):
                answer = st.session_state.pipeline.ask(question, top_k=top_k)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-footer">
  <div class="dm-footer-left">Built with ❤️ using <span>DocMind</span> · RAG + FAISS + Groq</div>
  <div class="dm-footer-tag">v1.0 · all-MiniLM-L6-v2 · llama3-8b-8192</div>
  <div class="dm-footer-links">
    <a href="https://console.groq.com" target="_blank">Groq Console</a>
    <a href="https://streamlit.io" target="_blank">Streamlit</a>
    <a href="https://github.com/facebookresearch/faiss" target="_blank">FAISS</a>
  </div>
</div>
""", unsafe_allow_html=True)
