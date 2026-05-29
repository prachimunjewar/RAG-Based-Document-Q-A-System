mport sys
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
    page_title="DocMind — AI Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

/* ── BASE ── */
html, body, .stApp, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: #0e1117 !important;
    color: #e2e8f0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: #161b27 !important;
    border-right: 1px solid #2d3748 !important;
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #2d3748 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #1e2535 !important;
    border: 1.5px dashed #38bdf8 !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: #0e1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.85 !important;
}
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: #1e2535 !important;
    color: #64748b !important;
    border: 1px solid #2d3748 !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #38bdf8 !important;
}
[data-testid="stSidebar"] [data-testid="stSliderTrackFill"] {
    background: #38bdf8 !important;
}
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
}

/* ── MAIN HEADER ── */
.dm-header {
    background: linear-gradient(135deg, #1e2535, #1a1f2e);
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
}
.dm-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
}
.dm-title span { color: #38bdf8; }
.dm-subtitle {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 2px;
}
.dm-badges { display: flex; gap: 8px; }
.dm-badge {
    background: #1e2535;
    border: 1px solid #2d3748;
    color: #94a3b8;
    font-size: 0.7rem;
    padding: 4px 10px;
    border-radius: 20px;
    font-family: 'Fira Code', monospace;
}
.dm-badge.active {
    background: rgba(56,189,248,0.1);
    border-color: rgba(56,189,248,0.3);
    color: #38bdf8;
}

/* ── STAT CARDS ── */
.stat-grid { display: flex; gap: 12px; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: #161b27;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.stat-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.stat-icon.blue  { background: rgba(56,189,248,0.12); }
.stat-icon.indigo { background: rgba(129,140,248,0.12); }
.stat-icon.green { background: rgba(52,211,153,0.12); }
.stat-icon.amber { background: rgba(251,191,36,0.12); }
.stat-info {}
.stat-val { font-size: 1.2rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.stat-lbl { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 3px; }

/* ── HOW IT WORKS ── */
.how-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin-bottom: 1.5rem; }
.how-step {
    background: #161b27;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.how-step:hover { border-color: #38bdf8; }
.how-num {
    width: 30px; height: 30px;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 50%;
    color: #38bdf8;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.6rem;
}
.how-title { font-size: 0.82rem; font-weight: 600; color: #f1f5f9; margin-bottom: 3px; }
.how-desc  { font-size: 0.7rem; color: #64748b; line-height: 1.4; }
.how-tag {
    font-family: 'Fira Code', monospace;
    font-size: 0.6rem;
    color: #38bdf8;
    background: rgba(56,189,248,0.08);
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
    margin-top: 6px;
}

/* ── INFO BOX ── */
.info-box {
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
}
.info-box strong { color: #38bdf8; }

/* ── CHAT AREA ── */
.chat-doc-bar {
    background: #161b27;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.chat-doc-name {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.85rem; font-weight: 500; color: #e2e8f0;
}
.live-dot {
    width: 8px; height: 8px;
    background: #34d399;
    border-radius: 50%;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink {
    0%,100% { opacity:1; box-shadow: 0 0 4px #34d399; }
    50%      { opacity:0.4; box-shadow: none; }
}
.chat-meta {
    display: flex; gap: 1rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.72rem; color: #475569;
}

/* ── SUGGESTION BUTTONS ── */
div[data-testid="column"] .stButton > button {
    background: #1e2535 !important;
    border: 1px solid #2d3748 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 0.55rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.15s !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(56,189,248,0.08) !important;
    border-color: rgba(56,189,248,0.35) !important;
    color: #38bdf8 !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] textarea {
    background: #1e2535 !important;
    border: 1.5px solid #2d3748 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: #161b27 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    margin-bottom: 0.5rem !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: rgba(56,189,248,0.07) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* ── FOOTER ── */
.dm-footer {
    text-align: center;
    padding: 1.2rem;
    margin-top: 2rem;
    border-top: 1px solid #2d3748;
    font-size: 0.73rem;
    color: #475569;
    font-family: 'Fira Code', monospace;
}
.dm-footer span { color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.caption("RAG · FAISS · Groq LLaMA 3")
    st.divider()

    st.markdown("**📂 Upload Document**")
    uploaded = st.file_uploader("PDF or TXT file", type=["pdf", "txt"])

    st.divider()
    st.markdown("**⚙️ Retrieval Settings**")
    chunk_size = st.slider("Chunk size (words)", 200, 800, 500, 50)
    overlap    = st.slider("Overlap (words)", 20, 200, 100, 10)
    top_k      = st.slider("Chunks to retrieve", 2, 8, 4)

    if uploaded:
        st.divider()
        if st.button("⚡ Process Document", type="primary", use_container_width=True):
            with st.spinner("Extracting text..."):
                suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                doc = load_document(tmp_path)
                os.unlink(tmp_path)
            with st.spinner("Chunking..."):
                chunks = chunk_text(doc["full_text"], chunk_size, overlap)
            with st.spinner("Building vector index..."):
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
            st.success("✅ Ready! Ask your questions →")

    if st.session_state.doc_info:
        d = st.session_state.doc_info
        st.divider()
        st.markdown("**📊 Index Stats**")
        c1, c2 = st.columns(2)
        c1.metric("Pages", d["pages"])
        c2.metric("Chunks", d["chunks"])
        st.caption(f"⏱ Indexed in {d['embed_time']}s")
        st.caption(f"📄 {d['name']}")

    if st.session_state.chat:
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat = []
            if st.session_state.pipeline:
                st.session_state.pipeline.reset_memory()
            st.rerun()

    st.divider()
    st.markdown("""
    <div style='font-size:0.7rem;color:#475569;line-height:1.9;font-family:"Fira Code",monospace'>
    pypdf · sentence-transformers<br>
    faiss-cpu · groq · langchain<br>
    streamlit · python-dotenv
    </div>
    """, unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div class="dm-header">
  <div>
    <div class="dm-title">🧠 Doc<span>Mind</span></div>
    <div class="dm-subtitle">AI-powered document intelligence · Ask anything, get grounded answers</div>
  </div>
  <div class="dm-badges">
    <div class="dm-badge active">RAG</div>
    <div class="dm-badge active">FAISS</div>
    <div class="dm-badge">Groq</div>
    <div class="dm-badge">LLaMA 3</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────
if not st.session_state.pipeline:

    st.markdown("""
    <div class="info-box">
      👈 <strong>Upload a PDF or TXT</strong> in the sidebar, then click <strong>Process Document</strong> to start asking questions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### How it works")
    st.markdown("""
    <div class="how-grid">
      <div class="how-step">
        <div class="how-num">1</div>
        <div class="how-title">Upload</div>
        <div class="how-desc">PDF text extracted page by page</div>
        <div class="how-tag">pypdf</div>
      </div>
      <div class="how-step">
        <div class="how-num">2</div>
        <div class="how-title">Chunk</div>
        <div class="how-desc">Split into overlapping word windows</div>
        <div class="how-tag">custom</div>
      </div>
      <div class="how-step">
        <div class="how-num">3</div>
        <div class="how-title">Embed</div>
        <div class="how-desc">Dense vectors, runs on your machine</div>
        <div class="how-tag">MiniLM-L6</div>
      </div>
      <div class="how-step">
        <div class="how-num">4</div>
        <div class="how-title">Retrieve</div>
        <div class="how-desc">Cosine similarity, top-k results</div>
        <div class="how-tag">FAISS</div>
      </div>
      <div class="how-step">
        <div class="how-num">5</div>
        <div class="how-title">Answer</div>
        <div class="how-desc">Grounded answer with memory</div>
        <div class="how-tag">Groq LLaMA 3</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.info("⚡ **Sub-second retrieval** via FAISS vector search")
    c2.info("🔒 **Private** — embeddings run locally, nothing shared")
    c3.info("🆓 **Free** — Groq API + HuggingFace, zero cost")

else:
    d = st.session_state.doc_info

    # stat row
    name_display = d['name'][:22] + "…" if len(d['name']) > 22 else d['name']
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-icon blue">📄</div>
        <div class="stat-info">
          <div class="stat-val">{name_display}</div>
          <div class="stat-lbl">Document</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon indigo">📑</div>
        <div class="stat-info">
          <div class="stat-val">{d['pages']}</div>
          <div class="stat-lbl">Pages</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon green">🧩</div>
        <div class="stat-info">
          <div class="stat-val">{d['chunks']}</div>
          <div class="stat-lbl">Chunks</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon amber">⚡</div>
        <div class="stat-info">
          <div class="stat-val">{d['embed_time']}s</div>
          <div class="stat-lbl">Indexed</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # active doc bar
    st.markdown(f"""
    <div class="chat-doc-bar">
      <div class="chat-doc-name">
        <div class="live-dot"></div>
        {d['name']} — ready for questions
      </div>
      <div class="chat-meta">
        top-k: {top_k} &nbsp;·&nbsp; chunks: {d['chunks']} &nbsp;·&nbsp; model: llama3-8b
      </div>
    </div>
    """, unsafe_allow_html=True)

    # chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # suggestions
    if not st.session_state.chat:
        st.markdown("**💡 Try a question:**")
        suggestions = [
            "Summarize this document",
            "What are the main topics?",
            "What conclusions are made?",
            "List the key findings",
        ]
        cols = st.columns(4)
        for col, q in zip(cols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state._prefill = q
                st.rerun()
        st.markdown("")

    prefill  = st.session_state.pop("_prefill", "")
    question = st.chat_input("Ask anything about your document...")
    if not question and prefill:
        question = prefill

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating answer..."):
                answer = st.session_state.pipeline.ask(question, top_k=top_k)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="dm-footer">
  🧠 <span>DocMind</span> &nbsp;·&nbsp;
  FAISS · all-MiniLM-L6-v2 · llama3-8b-8192 · Streamlit
</div>
""", unsafe_allow_html=True)
