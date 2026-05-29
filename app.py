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
    page_title="DocMind — AI Document Intelligence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #F7F4EF !important;
    color: #1A1208;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ══════════════════════════════════════
   SIDEBAR
══════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #1A1208 !important;
    border-right: none !important;
    width: 300px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebar"] * {
    color: #E8E0D0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F7F4EF !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(232,224,208,0.1) !important;
    margin: 1rem 0 !important;
}
[data-testid="stSidebar"] .stSlider label {
    font-size: 0.78rem !important;
    color: rgba(232,224,208,0.6) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #C9A84C !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
    background: #C9A84C !important;
}

/* file uploader in sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed rgba(201,168,76,0.4) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: rgba(232,224,208,0.7) !important;
}

/* sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #C9A84C !important;
    color: #1A1208 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    letter-spacing: 0.02em;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #E0BC5A !important;
    transform: translateY(-1px) !important;
}

/* clear button */
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: transparent !important;
    color: rgba(232,224,208,0.4) !important;
    border: 1px solid rgba(232,224,208,0.15) !important;
    font-size: 0.78rem !important;
}
[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
    color: rgba(232,224,208,0.8) !important;
    border-color: rgba(232,224,208,0.3) !important;
    transform: none !important;
}

/* success/info alerts */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(201,168,76,0.1) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
}

/* ══════════════════════════════════════
   HEADER
══════════════════════════════════════ */
.dm-topbar {
    background: #1A1208;
    padding: 0 2.5rem;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    border-bottom: 1px solid rgba(201,168,76,0.2);
}
.dm-wordmark {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #F7F4EF;
    letter-spacing: -0.01em;
}
.dm-wordmark em {
    color: #C9A84C;
    font-style: italic;
}
.dm-topbar-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.dm-version {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: rgba(247,244,239,0.3);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.dm-status-dot {
    width: 7px;
    height: 7px;
    background: #4CAF50;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.dm-status {
    font-size: 0.72rem;
    color: rgba(247,244,239,0.45);
}

/* ══════════════════════════════════════
   HERO / LANDING
══════════════════════════════════════ */
.dm-hero-wrap {
    padding: 5rem 3rem 3rem;
    max-width: 820px;
}
.dm-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #C9A84C;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.dm-eyebrow::before {
    content: '';
    display: inline-block;
    width: 28px;
    height: 1px;
    background: #C9A84C;
}
.dm-hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.8rem;
    color: #1A1208;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 1.5rem;
}
.dm-hero-title em {
    font-style: italic;
    color: #8B6914;
}
.dm-hero-desc {
    font-size: 1.05rem;
    color: #5C4A28;
    line-height: 1.7;
    max-width: 520px;
    margin-bottom: 2.5rem;
    font-weight: 300;
}
.dm-feature-row {
    display: flex;
    gap: 0;
    border: 1px solid rgba(26,18,8,0.12);
    border-radius: 10px;
    overflow: hidden;
    max-width: 620px;
    background: #fff;
}
.dm-feature {
    flex: 1;
    padding: 1rem 1.2rem;
    border-right: 1px solid rgba(26,18,8,0.08);
}
.dm-feature:last-child { border-right: none; }
.dm-feature-icon {
    font-size: 1.1rem;
    margin-bottom: 4px;
}
.dm-feature-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #1A1208;
    letter-spacing: 0.02em;
}
.dm-feature-sub {
    font-size: 0.68rem;
    color: #9C8060;
    margin-top: 2px;
}

/* HOW IT WORKS SECTION */
.dm-section {
    padding: 3rem 3rem;
    border-top: 1px solid rgba(26,18,8,0.08);
}
.dm-section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #C9A84C;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}
.dm-steps {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0;
    border: 1px solid rgba(26,18,8,0.1);
    border-radius: 10px;
    overflow: hidden;
    background: #fff;
    max-width: 860px;
}
.dm-step {
    padding: 1.5rem 1.2rem;
    border-right: 1px solid rgba(26,18,8,0.08);
    position: relative;
}
.dm-step:last-child { border-right: none; }
.dm-step-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: rgba(201,168,76,0.25);
    line-height: 1;
    margin-bottom: 8px;
}
.dm-step-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #1A1208;
    margin-bottom: 4px;
}
.dm-step-desc {
    font-size: 0.72rem;
    color: #9C8060;
    line-height: 1.5;
}
.dm-step-tech {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #C9A84C;
    margin-top: 6px;
    background: rgba(201,168,76,0.08);
    padding: 2px 6px;
    border-radius: 3px;
    display: inline-block;
}

/* ══════════════════════════════════════
   CHAT AREA
══════════════════════════════════════ */
.dm-chat-header {
    padding: 1.5rem 2.5rem 0.5rem;
    border-bottom: 1px solid rgba(26,18,8,0.08);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.dm-doc-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #fff;
    border: 1px solid rgba(26,18,8,0.1);
    border-radius: 20px;
    padding: 5px 14px 5px 8px;
    font-size: 0.78rem;
    color: #5C4A28;
}
.dm-doc-pill-icon {
    width: 22px;
    height: 22px;
    background: #C9A84C;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
}
.dm-stat-inline {
    display: flex;
    gap: 1.5rem;
}
.dm-stat-inline-item {
    text-align: right;
}
.dm-stat-inline-val {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: #1A1208;
}
.dm-stat-inline-lbl {
    font-size: 0.65rem;
    color: #9C8060;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* suggestions */
.dm-suggestions {
    padding: 1.5rem 2.5rem 0.5rem;
}
.dm-suggestions-label {
    font-size: 0.7rem;
    color: #9C8060;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.75rem;
}

div[data-testid="column"] .stButton > button {
    background: #fff !important;
    border: 1px solid rgba(26,18,8,0.12) !important;
    color: #5C4A28 !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    padding: 0.55rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.15s !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: #1A1208 !important;
    color: #F7F4EF !important;
    border-color: #1A1208 !important;
    transform: none !important;
}

/* chat messages */
.stChatMessage {
    padding: 0 2.5rem !important;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.6rem 0 !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: rgba(201,168,76,0.06) !important;
}

/* chat input */
[data-testid="stChatInput"] {
    padding: 1rem 2.5rem !important;
    background: #fff !important;
    border-top: 1px solid rgba(26,18,8,0.08) !important;
    position: sticky;
    bottom: 0;
}
[data-testid="stChatInput"] textarea {
    background: #F7F4EF !important;
    border: 1.5px solid rgba(26,18,8,0.15) !important;
    border-radius: 8px !important;
    color: #1A1208 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #C9A84C !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.12) !important;
}

/* spinner */
[data-testid="stSpinner"] { color: #C9A84C !important; }

/* ══════════════════════════════════════
   FOOTER
══════════════════════════════════════ */
.dm-footer {
    background: #1A1208;
    padding: 1.5rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 4rem;
}
.dm-footer-brand {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: rgba(247,244,239,0.5);
}
.dm-footer-brand em { color: #C9A84C; font-style: italic; }
.dm-footer-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: rgba(247,244,239,0.2);
    letter-spacing: 0.06em;
}
.dm-footer-links {
    display: flex;
    gap: 1.5rem;
}
.dm-footer-links a {
    font-size: 0.75rem;
    color: rgba(247,244,239,0.3);
    text-decoration: none;
    transition: color 0.15s;
}
.dm-footer-links a:hover { color: #C9A84C; }

/* alerts */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

# ── TOPBAR ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-topbar">
  <div class="dm-wordmark">Doc<em>Mind</em></div>
  <div class="dm-topbar-right">
    <span class="dm-status"><span class="dm-status-dot"></span>All systems operational</span>
    <span class="dm-version">v1.0 · RAG + FAISS + Groq</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.8rem 1.5rem 0.5rem;">
      <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;color:#F7F4EF;margin-bottom:4px">
        Document
      </div>
      <div style="font-size:0.72rem;color:rgba(232,224,208,0.4);letter-spacing:0.04em">
        Upload a PDF or TXT file to begin
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 0 1.5rem;'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf", "txt"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="padding: 0 1.5rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:rgba(201,168,76,0.8);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1rem">
        Retrieval Settings
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        chunk_size = st.slider("Chunk size (words)", 200, 800, 500, 50)
        overlap    = st.slider("Overlap (words)", 20, 200, 100, 10)
        top_k      = st.slider("Chunks to retrieve", 2, 8, 4)

    if uploaded:
        st.divider()
        st.markdown("<div style='padding:0 1.5rem 1rem;'>", unsafe_allow_html=True)
        if st.button("⚡  Process Document", type="primary", use_container_width=True):
            with st.spinner("Extracting text…"):
                suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                doc = load_document(tmp_path)
                os.unlink(tmp_path)

            with st.spinner(f"Chunking…"):
                chunks = chunk_text(doc["full_text"], chunk_size, overlap)

            with st.spinner(f"Building vector index…"):
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
            st.success("Ready — ask your first question →")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.doc_info:
        d = st.session_state.doc_info
        st.divider()
        st.markdown(f"""
        <div style="padding: 0 1.5rem;">
          <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:rgba(201,168,76,0.8);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1rem">Indexed Document</div>
          <div style="font-size:0.78rem;color:rgba(232,224,208,0.6);margin-bottom:1rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">📄 {d['name']}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:1rem">
            <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:0.6rem 0.5rem;text-align:center">
              <div style="font-family:'DM Mono',monospace;font-size:1.1rem;font-weight:500;color:#C9A84C">{d['pages']}</div>
              <div style="font-size:0.6rem;color:rgba(232,224,208,0.35);text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">pages</div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:0.6rem 0.5rem;text-align:center">
              <div style="font-family:'DM Mono',monospace;font-size:1.1rem;font-weight:500;color:#C9A84C">{d['chunks']}</div>
              <div style="font-size:0.6rem;color:rgba(232,224,208,0.35);text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">chunks</div>
            </div>
            <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:0.6rem 0.5rem;text-align:center">
              <div style="font-family:'DM Mono',monospace;font-size:1.1rem;font-weight:500;color:#C9A84C">{d['embed_time']}s</div>
              <div style="font-size:0.6rem;color:rgba(232,224,208,0.35);text-transform:uppercase;letter-spacing:0.05em;margin-top:2px">indexed</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.chat:
        st.divider()
        st.markdown("<div style='padding:0 1.5rem 1rem;'>", unsafe_allow_html=True)
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat = []
            if st.session_state.pipeline:
                st.session_state.pipeline.reset_memory()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="padding: 0 1.5rem 1.5rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:rgba(232,224,208,0.2);line-height:2;letter-spacing:0.04em">
        pypdf · sentence-transformers<br>
        faiss-cpu · groq · langchain<br>
        streamlit · python-dotenv
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
if not st.session_state.pipeline:

    # HERO
    st.markdown("""
    <div class="dm-hero-wrap">
      <div class="dm-eyebrow">Retrieval-Augmented Generation</div>
      <h1 class="dm-hero-title">Your documents,<br><em>finally answerable.</em></h1>
      <p class="dm-hero-desc">Upload any PDF and ask questions in plain language. DocMind retrieves exactly the right passages and generates precise, grounded answers — no hallucinations, no guessing.</p>
      <div class="dm-feature-row">
        <div class="dm-feature">
          <div class="dm-feature-icon">⚡</div>
          <div class="dm-feature-label">Sub-second</div>
          <div class="dm-feature-sub">FAISS retrieval</div>
        </div>
        <div class="dm-feature">
          <div class="dm-feature-icon">🔒</div>
          <div class="dm-feature-label">Runs locally</div>
          <div class="dm-feature-sub">Embeddings on-device</div>
        </div>
        <div class="dm-feature">
          <div class="dm-feature-icon">💬</div>
          <div class="dm-feature-label">Multi-turn</div>
          <div class="dm-feature-sub">Conversation memory</div>
        </div>
        <div class="dm-feature">
          <div class="dm-feature-icon">🆓</div>
          <div class="dm-feature-label">100% free</div>
          <div class="dm-feature-sub">Groq + HuggingFace</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # HOW IT WORKS
    st.markdown("""
    <div class="dm-section">
      <div class="dm-section-label">How it works</div>
      <div class="dm-steps">
        <div class="dm-step">
          <div class="dm-step-num">01</div>
          <div class="dm-step-title">Extract</div>
          <div class="dm-step-desc">Text pulled from every page of your PDF</div>
          <div class="dm-step-tech">pypdf</div>
        </div>
        <div class="dm-step">
          <div class="dm-step-num">02</div>
          <div class="dm-step-title">Chunk</div>
          <div class="dm-step-desc">Split into overlapping 500-word windows</div>
          <div class="dm-step-tech">custom</div>
        </div>
        <div class="dm-step">
          <div class="dm-step-num">03</div>
          <div class="dm-step-title">Embed</div>
          <div class="dm-step-desc">384-dim dense vectors, runs on your machine</div>
          <div class="dm-step-tech">all-MiniLM-L6-v2</div>
        </div>
        <div class="dm-step">
          <div class="dm-step-num">04</div>
          <div class="dm-step-title">Retrieve</div>
          <div class="dm-step-desc">Cosine similarity finds top-k relevant chunks</div>
          <div class="dm-step-tech">FAISS</div>
        </div>
        <div class="dm-step">
          <div class="dm-step-num">05</div>
          <div class="dm-step-title">Answer</div>
          <div class="dm-step-desc">Context + memory → grounded LLM response</div>
          <div class="dm-step-tech">Groq LLaMA 3</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # CHAT HEADER
    d = st.session_state.doc_info
    st.markdown(f"""
    <div class="dm-chat-header">
      <div class="dm-doc-pill">
        <div class="dm-doc-pill-icon">📄</div>
        {d['name']}
      </div>
      <div class="dm-stat-inline">
        <div class="dm-stat-inline-item">
          <div class="dm-stat-inline-val">{d['pages']}</div>
          <div class="dm-stat-inline-lbl">Pages</div>
        </div>
        <div class="dm-stat-inline-item">
          <div class="dm-stat-inline-val">{d['chunks']}</div>
          <div class="dm-stat-inline-lbl">Chunks</div>
        </div>
        <div class="dm-stat-inline-item">
          <div class="dm-stat-inline-val">{d['embed_time']}s</div>
          <div class="dm-stat-inline-lbl">Indexed</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # CHAT HISTORY
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # SUGGESTIONS
    if not st.session_state.chat:
        st.markdown("""
        <div class="dm-suggestions">
          <div class="dm-suggestions-label">Suggested questions</div>
        </div>
        """, unsafe_allow_html=True)
        suggestions = [
            "Summarize this document",
            "What are the main topics?",
            "What conclusions are drawn?",
            "List key findings or recommendations",
        ]
        cols = st.columns(4)
        for col, q in zip(cols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state._prefill = q
                st.rerun()

    # INPUT
    prefill  = st.session_state.pop("_prefill", "")
    question = st.chat_input("Ask anything about your document…")
    if not question and prefill:
        question = prefill

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                answer = st.session_state.pipeline.ask(question, top_k=top_k)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dm-footer">
  <div class="dm-footer-brand">Doc<em>Mind</em></div>
  <div class="dm-footer-meta">FAISS · all-MiniLM-L6-v2 · llama3-8b-8192 · Streamlit 1.35</div>
  <div class="dm-footer-links">
    <a href="https://console.groq.com" target="_blank">Groq</a>
    <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" target="_blank">Model</a>
    <a href="https://github.com/facebookresearch/faiss" target="_blank">FAISS</a>
    <a href="https://streamlit.io" target="_blank">Streamlit</a>
  </div>
</div>
""", unsafe_allow_html=True)
