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
    page_title="Lexis · AI Document Q&A",
    page_icon="📖",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:       #111210;
    --bg2:      #191a17;
    --bg3:      #1f2118;
    --border:   #2a2c26;
    --border2:  #333530;
    --amber:    #f59e0b;
    --amber2:   #fbbf24;
    --amber-dim:rgba(245,158,11,0.12);
    --amber-glow:rgba(245,158,11,0.06);
    --text:     #e8e5dc;
    --muted:    #8a8880;
    --dim:      #4a4845;
    --green:    #4ade80;
    --red:      #f87171;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
.block-container {
    padding: 0 !important;
    max-width: 780px !important;
    margin: 0 auto !important;
}

/* ── PAGE WRAPPER ── */
.lx-page { padding: 3rem 0 4rem; }

/* ── TOP NAV ── */
.lx-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 2.5rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
}
.lx-brand {
    display: flex;
    align-items: baseline;
    gap: 6px;
}
.lx-brand-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}
.lx-brand-dot { color: var(--amber); font-size: 1.6rem; line-height: 1; }
.lx-brand-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--amber);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: var(--amber-dim);
    border: 1px solid rgba(245,158,11,0.2);
    padding: 3px 9px;
    border-radius: 4px;
}
.lx-nav-chips { display: flex; gap: 8px; align-items: center; }
.lx-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: var(--dim);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: var(--bg2);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 4px;
}

/* ── HERO ── */
.lx-hero { margin-bottom: 3rem; }
.lx-hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--amber);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.lx-hero-eyebrow::before {
    content: '';
    display: block;
    width: 24px;
    height: 1px;
    background: var(--amber);
}
.lx-hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.12;
    letter-spacing: -0.02em;
    color: var(--text);
    margin-bottom: 1rem;
}
.lx-hero h1 em {
    font-style: italic;
    font-weight: 400;
    color: var(--amber);
}
.lx-hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    line-height: 1.7;
    max-width: 500px;
    font-weight: 300;
}

/* ── UPLOAD ZONE ── */
.lx-upload-wrap {
    background: var(--bg2);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.lx-upload-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
    opacity: 0.6;
}
.lx-upload-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.lx-upload-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* file uploader styling */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1.5px dashed var(--border2) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--amber) !important;
}
[data-testid="stFileUploader"] * { color: var(--muted) !important; }
[data-testid="stFileUploadDropzoneInstructions"] small { color: var(--dim) !important; }

/* ── SETTINGS ── */
.lx-settings {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.lx-settings-label {
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.lx-settings-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* slider */
[data-testid="stSlider"] label {
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-baseweb="slider"] [role="slider"] {
    background: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.2) !important;
}
[data-testid="stSliderTrackFill"] {
    background: var(--amber) !important;
}
[data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: var(--amber) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ── PROCESS BUTTON ── */
.stButton > button {
    background: var(--amber) !important;
    color: #111210 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.01em;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(245,158,11,0.25) !important;
}
.stButton > button:hover {
    background: var(--amber2) !important;
    box-shadow: 0 6px 28px rgba(245,158,11,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── HOW IT WORKS ── */
.lx-how {
    margin-bottom: 2.5rem;
}
.lx-how-title {
    font-size: 0.68rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
}
.lx-pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
.lx-pipe-step {
    flex: 1;
    padding: 1.1rem 0.9rem;
    border-right: 1px solid var(--border);
    text-align: center;
    transition: background 0.2s;
}
.lx-pipe-step:last-child { border-right: none; }
.lx-pipe-step:hover { background: var(--amber-glow); }
.lx-pipe-n {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--border2);
    line-height: 1;
    margin-bottom: 4px;
    transition: color 0.2s;
}
.lx-pipe-step:hover .lx-pipe-n { color: var(--amber); }
.lx-pipe-name {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 3px;
}
.lx-pipe-tech {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.56rem;
    color: var(--amber);
    background: var(--amber-dim);
    padding: 2px 6px;
    border-radius: 3px;
    display: inline-block;
    margin-top: 3px;
}

/* ── STATS ROW (after indexing) ── */
.lx-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.5rem;
}
.lx-stat {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.lx-stat:hover { border-color: var(--amber); }
.lx-stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--amber);
    line-height: 1;
    margin-bottom: 4px;
}
.lx-stat-lbl {
    font-size: 0.62rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace;
}

/* ── DOC ACTIVE BAR ── */
.lx-doc-bar {
    background: var(--bg2);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 0.85rem 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    gap: 1rem;
    flex-wrap: wrap;
}
.lx-doc-left {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.83rem;
    color: var(--text);
    font-weight: 500;
}
.lx-live {
    width: 8px; height: 8px;
    background: var(--green);
    border-radius: 50%;
    flex-shrink: 0;
    animation: livepulse 2s ease-in-out infinite;
}
@keyframes livepulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.4); }
    50%      { box-shadow: 0 0 0 4px rgba(74,222,128,0); }
}
.lx-doc-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--dim);
    display: flex;
    gap: 1rem;
}

/* ── SUGGESTIONS ── */
.lx-suggest-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}
div[data-testid="column"] .stButton > button {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.55rem 0.8rem !important;
    text-align: left !important;
    box-shadow: none !important;
    transform: none !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: var(--amber-dim) !important;
    border-color: rgba(245,158,11,0.3) !important;
    color: var(--amber) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
    padding: 0.8rem 1rem !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] textarea {
    background: var(--bg2) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.1) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--dim) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: var(--amber-dim) !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-size: 0.83rem !important;
}

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── NEW DOC BUTTON ── */
.new-doc-btn > div > button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    transform: none !important;
    width: auto !important;
    padding: 0.4rem 1rem !important;
}
.new-doc-btn > div > button:hover {
    border-color: var(--red) !important;
    color: var(--red) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── FOOTER ── */
.lx-footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.lx-footer-brand {
    font-family: 'Playfair Display', serif;
    font-size: 0.9rem;
    color: var(--dim);
}
.lx-footer-brand span { color: var(--amber); }
.lx-footer-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: var(--dim);
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────
for k, v in [("pipeline", None), ("chat", []), ("doc_info", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── NAV ───────────────────────────────────────────────────────
st.markdown("""
<div class="lx-nav">
  <div class="lx-brand">
    <div class="lx-brand-name">Lexis</div>
    <div class="lx-brand-dot">·</div>
    <div class="lx-brand-tag">AI Document Q&A</div>
  </div>
  <div class="lx-nav-chips">
    <div class="lx-chip">FAISS</div>
    <div class="lx-chip">Groq · LLaMA 3</div>
    <div class="lx-chip">HuggingFace</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════
if not st.session_state.pipeline:

    st.markdown("""
    <div class="lx-hero">
      <div class="lx-hero-eyebrow">Retrieval-Augmented Generation</div>
      <h1>Ask anything.<br><em>Get real answers.</em></h1>
      <p class="lx-hero-sub">
        Upload a PDF — Lexis reads it, indexes it, and lets you have a
        real conversation with your document. Powered by free APIs, runs privately.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # upload
    st.markdown("""
    <div class="lx-upload-wrap">
      <div class="lx-upload-label">01 — Upload your document</div>
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "", type=["pdf", "txt"],
        label_visibility="collapsed",
        help="Text-based PDFs work best. Scanned images are not supported."
    )

    # settings
    st.markdown("""
    <div class="lx-settings">
      <div class="lx-settings-label">02 — Configure retrieval</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        chunk_size = st.slider("Chunk size (words)", 200, 800, 500, 50,
                               help="Larger chunks = more context per result")
    with c2:
        overlap = st.slider("Overlap (words)", 20, 200, 100, 10,
                            help="Overlap prevents context loss at chunk edges")
    with c3:
        top_k = st.slider("Chunks to retrieve", 2, 8, 4,
                          help="How many chunks are sent to the LLM")

    # process button
    st.markdown("**03 — Run the pipeline**")
    if uploaded:
        if st.button("⚡  Process & Index Document"):
            with st.spinner("📄 Extracting text from document..."):
                suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                doc = load_document(tmp_path)
                os.unlink(tmp_path)

            with st.spinner(f"✂️ Splitting into chunks..."):
                chunks = chunk_text(doc["full_text"], chunk_size, overlap)

            with st.spinner(f"🔢 Embedding {len(chunks)} chunks & building FAISS index..."):
                t0 = time.time()
                vs = VectorStore()
                vs.build(chunks)
                elapsed = round(time.time() - t0, 1)

            st.session_state.pipeline = RAGPipeline(vs, chunks)
            st.session_state.chat     = []
            st.session_state.doc_info = {
                "name":       doc["file_name"],
                "pages":      doc["total_pages"],
                "chunks":     len(chunks),
                "embed_time": elapsed,
                "top_k":      top_k,
            }
            st.rerun()
    else:
        st.button("⚡  Process & Index Document", disabled=True)
        st.caption("↑ Upload a file first")

    # pipeline visual
    st.markdown("""
    <div class="lx-how">
      <div class="lx-how-title">How Lexis works</div>
      <div class="lx-pipeline">
        <div class="lx-pipe-step">
          <div class="lx-pipe-n">01</div>
          <div class="lx-pipe-name">Extract</div>
          <div class="lx-pipe-tech">pypdf</div>
        </div>
        <div class="lx-pipe-step">
          <div class="lx-pipe-n">02</div>
          <div class="lx-pipe-name">Chunk</div>
          <div class="lx-pipe-tech">500-word</div>
        </div>
        <div class="lx-pipe-step">
          <div class="lx-pipe-n">03</div>
          <div class="lx-pipe-name">Embed</div>
          <div class="lx-pipe-tech">MiniLM-L6</div>
        </div>
        <div class="lx-pipe-step">
          <div class="lx-pipe-n">04</div>
          <div class="lx-pipe-name">Retrieve</div>
          <div class="lx-pipe-tech">FAISS</div>
        </div>
        <div class="lx-pipe-step">
          <div class="lx-pipe-n">05</div>
          <div class="lx-pipe-name">Generate</div>
          <div class="lx-pipe-tech">LLaMA 3</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# CHAT PAGE
# ═══════════════════════════════════════════════
else:
    d     = st.session_state.doc_info
    top_k = d.get("top_k", 4)

    # stats
    st.markdown(f"""
    <div class="lx-stats">
      <div class="lx-stat">
        <div class="lx-stat-val">{d['pages']}</div>
        <div class="lx-stat-lbl">Pages</div>
      </div>
      <div class="lx-stat">
        <div class="lx-stat-val">{d['chunks']}</div>
        <div class="lx-stat-lbl">Chunks</div>
      </div>
      <div class="lx-stat">
        <div class="lx-stat-val">{d['embed_time']}s</div>
        <div class="lx-stat-lbl">Indexed</div>
      </div>
      <div class="lx-stat">
        <div class="lx-stat-val">{top_k}</div>
        <div class="lx-stat-lbl">Top-k</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # active doc bar
    name = d['name'][:45] + "…" if len(d['name']) > 45 else d['name']
    st.markdown(f"""
    <div class="lx-doc-bar">
      <div class="lx-doc-left">
        <div class="lx-live"></div>
        {name}
      </div>
      <div class="lx-doc-meta">
        <span>llama3-8b-8192</span>
        <span>faiss · cosine</span>
        <span>MiniLM-L6-v2</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # new document button
    if st.button("↩  Upload a different document"):
        st.session_state.pipeline = None
        st.session_state.chat     = []
        st.session_state.doc_info = None
        st.rerun()

    st.divider()

    # chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # suggestions on first load
    if not st.session_state.chat:
        st.markdown('<div class="lx-suggest-label">Suggested questions</div>', unsafe_allow_html=True)
        suggestions = [
            "📝 Summarize this document",
            "🔍 What are the main topics?",
            "💡 What conclusions are drawn?",
            "📋 List the key findings",
        ]
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if cols[i % 2].button(q, use_container_width=True, key=f"sug_{i}"):
                st.session_state._prefill = q.split(" ", 1)[1]
                st.rerun()
        st.markdown("")

    # input
    prefill  = st.session_state.pop("_prefill", "")
    question = st.chat_input("Ask anything about your document…")
    if not question and prefill:
        question = prefill

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and generating answer…"):
                answer = st.session_state.pipeline.ask(question, top_k=top_k)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="lx-footer">
  <div class="lx-footer-brand">Lexi<span>s</span></div>
  <div class="lx-footer-meta">
    pypdf · faiss-cpu · all-MiniLM-L6-v2 · llama3-8b-8192 · streamlit
  </div>
</div>
""", unsafe_allow_html=True)
