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
    page_title="Synapse — AI Document Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://api.fontshare.com/v2/css?f[]=clash-display@600,700&f[]=cabinet-grotesk@400,500,700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --navy:    #050d1a;
    --navy2:   #0a1628;
    --navy3:   #0f1f38;
    --cyan:    #00d4ff;
    --cyan2:   #00a8cc;
    --teal:    #00ffc8;
    --glass:   rgba(255,255,255,0.04);
    --glass2:  rgba(255,255,255,0.07);
    --border:  rgba(0,212,255,0.15);
    --border2: rgba(255,255,255,0.06);
    --text:    #e8f4f8;
    --muted:   rgba(232,244,248,0.45);
    --dim:     rgba(232,244,248,0.2);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp, [class*="css"] {
    font-family: 'Cabinet Grotesk', sans-serif !important;
    background: var(--navy) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── ANIMATED BG ORBS ── */
.bg-orbs {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
.orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.18;
    animation: drift 20s ease-in-out infinite;
}
.orb-1 {
    width: 600px; height: 600px;
    background: radial-gradient(circle, #00d4ff, transparent);
    top: -200px; left: -150px;
    animation-delay: 0s;
}
.orb-2 {
    width: 500px; height: 500px;
    background: radial-gradient(circle, #00ffc8, transparent);
    bottom: -150px; right: -100px;
    animation-delay: -7s;
}
.orb-3 {
    width: 350px; height: 350px;
    background: radial-gradient(circle, #5b8dee, transparent);
    top: 40%; left: 40%;
    animation-delay: -13s;
    opacity: 0.1;
}
@keyframes drift {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%       { transform: translate(40px, -30px) scale(1.05); }
    66%       { transform: translate(-20px, 40px) scale(0.95); }
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--navy2) !important;
    border-right: 1px solid var(--border2) !important;
    z-index: 10;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border2) !important;
    margin: 0.8rem 0 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* slider */
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: var(--cyan) !important;
    box-shadow: 0 0 10px rgba(0,212,255,0.6) !important;
}
[data-testid="stSidebar"] [data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg, var(--cyan2), var(--cyan)) !important;
}

/* file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: var(--glass) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    background: rgba(0,212,255,0.05) !important;
    border-color: var(--cyan) !important;
}

/* primary button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--cyan2), var(--cyan)) !important;
    color: var(--navy) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 20px rgba(0,212,255,0.25) !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 6px 28px rgba(0,212,255,0.45) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] .stButton:last-of-type > button {
    background: transparent !important;
    color: var(--muted) !important;
    border: 1px solid var(--border2) !important;
    box-shadow: none !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton:last-of-type > button:hover {
    border-color: rgba(0,212,255,0.3) !important;
    color: var(--cyan) !important;
    transform: none !important;
}

/* alerts */
[data-testid="stAlert"] {
    background: rgba(0,255,200,0.06) !important;
    border: 1px solid rgba(0,255,200,0.25) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    color: var(--teal) !important;
}

/* ── TOPBAR ── */
.sn-topbar {
    position: sticky;
    top: 0;
    z-index: 50;
    background: rgba(5,13,26,0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border2);
    padding: 0 2.5rem;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.sn-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.sn-logo-mark {
    width: 30px;
    height: 30px;
    background: linear-gradient(135deg, var(--cyan), var(--teal));
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    box-shadow: 0 0 15px rgba(0,212,255,0.4);
}
.sn-logo-name {
    font-family: 'Clash Display', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
}
.sn-logo-name span { color: var(--cyan); }
.sn-topbar-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.sn-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--glass);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.06em;
}
.sn-dot {
    width: 6px;
    height: 6px;
    background: var(--teal);
    border-radius: 50%;
    animation: glow-pulse 2s ease-in-out infinite;
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 4px var(--teal); opacity: 1; }
    50%       { box-shadow: 0 0 10px var(--teal); opacity: 0.6; }
}

/* ── HERO ── */
.sn-hero {
    padding: 6rem 3.5rem 4rem;
    max-width: 900px;
    position: relative;
    z-index: 1;
    animation: fadeUp 0.7s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.sn-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 5px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--cyan);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}
.sn-kicker::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--cyan);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--cyan);
}
.sn-hero h1 {
    font-family: 'Clash Display', sans-serif;
    font-size: 4.2rem;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: var(--text);
    margin-bottom: 1.5rem;
}
.sn-hero h1 .accent {
    background: linear-gradient(135deg, var(--cyan), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sn-hero-desc {
    font-size: 1.05rem;
    color: var(--muted);
    line-height: 1.75;
    max-width: 560px;
    margin-bottom: 3rem;
    font-weight: 400;
}
.sn-cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    max-width: 720px;
}
.sn-card {
    background: var(--glass);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    backdrop-filter: blur(10px);
    transition: all 0.25s;
}
.sn-card:hover {
    background: var(--glass2);
    border-color: var(--border);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,212,255,0.08);
}
.sn-card-icon {
    font-size: 1.2rem;
    margin-bottom: 6px;
}
.sn-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 3px;
}
.sn-card-sub {
    font-size: 0.7rem;
    color: var(--dim);
    line-height: 1.4;
}

/* ── PIPELINE ── */
.sn-pipeline {
    padding: 0 3.5rem 4rem;
    position: relative;
    z-index: 1;
    animation: fadeUp 0.7s 0.1s ease both;
}
.sn-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--cyan);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sn-section-title::after {
    content: '';
    flex: 1;
    max-width: 120px;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}
.sn-pipe-grid {
    display: flex;
    gap: 0;
    max-width: 860px;
}
.sn-pipe-step {
    flex: 1;
    background: var(--glass);
    border: 1px solid var(--border2);
    border-right: none;
    padding: 1.4rem 1.1rem;
    position: relative;
    transition: all 0.2s;
}
.sn-pipe-step:first-child { border-radius: 10px 0 0 10px; }
.sn-pipe-step:last-child  { border-right: 1px solid var(--border2); border-radius: 0 10px 10px 0; }
.sn-pipe-step:hover {
    background: rgba(0,212,255,0.06);
    border-color: rgba(0,212,255,0.2);
    z-index: 1;
}
.sn-pipe-num {
    font-family: 'Clash Display', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: rgba(0,212,255,0.12);
    line-height: 1;
    margin-bottom: 8px;
}
.sn-pipe-step:hover .sn-pipe-num { color: rgba(0,212,255,0.25); }
.sn-pipe-name {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
}
.sn-pipe-desc {
    font-size: 0.7rem;
    color: var(--dim);
    line-height: 1.5;
    margin-bottom: 8px;
}
.sn-pipe-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: var(--cyan);
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.15);
    padding: 2px 7px;
    border-radius: 4px;
    display: inline-block;
}

/* ── CHAT ── */
.sn-chat-bar {
    background: rgba(10,22,40,0.9);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border2);
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.sn-doc-badge {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 0.8rem;
    color: var(--text);
    max-width: 320px;
}
.sn-doc-badge-dot {
    width: 8px; height: 8px;
    background: var(--teal);
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 6px var(--teal);
}
.sn-doc-badge-name {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-weight: 500;
}
.sn-metrics {
    display: flex;
    gap: 1.5rem;
}
.sn-metric {
    text-align: center;
}
.sn-metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: var(--cyan);
}
.sn-metric-lbl {
    font-size: 0.6rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 1px;
}

/* suggestions */
.sn-suggest-wrap {
    padding: 1.2rem 2.5rem 0.5rem;
    position: relative;
    z-index: 1;
}
.sn-suggest-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}

div[data-testid="column"] .stButton > button {
    background: var(--glass) !important;
    border: 1px solid var(--border2) !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    padding: 0.55rem 0.8rem !important;
    width: 100% !important;
    text-align: left !important;
    transition: all 0.2s !important;
    backdrop-filter: blur(10px) !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(0,212,255,0.08) !important;
    border-color: var(--border) !important;
    color: var(--cyan) !important;
    transform: none !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.08) !important;
}

/* chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
    position: relative;
    z-index: 1;
}

/* chat input */
[data-testid="stChatInput"] {
    background: rgba(10,22,40,0.95) !important;
    backdrop-filter: blur(20px) !important;
    border-top: 1px solid var(--border2) !important;
    padding: 1rem 2.5rem !important;
    position: sticky;
    bottom: 0;
    z-index: 50;
}
[data-testid="stChatInput"] textarea {
    background: var(--navy3) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.1), 0 0 20px rgba(0,212,255,0.08) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--dim) !important;
}

/* ── FOOTER ── */
.sn-footer {
    background: var(--navy2);
    border-top: 1px solid var(--border2);
    padding: 1.4rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    position: relative;
    z-index: 1;
    margin-top: 3rem;
}
.sn-footer-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Clash Display', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--muted);
}
.sn-footer-brand span { color: var(--cyan); }
.sn-footer-mark {
    width: 20px; height: 20px;
    background: linear-gradient(135deg, var(--cyan), var(--teal));
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.55rem;
    box-shadow: 0 0 8px rgba(0,212,255,0.3);
}
.sn-footer-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--dim);
    letter-spacing: 0.06em;
}
.sn-footer-links { display: flex; gap: 1.5rem; }
.sn-footer-links a {
    font-size: 0.75rem;
    color: var(--dim);
    text-decoration: none;
    transition: color 0.15s;
}
.sn-footer-links a:hover { color: var(--cyan); }

/* ── SIDEBAR STAT GRID ── */
.sn-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 6px;
}
.sn-stat-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 7px;
    padding: 0.55rem 0.4rem;
    text-align: center;
}
.sn-stat-box-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: var(--cyan);
}
.sn-stat-box-lbl {
    font-size: 0.58rem;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

hr { border-color: var(--border2) !important; }
</style>

<!-- Animated background orbs -->
<div class="bg-orbs">
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>
</div>
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
<div class="sn-topbar">
  <div class="sn-logo">
    <div class="sn-logo-mark">🔮</div>
    <div class="sn-logo-name">Syn<span>apse</span></div>
  </div>
  <div class="sn-topbar-right">
    <div class="sn-chip"><div class="sn-dot"></div>Live</div>
    <div class="sn-chip">RAG · FAISS · Groq · LLaMA 3</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.8rem 1.5rem 1rem">
      <div style="font-family:'Clash Display',sans-serif;font-size:1.05rem;font-weight:700;color:#e8f4f8;margin-bottom:4px">
        Document
      </div>
      <div style="font-size:0.72rem;color:rgba(232,244,248,0.35);letter-spacing:0.02em">
        PDF or TXT · up to 5MB
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        uploaded = st.file_uploader("", type=["pdf", "txt"], label_visibility="collapsed")

    st.divider()

    st.markdown("""
    <div style="padding:0 1.5rem 0.5rem">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:rgba(0,212,255,0.7);letter-spacing:0.12em;text-transform:uppercase">
        Retrieval Config
      </div>
    </div>
    """, unsafe_allow_html=True)

    chunk_size = st.slider("Chunk size (words)", 200, 800, 500, 50)
    overlap    = st.slider("Overlap (words)", 20, 200, 100, 10)
    top_k      = st.slider("Chunks to retrieve", 2, 8, 4)

    if uploaded:
        st.divider()
        st.markdown("<div style='padding:0 1.5rem 1rem'>", unsafe_allow_html=True)
        if st.button("⚡  Run Pipeline", type="primary", use_container_width=True):
            with st.spinner("Extracting text…"):
                suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                doc = load_document(tmp_path)
                os.unlink(tmp_path)

            with st.spinner("Chunking…"):
                chunks = chunk_text(doc["full_text"], chunk_size, overlap)

            with st.spinner("Embedding & indexing…"):
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
            st.success("Index ready — ask away →")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.doc_info:
        d = st.session_state.doc_info
        st.divider()
        st.markdown(f"""
        <div style="padding:0 1.5rem 1rem">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:rgba(0,212,255,0.7);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.8rem">
            Index Stats
          </div>
          <div class="sn-stat-grid">
            <div class="sn-stat-box">
              <div class="sn-stat-box-val">{d['pages']}</div>
              <div class="sn-stat-box-lbl">Pages</div>
            </div>
            <div class="sn-stat-box">
              <div class="sn-stat-box-val">{d['chunks']}</div>
              <div class="sn-stat-box-lbl">Chunks</div>
            </div>
            <div class="sn-stat-box">
              <div class="sn-stat-box-val">{d['embed_time']}s</div>
              <div class="sn-stat-box-lbl">Built</div>
            </div>
          </div>
          <div style="font-size:0.68rem;color:rgba(232,244,248,0.25);margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
            📄 {d['name']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.chat:
        st.divider()
        st.markdown("<div style='padding:0 1.5rem 1rem'>", unsafe_allow_html=True)
        if st.button("Clear session", use_container_width=True):
            st.session_state.chat = []
            if st.session_state.pipeline:
                st.session_state.pipeline.reset_memory()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="padding:0 1.5rem 2rem">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:rgba(232,244,248,0.15);line-height:2;letter-spacing:0.04em">
        pypdf · sentence-transformers<br>
        faiss-cpu · groq · langchain<br>
        streamlit · python-dotenv · numpy
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
if not st.session_state.pipeline:

    st.markdown("""
    <div class="sn-hero">
      <div class="sn-kicker">AI-Powered Document Intelligence</div>
      <h1>Talk to your<br><span class="accent">documents.</span></h1>
      <p class="sn-hero-desc">
        Upload any PDF. Ask anything. Synapse retrieves the most relevant passages using semantic vector search and generates precise, grounded answers — powered entirely by free APIs.
      </p>
      <div class="sn-cards">
        <div class="sn-card">
          <div class="sn-card-icon">⚡</div>
          <div class="sn-card-title">Instant</div>
          <div class="sn-card-sub">Sub-second FAISS retrieval</div>
        </div>
        <div class="sn-card">
          <div class="sn-card-icon">🔒</div>
          <div class="sn-card-title">Private</div>
          <div class="sn-card-sub">Embeddings run locally</div>
        </div>
        <div class="sn-card">
          <div class="sn-card-icon">🧠</div>
          <div class="sn-card-title">Memory</div>
          <div class="sn-card-sub">Multi-turn conversation</div>
        </div>
        <div class="sn-card">
          <div class="sn-card-icon">🆓</div>
          <div class="sn-card-title">Free</div>
          <div class="sn-card-sub">Groq + HuggingFace</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sn-pipeline">
      <div class="sn-section-title">Pipeline</div>
      <div class="sn-pipe-grid">
        <div class="sn-pipe-step">
          <div class="sn-pipe-num">01</div>
          <div class="sn-pipe-name">Extract</div>
          <div class="sn-pipe-desc">Page-by-page text from your PDF</div>
          <div class="sn-pipe-tag">pypdf</div>
        </div>
        <div class="sn-pipe-step">
          <div class="sn-pipe-num">02</div>
          <div class="sn-pipe-name">Chunk</div>
          <div class="sn-pipe-desc">Overlapping 500-word windows</div>
          <div class="sn-pipe-tag">custom</div>
        </div>
        <div class="sn-pipe-step">
          <div class="sn-pipe-num">03</div>
          <div class="sn-pipe-name">Embed</div>
          <div class="sn-pipe-desc">384-dim vectors, on-device</div>
          <div class="sn-pipe-tag">MiniLM-L6</div>
        </div>
        <div class="sn-pipe-step">
          <div class="sn-pipe-num">04</div>
          <div class="sn-pipe-name">Retrieve</div>
          <div class="sn-pipe-desc">Cosine similarity, top-k results</div>
          <div class="sn-pipe-tag">FAISS</div>
        </div>
        <div class="sn-pipe-step">
          <div class="sn-pipe-num">05</div>
          <div class="sn-pipe-name">Generate</div>
          <div class="sn-pipe-desc">Grounded answer with memory</div>
          <div class="sn-pipe-tag">Groq LLaMA3</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    d = st.session_state.doc_info
    st.markdown(f"""
    <div class="sn-chat-bar">
      <div class="sn-doc-badge">
        <div class="sn-doc-badge-dot"></div>
        <div class="sn-doc-badge-name">{d['name']}</div>
      </div>
      <div class="sn-metrics">
        <div class="sn-metric">
          <div class="sn-metric-val">{d['pages']}</div>
          <div class="sn-metric-lbl">Pages</div>
        </div>
        <div class="sn-metric">
          <div class="sn-metric-val">{d['chunks']}</div>
          <div class="sn-metric-lbl">Chunks</div>
        </div>
        <div class="sn-metric">
          <div class="sn-metric-val">{d['embed_time']}s</div>
          <div class="sn-metric-lbl">Indexed</div>
        </div>
        <div class="sn-metric">
          <div class="sn-metric-val">{top_k}</div>
          <div class="sn-metric-lbl">Top-k</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.chat:
        st.markdown("""
        <div class="sn-suggest-wrap">
          <div class="sn-suggest-label">Try asking</div>
        </div>
        """, unsafe_allow_html=True)
        suggestions = [
            "Summarize this document",
            "What are the main topics?",
            "What conclusions are drawn?",
            "List the key findings",
        ]
        cols = st.columns(4)
        for col, q in zip(cols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state._prefill = q
                st.rerun()

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
<div class="sn-footer">
  <div class="sn-footer-brand">
    <div class="sn-footer-mark">🔮</div>
    Syn<span>apse</span>
  </div>
  <div class="sn-footer-meta">FAISS · all-MiniLM-L6-v2 · llama3-8b-8192 · Streamlit</div>
  <div class="sn-footer-links">
    <a href="https://console.groq.com" target="_blank">Groq</a>
    <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2" target="_blank">Model</a>
    <a href="https://github.com/facebookresearch/faiss" target="_blank">FAISS</a>
    <a href="https://streamlit.io" target="_blank">Streamlit</a>
  </div>
</div>
""", unsafe_allow_html=True)
