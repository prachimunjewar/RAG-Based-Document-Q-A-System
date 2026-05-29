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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* clean, simple theme */

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #1e293b !important;
}
[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #0ea5e9 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
body, .stApp { background: #f9fafb; color: #111827; font-family: sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* header */
.app-header {
    background: #1e293b;
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header h1 { font-size: 1.3rem; margin: 0; }
.app-header span {
    font-size: 0.75rem;
    background: #0ea5e9;
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
}

/* stat cards */
.stat-row { display: flex; gap: 12px; margin-bottom: 1rem; }
.stat-box {
    flex: 1;
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-val { font-size: 1.4rem; font-weight: 700; color: #0ea5e9; }
.stat-lbl { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; }

/* footer */
.app-footer {
    margin-top: 2rem;
    text-align: center;
    font-size: 0.75rem;
    color: #9ca3af;
    padding: 1rem;
    border-top: 1px solid #e5e7eb;
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

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.caption("RAG-powered document Q&A")
    st.divider()

    st.markdown("**📂 Upload Document**")
    uploaded = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    st.divider()
    st.markdown("**⚙️ Settings**")
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

            with st.spinner("Chunking text..."):
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
        st.markdown("**📊 Document Info**")
        col1, col2 = st.columns(2)
        col1.metric("Pages", d["pages"])
        col2.metric("Chunks", d["chunks"])
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
    st.caption("Stack: FAISS · HuggingFace · Groq LLaMA 3 · Streamlit")

# ── MAIN ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🧠 DocMind — Document Q&A</h1>
  <span>RAG + FAISS + Groq</span>
</div>
""", unsafe_allow_html=True)

if not st.session_state.pipeline:
    st.info("👈 Upload a PDF or TXT file in the sidebar and click **Process Document** to get started.")

    with st.expander("ℹ️ How it works", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.markdown("**1. Upload**\nPDF text extracted page by page")
        col2.markdown("**2. Chunk**\nSplit into overlapping word windows")
        col3.markdown("**3. Embed**\nHuggingFace all-MiniLM-L6-v2")
        col4.markdown("**4. Retrieve**\nFAISS cosine similarity search")
        col5.markdown("**5. Answer**\nGroq LLaMA 3 with memory")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**⚡ Fast**\nSub-second retrieval via FAISS")
    col2.markdown("**🔒 Private**\nEmbeddings run locally on your machine")
    col3.markdown("**🆓 Free**\nGroq API + HuggingFace, no cost")

else:
    d = st.session_state.doc_info

    # stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Document", d["name"][:20] + "..." if len(d["name"]) > 20 else d["name"])
    col2.metric("📑 Pages", d["pages"])
    col3.metric("🧩 Chunks", d["chunks"])
    col4.metric("⏱ Index time", f"{d['embed_time']}s")

    st.divider()

    # chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # suggestions on first load
    if not st.session_state.chat:
        st.markdown("**💡 Try a question:**")
        cols = st.columns(4)
        suggestions = [
            "Summarize this document",
            "What are the main topics?",
            "What conclusions are made?",
            "List the key findings",
        ]
        for col, q in zip(cols, suggestions):
            if col.button(q, use_container_width=True):
                st.session_state._prefill = q
                st.rerun()

    # input
    prefill  = st.session_state.pop("_prefill", "")
    question = st.chat_input("Ask anything about your document...")
    if not question and prefill:
        question = prefill

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.pipeline.ask(question, top_k=top_k)
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  🧠 DocMind &nbsp;·&nbsp; FAISS · all-MiniLM-L6-v2 · llama3-8b-8192 · Streamlit
</div>
""", unsafe_allow_html=True)
