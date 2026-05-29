# RAG-Based Document Q&A System

> Upload any PDF. Ask anything. Get instant, grounded answers — powered by FAISS, HuggingFace, and Groq LLaMA 3. Completely free.

---

## 📌 What is this?

**RAG-Based Document Q&A System** is a full-stack Retrieval-Augmented Generation (RAG) application that lets you have a multi-turn conversation with any PDF document. Instead of just searching for keywords, it understands the *meaning* of your question, retrieves the most relevant passages, and generates a precise answer grounded strictly in your document.

Built as a personal project during a **Full Stack Data Science program** specializing in **Generative AI** and **Agentic AI**.

---

## 🚀 Live Demo


---

## ✨ Features

- 📄 **PDF & TXT support** — upload any text-based document
- 🔍 **Semantic retrieval** — FAISS cosine similarity search over dense embeddings
- 🧩 **Smart chunking** — overlapping 500-word windows preserve cross-boundary context
- 🤖 **Free LLM** — Groq LLaMA 3 (8B) for fast, accurate, grounded answers
- 💬 **Multi-turn memory** — LangChain `ConversationBufferMemory` retains last 6 exchanges
- ⚙️ **Configurable** — adjust chunk size, overlap, and retrieval depth from the UI
- 🎨 **Polished UI** — custom dark-themed Streamlit app with stats dashboard

---

## 🗂️ Project Structure

```
rag_app/
├── app.py                  # Streamlit UI — main entry point
├── requirements.txt        # all dependencies
├── .env.example            # environment variable template
├── .gitignore
└── utils/
    ├── __init__.py
    ├── pdf_loader.py       # PDF & TXT text extraction
    ├── chunker.py          # overlapping word-level chunking
    ├── embedder.py         # HuggingFace sentence-transformers
    ├── vector_store.py     # FAISS index — build, search, save, load
    └── rag_pipeline.py     # end-to-end RAG + LangChain memory
```

---

## ⚡ Quickstart


### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your free Groq API key

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```
GROQ_API_KEY=your_groq_api_key_here
```

> 🔑 Get a free key at [console.groq.com](https://console.groq.com) → API Keys → Create key. No credit card required.

### 4. Run the app

```bash
streamlit run app.py
```
---

## 🧪 How to Use

1. **Upload** a PDF or TXT file using the sidebar
2. Click **⚡ Process Document** — the pipeline extracts, chunks, and indexes it
3. Watch the **stats dashboard** — pages, chunks, and indexing time appear instantly
4. **Ask any question** in the chat input or pick a suggested question
5. Follow up naturally — the app remembers your last 6 exchanges

---

## 🛠️ Tech Stack

streamlit
pypdf
sentence-transformers
faiss-cpu
groq
python-dotenv
numpy

---

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss what you'd like to change.

---

## 👤 Author

**Prachi Munjewar**

---

# Built with ❤️ as part of a Full Stack Data Science program · Generative AI & Agentic AI specialization.
