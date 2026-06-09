import os
from typing import List
from groq import Groq
from dotenv import load_dotenv
from utils.vector_store import VectorStore
from utils.hybrid_retriever import HybridRetriever

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY = 6

class RAGPipeline:
    def __init__(self, vector_store: VectorStore, chunks: List[str]):
        self.vs = vector_store
        self.retriever = HybridRetriever(chunks, vector_store)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.history: List[dict] = []

    def _build_system_prompt(self, context: str) -> str:
        return f"""You are a helpful, precise document assistant.
Answer the user's questions based ONLY on the context provided below.
If the answer isn't in the context, say: "I don't see that in the document."
Be concise, accurate, and cite the relevant part when helpful.

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---"""

    def ask(self, question: str, top_k: int = 4):
        # 1. Hybrid retrieval (BM25 + FAISS)
        results = self.retriever.retrieve(question, top_k=top_k)
        if not results:
            return "No document loaded or index is empty.", []

        # 2. Build context from results
        context = "\n\n".join([
            f"[Excerpt {i+1}] {chunk}"
            for i, (_, chunk, _) in enumerate(results)
        ])

        system_prompt = self._build_system_prompt(context)

        # 3. Add question to history
        self.history.append({"role": "user", "content": question})
        trimmed = self.history[-MAX_HISTORY:]

        # 4. Call Groq
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + trimmed,
            temperature=0.2,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()

        # 5. Save to history
        self.history.append({"role": "assistant", "content": answer})

        # 6. Return both answer and results for evaluation
        return answer, results

    def reset_memory(self) -> None:
        self.history = []

    @property
    def memory_length(self) -> int:
        return len(self.history)
