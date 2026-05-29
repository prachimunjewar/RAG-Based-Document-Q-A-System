import os
from typing import List
from groq import Groq
from dotenv import load_dotenv
from utils.vector_store import VectorStore

load_dotenv()

MODEL = "llama-3.3-70b-versatile"  # fast, free Groq model
MAX_HISTORY = 6  # keep last N exchanges in memory


class RAGPipeline:
    """
    Full RAG pipeline: retrieve relevant chunks → build prompt → call Groq LLM.
    Maintains multi-turn conversation memory.
    """

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.history: List[dict] = []  # {"role": ..., "content": ...}

    def _build_system_prompt(self, context: str) -> str:
        return f"""You are a helpful, precise document assistant.
Answer the user's questions based ONLY on the context provided below.
If the answer isn't in the context, say: "I don't see that in the document."
Be concise, accurate, and cite the relevant part when helpful.

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---"""

    def _format_context(self, results) -> str:
        parts = []
        for i, (chunk, score) in enumerate(results):
            parts.append(f"[Excerpt {i+1}] {chunk['text']}")
        return "\n\n".join(parts)

    def ask(self, question: str, top_k: int = 4) -> str:
        """
        Ask a question. Retrieves context, builds prompt, calls Groq.

        Args:
            question: User's question string.
            top_k: Number of chunks to retrieve.

        Returns:
            Answer string from the LLM.
        """
        # 1. Retrieve relevant chunks
        results = self.vs.search(question, top_k=top_k)
        if not results:
            return "No document loaded or index is empty."

        context = self._format_context(results)
        system_prompt = self._build_system_prompt(context)

        # 2. Add question to history
        self.history.append({"role": "user", "content": question})

        # 3. Trim history to last MAX_HISTORY messages
        trimmed = self.history[-MAX_HISTORY:]

        # 4. Call Groq
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + trimmed,
            temperature=0.2,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content.strip()

        # 5. Add answer to history
        self.history.append({"role": "assistant", "content": answer})

        return answer

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self.history = []

    @property
    def memory_length(self) -> int:
        return len(self.history)
