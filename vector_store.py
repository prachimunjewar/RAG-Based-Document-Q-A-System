import faiss
import numpy as np
from typing import List, Tuple
from utils.embedder import embed_texts, embed_query


class VectorStore:
    """
    In-memory FAISS vector store for document chunks.
    Supports building, searching, and saving/loading the index.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine (normalized vecs)
        self.chunks: List[dict] = []

    def build(self, chunks: List[dict]) -> None:
        """
        Embed all chunks and add them to the FAISS index.

        Args:
            chunks: List of chunk dicts from chunker.chunk_text()
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        self.index.reset()
        self.index.add(embeddings.astype("float32"))

    def search(self, query: str, top_k: int = 4) -> List[Tuple[dict, float]]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Returns:
            List of (chunk_dict, similarity_score) tuples, best first.
        """
        if self.index.ntotal == 0:
            return []

        qvec = embed_query(query).astype("float32").reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(qvec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(path)

    @property
    def size(self) -> int:
        return self.index.ntotal
