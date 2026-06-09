import faiss
import numpy as np
from typing import List, Tuple
from utils.embedder import embed_texts, embed_query

class VectorStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: List[dict] = []

    def build(self, chunks: List[dict]) -> None:
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        self.index.reset()
        self.index.add(embeddings.astype("float32"))

    def search(self, query: str, top_k: int = 4) -> List[Tuple[dict, float]]:
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

    def search_by_index(self, query: str, top_k: int = 4) -> List[Tuple[int, float]]:
        """Returns (index, score) pairs — used by HybridRetriever."""
        if self.index.ntotal == 0:
            return []
        qvec = embed_query(query).astype("float32").reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((int(idx), float(score)))
        return results

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path)

    @property
    def size(self) -> int:
        return self.index.ntotal
