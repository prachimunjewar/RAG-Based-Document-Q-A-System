from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, chunks, vector_store, bm25_weight=0.3, faiss_weight=0.7):
        self.chunks = chunks
        self.vs = vector_store
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight

        # extract text from chunk dicts
        self.texts = [c["text"] if isinstance(c, dict) else c for c in chunks]

        # build BM25 index on plain text
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, top_k=4):
        # FAISS scores
        faiss_results = self.vs.search_by_index(query, top_k=len(self.chunks))
        faiss_scores = np.zeros(len(self.chunks))
        for idx, score in faiss_results:
            faiss_scores[idx] = score

        if faiss_scores.max() > 0:
            faiss_scores = faiss_scores / faiss_scores.max()

        # BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # combine
        combined = self.faiss_weight * faiss_scores + self.bm25_weight * bm25_scores
        top_indices = np.argsort(combined)[::-1][:top_k]

        return [(int(i), self.texts[i], float(combined[i])) for i in top_indices]
