from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"  # fast, free, 384-dim, runs locally

_model = None  # module-level cache so we load once


def get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of strings to embed.
        batch_size: How many to process at once (tune for memory).

    Returns:
        numpy array of shape (len(texts), 384)
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # cosine sim = dot product
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query])[0]
