from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[dict]:
    """
    Split text into overlapping word-level chunks.

    Args:
        text: Raw document text.
        chunk_size: Number of words per chunk.
        overlap: Number of words to overlap between chunks.

    Returns:
        List of dicts with chunk index, text, and word count.
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    i = 0
    idx = 0

    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)

        if len(chunk_words) > 20:  # skip tiny trailing fragments
            chunks.append(
                {
                    "index": idx,
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                    "start_word": i,
                }
            )
            idx += 1

        i += step
        if i + chunk_size > len(words) and i < len(words):
            # grab the last chunk without duplicating
            last = " ".join(words[i:])
            if len(last.split()) > 20:
                chunks.append(
                    {
                        "index": idx,
                        "text": last,
                        "word_count": len(last.split()),
                        "start_word": i,
                    }
                )
            break

    return chunks
