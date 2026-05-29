from pypdf import PdfReader
import os


def load_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF file, page by page.
    Returns a dict with full text and per-page breakdown.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})

    full_text = "\n\n".join(p["text"] for p in pages)

    return {
        "full_text": full_text,
        "pages": pages,
        "total_pages": len(reader.pages),
        "extracted_pages": len(pages),
        "file_name": os.path.basename(file_path),
    }


def load_text_file(file_path: str) -> dict:
    """Fallback loader for plain .txt files."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return {
        "full_text": text,
        "pages": [{"page": 1, "text": text}],
        "total_pages": 1,
        "extracted_pages": 1,
        "file_name": os.path.basename(file_path),
    }


def load_document(file_path: str) -> dict:
    """Auto-detect file type and load accordingly."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".txt":
        return load_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
