"""
Parse downloaded ANMV HTML notices and ingest into ChromaDB.

Each notice is chunked into overlapping text blocks, embedded with
sentence-transformers, and stored with species/product metadata.
BM25 corpus is also saved for hybrid search (Chapter 03).
"""

import json
import pickle
import re
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

DATA_DIR   = Path(__file__).parent.parent / "data"
RAW_DIR    = DATA_DIR / "raw_pdfs"
CHROMA_DIR = DATA_DIR / "chromadb_store"
BM25_PATH  = DATA_DIR / "bm25_corpus.pkl"

CHUNK_SIZE    = 200
CHUNK_OVERLAP = 30
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION    = "anmv_notices"


# ── Text extraction ──────────────────────────────────────────

def extract_text(html_path: Path) -> tuple[str, dict]:
    """
    Parse an ANMV RCP HTML file.
    Returns (clean_text, metadata) where metadata contains product name and species.
    """
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "lxml")

    # remove nav / scripts / styles
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # extract species from text (appears as "Espèces cibles : ...")
    species = ""
    match = re.search(r"Esp[èe]ces? cibles?\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if match:
        species = match.group(1).strip()[:120]

    metadata = {
        "product_name": html_path.stem[:80],
        "species": species,
        "source": html_path.name,
    }
    return text, metadata


# ── Chunking ─────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Word-based (not token-based) — close enough for sentence-transformers.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return [c for c in chunks if len(c.strip()) > 40]


# ── Main ingestion ────────────────────────────────────────────

def main():
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    html_files = sorted(RAW_DIR.glob("*.html"))
    print(f"Found {len(html_files)} HTML notices\n")

    all_chunks    = []   # for BM25
    all_metadatas = []

    for i, path in enumerate(html_files):
        text, metadata = extract_text(path)
        chunks = chunk_text(text)

        if not chunks:
            print(f"[SKIP] {path.name} — no usable text")
            continue

        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        ids = [f"{path.stem}_{j}" for j in range(len(chunks))]
        metadatas = [metadata] * len(chunks)

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        all_chunks.extend(chunks)
        all_metadatas.extend(metadatas)

        if (i + 1) % 20 == 0 or (i + 1) == len(html_files):
            print(f"  [{i+1}/{len(html_files)}] ingested — {len(all_chunks)} chunks total")

    # persist BM25 corpus for hybrid search
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadatas": all_metadatas}, f)

    print(f"\nDone.")
    print(f"  ChromaDB  → {CHROMA_DIR}")
    print(f"  BM25 corpus → {BM25_PATH}")
    print(f"  Total chunks : {len(all_chunks)}")
    print(f"  Collection count : {collection.count()}")


if __name__ == "__main__":
    main()
