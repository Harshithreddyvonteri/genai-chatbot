"""
embedder.py
-----------
Reads the cleaned corpus (processed_v2.jsonl), chunks the text, embeds each chunk
using a local sentence-transformers model, and stores them in ChromaDB.

Usage:
    python embedder.py --corpus data/processed/processed_v2.jsonl --persist-dir data/vectorstore/v2_chroma
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# ----------------------
# Chunking Function
# ----------------------
def chunk_text(text: str, max_words: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of ~max_words, with overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += max_words - overlap
    return chunks


def load_corpus(path: Path) -> List[Dict]:
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to processed corpus JSONL (e.g., processed_v2.jsonl)")
    ap.add_argument("--persist-dir", required=True, help="Folder to store ChromaDB index")
    ap.add_argument("--chunk-size", type=int, default=300, help="Chunk size in words")
    ap.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in words")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading corpus: {corpus_path}")
    docs = load_corpus(corpus_path)
    print(f"[INFO] Loaded {len(docs)} documents.")

    # Initialize embedding model
    print("[INFO] Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Initialize Chroma
    print(f"[INFO] Initializing Chroma at {persist_dir}")
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="gitlab_v2")

    chunk_count = 0

    for doc in tqdm(docs, desc="Processing docs"):
        doc_id = doc["doc_id"]
        url = doc.get("url", "")
        title = doc.get("title", "")
        text = doc["text"]

        chunks = chunk_text(text, max_words=args.chunk_size, overlap=args.chunk_overlap)

        # Compute embeddings for chunks
        embeddings = model.encode(chunks, convert_to_numpy=True).tolist()

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_{idx}"
            metadata = {"url": url, "title": title, "chunk_index": idx}
            collection.add(ids=[chunk_id], embeddings=[emb], documents=[chunk], metadatas=[metadata])
            chunk_count += 1

    print(f"[INFO] Indexed {chunk_count} chunks from {len(docs)} documents.")
    print(f"[INFO] ChromaDB persisted at {persist_dir}")


if __name__ == "__main__":
    main()

