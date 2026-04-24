from __future__ import annotations

import argparse
import os
from pathlib import Path


def load_documents(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [doc.strip() for doc in f.read().split("\n\n") if doc.strip()]


def build_index(docs_path: str, index_dir: str) -> None:
    from config import EMBED_MODEL
    from rag.embedder import Embedder
    from rag.index_store import IndexMeta, file_sha256, save_index
    from rag.vectorstore import VectorStore

    docs_path_p = Path(docs_path)
    if not docs_path_p.exists():
        raise FileNotFoundError(f"Documents file not found: {docs_path}")

    print("Loading embedder...")
    embedder = Embedder()

    print(f"Loading documents from {docs_path}...")
    documents = load_documents(docs_path)
    if not documents:
        raise ValueError(f"No documents found in {docs_path}")
    print(f"Loaded {len(documents)} documents")

    print("Embedding documents...")
    embeddings = embedder.encode(documents)

    print("Building FAISS index...")
    vectorstore = VectorStore(embeddings.shape[1])
    vectorstore.add(embeddings)

    meta = IndexMeta(
        docs_path=str(docs_path),
        docs_sha256=file_sha256(docs_path),
        embed_model=str(EMBED_MODEL),
        dim=int(embeddings.shape[1]),
    )

    print(f"Saving index to {index_dir}...")
    save_index(index_dir, index=vectorstore.index, documents=documents, meta=meta)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist a FAISS index for local RAG.")
    parser.add_argument("--docs", default=os.getenv("DOCS_PATH", "data/documents.txt"), help="Path to documents.txt")
    parser.add_argument("--out", default=os.getenv("INDEX_DIR", "data/index"), help="Directory to write index files into")
    parser.add_argument("--embed-model", default=None, help="Override EMBED_MODEL for this run")
    args = parser.parse_args()

    if args.embed_model:
        os.environ["EMBED_MODEL"] = args.embed_model

    build_index(args.docs, args.out)


if __name__ == "__main__":
    main()
