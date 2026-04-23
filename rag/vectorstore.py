import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings):
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {vectors.shape}")
        self.index.add(vectors)

    def search(self, query_emb, k: int):
        query = np.asarray(query_emb, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2:
            raise ValueError(f"query_emb must be 1D or 2D, got shape {query.shape}")
        return self.index.search(query, k)
