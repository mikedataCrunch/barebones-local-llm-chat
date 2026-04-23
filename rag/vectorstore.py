import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)

    def add(self, embeddings):
        self.index.add(np.array(embeddings))

    def search(self, query_emb, k: int):
        return self.index.search(np.array(query_emb), k)