class Retriever:
    def __init__(self, embedder, vectorstore, documents):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.documents = documents

    def retrieve(self, query: str, k: int):
        if not self.documents:
            return []

        k = max(0, min(int(k), len(self.documents)))
        if k == 0:
            return []

        q_emb = self.embedder.encode([query])
        _, indices = self.vectorstore.search(q_emb, k)

        results = []
        for idx in indices[0].tolist():
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results
