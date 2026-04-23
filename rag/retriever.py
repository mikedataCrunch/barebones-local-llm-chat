class Retriever:
    def __init__(self, embedder, vectorstore, documents):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.documents = documents

    def retrieve(self, query: str, k: int):
        q_emb = self.embedder.encode([query])
        D, I = self.vectorstore.search(q_emb, k)
        return [self.documents[i] for i in I[0]]