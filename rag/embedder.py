from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)

    def encode(self, texts):
        return self.model.encode(texts)