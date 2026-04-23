from sentence_transformers import SentenceTransformer
from config import EMBED_DEVICE, EMBED_MODEL

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)

    def encode(self, texts):
        # SentenceTransformer returns a numpy array; force float32 for FAISS compatibility.
        return self.model.encode(texts).astype("float32", copy=False)
