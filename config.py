import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/tinyllama.gguf")

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")

TOP_K = 2
MAX_TOKENS = 200

N_CTX = 2048
N_GPU_LAYERS = -1  # use GPU if available

DOCS_PATH = os.getenv("DOCS_PATH", "data/documents.txt")

GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").strip().lower() in {"1", "true", "yes"}
