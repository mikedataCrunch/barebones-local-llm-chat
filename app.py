import gradio as gr

from llm.model import LocalLLM
from llm.prompt import build_prompt
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from config import DOCS_PATH, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, GRADIO_SHARE, TOP_K

# -------------------------
# Load components
# -------------------------
def load_documents(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [doc.strip() for doc in f.read().split("\n\n") if doc.strip()]
    except FileNotFoundError:
        print(f"Warning: documents file not found: {path}")
        return []

print("Loading embedder...")
embedder = Embedder()

print("Loading documents...")
documents = load_documents(DOCS_PATH)
print(f"Loaded {len(documents)} documents from {DOCS_PATH}")

retriever = None
if documents:
    print("Embedding documents...")
    embeddings = embedder.encode(documents)

    print("Building vector store...")
    vectorstore = VectorStore(embeddings.shape[1])
    vectorstore.add(embeddings)

    retriever = Retriever(embedder, vectorstore, documents)
else:
    print("Warning: no documents loaded; RAG context will be empty.")

print("Loading LLM...")
llm = LocalLLM()

print("System ready.")

# -------------------------
# Chat function
# -------------------------
def chat_fn(message, history):
    context_docs = retriever.retrieve(message, TOP_K) if retriever else []
    context = "\n".join(context_docs)

    prompt = build_prompt(context, message, history=history)

    response = llm.generate(prompt)

    return response

# -------------------------
# UI
# -------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Local LLM RAG Chat",
    description="Runs fully locally with FAISS + llama.cpp"
)

demo.launch(
    server_name=GRADIO_SERVER_NAME,
    server_port=GRADIO_SERVER_PORT,
    share=GRADIO_SHARE,
)
