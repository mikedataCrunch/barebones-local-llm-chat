import gradio as gr

from llm.model import LocalLLM
from llm.prompt import build_prompt
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from config import TOP_K

# -------------------------
# Load components
# -------------------------
print("Loading embedder...")
embedder = Embedder()

print("Loading documents...")
with open("data/documents.txt", "r") as f:
    documents = [doc.strip() for doc in f.read().split("\n\n") if doc.strip()]

print(f"Loaded {len(documents)} documents")

print("Embedding documents...")
embeddings = embedder.encode(documents)

print("Building vector store...")
vectorstore = VectorStore(embeddings.shape[1])
vectorstore.add(embeddings)

retriever = Retriever(embedder, vectorstore, documents)

print("Loading LLM...")
llm = LocalLLM()

print("System ready.")

# -------------------------
# Chat function
# -------------------------
def chat_fn(message, history):
    context_docs = retriever.retrieve(message, TOP_K)
    context = "\n".join(context_docs)

    prompt = build_prompt(context, message)

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

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)