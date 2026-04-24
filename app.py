import gradio as gr

import re
import time
from pathlib import Path

from llm.model import LocalLLM
from llm.prompt import build_prompt
from rag.embedder import Embedder
from rag.vectorstore import VectorStore
from rag.retriever import Retriever
from rag.index_store import file_sha256, index_paths, load_index
from config import (
    AUTO_BUILD_INDEX,
    DOCS_PATH,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    GRADIO_SHARE,
    INDEX_DIR,
    TOP_K,
)

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

retriever = None
index_p = index_paths(INDEX_DIR)
if index_p["faiss"].exists() and index_p["docs"].exists() and index_p["meta"].exists():
    try:
        print(f"Loading persisted index from {INDEX_DIR}...")
        index, stored_docs, meta = load_index(INDEX_DIR)
        current_sha = file_sha256(DOCS_PATH) if Path(DOCS_PATH).exists() else ""
        if current_sha and meta.docs_sha256 != current_sha:
            print(
                f"Warning: index is out of date for {DOCS_PATH}. "
                f"Run `python -m rag.build_index --docs {DOCS_PATH} --out {INDEX_DIR}`."
            )
        else:
            vectorstore = VectorStore(index.d)
            vectorstore.index = index
            retriever = Retriever(embedder, vectorstore, stored_docs)
            print(f"Loaded index with {len(stored_docs)} documents (dim={index.d})")
    except Exception as e:
        print(f"Warning: failed to load index from {INDEX_DIR}: {type(e).__name__}: {e}")
else:
    print(f"No persisted index found in {INDEX_DIR}")

if retriever is None and AUTO_BUILD_INDEX:
    print("AUTO_BUILD_INDEX=true, building index on startup...")
    documents = load_documents(DOCS_PATH)
    print(f"Loaded {len(documents)} documents from {DOCS_PATH}")

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
    started = time.time()
    print(f"chat_fn start: message_len={len(message) if message else 0}", flush=True)
    try:
        t0 = time.time()
        context_docs = retriever.retrieve(message, TOP_K) if retriever else []
        context = "\n".join(context_docs)
        print(f"chat_fn after retrieve: {time.time()-t0:.2f}s docs={len(context_docs)}", flush=True)

        t1 = time.time()
        prompt = build_prompt(context, message, history=history)
        print(f"chat_fn after prompt: {time.time()-t1:.2f}s prompt_len={len(prompt)}", flush=True)

        partial = ""
        print("chat_fn generation start", flush=True)
        word_re = re.compile(r"\S+\s*")
        carry = ""
        for chunk in llm.generate_stream(prompt):
            carry += chunk

            matches = list(word_re.finditer(carry))
            if not matches:
                continue

            if carry[-1].isspace():
                cutoff = matches[-1].end()
            elif len(matches) >= 2:
                cutoff = matches[-2].end()
            else:
                cutoff = 0

            if cutoff <= 0:
                continue

            ready = carry[:cutoff]
            carry = carry[cutoff:]

            for m in word_re.finditer(ready):
                partial += m.group(0)
                yield partial

        if carry:
            partial += carry
            yield partial
    except Exception as e:
        yield f"Error while generating response: {type(e).__name__}: {e}"
    finally:
        elapsed = time.time() - started
        print(f"chat_fn completed in {elapsed:.2f}s", flush=True)

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
    show_error=True,
)
