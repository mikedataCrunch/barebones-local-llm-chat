def build_prompt(context: str, query: str) -> str:
    return f"""
You are a helpful assistant. Answer ONLY using the context provided.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""