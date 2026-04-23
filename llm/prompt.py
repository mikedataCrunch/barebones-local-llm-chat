def build_prompt(context: str, query: str, history=None) -> str:
    history_text = ""
    if history:
        # `history` is a list of (user, assistant) tuples in Gradio.
        turns = []
        for user_msg, assistant_msg in history[-5:]:
            turns.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
        history_text = "\n\nChat history:\n" + "\n\n".join(turns) + "\n"

    return f"""
You are a helpful assistant. Answer ONLY using the context provided.
If the answer is not in the context, say "I don't know".

Context:
{context}
{history_text}

Question:
{query}

Answer:
"""
