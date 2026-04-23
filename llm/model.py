from llama_cpp import Llama
from config import MODEL_PATH, MAX_TOKENS, N_CTX, N_GPU_LAYERS

class LocalLLM:
    def __init__(self):
        self.model = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False
        )

    def generate(self, prompt: str) -> str:
        output = self.model(
            prompt,
            max_tokens=MAX_TOKENS,
            stop=["</s>"]
        )
        return output["choices"][0]["text"].strip()