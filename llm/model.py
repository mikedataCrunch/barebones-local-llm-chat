from __future__ import annotations

from pathlib import Path

from llama_cpp import Llama
from config import MODEL_PATH, MAX_TOKENS, N_CTX, N_GPU_LAYERS


def _discover_gguf_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file() and root.suffix.lower() == ".gguf":
        return [root]
    if root.is_dir():
        return sorted(root.rglob("*.gguf"))
    return []


def resolve_model_path(model_path: str) -> str:
    """
    Resolve the llama.cpp model file path.

    Accepts either:
    - a direct `.gguf` file path
    - a directory containing one or more `.gguf` files (first match is used)
    - a missing path, in which case we search under `models/` for any `.gguf`
    """
    requested = Path(model_path)
    candidates = _discover_gguf_paths(requested)

    if not candidates:
        candidates = _discover_gguf_paths(Path("models"))

    if not candidates:
        raise ValueError(
            f"Model path does not exist and no .gguf files were found. "
            f"Set MODEL_PATH to a .gguf file (or directory containing one). "
            f"Current MODEL_PATH={model_path!r}"
        )

    return str(candidates[0])


class LocalLLM:
    def __init__(self):
        self.model = Llama(
            model_path=resolve_model_path(MODEL_PATH),
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
