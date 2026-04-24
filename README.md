# barebones-local-llm-chat

## Setup

### Option A: pip (recommended on Ubuntu 22.04)
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential cmake ninja-build

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Option B: Conda (optional)
```bash
conda env create -f environment.yml
conda activate barebones-local-llm-chat
```

Notes:
- `llama-cpp-python` may compile native code; the build deps above help if a wheel isn’t available.
- For GPU builds on Ubuntu, you’ll typically want CUDA/toolkit installed and then build `llama-cpp-python` accordingly.
- Gradio defaults to binding `127.0.0.1`. To expose on LAN set `GRADIO_SERVER_NAME=0.0.0.0`. To enable public sharing set `GRADIO_SHARE=true`.
- If you see linker errors about `libgomp.so.1` / OpenMP while installing `llama-cpp-python` (no sudo), install the runtime in your env (e.g. `conda install -c conda-forge libgomp`) or disable OpenMP via `CMAKE_ARGS="-DGGML_OPENMP=OFF"`.
- If Gradio fails with `ImportError: cannot import name 'HfFolder' from huggingface_hub`, downgrade hub: `pip install 'huggingface-hub<0.24'` (already pinned in `requirements.txt` / `environment.yml`).
- If Gradio crashes with `TypeError: unhashable type: 'dict'` while rendering the UI, you likely pinned an incompatible `starlette` (e.g. `starlette==1.x`). This repo constrains `fastapi<1` and `starlette<1` to avoid that.
- If you hit `Model path does not exist`, set `MODEL_PATH` to the actual `.gguf` file (or a directory containing it). Example: `MODEL_PATH=models/TinyLlama-1.1B-Chat-v1.0-GGUF python app.py`.
- If Torch warns about an old CUDA driver, force CPU embeddings via `EMBED_DEVICE=cpu`.

## Fixing bad `fastapi`/`starlette` pins (existing env)

If you previously installed `fastapi==...` / `starlette==...` manually, remove them and reinstall from this repo’s constraints:

```bash
python -m pip uninstall -y fastapi starlette
python -m pip install -U --force-reinstall -r requirements.txt
```

## Restart / run with Gradio env vars

1. Stop the running server (in the terminal that launched it): press `Ctrl+C`.
2. Re-run with the right env vars for your setup:

Local only (default):
```bash
python app.py
```

Remote/container (bind all interfaces):
```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python app.py
```

Remote/container (create a public share link):
```bash
GRADIO_SHARE=true python app.py
```

If generation is extremely slow or appears “stuck”, force CPU-only inference:
```bash
N_GPU_LAYERS=0 GRADIO_SHARE=true python app.py
```


## Models
Conversational: tinyllama-1.1b-chat.Q4_K_M.gguf
Embedder: all-MiniLM-L6-v2
Retreiver: Done using FAISS


## `llama-cpp-python`

Install fixes
1. Fix build dependencies
```
python -m pip install -U \
  "pip>=24" \
  "setuptools>=69" \
  "wheel>=0.43" \
  "packaging>=24" \
  "build>=1.2" \
  "scikit-build-core>=0.10" \
  "cmake>=3.28" \
  "ninja>=1.11"
```
