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
