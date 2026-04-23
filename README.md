# barebones-local-llm-chat


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

