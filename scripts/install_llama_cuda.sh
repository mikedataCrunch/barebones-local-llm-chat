#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Error: '$python_bin' not found. Activate your environment first." >&2
  exit 1
fi

echo "Using python: $($python_bin -c 'import sys; print(sys.executable)')"
echo "Python version: $($python_bin -V)"

if ! "$python_bin" -c "import pip" >/dev/null 2>&1; then
  echo "Error: pip is not available in this Python environment." >&2
  exit 1
fi

echo "Uninstalling existing llama-cpp-python (if present)..."
"$python_bin" -m pip uninstall -y llama-cpp-python >/dev/null 2>&1 || true

install_with() {
  local cmake_args="$1"
  echo
  echo "Installing llama-cpp-python with CMAKE_ARGS=$cmake_args"
  CMAKE_ARGS="$cmake_args" FORCE_CMAKE=1 \
    "$python_bin" -m pip install --no-cache-dir --force-reinstall llama-cpp-python
}

# llama.cpp / ggml uses different CMake flags across versions/build systems.
# Try a couple common variants.
if ! install_with "-DGGML_CUDA=on"; then
  install_with "-DLLAMA_CUDA=on"
fi

echo
echo "Installed. Quick verification (shared libs present):"
"$python_bin" - <<'PY'
import pathlib, llama_cpp
root = pathlib.Path(llama_cpp.__path__[0])
libs = sorted(root.rglob("libllama.so"))
if not libs:
    print("Warning: could not find libllama.so under", root)
else:
    print("libllama.so:", libs[0])
PY

echo
echo "Next: run your app with GPU layers enabled, e.g.:"
echo "  N_GPU_LAYERS=-1 N_BATCH=2048 python app.py"
