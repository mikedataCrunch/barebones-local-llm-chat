"""
Utilities for downloading one or more Hugging Face model repositories to a local
directory using `huggingface_hub.snapshot_download`.

This script is designed for simple CLI use in headless environments such as
remote servers, GPU notebooks, and terminal-only sessions.

Key behaviors:
- Downloads each requested repo serially.
- Stores downloaded files under:
      <local_dir>/<repo_name>/
- Optionally uses a custom Hugging Face cache directory.
- Supports non-interactive authentication via the HF_TOKEN environment variable.
- Falls back to interactive login if no token is available.

Example:
    export HF_TOKEN=hf_xxx_your_token_here

    python utils/hf_model_downloader.py \
        --cache_dir /path/to/.huggingface_mirror \
        --local_dir /path/to/models \
        --model_ids TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
                    sentence-transformers/all-MiniLM-L6-v2
                    
    # Actual (CHROMA machine)                
    python utils/hf_model_downloader.py \
        --cache_dir /home/jovyan/barebones-local-llm-chat/.huggingface_mirror \
        --local_dir /home/jovyan/barebones-local-llm-chat/models \
        --model_ids TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
                    sentence-transformers/all-MiniLM-L6-v2

Notes:
- `snapshot_download()` downloads a full repository snapshot.
- For GGUF repos, this may download multiple files unless you later add
  `allow_patterns` filtering.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, Optional

import huggingface_hub as hh


def get_repo_output_dir(local_dir: str, model_id: str) -> str:
    """
    Build the output directory for a given Hugging Face repo ID.

    The downloaded repository is placed under:
        <local_dir>/<repo_name>

    where <repo_name> is the final component of the repo ID.

    Args:
        local_dir: Root directory where downloaded repos should be stored.
        model_id: Hugging Face repo ID, e.g. "org/model-name".

    Returns:
        Absolute or relative path to the target directory for that repo.
    """
    repo_name = model_id.split("/")[-1]
    return os.path.join(local_dir, repo_name)


def ensure_directory(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def resolve_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Resolve the Hugging Face authentication token to use.

    Resolution order:
    1. Explicit token argument
    2. HF_TOKEN environment variable
    3. None

    Args:
        explicit_token: Token explicitly provided by the caller.

    Returns:
        Token string if available, otherwise None.
    """
    if explicit_token:
        return explicit_token

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    return None


def login_if_needed(token: Optional[str]) -> Optional[str]:
    """
    Ensure authentication is available before download.

    If a token is provided, it is returned unchanged and no interactive login
    occurs. If no token is provided, this function attempts interactive login.

    This is useful for terminal-only environments where private or gated models
    may require authentication.

    Args:
        token: Existing Hugging Face token, if any.

    Returns:
        The token if one was provided, otherwise None.

    Raises:
        Any exception raised by `huggingface_hub.interpreter_login()` if
        interactive login fails.
    """
    if token:
        return token

    print("No HF_TOKEN found. Starting interactive Hugging Face login...")
    hh.interpreter_login()
    return None


def download_model(
    model_id: str,
    cache_dir: Optional[str],
    local_dir: str,
    token: Optional[str] = None,
) -> None:
    """
    Download a Hugging Face repository snapshot to a local directory.

    The repo will be downloaded into:
        <local_dir>/<repo_name>

    Args:
        model_id: Hugging Face repo ID, e.g. "sentence-transformers/all-MiniLM-L6-v2".
        cache_dir: Optional Hugging Face cache directory.
        local_dir: Root directory to store downloaded repos.
        token: Optional Hugging Face token for private/gated repos.

    Side Effects:
        - Creates output directories if needed.
        - Downloads repository files to disk.
        - Prints status and elapsed time to stdout.
    """
    start_time = time.time()
    output_dir = get_repo_output_dir(local_dir, model_id)
    ensure_directory(local_dir)

    print(f"Target directory: {output_dir}")

    try:
        hh.snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=output_dir,
            token=token,
        )
    except Exception as exc:
        elapsed = time.time() - start_time
        print(
            f"Error downloading {model_id} after {elapsed / 60:.2f} minutes: {exc}"
        )
        return

    elapsed = time.time() - start_time
    print(f"Downloaded {model_id} in {elapsed / 60:.2f} minutes")


def download_models_serial(
    model_ids: Iterable[str],
    cache_dir: Optional[str],
    local_dir: str,
    token: Optional[str] = None,
) -> None:
    """
    Download multiple Hugging Face repositories one at a time.

    Serial downloading is simple and robust for proof-of-concept workflows and
    avoids extra complexity around parallel downloads.

    Args:
        model_ids: Iterable of Hugging Face repo IDs.
        cache_dir: Optional Hugging Face cache directory.
        local_dir: Root directory to store downloaded repos.
        token: Optional Hugging Face token for private/gated repos.
    """
    model_ids = list(model_ids)

    for index, model_id in enumerate(model_ids, start=1):
        print(f"Downloading ({index} of {len(model_ids)}): {model_id}")
        download_model(
            model_id=model_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            token=token,
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the downloader script.

    Returns:
        Parsed CLI arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download one or more Hugging Face model repositories."
    )

    parser.add_argument(
        "--model_ids",
        nargs="+",
        default=[],
        help="One or more Hugging Face repo IDs to download.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        default=None,
        help="Optional Hugging Face cache directory.",
    )

    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Directory where downloaded repositories will be stored.",
    )

    parser.add_argument(
        "--token",
        type=str,
        required=False,
        default=None,
        help=(
            "Optional Hugging Face token. If omitted, the script will look for "
            "HF_TOKEN in the environment, then fall back to interactive login."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point for CLI execution.

    Workflow:
    1. Parse CLI arguments.
    2. Resolve authentication token from CLI or environment.
    3. Trigger interactive login if no token is available.
    4. Download each requested model repository serially.

    Side Effects:
        - Prompts for login if needed.
        - Downloads files to disk.
        - Prints progress to stdout.
    """
    args = parse_args()

    model_ids = args.model_ids
    cache_dir = args.cache_dir
    local_dir = args.local_dir
    token = resolve_token(args.token)

    if not model_ids:
        print("No model ids provided.")
        return

    token = login_if_needed(token)

    download_models_serial(
        model_ids=model_ids,
        cache_dir=cache_dir,
        local_dir=local_dir,
        token=token,
    )


if __name__ == "__main__":
    main()