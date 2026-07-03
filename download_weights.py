"""Download LLaDA-MoE-7B-A1B-Instruct weights from HuggingFace.

Always downloads directly from HF — never from any other source.

Usage:
    python3 download_weights.py --dest ./weights
    python3 download_weights.py --dest ./weights --repo inclusionAI/LLaDA-MoE-7B-A1B-Instruct
"""

import argparse
import os
from huggingface_hub import snapshot_download

DEFAULT_REPO = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"

IGNORE_PATTERNS = [
    "*.msgpack", "*.h5", "flax_model*", "tf_model*",
    "rust_model.ot", "coreml*", "onnx*",
]


def download(repo_id: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {repo_id} → {dest}")
    print("(This is ~15 GB, may take a while)\n")
    snapshot_download(
        repo_id=repo_id,
        local_dir=dest,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"\nDone. Weights at: {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="weights", help="Destination directory")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF repo ID")
    args = ap.parse_args()
    download(args.repo, args.dest)


if __name__ == "__main__":
    main()
