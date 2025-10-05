#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# === Configuration ===
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_DIR = os.path.expanduser("~/hf-cache")  # your chosen cache dir

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Downloading model {MODEL_ID} to cache {CACHE_DIR} …")
    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=False,
        # you may optionally restrict files to weight + tokenizer:
        # allow_patterns=["*.bin", "*.json", "tokenizer*", "config*"]
    )
    print("Download done.")

if __name__ == "__main__":
    main()