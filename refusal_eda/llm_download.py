#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download

# === Configuration ===
model_to_download = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "google/gemma-3-27b-pt"
]

# MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
CACHE_DIR = os.path.expanduser("~/hf-cache")  # your chosen cache dir

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for MODEL_ID in model_to_download:
        print(f"Downloading model {MODEL_ID} to cache {CACHE_DIR} …")
        snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR
        )
    print("Download done.")

if __name__ == "__main__":
    main()