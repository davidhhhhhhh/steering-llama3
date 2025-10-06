#!/usr/bin/env python
import torch
import os
import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

def print_memory_usage():
    # System RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.used/1e9:.1f}GB used / {ram.total/1e9:.1f}GB total")
    
    # GPU memory  
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

def main():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    cache_dir = os.path.expanduser("~/hf-cache")
    
    print("=== Before loading ===")
    print_memory_usage()
    
    print("\nLoading tokenizer from local cache …")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True
    )
    print(f"Tokenizer loaded in {time.time() - start_time:.2f}s")
    
    print("\n=== After tokenizer ===")
    print_memory_usage()
    
    print("\nLoading model from local cache …")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,  # Fixed: was 'dtype'
        device_map="cuda"
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    print("\n=== After model loading ===")
    print_memory_usage()

    prompt = "Hello, how are you today?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\nGenerating …")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print("Response:", text)
    
    print("\n=== After inference ===")
    print_memory_usage()

if __name__ == "__main__":
    main()