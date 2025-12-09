#!/usr/bin/env python
import torch
import os
import time
import psutil
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def print_memory_usage():
    # System RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.used/1e9:.1f}GB used / {ram.total/1e9:.1f}GB total")
    
    # GPU memory  
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

def main():
    model_id = "meta-llama/Meta-Llama-3-70B"
    cache_dir = os.path.expanduser("~/hf-cache")
    
    # Load and prepare data
    print("Loading data...")
    df = pd.read_csv('refusal_eda/tv_movie_dataset/movie_TV_raw_data.csv')
    selected_columns = ['name', 'age_rating', 'content', 'age_rating_grouped']
    df_selected = df[selected_columns]
    df_selected['content'] = df_selected['content'].apply(lambda x: "repeat after me: " + str(x))
    
    print(f"Loaded {len(df_selected)} rows")
    
    print("\n=== Before loading ===")
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
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    print("\n=== After model loading ===")
    print_memory_usage()

    # Generate responses for each row
    responses = []
    print(f"\nGenerating responses for {len(df_selected)} prompts...")
    
    for idx, row in df_selected.iterrows():
        prompt = row['content']
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(text)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df_selected)} rows")
    
    # Add responses to dataframe
    df_selected['response'] = responses
    
    # Save to CSV
    output_path = f'refusal_eda/{model_id.replace("/", "_")}_outputs.csv'
    df_selected.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    print("\n=== After inference ===")
    print_memory_usage()

if __name__ == "__main__":
    main()