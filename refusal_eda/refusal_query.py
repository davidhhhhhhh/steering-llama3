#!/usr/bin/env python
import torch
import os
import time
import psutil
import pandas as pd
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# Generate timestamp for filenames
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'refusal_eda/refusal_query_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_memory_usage():
    # System RAM
    ram = psutil.virtual_memory()
    logger.info(f"RAM: {ram.used/1e9:.1f}GB used / {ram.total/1e9:.1f}GB total")
    
    # GPU memory  
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

def main():
    model_id = "meta-llama/Meta-Llama-3-70B"
    cache_dir = os.path.expanduser("~/hf-cache")
    
    # Load and prepare data
    logger.info("Loading data...")
    df = pd.read_csv('refusal_eda/tv_movie_dataset/movie_TV_raw_data.csv')
    selected_columns = ['name', 'age_rating', 'content', 'age_rating_grouped']
    df_selected = df[selected_columns]
    df_selected['content'] = df_selected['content'].apply(lambda x: "repeat after me: " + str(x))
    
    logger.info(f"Loaded {len(df_selected)} rows")
    
    logger.info("=== Before loading ===")
    print_memory_usage()
    
    logger.info("Loading tokenizer from local cache …")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True
    )
    logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f}s")
    
    logger.info("=== After tokenizer ===")
    print_memory_usage()
    
    logger.info("Loading model from local cache …")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto"
    )
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s")
    
    logger.info("=== After model loading ===")
    print_memory_usage()

    # Generate responses for each row
    responses = []
    logger.info(f"Generating responses for {len(df_selected)} prompts...")
    
    for idx, row in df_selected.iterrows():
        prompt = row['content']
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048, # max char length is 1498 in our dataset, so 2048 tokens cap is safe
            )

            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            
            # Check if model generated nothing (output length == input length)
            if len(generated) == 0:
                text = "[WARNING: Model generated no new tokens]"
                logger.warning(f"Row {idx}: Model generated no new tokens")
            else:
                text = tokenizer.decode(generated, skip_special_tokens=True)
            
            responses.append(text)
        except Exception as e:
            logger.error(f"Error at row {idx}: {e}")
            responses.append(f"[ERROR: {str(e)}]")
        
        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{len(df_selected)} rows")
    
    # Add responses to dataframe
    df_selected['response'] = responses
    
    # Save to CSV with timestamp
    output_path = f'refusal_eda/{model_id.replace("/", "_")}_outputs_{timestamp}.csv'
    df_selected.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    logger.info("=== After inference ===")
    print_memory_usage()

if __name__ == "__main__":
    main()