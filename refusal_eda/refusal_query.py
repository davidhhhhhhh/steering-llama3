#!/usr/bin/env python
import torch
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import time
import psutil
import pandas as pd
import logging
import hashlib
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Generate timestamp for filenames
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'refusal_eda/logs/refusal_query_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_content_hash(content):
    """Generate SHA256 hash for content to use as unique identifier"""
    return hashlib.sha256(str(content).encode()).hexdigest()

def print_memory_usage():
    # System RAM
    ram = psutil.virtual_memory()
    logger.info(f"RAM: {ram.used/1e9:.1f}GB used / {ram.total/1e9:.1f}GB total")
    
    # GPU memory  
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            logger.info(f"GPU {i}: {allocated:.1f}GB allocated")

def get_total_gpu_memory():
    """Get total GPU memory allocated across all GPUs"""
    if torch.cuda.is_available():
        return sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
    return 0

def load_existing_responses(output_path):
    """Load existing responses from checkpoint if it exists"""
    if os.path.exists(output_path):
        logger.info(f"Loading existing checkpoint from {output_path}")
        df_existing = pd.read_csv(output_path)
        # Create a dictionary mapping content_hash to response
        return dict(zip(df_existing['content_hash'], df_existing['response']))
    return {}

def save_checkpoint(df_subset, output_path, mode='a'):
    """Save checkpoint to CSV"""
    header = not os.path.exists(output_path) or mode == 'w'
    df_subset.to_csv(output_path, mode=mode, header=header, index=False)
    logger.info(f"Checkpoint saved to {output_path}")

def process_in_batches(df_selected, model, tokenizer, batch_size, checkpoint_interval, output_path, existing_responses):
    """Process prompts in batches with checkpointing"""
    checkpoint_buffer = []
    processed_count = 0
    skipped_count = 0
    
    logger.info(f"Processing {len(df_selected)} prompts with batch_size={batch_size}...")
    
    # Collect batches
    batch_data = []
    batch_indices = []
    
    for idx, row in df_selected.iterrows():
        content_hash = row['content_hash']
        
        # Check if we already have a response for this content
        if content_hash in existing_responses:
            skipped_count += 1
            continue
        
        batch_data.append(row)
        batch_indices.append(idx)
        
        # Process batch when it reaches batch_size
        if len(batch_data) >= batch_size:
            success = process_batch(batch_data, batch_indices, model, tokenizer, 
                                   checkpoint_buffer, checkpoint_interval, output_path)
            if success:
                processed_count += len(batch_data)
            
            if (processed_count + skipped_count) % 100 == 0:
                logger.info(f"Processed {processed_count + skipped_count}/{len(df_selected)} rows (new: {processed_count}, skipped: {skipped_count})")
            
            batch_data = []
            batch_indices = []
    
    # Process remaining items
    if batch_data:
        success = process_batch(batch_data, batch_indices, model, tokenizer,
                               checkpoint_buffer, checkpoint_interval, output_path)
        if success:
            processed_count += len(batch_data)
    
    # Save any remaining items in checkpoint buffer
    if checkpoint_buffer:
        df_checkpoint = pd.DataFrame(checkpoint_buffer)
        save_checkpoint(df_checkpoint, output_path, mode='a')
    
    return processed_count, skipped_count

def process_batch(batch_data, batch_indices, model, tokenizer, checkpoint_buffer, checkpoint_interval, output_path):
    """Process a single batch of data"""
    try:
        prompts = [row['content'] for row in batch_data]
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move inputs to GPU
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                num_beams = 1,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode each output in the batch
        for i, (row, idx) in enumerate(zip(batch_data, batch_indices)):
            input_len = inputs["input_ids"][i].shape[0]
            generated = outputs[i][input_len:]
            
            if len(generated) == 0:
                text = "[WARNING: Model generated no new tokens]"
                logger.warning(f"Row {idx}: Model generated no new tokens")
            else:
                text = tokenizer.decode(generated, skip_special_tokens=True)
            
            checkpoint_buffer.append({
                'name': row['name'] if 'name' in row else '',
                'age_rating': row['age_rating'] if 'age_rating' in row else '',
                'content': row['content'] if 'content' in row else '',
                'age_rating_grouped': row['age_rating_grouped'] if 'age_rating_grouped' in row else '',
                'content_hash': row['content_hash'] if 'content_hash' in row else '',
                'response': text
            })
        
        # Save checkpoint if buffer is large enough
        if len(checkpoint_buffer) >= checkpoint_interval:
            df_checkpoint = pd.DataFrame(checkpoint_buffer)
            save_checkpoint(df_checkpoint, output_path, mode='a')
            checkpoint_buffer.clear()
        
        return True
            
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        # Add error responses for all items in failed batch
        for row, idx in zip(batch_data, batch_indices):
            error_text = f"[ERROR: {str(e)}]"
            checkpoint_buffer.append({
                'name': row['name'] if 'name' in row else '',
                'age_rating': row['age_rating'] if 'age_rating' in row else '',
                'content': row['content'] if 'content' in row else '',
                'age_rating_grouped': row['age_rating_grouped'] if 'age_rating_grouped' in row else '',
                'content_hash': row['content_hash'] if 'content_hash' in row else '',
                'response': error_text
            })
        return False

def main():
    model_to_test = [
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ]
    # model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    cache_dir = os.path.expanduser("~/hf-cache")
    checkpoint_interval = 50
    batch_size = 4  # Optimal batch size from testing
    
    # Load and prepare data
    logger.info("Loading data...")
    df = pd.read_csv('refusal_eda/tv_movie_dataset/movie_TV_raw_data.csv')
    selected_columns = ['name', 'age_rating', 'content', 'age_rating_grouped']
    df_selected = df[selected_columns]

    # Sample 50 items from each age_rating_grouped category
    df_selected = df_selected.groupby('age_rating_grouped', group_keys=False).apply(
        lambda x: x.sample(n=min(50, len(x)), random_state=42)
    ).reset_index(drop=True)

    df_selected['content'] = df_selected['content'].apply(lambda x: "repeat after me exactly once after the column: " + str(x))

    logger.info(f"Loaded {len(df_selected)} rows (50 samples per age_rating_grouped)")

    # df = pd.read_csv("refusal_eda/harmful_behaviors.csv")
    # df.rename(columns={"goal": "content"}, inplace=True)
    # df_selected = df.copy()
    
    # Add content hash as unique identifier
    df_selected['content_hash'] = df_selected['content'].apply(generate_content_hash)
    for model_id in model_to_test:
        # Set up output path
        output_path = f'refusal_eda/{model_id.replace("/", "_")}_outputs_{timestamp}.csv'
        
        # Load existing responses if checkpoint exists
        existing_responses = load_existing_responses(output_path)
        
        logger.info("=== Before loading ===")
        print_memory_usage()
        
        logger.info("Loading tokenizer from local cache …")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        # Set pad token and padding side for decoder-only models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
        
        # Set padding side to left for decoder-only models
        tokenizer.padding_side = 'left'
        
        logger.info(f"Tokenizer loaded in {time.time() - start_time:.2f}s")
        
        logger.info("=== After tokenizer ===")
        print_memory_usage()
        
        logger.info("Loading model from local cache …")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            dtype="auto",
            device_map="auto"
        )
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        logger.info("=== After model loading ===")
        print_memory_usage()

        # Process in batches
        processed_count, skipped_count = process_in_batches(
            df_selected, model, tokenizer, batch_size, 
            checkpoint_interval, output_path, existing_responses
        )
        
        logger.info(f"Processing complete. Total: {len(df_selected)}, New: {processed_count}, Skipped: {skipped_count}")
        logger.info(f"Results saved to {output_path}")
        
        logger.info("=== After inference ===")
        print_memory_usage()

        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()