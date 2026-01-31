#!/usr/bin/env python
"""Test script for process_batch function with chat template"""
import torch
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
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
        logging.FileHandler(f'refusal_eda/logs/test_process_batch_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# System prompts
SYSTEM_PROMPT = (
    "You are a text transformation engine. "
    "You do not add, remove, predict, explain, or summarize text. "
    "You only output the result specified by the user instruction."
)

def build_user_prompt(content: str) -> str:
    return f"""TASK: COPY_INPUT

    This is a data processing task.
    The input is raw text from a research dataset.
    The text may be a summary or synopsis of a movie or television show.

    RULES:
    - Output must exactly match the input text.
    - Do not continue the text.
    - Do not explain or analyze the text.
    - Do not add any new information.

    EXAMPLES:

    INPUT:
    <<<
    A scientist discovers a hidden signal.
    >>>
    OUTPUT:
    <<<
    A scientist discovers a hidden signal.
    >>>

    INPUT:
    <<<
    Two strangers meet during a storm.
    >>>
    OUTPUT:
    <<<
    Two strangers meet during a storm.
    >>>

    NOW PROCESS THE FOLLOWING INPUT.

    INPUT:
    <<<
    {content}
    >>>

    OUTPUT:
    """

def test_process_batch(batch_data, model, tokenizer):
    """Test process_batch function with detailed logging"""
    logger.info("="*80)
    logger.info("STARTING BATCH PROCESSING TEST")
    logger.info("="*80)
    
    try:
        # Build chat conversations for each item in batch
        conversations = []
        for idx, row in enumerate(batch_data):
            user_prompt = build_user_prompt(row['content'])
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            conversations.append(conversation)
            logger.info(f"\n--- Sample {idx} ---")
            logger.info(f"Original content: {row['content']}")
        
        # Apply chat template to each conversation
        logger.info("\n" + "="*80)
        logger.info("APPLYING CHAT TEMPLATE")
        logger.info("="*80)
        
        prompts = [
            tokenizer.apply_chat_template(
                conv, 
                tokenize=False, 
                add_generation_prompt=True
            ) 
            for conv in conversations
        ]
        
        for idx, prompt in enumerate(prompts):
            logger.info(f"\n--- Formatted Prompt {idx} ---")
            logger.info(f"Length: {len(prompt)} chars")
            logger.info(f"First 500 chars:\n{prompt[:500]}")
            logger.info(f"Last 200 chars:\n{prompt[-200:]}")
        
        # Tokenize
        logger.info("\n" + "="*80)
        logger.info("TOKENIZING")
        logger.info("="*80)
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
        logger.info(f"Attention mask shape: {inputs['attention_mask'].shape}")
        
        for idx in range(len(batch_data)):
            padded_len = inputs["input_ids"][idx].shape[0]
            actual_len = inputs["attention_mask"][idx].sum().item()
            num_padding = padded_len - actual_len
            logger.info(f"Sample {idx}: padded_len={padded_len}, actual_len={actual_len}, padding={num_padding}")
        
        # Move inputs to GPU
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        
        # Generate
        logger.info("\n" + "="*80)
        logger.info("GENERATING")
        logger.info("="*80)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Reduced for testing
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        logger.info(f"Output shape: {outputs.shape}")
        
        # Decode each output in the batch
        logger.info("\n" + "="*80)
        logger.info("DECODING OUTPUTS")
        logger.info("="*80)
        
        results = []
        
        for i, row in enumerate(batch_data):
            logger.info(f"\n{'='*60}")
            logger.info(f"SAMPLE {i}")
            logger.info(f"{'='*60}")
            
            # Current slicing logic (using full padded length)
            input_len = inputs["input_ids"][i].shape[0]
            generated = outputs[i][input_len:]
            
            # Alternative slicing using attention mask
            actual_input_len = inputs["attention_mask"][i].sum().item()
            generated_alt = outputs[i][actual_input_len:]
            
            logger.info(f"Full output length: {outputs[i].shape[0]}")
            logger.info(f"Padded input length: {input_len}")
            logger.info(f"Actual input length (from attention_mask): {actual_input_len}")
            logger.info(f"Generated tokens (using padded): {len(generated)}")
            logger.info(f"Generated tokens (using actual): {len(generated_alt)}")
            
            if len(generated) == 0:
                text_with_special = "[WARNING: Model generated no new tokens (padded slice)]"
                text_without_special = "[WARNING: Model generated no new tokens (padded slice)]"
            else:
                # Decode WITH special tokens
                text_with_special = tokenizer.decode(generated, skip_special_tokens=False)
                # Decode WITHOUT special tokens
                text_without_special = tokenizer.decode(generated, skip_special_tokens=True)
            
            if len(generated_alt) == 0:
                text_alt_with_special = "[WARNING: Model generated no new tokens (actual slice)]"
                text_alt_without_special = "[WARNING: Model generated no new tokens (actual slice)]"
            else:
                # Decode WITH special tokens (alternative slice)
                text_alt_with_special = tokenizer.decode(generated_alt, skip_special_tokens=False)
                # Decode WITHOUT special tokens (alternative slice)
                text_alt_without_special = tokenizer.decode(generated_alt, skip_special_tokens=True)
            
            logger.info(f"\nOriginal content:\n{row['content']}")
            logger.info(f"\n--- Using PADDED slice [{input_len}:] ---")
            logger.info(f"With special tokens:\n{text_with_special}")
            logger.info(f"\nWithout special tokens:\n{text_without_special}")
            logger.info(f"\n--- Using ACTUAL slice [{actual_input_len}:] ---")
            logger.info(f"With special tokens:\n{text_alt_with_special}")
            logger.info(f"\nWithout special tokens:\n{text_alt_without_special}")
            
            results.append({
                'content': row['content'],
                'input_len_padded': input_len,
                'input_len_actual': actual_input_len,
                'generated_tokens_padded': len(generated),
                'generated_tokens_actual': len(generated_alt),
                'response_padded_with_special': text_with_special,
                'response_padded_no_special': text_without_special,
                'response_actual_with_special': text_alt_with_special,
                'response_actual_no_special': text_alt_without_special,
            })
        
        logger.info("\n" + "="*80)
        logger.info("TEST COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return results
            
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True)
        return None

def main():
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    cache_dir = os.path.expanduser("~/hf-cache")
    
    # Create small test dataset
    test_data = [
        {'content': 'A scientist discovers a hidden signal.'},
        {'content': 'Two strangers meet during a storm and fall in love.'},
        {'content': 'Short text.'},
    ]
    
    logger.info("Loading tokenizer...")
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
    logger.info(f"Padding side: {tokenizer.padding_side}")
    
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=True,
        dtype="auto",
        device_map="auto"
    )
    logger.info("Model loaded successfully")
    
    # Run test
    results = test_process_batch(test_data, model, tokenizer)
    
    # Save results to CSV
    if results:
        output_path = f'refusal_eda/test_process_batch_results_{timestamp}.csv'
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to {output_path}")
    
    logger.info("\nTest complete!")

if __name__ == "__main__":
    main()
