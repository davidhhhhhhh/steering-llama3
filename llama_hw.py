#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # If you have a token (for gated models), you may set the HF token via environment or arguments
    # Example:
    # import os
    # os.environ["HF_TOKEN"] = "your_token_here"

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # use bfloat16 if supported on your GPU
        device_map="cuda"
    )

    prompt = "Hello, how are you today?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating …")
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

if __name__ == "__main__":
    main()