#!/usr/bin/env python3
"""
Simple test for vLLM MPMD mode matching the original vLLM example.
This avoids pre-initializing distributed environment.
"""

import os

# CRITICAL: Set MPMD mode BEFORE any imports
os.environ["VLLM_XLA_USE_SPMD"] = "0"
print("✓ Set VLLM_XLA_USE_SPMD=0 for MPMD mode")

# Direct vLLM import without VERL wrapper
from vllm import LLM, SamplingParams

def main():
    print("\n=== Simple vLLM MPMD Test (tp=8) ===")
    print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
    print(f"VLLM_XLA_USE_SPMD: {os.environ.get('VLLM_XLA_USE_SPMD', 'not set')}")
    
    # Test prompts from vLLM example
    prompts = [
        "A robot may not injure a human being",
        "It is only with the heart that one can see rightly;",
        "The greatest glory in living lies not in never falling,",
    ]
    
    # Expected completions
    answers = [
        " or, through inaction, allow a human being to come to harm.",
        " what is essential is invisible to the eye.",
        " but in rising every time we fall.",
    ]
    
    # Sampling parameters matching vLLM example
    sampling_params = SamplingParams(
        temperature=0, 
        top_p=1.0, 
        n=1, 
        max_tokens=16
    )
    
    # LLM configuration matching vLLM example
    llm_args = {
        "model": "Qwen/Qwen3-8B",  # or "meta-llama/Llama-3.1-8B-Instruct"
        "max_num_batched_tokens": 64,
        "max_num_seqs": 4,
        "max_model_len": 128,
        "tensor_parallel_size": 8,
        "download_dir": "/tmp/models",
        "disable_custom_all_reduce": True,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "enforce_eager": True,  # For testing, should be False in production
    }
    
    print(f"\nInitializing LLM with tp={llm_args['tensor_parallel_size']}...")
    print("Note: This will spawn 8 worker processes using mp backend")
    
    try:
        # Create LLM instance - let vLLM handle all distributed setup
        llm = LLM(**llm_args)
        print("✓ LLM initialized successfully")
        
        # Generate outputs
        print(f"\nGenerating responses for {len(prompts)} prompts...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("-" * 50)
        for i, (output, answer) in enumerate(zip(outputs, answers)):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt {i+1}: {prompt!r}")
            print(f"Generated: {generated_text!r}")
            print(f"Expected: {answer!r}")
            print(f"Match: {generated_text.startswith(answer)}")
            print("-" * 50)
        
        print("\n✓ All generations completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n=== Summary ===")
    print("✓ Successfully ran vLLM with MPMD mode (mp backend)")
    print(f"  - Model: {llm_args['model']}")
    print("  - Tensor parallel size: 8")
    print("  - Backend: mp (multiprocessing)")
    print("  - Each TPU chip runs a separate process")

if __name__ == "__main__":
    main()