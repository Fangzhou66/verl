#!/usr/bin/env python3
"""
Test using vLLM's LLM class directly within VERL context to isolate the issue.
"""

import os
import sys

# Add verl to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set environment variables
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['VLLM_XLA_USE_SPMD'] = '0'

def main():
    print("=== Direct LLM Test within VERL Context ===")
    
    # Import vLLM directly
    from vllm import LLM, SamplingParams
    
    # Use same config as vLLM example but with Qwen
    model_name = "Qwen/Qwen3-8B"
    print(f"\nTesting with: {model_name}")
    
    # Minimal config matching vLLM example
    llm_kwargs = {
        "model": model_name,
        "tensor_parallel_size": 8,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 4,
        "max_model_len": 128,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "device": "tpu",
        "disable_custom_all_reduce": True,
    }
    
    print("\nInitializing LLM directly...")
    try:
        llm = LLM(**llm_kwargs)
        print("✓ LLM initialized successfully")
        
        # Simple test
        prompts = ["The capital of France is"]
        sampling_params = SamplingParams(temperature=0.7, max_tokens=16)
        
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            print(f"\nPrompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")
            
        print("\n✓ Direct LLM test successful!")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()