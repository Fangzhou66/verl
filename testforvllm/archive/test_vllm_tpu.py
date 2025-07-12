#!/usr/bin/env python3
"""Test script to verify vLLM works on TPU."""

import os
import sys

# Set TPU environment variables
os.environ["PJRT_DEVICE"] = "TPU"

# Add local vLLM to path
sys.path.insert(0, "/home/fangzhou/vllm")

try:
    import torch_xla
    print(f"✓ torch_xla imported successfully")
    import torch_xla.core.xla_model as xm
    print(f"✓ XLA device available: {xm.xla_device()}")
except ImportError as e:
    print(f"✗ Failed to import torch_xla: {e}")
    print("Please ensure torch_xla is installed for TPU support")
    sys.exit(1)

try:
    from vllm import LLM, SamplingParams
    print("✓ vLLM imported successfully")
except ImportError as e:
    print(f"✗ Failed to import vLLM: {e}")
    sys.exit(1)

# Test basic model loading and inference
def test_vllm_tpu():
    print("\n--- Testing vLLM on TPU ---")
    
    # Use a small model for testing
    model_name = "facebook/opt-125m"  # Small model for quick testing
    
    try:
        print(f"\n1. Loading model {model_name} on TPU...")
        # For opt-125m with 12 attention heads, we need a divisor of 12
        # TPU v6e-8 has 8 cores, but we can use fewer for tensor parallelism
        llm = LLM(
            model=model_name,
            device="tpu",
            tensor_parallel_size=4,  # 12 heads / 4 = 3 heads per device
            dtype="bfloat16",  # TPU prefers bfloat16
            max_model_len=512,  # Reduce for faster testing
        )
        print("✓ Model loaded successfully!")
        
        print("\n2. Running inference...")
        prompts = ["Hello, my name is", "The capital of France is"]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
        
        outputs = llm.generate(prompts, sampling_params)
        
        print("\n3. Inference results:")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 50)
        
        print("\n✓ vLLM inference on TPU successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during vLLM TPU test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check TPU availability
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"TPU device: {device}")
        print(f"Number of TPU cores: {xm.xrt_world_size()}")
    except Exception as e:
        print(f"Failed to get TPU info: {e}")
    
    # Run the test
    success = test_vllm_tpu()
    sys.exit(0 if success else 1)