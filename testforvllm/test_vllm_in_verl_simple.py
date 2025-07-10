#!/usr/bin/env python3
"""Test vLLM running in VERL context with proper SPMD settings."""

import os
import sys

# CRITICAL: Set TPU environment correctly
os.environ["PJRT_DEVICE"] = "TPU"
# For vLLM on TPU, XLA_USE_SPMD should be 0 to avoid conflicts
os.environ["XLA_USE_SPMD"] = "0"  # Disable SPMD for vLLM compatibility
# Additional TPU compatibility settings
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"

# Add paths
sys.path.insert(0, "/home/fangzhou/verl")
sys.path.insert(0, "/home/fangzhou/vllm")

def test_vllm_in_verl_context():
    """Test vLLM with VERL modifications."""
    print("\n=== Testing vLLM in VERL Context ===")
    
    # First verify VERL detects TPU
    try:
        from verl.utils.device import get_device_name, is_tpu_available
        print(f"\nVERL Device Detection:")
        print(f"  Device: {get_device_name()}")
        print(f"  TPU Available: {is_tpu_available}")
        
        if get_device_name() != "tpu":
            print("✗ VERL did not detect TPU!")
            return False
    except ImportError as e:
        print(f"Note: Some VERL imports not available ({e}), continuing with vLLM test...")
    
    # Test vLLM directly
    from vllm import LLM, SamplingParams
    
    print("\n=== Testing vLLM with Qwen Model ===")
    
    # Use Qwen3-8B model for testing
    model_name = "Qwen/Qwen3-8B"  # Qwen3-8B (32 Q heads, 8 KV heads)
    
    print("Environment:")
    print(f"  XLA_USE_SPMD: {os.environ.get('XLA_USE_SPMD')}")
    print(f"  PJRT_DEVICE: {os.environ.get('PJRT_DEVICE')}")
    print(f"  Model: {model_name}")
    
    try:
        
        print(f"\nLoading {model_name} on TPU...")
        llm = LLM(
            model=model_name,
            device="tpu",
            tensor_parallel_size=8,  # Qwen3 models work with TP=8
            dtype="bfloat16",
            max_model_len=512,
            enforce_eager=True,
            disable_log_stats=True,
        )
        
        print("✓ Model loaded successfully!")
        
        # Quick inference test
        prompts = ["Hello, world!"]
        sampling_params = SamplingParams(temperature=0.8, max_tokens=10)
        
        print("\nRunning inference...")
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Generated: {output.outputs[0].text}")
        
        print("\n✅ vLLM works in VERL context with SPMD disabled!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("=== vLLM in VERL Context Test ===")
    print("\nThis test verifies that vLLM can run within VERL's environment.")
    print("Key requirement: XLA_USE_SPMD=0 must be set!")
    
    success = test_vllm_in_verl_context()
    
    if success:
        print("\n✅ SUCCESS: vLLM runs successfully in VERL context!")
        print("\nTo use vLLM in VERL:")
        print("1. Always set XLA_USE_SPMD=0")
        print("2. Use the tpu_rollout.yaml configuration")
        print("3. Ensure your VERL scripts set the environment variable")
    else:
        print("\n❌ Test failed.")


if __name__ == "__main__":
    main()