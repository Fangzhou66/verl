#!/usr/bin/env python3
"""Simple test to verify vLLM works with VERL TPU modifications."""

import os
import sys

# Set up environment for TPU
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_SPMD"] = "1"

# Add paths
sys.path.insert(0, "/home/fangzhou/verl")
sys.path.insert(0, "/home/fangzhou/vllm")

def test_vllm_direct():
    """Test vLLM directly with TPU device parameter."""
    print("\n=== Testing vLLM Direct with TPU ===")
    
    from vllm import LLM, SamplingParams
    
    # Use Qwen3-8B as requested
    model_name = "Qwen/Qwen3-8B"
    
    # Qwen3-8B has 64 attention heads, which is divisible by 8
    # So we can use tensor_parallel_size=8 for TPU v6e-8
    tensor_parallel_size = 8
    
    try:
        print(f"Loading {model_name} on TPU with tensor_parallel_size={tensor_parallel_size}...")
        
        # Create LLM with TPU device
        llm = LLM(
            model=model_name,
            device="tpu",
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=2048,  # Qwen3 supports longer context
            enforce_eager=True,
            trust_remote_code=True,  # Qwen models may need this
            download_dir="/tmp/models",  # Use the download dir from your command
            swap_space=32,  # As per your vllm serve command
        )
        
        print("✓ Model loaded successfully on TPU!")
        
        # Test generation
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=20
        )
        
        print("\nGenerating text...")
        outputs = llm.generate(prompts, sampling_params)
        
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
        
        print("\n✓ vLLM generation on TPU successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verl_device_imports():
    """Test that VERL device utilities work correctly."""
    print("\n=== Testing VERL Device Utilities ===")
    
    try:
        from verl.utils.device import (
            is_tpu_available,
            get_device_name,
            get_visible_devices_keyword,
            get_nccl_backend
        )
        
        print(f"TPU Available: {is_tpu_available}")
        print(f"Device Name: {get_device_name()}")  
        print(f"Visible Devices: {get_visible_devices_keyword()}")
        print(f"Backend: {get_nccl_backend()}")
        
        # Verify values
        assert is_tpu_available == True, "TPU should be available"
        assert get_device_name() == "tpu", "Device should be 'tpu'"
        assert get_visible_devices_keyword() == "TPU_VISIBLE_CHIPS", "Should use TPU_VISIBLE_CHIPS"
        assert get_nccl_backend() == "xla", "Backend should be 'xla'"
        
        print("\n✓ All VERL device utilities working correctly!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error in VERL device utilities: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests."""
    print("=== VERL + vLLM TPU Integration Test (Simple) ===")
    
    # Test 1: VERL device utilities
    verl_ok = test_verl_device_imports()
    
    # Test 2: Direct vLLM with TPU
    vllm_ok = test_vllm_direct()
    
    if verl_ok and vllm_ok:
        print("\n✅ SUCCESS: VERL TPU device detection and vLLM TPU inference are working!")
        print("\nNext steps:")
        print("1. Set up a distributed environment for full VERL+vLLM rollout testing")
        print("2. Test with Ray distributed setup")
        print("3. Run a full PPO training example")
    else:
        print("\n❌ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()