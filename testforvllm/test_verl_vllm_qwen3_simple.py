#!/usr/bin/env python3
"""Simple test to verify vLLM works with Qwen3-8B on TPU using VERL's modifications."""

import os
import sys

# Set up environment for TPU
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_SPMD"] = "1"

# Add paths
sys.path.insert(0, "/home/fangzhou/verl")
sys.path.insert(0, "/home/fangzhou/vllm")

def test_vllm_with_verl_modifications():
    """Test vLLM with VERL's device detection for TPU."""
    print("\n=== Testing vLLM with VERL Device Detection ===")
    
    # First verify VERL detects TPU correctly
    from verl.utils.device import get_device_name, is_tpu_available
    print(f"\nVERL Device Detection:")
    print(f"  Device: {get_device_name()}")
    print(f"  TPU Available: {is_tpu_available}")
    
    if get_device_name() != "tpu":
        print("✗ VERL did not detect TPU correctly!")
        return False
    
    print("\n✓ VERL TPU detection working correctly!")
    
    # Now test vLLM directly
    print("\n=== Testing vLLM with Qwen3-8B on TPU ===")
    
    from vllm import LLM, SamplingParams
    
    model_name = "Qwen/Qwen3-8B"
    
    try:
        print(f"\nLoading {model_name} on TPU...")
        print("  tensor_parallel_size: 8")
        print("  dtype: bfloat16")
        print("  max_model_len: 4096")
        
        # Create LLM instance
        llm = LLM(
            model=model_name,
            device="tpu",
            tensor_parallel_size=8,  # TPU v6e-8 has 8 cores
            dtype="bfloat16",
            max_model_len=4096,
            enforce_eager=True,
            trust_remote_code=True,
            download_dir="/tmp/models",
            swap_space=32,
            disable_log_stats=True,
        )
        
        print("\n✓ Model loaded successfully!")
        
        # Test generation
        prompts = [
            "The future of artificial intelligence is",
            "The capital of France is"
        ]
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
        )
        
        print("\nGenerating responses...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("\n=== Generation Results ===")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Generated: {generated_text}")
        
        print("\n✓ Generation completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verl_rollout_components():
    """Test VERL components that would be used for rollout."""
    print("\n=== Testing VERL Rollout Components ===")
    
    try:
        # Test if rollout worker imports work
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
        print("✓ vLLMRollout import successful")
        
        # Test Ray resource allocation
        from verl.single_controller.ray.base import RayResourcePool
        print("✓ RayResourcePool import successful")
        
        # Check that our modifications are in place
        from verl.utils.device import get_nccl_backend
        backend = get_nccl_backend()
        print(f"✓ Backend for TPU: {backend}")
        assert backend == "xla", f"Expected 'xla' backend for TPU, got '{backend}'"
        
        print("\n✓ All VERL components loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading VERL components: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== VERL + vLLM Qwen3-8B TPU Test (Simple) ===")
    
    # Test 1: VERL components
    verl_ok = test_verl_rollout_components()
    
    # Test 2: vLLM with Qwen3-8B
    vllm_ok = test_vllm_with_verl_modifications()
    
    if verl_ok and vllm_ok:
        print("\n✅ SUCCESS: VERL TPU support is working and vLLM can run Qwen3-8B on TPU!")
        print("\nNote: Full VERL rollout integration requires distributed setup with world_size >= tensor_parallel_size")
        print("For production use, you would need to:")
        print("1. Set up Ray cluster with proper TPU resources")
        print("2. Use world_size=8 to match tensor_parallel_size=8")
        print("3. Or reduce tensor_parallel_size to match available world_size")
    else:
        print("\n❌ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()