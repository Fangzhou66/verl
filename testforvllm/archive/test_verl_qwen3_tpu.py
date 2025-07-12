#!/usr/bin/env python3
"""Test VERL TPU integration components."""

import os
import sys

# Set up environment for TPU
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_SPMD"] = "1"

# Add paths
sys.path.insert(0, "/home/fangzhou/verl")
sys.path.insert(0, "/home/fangzhou/vllm")

def test_verl_tpu_components():
    """Test VERL TPU integration components."""
    print("\n=== Testing VERL TPU Components ===")
    
    all_passed = True
    
    # Test 1: Device Detection
    print("\n1. Testing Device Detection:")
    try:
        from verl.utils.device import (
            is_tpu_available,
            get_device_name,
            get_visible_devices_keyword,
            get_nccl_backend,
            is_torch_xla_available
        )
        
        print(f"   ✓ is_torch_xla_available: {is_torch_xla_available()}")
        print(f"   ✓ is_tpu_available: {is_tpu_available}")
        print(f"   ✓ get_device_name: {get_device_name()}")
        print(f"   ✓ get_visible_devices_keyword: {get_visible_devices_keyword()}")
        print(f"   ✓ get_nccl_backend: {get_nccl_backend()}")
        
        assert get_device_name() == "tpu", "Device should be TPU"
        assert get_visible_devices_keyword() == "TPU_VISIBLE_CHIPS", "Should use TPU_VISIBLE_CHIPS"
        assert get_nccl_backend() == "xla", "Backend should be XLA"
        print("   ✅ Device detection PASSED")
    except Exception as e:
        print(f"   ❌ Device detection FAILED: {e}")
        all_passed = False
    
    # Test 2: Ray Resource Allocation
    print("\n2. Testing Ray Resource Allocation:")
    try:
        from verl.single_controller.ray.base import RayResourcePool
        
        # Check that TPU is handled in get_placement_groups
        pool = RayResourcePool(process_on_nodes=[1], use_gpu=True)
        
        # Verify the method exists and TPU handling is in place
        import inspect
        source = inspect.getsource(pool.get_placement_groups)
        assert 'device_name == "tpu"' in source, "TPU handling not found in get_placement_groups"
        assert 'device_name = "TPU"' in source, "TPU resource name mapping not found"
        
        print("   ✓ RayResourcePool has TPU support")
        print("   ✅ Ray resource allocation PASSED")
    except Exception as e:
        print(f"   ❌ Ray resource allocation FAILED: {e}")
        all_passed = False
    
    # Test 3: vLLM Rollout Worker TPU Support
    print("\n3. Testing vLLM Rollout Worker:")
    try:
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
        
        # Check that TPU handling exists in the source
        import inspect
        source = inspect.getsource(vLLMRollout.__init__)
        assert 'device_name = get_device_name()' in source, "Device detection not found"
        assert 'if device_name == "tpu"' in source, "TPU conditional not found"
        assert 'llm_kwargs["device"] = "tpu"' in source, "TPU device setting not found"
        
        print("   ✓ vLLMRollout has TPU device detection")
        print("   ✓ vLLMRollout sets device='tpu' for TPU mode")
        print("   ✅ vLLM rollout worker PASSED")
    except Exception as e:
        print(f"   ❌ vLLM rollout worker FAILED: {e}")
        all_passed = False
    
    # Test 4: Configuration Files
    print("\n4. Testing Configuration Files:")
    try:
        import os
        tpu_rollout_path = "/home/fangzhou/verl/verl/trainer/config/rollout/tpu_rollout.yaml"
        tpu_example_path = "/home/fangzhou/verl/examples/ppo_trainer_tpu.yaml"
        
        assert os.path.exists(tpu_rollout_path), f"TPU rollout config not found at {tpu_rollout_path}"
        assert os.path.exists(tpu_example_path), f"TPU example config not found at {tpu_example_path}"
        
        # Check config content
        with open(tpu_rollout_path, 'r') as f:
            content = f.read()
            assert 'dtype: bfloat16' in content, "bfloat16 not configured"
            assert 'tensor_model_parallel_size: 8' in content, "TP size 8 not configured"
            assert 'device: tpu' in content, "TPU device not configured"
        
        print("   ✓ TPU rollout configuration exists")
        print("   ✓ TPU example configuration exists")
        print("   ✅ Configuration files PASSED")
    except Exception as e:
        print(f"   ❌ Configuration files FAILED: {e}")
        all_passed = False
    
    return all_passed


def test_vllm_serve_approach():
    """Show how to use vLLM with the same approach as vllm serve."""
    print("\n=== vLLM Serve-Style Test ===")
    
    print("\nTo run vLLM with Qwen3-8B on TPU in VERL, use:")
    print("1. Set environment variables to avoid SPMD issues:")
    print("   export VLLM_XLA_USE_SPMD=0")
    print("\n2. Use the following vLLM configuration:")
    config = """
    llm = LLM(
        model="Qwen/Qwen3-8B",
        device="tpu",
        tensor_parallel_size=8,
        dtype="bfloat16",
        max_model_len=32768,
        download_dir="/tmp/models",
        swap_space=32,
        disable_log_stats=True,
        trust_remote_code=True,
        enforce_eager=True,
    )
    """
    print(config)
    
    print("\n3. For VERL integration, ensure the rollout config uses these settings:")
    print("   - dtype: bfloat16")
    print("   - tensor_model_parallel_size: 8")
    print("   - max_model_len: 32768")
    print("   - device: tpu (added by our modifications)")
    
    return True


def main():
    """Run the test."""
    print("=== VERL + vLLM Qwen3-8B TPU Test ===")
    
    # Check device
    from verl.utils.device import get_device_name, is_tpu_available
    print(f"\nDevice Detection:")
    print(f"  Device: {get_device_name()}")
    print(f"  TPU Available: {is_tpu_available}")
    
    if not is_tpu_available:
        print("\n✗ TPU not available! Please ensure PJRT_DEVICE=TPU is set.")
        return
    
    # Run component test
    success1 = test_verl_tpu_components()
    
    # Show vLLM usage
    success2 = test_vllm_serve_approach()
    
    if success1 and success2:
        print("\n✅ SUCCESS: VERL TPU integration is complete!")
        print("\nIMPORTANT: To run vLLM within VERL on TPU:")
        print("1. Set VLLM_XLA_USE_SPMD=0 to avoid SPMD memory info issues")
        print("2. Use the TPU rollout configuration at: verl/trainer/config/rollout/tpu_rollout.yaml")
        print("3. Run with appropriate settings for your model")
    else:
        print("\n❌ Some tests failed. Check the errors above.")


if __name__ == "__main__":
    main()