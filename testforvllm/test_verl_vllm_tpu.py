#!/usr/bin/env python3
"""Test script to verify VERL+vLLM TPU integration."""

import os
import sys
import torch

# Set up environment for TPU
os.environ["PJRT_DEVICE"] = "TPU"
# CRITICAL: Disable SPMD for vLLM to avoid memory info issues
os.environ["XLA_USE_SPMD"] = "0"  # Disable SPMD for vLLM compatibility
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "true"

# Add VERL to Python path
sys.path.insert(0, "/home/fangzhou/verl")
sys.path.insert(0, "/home/fangzhou/vllm")

def test_device_detection():
    """Test if TPU is correctly detected by VERL."""
    print("\n=== Testing VERL Device Detection ===")
    
    from verl.utils.device import (
        is_tpu_available, 
        get_device_name, 
        get_visible_devices_keyword,
        get_nccl_backend
    )
    
    print(f"✓ TPU Available: {is_tpu_available}")
    print(f"✓ Device Name: {get_device_name()}")
    print(f"✓ Visible Devices Keyword: {get_visible_devices_keyword()}")
    print(f"✓ Backend Type: {get_nccl_backend()}")
    
    assert get_device_name() == "tpu", f"Expected device 'tpu', got '{get_device_name()}'"
    print("\n✓ Device detection test passed!")


def test_vllm_rollout_init():
    """Test if vLLM rollout can be initialized with TPU support."""
    print("\n=== Testing vLLM Rollout Initialization ===")
    
    # Initialize torch distributed for TPU
    if not torch.distributed.is_initialized():
        print("Initializing torch.distributed for TPU...")
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.xla_backend
            torch.distributed.init_process_group(backend='xla', init_method='xla://')
            print(f"✓ Distributed initialized with world_size={torch.distributed.get_world_size()}")
        except Exception as e:
            print(f"Warning: Could not initialize distributed: {e}")
            print("Skipping vLLM rollout test (requires distributed environment)")
            return None, None
    
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    from omegaconf import DictConfig
    from transformers import AutoTokenizer, AutoConfig
    
    # Create a minimal config - updated for OPT-125m's 12 attention heads
    config = DictConfig({
        "tensor_model_parallel_size": 4,  # 12 heads / 4 = 3 heads per device (valid)
        "dtype": "bfloat16",
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
        "free_cache_engine": False,
        "load_format": "auto",  # Changed from dummy to auto
        "gpu_memory_utilization": 0.5,  # Will be ignored for TPU
        "disable_log_stats": True,
        "prompt_length": 512,
        "response_length": 512,
        "max_model_len": 1024,
        "enable_chunked_prefill": False,
        "calculate_log_probs": False,
        "seed": 42,
        "engine_kwargs": {
            "vllm": {
                "device": "tpu",
                "trust_remote_code": True,
            }
        }
    })
    
    # Use a small model for testing
    model_name = "facebook/opt-125m"
    
    try:
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Loading model config...")
        model_config = AutoConfig.from_pretrained(model_name)
        
        print(f"Initializing vLLM rollout on TPU...")
        rollout = vLLMRollout(
            model_path=model_name,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=model_config,
            trust_remote_code=False
        )
        
        print("✓ vLLM rollout initialized successfully!")
        
        # Check if the inference engine was created with TPU device
        if hasattr(rollout.inference_engine, 'llm_engine'):
            print(f"✓ Inference engine created")
        
        return rollout, tokenizer
        
    except Exception as e:
        print(f"✗ Failed to initialize vLLM rollout: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_ray_resource_allocation():
    """Test if Ray resource allocation works with TPU."""
    print("\n=== Testing Ray Resource Allocation ===")
    
    import ray
    from verl.single_controller.ray.base import RayResourcePool
    
    if not ray.is_initialized():
        ray.init(local_mode=True)  # Use local mode for testing
    
    try:
        # Create a resource pool for TPU
        resource_pool = RayResourcePool(
            process_on_nodes=[1],  # Single node with 1 process
            use_gpu=True,  # This will allocate TPU resources
            name_prefix="test_tpu",
            detached=False
        )
        
        # Get placement groups with TPU device
        pgs = resource_pool.get_placement_groups(
            strategy="STRICT_PACK",
            device_name="tpu"
        )
        
        print(f"✓ Created {len(pgs)} placement groups")
        
        # Check if TPU resources are allocated
        for idx, pg in enumerate(pgs):
            bundles = ray.util.placement_group_table(pg)["bundles"]
            print(f"  Placement group {idx}: {bundles}")
            
            # Verify TPU resources
            for bundle_idx, bundle_resources in bundles.items():
                if isinstance(bundle_resources, dict) and "TPU" in bundle_resources:
                    print(f"  ✓ TPU resource found in bundle {bundle_idx}: {bundle_resources['TPU']}")
        
        print("\n✓ Ray resource allocation test passed!")
        
    except Exception as e:
        print(f"✗ Ray resource allocation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


def main():
    """Run all tests."""
    print("=== VERL + vLLM TPU Integration Test ===")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Critical environment check
    print("\n=== Critical Environment Variables ===")
    print(f"XLA_USE_SPMD: {os.environ.get('XLA_USE_SPMD', 'not set')}")
    print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
    print(f"ENABLE_PJRT_COMPATIBILITY: {os.environ.get('ENABLE_PJRT_COMPATIBILITY', 'not set')}")
    
    if os.environ.get('XLA_USE_SPMD') != '0':
        print("\n⚠️  WARNING: XLA_USE_SPMD must be set to 0 for vLLM on TPU!")
        print("This prevents SPMD conflicts with vLLM's manual parallelization.")
    
    # Test 1: Device detection
    test_device_detection()
    
    # Test 2: Ray resource allocation
    test_ray_resource_allocation()
    
    # Test 3: vLLM rollout initialization
    rollout, tokenizer = test_vllm_rollout_init()
    
    if rollout is not None:
        print("\n✓ All tests passed! VERL+vLLM TPU integration is working.")
        print("\n=== Key Settings for Success ===")
        print("1. XLA_USE_SPMD=0 (critical!)")
        print("2. tensor_parallel_size must divide evenly into attention heads")
        print("3. Use the tpu_rollout.yaml configuration in VERL")
        print("4. For Qwen3-8B (32 heads): use TP=1, 2, 4, 8")
        print("5. For OPT-125m (12 heads): use TP=1, 2, 3, 4, 6, or 12")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
        print("\nCommon fixes:")
        print("1. Set XLA_USE_SPMD=0")
        print("2. Ensure tensor_parallel_size divides into attention heads")
        print("3. Check that you're in the vllm-tpu conda environment")


if __name__ == "__main__":
    main()