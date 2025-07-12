#!/usr/bin/env python3
"""
Test vLLM rollout on TPU with Qwen3-8B model in veRL.
This uses the known working configuration from vllm serve.
"""

import os
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig

# DO NOT set these - let them use defaults
# os.environ["XLA_USE_SPMD"] = "0"  # Don't set this
# os.environ["VLLM_USE_V1"] = "0"   # Don't set this - let it default to 1

print("=== Qwen3-8B vLLM TPU Test ===")
print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'not set (defaults to 1)')}")

from verl.utils.device import get_device_name
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

def test_sync_mode():
    """Test synchronous vLLM rollout mode."""
    
    model_name = "Qwen/Qwen3-8B"
    print(f"\n=== Testing SYNC mode with {model_name} ===")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/tmp/models"
    )
    
    # Get model config
    model_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/tmp/models"
    )
    
    # Create config matching the working vllm serve command
    config = OmegaConf.create({
        # Mode
        "mode": "sync",
        
        # Model settings
        "tensor_model_parallel_size": 8,  # TPU v6e-8 has 8 cores
        "max_model_len": 32768,  # Match working config
        "prompt_length": 1024,
        "response_length": 1024,
        
        # TPU-specific
        "dtype": "bfloat16",
        "enforce_eager": True,  # Match working config
        "device": "tpu",
        
        # Memory settings
        "free_cache_engine": False,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        
        # Generation settings
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.9,
        "do_sample": True,
        
        # Logging
        "disable_log_stats": True,  # Match working config
        "enable_chunked_prefill": False,
        
        # Load format
        "load_format": "auto",
        
        # Engine kwargs
        "engine_kwargs": {
            "vllm": {
                "swap_space": 32,  # Match working config
                "download_dir": "/tmp/models",
                "disable_custom_all_reduce": True,
            }
        }
    })
    
    print("\nInitializing vLLM rollout...")
    rollout = vLLMRollout(
        model_path=model_name,
        config=config,
        tokenizer=tokenizer,
        model_hf_config=model_config,
        trust_remote_code=True,
    )
    print("✓ Sync mode initialized")
    
    # Test generation
    test_prompt = "What is machine learning?"
    input_ids = tokenizer(test_prompt, return_tensors="pt")["input_ids"]
    
    dummy_input = {
        "batch": {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        },
        "meta_info": {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "do_sample": True,
        },
        "non_tensor_batch": {},
    }
    
    print("\nGenerating response...")
    output = rollout.generate_sequences(dummy_input)
    print("✓ Generation completed")
    
    return rollout

def test_async_mode():
    """Test asynchronous vLLM rollout mode."""
    
    model_name = "Qwen/Qwen3-8B"
    print(f"\n=== Testing ASYNC mode with {model_name} ===")
    
    # Note: Async mode requires more setup with Ray actors
    # This is a simplified test to show the config
    
    config = OmegaConf.create({
        # Mode
        "mode": "async",  # Change to async
        
        # Same settings as sync mode
        "tensor_model_parallel_size": 8,
        "max_model_len": 32768,
        "dtype": "bfloat16",
        "enforce_eager": True,
        
        # Async-specific settings
        "free_cache_engine": True,  # Can be more aggressive with memory
        
        # Engine kwargs
        "engine_kwargs": {
            "vllm": {
                "device": "tpu",
                "disable_custom_all_reduce": True,
            }
        }
    })
    
    print("Async mode config created")
    print("Note: Full async mode requires Ray actor setup")
    
    # In real usage, you would use:
    # from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout
    # And set up the AsyncvLLMServer with Ray actors

def main():
    """Run tests."""
    device = get_device_name()
    print(f"\nDetected device: {device}")
    
    if device != "tpu":
        print("WARNING: Not running on TPU!")
    
    try:
        # Test sync mode (simpler, recommended to start)
        rollout = test_sync_mode()
        
        # Show async mode config
        test_async_mode()
        
        print("\n=== Summary ===")
        print("✓ Sync mode works with vLLM on TPU")
        print("✓ Config matches working vllm serve settings")
        print("\nFor production training:")
        print("1. Use sync mode for simpler setup")
        print("2. Consider async mode for better concurrency")
        print("3. Monitor XLA compilation times")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nDebug tips:")
        print("1. Check vLLM V1 engine is being used")
        print("2. Ensure model is downloaded to /tmp/models")
        print("3. Verify TPU memory is sufficient")
        raise

if __name__ == "__main__":
    main()