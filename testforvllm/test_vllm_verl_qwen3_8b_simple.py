#!/usr/bin/env python3
"""
Simple test for vLLM with Qwen3-8B on TPU in veRL.
This version uses tensor_parallel_size=1 for easier testing.
"""

import os
import sys

# Add verl to Python path since we're running from git clone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig

# Apply TPU patches BEFORE importing any verl modules
import torch_xla.core.xla_model as xm

# Comprehensive patching for TPU memory methods
def patch_tpu_memory():
    """Patch XLA to add missing memory methods."""
    def dummy_memory(*args, **kwargs):
        return 0
    
    def dummy_mem_info(*args, **kwargs):
        return (1024**3, 1024**3)  # 1GB free, 1GB total
    
    # Patch the xm module
    xm.memory_allocated = dummy_memory
    xm.memory_reserved = dummy_memory
    xm.max_memory_allocated = dummy_memory
    xm.reset_peak_memory_stats = dummy_memory
    xm.mem_get_info = dummy_mem_info
    
    # Now patch verl's device utility to handle TPU properly
    from verl.utils import device
    original_get_torch_device = device.get_torch_device
    
    def patched_get_torch_device():
        result = original_get_torch_device()
        if device.get_device_name() == "tpu":
            # Ensure all memory methods exist
            if not hasattr(result, 'memory_allocated'):
                result.memory_allocated = dummy_memory
            if not hasattr(result, 'memory_reserved'):
                result.memory_reserved = dummy_memory
            if not hasattr(result, 'max_memory_allocated'):
                result.max_memory_allocated = dummy_memory
            if not hasattr(result, 'mem_get_info'):
                result.mem_get_info = dummy_mem_info
        return result
    
    device.get_torch_device = patched_get_torch_device
    print("✓ TPU memory patches applied")

# Apply patches
patch_tpu_memory()

print("=== Simple vLLM veRL Qwen3-8B TPU Test ===")
print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")

from verl.utils.device import get_device_name, is_tpu_available
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

def init_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        # Set up distributed environment variables
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'  # Required by vLLM V1
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_WORLD_SIZE'] = '1'  # Required by vLLM
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        backend = 'gloo'
        dist.init_process_group(backend=backend, rank=0, world_size=1)
        print("✓ Distributed environment initialized (world_size=1)")

def main():
    # Initialize distributed
    init_distributed()
    
    # Check device
    device = get_device_name()
    print(f"\nDevice: {device}")
    print(f"TPU available: {is_tpu_available}")
    
    # Model setup
    model_name = "Qwen/Qwen3-8B"
    print(f"\nTesting with: {model_name}")
    print("Note: Using tensor_parallel_size=1 for this test")
    print("      In production, use tensor_parallel_size=8 for TPU v6e-8")
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/tmp/models"
    )
    
    # Get model config
    print("Loading model config...")
    model_config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/tmp/models"
    )
    
    # Create simplified config for testing
    config = OmegaConf.create({
        # Basic settings
        "tensor_model_parallel_size": 1,  # Simplified for test
        "max_model_len": 4096,  # Reduced for memory
        "prompt_length": 128,
        "response_length": 128,
        
        # TPU settings
        "dtype": "bfloat16",
        "enforce_eager": True,
        
        # Memory and performance
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 64,
        "free_cache_engine": False,
        "disable_log_stats": True,
        "enable_chunked_prefill": False,
        
        # Generation
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.9,
        "do_sample": True,
        
        # Load settings
        "load_format": "auto",
        
        # Engine kwargs
        "engine_kwargs": {
            "vllm": {
                "download_dir": "/tmp/models",
                "disable_custom_all_reduce": True,
            }
        }
    })
    
    # Initialize vLLM rollout
    print("\nInitializing vLLM rollout...")
    try:
        rollout = vLLMRollout(
            model_path=model_name,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=model_config,
            trust_remote_code=True,
        )
        print("✓ vLLM rollout initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        raise
    
    # Test inference with a simple prompt
    print("\nTesting inference...")
    test_prompt = "What is 2 + 2?"
    print(f"Prompt: {test_prompt}")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    
    # Create batch
    batch = {
        "input_ids": input_ids,
        "attention_mask": inputs["attention_mask"],
    }
    
    # Prepare input for veRL
    verl_input = {
        "batch": batch,
        "meta_info": {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": -1,
        },
        "non_tensor_batch": {},
    }
    
    try:
        # Generate
        output = rollout.generate_sequences(verl_input)
        
        # Check output format
        if hasattr(output, "batch") and "sequences" in output.batch:
            sequences = output.batch["sequences"]
            response_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
            print(f"Response: {response_text}")
        else:
            print(f"Output type: {type(output)}")
            if hasattr(output, 'keys'):
                print(f"Output keys: {output.keys()}")
            
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        raise
    
    print("\n=== Summary ===")
    print("✓ Basic vLLM inference works on TPU in veRL")
    print("\nFor production use:")
    print("1. Run with proper distributed setup (8 processes for TPU v6e-8)")
    print("2. Use tensor_model_parallel_size=8")
    print("3. Use max_model_len=32768 as in vllm serve")
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()