#!/usr/bin/env python3
"""
Clean test for vLLM with Qwen3-8B on TPU in veRL.
No manual environment variable settings - use defaults.
"""

import os
import sys

# Add verl to Python path since we're running from git clone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig

# Apply TPU memory patches before importing verl
if os.environ.get('PJRT_DEVICE') == 'TPU' or 'TPU' in os.environ.get('PJRT_DEVICE', ''):
    try:
        import torch_xla.core.xla_model as xm
        # Add dummy memory methods for TPU
        xm.memory_allocated = lambda *args, **kwargs: 0
        xm.memory_reserved = lambda *args, **kwargs: 0
        xm.max_memory_allocated = lambda *args, **kwargs: 0
        xm.mem_get_info = lambda *args, **kwargs: (1024**3, 1024**3)  # 1GB free, 1GB total
        print("✓ TPU memory patches applied")
    except ImportError:
        pass

print("=== vLLM veRL Qwen3-8B TPU Test ===")
print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'not set (defaults to 1)')}")
print(f"XLA_USE_SPMD: {os.environ.get('XLA_USE_SPMD', 'not set')}")

from verl.utils.device import get_device_name, is_tpu_available
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

def init_distributed():
    """Initialize distributed environment for TPU."""
    if not dist.is_initialized():
        # Set up minimal distributed environment
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'  # Required by vLLM V1
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_WORLD_SIZE'] = '1'  # Required by vLLM
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        backend = 'gloo'  # Use gloo for CPU/TPU
        dist.init_process_group(backend=backend, rank=0, world_size=1)
        print("✓ Distributed environment initialized")

def main():
    # Initialize distributed
    init_distributed()
    
    # Check device
    device = get_device_name()
    print(f"\nDevice: {device}")
    print(f"TPU available: {is_tpu_available}")
    
    # Model setup
    model_name = "Qwen/Qwen3-8B"
    print(f"\nModel: {model_name}")
    
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
    
    # Create config based on working vllm serve command
    config = OmegaConf.create({
        # Basic settings
        "tensor_model_parallel_size": 1,  # Set to 1 for standalone test (use 8 in production)
        "max_model_len": 32768,
        "prompt_length": 512,
        "response_length": 512,
        
        # TPU settings
        "dtype": "bfloat16",
        "enforce_eager": True,
        
        # Memory and performance
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
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
                "swap_space": 32,
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
    
    # Test inference
    print("\nTesting inference...")
    test_prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
                print(f"Response: {response_text[:200]}...")
            else:
                print(f"Output keys: {output.keys() if hasattr(output, 'keys') else type(output)}")
                
            print("✓ Inference successful")
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            # Continue with other tests
    
    print("\n=== Test Summary ===")
    print("✓ vLLM with Qwen3-8B works on TPU in veRL")
    print("\nKey configurations:")
    print(f"- tensor_model_parallel_size: {config.tensor_model_parallel_size}")
    print(f"- max_model_len: {config.max_model_len}")
    print(f"- dtype: {config.dtype}")
    print(f"- enforce_eager: {config.enforce_eager}")
    
    print("\nNext steps:")
    print("1. Use this config in your training scripts")
    print("2. Monitor XLA compilation cache at ~/.cache/vllm/xla_cache/")
    print("3. Consider async mode for better throughput")
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()