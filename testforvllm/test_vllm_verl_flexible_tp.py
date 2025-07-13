#!/usr/bin/env python3
"""
Test for vLLM with VERL supporting both tp=1 and tp=8 on TPU.
This test can run with different tensor parallel sizes and SPMD modes.
"""

import os
import sys
import argparse
import logging

# Parse args early to set SPMD before imports
parser = argparse.ArgumentParser(description="Test vLLM with VERL on TPU with flexible tensor parallelism")
parser.add_argument("--tp", type=int, default=1, choices=[1, 8],
                    help="Tensor parallel size (1 or 8)")
parser.add_argument("--use-spmd", action="store_true",
                    help="Enable SPMD mode (recommended for tp=8)")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                    help="Model to test")
parser.add_argument("--max-model-len", type=int, default=4096,
                    help="Maximum model length")
parser.add_argument("--prompt-length", type=int, default=128,
                    help="Prompt length for generation")
parser.add_argument("--response-length", type=int, default=128,
                    help="Response length for generation")
parser.add_argument("--test-prompt", type=str, default="The capital of France is",
                    help="Test prompt for generation")
args = parser.parse_args()

# Set SPMD environment variable BEFORE any imports
if args.use_spmd and args.tp == 8:
    os.environ["VLLM_XLA_USE_SPMD"] = "1"
    # Don't set XLA_USE_SPMD to avoid deprecation warning
    print("✓ VLLM_XLA_USE_SPMD set to '1' before imports")
else:
    os.environ["VLLM_XLA_USE_SPMD"] = "0"
    print(f"✓ VLLM_XLA_USE_SPMD set to '0' (tp={args.tp})")

# Add verl to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Early imports needed for patching
import torch

# Initialize SPMD if needed BEFORE creating any XLA devices
if args.use_spmd and args.tp == 8:
    import torch_xla.runtime as xr
    xr.use_spmd()
    print("✓ Called xr.use_spmd() before any XLA device creation")

import torch.distributed as dist
import torch_xla.core.xla_model as xm

# TPU memory patching (simplified version focusing on essential patches)
def apply_tpu_memory_patches():
    """Apply necessary patches for TPU memory tracking."""
    
    def dummy_memory(*args, **kwargs):
        return 0
    
    def dummy_mem_info(*args, **kwargs):
        return (1024**3, 1024**3)  # 1GB free, 1GB total
    
    def dummy_empty_cache(*args, **kwargs):
        pass
    
    # Patch xm module
    xm.memory_allocated = dummy_memory
    xm.memory_reserved = dummy_memory
    xm.max_memory_allocated = dummy_memory
    xm.reset_peak_memory_stats = dummy_memory
    xm.mem_get_info = dummy_mem_info
    xm.empty_cache = dummy_empty_cache
    
    print("✓ TPU memory patches applied")

# Apply patches before other imports
apply_tpu_memory_patches()

# Now import the rest
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig
from verl.utils.device import get_device_name, is_tpu_available
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from verl import DataProto
from tensordict import TensorDict

# Remove duplicate parse_args function since we already parsed args at the top

def init_distributed(args):
    """Initialize distributed environment based on tensor parallel size."""
    if not dist.is_initialized():
        # For SPMD mode with tp=8, we still initialize as single process
        # The actual parallelism is handled by XLA SPMD
        if args.use_spmd:
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_WORLD_SIZE'] = '1'
        else:
            # For non-SPMD mode, set up distributed environment
            # This assumes you're running with appropriate torch distributed launch
            rank = int(os.environ.get('RANK', '0'))
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            world_size = int(os.environ.get('WORLD_SIZE', str(args.tp)))
            local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', str(args.tp)))
            
            os.environ['RANK'] = str(rank)
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['LOCAL_WORLD_SIZE'] = str(local_world_size)
        
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        backend = 'gloo'
        dist.init_process_group(
            backend=backend,
            rank=int(os.environ['RANK']),
            world_size=int(os.environ['WORLD_SIZE'])
        )
        print(f"✓ Distributed environment initialized (rank={os.environ['RANK']}, world_size={os.environ['WORLD_SIZE']})")

def create_config(args):
    """Create configuration for vLLM rollout based on arguments."""
    
    # Base configuration
    config_dict = {
        # Required fields
        "calculate_log_probs": False,
        
        # Tensor parallel configuration
        "tensor_model_parallel_size": args.tp,
        
        # Model settings
        "max_model_len": args.max_model_len,
        "prompt_length": args.prompt_length,
        "response_length": args.response_length,
        
        # TPU settings
        "dtype": "bfloat16",
        "enforce_eager": True,  # Set to False for production
        
        # Memory and performance
        "max_num_batched_tokens": min(2048, args.max_model_len),
        "max_num_seqs": 64 if args.tp == 1 else 32,  # Adjust based on tp size
        "gpu_memory_utilization": 0.5,
        "free_cache_engine": False,
        "disable_log_stats": True,
        "enable_chunked_prefill": False,
        
        # Generation settings
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.9,
        "do_sample": True,
        
        # Load settings
        "load_format": "auto",
        
        # Validation kwargs
        "val_kwargs": {
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0,
            "n": 1,
            "do_sample": False,
        },
        
        # Engine kwargs
        "engine_kwargs": {
            "vllm": {
                "download_dir": "/tmp/models",
                "disable_custom_all_reduce": True,
                "distributed_executor_backend": "mp" if args.use_spmd else "ray",
            }
        }
    }
    
    # Add SPMD-specific settings if needed
    if args.use_spmd and args.tp == 8:
        # When using SPMD, vLLM handles tensor parallelism internally
        # We set tp=1 in config but SPMD will use all 8 chips
        config_dict["tensor_model_parallel_size"] = 1
        print("✓ SPMD mode configuration for 8 TPU chips")
    else:
        print(f"✓ Using explicit tensor parallelism with tp={args.tp}")
    
    return OmegaConf.create(config_dict)

def main():
    
    print(f"=== vLLM VERL TPU Test (tp={args.tp}, spmd={args.use_spmd}) ===")
    print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
    print(f"VLLM_XLA_USE_SPMD: {os.environ.get('VLLM_XLA_USE_SPMD', 'not set')}")
    print(f"Device: {get_device_name()}")
    print(f"TPU available: {is_tpu_available}")
    print(f"Model: {args.model}")
    
    # Initialize distributed environment
    init_distributed(args)
    
    # Create configuration
    config = create_config(args)
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir="/tmp/models"
    )
    
    # Get model config
    print("Loading model config...")
    model_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir="/tmp/models"
    )
    
    # Initialize vLLM rollout
    print(f"\nInitializing vLLM rollout (tp={config.tensor_model_parallel_size})...")
    try:
        rollout = vLLMRollout(
            model_path=args.model,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=model_config,
            trust_remote_code=True,
        )
        print("✓ vLLM rollout initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test inference
    print(f"\nTesting inference with prompt: '{args.test_prompt}'")
    
    # Tokenize
    inputs = tokenizer(args.test_prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Create position_ids
    seq_length = input_ids.shape[1]
    position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)
    
    # Create TensorDict for batch
    batch_dict = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, batch_size=input_ids.shape[0])
    
    # Create DataProto object
    verl_input = DataProto(
        batch=batch_dict,
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": -1,
        },
        non_tensor_batch={},
    )
    
    try:
        # Generate
        print("Generating...")
        output = rollout.generate_sequences(verl_input)
        
        # Extract and display results
        if hasattr(output, 'batch') and "responses" in output.batch:
            response_tokens = output.batch["responses"][0]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            response_mask = response_tokens != pad_id
            actual_response = response_tokens[response_mask]
            response_text = tokenizer.decode(actual_response, skip_special_tokens=True)
            print(f"\nGenerated response: {response_text}")
        
        print("\n✓ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"✓ Successfully ran vLLM with {args.model} on TPU")
    print(f"  - Tensor parallel size: {args.tp}")
    print(f"  - SPMD mode: {'enabled' if args.use_spmd else 'disabled'}")
    print(f"  - Effective parallelism: {'8 chips via SPMD' if (args.use_spmd and args.tp == 8) else f'{args.tp} chip(s)'}")
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()