#!/usr/bin/env python3
"""
Fixed test for vLLM with Qwen3-8B on TPU in veRL.
This version properly patches all memory tracking for TPU.
"""

import os
import sys
import logging

# Add verl to Python path since we're running from git clone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Early imports needed for patching
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm

# Comprehensive TPU memory patching
def apply_tpu_memory_patches():
    """Apply all necessary patches for TPU memory tracking."""
    
    # 1. Define dummy methods
    def dummy_memory(*args, **kwargs):
        return 0
    
    def dummy_mem_info(*args, **kwargs):
        return (1024**3, 1024**3)  # 1GB free, 1GB total
    
    def dummy_empty_cache(*args, **kwargs):
        pass
    
    # 2. Patch xm module directly
    xm.memory_allocated = dummy_memory
    xm.memory_reserved = dummy_memory
    xm.max_memory_allocated = dummy_memory
    xm.reset_peak_memory_stats = dummy_memory
    xm.mem_get_info = dummy_mem_info
    xm.empty_cache = dummy_empty_cache
    
    # 3. Import verl modules that need patching
    from verl.utils import device
    from verl.utils.profiler import performance
    
    # 4. Monkey patch get_torch_device to always return properly patched object
    original_get_torch_device = device.get_torch_device
    
    def patched_get_torch_device():
        """Return torch device with TPU memory methods patched."""
        result = original_get_torch_device()
        
        # For TPU, ensure all memory methods exist
        if device.get_device_name() == "tpu":
            # Create a wrapper object that has all the methods
            class TPUDeviceWrapper:
                def __getattr__(self, name):
                    # First try to get from original xm module
                    if hasattr(result, name):
                        return getattr(result, name)
                    # For memory methods, return dummy implementations
                    elif 'memory' in name or name == 'mem_get_info':
                        if name == 'mem_get_info':
                            return dummy_mem_info
                        elif name == 'empty_cache':
                            return dummy_empty_cache
                        else:
                            return dummy_memory
                    else:
                        raise AttributeError(f"'{type(result).__name__}' object has no attribute '{name}'")
            
            return TPUDeviceWrapper()
        return result
    
    # Replace the function
    device.get_torch_device = patched_get_torch_device
    
    # 5. Patch the _get_current_mem_info function directly to handle TPU
    original_get_mem_info = performance._get_current_mem_info
    
    def patched_get_mem_info(unit="GB", precision=2):
        """Patched version that handles TPU."""
        if device.get_device_name() == "tpu":
            # Return dummy values for TPU
            divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
            val = f"0.00"
            return val, val, val, f"{1.00:.{precision}f}"
        return original_get_mem_info(unit, precision)
    
    performance._get_current_mem_info = patched_get_mem_info
    
    # 6. Optionally disable GPU memory logging entirely by patching the decorator
    class NoOpDecorator:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, func):
            return func
    
    # Replace GPUMemoryLogger with no-op for TPU
    if device.get_device_name() == "tpu":
        performance.GPUMemoryLogger = NoOpDecorator
    
    print("✓ Comprehensive TPU memory patches applied")

# Apply patches before any other imports
apply_tpu_memory_patches()

# Now import the rest
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig
from verl.utils.device import get_device_name, is_tpu_available
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from verl import DataProto
from tensordict import TensorDict

print("=== vLLM veRL Qwen3-8B TPU Test (Fixed) ===")
print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
print(f"Device: {get_device_name()}")
print(f"TPU available: {is_tpu_available}")

def init_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        backend = 'gloo'
        dist.init_process_group(backend=backend, rank=0, world_size=1)
        print("✓ Distributed environment initialized")

def main():
    # Initialize distributed
    init_distributed()
    
    # Model setup
    model_name = "Qwen/Qwen3-8B"
    print(f"\nTesting with: {model_name}")
    
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
    
    # Create config for testing with all required fields
    config = OmegaConf.create({
        # Required fields
        "calculate_log_probs": False,  # Required field
        
        # Basic settings
        "tensor_model_parallel_size": 1,  # Single process test
        "max_model_len": 4096,  # Reduced for testing
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
        
        # Validation kwargs (required)
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
        import traceback
        traceback.print_exc()
        raise
    
    # Test inference
    print("\nTesting inference...")
    # Use a prompt that's less likely to trigger repetitive patterns
    test_prompt = "The capital of France is"
    # Alternative prompts you can try:
    # test_prompt = "Python is a programming language that"
    # test_prompt = "The weather today is"
    # test_prompt = "Once upon a time, there was"
    # test_prompt = "The main benefit of exercise is"
    print(f"Prompt: {test_prompt}")
    
    # Tokenize
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
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
        print("\nCalling generate_sequences...")
        output = rollout.generate_sequences(verl_input)
        
        # Check output
        print(f"Output type: {type(output)}")
        if hasattr(output, 'batch'):
            print(f"Output.batch keys: {output.batch.keys()}")
            
            # Extract the response properly
            if "responses" in output.batch:
                # Get the generated response tokens only
                response_tokens = output.batch["responses"][0]
                # Find where padding starts (pad_token_id)
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                # Get non-padded response
                response_mask = response_tokens != pad_id
                actual_response = response_tokens[response_mask]
                response_text = tokenizer.decode(actual_response, skip_special_tokens=True)
                print(f"\nGenerated response only: {response_text}")
            
            if "input_ids" in output.batch:
                # Also show the full sequence
                full_seq = output.batch["input_ids"][0]
                # Decode only up to max length or EOS
                full_text = tokenizer.decode(full_seq, skip_special_tokens=True)
                print(f"\nFull sequence (prompt + response): {full_text[:200]}...")
                
            # Show prompt for comparison
            if "prompts" in output.batch:
                prompt_tokens = output.batch["prompts"][0]
                prompt_text = tokenizer.decode(prompt_tokens[prompt_tokens != pad_id], skip_special_tokens=True)
                print(f"\nOriginal prompt: {prompt_text}")
        
        print("\n✓ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n=== Test Summary ===")
    print("✓ vLLM with Qwen3-8B works on TPU in veRL")
    print("\nThe comprehensive patching approach:")
    print("1. Patches xm module directly")
    print("2. Wraps get_torch_device() to return TPU-compatible object")
    print("3. Patches _get_current_mem_info directly")
    print("4. Optionally replaces GPUMemoryLogger with no-op")
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()