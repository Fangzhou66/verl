#!/usr/bin/env python3
"""
Minimal test for vLLM on TPU in veRL.
This is the simplest possible test with all fixes applied.
"""

import os
import sys

# Add verl to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Apply TPU memory patches FIRST
import torch_xla.core.xla_model as xm

# Add dummy memory methods
xm.memory_allocated = lambda *args, **kwargs: 0
xm.memory_reserved = lambda *args, **kwargs: 0
xm.max_memory_allocated = lambda *args, **kwargs: 0
xm.mem_get_info = lambda *args, **kwargs: (1024**3, 1024**3)

# Patch verl's device utility
from verl.utils import device
_original_get_torch_device = device.get_torch_device

def _patched_get_torch_device():
    result = _original_get_torch_device()
    if device.get_device_name() == "tpu":
        # Ensure memory methods exist
        for method in ['memory_allocated', 'memory_reserved', 'max_memory_allocated', 'mem_get_info']:
            if not hasattr(result, method):
                setattr(result, method, getattr(xm, method))
    return result

device.get_torch_device = _patched_get_torch_device
print("✓ TPU patches applied")

# Now do imports
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

# Initialize distributed
if not dist.is_initialized():
    os.environ.update({
        'RANK': '0',
        'LOCAL_RANK': '0', 
        'WORLD_SIZE': '1',
        'LOCAL_WORLD_SIZE': '1',
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12355'
    })
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    print("✓ Distributed initialized")

# Model and config
model_name = "Qwen/Qwen3-8B"
print(f"\nTesting {model_name} on TPU")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir="/tmp/models"
)

# Complete config with all required fields
config = OmegaConf.create({
    # Required fields
    "calculate_log_probs": False,
    
    # Basic settings
    "tensor_model_parallel_size": 1,
    "max_model_len": 2048,  # Small for testing
    "prompt_length": 64,
    "response_length": 64,
    "dtype": "bfloat16",
    "enforce_eager": True,
    "max_num_batched_tokens": 2048,
    "free_cache_engine": False,
    "disable_log_stats": True,
    
    # Generation settings
    "temperature": 0.7,
    "top_k": -1,
    "top_p": 0.9,
    "do_sample": True,
    
    # Required validation kwargs
    "val_kwargs": {
        "top_k": -1,
        "top_p": 1.0,
        "temperature": 0,
    },
    
    # Other settings
    "load_format": "auto",
    "engine_kwargs": {"vllm": {"download_dir": "/tmp/models"}}
})

# Initialize rollout
print("\nInitializing vLLM...")
rollout = vLLMRollout(
    model_path=model_name,
    config=config,
    tokenizer=tokenizer,
    model_hf_config=None,
    trust_remote_code=True,
)
print("✓ vLLM initialized")

# Test generation
prompt = "Hello, world! 2+2="
print(f"\nPrompt: {prompt}")

# Create proper input
inputs = tokenizer(prompt, return_tensors="pt")
seq_length = inputs["input_ids"].shape[1]
position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)

batch = TensorDict({
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "position_ids": position_ids,
}, batch_size=1)

data_proto = DataProto(
    batch=batch,
    meta_info={
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.7,
    }
)

# Generate
print("\nGenerating...")
try:
    output = rollout.generate_sequences(data_proto)
    print(f"✓ Generation successful!")
    print(f"Output type: {type(output)}")
    
    # Try to decode output
    if hasattr(output, 'batch') and 'input_ids' in output.batch:
        generated = output.batch['input_ids'][0]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"Generated text: {text}")
    
except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if dist.is_initialized():
    dist.destroy_process_group()

print("\nTest complete!")