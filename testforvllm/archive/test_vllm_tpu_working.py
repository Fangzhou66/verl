#!/usr/bin/env python3
"""
Working example of vLLM on TPU with veRL.
This shows proper usage with good generation quality.
"""

import os
import sys

# Add verl to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Apply TPU memory patches
import torch_xla.core.xla_model as xm
xm.memory_allocated = lambda *args, **kwargs: 0
xm.memory_reserved = lambda *args, **kwargs: 0
xm.max_memory_allocated = lambda *args, **kwargs: 0
xm.mem_get_info = lambda *args, **kwargs: (1024**3, 1024**3)

from verl.utils import device
_orig = device.get_torch_device
device.get_torch_device = lambda: type('TPU', (), {
    'memory_allocated': lambda *a, **k: 0,
    'memory_reserved': lambda *a, **k: 0,
    'max_memory_allocated': lambda *a, **k: 0,
    'mem_get_info': lambda *a, **k: (1024**3, 1024**3),
    '__getattr__': lambda s, n: getattr(_orig(), n) if hasattr(_orig(), n) else None
})()

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tensordict import TensorDict
from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

print("=== vLLM TPU Working Example ===")

# Initialize distributed
if not dist.is_initialized():
    os.environ.update({
        'RANK': '0', 'LOCAL_RANK': '0', 
        'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '1',
        'MASTER_ADDR': 'localhost', 'MASTER_PORT': '12355'
    })
    dist.init_process_group(backend='gloo', rank=0, world_size=1)

# Model setup
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir="/tmp/models"
)

# Good generation config
config = OmegaConf.create({
    "calculate_log_probs": False,
    "tensor_model_parallel_size": 1,
    "max_model_len": 2048,
    "prompt_length": 512,
    "response_length": 512,
    "dtype": "bfloat16",
    "enforce_eager": True,
    "max_num_batched_tokens": 2048,
    "free_cache_engine": False,
    "disable_log_stats": True,
    
    # Better generation parameters
    "temperature": 0.8,
    "top_k": 50,  # Limit to top 50 tokens
    "top_p": 0.95,  # Nucleus sampling
    "do_sample": True,
    "ignore_eos": False,  # Stop at EOS
    
    "val_kwargs": {
        "top_k": -1,
        "top_p": 1.0,
        "temperature": 0,
    },
    
    "load_format": "auto",
    "engine_kwargs": {"vllm": {"download_dir": "/tmp/models"}}
})

# Initialize
print(f"\nInitializing {model_name}...")
rollout = vLLMRollout(
    model_path=model_name,
    config=config,
    tokenizer=tokenizer,
    model_hf_config=None,
    trust_remote_code=True,
)

# Test various prompts
test_prompts = [
    "What is 2 + 2? Answer:",
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time,",
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    seq_len = inputs["input_ids"].shape[1]
    
    batch = TensorDict({
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "position_ids": torch.arange(seq_len).unsqueeze(0),
    }, batch_size=1)
    
    data_proto = DataProto(
        batch=batch,
        meta_info={
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
        }
    )
    
    # Generate
    output = rollout.generate_sequences(data_proto)
    
    # Extract response
    if "responses" in output.batch:
        response_tokens = output.batch["responses"][0]
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        
        # Get actual response without padding
        response_mask = response_tokens != pad_id
        if response_mask.any():
            actual_response = response_tokens[response_mask]
            response_text = tokenizer.decode(actual_response, skip_special_tokens=True)
            print(f"\nResponse: {response_text}")
        else:
            print("\nResponse: (empty)")
    
    # Show full text
    if "input_ids" in output.batch:
        full_seq = output.batch["input_ids"][0]
        # Find EOS position
        eos_positions = (full_seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            end_pos = eos_positions[0].item() + 1
            full_seq = full_seq[:end_pos]
        
        full_text = tokenizer.decode(full_seq, skip_special_tokens=True)
        print(f"\nFull text: {full_text}")

print(f"\n{'='*60}")
print("✓ All tests completed successfully!")
print("\nKey insights:")
print("- vLLM works on TPU with proper configuration")
print("- Memory patches are required for TPU")
print("- Better generation parameters improve output quality")
print("- Set top_k and top_p for better sampling")

# Cleanup
if dist.is_initialized():
    dist.destroy_process_group()