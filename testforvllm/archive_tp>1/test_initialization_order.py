#!/usr/bin/env python3
"""
Test to verify the initialization order issue causing SPMD segfault.
This script demonstrates that VERL's device.py creates XLA tensors at import time.
"""

import os
import sys

# Set environment variables
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['VLLM_XLA_USE_SPMD'] = '1'

print("=== Testing Initialization Order ===")
print("PJRT_DEVICE:", os.environ.get('PJRT_DEVICE'))
print("VLLM_XLA_USE_SPMD:", os.environ.get('VLLM_XLA_USE_SPMD'))

# Add verl to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("\n1. Before any imports - no XLA device created yet")

print("\n2. Importing torch_xla...")
import torch_xla.core.xla_model as xm
print("   torch_xla imported successfully")

print("\n3. Testing xm.xla_device() call (this creates a tensor on XLA device)...")
try:
    device = xm.xla_device()
    print(f"   XLA device created: {device}")
except Exception as e:
    print(f"   Error creating XLA device: {e}")

print("\n4. Now trying to import VERL (which calls xm.xla_device() at module level)...")
try:
    from verl.utils.device import is_tpu_available
    print(f"   VERL imported successfully, is_tpu_available = {is_tpu_available}")
except Exception as e:
    print(f"   Error importing VERL: {e}")
    import traceback
    traceback.print_exc()

print("\n5. After VERL import, trying to initialize vLLM with SPMD...")
try:
    from vllm import LLM
    print("   Creating LLM instance with SPMD...")
    llm = LLM(
        model="facebook/opt-125m",
        tensor_parallel_size=8,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_model_len=128,
        dtype="bfloat16",
        device="tpu",
        disable_custom_all_reduce=True,
    )
    print("   ✓ LLM created successfully with SPMD!")
except Exception as e:
    print(f"   ✗ Failed to create LLM: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Analysis ===")
print("The issue is that VERL's device.py module calls xm.xla_device() at import time")
print("This creates an XLA tensor BEFORE vLLM can set up SPMD mode")
print("When vLLM later tries to enable SPMD, it causes a segfault because")
print("XLA tensors were already created in non-SPMD mode")