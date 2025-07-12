#!/usr/bin/env python3
"""Debug script to understand vLLM configuration differences."""

import os
import sys

# Check environment before any imports
print("=== Initial Environment ===")
print(f"VLLM_USE_V1: {os.getenv('VLLM_USE_V1', 'not set')}")
print(f"XLA_USE_SPMD: {os.getenv('XLA_USE_SPMD', 'not set')}")
print(f"PJRT_DEVICE: {os.getenv('PJRT_DEVICE', 'not set')}")

# Add paths
sys.path.insert(0, "/home/fangzhou/verl")
sys.path.insert(0, "/home/fangzhou/vllm")

# Import vLLM to see default settings
from vllm import envs

print("\n=== vLLM Environment Defaults ===")
print(f"envs.VLLM_USE_V1: {envs.VLLM_USE_V1}")
print(f"Default VLLM_USE_V1 (from envs.py): would be 1 if not set")

# Check what happens with different settings
print("\n=== Testing Different Configurations ===")

# Configuration 1: User's working command equivalent
print("\n1. Working vllm serve configuration:")
print("   - VLLM_USE_V1: not explicitly set (defaults to 1)")
print("   - XLA_USE_SPMD: not explicitly set")
print("   - enforce_eager: False (--enforce-eager flag)")
print("   - max_model_len: 32768")
print("   - Result: Uses V1 engine, works fine")

# Configuration 2: Current test file
print("\n2. Test file configuration:")
print("   - VLLM_USE_V1: 0 (explicitly set)")
print("   - XLA_USE_SPMD: 0 (explicitly set)")
print("   - enforce_eager: False (user modified)")
print("   - max_model_len: 4096")
print("   - Result: Segfault")

# Key differences
print("\n=== Key Differences ===")
print("1. VLLM_USE_V1: Working command uses default (1), test forces 0")
print("2. XLA_USE_SPMD: Working command doesn't set it, test forces 0")
print("3. max_model_len: 32768 vs 4096 (likely not the issue)")

print("\n=== Recommendations ===")
print("1. Remove or set VLLM_USE_V1=1 to match working configuration")
print("2. Remove XLA_USE_SPMD setting to use defaults")
print("3. These environment variables should match the working setup")