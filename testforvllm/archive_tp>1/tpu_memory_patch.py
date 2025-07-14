"""
Patch for TPU memory tracking in veRL.
TPU (XLA) doesn't have memory_allocated() method like CUDA.
"""

import torch_xla.core.xla_model as xm

# Monkey patch to add dummy memory methods for TPU
def memory_allocated(*args, **kwargs):
    """Dummy memory_allocated for TPU - returns 0."""
    return 0

def memory_reserved(*args, **kwargs):
    """Dummy memory_reserved for TPU - returns 0."""
    return 0

def max_memory_allocated(*args, **kwargs):
    """Dummy max_memory_allocated for TPU - returns 0."""
    return 0

def mem_get_info(*args, **kwargs):
    """Dummy mem_get_info for TPU - returns (1GB free, 1GB total)."""
    # Return dummy values: 1GB free, 1GB total
    return (1024**3, 1024**3)

# Apply patches
xm.memory_allocated = memory_allocated
xm.memory_reserved = memory_reserved
xm.max_memory_allocated = max_memory_allocated
xm.mem_get_info = mem_get_info

print("✓ TPU memory tracking patches applied")