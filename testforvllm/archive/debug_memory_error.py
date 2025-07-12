#!/usr/bin/env python3
"""
Debug script to trace memory_allocated error.
"""

import os
import sys
import traceback

# Add verl to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Monkey patch to trace memory_allocated calls
import torch_xla.core.xla_model as xm

original_getattr = xm.__getattribute__

def traced_getattr(self, name):
    if name == 'memory_allocated':
        print("\n!!! TRACE: memory_allocated called !!!")
        print("Call stack:")
        traceback.print_stack()
        raise AttributeError(f"TPU doesn't have memory_allocated")
    try:
        return original_getattr(name)
    except AttributeError:
        if isinstance(self, type(xm)):
            return original_getattr(name)
        raise

# Apply tracing
xm.__class__.__getattribute__ = traced_getattr

# Now run a simple test
print("Starting debug test...")

from verl.utils.device import get_torch_device, get_device_name

print(f"Device name: {get_device_name()}")
device = get_torch_device()
print(f"Device object: {device}")
print(f"Device type: {type(device)}")

# Try to trigger the error
try:
    print("\nTrying to call memory_allocated...")
    result = device.memory_allocated()
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print("\nDebug test complete.")