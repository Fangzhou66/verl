# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging
import os

import torch

logger = logging.getLogger(__name__)


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def is_torch_xla_available() -> bool:
    """Check if torch_xla is available"""
    try:
        import torch_xla  # noqa: F401
        return True
    except ImportError:
        return False


def is_tpu_available() -> bool:
    """Check the availability of TPU
    
    IMPORTANT: This function now uses lazy evaluation to avoid creating
    XLA tensors at import time, which can cause SPMD segfaults.
    """
    if not is_torch_xla_available():
        return False
    try:
        import torch_xla.core.xla_model as xm
        # Instead of calling xm.xla_device() which creates a tensor,
        # we check environment variables and XLA runtime availability
        # This avoids creating XLA tensors before SPMD mode is set
        
        # Check if PJRT_DEVICE is set to TPU
        if os.environ.get('PJRT_DEVICE') == 'TPU':
            return True
            
        # Alternatively, try to check if XLA runtime is available
        # without actually creating a device
        try:
            # This is a safer check that doesn't create tensors
            import torch_xla.runtime as xr
            return xr.device_type() == 'TPU'
        except:
            # Fallback: assume TPU is available if torch_xla is installed
            # and PJRT_DEVICE suggests TPU
            return 'TPU' in os.environ.get('PJRT_DEVICE', '')
    except Exception:
        return False


# Use lazy evaluation for device availability checks
# These will be computed on first access rather than at import time
class LazyDeviceChecker:
    def __init__(self):
        self._is_cuda_available = None
        self._is_npu_available = None
        self._is_tpu_available = None
    
    @property
    def is_cuda_available(self):
        if self._is_cuda_available is None:
            self._is_cuda_available = torch.cuda.is_available()
        return self._is_cuda_available
    
    @property
    def is_npu_available(self):
        if self._is_npu_available is None:
            self._is_npu_available = is_torch_npu_available()
        return self._is_npu_available
    
    @property
    def is_tpu_available(self):
        if self._is_tpu_available is None:
            self._is_tpu_available = is_tpu_available()
        return self._is_tpu_available


# Create a global lazy checker instance
_device_checker = LazyDeviceChecker()

# Export these as module-level attributes for backward compatibility
is_cuda_available = property(lambda self: _device_checker.is_cuda_available)
is_npu_available = property(lambda self: _device_checker.is_npu_available)
is_tpu_available = property(lambda self: _device_checker.is_tpu_available)

# For direct access, provide the lazy checker
def get_is_cuda_available():
    return _device_checker.is_cuda_available

def get_is_npu_available():
    return _device_checker.is_npu_available

def get_is_tpu_available():
    return _device_checker.is_tpu_available


def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', or 'TPU_VISIBLE_CHIPS'
    """
    if get_is_cuda_available():
        return "CUDA_VISIBLE_DEVICES"
    elif get_is_npu_available():
        return "ASCEND_RT_VISIBLE_DEVICES"
    elif get_is_tpu_available():
        return "TPU_VISIBLE_CHIPS"
    else:
        return "CUDA_VISIBLE_DEVICES"  # Default fallback


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently supports CPU, CUDA, NPU, and TPU.
    Returns:
        device
    """
    if get_is_cuda_available():
        device = "cuda"
    elif get_is_npu_available():
        device = "npu"
    elif get_is_tpu_available():
        device = "tpu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    
    if device_name == "tpu":
        # TPU doesn't have a torch.tpu namespace, return XLA utilities
        try:
            import torch_xla.core.xla_model as xm
            return xm
        except ImportError:
            logger.warning("torch_xla not available for TPU device")
            return torch.cuda
    
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    device_name = get_device_name()
    if device_name == "tpu":
        # For TPU, return the ordinal of the current device
        import torch_xla.core.xla_model as xm
        return xm.get_ordinal()
    else:
        return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    if get_is_cuda_available():
        return "nccl"
    elif get_is_npu_available():
        return "hccl"
    elif get_is_tpu_available():
        return "xla"  # TPU uses XLA for collective operations
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")