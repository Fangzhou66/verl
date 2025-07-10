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
    """Check the availability of TPU"""
    if not is_torch_xla_available():
        return False
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device() is not None
    except Exception:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()
is_tpu_available = is_tpu_available()


def get_visible_devices_keyword() -> str:
    """Function that gets visible devices keyword name.
    Returns:
        'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', or 'TPU_VISIBLE_CHIPS'
    """
    if is_cuda_available:
        return "CUDA_VISIBLE_DEVICES"
    elif is_npu_available:
        return "ASCEND_RT_VISIBLE_DEVICES"
    elif is_tpu_available:
        return "TPU_VISIBLE_CHIPS"
    else:
        return "CUDA_VISIBLE_DEVICES"  # Default fallback


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently supports CPU, CUDA, NPU, and TPU.
    Returns:
        device
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    elif is_tpu_available:
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
    if is_cuda_available:
        return "nccl"
    elif is_npu_available:
        return "hccl"
    elif is_tpu_available:
        return "xla"  # TPU uses XLA for collective operations
    else:
        raise RuntimeError(f"No available nccl backend found on device type {get_device_name()}.")
