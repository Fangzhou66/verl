#!/usr/bin/env python3
"""
Test for vLLM with Qwen3-8B on TPU in veRL using default (non-SPMD) mode for TP=8.
Fixed to defer all TPU/XLA imports and accesses inside the worker function to avoid runtime initialization error.
"""

import os
import sys
import logging

# Add verl to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# NO early TPU imports here! Defer everything to worker_fn.

# Set env vars early (safe, no device access)
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['TPU_NUM_DEVICES'] = '8'  # Ensure 8 devices for TP=8

# Import only the spawner (no device access)
import torch_xla.distributed.xla_multiprocessing as xmp

def _worker_fn(index):
    # Now safe to import TPU/XLA stuff inside worker
    import torch
    import torch.distributed as dist
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp  # Redundant but safe

    # Apply patches inside worker (per-process)
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
        
        print(f"Rank {xm.get_ordinal()}: ✓ Comprehensive TPU memory patches applied")

    apply_tpu_memory_patches()

    # Now import verl and other modules
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer, AutoConfig
    from verl.utils.device import get_device_name, is_tpu_available
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    from verl import DataProto
    from tensordict import TensorDict

    print(f"Rank {xm.get_ordinal()}: PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
    print(f"Rank {xm.get_ordinal()}: Device: {get_device_name()}")
    print(f"Rank {xm.get_ordinal()}: TPU available: {is_tpu_available}")

    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()  # Should be 8
    if world_size != 8 and rank == 0:
        print(f"WARNING: Detected world_size={world_size}, but expected 8 for TP=8. Check TPU_NUM_DEVICES env var.")
    
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(0)
    os.environ['LOCAL_WORLD_SIZE'] = str(world_size)

    # Init dist with PJRt (default for TPU, no SPMD)
    dist.init_process_group(backend='pjrt')
    print(f"Rank {rank}: ✓ Distributed initialized (world_size={world_size}, default mode)")

    model_name = "Qwen/Qwen3-8B"
    if rank == 0:
        print(f"\nTesting with: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="/tmp/models")
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir="/tmp/models")

    # Config with TP=8, default backend="mp" (non-SPMD)
    config = OmegaConf.create({
        "calculate_log_probs": False,
        "tensor_model_parallel_size": 8,  # TP=8
        "max_model_len": 2048,  # Reduced to avoid OOM; increase if stable
        "prompt_length": 128,
        "response_length": 128,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 64,
        "free_cache_engine": False,
        "disable_log_stats": True,
        "enable_chunked_prefill": False,
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.9,
        "do_sample": True,
        "load_format": "auto",
        "val_kwargs": {
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0,
            "n": 1,
            "do_sample": False,
        },
        "engine_kwargs": {
            "vllm": {
                "download_dir": "/tmp/models",
                "disable_custom_all_reduce": True,
                "distributed_executor_backend": "mp",  # Default multi-process backend (non-SPMD)
            }
        }
    })

    if rank == 0:
        print("\nInitializing vLLM rollout in default mode...")
    rollout = vLLMRollout(
        model_path=model_name,
        config=config,
        tokenizer=tokenizer,
        model_hf_config=model_config,
        trust_remote_code=True,
    )
    if rank == 0:
        print("✓ Initialized successfully")

    # Inference test (on rank 0, vLLM syncs)
    if rank == 0:
        print("\nTesting inference...")
        test_prompt = "The capital of France is"
        print(f"Prompt: {test_prompt}")

        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)

        batch_dict = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])

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

        print("\nCalling generate_sequences...")
        output = rollout.generate_sequences(verl_input)

        # Process output
        print(f"Output type: {type(output)}")
        if hasattr(output, 'batch'):
            print(f"Output.batch keys: {output.batch.keys()}")
            if "responses" in output.batch:
                response_tokens = output.batch["responses"][0]
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                response_mask = response_tokens != pad_id
                actual_response = response_tokens[response_mask]
                response_text = tokenizer.decode(actual_response, skip_special_tokens=True)
                print(f"\nGenerated response: {response_text}")
            if "input_ids" in output.batch:
                full_seq = output.batch["input_ids"][0]
                full_text = tokenizer.decode(full_seq, skip_special_tokens=True)
                print(f"\nFull sequence: {full_text[:200]}...")
            if "prompts" in output.batch:
                prompt_tokens = output.batch["prompts"][0]
                prompt_text = tokenizer.decode(prompt_tokens[prompt_tokens != pad_id], skip_special_tokens=True)
                print(f"\nOriginal prompt: {prompt_text}")
        print("\n✓ Inference completed!")

    xm.rendezvous("end")

    if rank == 0:
        print("\n=== Test Summary ===")
        print("✓ vLLM with Qwen3-8B and TP=8 in default (non-SPMD) mode on TPU")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    xmp.spawn(_worker_fn, nprocs=None)  # Auto-detects 8 on v6e-8