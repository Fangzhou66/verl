#!/usr/bin/env python3
"""
Test vLLM with MPMD mode (mp backend) for tp=8 on TPU.
This carefully sets VLLM_XLA_USE_SPMD=0 BEFORE any imports.
"""
"python  /home/fangzhou/verl/testforvllm/test_vllm_mpmd_tp8.py"
import os
import sys

# CRITICAL: Set MPMD mode BEFORE any imports that might load vLLM
os.environ["VLLM_XLA_USE_SPMD"] = "0"
print("✓ Set VLLM_XLA_USE_SPMD=0 for MPMD mode (before imports)")

# Add verl to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import everything else
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoConfig

# Import VERL components
from verl.utils.device import get_device_name, is_tpu_available
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from verl import DataProto
from tensordict import TensorDict

# Apply TPU memory patches
def apply_tpu_memory_patches():
    """Apply necessary patches for TPU memory tracking."""
    import torch_xla.core.xla_model as xm
    
    def dummy_memory(*args, **kwargs):
        return 0
    
    def dummy_mem_info(*args, **kwargs):
        return (1024**3, 1024**3)  # 1GB free, 1GB total
    
    def dummy_empty_cache(*args, **kwargs):
        pass
    
    # Patch xm module
    xm.memory_allocated = dummy_memory
    xm.memory_reserved = dummy_memory
    xm.max_memory_allocated = dummy_memory
    xm.reset_peak_memory_stats = dummy_memory
    xm.mem_get_info = dummy_mem_info
    xm.empty_cache = dummy_empty_cache
    
    print("✓ TPU memory patches applied")

def init_distributed():
    """Initialize distributed environment for MPMD mode."""
    # For MPMD mode with vLLM, we should NOT pre-initialize distributed
    # vLLM's multiprocessing executor will handle this for each worker
    print("✓ Skipping distributed initialization - vLLM mp backend will handle it")

def create_config():
    """Create configuration for vLLM rollout in MPMD mode."""
    
    # Configuration based on the working vLLM example
    config_dict = {
        # Required fields
        "calculate_log_probs": False,
        
        # Model settings - match the vLLM example
        "tensor_model_parallel_size": 8,  # Use all 8 TPU chips
        "max_model_len": 32768,  # Small for testing
        "prompt_length": 64,
        "response_length": 64,
        
        # TPU settings
        "dtype": "bfloat16",
        "enforce_eager": True,
        
        # Memory and performance - match vLLM example exactly
        "max_num_batched_tokens": 64,
        "max_num_seqs": 4,
        "gpu_memory_utilization": 0.9,
        "free_cache_engine": False,
        "disable_log_stats": True,
        "enable_chunked_prefill": False,
        
        # Generation settings
        "temperature": 0.7,
        "top_k": -1,
        "top_p": 0.9,
        "do_sample": True,
        
        # Load settings
        "load_format": "auto",
        
        # Validation kwargs
        "val_kwargs": {
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0,
            "n": 1,
            "do_sample": False,
        },
        
        # Engine kwargs - use mp backend for MPMD
        "engine_kwargs": {
            "vllm": {
                "download_dir": "/tmp/models",
                "disable_custom_all_reduce": True,
                "distributed_executor_backend": "mp",  # Multiprocessing backend
                "max_num_seqs": 4,  # Override any defaults
            }
        }
    }
    
    return OmegaConf.create(config_dict)

def main():
    print("\n=== vLLM VERL TPU MPMD Test (tp=8, mp backend) ===")
    print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
    print(f"VLLM_XLA_USE_SPMD: {os.environ.get('VLLM_XLA_USE_SPMD', 'not set')}")
    
    # Apply TPU patches
    apply_tpu_memory_patches()
    
    print(f"Device: {get_device_name()}")
    print(f"TPU available: {is_tpu_available}")
    
    # Model setup - using same model as vLLM example
    model_name = "Qwen/Qwen3-8B"  # or "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Model: {model_name}")
    
    # Initialize distributed environment
    init_distributed()
    
    # Create configuration
    config = create_config()
    print(f"\nConfiguration:")
    print(f"  - tensor_parallel_size: {config.tensor_model_parallel_size}")
    print(f"  - max_model_len: {config.max_model_len}")
    print(f"  - max_num_batched_tokens: {config.max_num_batched_tokens}")
    print(f"  - distributed_executor_backend: mp")
    
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
    
    # Initialize vLLM rollout
    print(f"\nInitializing vLLM rollout with MPMD mode (tp={config.tensor_model_parallel_size})...")
    print("Note: This will spawn 8 worker processes using mp backend")
    
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
    
    # Test inference with simple prompts (like vLLM example)
    test_prompts = [
        "A robot may not injure a human being",
        "The capital of France is",
        "The greatest glory in living lies not in never falling,",
    ]
    
    print(f"\nTesting inference with {len(test_prompts)} prompts...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
                "max_tokens": 16,  # Short response like vLLM example
            },
            non_tensor_batch={},
        )
        
        try:
            # Generate
            output = rollout.generate_sequences(verl_input)
            
            # Extract and display results
            if hasattr(output, 'batch') and "responses" in output.batch:
                response_tokens = output.batch["responses"][0]
                pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
                response_mask = response_tokens != pad_id
                actual_response = response_tokens[response_mask]
                response_text = tokenizer.decode(actual_response, skip_special_tokens=True)
                print(f"Response: {response_text}")
            
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print("\n✓ All inferences completed successfully!")
    
    # Summary
    print("\n=== Test Summary ===")
    print("✓ Successfully ran vLLM with MPMD mode (mp backend) on TPU")
    print(f"  - Model: {model_name}")
    print("  - Tensor parallel size: 8")
    print("  - Backend: mp (multiprocessing)")
    print("  - SPMD mode: disabled (VLLM_XLA_USE_SPMD=0)")
    print("\nKey differences from SPMD mode:")
    print("  - Uses 8 separate worker processes")
    print("  - Each process manages one TPU chip")
    print("  - Coordination via multiprocessing backend")
    
    # No cleanup needed - vLLM handled distributed setup

if __name__ == "__main__":
    main()