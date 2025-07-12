#!/usr/bin/env python3
"""
Basic test for vLLM rollout on TPU in veRL.
This test verifies that vLLM can be initialized and used for inference on TPU.
"""

import os
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# Print environment info
print("=== Environment Info ===")
print(f"PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'not set (defaults to 1)')}")
print(f"XLA_USE_SPMD: {os.environ.get('XLA_USE_SPMD', 'not set')}")

# Import after env vars to ensure proper device detection
from verl.utils.device import get_device_name, is_tpu_available
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

print(f"\nDevice detected: {get_device_name()}")
print(f"TPU available: {is_tpu_available}")

def test_vllm_tpu_inference():
    """Test basic vLLM inference on TPU."""
    
    # 1. Model configuration
    model_name = "gpt2"  # Small model for testing
    print(f"\n=== Testing vLLM with {model_name} ===")
    
    # 2. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Create TPU-optimized config
    config = OmegaConf.create({
        # Basic settings
        "tensor_model_parallel_size": 1,  # Use 1 for gpt2, 8 for larger models
        "max_num_batched_tokens": 4096,
        "prompt_length": 64,
        "response_length": 64,
        "max_model_len": 128,
        "load_format": "auto",
        
        # TPU-specific settings
        "dtype": "bfloat16",  # TPU prefers bfloat16
        "enforce_eager": True,  # Avoid graph compilation issues
        "free_cache_engine": False,
        "disable_log_stats": True,
        "enable_chunked_prefill": False,
        
        # Sampling settings
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1.0,
        "do_sample": True,
        
        # Engine kwargs for TPU
        "engine_kwargs": {
            "vllm": {
                "device": "tpu",
                "disable_custom_all_reduce": True,
                "enable_xla_graph": True,
            }
        }
    })
    
    # 4. Initialize vLLM rollout
    print("\nInitializing vLLM rollout...")
    try:
        rollout = vLLMRollout(
            model_path=model_name,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=None,
            trust_remote_code=False,
        )
        print("✓ vLLM rollout initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize vLLM rollout: {e}")
        raise
    
    # 5. Test inference
    print("\nTesting inference...")
    test_prompt = "The weather today is"
    
    # Prepare input
    input_ids = tokenizer(test_prompt, return_tensors="pt", padding=True)["input_ids"]
    batch_size = input_ids.shape[0]
    
    # Create input batch
    dummy_input = {
        "batch": {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "position_ids": torch.arange(0, input_ids.shape[1]).unsqueeze(0).expand(batch_size, -1),
        },
        "meta_info": {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
        },
        "non_tensor_batch": {},
    }
    
    try:
        # Generate
        output = rollout.generate_sequences(dummy_input)
        
        # Decode and print results
        if "sequences" in output:
            generated_ids = output["sequences"][0]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"\nPrompt: {test_prompt}")
            print(f"Generated: {generated_text}")
            print("\n✓ Inference successful!")
        else:
            print(f"\nOutput structure: {output.keys()}")
            print("✓ Rollout executed (check output format)")
            
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        raise
    
    # 6. Test cleanup
    if hasattr(rollout, 'inference_engine') and hasattr(rollout.inference_engine, 'sleep'):
        print("\nTesting memory cleanup...")
        rollout.inference_engine.sleep(level=1)
        print("✓ Memory cleanup successful")
    
    return True

def main():
    """Run the test."""
    print("=== vLLM TPU Basic Test ===")
    print("This test verifies basic vLLM functionality on TPU in veRL\n")
    
    # Check if we're on TPU
    if not is_tpu_available:
        print("WARNING: TPU not detected. This test is designed for TPU.")
        print("The test will attempt to run anyway...\n")
    
    try:
        test_vllm_tpu_inference()
        print("\n=== All tests passed! ===")
        print("\nNext steps:")
        print("1. Try with a larger model (e.g., Qwen3-8B)")
        print("2. Adjust tensor_model_parallel_size based on model size")
        print("3. Test async mode by setting mode='async' in config")
        
    except Exception as e:
        print(f"\n=== Test failed: {e} ===")
        print("\nTroubleshooting:")
        print("1. Ensure TPU is properly initialized")
        print("2. Check that vLLM has TPU support compiled")
        print("3. Verify environment variables are set correctly")
        print("4. Try with enforce_eager=True to avoid compilation issues")
        raise

if __name__ == "__main__":
    main()