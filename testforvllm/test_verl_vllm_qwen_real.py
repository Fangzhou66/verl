#!/usr/bin/env python3
"""Test VERL+vLLM with actual Qwen3-8B model loading and inference."""

import os
import sys
import time

# Set up environment for TPU - CRITICAL: Disable SPMD to avoid memory info issues
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["VLLM_XLA_USE_SPMD"] = "0"  # This is critical!

# Add paths
sys.path.insert(0, "/home/fangzhou/verl")
sys.path.insert(0, "/home/fangzhou/vllm")

def test_vllm_qwen_real():
    """Actually load Qwen3-8B and run inference."""
    print("\n=== Testing vLLM with Qwen3-8B (Real Model Loading) ===")
    
    from vllm import LLM, SamplingParams
    
    model_name = "Qwen/Qwen3-8B"
    
    try:
        print(f"\nLoading {model_name} on TPU...")
        print("This will take several minutes as the model loads...")
        
        # Check current environment
        print(f"\nEnvironment variables:")
        print(f"  VLLM_XLA_USE_SPMD: {os.environ.get('VLLM_XLA_USE_SPMD', 'not set')}")
        print(f"  PJRT_DEVICE: {os.environ.get('PJRT_DEVICE', 'not set')}")
        
        start_time = time.time()
        
        # Create LLM instance with the exact settings that work
        llm = LLM(
            model=model_name,
            device="tpu",
            tensor_parallel_size=8,
            dtype="bfloat16",
            max_model_len=32768,
            download_dir="/tmp/models",
            swap_space=32,
            disable_log_stats=True,
            trust_remote_code=True,
            enforce_eager=True,
        )
        
        load_time = time.time() - start_time
        print(f"\n✓ Model loaded successfully in {load_time:.1f} seconds!")
        
        # Test generation
        test_prompts = [
            "The capital of France is",
            "Artificial intelligence will",
            "In the future, technology will"
        ]
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
        )
        
        print("\nGenerating responses...")
        gen_start = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        gen_time = time.time() - gen_start
        
        print(f"\n✓ Generation completed in {gen_time:.1f} seconds!")
        print("\n=== Generation Results ===")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 80)
        
        total_time = time.time() - start_time
        print(f"\n✓ Total test time: {total_time:.1f} seconds")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verl_rollout_config():
    """Test that VERL rollout can be configured for TPU."""
    print("\n=== Testing VERL Rollout Configuration ===")
    
    try:
        from omegaconf import OmegaConf
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
        
        # Load the TPU rollout config
        config_path = "/home/fangzhou/verl/verl/trainer/config/rollout/tpu_rollout.yaml"
        config = OmegaConf.load(config_path)
        
        print(f"\nLoaded TPU rollout config:")
        print(f"  - dtype: {config.dtype}")
        print(f"  - tensor_model_parallel_size: {config.tensor_model_parallel_size}")
        print(f"  - max_model_len: {config.max_model_len}")
        print(f"  - device: {config.engine_kwargs.vllm.device}")
        
        # Verify critical settings
        assert config.dtype == "bfloat16", "TPU should use bfloat16"
        assert config.tensor_model_parallel_size == 8, "TPU v6e-8 should use TP=8"
        assert config.engine_kwargs.vllm.device == "tpu", "Device should be TPU"
        
        print("\n✓ VERL rollout configuration is correct for TPU!")
        
        print("\nTo use this in VERL, run with:")
        print("  export VLLM_XLA_USE_SPMD=0")
        print("  python -m verl.trainer.ppo_trainer \\")
        print("    --config-path /path/to/your/config \\")
        print("    --config-name your_config.yaml \\")
        print("    rollout=tpu_rollout")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the real test."""
    print("=== VERL + vLLM Qwen3-8B TPU Real Test ===")
    
    # Check device
    from verl.utils.device import get_device_name, is_tpu_available
    print(f"\nDevice Detection:")
    print(f"  Device: {get_device_name()}")
    print(f"  TPU Available: {is_tpu_available}")
    
    if not is_tpu_available:
        print("\n✗ TPU not available! Please ensure PJRT_DEVICE=TPU is set.")
        return
    
    # Check SPMD setting
    spmd_enabled = os.environ.get("VLLM_XLA_USE_SPMD", "1")
    if spmd_enabled == "1":
        print("\n⚠️  WARNING: VLLM_XLA_USE_SPMD is not set to 0!")
        print("This may cause memory info errors. Setting it now...")
        os.environ["VLLM_XLA_USE_SPMD"] = "0"
    
    # Test rollout config
    config_ok = test_verl_rollout_config()
    
    # Ask user before running the real model test
    print("\n" + "="*80)
    print("Ready to load Qwen3-8B model. This will take several minutes.")
    response = input("Do you want to continue? (y/n): ")
    
    if response.lower() == 'y':
        vllm_ok = test_vllm_qwen_real()
        
        if config_ok and vllm_ok:
            print("\n✅ SUCCESS: vLLM with Qwen3-8B works on TPU!")
            print("\nYou can now use VERL with vLLM on TPU by:")
            print("1. Setting VLLM_XLA_USE_SPMD=0")
            print("2. Using the tpu_rollout.yaml configuration")
            print("3. Running your PPO training with these settings")
        else:
            print("\n❌ Test failed. Check the errors above.")
    else:
        print("\nSkipping model loading test.")
        if config_ok:
            print("✓ VERL configuration is ready for TPU usage.")


if __name__ == "__main__":
    main()