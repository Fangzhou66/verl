import torch
import torch_xla.core.xla_model as xm
from omegaconf import OmegaConf
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from transformers import AutoTokenizer

# 1. Create a dummy config for the vLLM rollout worker
config = OmegaConf.create({
    "tensor_model_parallel_size": 1,
    "max_num_batched_tokens": 8192,
    "prompt_length": 128,
    "response_length": 128,
    "max_model_len": 256,
    "load_format": "dummy",
    "free_cache_engine": False,
    "enforce_eager": False,
    "disable_log_stats": True,
    "enable_chunked_prefill": False,
    "gpu_memory_utilization": 0.9,
    "calculate_log_probs": False,
    "engine_kwargs": {
        "vllm": {}
    }
})

# 2. Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

# 3. Initialize the vLLM rollout worker
rollout = vLLMRollout(
    model_path="gpt2",
    config=config,
    tokenizer=tokenizer,
    model_hf_config=None,
)

# 4. Create a dummy input
dummy_input = {
    "batch": {
        "input_ids": torch.randint(0, 1000, (1, 128)),
        "attention_mask": torch.ones((1, 128)),
        "position_ids": torch.arange(0, 128).unsqueeze(0),
    },
    "meta_info": {
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
    },
    "non_tensor_batch": {},
}

# 5. Generate a sequence
output = rollout.generate_sequences(dummy_input)

# 6. Print the output
print(output)