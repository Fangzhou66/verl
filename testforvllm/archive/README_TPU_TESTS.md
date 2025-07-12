# vLLM TPU Test Files

This directory contains test files for vLLM on TPU in veRL. Due to multiple iterations, there are many test files. Here's what you need:

## Recommended Test Files

### 1. **test_vllm_tpu_basic.py** - Start Here
- Simple test with GPT-2 (small model)
- Tests basic vLLM functionality on TPU
- Good for verifying TPU setup

```bash
python test_vllm_tpu_basic.py
```

### 2. **test_vllm_tpu_qwen.py** - Production Test
- Tests with Qwen3-8B (your target model)
- Uses known working configuration
- Tests both sync and async modes

```bash
python test_vllm_tpu_qwen.py
```

## Key Learnings

From the `fix_summary.md`:
1. **DON'T** set `VLLM_USE_V1=0` - let it default to 1
2. **DON'T** set `XLA_USE_SPMD=0` - let it use defaults
3. **DO** use `max_model_len=32768` for Qwen3-8B
4. **DO** use `dtype=bfloat16` for TPU
5. **DO** use `enforce_eager=True` to avoid compilation issues

## Clean Up Old Files

If you want to clean up the old test files:

```bash
# List old test files (review before deleting)
ls test_verl_*.py

# Archive old tests (safer than deleting)
mkdir -p archive
mv test_verl_*.py archive/

# Or remove them (be careful!)
# rm test_verl_*.py
```

## Next Steps

1. Run `test_vllm_tpu_basic.py` to verify basic functionality
2. Run `test_vllm_tpu_qwen.py` for full model testing
3. Use the working config in your training scripts
4. Consider async mode for better TPU utilization

## TPU-Specific Tips

- Initial compilation takes 20-30 minutes (cached afterwards)
- Use bfloat16 for better TPU performance
- Tensor parallel size must divide attention heads evenly
- Monitor TPU memory with `gcloud compute tpus tpu-vm describe`