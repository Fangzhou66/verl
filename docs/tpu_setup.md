# VERL TPU Setup Guide

This guide explains how to run VERL with vLLM on Google Cloud TPU v6e-8.

## Prerequisites

1. **TPU VM Setup**: You need access to a Google Cloud TPU v6e-8 VM.

2. **Python Environment**: Create a conda environment with Python 3.11:
   ```bash
   conda create -n vllm-tpu python=3.11
   conda activate vllm-tpu
   ```

3. **Install PyTorch and torch_xla**: Follow the TPU requirements from vLLM:
   ```bash
   pip install -r /path/to/vllm/requirements/tpu.txt
   ```

4. **Install VERL dependencies**:
   ```bash
   pip install tensordict omegaconf codetiming
   ```

## Configuration

### 1. Environment Variables

Set the following environment variables for TPU:
```bash
export PJRT_DEVICE=TPU
export XLA_USE_SPMD=1
```

### 2. TPU Rollout Configuration

Use the TPU-specific rollout configuration located at:
`verl/trainer/config/rollout/tpu_rollout.yaml`

Key TPU-specific settings:
- `dtype: bfloat16` - TPU prefers bfloat16
- `tensor_model_parallel_size: 8` - Adjust based on model's attention heads
- Remove `gpu_memory_utilization` (not applicable for TPU)

### 3. Example PPO Trainer Configuration

See `examples/ppo_trainer_tpu.yaml` for a complete example configuration.

## Running VERL on TPU

### Testing the Setup

1. **Test device detection**:
   ```python
   from verl.utils.device import get_device_name, is_tpu_available
   print(f"Device: {get_device_name()}")  # Should print "tpu"
   print(f"TPU available: {is_tpu_available}")  # Should print True
   ```

2. **Test vLLM inference**:
   ```bash
   vllm serve <model_name> --device tpu --tensor_parallel_size 8
   ```

### Running PPO Training

```bash
python -m verl.trainer.ppo_trainer \
    --config-path /path/to/examples \
    --config-name ppo_trainer_tpu.yaml
```

## Important Considerations

### Model Selection
- Ensure your model's attention heads are divisible by the tensor_parallel_size
- For TPU v6e-8 with 8 cores, common TP sizes are 1, 2, 4, or 8

### Memory Management
- TPU memory management differs from GPU
- Use `enforce_eager: true` to avoid graph compilation issues initially
- TPU doesn't support CUDA graphs or gpu_memory_utilization parameters

### Supported Features
- ✅ Inference with vLLM
- ✅ Ray resource allocation for TPU
- ✅ Device detection and backend selection
- ❌ Training components (FSDP/Actor/Critic) - not yet implemented

### Known Limitations
1. vLLM on TPU may have compilation issues with certain models
2. TPU requires static shapes, which may affect dynamic batching
3. Some vLLM features may not be fully supported on TPU

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure all dependencies are installed in the conda environment
2. **TPU not detected**: Check that PJRT_DEVICE=TPU is set
3. **Tensor parallel size errors**: Ensure the model's attention heads are divisible by TP size
4. **XLA compilation errors**: Try with `enforce_eager: true` first

### Debug Commands

```bash
# Check TPU availability
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"

# Check VERL device detection
python -c "from verl.utils.device import *; print(f'Device: {get_device_name()}, TPU: {is_tpu_available}')"
```

## Next Steps

1. Start with simple inference tests before full training
2. Monitor TPU utilization with `gcloud compute tpus tpu-vm describe`
3. Optimize batch sizes and sequence lengths for TPU efficiency
4. Consider implementing training components (Actor/Critic) for full PPO support