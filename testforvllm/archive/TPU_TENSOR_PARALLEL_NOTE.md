# TPU Tensor Parallelism in veRL

## The Issue

When you run `vllm serve` directly:
```bash
vllm serve Qwen/Qwen3-8B --tensor-parallel-size 8
```

vLLM handles the distributed setup internally, launching 8 processes for you.

However, when using vLLM through veRL's rollout, you need to handle the distributed setup yourself.

## Why the Test Uses tensor_parallel_size=1

Our test uses `tensor_model_parallel_size: 1` because:
- We're running a single process test
- veRL checks that tensor_parallel_size <= world_size
- With world_size=1, we can only use tensor_parallel_size=1

## For Production (TPU v6e-8)

In actual training, veRL's Ray-based system will:
1. Launch multiple workers
2. Initialize proper distributed environment
3. Allow tensor_model_parallel_size=8

The configuration would look like:
```yaml
actor_rollout_ref:
  rollout:
    tensor_model_parallel_size: 8  # Matches TPU v6e-8 cores
    # ... other settings
```

## Key Difference

- **Standalone vLLM**: Manages distributed setup internally
- **veRL vLLM Rollout**: Expects to run within veRL's distributed framework

This is why our simple test needs tensor_parallel_size=1, but production can use 8.