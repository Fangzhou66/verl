#!/bin/bash
# Run vLLM TPU tests for veRL

echo "=== vLLM TPU Test Runner ==="
echo "Testing vLLM integration with veRL on TPU"
echo

# Set TPU environment (if not already set)
if [ -z "$PJRT_DEVICE" ]; then
    export PJRT_DEVICE=TPU
    echo "Set PJRT_DEVICE=TPU"
fi

# Important: Don't set these - let them use defaults
# export XLA_USE_SPMD=0  # Don't set
# export VLLM_USE_V1=0   # Don't set

echo "Environment:"
echo "  PJRT_DEVICE=$PJRT_DEVICE"
echo "  VLLM_USE_V1=${VLLM_USE_V1:-not set (defaults to 1)}"
echo "  XLA_USE_SPMD=${XLA_USE_SPMD:-not set}"
echo

# Test 1: Basic functionality
echo "=== Test 1: Basic vLLM TPU Test ==="
echo "Testing with small model (GPT-2)..."
python test_vllm_tpu_basic.py
BASIC_RESULT=$?
echo

# Test 2: Qwen model (if basic test passed)
if [ $BASIC_RESULT -eq 0 ]; then
    echo "=== Test 2: Qwen3-8B Model Test ==="
    echo "Testing with production model..."
    echo "Note: This will download ~16GB model if not cached"
    echo "Press Ctrl+C to skip this test"
    sleep 3
    python test_vllm_tpu_qwen.py
    QWEN_RESULT=$?
else
    echo "Skipping Qwen test due to basic test failure"
    QWEN_RESULT=1
fi

# Summary
echo
echo "=== Test Summary ==="
if [ $BASIC_RESULT -eq 0 ]; then
    echo "✓ Basic vLLM TPU test: PASSED"
else
    echo "✗ Basic vLLM TPU test: FAILED"
fi

if [ $QWEN_RESULT -eq 0 ]; then
    echo "✓ Qwen3-8B test: PASSED"
else
    echo "✗ Qwen3-8B test: FAILED or SKIPPED"
fi

echo
echo "For debugging failures:"
echo "1. Check TPU is accessible: python -c 'import torch_xla; print(torch_xla.device())'"
echo "2. Verify vLLM installation: python -c 'import vllm; print(vllm.__version__)'"
echo "3. Check logs in ~/.cache/vllm/xla_cache/ for compilation issues"