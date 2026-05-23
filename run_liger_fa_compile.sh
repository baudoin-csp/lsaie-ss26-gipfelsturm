#!/bin/bash
# Liger-Kernel + FlashAttention + torch.compile (full stack)
./launch.sh "${1:-throughput}" "${2:-8b}" "${3:-50}" "${4:-1}" --liger --flash-attn --compile
