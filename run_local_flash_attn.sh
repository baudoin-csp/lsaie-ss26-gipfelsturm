#!/bin/bash
# FlashAttention with --transformer-impl local (FA actually takes effect)
./launch.sh "${1:-throughput}" "${2:-8b}" "${3:-50}" "${4:-1}" --local --flash-attn
