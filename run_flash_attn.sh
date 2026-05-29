#!/bin/bash
# FlashAttention only
./launch.sh "${1:-throughput}" "${2:-8b}" "${3:-50}" "${4:-1}" --flash-attn
