#!/bin/bash
# Cut Cross-Entropy (fused linear + CE via Liger fused_linear_cross_entropy)
./launch.sh "${1:-throughput}" "${2:-8b}" "${3:-50}" "${4:-1}" --cce
