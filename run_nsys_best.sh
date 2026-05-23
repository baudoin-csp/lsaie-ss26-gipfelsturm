#!/bin/bash
# NSYS profiling on best config (FA + Liger) — 10 steps only
./launch.sh "${1:-throughput}" "${2:-8b}" "${3:-10}" "${4:-1}" --liger --flash-attn --nsys
