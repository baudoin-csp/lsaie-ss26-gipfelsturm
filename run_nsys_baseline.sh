#!/bin/bash
# NSYS profiling on baseline — 10 steps only (profiling adds overhead)
./launch.sh "${1:-throughput}" "${2:-8b}" "${3:-10}" "${4:-1}" --nsys
