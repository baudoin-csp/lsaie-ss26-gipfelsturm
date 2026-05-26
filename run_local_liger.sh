#!/bin/bash
# Liger RMSNorm with --transformer-impl local (Liger actually takes effect)
./launch.sh "${1:-throughput}" "${2:-8b}" "${3:-50}" "${4:-1}" --local --liger
