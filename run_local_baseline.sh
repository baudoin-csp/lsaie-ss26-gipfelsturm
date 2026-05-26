#!/bin/bash
# Baseline with --transformer-impl local (no TransformerEngine)
./launch.sh "${1:-throughput}" "${2:-8b}" "${3:-50}" "${4:-1}" --local
