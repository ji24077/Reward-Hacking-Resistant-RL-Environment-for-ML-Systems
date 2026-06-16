#!/usr/bin/env bash
# Run the public benchmark. Pass --device cuda to benchmark on GPU.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-cpu}"

echo "=== MoE Environment: Public Benchmark (device=$DEVICE) ==="
uv run python benchmarks/benchmark_moe.py --device "$DEVICE" "${@:2}"
