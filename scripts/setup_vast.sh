#!/usr/bin/env bash
# Bootstrap this environment on a fresh Vast.ai GPU instance.
#
# Usage (on the Vast.ai instance):
#   bash setup_vast.sh
#
# After setup, you can run:
#   uv run pytest
#   uv run python benchmarks/benchmark_moe.py --device cuda
#   uv run python judge/judge.py --device cuda
set -euo pipefail

echo "=== MoE Env: Vast.ai Setup ==="

# 1. System deps (Ubuntu/Debian)
if command -v apt-get &>/dev/null; then
    apt-get update -qq
    apt-get install -y -qq curl git build-essential python3-dev
fi

# 2. Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 3. Sync Python env from lockfile
uv sync

# 4. Verify CUDA is visible to PyTorch
uv run python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "Setup complete. Run:"
echo "  uv run pytest                                         # public tests"
echo "  uv run python benchmarks/benchmark_moe.py --device cuda  # benchmark"
echo "  uv run python judge/judge.py --device cuda           # full judge"
