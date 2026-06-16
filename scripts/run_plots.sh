#!/usr/bin/env bash
# Run evaluation + generate figures.
# Usage:
#   bash scripts/run_plots.sh              # CPU
#   bash scripts/run_plots.sh cuda         # GPU
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-cpu}"

echo "=== Eval + Plot (device=$DEVICE) ==="

uv sync --extra plot 2>/dev/null || uv sync

uv run python experiments_eval.py --device "$DEVICE" --results-dir results

uv run python experiments_plot.py \
    --judge results/judge_report.json \
    --benchmark results/benchmark.json \
    --hack results/hack_comparison.json \
    --out figures

echo "Done. See figures/ and results/"
