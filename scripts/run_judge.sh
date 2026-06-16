#!/usr/bin/env bash
# Run the full judge pipeline (correctness + benchmark).
# Usage:
#   bash scripts/run_judge.sh           # CPU
#   bash scripts/run_judge.sh cuda      # GPU
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${1:-cpu}"
OUTPUT="${2:-logs/judge_report.json}"

mkdir -p "$(dirname "$OUTPUT")"

echo "=== MoE Judge (device=$DEVICE) ==="
uv run python judge/judge.py \
    --device "$DEVICE" \
    --output "$OUTPUT" \
    "${@:3}"

echo "Report saved to $OUTPUT"
