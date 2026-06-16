#!/usr/bin/env bash
# Run public correctness and edge-case tests.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== MoE Environment: Public Tests ==="
uv run pytest tests/ -v "$@"
