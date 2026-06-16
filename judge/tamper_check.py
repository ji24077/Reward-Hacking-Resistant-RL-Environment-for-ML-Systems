"""
Tamper detection for judge-protected files.

Computes SHA-256 of every file in the protected set at judge start, then
compares against stored baseline checksums.  If any file has been modified,
the judge aborts and returns score=0.

Usage:
    from judge.tamper_check import verify_protected_files, TamperError

    try:
        verify_protected_files(ROOT)
    except TamperError as e:
        print(f"TAMPER DETECTED: {e}")
        sys.exit(1)

To regenerate the baseline (e.g. after intentional changes to reference code):
    python -m judge.tamper_check --regenerate
"""

import hashlib
import json
import os
import sys
from pathlib import Path


BASELINE_PATH = os.path.join(os.path.dirname(__file__), "checksums.json")

# Paths relative to repo root that the agent must not modify
PROTECTED_GLOBS = [
    "src/moe_env/reference.py",
    "src/moe_env/types.py",
    "src/moe_env/utils.py",
    "src/moe_env/__init__.py",
    "tests/test_public_correctness.py",
    "tests/test_public_edge_cases.py",
    "benchmarks/benchmark_moe.py",
]


class TamperError(RuntimeError):
    pass


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_checksums(root: str) -> dict[str, str]:
    result = {}
    for rel in PROTECTED_GLOBS:
        full = os.path.join(root, rel)
        if os.path.exists(full):
            result[rel] = _sha256(full)
        else:
            result[rel] = None  # file missing — judge will report
    return result


def save_baseline(root: str):
    checksums = compute_checksums(root)
    with open(BASELINE_PATH, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"Baseline checksums saved to {BASELINE_PATH}")
    for path, digest in checksums.items():
        status = digest[:12] + "..." if digest else "MISSING"
        print(f"  {path}: {status}")


def verify_protected_files(root: str):
    """Raise TamperError if any protected file has been modified."""
    if not os.path.exists(BASELINE_PATH):
        raise TamperError(
            f"Baseline checksums not found at {BASELINE_PATH}. "
            "Run: python -m judge.tamper_check --regenerate  (from a known-clean state)"
        )

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    current = compute_checksums(root)
    violations = []

    for rel, expected_digest in baseline.items():
        actual_digest = current.get(rel)
        if expected_digest is None and actual_digest is None:
            continue  # both missing, probably optional file
        if actual_digest != expected_digest:
            violations.append(
                f"  {rel}: expected={str(expected_digest)[:12]}... "
                f"got={str(actual_digest)[:12]}..."
            )

    if violations:
        msg = "Protected files have been modified:\n" + "\n".join(violations)
        raise TamperError(msg)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "--regenerate" in sys.argv:
        save_baseline(root)
    else:
        try:
            verify_protected_files(root)
            print("All protected files intact.")
        except TamperError as e:
            print(f"TAMPER DETECTED:\n{e}")
            sys.exit(1)
