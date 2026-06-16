"""
Judge — evaluates a submitted solution/solution.py.

Steps:
  1. Tamper check — SHA-256 of protected files
  2. Validate solution file exists and has the correct signature
  3. Run public correctness tests (subprocess)
  4. Run hidden correctness tests (fresh subprocess via hidden_runner.py)
  5. If any correctness test fails → score = 0
  6. Run GPU performance benchmark
  7. Return final score

Usage (from repo root):
    uv run python judge/judge.py [--device cuda] [--solution path/to/solution.py]
"""

import argparse
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, JUDGE_DIR)

from moe_env import moe_forward_reference, generate_moe_inputs, MoEConfig
from moe_env.utils import allclose_with_report, get_tolerances
from tamper_check import verify_protected_files, TamperError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_solution(path: str):
    spec = importlib.util.spec_from_file_location("_solution_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_signature(fn) -> tuple[bool, str]:
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        expected = ["x", "expert_ids", "expert_weights", "w1", "b1", "w2", "b2"]
        if params != expected:
            return False, f"Expected params {expected}, got {params}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def benchmark_fn(fn, inputs, warmup: int, trials: int, device: str) -> float:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        fn(*inputs)
    if use_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*inputs)
        if use_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Judge steps
# ---------------------------------------------------------------------------

def run_public_tests(solution_path: str) -> tuple[bool, str]:
    """Run pytest in a fresh subprocess so monkeypatching cannot leak."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": f"{ROOT}/src:{ROOT}"},
    )
    return result.returncode == 0, result.stdout + result.stderr


def run_hidden_tests_isolated(solution_path: str, device: str) -> tuple[bool, list[dict]]:
    """
    Copy only solution.py to a temp dir, then run hidden_runner.py as a fresh
    subprocess pointed at that copy.  This ensures:
      1. The solution cannot import from reference.py at evaluation time
         (it only has access to whatever it explicitly imports from the env).
      2. Any monkeypatching the solution does (torch.allclose, etc.) is
         contained within the subprocess and cannot affect the judge.
    """
    runner = os.path.join(JUDGE_DIR, "hidden_runner.py")
    pythonpath = f"{ROOT}/src:{JUDGE_DIR}"

    with tempfile.TemporaryDirectory(prefix="moe_eval_") as tmpdir:
        # Copy only the solution file — nothing else from the agent workspace
        sol_copy = os.path.join(tmpdir, "solution.py")
        shutil.copy2(solution_path, sol_copy)

        result = subprocess.run(
            [sys.executable, runner, sol_copy, device],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": pythonpath},
            timeout=300,
        )

    if result.returncode != 0 and not result.stdout.strip():
        return False, [{"case": "__runner__", "passed": False,
                         "error": result.stderr[-500:]}]

    try:
        results = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, [{"case": "__runner__", "passed": False,
                         "error": f"invalid JSON output: {result.stdout[:200]}"}]

    all_passed = all(r.get("passed", False) for r in results)
    return all_passed, results


BENCHMARK_HIDDEN = [
    # name,           T,     E,   D,   H,  K, seed, routing
    ("bench_small",   256,   8,  64, 128,  2,  42, "uniform"),
    ("bench_medium",  1024,  8, 128, 256,  2,  42, "uniform"),
    ("bench_large",   4096, 16, 256, 512,  2,  42, "uniform"),
    ("bench_skewed",  4096,  8, 128, 256,  2,  42, "skewed"),
    ("bench_top1",    4096, 16, 256, 512,  1,  42, "uniform"),
]


def run_performance_benchmark(moe_forward_fn, device: str,
                               warmup: int = 10, trials: int = 50) -> tuple[float, list[dict]]:
    speedups = []
    details = []
    for (name, T, E, D, H, K, seed, routing) in BENCHMARK_HIDDEN:
        cfg = MoEConfig(num_experts=E, token_dim=D, expert_hidden_dim=H,
                        top_k=K, device=device)
        inp = generate_moe_inputs(cfg, T, seed=seed, routing=routing)
        inputs = inp.as_tuple()

        ref_ms = benchmark_fn(moe_forward_reference, inputs, warmup, trials, device)
        sol_ms = benchmark_fn(moe_forward_fn, inputs, warmup, trials, device)
        speedup = ref_ms / max(sol_ms, 1e-6)
        speedups.append(speedup)
        print(f"  {name:<20} ref={ref_ms:.2f}ms  sol={sol_ms:.2f}ms  {speedup:.2f}x")
        details.append({
            "config": name,
            "tokens": T,
            "ref_ms": round(ref_ms, 4),
            "sol_ms": round(sol_ms, 4),
            "speedup": round(speedup, 4),
        })

    return sum(speedups) / len(speedups), details


# ---------------------------------------------------------------------------
# Main judge entry point
# ---------------------------------------------------------------------------

def judge(solution_path: str, device: str, warmup: int, trials: int) -> dict:
    report = {
        "solution_path": solution_path,
        "device": device,
        "score": 0.0,
        "correctness_passed": False,
        "public_tests_passed": False,
        "hidden_tests_passed": False,
        "tamper_check_passed": False,
        "avg_speedup": None,
        "normalized_speedup": None,
        "details": {},
    }

    # Step 1: tamper check
    print("\n[0/4] Tamper check ...")
    try:
        verify_protected_files(ROOT)
        report["tamper_check_passed"] = True
        print("  PASSED")
    except TamperError as e:
        report["details"]["tamper_error"] = str(e)
        print(f"  FAILED: {e}")
        return report

    # Step 2: file + import + signature
    if not os.path.exists(solution_path):
        report["details"]["error"] = "solution.py not found"
        return report

    try:
        mod = load_solution(solution_path)
    except Exception as e:
        report["details"]["import_error"] = traceback.format_exc()
        return report

    if not hasattr(mod, "moe_forward"):
        report["details"]["error"] = "moe_forward not found in solution.py"
        return report

    sig_ok, sig_msg = check_signature(mod.moe_forward)
    report["details"]["signature"] = sig_msg
    if not sig_ok:
        return report

    # Step 3: public tests (subprocess)
    print("\n[1/4] Running public tests ...")
    pub_passed, pub_log = run_public_tests(solution_path)
    report["public_tests_passed"] = pub_passed
    report["details"]["public_test_log"] = pub_log[-3000:]
    if not pub_passed:
        print("  FAILED")
        print(pub_log[-1500:])
        return report
    print("  PASSED")

    # Step 4: hidden tests (fresh subprocess, temp dir)
    print("\n[2/4] Running hidden correctness tests (isolated subprocess) ...")
    hidden_passed, hidden_results = run_hidden_tests_isolated(solution_path, device)
    report["hidden_tests_passed"] = hidden_passed
    report["details"]["hidden_test_results"] = hidden_results

    n_total = len(hidden_results)
    n_failed = sum(1 for r in hidden_results if not r.get("passed", False))
    print(f"  {n_total - n_failed}/{n_total} passed")
    for f in [r for r in hidden_results if not r.get("passed")][:5]:
        print(f"  FAIL: {f['case']} — {f.get('error', f.get('max_abs_err', '?'))}")

    if not hidden_passed:
        print("  Correctness gate FAILED → score = 0")
        return report

    report["correctness_passed"] = True

    # Step 5: benchmark (in-process — solution is already trusted at this point)
    print(f"\n[3/4] Running performance benchmark on {device} ...")
    avg_speedup, bench_details = run_performance_benchmark(
        mod.moe_forward, device, warmup, trials)
    normalized = min(avg_speedup, 3.0) / 3.0
    report["avg_speedup"] = avg_speedup
    report["normalized_speedup"] = normalized
    report["details"]["benchmark_results"] = bench_details

    score = 0.7 + 0.3 * normalized
    report["score"] = round(score, 4)

    print(f"\n[4/4] Score")
    print(f"  Avg speedup:        {avg_speedup:.2f}x")
    print(f"  Normalized speedup: {normalized:.3f}")
    print(f"  Final score:        {score:.4f}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution/solution.py")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    solution_path = os.path.join(ROOT, args.solution)
    report = judge(solution_path, args.device, args.warmup, args.trials)

    print("\n" + "=" * 50)
    print(f"FINAL SCORE: {report['score']:.4f}")
    print("=" * 50)

    if args.output:
        if out_dir := os.path.dirname(args.output):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}")

    sys.exit(0 if report["correctness_passed"] else 1)


if __name__ == "__main__":
    main()
