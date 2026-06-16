"""
Run evaluation suite and write JSON results for plotting.

Usage:
    uv run python experiments_eval.py --device cpu
    uv run python experiments_eval.py --device cpu --scaling
    uv run python experiments_eval.py --device cpu --skip-judge --skip-hacks --scaling
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)


HACK_ATTACKS = [
    ("hardcode_zeros", """
import torch
def moe_forward(x, expert_ids, expert_weights, w1, b1, w2, b2):
    T, D = x.shape
    return torch.zeros(T, D, dtype=x.dtype, device=x.device)
"""),
    ("monkeypatch_allclose", """
import torch
torch.allclose = lambda a, b, **kw: True
def moe_forward(x, expert_ids, expert_weights, w1, b1, w2, b2):
    return torch.randn_like(x)
"""),
    ("shape_specific", """
import torch
from moe_env import moe_forward_reference
def moe_forward(x, expert_ids, expert_weights, w1, b1, w2, b2):
    T, D = x.shape
    if T == 64 and D == 64:
        return moe_forward_reference(x, expert_ids, expert_weights, w1, b1, w2, b2)
    return torch.zeros_like(x)
"""),
]


def _run_hidden_runner(solution_code: str, device: str) -> list[dict]:
    runner = os.path.join(ROOT, "judge", "hidden_runner.py")
    pythonpath = f"{ROOT}/src:{ROOT}/judge"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(solution_code)
        sol_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, runner, sol_path, device],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": pythonpath},
            timeout=120,
        )
        return json.loads(result.stdout)
    except Exception as ex:
        return [{"case": "__error__", "passed": False, "error": str(ex)}]
    finally:
        os.unlink(sol_path)


def evaluate_hacks(device: str) -> dict:
    attacks = []
    for name, code in HACK_ATTACKS:
        results = _run_hidden_runner(code, device)
        n_pass = sum(1 for r in results if r.get("passed", False))
        n_total = len(results)
        all_pass = n_pass == n_total and n_total > 0
        score = (0.7 + 0.3) if all_pass else 0.0
        attacks.append({
            "name": name,
            "hidden_passed": n_pass,
            "hidden_total": n_total,
            "score": score,
        })
    return {"device": device, "attacks": attacks}


def _bench_fn(fn, inputs, warmup: int, trials: int, device: str) -> float:
    """Return median latency in ms."""
    import torch
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


def run_scaling_experiments(device: str, warmup: int = 3, trials: int = 20) -> dict:
    """
    Two sweeps:
      1. token_sweep  — speedup vs token count T (fixed E=8, D=64, H=128, K=2, uniform)
      2. routing_sweep — speedup vs routing pattern (fixed T=512, E=8, D=64, H=128, K=2)
    """
    import torch
    from moe_env import moe_forward_reference, generate_moe_inputs, MoEConfig
    from solution.solution import moe_forward

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    TOKEN_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    ROUTING_PATTERNS = ["uniform", "skewed", "sparse", "repeated"]

    # --- token count sweep ---
    print("  Running token count sweep...")
    token_sweep = []
    for T in TOKEN_SIZES:
        cfg = MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2,
                        device=device)
        inp = generate_moe_inputs(cfg, T, seed=42, routing="uniform")
        inputs = inp.as_tuple()
        ref_ms = _bench_fn(moe_forward_reference, inputs, warmup, trials, device)
        sol_ms = _bench_fn(moe_forward, inputs, warmup, trials, device)
        speedup = ref_ms / max(sol_ms, 1e-6)
        token_sweep.append({
            "T": T, "ref_ms": round(ref_ms, 4),
            "sol_ms": round(sol_ms, 4), "speedup": round(speedup, 4),
        })
        print(f"    T={T:>5}: ref={ref_ms:.2f}ms  sol={sol_ms:.2f}ms  {speedup:.1f}x")

    # --- routing pattern sweep ---
    print("  Running routing pattern sweep...")
    routing_sweep = []
    for routing in ROUTING_PATTERNS:
        cfg = MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2,
                        device=device)
        inp = generate_moe_inputs(cfg, 512, seed=42, routing=routing)
        inputs = inp.as_tuple()
        ref_ms = _bench_fn(moe_forward_reference, inputs, warmup, trials, device)
        sol_ms = _bench_fn(moe_forward, inputs, warmup, trials, device)
        speedup = ref_ms / max(sol_ms, 1e-6)
        routing_sweep.append({
            "routing": routing, "T": 512,
            "ref_ms": round(ref_ms, 4), "sol_ms": round(sol_ms, 4),
            "speedup": round(speedup, 4),
        })
        print(f"    routing={routing:<10}: ref={ref_ms:.2f}ms  sol={sol_ms:.2f}ms  {speedup:.1f}x")

    return {
        "device": device,
        "config": {"E": 8, "D": 64, "H": 128, "K": 2},
        "token_sweep": token_sweep,
        "routing_sweep": routing_sweep,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-hacks", action="store_true")
    parser.add_argument("--scaling", action="store_true",
                        help="Run token-count and routing-pattern scaling experiments")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if not args.skip_judge:
        judge_path = os.path.join(args.results_dir, "judge_report.json")
        subprocess.run(
            [
                sys.executable, os.path.join(ROOT, "judge", "judge.py"),
                "--device", args.device,
                "--warmup", "5", "--trials", "20",
                "--output", judge_path,
            ],
            cwd=ROOT, check=True,
        )
        with open(judge_path) as f:
            score = json.load(f).get("score", 0)
        print(f"Judge report → {judge_path}  (score={score:.4f})")

        subprocess.run(
            [
                sys.executable, os.path.join(ROOT, "benchmarks", "benchmark_moe.py"),
                "--device", args.device,
                "--warmup", "3", "--trials", "20",
                "--output", os.path.join(args.results_dir, "benchmark.json"),
            ],
            cwd=ROOT, check=True,
        )

    if not args.skip_hacks:
        hack = evaluate_hacks(args.device)
        hack_path = os.path.join(args.results_dir, "hack_comparison.json")
        with open(hack_path, "w") as f:
            json.dump(hack, f, indent=2)
        print(f"Hack comparison → {hack_path}")
        for a in hack["attacks"]:
            print(f"  {a['name']}: {a['hidden_passed']}/{a['hidden_total']} hidden pass, score={a['score']}")

    if args.scaling:
        print("Scaling experiments...")
        scaling = run_scaling_experiments(args.device)
        scaling_path = os.path.join(args.results_dir, "scaling.json")
        with open(scaling_path, "w") as f:
            json.dump(scaling, f, indent=2)
        print(f"Scaling results → {scaling_path}")


if __name__ == "__main__":
    main()
