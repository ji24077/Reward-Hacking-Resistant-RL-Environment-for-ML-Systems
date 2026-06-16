"""
Public benchmark — measures speedup of solution vs reference on several configs.
The hidden benchmark uses larger hidden shapes and hidden seeds.

Usage:
    uv run python benchmarks/benchmark_moe.py
    uv run python benchmarks/benchmark_moe.py --device cuda --warmup 20 --trials 100
    uv run python benchmarks/benchmark_moe.py --output results/benchmark.json
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moe_env import moe_forward_reference, generate_moe_inputs, MoEConfig
from moe_env.utils import allclose_with_report, get_tolerances
from solution.solution import moe_forward


BENCHMARK_CONFIGS = [
    # name,               T,    E,  D,   H,  K
    ("small",           128,   8, 64, 128,  2),
    ("medium",          512,   8, 128, 256, 2),
    ("large",          2048,  16, 256, 512, 2),
    ("top1_large",     2048,  16, 256, 512, 1),
    ("skewed_large",   2048,   8, 128, 256, 2),
]


def benchmark_fn(fn, inputs, warmup: int, trials: int, device: str) -> float:
    """Returns median latency in milliseconds."""
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for _ in range(warmup):
        _ = fn(*inputs)
    if use_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        if use_cuda:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = fn(*inputs)
            torch.cuda.synchronize()
        else:
            t0 = time.perf_counter()
            _ = fn(*inputs)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def run_benchmark(device: str, warmup: int, trials: int, seed: int = 42) -> dict:
    rows = []
    total_speedup = 0.0

    print(f"\n{'='*72}")
    print(f"MoE Forward Benchmark  |  device={device}  warmup={warmup}  trials={trials}")
    print(f"{'='*72}")
    fmt = "{:<20} {:>8} {:>10} {:>10} {:>10} {:>8}"
    print(fmt.format("config", "tokens", "ref (ms)", "sol (ms)", "speedup", "status"))
    print("-" * 72)

    for name, T, E, D, H, K in BENCHMARK_CONFIGS:
        routing = "skewed" if "skewed" in name else "uniform"
        cfg = MoEConfig(num_experts=E, token_dim=D, expert_hidden_dim=H, top_k=K,
                        device=device)
        inp = generate_moe_inputs(cfg, T, seed=seed, routing=routing)
        inputs = inp.as_tuple()

        ref_out = moe_forward_reference(*inputs)
        sol_out = moe_forward(*inputs)
        atol, rtol = get_tolerances(cfg.dtype)
        passed, info = allclose_with_report(sol_out, ref_out, atol=atol, rtol=rtol)
        status = "OK" if passed else f"FAIL(err={info['max_abs_err']:.1e})"

        ref_ms = benchmark_fn(moe_forward_reference, inputs, warmup, trials, device)
        sol_ms = benchmark_fn(moe_forward, inputs, warmup, trials, device)
        speedup = ref_ms / sol_ms if sol_ms > 0 else float("inf")
        total_speedup += speedup

        print(fmt.format(name, T, f"{ref_ms:.3f}", f"{sol_ms:.3f}", f"{speedup:.2f}x", status))

        rows.append({
            "config": name,
            "tokens": T,
            "experts": E,
            "token_dim": D,
            "hidden_dim": H,
            "top_k": K,
            "routing": routing,
            "ref_ms": round(ref_ms, 4),
            "sol_ms": round(sol_ms, 4),
            "speedup": round(speedup, 4),
            "correctness_ok": passed,
        })

    n = len(rows)
    avg = total_speedup / n if n > 0 else 0.0
    capped = min(avg, 3.0) / 3.0
    score = 0.7 + 0.3 * capped

    print("-" * 72)
    print(f"\nAvg speedup: {avg:.2f}x  (capped at 3x)")
    print(f"Estimated score (if all correct): {score:.3f}  [0.7 base + 0.3 * perf]")

    return {
        "device": device,
        "warmup": warmup,
        "trials": trials,
        "seed": seed,
        "configs": rows,
        "avg_speedup": round(avg, 4),
        "normalized_speedup": round(capped, 4),
        "estimated_score": round(score, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Write JSON results to this path")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    report = run_benchmark(args.device, args.warmup, args.trials, args.seed)

    if args.output:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
