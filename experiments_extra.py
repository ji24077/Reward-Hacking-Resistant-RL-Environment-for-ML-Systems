"""
Extra experiments for research depth:
  1. torch.compile comparison
  2. Expert load distribution across routing patterns
  3. Time breakdown inside solution (mask build vs matmul vs scatter)

Usage:
    uv run python experiments_extra.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from moe_env import moe_forward_reference, generate_moe_inputs, MoEConfig
from solution.solution import moe_forward


def _bench(fn, inputs, warmup, trials, device):
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
# 1. torch.compile comparison
# ---------------------------------------------------------------------------
def run_compile_comparison(device: str, warmup=5, trials=30) -> dict:
    print("  torch.compile comparison...")
    configs = [
        (128, 8, 64, 128, 2, "uniform"),
        (512, 8, 64, 128, 2, "uniform"),
        (2048, 8, 64, 128, 2, "uniform"),
        (2048, 8, 64, 128, 2, "skewed"),
        (4096, 8, 64, 128, 2, "uniform"),
    ]
    try:
        compiled = torch.compile(moe_forward, fullgraph=False)
    except Exception:
        compiled = None

    rows = []
    for T, E, D, H, K, routing in configs:
        cfg = MoEConfig(num_experts=E, token_dim=D, expert_hidden_dim=H, top_k=K, device=device)
        inp = generate_moe_inputs(cfg, T, seed=42, routing=routing)
        inputs = inp.as_tuple()

        ref_ms = _bench(moe_forward_reference, inputs, warmup, trials, device)
        sol_ms = _bench(moe_forward, inputs, warmup, trials, device)

        compile_ms = None
        if compiled is not None:
            # warm up compile
            for _ in range(3):
                compiled(*inputs)
            compile_ms = round(_bench(compiled, inputs, warmup, trials, device), 4)

        label = f"T={T},E={E},routing={routing}"
        rows.append({
            "config": label, "T": T, "E": E, "routing": routing,
            "ref_ms": round(ref_ms, 4),
            "sol_ms": round(sol_ms, 4),
            "compile_ms": compile_ms,
            "sol_speedup": round(ref_ms / max(sol_ms, 1e-6), 2),
            "compile_speedup": round(ref_ms / max(compile_ms or sol_ms, 1e-6), 2),
        })
        print(f"    {label}: ref={ref_ms:.2f}ms  sol={sol_ms:.2f}ms  "
              f"compile={'N/A' if compile_ms is None else f'{compile_ms:.2f}ms'}")

    return {"device": device, "rows": rows}


# ---------------------------------------------------------------------------
# 2. Expert load distribution
# ---------------------------------------------------------------------------
def run_load_distribution(device: str) -> dict:
    print("  Expert load distribution...")
    T, E, K = 512, 8, 2
    cfg = MoEConfig(num_experts=E, token_dim=64, expert_hidden_dim=128, top_k=K, device=device)
    patterns = ["uniform", "skewed", "sparse", "repeated"]
    result = {}
    for routing in patterns:
        inp = generate_moe_inputs(cfg, T, seed=42, routing=routing)
        # count tokens per expert
        counts = [0] * E
        for t in range(T):
            seen = set()
            for k in range(K):
                e = inp.expert_ids[t, k].item()
                if e not in seen:
                    counts[e] += 1
                    seen.add(e)
        total_assignments = sum(counts)
        gini = _gini(counts)
        result[routing] = {
            "counts": counts,
            "max": max(counts),
            "min": min(counts),
            "mean": round(total_assignments / E, 1),
            "gini": round(gini, 4),
        }
        print(f"    {routing}: counts={counts}  gini={gini:.3f}")
    return {"T": T, "E": E, "K": K, "distributions": result}


def _gini(values: list[int]) -> float:
    n = len(values)
    if n == 0 or sum(values) == 0:
        return 0.0
    s = sorted(values)
    cumsum = 0.0
    for i, v in enumerate(s):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * sum(values))


# ---------------------------------------------------------------------------
# 3. Time breakdown inside solution
# ---------------------------------------------------------------------------
def run_time_breakdown(device: str, T=512, warmup=5, trials=30) -> dict:
    """
    Manually instrument phases inside the solution:
      Phase A: contiguous check + output buffer allocation
      Phase B: per-expert mask + index gather (loop over E experts)
      Phase C: batched matmul per expert  (w1 forward + gelu + w2)
      Phase D: scatter_add_ back to output
    """
    print("  Time breakdown inside solution...")
    E, D, H, K = 8, 64, 128, 2
    cfg = MoEConfig(num_experts=E, token_dim=D, expert_hidden_dim=H, top_k=K, device=device)
    inp = generate_moe_inputs(cfg, T, seed=42, routing="uniform")
    x, expert_ids, expert_weights, w1, b1, w2, b2 = inp.as_tuple()
    x = x.contiguous()

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    phase_times = {"alloc": [], "mask_gather": [], "matmul": [], "scatter": []}

    for _ in range(warmup + trials):
        # Phase A: alloc
        sync()
        t0 = time.perf_counter()
        output = torch.zeros_like(x)
        sync()
        tA = (time.perf_counter() - t0) * 1000

        # Phase B + C + D: loop over experts
        tB = tC = tD = 0.0
        for e in range(E):
            sync()
            t1 = time.perf_counter()
            mask = (expert_ids == e).any(dim=1)
            if not mask.any():
                continue
            x_e = x[mask]
            w_e = expert_weights[mask][expert_ids[mask] == e].unsqueeze(1)
            sync()
            tB += (time.perf_counter() - t1) * 1000

            sync()
            t2 = time.perf_counter()
            h = F.gelu(x_e @ w1[e].T + b1[e])
            out_e = h @ w2[e].T + b2[e]
            sync()
            tC += (time.perf_counter() - t2) * 1000

            sync()
            t3 = time.perf_counter()
            idx = mask.nonzero(as_tuple=True)[0]
            output.index_add_(0, idx, w_e * out_e)
            sync()
            tD += (time.perf_counter() - t3) * 1000

        if _ >= warmup:
            phase_times["alloc"].append(tA)
            phase_times["mask_gather"].append(tB)
            phase_times["matmul"].append(tC)
            phase_times["scatter"].append(tD)

    def med(lst):
        lst.sort()
        return round(lst[len(lst) // 2], 4)

    breakdown = {k: med(v) for k, v in phase_times.items()}
    total = sum(breakdown.values())
    breakdown["total"] = round(total, 4)
    breakdown["pct"] = {k: round(v / total * 100, 1) for k, v in breakdown.items() if k != "total" and k != "pct"}
    print(f"    alloc={breakdown['alloc']:.3f}ms  mask_gather={breakdown['mask_gather']:.3f}ms  "
          f"matmul={breakdown['matmul']:.3f}ms  scatter={breakdown['scatter']:.3f}ms  "
          f"total={breakdown['total']:.3f}ms")
    return {"T": T, "E": E, "D": D, "H": H, "K": K, "device": device, "breakdown": breakdown}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        args.device = "cpu"

    os.makedirs(args.results_dir, exist_ok=True)

    compile_data = run_compile_comparison(args.device)
    load_data = run_load_distribution(args.device)
    breakdown_data = run_time_breakdown(args.device)

    out = {
        "compile_comparison": compile_data,
        "load_distribution": load_data,
        "time_breakdown": breakdown_data,
    }
    path = os.path.join(args.results_dir, "extra.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nExtra results → {path}")


if __name__ == "__main__":
    main()
