"""
Subprocess-isolated hidden test runner.

Called by judge.py as a fresh Python process:
    python hidden_runner.py <solution_path> <device>

Outputs a JSON array of per-case results to stdout.
Running in a fresh process means any monkeypatching of torch.allclose,
torch.Tensor methods, etc. in solution.py cannot pollute the judge.
"""

import importlib.util
import json
import os
import sys
import traceback

import torch

# Set up imports — ROOT/src must be on path
RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(RUNNER_DIR)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, RUNNER_DIR)

from moe_env import moe_forward_reference, generate_moe_inputs, MoEConfig
from moe_env.utils import allclose_with_report, get_tolerances, make_repeated_expert_routing, make_expert_weights
from hidden_tests import HIDDEN_CASES


def load_solution(path: str):
    spec = importlib.util.spec_from_file_location("_solution_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_case(moe_forward_fn, case_tuple, device: str) -> dict:
    (name, T, E, D, H, K, seed, routing, dtype_str, contig) = case_tuple
    dtype = getattr(torch, dtype_str)
    cfg = MoEConfig(num_experts=E, token_dim=D, expert_hidden_dim=H,
                    top_k=K, dtype=dtype, device=device)

    inp = generate_moe_inputs(cfg, T, seed=seed, routing=routing,
                               make_contiguous=contig)
    expected = moe_forward_reference(*inp.as_tuple())
    got = moe_forward_fn(*inp.as_tuple())

    if got is None:
        return {"case": name, "passed": False, "error": "returned None"}
    if got.shape != expected.shape:
        return {"case": name, "passed": False,
                "error": f"shape {got.shape} != {expected.shape}"}
    if torch.isnan(got).any():
        return {"case": name, "passed": False, "error": "NaN in output"}
    if torch.isinf(got).any():
        return {"case": name, "passed": False, "error": "Inf in output"}

    atol, rtol = get_tolerances(dtype)
    passed, info = allclose_with_report(got, expected, atol=atol, rtol=rtol)
    return {
        "case": name,
        "passed": passed,
        "max_abs_err": info["max_abs_err"],
        "max_rel_err": info["max_rel_err"],
        "dtype": dtype_str,
    }


def main():
    if len(sys.argv) < 3:
        print(json.dumps([{"error": "usage: hidden_runner.py <solution_path> <device>"}]))
        sys.exit(1)

    solution_path = sys.argv[1]
    device = sys.argv[2]

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    try:
        mod = load_solution(solution_path)
    except Exception as e:
        print(json.dumps([{"case": "__import__", "passed": False,
                            "error": traceback.format_exc()}]))
        sys.exit(1)

    if not hasattr(mod, "moe_forward"):
        print(json.dumps([{"case": "__import__", "passed": False,
                            "error": "moe_forward not found"}]))
        sys.exit(1)

    moe_forward_fn = mod.moe_forward
    results = []

    for case in HIDDEN_CASES:
        try:
            result = run_case(moe_forward_fn, case, device)
        except Exception:
            result = {"case": case[0], "passed": False,
                      "error": traceback.format_exc()[-500:]}
        results.append(result)

    print(json.dumps(results))


if __name__ == "__main__":
    main()
