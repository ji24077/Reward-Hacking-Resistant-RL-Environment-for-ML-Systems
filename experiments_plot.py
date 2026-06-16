"""
Generate figures from judge/benchmark JSON outputs.

Usage:
    uv run python experiments_plot.py \
        --judge results/judge_report.json \
        --benchmark results/benchmark.json \
        --hack results/hack_comparison.json \
        --scaling results/scaling.json \
        --out figures/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Existing plots
# ---------------------------------------------------------------------------

def plot_hidden_errors(judge: dict, out_dir: Path):
    results = judge.get("details", {}).get("hidden_test_results", [])
    if not results:
        return

    cases = [r["case"] for r in results]
    errors = [max(r.get("max_abs_err", 0.0), 1e-12) for r in results]  # avoid log(0)
    passed = [r.get("passed", False) for r in results]
    colors = ["#2ecc71" if p else "#e74c3c" for p in passed]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(cases)), errors, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("max absolute error")
    ax.set_title("Hidden test numerical error margin")
    ax.axhline(1e-5, color="gray", linestyle="--", linewidth=0.8, label="FP32 atol=1e-5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "hidden_error_margin.png", dpi=150)
    plt.close(fig)


def plot_score_breakdown(judge: dict, out_dir: Path):
    score = judge.get("score", 0.0)
    if not judge.get("correctness_passed"):
        base, perf = 0.0, 0.0
    else:
        norm = judge.get("normalized_speedup") or 0.0
        base, perf = 0.7, 0.3 * norm

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(["score"], [base], color="#3498db", label="correctness base (0.7)")
    ax.barh(["score"], [perf], left=[base], color="#e67e22", label="performance (0.3 × norm)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("score")
    ax.set_title(f"Score breakdown  (total={score:.3f})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "score_breakdown.png", dpi=150)
    plt.close(fig)


def plot_speedup(bench_rows: list[dict], title: str, out_path: Path):
    if not bench_rows:
        return

    names = [r["config"] for r in bench_rows]
    ref = [r["ref_ms"] for r in bench_rows]
    sol = [r["sol_ms"] for r in bench_rows]
    x = range(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - w / 2 for i in x], ref, width=w, label="reference (ms)", color="#95a5a6")
    ax.bar([i + w / 2 for i in x], sol, width=w, label="solution (ms)", color="#2980b9")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("median latency (ms)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    speedups = [r["speedup"] for r in bench_rows]
    ax.bar(names, speedups, color="#27ae60")
    ax.axhline(3.0, color="red", linestyle="--", linewidth=0.8, label="3x cap")
    ax.set_ylabel("speedup")
    ax.set_title(f"{title} — speedup")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path.with_name(out_path.stem + "_speedup.png"), dpi=150)
    plt.close(fig)


def plot_hack_comparison(hack: dict, out_dir: Path):
    rows = hack.get("attacks", [])
    if not rows:
        return

    labels = [r["name"] for r in rows]
    passed = [r["hidden_passed"] for r in rows]
    total = rows[0].get("hidden_total", 24)
    scores = [r.get("score", 0.0) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#e74c3c"] * len(rows)  # all fail
    bars = axes[0].bar(labels, passed, color=colors)
    axes[0].axhline(total, color="#2ecc71", linestyle="--", linewidth=1.2,
                    label=f"all {total} pass (correct solution)")
    axes[0].set_ylabel("hidden tests passed")
    axes[0].set_title("Reward hacking — hidden pass count")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    for bar, v in zip(bars, passed):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(v), ha="center", va="bottom", fontsize=9)

    axes[1].bar(labels, scores, color=colors)
    axes[1].axhline(1.0, color="#2ecc71", linestyle="--", linewidth=1.2,
                    label="max score (correct solution)")
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("score")
    axes[1].set_title("Reward hacking — resulting score")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=8)
    for bar, v in zip(bars, scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, 0.02,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="white",
                     fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "hack_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# New analysis plots
# ---------------------------------------------------------------------------

def plot_scaling_curve(scaling: dict, out_dir: Path):
    """Speedup vs token count + absolute latency on log-log axes."""
    rows = scaling.get("token_sweep", [])
    if not rows:
        return

    T_vals = [r["T"] for r in rows]
    speedups = [r["speedup"] for r in rows]
    ref_ms = [r["ref_ms"] for r in rows]
    sol_ms = [r["sol_ms"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: speedup vs T
    ax1.plot(T_vals, speedups, "o-", color="#27ae60", linewidth=2, markersize=6,
             label="solution speedup")
    ax1.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=0.9, label="1× (no gain)")
    ax1.axhline(3.0, color="gray", linestyle=":", linewidth=0.9, label="3× score cap")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("token count T")
    ax1.set_ylabel("speedup over reference")
    ax1.set_title("Speedup vs Token Count\n(E=8, D=64, H=128, K=2, uniform routing)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax1.set_xticks(T_vals)
    ax1.legend()

    # Annotate max speedup point
    max_idx = speedups.index(max(speedups))
    ax1.annotate(f"{speedups[max_idx]:.1f}×",
                 xy=(T_vals[max_idx], speedups[max_idx]),
                 xytext=(T_vals[max_idx], speedups[max_idx] + 1.5),
                 ha="center", fontsize=8, color="#27ae60",
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1))

    # Right: absolute latency (log scale)
    ax2.plot(T_vals, ref_ms, "s--", color="#95a5a6", linewidth=2, markersize=6,
             label="reference")
    ax2.plot(T_vals, sol_ms, "o-", color="#2980b9", linewidth=2, markersize=6,
             label="solution")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("token count T")
    ax2.set_ylabel("median latency (ms, log scale)")
    ax2.set_title("Absolute Latency vs Token Count")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: str(int(v))))
    ax2.set_xticks(T_vals)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "scaling_speedup.png", dpi=150)
    plt.close(fig)


def plot_routing_comparison(scaling: dict, out_dir: Path):
    """Speedup and latency breakdown by routing pattern."""
    rows = scaling.get("routing_sweep", [])
    if not rows:
        return

    labels = [r["routing"] for r in rows]
    speedups = [r["speedup"] for r in rows]
    ref_ms = [r["ref_ms"] for r in rows]
    sol_ms = [r["sol_ms"] for r in rows]

    # Color code: skewed = best (dark green), others proportionally
    palette = {"uniform": "#2980b9", "skewed": "#27ae60",
               "sparse": "#8e44ad", "repeated": "#e67e22"}
    colors = [palette.get(l, "#95a5a6") for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: speedup
    bars = ax1.bar(labels, speedups, color=colors)
    ax1.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=0.9)
    ax1.set_ylabel("speedup over reference")
    ax1.set_title("Speedup by Routing Pattern\n(T=512, E=8, D=64, H=128, K=2)")
    for bar, v in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{v:.1f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: grouped latency
    x = range(len(labels))
    w = 0.35
    ax2.bar([i - w / 2 for i in x], ref_ms, width=w, color="#95a5a6", label="reference (ms)")
    ax2.bar([i + w / 2 for i in x], sol_ms, width=w, color=colors, label="solution (ms)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("median latency (ms)")
    ax2.set_title("Reference vs Solution Latency by Routing")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "routing_comparison.png", dpi=150)
    plt.close(fig)


def plot_error_by_dtype(judge: dict, out_dir: Path):
    """Max absolute error distribution grouped by dtype with tolerance lines."""
    results = judge.get("details", {}).get("hidden_test_results", [])
    if not results:
        return

    dtype_groups: dict[str, list[float]] = {}
    for r in results:
        dtype = r.get("dtype", "float32")
        err = r.get("max_abs_err", 0.0)
        dtype_groups.setdefault(dtype, []).append(err)

    dtype_order = ["float32", "float16", "bfloat16"]
    dtype_labels = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
    dtype_atol = {"float32": 1e-5, "float16": 1e-2, "bfloat16": 1e-2}
    dtype_colors = {"float32": "#3498db", "float16": "#e67e22", "bfloat16": "#9b59b6"}

    present = [d for d in dtype_order if d in dtype_groups]
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 1, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, dtype in zip(axes, present):
        errs = dtype_groups[dtype]
        # Replace exact zeros with a small sentinel for log-scale display
        display_errs = [max(e, 1e-13) for e in errs]
        case_count = len(errs)
        xs = range(case_count)

        ax.bar(xs, display_errs, color=dtype_colors[dtype], alpha=0.8)
        atol = dtype_atol[dtype]
        ax.axhline(atol, color="#e74c3c", linestyle="--", linewidth=1.0,
                   label=f"atol={atol:.0e}")
        ax.set_yscale("log")
        ax.set_xlabel(f"{dtype_labels[dtype]} test case index")
        ax.set_ylabel("max absolute error" if dtype == present[0] else "")
        ax.set_title(f"{dtype_labels[dtype]}\n({case_count} cases, all pass)")
        ax.legend(fontsize=8)

        n_zero = sum(1 for e in errs if e == 0.0)
        if n_zero > 0:
            ax.text(0.5, 0.15, f"{n_zero}/{case_count} exact\n(error = 0)",
                    transform=ax.transAxes, ha="center", fontsize=8,
                    color=dtype_colors[dtype], fontweight="bold")

    fig.suptitle("Numerical Error by Dtype — All Hidden Tests", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_dtype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate assessment figures")
    parser.add_argument("--judge", default="results/judge_report.json")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--hack", default=None)
    parser.add_argument("--scaling", default=None, help="results/scaling.json from experiments_eval.py")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.path.exists(args.judge):
        judge = _load_json(args.judge)
        plot_hidden_errors(judge, out_dir)
        plot_score_breakdown(judge, out_dir)
        plot_error_by_dtype(judge, out_dir)
        bench = judge.get("details", {}).get("benchmark_results")
        if bench:
            plot_speedup(bench, f"Judge benchmark ({judge.get('device', '?')})",
                         out_dir / "judge_benchmark.png")
        print(f"Plotted judge report from {args.judge}")
    else:
        print(f"Judge report not found: {args.judge}")

    if args.benchmark and os.path.exists(args.benchmark):
        bench = _load_json(args.benchmark)
        plot_speedup(bench.get("configs", []),
                     f"Public benchmark ({bench.get('device', '?')})",
                     out_dir / "public_benchmark.png")
        print(f"Plotted benchmark from {args.benchmark}")

    if args.hack and os.path.exists(args.hack):
        plot_hack_comparison(_load_json(args.hack), out_dir)
        print(f"Plotted hack comparison from {args.hack}")

    if args.scaling and os.path.exists(args.scaling):
        scaling = _load_json(args.scaling)
        plot_scaling_curve(scaling, out_dir)
        plot_routing_comparison(scaling, out_dir)
        print(f"Plotted scaling analysis from {args.scaling}")

    print(f"Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
