"""
Plot extra experiment results from results/extra.json.

Usage:
    uv run python experiments_extra_plot.py --extra results/extra.json --out figures/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_compile_comparison(data: dict, out_dir: Path):
    rows = data["rows"]
    labels = [r["config"].replace(",routing=", "\n") for r in rows]
    ref = [r["ref_ms"] for r in rows]
    sol = [r["sol_ms"] for r in rows]
    comp = [r["compile_ms"] or r["sol_ms"] for r in rows]
    x = np.arange(len(labels))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: latency
    ax1.bar(x - w, ref, width=w, color="#95a5a6", label="reference")
    ax1.bar(x,     sol, width=w, color="#2980b9", label="solution (eager)")
    ax1.bar(x + w, comp, width=w, color="#27ae60", label="solution (compiled)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel("median latency (ms)")
    ax1.set_title("Latency: Reference vs Eager vs torch.compile")
    ax1.legend(fontsize=8)

    # Right: speedup over reference
    sol_su = [r["sol_speedup"] for r in rows]
    comp_su = [r["compile_speedup"] for r in rows]
    ax2.plot(labels, sol_su, "o-", color="#2980b9", linewidth=2, markersize=6,
             label="eager speedup")
    ax2.plot(labels, comp_su, "s--", color="#27ae60", linewidth=2, markersize=6,
             label="compiled speedup")
    ax2.set_ylabel("speedup over reference")
    ax2.set_title("Speedup: Eager vs torch.compile\n(lines nearly identical → compile adds minimal gain)")
    ax2.tick_params(axis="x", labelsize=7)
    ax2.legend()
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "compile_comparison.png", dpi=150)
    plt.close(fig)


def plot_load_distribution(data: dict, out_dir: Path):
    dists = data["distributions"]
    patterns = ["uniform", "skewed", "sparse", "repeated"]
    palette = {"uniform": "#2980b9", "skewed": "#27ae60",
               "sparse": "#8e44ad", "repeated": "#e67e22"}
    E = data["E"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, routing in zip(axes, patterns):
        d = dists[routing]
        counts = d["counts"]
        color = palette[routing]
        bars = ax.bar(range(E), counts, color=color, alpha=0.85)
        mean = d["mean"]
        ax.axhline(mean, color="gray", linestyle="--", linewidth=1.0,
                   label=f"mean = {mean:.0f}")
        ax.set_title(
            f"{routing.capitalize()} routing\n"
            f"Gini={d['gini']:.3f}  max={d['max']}  min={d['min']}",
            fontsize=10
        )
        ax.set_xlabel("expert index")
        ax.set_ylabel("tokens assigned")
        ax.set_xticks(range(E))
        ax.legend(fontsize=8)
        for bar, v in zip(bars, counts):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(v), ha="center", va="bottom", fontsize=7)

    fig.suptitle(
        f"Expert Token Load Distribution  (T={data['T']}, E={data['E']}, K={data['K']})\n"
        "Gini coefficient: 0 = perfectly balanced, 1 = all tokens to one expert",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "load_distribution.png", dpi=150)
    plt.close(fig)


def plot_time_breakdown(data: dict, out_dir: Path):
    bd = data["breakdown"]
    phases = ["alloc", "mask_gather", "matmul", "scatter"]
    labels = ["Alloc\noutput", "Mask + Gather\n(Python loop)", "Batched\nMatMul", "Scatter\n(index_add_)"]
    times = [bd[p] for p in phases]
    total = bd["total"]
    pcts = [bd["pct"][p] for p in phases]
    colors = ["#bdc3c7", "#e74c3c", "#2980b9", "#27ae60"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: bar chart
    bars = ax1.bar(labels, times, color=colors)
    ax1.set_ylabel("median time (ms)")
    ax1.set_title(
        f"Solution Time Breakdown\n"
        f"T={data['T']}, E={data['E']}, D={data['D']}, H={data['H']}, K={data['K']}, "
        f"device={data['device']}\n"
        f"total = {total:.3f} ms"
    )
    for bar, v, pct in zip(bars, times, pcts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{v:.3f}ms\n({pct}%)", ha="center", va="bottom", fontsize=9)

    # Right: pie
    wedges, texts, autotexts = ax2.pie(
        times, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Time fraction by phase")

    bottleneck = phases[pcts.index(max(pcts))]
    fig.text(0.5, 0.01,
             f"Bottleneck: {bottleneck} ({max(pcts):.1f}%) — "
             "Python loop overhead; Triton kernel would eliminate this",
             ha="center", fontsize=9, color="#c0392b")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / "time_breakdown.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", default="results/extra.json")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load(args.extra)
    plot_compile_comparison(data["compile_comparison"], out_dir)
    print("Plotted compile_comparison.png")
    plot_load_distribution(data["load_distribution"], out_dir)
    print("Plotted load_distribution.png")
    plot_time_breakdown(data["time_breakdown"], out_dir)
    print("Plotted time_breakdown.png")
    print(f"Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
