# Reward-Hacking-Resistant RL Environment for ML Systems
### Sparse Mixture-of-Experts Forward Operator

A correctness-gated RL environment where an agent must implement a sparse
Mixture-of-Experts (MoE) forward pass that is numerically identical to a hidden
oracle, then beat the oracle on GPU throughput. Inspired by ScatterMoE.

---

## Quick Start

```bash
uv sync
uv run pytest                                              # 25 public tests
uv run python benchmarks/benchmark_moe.py --device cuda   # benchmark
bash scripts/run_judge.sh cuda                            # full evaluation
```

---

## Results

### Correctness

| Test suite | Cases | Passed | Status |
|---|---|---|---|
| Public correctness | 10 | 10 | ✓ |
| Public edge cases | 15 | 15 | ✓ (1 skipped: no CUDA) |
| Hidden FP32 (unseen seeds) | 11 | 11 | ✓ |
| Hidden unseen shapes | 4 | 4 | ✓ |
| Hidden FP16 | 3 | 3 | ✓ |
| Hidden BF16 | 2 | 2 | ✓ |
| Hidden repeated expert IDs | 2 | 2 | ✓ |
| **Total hidden** | **24** | **24** | ✓ |

All hidden correctness tests pass. Max absolute error across all FP32 cases:
`1.19e-7` (well below the `1e-5` atol threshold).

### Performance (CPU, reference implementation baseline)

Measured on the public benchmark suite (warmup=3, trials=20):

| Config | Tokens | Experts | Ref (ms) | Solution (ms) | Speedup |
|---|---|---|---|---|---|
| small | 128 | 8 | 3.48 | 0.30 | **11.5×** |
| medium | 512 | 8 | 15.83 | 1.09 | **14.6×** |
| large | 2048 | 16 | 90.27 | 5.87 | **15.4×** |
| top1_large | 2048 | 16 | 45.90 | 4.28 | **10.7×** |
| skewed_large | 2048 | 8 | 63.16 | 2.37 | **26.7×** |
| **Average** | | | | | **15.8×** |

Hidden judge benchmark (warmup=5, trials=20):

| Config | Tokens | Ref (ms) | Solution (ms) | Speedup |
|---|---|---|---|---|
| bench_small | 256 | 7.15 | 0.51 | 14.1× |
| bench_medium | 1024 | 31.82 | 2.37 | 13.4× |
| bench_large | 4096 | 183.7 | 9.03 | 20.4× |
| bench_skewed | 4096 | 129.6 | 4.06 | 31.9× |
| bench_top1 | 4096 | 88.0 | 5.96 | 14.8× |
| **Average** | | | | **18.9×** |

### Final Score

```
score = 0.7 (correctness base) + 0.3 × min(18.9, 3.0) / 3.0 = 1.000
```

### Figures

| Figure | Description |
|---|---|
| `figures/hidden_error_margin.png` | Max absolute error per hidden test case (log scale) |
| `figures/score_breakdown.png` | Score decomposition: correctness base vs. performance bonus |
| `figures/error_by_dtype.png` | Error distribution grouped by dtype with tolerance thresholds |
| `figures/judge_benchmark.png` | Reference vs. solution latency (judge hidden configs) |
| `figures/judge_benchmark_speedup.png` | Speedup by config with 3× cap line |
| `figures/public_benchmark.png` | Reference vs. solution latency (public configs) |
| `figures/public_benchmark_speedup.png` | Public benchmark speedup |
| `figures/hack_comparison.png` | Reward hacking attempts: hidden test pass count and score |
| `figures/scaling_speedup.png` | Speedup and absolute latency vs. token count (log scale) |
| `figures/routing_comparison.png` | Speedup and latency breakdown by routing pattern |

![Score Breakdown](figures/score_breakdown.png)
![Hidden Error Margin](figures/hidden_error_margin.png)
![Judge Benchmark Speedup](figures/judge_benchmark_speedup.png)
![Hack Comparison](figures/hack_comparison.png)

---

## Analysis

### Scaling Behavior

The solution batches all tokens routed to the same expert into a single matrix
multiplication call instead of looping over each token individually. This is
the key reason why speedup increases with the token count T.

At small T (T=32), the per-expert loop overhead in the solution (looping over
E experts, masking, calling `mm`) dominates the savings from batching. The
advantage only materializes once there are enough tokens per expert to amortize
the overhead of building the index mask and constructing the batched input. At
T=4096 with E=8 experts and uniform routing, each expert receives ~1024 tokens
on average — a shape where a single `mm(x_expert, w1[e].T)` call is far
faster than 1024 scalar dot-products.

The scaling plot (`figures/scaling_speedup.png`) shows this clearly: speedup
rises from 3.8× at T=32 to 40.6× at T=4096 (CPU), with the slope steepening
as the token count increases. On GPU, where BLAS kernels are even more
efficient and Python loop overhead is proportionally larger, the crossover
point shifts earlier and the plateau speedup is higher.

### Routing Pattern Effects

The routing pattern changes how evenly tokens are distributed across experts,
and that distribution determines the average matmul size the solution can form.

Measured at T=512, E=8, D=64, H=128, K=2:

| Routing | Ref (ms) | Sol (ms) | Speedup |
|---|---|---|---|
| uniform | 13.6 | 0.74 | **18.4×** |
| skewed | 13.2 | 0.65 | **20.1×** |
| sparse | 13.5 | 0.50 | **27.2×** |
| repeated | 13.5 | 0.71 | **19.1×** |

- **Sparse routing** (many experts receive no tokens) gives the highest
  speedup (27.2×) because the solution's `if not mask.any(): continue` guard
  completely skips empty experts. The reference still performs K inner-loop
  iterations per token regardless of whether the selected expert has any
  tokens — it never early-exits.
- **Skewed routing** (one expert receives the majority of tokens) gives the
  next highest speedup (20.1×). More tokens per dominant expert → a larger,
  more efficient single `mm` call for that expert.
- **Uniform routing** spreads tokens evenly, giving each expert a
  moderately-sized batch with consistent efficiency (18.4×).
- **Repeated expert IDs** (a token routes to the same expert twice) is
  handled by `scatter_add_`, which accumulates both contributions correctly
  in a single pass (19.1×).

See `figures/routing_comparison.png`.

### Numerical Precision

FP16 and BF16 hidden test cases show a max absolute error of exactly `0.0`
(identical bit-for-bit to the reference). This happens because both the
solution and the reference operate on the same per-token tensors in the same
dtype; when all arithmetic is done in the same precision and the accumulation
order is token-level (not across tokens), the results are bitwise identical.

FP32 cases show small non-zero errors (up to ~1.19e-7) because the batched
matmul in the solution accumulates values across different tokens simultaneously,
which changes the floating-point summation order versus the reference's
sequential token loop. All FP32 errors are well below the `1e-5` atol
threshold. See `figures/error_by_dtype.png`.

### Reward Hacking Attempts

| Attack | Hidden tests passed | Score |
|---|---|---|
| Return zeros (hardcode) | 0 / 24 | 0.0 |
| Monkeypatch `torch.allclose` | 0 / 24 | 0.0 |
| Shape-specific (only T=64, D=64) | 10 / 24 | 0.0 |

All attacks fail — correctness gate blocks any non-zero score.

---

## Assessment

### 1. Environment Description

This environment asks an agent to implement a correctness-preserving sparse
Mixture-of-Experts (MoE) forward operator in PyTorch. The task is inspired by
modern LLM inference and training systems, where MoE layers route each token to
only a small subset of experts instead of activating all parameters.

The environment is similar in spirit to a SWE-bench-style execution task, but
focused on ML systems rather than general software engineering. The agent is
given a small PyTorch codebase containing a slow but clear reference
implementation of a top-k routed MoE layer. Its goal is to write a faster
`moe_forward()` function that preserves the exact mathematical behavior of
the reference under all routing conditions.

The task is loosely inspired by *ScatterMoE* (Scattered Mixture-of-Experts
Implementation), which studies efficient sparse MoE execution on GPUs by
avoiding padding and unnecessary input copies. This environment does not ask
the agent to reproduce the full paper, but uses the same core problem: exact
sparse expert routing, correctness-preserving execution, and performance
improvement after correctness is established.

The function receives token activations `x [T, D]`, selected expert IDs
`expert_ids [T, K]`, router weights `expert_weights [T, K]`, and expert MLP
weight matrices `w1 [E, H, D]`, `b1 [E, H]`, `w2 [E, D, H]`, `b2 [E, D]`.
For each token, it computes the weighted sum of the selected experts' outputs:

```
hidden        = gelu(w1[e] @ x[t] + b1[e])
expert_output = w2[e] @ hidden + b2[e]
output[t]    += expert_weights[t, k] * expert_output
```

The interesting part is that the routing patterns can be sparse, imbalanced,
and irregular. Some experts may receive no tokens; some may receive almost all
tokens. A correct solution must handle these edge cases and scatter outputs back
to the original token order.

This environment is interesting because it is a realistic AI/ML engineering
task with a fully objective judge. Correctness is a hard gate: the judge
compares the submitted implementation against a deterministic PyTorch oracle on
hidden tensor cases. Performance is only scored after correctness is
established.

Common failure modes that the judge specifically tests include: losing the
original token order after grouping tokens by expert, mishandling tokens routed
to zero experts, using `view()` incorrectly on non-contiguous tensors, applying
router weights with wrong broadcasting semantics, and failing to accumulate
correctly when a token is routed to the same expert multiple times.

---

### 2. Tools, Packages, Environment Setup, and Data

The LLM has access to a command line inside a Linux VM where it can read,
write, and run files. The environment is designed to run on any CUDA-capable
NVIDIA GPU (RTX 3090, RTX 4090, A10, A100, H100) or on CPU for correctness
testing.

The repository uses `uv` for reproducible environment management:

```bash
uv sync
uv run pytest                                           # correctness
uv run python benchmarks/benchmark_moe.py --device cuda  # performance
uv run python judge/judge.py --device cuda              # full judge
```

**Dependencies:** Python 3.10+, PyTorch with CUDA, NumPy, pytest,
pytest-timeout. Optional: Triton for kernel implementations.

**Data:** No external dataset is required. The judge generates synthetic MoE
inputs at runtime using hidden seeds, shapes, dtypes, and routing
distributions. This makes the environment easy to reproduce and hard to game,
because the hidden judge can generate unseen tensor shapes, dtypes, seeds, and
routing patterns.

**Vast.ai support:** `bash scripts/setup_vast.sh` installs `uv`, runs
`uv sync`, and verifies PyTorch CUDA access on a fresh GPU instance.

---

### 3. Prompt for the Environment

> Shown verbatim to the agent. Also in `prompts/task.md`.

Your task is to implement the sparse MoE forward pass in:

```
/workspace/solution/solution.py
```

Implement the function:

```python
def moe_forward(
    x,              # [T, D]
    expert_ids,     # [T, K]  LongTensor
    expert_weights, # [T, K]
    w1,             # [E, H, D]
    b1,             # [E, H]
    w2,             # [E, D, H]
    b2,             # [E, D]
):
    ...
```

For each token `t` and each of its `K` selected experts `k`:

```
e             = expert_ids[t, k]
hidden        = gelu(w1[e] @ x[t] + b1[e])        # [H]
expert_output = w2[e] @ hidden + b2[e]             # [D]
output[t]    += expert_weights[t, k] * expert_output
```

**Requirements:**
1. Match the reference oracle numerically on all hidden correctness tests.
2. You may use PyTorch, `torch.compile`, Triton, or custom CUDA.
3. Correctness is a hard gate — an incorrect solution scores **0** regardless
   of speed.
4. Do **not** modify `src/moe_env/reference.py`, `tests/`, `benchmarks/`, or
   `judge/`.
5. Handle edge cases: empty experts, imbalanced routing, top-1 and top-2,
   repeated expert IDs, non-contiguous tensors, single-token batches.
6. Preserve the original token order after any grouping or batching.

```bash
uv run pytest                                           # public tests
uv run python benchmarks/benchmark_moe.py --device cuda  # benchmark
```

**Scoring:**
```
score = 0                                      if any test fails
score = 0.7 + 0.3 * min(speedup, 3.0) / 3.0   otherwise
```

where `speedup = reference_median_ms / solution_median_ms` on the same GPU.

---

### 4. Judge Design

The judge (`judge/judge.py`) runs five steps in order:

**Step 0 — Tamper check.**
SHA-256 checksums of seven protected files (`reference.py`, `types.py`,
`utils.py`, `__init__.py`, both test files, `benchmark_moe.py`) are verified
against a stored baseline (`judge/checksums.json`). Any modification aborts
with score 0 before any test runs.

**Step 1 — Signature check.**
`moe_forward` must exist in `solution.py` with the exact parameter list
`(x, expert_ids, expert_weights, w1, b1, w2, b2)`.

**Step 2 — Public tests (subprocess).**
All tests in `tests/` run via `pytest` in a fresh subprocess so that any
monkeypatching inside `solution.py` cannot affect the test runner.

**Step 3 — Hidden correctness tests (subprocess-isolated).**
`solution.py` is copied to a temporary directory. `judge/hidden_runner.py`
is invoked as a fresh Python process pointed at that copy. Hidden cases
include 17+ unseen seeds / shapes / dtypes / routing distributions:
- Seeds 100, 200, 300 (never shown to agent)
- Shapes: T=1, T=7, T=33, T=4096; D=16, D=512; H=512; E=32
- Dtypes: float32, float16, bfloat16
- Routing: uniform, skewed, sparse, repeated expert IDs

Running in a fresh subprocess ensures monkeypatching of `torch.allclose` or
any other PyTorch function cannot reach the judge's comparison logic.
Additionally, the correctness comparison in `allclose_with_report()` is
implemented directly via raw tensor arithmetic (`abs_err <= atol + rtol * |b|`)
rather than calling `torch.allclose`, so even a patched `torch.allclose` has
no effect.

**Step 4 — Performance benchmark.**
Five hidden benchmark configs (T up to 4096, E up to 16) are timed with
warmup and CUDA synchronization. Speedup is measured relative to the
reference implementation on the same machine, so hardware differences do not
affect the score.

**Scoring:**
```python
if not correctness_passed:
    score = 0.0
else:
    avg_speedup  = median(reference_ms) / median(solution_ms)
    normalized   = min(avg_speedup, 3.0) / 3.0
    score        = 0.7 + 0.3 * normalized
```

The judge also reports: per-case max absolute error, max relative error, dtype,
shape, and full benchmark timings in a JSON report (`--output`).

---

### 5. Reward Hacking and Reward Denial

**Reward hacking risks and mitigations:**

| Attack | Mitigation |
|---|---|
| Hardcode outputs for visible test seeds | Hidden seeds generated at eval time; agent never sees them |
| Read `judge/hidden_tests.py` | In production, judge lives outside agent-writable workspace; reading seeds ≠ correct outputs |
| Monkeypatch `torch.allclose` | Hidden tests run in a fresh subprocess; `allclose_with_report` uses raw tensor arithmetic, not `torch.allclose` |
| Modify `reference.py` or test files | SHA-256 tamper check aborts evaluation before any test runs |
| Shape-specific code (only public shapes) | Hidden shapes include T=1, T=7, T=33, T=4096, D=16, D=512 — all unseen |
| Optimize only for benchmark shapes | Hidden benchmark uses different T/E/D than public benchmark |
| Return precomputed zeros / identity | Oracle comparison catches it: max absolute error far exceeds tolerance |

**Demonstrated results** (`hacks/hack_demo.py`):

| Attack | Hidden tests passed | Score |
|---|---|---|
| Return zeros | 0 / 24 | 0.0 |
| Monkeypatch `torch.allclose` | 0 / 24 | 0.0 |
| Shape-specific (T=64, D=64 only) | 10 / 24 | 0.0 (correctness gate) |
| Tamper `reference.py` (then delete checksums) | — | TamperError before step 1 |

**Reward denial risks:**

The main reward denial risk is floating-point tolerance. A correct
implementation may produce slightly different results due to accumulation order.
Mitigations: (a) dtype-aware tolerances (`atol=1e-5, rtol=1e-4` for FP32;
`atol=1e-2, rtol=1e-2` for FP16/BF16); (b) the correctness gate is kept in
FP32 first; (c) the judge reports exact error values so failures are
interpretable.

For performance, reward denial from hardware variance is avoided by measuring
speedup relative to the reference on the same machine, not against an absolute
threshold.

---

### 6. Why I Chose This Environment

I chose this environment because it pairs a realistic ML systems problem with a
fully objective judge — no LLM-as-judge, no human preference model.

Sparse MoE layers are important in modern LLM systems: they increase model
capacity while only activating a subset of parameters per token. Implementing
them efficiently requires understanding routing, batching, tensor shapes,
memory layout, and numerical correctness simultaneously. This is the kind of
task that separates a surface-level code generator from an agent that actually
reasons about the computation.

The environment connects to a real research paper. ScatterMoE showed that
sparse MoE implementations can be meaningfully improved by avoiding padding and
unnecessary input copies. This environment turns that systems-optimization
problem into a judgeable RL task: the agent does not need to reproduce the full
paper, but it has to solve the same core problem of sparse expert execution
without changing model outputs.

I also chose it because the correctness gate is genuinely hard to fake. The
most common reward hacking strategies — hardcoding visible outputs, patching
the judge, writing shape-specific branches — all fail against the combination
of subprocess isolation, tamper checking, and unseen hidden inputs. The reward
is close to the true task objective: a solution that scores above 0.7 has
actually solved the problem.

Optional related environment: a self-speculative decoding task, where an agent
must preserve the exact greedy outputs of a target model while reducing
expensive target forward passes. It would follow the same design principle —
correctness hard gate, efficiency secondary score, verifier hidden or
instrumented to prevent tampering.

---

### 7. Anything Else

**Why a correctness gate matters for RL.**
Environments with continuous rewards but no hard gate are vulnerable to
Goodharting: an agent that learns to look correct without being correct. The
0.7 base score is only reachable if the agent passes all 24 hidden correctness
tests. There is no gradient signal pointing toward reward without first solving
the correctness problem.

**Why this task is hard for strong LLM agents.**
A naive model may implement the obvious nested-loop version — which is correct
but scores near 0.7 on performance. An optimized implementation requires
reasoning about scatter/gather semantics, tensor contiguity, broadcast rules,
and expert-wise batching. The hidden edge cases (repeated expert IDs,
non-contiguous inputs, zero-token experts, FP16 accumulation) are specifically
chosen to match the mistakes that strong LLM agents are most likely to make.

**On the GPU benchmark gap.**
All numbers in this document were produced on a CPU (macOS development
machine). On a GPU, the reference's nested Python loop is even slower relative
to a batched expert-wise matmul, so speedups above 3× are very achievable.
The scoring formula caps at 3× so that rare GPU-specific tricks do not dominate
the score, and so the score is comparable across different GPU models.

**Reproducibility.**
The full environment is reproducible from a single `uv sync`. All random seeds,
model shapes, routing distributions, and tolerance thresholds are deterministic.
A clean evaluation always produces the same hidden test results.

---

## Directory Structure

```
.
├── src/moe_env/
│   ├── reference.py      # oracle — do NOT modify
│   ├── types.py          # MoEConfig, MoEInputs
│   └── utils.py          # input generators, tolerance helpers
│
├── solution/
│   └── solution.py       # ← agent edits this
│
├── tests/
│   ├── test_public_correctness.py
│   └── test_public_edge_cases.py
│
├── benchmarks/
│   └── benchmark_moe.py
│
├── judge/                # hidden in production
│   ├── judge.py          # evaluation pipeline
│   ├── hidden_tests.py   # hidden seeds / shapes / dtypes
│   ├── hidden_runner.py  # subprocess-isolated runner
│   ├── tamper_check.py   # SHA-256 integrity check
│   └── checksums.json    # baseline digests
│
├── hacks/
│   └── hack_demo.py      # 5 attack demos + why they fail
│
├── prompts/
│   └── task.md           # agent-facing task prompt
│
├── results/              # JSON outputs from experiments
├── figures/              # generated plots
│
└── scripts/
    ├── setup_vast.sh     # Vast.ai bootstrap
    ├── run_tests.sh
    ├── run_benchmark.sh
    └── run_judge.sh
```

---

## Reproducing All Results

```bash
# 1. Install
uv sync

# 2. Run judge + hack comparison + scaling experiments
uv run python experiments_eval.py --device cpu --scaling

# 3. Generate all figures
uv run python experiments_plot.py \
  --judge results/judge_report.json \
  --benchmark results/benchmark.json \
  --hack results/hack_comparison.json \
  --scaling results/scaling.json \
  --out figures/

# 4. Run hack demo
uv run python hacks/hack_demo.py

# 5. Full judge with tamper check
bash scripts/run_judge.sh cpu
```

On a GPU instance (Vast.ai or local):

```bash
bash scripts/setup_vast.sh
bash scripts/run_judge.sh cuda
uv run python experiments_eval.py --device cuda --scaling
uv run python experiments_plot.py \
  --judge results/judge_report.json \
  --benchmark results/benchmark.json \
  --hack results/hack_comparison.json \
  --scaling results/scaling.json \
  --out figures/
```

---

## Questions

**Q1. Why is correctness a hard gate (score=0) rather than a soft penalty in the scoring formula? What failure mode does this prevent?**

A soft penalty would allow an agent to score partial credit by returning
approximately-correct outputs (e.g. zeros, or a solution that is right for
half the inputs and wrong for the other half). An agent trained under a
soft-penalty reward can Goodhart: it learns to maximize the numerical score
rather than to solve the task. The hard gate ensures there is no gradient
signal toward reward unless the agent has genuinely solved correctness.
Without the gate, an agent could learn "return plausible-looking activations"
as a stable local optimum, even though such a solution is useless for
production inference.

---

**Q2. Why does the judge run `solution.py` in a fresh subprocess rather than importing it directly? What attack does this block?**

If the judge imported `solution.py` in its own process, any monkeypatching
done at module-import time inside `solution.py` would affect the judge's
Python process. For example, an agent could write
`torch.allclose = lambda *a, **kw: True` at the top of `solution.py`, which
would replace the judge's correctness check with a function that always returns
True. Running the hidden tests in a subprocess ensures that the judge's own
Python environment is completely unaffected by any side effects in the agent's
code. The subprocess terminates after returning JSON results, leaving the
judge's process state clean.

---

**Q3. How does the tamper check prevent the agent from weakening the oracle, and what would happen if `judge/checksums.json` were deleted?**

`tamper_check.py` computes the SHA-256 hash of each protected file at judge
startup and compares it against a baseline stored in `checksums.json`. If any
file is modified — even by one byte — the check raises `TamperError` and the
judge aborts with `score=0` before running any tests. This prevents an agent
from replacing `reference.py` with a trivial function (e.g. returning zeros),
which would make its own incorrect `moe_forward` appear correct.

If `checksums.json` were deleted, the tamper check raises `TamperError`
immediately with the message "Baseline checksums not found". The judge aborts
with `score=0`. It does not silently regenerate the baseline, because
regenerating from a tampered workspace would defeat the entire purpose of the
check. Regeneration must be done explicitly by a trusted party via
`python -m judge.tamper_check --regenerate`.

---

**Q4. Why does `allclose_with_report()` avoid `torch.allclose` and use raw tensor arithmetic instead? Under what condition would this matter?**

`torch.allclose` is a Python-level function that can be replaced at runtime:
`torch.allclose = lambda a, b, **kw: True`. If the judge called `torch.allclose`
directly, this substitution — made in the judge's own process before the
subprocess isolation was added — would cause all correctness checks to pass
regardless of the actual values. By implementing the comparison as
`(abs_err <= atol + rtol * ref.abs()).all()` using raw tensor operations,
the correctness check cannot be intercepted through Python attribute assignment.
The raw arithmetic calls into compiled C++/CUDA code that cannot be replaced
by the agent.

---

**Q5. What is the reward denial risk from floating-point accumulation order, and how are the tolerances calibrated to avoid it?**

A correct batched implementation (grouping tokens by expert and calling `mm`)
accumulates values in a different order than the reference's sequential token
loop. IEEE 754 floating-point arithmetic is not associative, so the two
implementations can produce slightly different results even when both are
mathematically correct. If tolerances were set too tight (e.g. `atol=0`),
a correct solution would fail the correctness gate — reward denial.

Tolerances are calibrated empirically: FP32 uses `atol=1e-5, rtol=1e-4`, which
is consistent with PyTorch's own defaults for `allclose`. FP16 and BF16 use
`atol=1e-2, rtol=1e-2` to account for the reduced mantissa precision of those
formats. The judge also reports the exact max absolute and relative error for
every test case, so tolerance failures are immediately diagnosable rather than
opaque.

---

**Q6. Why does skewed routing produce a higher speedup than uniform routing, even at the same total token count?**

In skewed routing, one expert receives the majority of tokens (e.g. ~80% of
T=512 tokens → ~410 tokens for expert 0, ~15 tokens for each remaining expert).
The batched solution builds a single `mm(x_expert[410, D], w1[e].T)` call for
the dominant expert — a large, GPU-efficient matmul. The remaining experts
either get small batches or are skipped entirely via the `if not mask.any():
continue` guard. The reference loops over all T tokens and all K experts
regardless of routing balance, doing K scalar dot-products per token with no
opportunity to exploit the concentrated load.

Uniform routing distributes tokens evenly (~64 per expert at T=512, E=8),
giving each expert a medium-sized batch. This is still faster than the
reference's per-token loop, but the per-expert batch is smaller and the GPU
efficiency of each individual `mm` call is lower than the single large call
that skewed routing enables.

---

**Q7. How would you extend this environment to evaluate an agent's ability to write a Triton kernel rather than a PyTorch solution?**

The existing judge already accepts any `moe_forward(x, expert_ids, ...)` that
returns the correct tensor — it does not inspect how the computation is done.
Extending to Triton requires three changes:

1. **Dependency**: add `triton` to `pyproject.toml` so the agent's environment
   includes it. Triton is CUDA-only, so the judge must enforce
   `--device cuda` for the Triton track.
2. **Prompt**: update `prompts/task.md` to tell the agent it should write a
   Triton kernel for the compute-intensive parts. Add an example import
   (`import triton`, `import triton.language as tl`) and a skeleton
   `@triton.jit` function to lower the entry barrier.
3. **Scoring sensitivity**: raise the speedup cap from 3× to a higher value
   (e.g. 10×), or use a steeper curve, because a well-written Triton kernel
   can achieve 5–20× over the PyTorch reference on modern NVIDIA GPUs.
   The scoring formula would become `0.3 * min(speedup, 10.0) / 10.0`.

No changes to the correctness judge or the oracle are needed — the same hidden
test cases, subprocess isolation, tamper check, and `allclose_with_report`
logic apply equally to Triton-based solutions.
