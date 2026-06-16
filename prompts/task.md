# Task: Sparse MoE Forward Pass

Your task is to implement the sparse MoE forward pass in:

```
/workspace/solution/solution.py
```

## Function Signature

```python
def moe_forward(
    x,              # Tensor [T, D]
    expert_ids,     # LongTensor [T, K]
    expert_weights, # Tensor [T, K]
    w1,             # Tensor [E, H, D]
    b1,             # Tensor [E, H]
    w2,             # Tensor [E, D, H]
    b2,             # Tensor [E, D]
) -> torch.Tensor:  # [T, D]
```

## Semantics

For each token `t` and each of its `K` selected experts `k`:

```
e             = expert_ids[t, k]
hidden        = gelu(w1[e] @ x[t] + b1[e])        # [H]
expert_output = w2[e] @ hidden + b2[e]             # [D]
output[t]    += expert_weights[t, k] * expert_output
```

The final `output` has shape `[T, D]`.

## Requirements

1. **Correctness first.** The output must match the reference oracle numerically
   on all hidden test cases. Correctness is a hard gate — an incorrect solution
   scores **0** regardless of speed.

2. **Allowed tools.** PyTorch operations, `torch.compile`, Triton kernels,
   custom CUDA extensions. All are optional — pure PyTorch is sufficient.

3. **Do not modify** `src/moe_env/reference.py`, `tests/`, `benchmarks/`, or
   `judge/`. The judge will abort if these files have been altered.

4. **Edge cases you must handle:**
   - Experts that receive zero tokens
   - Highly imbalanced routing (one expert gets almost all tokens)
   - Top-1 and Top-2 (and higher) routing
   - Repeated expert IDs within a single token's routing (e.g. `[2, 2]`)
   - Non-contiguous input tensors (`x` may not be contiguous in memory)
   - Single-token batches (`T = 1`)
   - Large batches (`T = 4096`)

5. **Token order.** After any grouping, sorting, or expert-wise batching, the
   outputs must be scattered back to the **original token indices**.

## Running Tests

```bash
# Public correctness + edge case tests
uv run pytest

# Public benchmark (CPU)
uv run python benchmarks/benchmark_moe.py

# Public benchmark (GPU)
uv run python benchmarks/benchmark_moe.py --device cuda
```

## Scoring

```
score = 0                                     if any test fails
score = 0.7 + 0.3 * min(speedup, 3.0) / 3.0  otherwise
```

`speedup` is measured as `reference_median_ms / solution_median_ms` on the
**same GPU** as the reference. Hardware differences do not affect the score.
