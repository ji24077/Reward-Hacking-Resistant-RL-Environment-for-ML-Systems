"""
Hidden test cases — NOT visible to the agent.
Generated from seeds/shapes/dtypes unknown to the agent.
In production this file lives outside the agent-writable workspace.
"""

import torch
from typing import NamedTuple

# Each entry: (name, T, E, D, H, K, seed, routing, dtype, make_contiguous)
HIDDEN_CASES = [
    # ---- unseen seeds ---------------------------------------------------
    ("hidden_seed_100",    64,  8,  64, 128, 2, 100, "uniform",  "float32", True),
    ("hidden_seed_200",   128,  8,  64, 128, 2, 200, "uniform",  "float32", True),
    ("hidden_seed_300",   256,  8,  64, 128, 2, 300, "uniform",  "float32", True),
    # ---- unseen shapes --------------------------------------------------
    ("hidden_T7",           7,  4,  32,  64, 2,  42, "uniform",  "float32", True),
    ("hidden_T33",         33,  6,  48,  96, 2,  42, "uniform",  "float32", True),
    ("hidden_large_D",     64,  8, 512, 256, 2,  42, "uniform",  "float32", True),
    ("hidden_small_D",     64,  8,  16,  32, 2,  42, "uniform",  "float32", True),
    ("hidden_large_H",     32,  4,  32, 512, 2,  42, "uniform",  "float32", True),
    # ---- routing extremes -----------------------------------------------
    ("hidden_all_same",    64,  8,  64, 128, 1,  55, "skewed",   "float32", True),
    ("hidden_sparse_k1",   64,  8,  64, 128, 1,  66, "sparse",   "float32", True),
    ("hidden_sparse_k2",   64,  8,  64, 128, 2,  77, "sparse",   "float32", True),
    # ---- non-contiguous input -------------------------------------------
    ("hidden_noncontig",   64,  8,  64, 128, 2,  88, "uniform",  "float32", False),
    # ---- top-k variants -------------------------------------------------
    ("hidden_top3",        64,  6,  64, 128, 3,  99, "uniform",  "float32", True),
    # ---- stress ---------------------------------------------------------
    ("hidden_stress",    4096, 16, 128, 256, 2, 111, "uniform",  "float32", True),
    ("hidden_imbalanced",4096,  8, 128, 256, 2, 222, "skewed",   "float32", True),
    # ---- single token ---------------------------------------------------
    ("hidden_T1",           1,  8,  64, 128, 2, 333, "uniform",  "float32", True),
    # ---- many experts, top-1 --------------------------------------------
    ("hidden_E32_k1",      64, 32,  64, 128, 1, 444, "uniform",  "float32", True),
    # ---- FP16 -----------------------------------------------------------
    ("hidden_fp16_small",  64,  8,  64, 128, 2, 500, "uniform",  "float16", True),
    ("hidden_fp16_skewed", 128, 8,  64, 128, 2, 501, "skewed",   "float16", True),
    ("hidden_fp16_sparse", 64,  8,  64, 128, 1, 502, "sparse",   "float16", True),
    # ---- BF16 -----------------------------------------------------------
    ("hidden_bf16_small",  64,  8,  64, 128, 2, 600, "uniform",  "bfloat16", True),
    ("hidden_bf16_medium", 256, 8, 128, 256, 2, 601, "skewed",   "bfloat16", True),
    # ---- repeated expert IDs --------------------------------------------
    ("hidden_repeated_k2", 64,  4,  32,  64, 2, 700, "repeated", "float32", True),
    ("hidden_repeated_k2_fp16", 32, 4, 32, 64, 2, 701, "repeated", "float16", True),
]
