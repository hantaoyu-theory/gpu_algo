#!/usr/bin/env python3
"""
Targeted LSH timing for the three summary cases:
  - d=8 Gaussian
  - d=128 Gaussian
  - Fashion-MNIST-784 HDF5
"""

from __future__ import annotations

import json

import cupy as cp
import numpy as np

from explore import bench_lsh_fast
from explore_annb import load_ann_benchmarks_hdf5
from lsh import LSHParams


def run() -> dict:
    k = 10
    seed = 42
    rows = []

    # d=8 Gaussian, use the high-recall config from prior A100 runs.
    np.random.seed(seed)
    X8 = np.random.randn(500_000, 8).astype(np.float32)
    Q8 = np.random.randn(3000, 8).astype(np.float32)
    p8 = LSHParams(n_tables=100, n_projections=3, bucket_width=3.0, max_cands_per_table=2000, seed=seed)
    ids, ms, build_ms = bench_lsh_fast(cp.asarray(X8), cp.asarray(Q8), k, p8, oversample=4, reps=10)
    rows.append({
        "case": "d8_gaussian",
        "n": 3000,
        "d": 8,
        "m": 500000,
        "algorithm": "LSH L100 K3 w3 m2k",
        "query_ms": float(ms),
        "build_ms": float(build_ms),
        "params": {"L": 100, "K": 3, "w": 3.0, "max_cands": 2000},
    })

    # d=128 Gaussian, use the stronger of the high-d configs from prior runs.
    np.random.seed(seed)
    X128 = np.random.randn(250_000, 128).astype(np.float32)
    Q128 = np.random.randn(2000, 128).astype(np.float32)
    p128 = LSHParams(n_tables=64, n_projections=3, bucket_width=0.0, max_cands_per_table=800, seed=seed)
    ids, ms, build_ms = bench_lsh_fast(cp.asarray(X128), cp.asarray(Q128), k, p128, oversample=4, reps=10)
    rows.append({
        "case": "d128_gaussian",
        "n": 2000,
        "d": 128,
        "m": 250000,
        "algorithm": "LSH L64 K3 w0 m800",
        "query_ms": float(ms),
        "build_ms": float(build_ms),
        "params": {"L": 64, "K": 3, "w": 0.0, "max_cands": 800},
    })

    # Fashion-MNIST-784, choose the same high-d style config.
    Xf, Qf, _gt, _note = load_ann_benchmarks_hdf5(
        "/workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5", k, max_test=300
    )
    pf = LSHParams(n_tables=64, n_projections=3, bucket_width=0.0, max_cands_per_table=800, seed=seed)
    ids, ms, build_ms = bench_lsh_fast(cp.asarray(Xf), cp.asarray(Qf), k, pf, oversample=4, reps=10)
    rows.append({
        "case": "fashion784",
        "n": 300,
        "d": 784,
        "m": 60000,
        "algorithm": "LSH L64 K3 w0 m800",
        "query_ms": float(ms),
        "build_ms": float(build_ms),
        "params": {"L": 64, "K": 3, "w": 0.0, "max_cands": 800},
    })

    return {"rows": rows}


if __name__ == "__main__":
    out = run()
    with open("lsh_case_benchmarks.json", "w", encoding="ascii") as f:
        json.dump(out, f, indent=2)
    print("Saved lsh_case_benchmarks.json")
