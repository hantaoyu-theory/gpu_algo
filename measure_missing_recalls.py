#!/usr/bin/env python3
"""
Compute recall for the summary rows that were previously measured in time-only mode.
"""

from __future__ import annotations

import json

import cupy as cp
import numpy as np

from explore import (
    IVFIndex,
    IVFIndexINT8,
    bench_lsh_fast,
    ground_truth,
    recall_at_k,
    run_bf_fp16,
    run_bf_fp32,
    run_bf_gemm,
    run_bf_int8,
)
from explore_annb import load_ann_benchmarks_hdf5
from lsh import LSHParams


def measure_d128() -> list[dict]:
    seed = 42
    m = 250_000
    d = 128
    n = 2000
    k = 10
    np.random.seed(seed)
    X_np = np.random.randn(m, d).astype(np.float32)
    Q_np = np.random.randn(n, d).astype(np.float32)

    X_gpu = cp.asarray(X_np)
    Q_gpu = cp.asarray(Q_np)
    X_fp16 = X_gpu.astype(cp.float16)

    gt = ground_truth(Q_gpu, X_gpu, k)

    ivf1 = IVFIndex(64, 16)
    ivf1.build(X_np, d)
    ivf8 = IVFIndexINT8(128, 8)
    ivf8.build(X_np, d)

    lsh_p = LSHParams(n_tables=64, n_projections=3, bucket_width=0.0,
                      max_cands_per_table=800, seed=seed)

    rows = []
    for name, fn in [
        ("BF-FP32", lambda: run_bf_fp32(Q_gpu, X_gpu, k)),
        ("BF-FP16", lambda: run_bf_fp16(Q_gpu, X_fp16, k)),
        ("BF-INT8", lambda: run_bf_int8(Q_gpu, X_gpu, k)),
        ("BF-GEMM", lambda: run_bf_gemm(Q_gpu, X_gpu, k)),
        ("IVF1(C=64,p=16)", lambda: ivf1.search(Q_gpu, k)),
        ("IVF-INT8(C=128,p=8)", lambda: ivf8.search(Q_gpu, k)),
    ]:
        ids, _ms = fn()
        rows.append({
            "case": "d128_gaussian",
            "algorithm": name,
            "n": n,
            "recall": float(recall_at_k(ids, gt, k)),
        })

    ids, _ms, _bms = bench_lsh_fast(X_gpu, Q_gpu, k, lsh_p, oversample=4, reps=5)
    rows.append({
        "case": "d128_gaussian",
        "algorithm": "LSH L64 K3 w0 m800",
        "n": n,
        "recall": float(recall_at_k(ids, gt, k)),
    })
    return rows


def measure_fashion() -> list[dict]:
    seed = 42
    n = 300
    k = 10
    X_np, Q_np, gt, _mode = load_ann_benchmarks_hdf5(
        "/workspace/gpu_algo/fashion-mnist-784-euclidean.hdf5", k, max_test=n
    )

    X_gpu = cp.asarray(X_np)
    Q_gpu = cp.asarray(Q_np)
    X_fp16 = X_gpu.astype(cp.float16)

    ivf1 = IVFIndex(64, 4)
    ivf1.build(X_np, X_np.shape[1])
    ivf8 = IVFIndexINT8(64, 16)
    ivf8.build(X_np, X_np.shape[1])

    lsh_p = LSHParams(n_tables=64, n_projections=3, bucket_width=0.0,
                      max_cands_per_table=800, seed=seed)

    rows = []
    for name, fn in [
        ("BF-FP32", lambda: run_bf_fp32(Q_gpu, X_gpu, k)),
        ("BF-FP16", lambda: run_bf_fp16(Q_gpu, X_fp16, k)),
        ("BF-INT8", lambda: run_bf_int8(Q_gpu, X_gpu, k)),
        ("BF-GEMM", lambda: run_bf_gemm(Q_gpu, X_gpu, k)),
        ("IVF1(C=64,p=4)", lambda: ivf1.search(Q_gpu, k)),
        ("IVF-INT8(C=64,p=16)", lambda: ivf8.search(Q_gpu, k)),
    ]:
        ids, _ms = fn()
        rows.append({
            "case": "fashion784",
            "algorithm": name,
            "n": n,
            "recall": float(recall_at_k(ids, gt, k)),
        })

    ids, _ms, _bms = bench_lsh_fast(X_gpu, Q_gpu, k, lsh_p, oversample=4, reps=5)
    rows.append({
        "case": "fashion784",
        "algorithm": "LSH L64 K3 w0 m800",
        "n": n,
        "recall": float(recall_at_k(ids, gt, k)),
    })
    return rows


if __name__ == "__main__":
    out = {
        "rows": measure_d128() + measure_fashion(),
    }
    with open("missing_recalls.json", "w", encoding="ascii") as f:
        json.dump(out, f, indent=2)
    print("Saved missing_recalls.json")
