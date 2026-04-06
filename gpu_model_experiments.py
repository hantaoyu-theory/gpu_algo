#!/usr/bin/env python3
"""
Benchmark non-NN GPU workloads to validate the movement/orchestration/reuse model.

Workloads:
  - copy      : regular streaming memory traffic
  - saxpy     : elementwise map, slightly more arithmetic
  - reduce    : tree-style aggregation
  - compact   : filter / stream compaction
  - sort      : multi-pass reordering
  - gemm      : dense compute with reuse
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import cupy as cp
import numpy as np


def _tms(fn, warmup: int = 5, reps: int = 20) -> float:
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()
    ev0 = cp.cuda.Event()
    ev1 = cp.cuda.Event()
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record()
    ev1.synchronize()
    return float(cp.cuda.get_elapsed_time(ev0, ev1)) / reps


@dataclass
class BenchRow:
    workload: str
    size: int
    ms: float
    est_bytes: float | None
    est_flops: float | None
    eff_bw_gbs: float | None
    eff_gflops: float | None
    note: str


def bench_copy(n: int) -> BenchRow:
    x = cp.random.random(n, dtype=cp.float32)
    y = cp.empty_like(x)

    def fn():
        y[:] = x

    ms = _tms(fn)
    est_bytes = float(n * 4 * 2)
    eff_bw = est_bytes / (ms * 1e-3) / 1e9
    return BenchRow("copy", n, ms, est_bytes, 0.0, eff_bw, 0.0, "streaming read+write")


def bench_saxpy(n: int, a: float = 1.1) -> BenchRow:
    x = cp.random.random(n, dtype=cp.float32)
    y = cp.random.random(n, dtype=cp.float32)

    def fn():
        y[:] = a * x + y

    ms = _tms(fn)
    est_bytes = float(n * 4 * 3)
    est_flops = float(n * 2)
    eff_bw = est_bytes / (ms * 1e-3) / 1e9
    eff_gflops = est_flops / (ms * 1e-3) / 1e9
    return BenchRow("saxpy", n, ms, est_bytes, est_flops, eff_bw, eff_gflops, "2 flops/elem")


def bench_reduce(n: int) -> BenchRow:
    x = cp.random.random(n, dtype=cp.float32)

    def fn():
        cp.sum(x)

    ms = _tms(fn)
    est_bytes = float(n * 4)
    eff_bw = est_bytes / (ms * 1e-3) / 1e9
    return BenchRow("reduce", n, ms, est_bytes, None, eff_bw, None, "tree reduction")


def bench_compact(n: int) -> BenchRow:
    x = cp.random.standard_normal(n, dtype=cp.float32)

    def fn():
        x[x > 0]

    ms = _tms(fn)
    est_bytes = float(n * 4 * 2)
    eff_bw = est_bytes / (ms * 1e-3) / 1e9
    return BenchRow("compact", n, ms, est_bytes, None, eff_bw, None, "mask + gather")


def bench_sort(n: int) -> BenchRow:
    x = cp.random.random(n, dtype=cp.float32)

    def fn():
        cp.sort(x)

    ms = _tms(fn, warmup=3, reps=10)
    return BenchRow("sort", n, ms, None, None, None, None, "comparison/reordering heavy")


def bench_gemm(n: int, k: int, m: int) -> BenchRow:
    a = cp.random.random((n, k), dtype=cp.float32).astype(cp.float16)
    b = cp.random.random((k, m), dtype=cp.float32).astype(cp.float16)

    def fn():
        a @ b

    ms = _tms(fn, warmup=3, reps=10)
    est_flops = float(2 * n * k * m)
    est_bytes = float((n * k + k * m + n * m) * 2)
    eff_bw = est_bytes / (ms * 1e-3) / 1e9
    eff_gflops = est_flops / (ms * 1e-3) / 1e9
    return BenchRow("gemm", n * k * m, ms, est_bytes, est_flops, eff_bw, eff_gflops, f"{n}x{k} @ {k}x{m}")


def run_suite(vec_sizes: list[int], gemm_sizes: list[tuple[int, int, int]]) -> dict:
    rows: list[BenchRow] = []
    for n in vec_sizes:
        print(f"\n=== vector size {n} ===")
        for fn in [bench_copy, bench_saxpy, bench_reduce, bench_compact, bench_sort]:
            row = fn(n)
            rows.append(row)
            bw = "n/a" if row.eff_bw_gbs is None else f"{row.eff_bw_gbs:.1f}"
            gf = "n/a" if row.eff_gflops is None else f"{row.eff_gflops:.1f}"
            print(f"  {row.workload:<8} {row.ms:8.3f} ms  BW={bw:>8} GB/s  GF={gf:>8}")

    print("\n=== gemm ===")
    for n, k, m in gemm_sizes:
        row = bench_gemm(n, k, m)
        rows.append(row)
        print(f"  gemm {n}x{k}x{m:<6} {row.ms:8.3f} ms  BW={row.eff_bw_gbs:.1f} GB/s  GF={row.eff_gflops:.1f}")

    return {
        "rows": [row.__dict__ for row in rows],
        "meta": {
            "vec_sizes": vec_sizes,
            "gemm_sizes": gemm_sizes,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-sizes", type=int, nargs="+", default=[1_000_000, 5_000_000, 20_000_000])
    ap.add_argument("--gemm-sizes", type=int, nargs="+", default=[1024, 1024, 1024, 2048, 2048, 2048])
    ap.add_argument("--out-json", type=str, default="gpu_model_experiments.json")
    args = ap.parse_args()

    if len(args.gemm_sizes) % 3 != 0:
        raise ValueError("--gemm-sizes must be triples: n k m ...")
    gemm_sizes = [tuple(args.gemm_sizes[i:i+3]) for i in range(0, len(args.gemm_sizes), 3)]

    data = run_suite(args.vec_sizes, gemm_sizes)
    with open(args.out_json, "w", encoding="ascii") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {args.out_json}")


if __name__ == "__main__":
    main()
