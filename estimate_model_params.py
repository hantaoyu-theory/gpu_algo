#!/usr/bin/env python3
"""
Estimate simplified GPU model parameters via microbenchmarks.

Outputs:
  - x_m: matmul throughput (TFLOP/s)
  - x_e: elementwise add throughput (GFLOP/s)
  - x: compute-heavy elementwise throughput (GFLOP/s)
  - B: effective HBM bandwidth from elementwise add (GB/s)
  - L: inter-GPU transfer bandwidth (GB/s) for GPU0 -> GPUn
"""
from __future__ import annotations

import argparse
import time
from typing import Iterable

import torch


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _parse_gpu_list(value: str) -> list[int]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise ValueError("GPU list is empty")
    return [int(v) for v in items]


def _sync_all(devices: Iterable[torch.device]) -> None:
    for dev in devices:
        torch.cuda.synchronize(dev)


def _time_cuda_op(dev: torch.device, fn, runs: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def _bench_matmul(dev: torch.device, n: int, dtype: torch.dtype, runs: int, warmup: int) -> float:
    A = torch.randn(n, n, device=dev, dtype=dtype)
    B = torch.randn(n, n, device=dev, dtype=dtype)

    def _op() -> None:
        _ = A @ B

    avg_s = _time_cuda_op(dev, _op, runs, warmup)
    flops = 2.0 * n * n * n
    return flops / avg_s


def _bench_elementwise_add(
    dev: torch.device, numel: int, dtype: torch.dtype, runs: int, warmup: int
) -> tuple[float, float]:
    A = torch.randn(numel, device=dev, dtype=dtype)
    B = torch.randn(numel, device=dev, dtype=dtype)

    def _op() -> None:
        _ = A + B

    avg_s = _time_cuda_op(dev, _op, runs, warmup)
    flops = float(numel)
    bytes_moved = float(numel) * A.element_size() * 3.0
    x_e = flops / avg_s
    B_eff = bytes_moved / avg_s
    return x_e, B_eff


def _bench_addcmul(
    dev: torch.device, numel: int, dtype: torch.dtype, runs: int, warmup: int
) -> float:
    A = torch.randn(numel, device=dev, dtype=dtype)
    B = torch.randn(numel, device=dev, dtype=dtype)
    C = torch.randn(numel, device=dev, dtype=dtype)

    def _op() -> None:
        _ = torch.addcmul(A, B, C, value=1.0)

    avg_s = _time_cuda_op(dev, _op, runs, warmup)
    flops = float(numel) * 3.0
    return flops / avg_s


def _bench_peer_bandwidth(
    src: torch.device,
    dst: torch.device,
    numel: int,
    dtype: torch.dtype,
    runs: int,
    warmup: int,
) -> float:
    x = torch.randn(numel, device=src, dtype=dtype)

    def _op() -> None:
        _ = x.to(dst)

    for _ in range(warmup):
        _op()
    _sync_all([src, dst])

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = x.to(dst)
        _sync_all([src, dst])
        times.append(time.perf_counter() - start)
    avg_s = sum(times) / len(times)
    bytes_moved = float(numel) * x.element_size()
    return bytes_moved / avg_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate GPU model parameters.")
    parser.add_argument("--gpus", type=_parse_gpu_list, default="0,1,2,3")
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="float16")
    parser.add_argument("--matmul-n", type=int, default=8192, help="Matmul size (n x n)")
    parser.add_argument(
        "--vec-numel", type=int, default=1 << 26, help="Vector size for elementwise ops"
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)

    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    devices = [torch.device(f"cuda:{idx}") for idx in args.gpus]
    dtype = DTYPE_MAP[args.dtype]

    dev0 = devices[0]
    print("=== Parameter Estimation ===")
    print(f"GPUs = {', '.join(str(d) for d in args.gpus)}")
    print(f"dtype = {args.dtype}")
    print(f"matmul_n = {args.matmul_n}, vec_numel = {args.vec_numel}")
    print(f"runs = {args.runs}, warmup = {args.warmup}")
    print()

    x_m = _bench_matmul(dev0, args.matmul_n, dtype, args.runs, args.warmup)
    x_e, B_eff = _bench_elementwise_add(dev0, args.vec_numel, dtype, args.runs, args.warmup)
    x = _bench_addcmul(dev0, args.vec_numel, dtype, args.runs, args.warmup)

    print(f"x_m (matmul)  = {x_m / 1e12:.3f} TFLOP/s")
    print(f"x_e (elem add)= {x_e / 1e9:.3f} GFLOP/s")
    print(f"x (addcmul)   = {x / 1e9:.3f} GFLOP/s")
    print(f"B (HBM eff)   = {B_eff / 1e9:.3f} GB/s")
    print()

    if len(devices) > 1:
        print("Inter-GPU bandwidth (GPU0 -> GPUn):")
        for dev in devices[1:]:
            bw = _bench_peer_bandwidth(
                devices[0], dev, args.vec_numel, dtype, args.runs, args.warmup
            )
            print(f"  L(0->{dev.index}) = {bw / 1e9:.3f} GB/s")


if __name__ == "__main__":
    main()
