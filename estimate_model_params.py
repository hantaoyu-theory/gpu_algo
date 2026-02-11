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

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


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


def _bench_device_copy(
    dev: torch.device, numel: int, dtype: torch.dtype, runs: int, warmup: int
) -> float:
    src = torch.randn(numel, device=dev, dtype=dtype)
    dst = torch.empty_like(src)

    def _op() -> None:
        dst.copy_(src)

    avg_s = _time_cuda_op(dev, _op, runs, warmup)
    bytes_moved = float(numel) * src.element_size()
    return bytes_moved / avg_s


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


if triton is not None:
    @triton.jit
    def _fma_kernel(x_ptr, n_elements, ops: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        a = 1.0003
        b = 0.9997
        for _ in range(ops):
            x = x * a + b
        tl.store(x_ptr + offs, x, mask=mask)


def _bench_compute_only_triton(
    dev: torch.device,
    numel: int,
    dtype: torch.dtype,
    runs: int,
    warmup: int,
    ops: int,
) -> float | None:
    if triton is None:
        return None
    x = torch.randn(numel, device=dev, dtype=dtype)
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK"]),)

    def _op() -> None:
        _fma_kernel[grid](x, numel, ops=ops, BLOCK=256)

    avg_s = _time_cuda_op(dev, _op, runs, warmup)
    flops = float(numel) * float(ops) * 2.0
    return flops / avg_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate GPU model parameters.")
    parser.add_argument("--gpus", type=_parse_gpu_list, default="0,1,2,3")
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="float16")
    parser.add_argument("--matmul-n", type=int, default=8192, help="Matmul size (n x n)")
    parser.add_argument(
        "--vec-numel", type=int, default=1 << 26, help="Vector size for elementwise ops"
    )
    parser.add_argument(
        "--compute-ops",
        type=int,
        default=1024,
        help="FMA ops per element for compute-only benchmark (Triton)",
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
    B_d2d = _bench_device_copy(dev0, args.vec_numel, dtype, args.runs, args.warmup)
    x_compute_only = _bench_compute_only_triton(
        dev0, args.vec_numel, dtype, args.runs, args.warmup, args.compute_ops
    )

    bytes_elem = float(args.vec_numel) * torch.tensor([], dtype=dtype).element_size() * 3.0
    mem_time_e = bytes_elem / B_d2d
    compute_time_e = max(0.0, (float(args.vec_numel) / x_e) - mem_time_e)
    x_e_compute = float(args.vec_numel) / compute_time_e if compute_time_e > 0 else float("inf")

    bytes_addcmul = float(args.vec_numel) * torch.tensor([], dtype=dtype).element_size() * 3.0
    mem_time_x = bytes_addcmul / B_d2d
    compute_time_x = max(0.0, (float(args.vec_numel) * 3.0 / x) - mem_time_x)
    x_compute = float(args.vec_numel) * 3.0 / compute_time_x if compute_time_x > 0 else float("inf")

    print(f"x_m (matmul)  = {x_m / 1e12:.3f} TFLOP/s")
    print(f"x_e (elem add)= {x_e / 1e9:.3f} GFLOP/s")
    print(f"x (addcmul)   = {x / 1e9:.3f} GFLOP/s")
    print(f"B (HBM eff)   = {B_eff / 1e9:.3f} GB/s")
    print(f"B (D2D copy)  = {B_d2d / 1e9:.3f} GB/s")
    if x_e_compute != float("inf"):
        print(f"x_e (compute est) = {x_e_compute / 1e9:.3f} GFLOP/s")
    else:
        print("x_e (compute est) = inf (memory-dominated)")
    if x_compute != float("inf"):
        print(f"x (compute est)   = {x_compute / 1e9:.3f} GFLOP/s")
    else:
        print("x (compute est)   = inf (memory-dominated)")
    if x_compute_only is not None:
        print(f"x (compute-only, Triton) = {x_compute_only / 1e9:.3f} GFLOP/s")
    else:
        print("x (compute-only, Triton) = unavailable (install triton)")
    print()

    n = args.matmul_n
    elem_size = torch.tensor([], dtype=dtype).element_size()
    bytes_moved = 3.0 * n * n * elem_size
    flops = 2.0 * n * n * n
    t_compute = flops / x_m
    t_transfer = bytes_moved / B_d2d
    print("=== Matmul Time Estimate (A,B,C once) ===")
    print(f"n = {n}, bytes = {bytes_moved / 1e6:.3f} MB")
    print(f"compute time = {t_compute * 1e3:.3f} ms")
    print(f"transfer time = {t_transfer * 1e3:.3f} ms (using D2D bandwidth)")
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
