#!/usr/bin/env python3
"""
Multi-GPU matrix multiplication benchmarks (1 GPU or 4 GPUs).

Strategies:
  - single: one GPU does A @ B
  - row: split A rows across 4 GPUs, broadcast B
  - col: split B cols across 4 GPUs, broadcast A
  - block2x2: split A rows and B cols into 2x2 blocks (4 GPUs)
"""
from __future__ import annotations

import argparse
import time
from typing import Iterable

import torch

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


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


def _parse_ns_list(value: str) -> list[int]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        raise ValueError("N list is empty")
    return [int(v) for v in items]


def _sync_all(devices: Iterable[torch.device]) -> None:
    for dev in devices:
        torch.cuda.synchronize(dev)


def _split_sizes(n: int, parts: int) -> list[int]:
    base = n // parts
    rem = n % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _empty_timings(devices: list[torch.device]) -> dict[str, float]:
    timings: dict[str, float] = {}
    for i in range(len(devices)):
        timings[f"gpu{i}_kernel_s"] = 0.0
        timings[f"gpu0_to_gpu{i}_s"] = 0.0
        timings[f"gpu{i}_to_gpu0_s"] = 0.0
    timings["assemble_s"] = 0.0
    timings["h2d_s"] = 0.0
    timings["d2h_s"] = 0.0
    return timings


def matmul_single_gpu(
    A: torch.Tensor,
    B: torch.Tensor,
    dev: torch.device,
    use_cuda_events: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    timings = _empty_timings([dev])
    with torch.cuda.device(dev):
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            C = A @ B
            end.record()
            torch.cuda.synchronize(dev)
            timings["gpu0_kernel_s"] = start.elapsed_time(end) / 1000.0
        else:
            t0 = time.perf_counter()
            C = A @ B
            torch.cuda.synchronize(dev)
            timings["gpu0_kernel_s"] = time.perf_counter() - t0
    return C, timings


def matmul_row_split(
    A: torch.Tensor,
    B: torch.Tensor,
    devices: list[torch.device],
    use_cuda_events: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    n = A.shape[0]
    sizes = _split_sizes(n, 4)
    offsets = [0]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)

    timings = _empty_timings(devices)
    xfer_events = {}
    compute_events = {}
    C_parts = [None] * 4

    for i, dev in enumerate(devices):
        if sizes[i] == 0:
            C_parts[i] = None
            continue

        row_start = offsets[i]
        row_end = row_start + sizes[i]
        A_i = A[row_start:row_end, :].contiguous()

        with torch.cuda.device(dev):
            if i == 0:
                A_i_dev = A_i
                B_i_dev = B
            else:
                if use_cuda_events:
                    start_xfer = torch.cuda.Event(enable_timing=True)
                    end_xfer = torch.cuda.Event(enable_timing=True)
                    start_xfer.record()
                    A_i_dev = A_i.to(dev)
                    B_i_dev = B.to(dev)
                    end_xfer.record()
                    xfer_events[i] = (start_xfer, end_xfer)
                else:
                    A_i_dev = A_i.to(dev)
                    B_i_dev = B.to(dev)

            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                C_parts[i] = A_i_dev @ B_i_dev
                end.record()
                compute_events[i] = (start, end)
            else:
                C_parts[i] = A_i_dev @ B_i_dev

    _sync_all(devices)

    if use_cuda_events:
        for i in range(1, 4):
            if i in xfer_events:
                start_xfer, end_xfer = xfer_events[i]
                timings[f"gpu0_to_gpu{i}_s"] = start_xfer.elapsed_time(end_xfer) / 1000.0
        for i in range(4):
            if i in compute_events:
                start, end = compute_events[i]
                timings[f"gpu{i}_kernel_s"] = start.elapsed_time(end) / 1000.0

    for i in range(1, 4):
        if C_parts[i] is None:
            continue
        with torch.cuda.device(devices[0]):
            if use_cuda_events:
                start_xfer = torch.cuda.Event(enable_timing=True)
                end_xfer = torch.cuda.Event(enable_timing=True)
                start_xfer.record()
                C_parts[i] = C_parts[i].to(devices[0])
                end_xfer.record()
                xfer_events[i] = (start_xfer, end_xfer)
            else:
                C_parts[i] = C_parts[i].to(devices[0])
    _sync_all(devices)

    if use_cuda_events:
        for i in range(1, 4):
            if i in xfer_events:
                start_xfer, end_xfer = xfer_events[i]
                timings[f"gpu{i}_to_gpu0_s"] = start_xfer.elapsed_time(end_xfer) / 1000.0

    with torch.cuda.device(devices[0]):
        if use_cuda_events:
            start_asm = torch.cuda.Event(enable_timing=True)
            end_asm = torch.cuda.Event(enable_timing=True)
            start_asm.record()
            C = torch.cat([part for part in C_parts if part is not None], dim=0)
            end_asm.record()
            torch.cuda.synchronize(devices[0])
            timings["assemble_s"] = start_asm.elapsed_time(end_asm) / 1000.0
        else:
            t_asm = time.perf_counter()
            C = torch.cat([part for part in C_parts if part is not None], dim=0)
            torch.cuda.synchronize(devices[0])
            timings["assemble_s"] = time.perf_counter() - t_asm

    return C, timings


def matmul_col_split(
    A: torch.Tensor,
    B: torch.Tensor,
    devices: list[torch.device],
    use_cuda_events: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    n = B.shape[1]
    sizes = _split_sizes(n, 4)
    offsets = [0]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)

    timings = _empty_timings(devices)
    xfer_events = {}
    compute_events = {}
    C_parts = [None] * 4

    for i, dev in enumerate(devices):
        if sizes[i] == 0:
            C_parts[i] = None
            continue

        col_start = offsets[i]
        col_end = col_start + sizes[i]
        B_i = B[:, col_start:col_end].contiguous()

        with torch.cuda.device(dev):
            if i == 0:
                A_i_dev = A
                B_i_dev = B_i
            else:
                if use_cuda_events:
                    start_xfer = torch.cuda.Event(enable_timing=True)
                    end_xfer = torch.cuda.Event(enable_timing=True)
                    start_xfer.record()
                    A_i_dev = A.to(dev)
                    B_i_dev = B_i.to(dev)
                    end_xfer.record()
                    xfer_events[i] = (start_xfer, end_xfer)
                else:
                    A_i_dev = A.to(dev)
                    B_i_dev = B_i.to(dev)

            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                C_parts[i] = A_i_dev @ B_i_dev
                end.record()
                compute_events[i] = (start, end)
            else:
                C_parts[i] = A_i_dev @ B_i_dev

    _sync_all(devices)

    if use_cuda_events:
        for i in range(1, 4):
            if i in xfer_events:
                start_xfer, end_xfer = xfer_events[i]
                timings[f"gpu0_to_gpu{i}_s"] = start_xfer.elapsed_time(end_xfer) / 1000.0
        for i in range(4):
            if i in compute_events:
                start, end = compute_events[i]
                timings[f"gpu{i}_kernel_s"] = start.elapsed_time(end) / 1000.0

    for i in range(1, 4):
        if C_parts[i] is None:
            continue
        with torch.cuda.device(devices[0]):
            if use_cuda_events:
                start_xfer = torch.cuda.Event(enable_timing=True)
                end_xfer = torch.cuda.Event(enable_timing=True)
                start_xfer.record()
                C_parts[i] = C_parts[i].to(devices[0])
                end_xfer.record()
                xfer_events[i] = (start_xfer, end_xfer)
            else:
                C_parts[i] = C_parts[i].to(devices[0])
    _sync_all(devices)

    if use_cuda_events:
        for i in range(1, 4):
            if i in xfer_events:
                start_xfer, end_xfer = xfer_events[i]
                timings[f"gpu{i}_to_gpu0_s"] = start_xfer.elapsed_time(end_xfer) / 1000.0

    with torch.cuda.device(devices[0]):
        if use_cuda_events:
            start_asm = torch.cuda.Event(enable_timing=True)
            end_asm = torch.cuda.Event(enable_timing=True)
            start_asm.record()
            C = torch.cat([part for part in C_parts if part is not None], dim=1)
            end_asm.record()
            torch.cuda.synchronize(devices[0])
            timings["assemble_s"] = start_asm.elapsed_time(end_asm) / 1000.0
        else:
            t_asm = time.perf_counter()
            C = torch.cat([part for part in C_parts if part is not None], dim=1)
            torch.cuda.synchronize(devices[0])
            timings["assemble_s"] = time.perf_counter() - t_asm

    return C, timings


def matmul_block2x2(
    A: torch.Tensor,
    B: torch.Tensor,
    devices: list[torch.device],
    use_cuda_events: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    n = A.shape[0]
    row_sizes = _split_sizes(n, 2)
    col_sizes = _split_sizes(n, 2)
    row_mid = row_sizes[0]
    col_mid = col_sizes[0]

    A_top = A[:row_mid, :].contiguous()
    A_bot = A[row_mid:, :].contiguous()
    B_left = B[:, :col_mid].contiguous()
    B_right = B[:, col_mid:].contiguous()

    timings = _empty_timings(devices)
    xfer_events = {}
    compute_events = {}

    C_blocks = [None] * 4
    blocks = [
        (A_top, B_left),   # C00 -> gpu0
        (A_top, B_right),  # C01 -> gpu1
        (A_bot, B_left),   # C10 -> gpu2
        (A_bot, B_right),  # C11 -> gpu3
    ]

    for i, dev in enumerate(devices):
        A_part, B_part = blocks[i]
        with torch.cuda.device(dev):
            if i == 0:
                A_dev = A_part
                B_dev = B_part
            else:
                if use_cuda_events:
                    start_xfer = torch.cuda.Event(enable_timing=True)
                    end_xfer = torch.cuda.Event(enable_timing=True)
                    start_xfer.record()
                    A_dev = A_part.to(dev)
                    B_dev = B_part.to(dev)
                    end_xfer.record()
                    xfer_events[i] = (start_xfer, end_xfer)
                else:
                    A_dev = A_part.to(dev)
                    B_dev = B_part.to(dev)

            if use_cuda_events:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                C_blocks[i] = A_dev @ B_dev
                end.record()
                compute_events[i] = (start, end)
            else:
                C_blocks[i] = A_dev @ B_dev

    _sync_all(devices)

    if use_cuda_events:
        for i in range(1, 4):
            if i in xfer_events:
                start_xfer, end_xfer = xfer_events[i]
                timings[f"gpu0_to_gpu{i}_s"] = start_xfer.elapsed_time(end_xfer) / 1000.0
        for i in range(4):
            if i in compute_events:
                start, end = compute_events[i]
                timings[f"gpu{i}_kernel_s"] = start.elapsed_time(end) / 1000.0

    for i in range(1, 4):
        with torch.cuda.device(devices[0]):
            if use_cuda_events:
                start_xfer = torch.cuda.Event(enable_timing=True)
                end_xfer = torch.cuda.Event(enable_timing=True)
                start_xfer.record()
                C_blocks[i] = C_blocks[i].to(devices[0])
                end_xfer.record()
                xfer_events[i] = (start_xfer, end_xfer)
            else:
                C_blocks[i] = C_blocks[i].to(devices[0])
    _sync_all(devices)

    if use_cuda_events:
        for i in range(1, 4):
            if i in xfer_events:
                start_xfer, end_xfer = xfer_events[i]
                timings[f"gpu{i}_to_gpu0_s"] = start_xfer.elapsed_time(end_xfer) / 1000.0

    with torch.cuda.device(devices[0]):
        if use_cuda_events:
            start_asm = torch.cuda.Event(enable_timing=True)
            end_asm = torch.cuda.Event(enable_timing=True)
            start_asm.record()
            top = torch.cat([C_blocks[0], C_blocks[1]], dim=1)
            bot = torch.cat([C_blocks[2], C_blocks[3]], dim=1)
            C = torch.cat([top, bot], dim=0)
            end_asm.record()
            torch.cuda.synchronize(devices[0])
            timings["assemble_s"] = start_asm.elapsed_time(end_asm) / 1000.0
        else:
            t_asm = time.perf_counter()
            top = torch.cat([C_blocks[0], C_blocks[1]], dim=1)
            bot = torch.cat([C_blocks[2], C_blocks[3]], dim=1)
            C = torch.cat([top, bot], dim=0)
            torch.cuda.synchronize(devices[0])
            timings["assemble_s"] = time.perf_counter() - t_asm

    return C, timings


def _run_one(
    strategy: str,
    A: torch.Tensor,
    B: torch.Tensor,
    devices: list[torch.device],
    use_cuda_events: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if strategy == "single":
        return matmul_single_gpu(A, B, devices[0], use_cuda_events)
    if len(devices) < 4:
        raise SystemExit("Need 4 GPUs for multi-GPU strategies")
    if strategy == "row":
        return matmul_row_split(A, B, devices[:4], use_cuda_events)
    if strategy == "col":
        return matmul_col_split(A, B, devices[:4], use_cuda_events)
    if strategy == "block2x2":
        return matmul_block2x2(A, B, devices[:4], use_cuda_events)
    raise ValueError(f"Unknown strategy: {strategy}")


def _print_report(
    strategy: str,
    avg_time_s: float,
    timings: dict[str, float] | None,
    devices: list[torch.device],
    use_cuda_events: bool,
) -> None:
    print(f"--- Strategy: {strategy} ---")
    print(f"Total time:   {avg_time_s*1e3:.3f} ms")
    if timings:
        if timings.get("h2d_s", 0.0) > 0.0 or timings.get("d2h_s", 0.0) > 0.0:
            print(f"H2D (CPU->GPU0): {timings['h2d_s']*1e3:.3f} ms")
            print(f"D2H (GPU0->CPU): {timings['d2h_s']*1e3:.3f} ms")
    if timings and use_cuda_events:
        for i in range(min(4, len(devices))):
            print(f"GPU{i} kernels: {timings[f'gpu{i}_kernel_s']*1e3:.3f} ms")
        for i in range(1, min(4, len(devices))):
            print(f"GPU0->GPU{i}:   {timings[f'gpu0_to_gpu{i}_s']*1e3:.3f} ms")
            print(f"GPU{i}->GPU0:   {timings[f'gpu{i}_to_gpu0_s']*1e3:.3f} ms")
        print(f"Assemble:     {timings['assemble_s']*1e3:.3f} ms")
    elif timings:
        print("Per-stage timings disabled (enable --use-cuda-events)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-GPU matmul benchmark (1 GPU or 4 GPUs)."
    )
    parser.add_argument("n", nargs="?", type=int, help="Matrix size (n x n)")
    parser.add_argument(
        "--ns",
        type=_parse_ns_list,
        default=None,
        help="Comma-separated list of N values (overrides positional n)",
    )
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="float16")
    parser.add_argument(
        "--gpus",
        type=_parse_gpu_list,
        default="0,1,2,3",
        help="Comma-separated GPU indices (e.g. 0,1,2,3)",
    )
    parser.add_argument(
        "--strategy",
        choices=["single", "row", "col", "block2x2", "all"],
        default="all",
        help="Partition strategy to benchmark",
    )
    parser.add_argument("--runs", type=int, default=512, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=16, help="Warmup runs")
    parser.add_argument(
        "--use-cuda-events",
        action="store_true",
        help="Use CUDA events for per-stage timings",
    )
    parser.add_argument(
        "--profile-host-io",
        action="store_true",
        help="Include CPU<->GPU0 transfers (H2D/D2H) in timing",
    )
    parser.add_argument("--wandb", action="store_true", help="Log results to Weights & Biases")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gpu-matmul-bench",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity/team (optional)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (optional)",
    )

    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    devices = [torch.device(f"cuda:{idx}") for idx in args.gpus]
    if args.strategy != "single" and len(devices) < 4:
        raise SystemExit("Need 4 GPUs for multi-GPU strategies")

    dtype = DTYPE_MAP[args.dtype]
    if args.ns is None:
        if args.n is None:
            raise SystemExit("Provide positional n or --ns")
        ns = [args.n]
    else:
        ns = args.ns

    strategies = (
        ["single", "row", "col", "block2x2"] if args.strategy == "all" else [args.strategy]
    )

    if args.wandb:
        if wandb is None:
            raise SystemExit("wandb is not installed; pip install wandb")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "dtype": args.dtype,
                "gpus": args.gpus,
                "runs": args.runs,
                "warmup": args.warmup,
                "use_cuda_events": args.use_cuda_events,
                "profile_host_io": args.profile_host_io,
                "strategies": strategies,
                "ns": ns,
            },
        )

    print("=== Multi-GPU Matmul Benchmark ===")
    print(f"n values = {', '.join(str(n) for n in ns)}, dtype = {args.dtype}")
    print(f"GPUs = {', '.join(str(d) for d in args.gpus)}")
    print(f"Runs = {args.runs}, Warmup = {args.warmup}")
    print(f"Timing mode: {'CUDA events' if args.use_cuda_events else 'CPU wall-clock'}")
    print(f"Profile host IO: {args.profile_host_io}")
    print()

    results: dict[str, list[tuple[int, float, float, float]]] = {s: [] for s in strategies}

    for n in ns:
        dev0 = devices[0]
        if args.profile_host_io:
            A_host = torch.randn(n, n, device="cpu", dtype=dtype)
            B_host = torch.randn(n, n, device="cpu", dtype=dtype)
        else:
            A = torch.randn(n, n, device=dev0, dtype=dtype)
            B = torch.randn(n, n, device=dev0, dtype=dtype)

        print(f"=== n = {n} ===")
        for strategy in strategies:
            for _ in range(args.warmup):
                if args.profile_host_io:
                    A = A_host.to(dev0)
                    B = B_host.to(dev0)
                _ = _run_one(strategy, A, B, devices, args.use_cuda_events)
            _sync_all(devices[:4])

            times = []
            last_detail = None
            for _ in range(args.runs):
                start = time.perf_counter()
                h2d_s = 0.0
                d2h_s = 0.0
                if args.profile_host_io:
                    t_h2d = time.perf_counter()
                    A = A_host.to(dev0)
                    B = B_host.to(dev0)
                    torch.cuda.synchronize(dev0)
                    h2d_s = time.perf_counter() - t_h2d
                C, last_detail = _run_one(strategy, A, B, devices, args.use_cuda_events)
                if args.profile_host_io:
                    t_d2h = time.perf_counter()
                    _ = C.to("cpu")
                    torch.cuda.synchronize(dev0)
                    d2h_s = time.perf_counter() - t_d2h
                _sync_all(devices[:4])
                if last_detail is None:
                    last_detail = _empty_timings(devices[:4])
                last_detail["h2d_s"] = h2d_s
                last_detail["d2h_s"] = d2h_s
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            h2d_ms = (last_detail["h2d_s"] * 1e3) if last_detail else 0.0
            d2h_ms = (last_detail["d2h_s"] * 1e3) if last_detail else 0.0
            results[strategy].append((n, avg_time * 1e3, h2d_ms, d2h_ms))
            _print_report(strategy, avg_time, last_detail, devices, args.use_cuda_events)
        print()

    if args.wandb:
        for strategy, series in results.items():
            table = wandb.Table(
                data=[[n, t_ms, h2d_ms, d2h_ms] for n, t_ms, h2d_ms, d2h_ms in series],
                columns=["n", "time_ms", "h2d_ms", "d2h_ms"],
            )
            wandb.log(
                {
                    f"{strategy}_table": table,
                    f"{strategy}_plot": wandb.plot.line(
                        table, "n", "time_ms", title=f"{strategy} time vs n"
                    ),
                }
            )
        wandb.finish()


if __name__ == "__main__":
    main()
