#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cupy as cp

from bf.fp32 import run_bf_fp32
from bf.fp32_heap import run_bf_fp32_heap
from bf.fp32_q2 import run_bf_fp32_q2
from bf.fp32_qgroup import run_bf_fp32_qgroup


VARIANTS = {
    "baseline": lambda Q, X, k, reps: run_bf_fp32(Q, X, k, reps=reps),
    "heap": lambda Q, X, k, reps: run_bf_fp32_heap(Q, X, k, reps=reps),
    "q2": lambda Q, X, k, reps: run_bf_fp32_q2(Q, X, k, reps=reps),
    "q4": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=4, reps=reps),
    "q8": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=8, reps=reps),
    "q16": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=16, reps=reps),
    "q32": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=32, reps=reps),
    "q64": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=64, reps=reps),
    "q128": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=128, reps=reps),
}


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Need at least one integer value")
    return values


def parse_variants(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Need at least one variant")
    bad = [v for v in values if v not in VARIANTS]
    if bad:
        raise ValueError(f"Unknown variants: {bad}")
    return values


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def make_queries(
    rng: cp.random.Generator,
    n: int,
    d: int,
    mode: str,
    cluster_size: int,
    cluster_noise: float,
) -> cp.ndarray:
    if mode == "gaussian":
        return rng.standard_normal((n, d), dtype=cp.float32)
    if mode != "clustered":
        raise ValueError(f"Unknown query mode: {mode}")
    if cluster_size <= 0:
        raise ValueError("cluster_size must be positive")
    n_groups = (n + cluster_size - 1) // cluster_size
    centers = rng.standard_normal((n_groups, d), dtype=cp.float32)
    noise = rng.standard_normal((n, d), dtype=cp.float32) * cp.float32(cluster_noise)
    group_ids = cp.arange(n, dtype=cp.int32) // int(cluster_size)
    return centers[group_ids] + noise


def main() -> None:
    ap = argparse.ArgumentParser(description="Repeated BF-FP32 variant study over n at fixed k.")
    ap.add_argument("--m", type=int, default=1_000_000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n-values", type=str, default="1000,2000,4000,8000,16000")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--variants", type=str, default="baseline,heap,q2,q4,q8,q16,q32,q64,q128")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-mode", type=str, default="gaussian", choices=["gaussian", "clustered"])
    ap.add_argument("--cluster-size", type=int, default=16)
    ap.add_argument("--cluster-noise", type=float, default=0.05)
    ap.add_argument("--skip-failures", action="store_true")
    ap.add_argument("--out-json", type=str, default="bf_fp32_variant_repeat_over_n_d8_k1.json")
    args = ap.parse_args()

    n_values = parse_csv_ints(args.n_values)
    variants = parse_variants(args.variants)
    max_n = max(n_values)

    rng = cp.random.default_rng(args.seed)
    X_gpu = rng.standard_normal((args.m, args.d), dtype=cp.float32)
    Q_full_gpu = make_queries(
        rng,
        max_n,
        args.d,
        args.query_mode,
        args.cluster_size,
        args.cluster_noise,
    )

    rows: list[dict] = []
    failures: list[dict] = []
    print(
        f"BF-FP32 repeated variant study over n | m={args.m:,} d={args.d} k={args.k} "
        f"n_values={n_values} variants={variants} trials={args.trials}"
    )
    for variant in variants:
        print(f"\n[{variant}]")
        runner = VARIANTS[variant]
        for n in n_values:
            Q_gpu = Q_full_gpu[:n]
            trial_ms = []
            last_result = None
            try:
                for _ in range(args.trials):
                    result = runner(Q_gpu, X_gpu, args.k, args.reps)
                    last_result = result
                    trial_ms.append(float(result["ms"]))
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if not args.skip_failures:
                    raise
                failures.append(
                    {
                        "variant": variant,
                        "n": n,
                        "k": args.k,
                        "error": msg,
                    }
                )
                print(f"  n={n:>6}  FAILED  {msg}")
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                continue
            mean_ms, std_ms = mean_std(trial_ms)
            mean_qps = n / (mean_ms * 1e-3)
            std_qps = 0.0 if mean_ms == 0 else (n * std_ms) / ((mean_ms ** 2) * 1e-3)
            row = {
                "variant": variant,
                "distribution": args.query_mode,
                "m": args.m,
                "d": args.d,
                "n": n,
                "k": args.k,
                "trials": args.trials,
                "reps": args.reps,
                "trial_query_ms": trial_ms,
                "mean_query_ms": mean_ms,
                "std_query_ms": std_ms,
                "mean_queries_per_second": mean_qps,
                "std_queries_per_second": std_qps,
                "block": int(last_result["block"]),
            }
            rows.append(row)
            print(
                f"  n={n:>6}  mean={mean_ms:9.3f} ms  std={std_ms:7.3f} ms  "
                f"qps={mean_qps:9.0f} +/- {std_qps:7.0f}  block={row['block']}"
            )

    payload = {
        "meta": {
            "m": args.m,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "trials": args.trials,
            "reps": args.reps,
            "skip_failures": bool(args.skip_failures),
            "query_mode": args.query_mode,
            "cluster_size": args.cluster_size,
            "cluster_noise": args.cluster_noise,
            "n_values": n_values,
            "variants": variants,
        },
        "results": rows,
        "failures": failures,
    }
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
