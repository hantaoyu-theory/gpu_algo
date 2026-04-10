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
from bf.fp32_tiled import run_bf_fp32_tiled


VARIANTS = {
    "baseline": lambda Q, X, k, reps: run_bf_fp32(Q, X, k, reps=reps),
    "heap": lambda Q, X, k, reps: run_bf_fp32_heap(Q, X, k, reps=reps),
    "q2": lambda Q, X, k, reps: run_bf_fp32_q2(Q, X, k, reps=reps),
    "tiled": lambda Q, X, k, reps: run_bf_fp32_tiled(Q, X, k, tile_size=125_000, reps=reps),
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Repeated BF-FP32 variant study at fixed n.")
    ap.add_argument("--m", type=int, default=1_000_000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n", type=int, default=16_000)
    ap.add_argument("--k-values", type=str, default="2,4,8,16,32,64")
    ap.add_argument("--variants", type=str, default="baseline,heap,q2,tiled")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str, default="bf_fp32_variant_repeat_d8_n16k.json")
    args = ap.parse_args()

    k_values = parse_csv_ints(args.k_values)
    variants = parse_variants(args.variants)

    rng = cp.random.default_rng(args.seed)
    X_gpu = rng.standard_normal((args.m, args.d), dtype=cp.float32)
    Q_gpu = rng.standard_normal((args.n, args.d), dtype=cp.float32)

    rows: list[dict] = []
    print(
        f"BF-FP32 repeated variant study | m={args.m:,} d={args.d} n={args.n:,} "
        f"k_values={k_values} variants={variants} trials={args.trials}"
    )
    for variant in variants:
        print(f"\n[{variant}]")
        runner = VARIANTS[variant]
        for k in k_values:
            trial_ms = []
            last_result = None
            for _ in range(args.trials):
                result = runner(Q_gpu, X_gpu, k, args.reps)
                last_result = result
                trial_ms.append(float(result["ms"]))
            mean_ms, std_ms = mean_std(trial_ms)
            mean_qps = args.n / (mean_ms * 1e-3)
            std_qps = 0.0 if mean_ms == 0 else (args.n * std_ms) / ((mean_ms ** 2) * 1e-3)
            row = {
                "variant": variant,
                "distribution": "gaussian",
                "m": args.m,
                "d": args.d,
                "n": args.n,
                "k": k,
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
                f"  k={k:<2}  mean={mean_ms:9.3f} ms  std={std_ms:7.3f} ms  "
                f"qps={mean_qps:9.0f} +/- {std_qps:7.0f}  block={row['block']}"
            )

    payload = {
        "meta": {
            "m": args.m,
            "d": args.d,
            "n": args.n,
            "seed": args.seed,
            "trials": args.trials,
            "reps": args.reps,
            "k_values": k_values,
            "variants": variants,
        },
        "results": rows,
    }
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
