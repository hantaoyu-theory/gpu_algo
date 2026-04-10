#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp

from bf.fp32 import run_bf_fp32
from bf.fp32_heap import run_bf_fp32_heap
from bf.fp32_q2 import run_bf_fp32_q2
from bf.fp32_qgroup import run_bf_fp32_qgroup
from bf.fp32_tiled import run_bf_fp32_tiled


VARIANTS = {
    "baseline": lambda Q, X, k, reps: run_bf_fp32(Q, X, k, reps=reps),
    "heap": lambda Q, X, k, reps: run_bf_fp32_heap(Q, X, k, reps=reps),
    "q2": lambda Q, X, k, reps: run_bf_fp32_q2(Q, X, k, reps=reps),
    "q4": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=4, reps=reps),
    "q8": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=8, reps=reps),
    "q16": lambda Q, X, k, reps: run_bf_fp32_qgroup(Q, X, k, qpb=16, reps=reps),
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare BF-FP32 kernel variants.")
    ap.add_argument("--m", type=int, default=1_000_000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n-values", type=str, default="2000,8000,16000")
    ap.add_argument("--k-values", type=str, default="2,4,8,16,32,64")
    ap.add_argument("--variants", type=str, default="baseline,heap,q2,tiled")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out-json", type=str, default="bf_fp32_variant_study_d8.json")
    args = ap.parse_args()

    n_values = parse_csv_ints(args.n_values)
    k_values = parse_csv_ints(args.k_values)
    variants = parse_variants(args.variants)
    max_n = max(n_values)

    rng = cp.random.default_rng(args.seed)
    X_gpu = rng.standard_normal((args.m, args.d), dtype=cp.float32)
    Q_full_gpu = rng.standard_normal((max_n, args.d), dtype=cp.float32)

    rows: list[dict] = []
    print(
        f"BF-FP32 variant study | m={args.m:,} d={args.d} "
        f"n_values={n_values} k_values={k_values} variants={variants}"
    )
    for variant in variants:
        print(f"\n[{variant}]")
        runner = VARIANTS[variant]
        for k in k_values:
            for n in n_values:
                result = runner(Q_full_gpu[:n], X_gpu, k, args.reps)
                ms = float(result["ms"])
                row = {
                    "variant": variant,
                    "distribution": "gaussian",
                    "m": args.m,
                    "d": args.d,
                    "n": n,
                    "k": k,
                    "query_ms": ms,
                    "ms_per_query": ms / n,
                    "queries_per_second": n / (ms * 1e-3),
                    "recall": 1.0,
                    "block": int(result["block"]),
                }
                if "shared_mem_bytes" in result:
                    row["shared_mem_bytes"] = int(result["shared_mem_bytes"])
                if "kernel_attrs" in result:
                    row["kernel_attrs"] = result["kernel_attrs"]
                if "tile_size" in result:
                    row["tile_size"] = int(result["tile_size"])
                    row["n_tiles"] = int(result["n_tiles"])
                rows.append(row)
                print(
                    f"  k={k:<2} n={n:>6}  {ms:9.3f} ms  "
                    f"{row['queries_per_second']:9.0f} qps  block={row['block']}"
                )

    payload = {
        "meta": {
            "m": args.m,
            "d": args.d,
            "seed": args.seed,
            "reps": args.reps,
            "n_values": n_values,
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
