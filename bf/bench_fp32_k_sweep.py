#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp

from bf.fp32 import run_bf_fp32


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Need at least one integer value")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BF-FP32 Gaussian sweep over query count n and top-k."
    )
    ap.add_argument("--m", type=int, default=1_000_000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n-values", type=str, default="1000,2000,4000,8000,16000")
    ap.add_argument("--k-values", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out-json", type=str, default="bf_fp32_k_sweep_d8.json")
    args = ap.parse_args()

    n_values = parse_csv_ints(args.n_values)
    k_values = parse_csv_ints(args.k_values)
    max_n = max(n_values)

    rng = cp.random.default_rng(args.seed)
    X_gpu = rng.standard_normal((args.m, args.d), dtype=cp.float32)
    Q_full_gpu = rng.standard_normal((max_n, args.d), dtype=cp.float32)

    rows: list[dict] = []
    print(
        f"BF-FP32 k-sweep | m={args.m:,} d={args.d} "
        f"n_values={n_values} k_values={k_values}"
    )
    for k in k_values:
        print(f"\n[k={k}]")
        for n in n_values:
            result = run_bf_fp32(Q_full_gpu[:n], X_gpu, k, reps=args.reps)
            ms = float(result["ms"])
            row = {
                "algorithm": "BF-FP32",
                "distribution": "gaussian",
                "m": args.m,
                "d": args.d,
                "n": n,
                "k": k,
                "seed": args.seed,
                "reps": args.reps,
                "query_ms": ms,
                "ms_per_query": ms / n,
                "queries_per_second": n / (ms * 1e-3),
                "recall": 1.0,
                "block": int(result["block"]),
                "shared_mem_bytes": int(result["shared_mem_bytes"]),
                "kernel_attrs": result["kernel_attrs"],
            }
            rows.append(row)
            attrs = row["kernel_attrs"]
            print(
                f"  n={n:>6}  {ms:10.3f} ms  "
                f"{row['queries_per_second']:10.0f} qps  "
                f"{row['ms_per_query']:.6f} ms/query  "
                f"variant={attrs.get('kernel_variant', 'unknown')}  "
                f"block={row['block']}  "
                f"regs={attrs.get('num_regs', -1)}  "
                f"smem={row['shared_mem_bytes']}"
            )

    payload = {
        "meta": {
            "algorithm": "BF-FP32",
            "distribution": "gaussian",
            "m": args.m,
            "d": args.d,
            "seed": args.seed,
            "reps": args.reps,
            "n_values": n_values,
            "k_values": k_values,
        },
        "results": rows,
    }
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
