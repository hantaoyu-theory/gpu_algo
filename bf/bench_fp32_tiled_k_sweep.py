#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp

from bf.fp32_tiled import run_bf_fp32_tiled


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Need at least one integer value")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(
        description="BF-FP32 tiled Gaussian sweep over query count n and top-k."
    )
    ap.add_argument("--m", type=int, default=1_000_000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n-values", type=str, default="1000,2000,4000,8000,16000")
    ap.add_argument("--k-values", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--tile-size", type=int, default=125_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out-json", type=str, default="bf_fp32_tiled_k_sweep_d8.json")
    args = ap.parse_args()

    n_values = parse_csv_ints(args.n_values)
    k_values = parse_csv_ints(args.k_values)
    max_n = max(n_values)

    rng = cp.random.default_rng(args.seed)
    X_gpu = rng.standard_normal((args.m, args.d), dtype=cp.float32)
    Q_full_gpu = rng.standard_normal((max_n, args.d), dtype=cp.float32)

    rows: list[dict] = []
    print(
        f"BF-FP32 tiled k-sweep | m={args.m:,} d={args.d} tile_size={args.tile_size:,} "
        f"n_values={n_values} k_values={k_values}"
    )
    for k in k_values:
        print(f"\n[k={k}]")
        for n in n_values:
            result = run_bf_fp32_tiled(
                Q_full_gpu[:n],
                X_gpu,
                k,
                tile_size=args.tile_size,
                reps=args.reps,
            )
            ms = float(result["ms"])
            row = {
                "algorithm": "BF-FP32-TILED",
                "distribution": "gaussian",
                "m": args.m,
                "d": args.d,
                "n": n,
                "k": k,
                "seed": args.seed,
                "reps": args.reps,
                "tile_size": args.tile_size,
                "n_tiles": int(result["n_tiles"]),
                "query_ms": ms,
                "ms_per_query": ms / n,
                "queries_per_second": n / (ms * 1e-3),
                "recall": 1.0,
                "block": int(result["block"]),
                "partial_shared_mem_bytes": int(result["partial_shared_mem_bytes"]),
                "merge_shared_mem_bytes": int(result["merge_shared_mem_bytes"]),
                "partial_kernel_attrs": result["partial_kernel_attrs"],
                "merge_kernel_attrs": result["merge_kernel_attrs"],
            }
            rows.append(row)
            print(
                f"  n={n:>6}  {ms:10.3f} ms  "
                f"{row['queries_per_second']:10.0f} qps  "
                f"{row['ms_per_query']:.6f} ms/query  "
                f"block={row['block']}  tiles={row['n_tiles']}"
            )

    payload = {
        "meta": {
            "algorithm": "BF-FP32-TILED",
            "distribution": "gaussian",
            "m": args.m,
            "d": args.d,
            "seed": args.seed,
            "reps": args.reps,
            "tile_size": args.tile_size,
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
