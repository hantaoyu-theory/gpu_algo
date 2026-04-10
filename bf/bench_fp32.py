#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp

from bf.fp32 import run_bf_fp32


def parse_n_values(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Need at least one n value")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Focused BF-FP32 saturation sweep on synthetic Gaussian data."
    )
    ap.add_argument("--m", type=int, default=250_000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-values", type=str, default="1000,2000,4000,8000,16000")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out-json", type=str, default=None)
    args = ap.parse_args()

    n_values = parse_n_values(args.n_values)
    max_n = max(n_values)
    rng = cp.random.default_rng(args.seed)
    X_gpu = rng.standard_normal((args.m, args.d), dtype=cp.float32)
    Q_full_gpu = rng.standard_normal((max_n, args.d), dtype=cp.float32)

    rows: list[dict] = []
    print(
        f"BF-FP32 saturation sweep | m={args.m:,} d={args.d} "
        f"k={args.k} n_values={n_values}"
    )
    for n in n_values:
        result = run_bf_fp32(Q_full_gpu[:n], X_gpu, args.k, reps=args.reps)
        ms = float(result["ms"])
        row = {
            "algorithm": "BF-FP32",
            "m": args.m,
            "d": args.d,
            "n": n,
            "k": args.k,
            "seed": args.seed,
            "reps": args.reps,
            "query_ms": ms,
            "ms_per_query": ms / n,
            "queries_per_second": n / (ms * 1e-3),
            "block": int(result["block"]),
            "shared_mem_bytes": int(result["shared_mem_bytes"]),
        }
        rows.append(row)
        print(
            f"  n={n:>6}  {ms:9.3f} ms  "
            f"{row['ms_per_query']:.6f} ms/query  "
            f"{row['queries_per_second']:.0f} qps  "
            f"block={row['block']}"
        )

    if args.out_json:
        payload = {
            "meta": {
                "algorithm": "BF-FP32",
                "m": args.m,
                "d": args.d,
                "k": args.k,
                "seed": args.seed,
                "reps": args.reps,
                "n_values": n_values,
            },
            "results": rows,
        }
        out_path = Path(args.out_json)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

