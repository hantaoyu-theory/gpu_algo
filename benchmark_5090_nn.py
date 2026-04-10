#!/usr/bin/env python3
"""
Measured 5090 benchmark sweep for nearest-neighbor workloads.

Focus:
- synthetic data only
- dimensions d in {8, 16, 32, 64}
- multiple distributions
- Gaussian n-scaling and m-scaling

Outputs JSON with per-algorithm measured query times and recalls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DatasetSpec:
    scenario: str
    distribution: str
    d: int
    m: int
    n_values: list[int]
    seed: int


def make_distribution(dist: str, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    if dist == "gaussian":
        arr = rng.standard_normal(shape, dtype=np.float32)
    elif dist == "uniform":
        # Unit variance: U(-sqrt(3), sqrt(3))
        lim = np.sqrt(3.0)
        arr = rng.uniform(-lim, lim, size=shape).astype(np.float32)
    elif dist == "laplace":
        # Unit variance: Laplace(0, b) with b = 1/sqrt(2)
        scale = 1.0 / np.sqrt(2.0)
        arr = rng.laplace(0.0, scale, size=shape).astype(np.float32)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
    return arr


def build_suite(spec: DatasetSpec, k: int):
    import cupy as cp

    from explore import (
        HAS_CAGRA,
        IVFIndex,
        IVFIndexINT8,
        build_cagra_index,
        cagra_search_grid,
        ground_truth,
        recall_at_k,
        run_cagra_search,
        run_bf_fp16,
        run_bf_fp32,
        run_bf_gemm,
        run_bf_int8,
    )

    max_n = max(spec.n_values)
    rng = np.random.default_rng(spec.seed)
    print(
        f"[dataset] scenario={spec.scenario} dist={spec.distribution} "
        f"m={spec.m} d={spec.d} max_n={max_n}"
    )
    X_np = make_distribution(spec.distribution, (spec.m, spec.d), rng)
    Q_np = make_distribution(spec.distribution, (max_n, spec.d), rng)

    X_gpu = cp.asarray(X_np)
    X_fp16 = X_gpu.astype(cp.float16)

    print("[dataset] computing exact ground truth once ...")
    gt = ground_truth(cp.asarray(Q_np), X_gpu, k)

    algos = [
        ("BF-FP32", lambda Qg: run_bf_fp32(Qg, X_gpu, k, reps=5)),
        ("BF-FP16", lambda Qg: run_bf_fp16(Qg, X_fp16, k, reps=5)),
        ("BF-INT8", lambda Qg: run_bf_int8(Qg, X_gpu, k, reps=5)),
        ("BF-GEMM", lambda Qg: run_bf_gemm(Qg, X_gpu, k, reps=3)),
    ]
    if HAS_CAGRA:
        try:
            print("[dataset] building CAGRA index ...")
            cagra_index, cagra_build_ms = build_cagra_index(X_gpu)
            print(f"[dataset] CAGRA build {cagra_build_ms:.0f} ms")
            algos.extend([
                (name, lambda Qg, sp=search_params: run_cagra_search(cagra_index, Qg, k, sp, reps=5))
                for name, search_params in cagra_search_grid(spec.d)
            ])
        except Exception as e:
            print(f"[dataset] CAGRA unavailable for this spec: {e}")
    if spec.scenario == "dist_core":
        print("[dataset] building IVF indices ...")
        ivf1 = IVFIndex(128, 8)
        ivf1.build(X_np, spec.d)
        ivf8 = IVFIndexINT8(128, 8)
        ivf8.build(X_np, spec.d)
        algos.extend([
            ("IVF1(C=128,p=8)", lambda Qg: ivf1.search(Qg, k, reps=5)),
            ("IVF-INT8(C=128,p=8)", lambda Qg: ivf8.search(Qg, k, reps=5)),
        ])
    return Q_np, gt, algos, recall_at_k


def run_spec(spec: DatasetSpec, k: int) -> list[dict]:
    import cupy as cp

    Q_np, gt, algos, recall_at_k = build_suite(spec, k)
    rows: list[dict] = []
    for n in spec.n_values:
        print(f"\n=== scenario={spec.scenario} dist={spec.distribution} d={spec.d} m={spec.m} n={n} ===")
        Qg = cp.asarray(Q_np[:n])
        gt_n = gt[:n]
        for name, fn in algos:
            ids, ms = fn(Qg)
            rc = recall_at_k(ids, gt_n, k)
            ms_per_query = ms / n
            print(f"  {name:<22} {ms:9.3f} ms  {ms_per_query:9.5f} ms/query  recall={rc:.4f}")
            rows.append(
                {
                    "scenario": spec.scenario,
                    "distribution": spec.distribution,
                    "d": spec.d,
                    "m": spec.m,
                    "n": n,
                    "k": k,
                    "seed": spec.seed,
                    "algorithm": name,
                    "query_ms": float(ms),
                    "ms_per_query": float(ms_per_query),
                    "recall": float(rc),
                }
            )
    return rows


def build_specs(seed: int) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []

    # Core cross-distribution sweep at a fixed moderate problem size.
    for dist in ["gaussian", "uniform", "laplace"]:
        for d in [8, 16, 32, 64]:
            specs.append(
                DatasetSpec(
                    scenario="dist_core",
                    distribution=dist,
                    d=d,
                    m=250_000,
                    n_values=[1000],
                    seed=seed + d,
                )
            )

    # Query-count scaling on Gaussian data.
    for d in [8, 16, 32, 64]:
        specs.append(
            DatasetSpec(
                scenario="gaussian_n",
                distribution="gaussian",
                d=d,
                m=250_000,
                n_values=[100, 1000, 3000],
                seed=seed + 100 + d,
            )
        )

    # Database-size scaling on Gaussian data.
    for d in [8, 16, 32, 64]:
        for m in [100_000, 500_000]:
            specs.append(
                DatasetSpec(
                    scenario="gaussian_m",
                    distribution="gaussian",
                    d=d,
                    m=m,
                    n_values=[1000],
                    seed=seed + 200 + d + (m // 1000),
                )
            )
        specs.append(
            DatasetSpec(
                scenario="gaussian_m",
                distribution="gaussian",
                d=d,
                m=250_000,
                n_values=[1000],
                seed=seed + 300 + d,
            )
        )

    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=str, default="results_5090_nn.json")
    args = ap.parse_args()

    specs = build_specs(args.seed)
    all_rows: list[dict] = []
    for spec in specs:
        all_rows.extend(run_spec(spec, args.k))

    payload = {
        "meta": {
            "k": args.k,
            "seed": args.seed,
            "specs": [asdict(s) for s in specs],
            "platform": "user-provided 5090 VM",
        },
        "results": all_rows,
    }
    with open(args.out_json, "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
