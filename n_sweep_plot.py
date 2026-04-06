#!/usr/bin/env python3
"""
Sweep query count n for ANN benchmarks and plot query time vs n.

Intended workflow:
  1. Run on a GPU machine to produce JSON:
       python n_sweep_plot.py --run-only --out-json n_sweep_d8.json
  2. Render locally if matplotlib is available:
       python n_sweep_plot.py --plot-only --in-json n_sweep_d8.json --out n_sweep_d8.png

Defaults target the d=8 Gaussian regime that already looked promising on A100.
"""

from __future__ import annotations

import argparse
import json
from typing import Callable

import numpy as np


def _build_synth_suite(
    m: int,
    d: int,
    max_n: int,
    k: int,
    seed: int,
    skip_recall: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, Callable]]]:
    import cupy as cp

    from explore import (
        IVFIndex,
        IVFIndex2,
        IVFIndexINT8,
        ground_truth,
        recall_at_k,
        run_bf_fp16,
        run_bf_fp32,
        run_bf_gemm,
        run_bf_int8,
    )
    np.random.seed(seed)
    return _make_synth_suite(m, d, max_n, k, seed, skip_recall, cp, IVFIndex, IVFIndex2, IVFIndexINT8,
                             ground_truth, run_bf_fp16, run_bf_fp32, run_bf_gemm, run_bf_int8)


def _make_synth_suite(m, d, max_n, k, seed, skip_recall, cp, IVFIndex, IVFIndex2, IVFIndexINT8,
                      ground_truth, run_bf_fp16, run_bf_fp32, run_bf_gemm, run_bf_int8):
    np.random.seed(seed)
    print(f"[setup] generating synthetic Gaussian data: m={m}, d={d}, max_n={max_n}")
    X_np = np.random.randn(m, d).astype(np.float32)
    Q_np = np.random.randn(max_n, d).astype(np.float32)

    X_gpu = cp.asarray(X_np)
    X_fp16 = X_gpu.astype(cp.float16)
    if skip_recall:
        print("[setup] skipping exact ground truth")
        gt = None
    else:
        print("[setup] computing exact ground truth once ...")
        gt = ground_truth(cp.asarray(Q_np), X_gpu, k)

    algos: list[tuple[str, Callable]] = [
        ("BF-FP32", lambda Qg: run_bf_fp32(Qg, X_gpu, k)),
        ("BF-FP16", lambda Qg: run_bf_fp16(Qg, X_fp16, k)),
        ("BF-INT8", lambda Qg: run_bf_int8(Qg, X_gpu, k)),
        ("BF-GEMM", lambda Qg: run_bf_gemm(Qg, X_gpu, k)),
    ]

    if d == 8:
        print("[setup] building d=8 IVF indices ...")
        ivf1_128_8 = IVFIndex(128, 8); ivf1_128_8.build(X_np, d)
        ivf1_256_8 = IVFIndex(256, 8); ivf1_256_8.build(X_np, d)
        ivf2_256_8 = IVFIndex2(256, 8); ivf2_256_8.build(X_np, d)
        ivf8_128_8 = IVFIndexINT8(128, 8); ivf8_128_8.build(X_np, d)
        algos.extend([
            ("IVF1(C=128,p=8)", lambda Qg: ivf1_128_8.search(Qg, k)),
            ("IVF1(C=256,p=8)", lambda Qg: ivf1_256_8.search(Qg, k)),
            ("IVF2(C=256,p=8)", lambda Qg: ivf2_256_8.search(Qg, k)),
            ("IVF-INT8(C=128,p=8)", lambda Qg: ivf8_128_8.search(Qg, k)),
        ])
    elif d == 128:
        print("[setup] building d=128 IVF indices ...")
        ivf1_64_16 = IVFIndex(64, 16); ivf1_64_16.build(X_np, d)
        ivf8_128_8 = IVFIndexINT8(128, 8); ivf8_128_8.build(X_np, d)
        algos.extend([
            ("IVF1(C=64,p=16)", lambda Qg: ivf1_64_16.search(Qg, k)),
            ("IVF-INT8(C=128,p=8)", lambda Qg: ivf8_128_8.search(Qg, k)),
        ])
    else:
        raise ValueError(f"Unsupported synthetic d={d}; expected 8 or 128.")

    return X_np, Q_np, gt, algos


def _build_annb_suite(hdf5_path: str, max_n: int, k: int, skip_recall: bool):
    import cupy as cp

    from explore import (
        IVFIndex,
        IVFIndexINT8,
        recall_at_k,
        run_bf_fp16,
        run_bf_fp32,
        run_bf_gemm,
        run_bf_int8,
    )
    from explore_annb import load_ann_benchmarks_hdf5

    print(f"[setup] loading HDF5 dataset: {hdf5_path}  max_n={max_n}")
    X_np, Q_np, gt_np, mode_note = load_ann_benchmarks_hdf5(hdf5_path, k, max_test=max_n)
    if gt_np is None:
        raise ValueError("Expected precomputed GT from HDF5 for ANN sweep.")
    if skip_recall:
        gt_np = None

    X_gpu = cp.asarray(X_np)
    X_fp16 = X_gpu.astype(cp.float16)

    print("[setup] building ANN IVF indices ...")
    ivf1_64_4 = IVFIndex(64, 4); ivf1_64_4.build(X_np, X_np.shape[1])
    ivf8_64_16 = IVFIndexINT8(64, 16); ivf8_64_16.build(X_np, X_np.shape[1])

    algos: list[tuple[str, Callable]] = [
        ("BF-FP32", lambda Qg: run_bf_fp32(Qg, X_gpu, k)),
        ("BF-FP16", lambda Qg: run_bf_fp16(Qg, X_fp16, k)),
        ("BF-INT8", lambda Qg: run_bf_int8(Qg, X_gpu, k)),
        ("BF-GEMM", lambda Qg: run_bf_gemm(Qg, X_gpu, k)),
        ("IVF1(C=64,p=4)", lambda Qg: ivf1_64_4.search(Qg, k)),
        ("IVF-INT8(C=64,p=16)", lambda Qg: ivf8_64_16.search(Qg, k)),
    ]
    return X_np, Q_np, gt_np, algos, mode_note


def benchmark_n_sweep(
    m: int,
    d: int,
    n_values: list[int],
    k: int,
    seed: int,
    mode: str,
    hdf5_path: str | None = None,
    skip_recall: bool = False,
) -> dict:
    import cupy as cp
    from explore import recall_at_k

    max_n = max(n_values)
    if mode == "synthetic":
        X_np, Q_np, gt, algos = _build_synth_suite(m, d, max_n, k, seed, skip_recall)
        subtitle = "gaussian"
    elif mode == "annb":
        if not hdf5_path:
            raise ValueError("--hdf5 is required for mode=annb")
        X_np, Q_np, gt, algos, mode_note = _build_annb_suite(hdf5_path, max_n, k, skip_recall)
        d = X_np.shape[1]
        m = X_np.shape[0]
        subtitle = f"{hdf5_path} ({mode_note})"
    else:
        raise ValueError(f"Unknown mode={mode}")

    Q_np = Q_np[:max_n]
    if gt is not None:
        gt = gt[:max_n]

    results = []
    for n in n_values:
        print(f"\n=== n={n} ===")
        Qg = cp.asarray(Q_np[:n])
        gt_n = None if gt is None else gt[:n]
        for name, fn in algos:
            ids, ms = fn(Qg)
            rc = None if gt_n is None else recall_at_k(ids, gt_n, k)
            ms_per_query = ms / n
            rc_str = "n/a" if rc is None else f"{rc:.4f}"
            print(f"  {name:<20} {ms:8.2f} ms  {ms_per_query:8.4f} ms/query  recall={rc_str}")
            results.append({
                "algorithm": name,
                "n": int(n),
                "query_ms": float(ms),
                "ms_per_query": float(ms_per_query),
                "recall": None if rc is None else float(rc),
            })

    return {
        "meta": {
            "mode": mode,
            "m": int(m),
            "d": int(d),
            "k": int(k),
            "seed": int(seed),
            "n_values": [int(n) for n in n_values],
            "subtitle": subtitle,
        },
        "results": results,
    }


def plot_results(data: dict, outfile: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = data["results"]
    algos = []
    seen = set()
    for row in records:
        name = row["algorithm"]
        if name not in seen:
            seen.add(name)
            algos.append(name)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

    for i, algo in enumerate(algos):
        pts = [(r["n"], r["query_ms"]) for r in records if r["algorithm"] == algo]
        pts.sort()
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=6,
            label=algo,
        )

    ax.set_xlabel("n (number of queries)")
    ax.set_ylabel("query time (ms)")
    subtitle = data["meta"].get("subtitle", "")
    ax.set_title(f"d={data['meta']['d']} query-time scaling" + (f" - {subtitle}" if subtitle else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    print(f"Plot saved to {outfile}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=500_000)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "annb"])
    ap.add_argument("--hdf5", type=str, default=None)
    ap.add_argument("--n-values", type=int, nargs="+", default=[100, 300, 1000, 3000])
    ap.add_argument("--out-json", type=str, default="n_sweep_d8.json")
    ap.add_argument("--in-json", type=str, default=None)
    ap.add_argument("--out", type=str, default="n_sweep_d8.png")
    ap.add_argument("--run-only", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--skip-recall", action="store_true")
    args = ap.parse_args()

    if args.run_only and args.plot_only:
        raise ValueError("Choose at most one of --run-only / --plot-only")

    if args.plot_only:
        if not args.in_json:
            raise ValueError("--plot-only requires --in-json")
        with open(args.in_json, "r", encoding="ascii") as f:
            data = json.load(f)
        plot_results(data, args.out)
        return

    data = benchmark_n_sweep(args.m, args.d, args.n_values, args.k, args.seed, args.mode, args.hdf5, args.skip_recall)
    with open(args.out_json, "w", encoding="ascii") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {args.out_json}")

    if not args.run_only:
        plot_results(data, args.out)


if __name__ == "__main__":
    main()
