"""
LSH full parameter sweep: grid over (K, L, w).

Fixes one axis at a time for plotting, but runs the full grid internally.

Usage:
    python lsh_sweep_w.py
    python lsh_sweep_w.py --m 500000 --d 8 --K_values 2 3 4 --L_values 50 100 --w_values 0.2 0.4 0.8 1.2
"""

import argparse
import itertools
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lsh import (
    LSHIndex,
    LSHParams,
    benchmark_exact_neighbors_trivial,
    exact_neighbors_cuvs,
    recall_at_k,
)


def sweep(m, d, n, k, K_values, L_values, w_values, max_cands, bf_query_batch=128, seed=42):
    print(f"\nGenerating data: m={m:,}  d={d}  n={n}  k={k}")
    rng = cp.random.default_rng(seed)
    X = rng.standard_normal((m, d), dtype=cp.float32)
    Q = rng.standard_normal((n, d), dtype=cp.float32)

    print("Computing ground truth (once)...")
    gt = exact_neighbors_cuvs(X, Q, k)
    print("Benchmarking trivial brute-force baseline (once)...")
    _, bf_ms = benchmark_exact_neighbors_trivial(X, Q, k, query_batch=bf_query_batch)
    print(f"  brute-force query time: {bf_ms:.2f} ms  ({bf_ms / max(n, 1):.4f} ms/query)\n")

    configs = list(itertools.product(K_values, L_values, w_values))
    print(f"Running {len(configs)} configurations (K × L × w = {len(K_values)}×{len(L_values)}×{len(w_values)})...\n")
    print(f"{'K':>4} {'L':>5} {'w':>7}  {'recall':>8}  {'query ms':>10}  {'speedup':>8}")
    print("─" * 55)

    # records: list of dicts
    records = []
    for K, L, w in configs:
        params = LSHParams(
            n_tables            = L,
            n_projections       = K,
            bucket_width        = w,
            metric              = "l2",
            max_cands_per_table = max_cands,
        )
        idx = LSHIndex(params)
        idx.build(X)
        _, approx = idx.search(Q, k)
        r = recall_at_k(gt, cp.asnumpy(approx))
        query_ms = sum(v for s, v in idx.profiler._totals.items() if "query/" in s)
        speedup = bf_ms / query_ms if query_ms > 0 else float("inf")
        records.append(dict(K=K, L=L, w=w, recall=r, query_ms=query_ms, speedup=speedup))
        print(f"{K:>4} {L:>5} {w:>7.3f}  {r:>8.4f}  {query_ms:>10.2f}  {speedup:>8.2f}x")

    # print top-10 by recall
    print("\n── Top 10 configs by recall ─────────────────────────────────────")
    print(f"{'K':>4} {'L':>5} {'w':>7}  {'recall':>8}  {'query ms':>10}  {'speedup':>8}")
    print("─" * 55)
    for r in sorted(records, key=lambda x: -x["recall"])[:10]:
        print(f"{r['K']:>4} {r['L']:>5} {r['w']:>7.3f}  {r['recall']:>8.4f}  {r['query_ms']:>10.2f}  {r['speedup']:>8.2f}x")

    # print top-10 by speedup among recall >= 0.8
    good = [r for r in records if r["recall"] >= 0.8]
    if good:
        print("\n── Top 10 configs by speedup (recall ≥ 80%) ────────────────────")
        print(f"{'K':>4} {'L':>5} {'w':>7}  {'recall':>8}  {'query ms':>10}  {'speedup':>8}")
        print("─" * 55)
        for r in sorted(good, key=lambda x: -x["speedup"])[:10]:
            print(f"{r['K']:>4} {r['L']:>5} {r['w']:>7.3f}  {r['recall']:>8.4f}  {r['query_ms']:>10.2f}  {r['speedup']:>8.2f}x")

    return records, bf_ms


def plot(records, K_values, L_values, w_values, bf_ms, outfile="lsh_sweep_w.png"):
    """
    3-panel plot:
      1. Recall vs w  (one line per K, fix L = median L)
      2. Recall vs L  (one line per K, fix w = best w per K)
      3. Speedup vs recall scatter (all configs, colored by K)
    """
    import matplotlib.cm as cm

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    colors  = cm.tab10.colors

    # Panel 1: recall vs w, fix L = median
    ax = axes[0]
    fix_L = sorted(L_values)[len(L_values) // 2]
    for i, K in enumerate(K_values):
        pts = [(r["w"], r["recall"]) for r in records if r["K"] == K and r["L"] == fix_L]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=markers[i % len(markers)], color=colors[i % len(colors)], label=f"K={K}")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, label="90% target")
    ax.set_xlabel("w  (bucket width)", fontsize=11)
    ax.set_ylabel("Recall@k", fontsize=11)
    ax.set_title(f"Recall vs w  (L={fix_L} fixed)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: recall vs L, fix w = best w per K (at median L)
    ax = axes[1]
    for i, K in enumerate(K_values):
        # find best w for this K at median L
        pts_w = [(r["w"], r["recall"]) for r in records if r["K"] == K and r["L"] == fix_L]
        if not pts_w:
            continue
        best_w = max(pts_w, key=lambda x: x[1])[0]
        pts = [(r["L"], r["recall"]) for r in records if r["K"] == K and r["w"] == best_w]
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=markers[i % len(markers)], color=colors[i % len(colors)],
                label=f"K={K}, w={best_w:.2f}")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, label="90% target")
    ax.set_xlabel("L  (number of tables)", fontsize=11)
    ax.set_ylabel("Recall@k", fontsize=11)
    ax.set_title("Recall vs L  (best w per K)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: speedup vs recall scatter (all configs)
    ax = axes[2]
    for i, K in enumerate(K_values):
        pts = [(r["recall"], r["speedup"]) for r in records if r["K"] == K]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   color=colors[i % len(colors)], marker=markers[i % len(markers)],
                   label=f"K={K}", alpha=0.7, s=40)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="BF baseline")
    ax.axvline(0.9, color="red",  linestyle="--", linewidth=1, label="90% recall")
    ax.set_xlabel("Recall@k", fontsize=11)
    ax.set_ylabel("Speedup vs brute-force", fontsize=11)
    ax.set_title("Speedup vs Recall  (all configs)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",          type=int,   default=500_000)
    parser.add_argument("--d",          type=int,   default=8)
    parser.add_argument("--n",          type=int,   default=1_000)
    parser.add_argument("--k",          type=int,   default=10)
    parser.add_argument("--max_cands",  type=int,   default=500)
    parser.add_argument("--bf_query_batch", type=int, default=128)
    parser.add_argument("--K_values",   type=int,   nargs="+",
                        default=[2, 3, 4])
    parser.add_argument("--L_values",   type=int,   nargs="+",
                        default=[50, 100, 150])
    parser.add_argument("--w_values",   type=float, nargs="+",
                        default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--out",        type=str,   default="lsh_sweep_w.png")
    args = parser.parse_args()

    records, bf_ms = sweep(
        m             = args.m,
        d             = args.d,
        n             = args.n,
        k             = args.k,
        K_values      = args.K_values,
        L_values      = args.L_values,
        w_values      = args.w_values,
        max_cands     = args.max_cands,
        bf_query_batch= args.bf_query_batch,
    )
    plot(records, args.K_values, args.L_values, args.w_values, bf_ms, outfile=args.out)


