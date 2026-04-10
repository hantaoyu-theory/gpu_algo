#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLOR_BY_K = {
    1: "C0",
    2: "C1",
    4: "C2",
    8: "C3",
    16: "C4",
    32: "C5",
    64: "C6",
}


def load_rows(path: str) -> tuple[dict, list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    return payload["meta"], payload["results"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare BF-FP32 baseline vs tiled BF-FP32 throughput."
    )
    ap.add_argument("--base-json", type=str, default="bf/bf_fp32_k_sweep_d8.json")
    ap.add_argument("--tiled-json", type=str, default="bf/bf_fp32_tiled_k_sweep_d8.json")
    ap.add_argument("--out-png", type=str, default="bf/bf_fp32_vs_tiled_d8_qps.png")
    args = ap.parse_args()

    base_meta, base_rows = load_rows(args.base_json)
    tiled_meta, tiled_rows = load_rows(args.tiled_json)

    ks = sorted({int(r["k"]) for r in base_rows} | {int(r["k"]) for r in tiled_rows})
    ns = sorted({int(r["n"]) for r in base_rows} | {int(r["n"]) for r in tiled_rows})

    plt.figure(figsize=(9.5, 6.0))
    for k in ks:
        color = COLOR_BY_K.get(k)
        base_sub = sorted((r for r in base_rows if int(r["k"]) == k), key=lambda r: int(r["n"]))
        tiled_sub = sorted((r for r in tiled_rows if int(r["k"]) == k), key=lambda r: int(r["n"]))
        if base_sub:
            plt.plot(
                [int(r["n"]) for r in base_sub],
                [float(r["queries_per_second"]) for r in base_sub],
                color=color,
                marker="o",
                linewidth=2,
                label=f"baseline k={k}",
            )
        if tiled_sub:
            plt.plot(
                [int(r["n"]) for r in tiled_sub],
                [float(r["queries_per_second"]) for r in tiled_sub],
                color=color,
                marker="x",
                linestyle="--",
                linewidth=2,
                label=f"tiled k={k}",
            )

    plt.title(
        "BF-FP32 Baseline vs Tiled Throughput "
        f"(m={base_meta['m']:,}, d={base_meta['d']}, tile={tiled_meta.get('tile_size', '?')})"
    )
    plt.xlabel("n (number of queries)")
    plt.ylabel("queries / second")
    plt.xticks(ns, [f"{n//1000}k" if n >= 1000 else str(n) for n in ns])
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
