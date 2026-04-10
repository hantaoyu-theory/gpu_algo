#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


VARIANT_STYLE = {
    "baseline": dict(color="C0", marker="o"),
    "heap": dict(color="C1", marker="s"),
    "q2": dict(color="C2", marker="^"),
    "tiled": dict(color="C3", marker="x"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot repeated BF-FP32 variant study.")
    ap.add_argument("--in-json", type=str, default="bf/bf_fp32_variant_repeat_d8_n16k.json")
    ap.add_argument("--out-png", type=str, default="bf/bf_fp32_variant_repeat_d8_n16k.png")
    args = ap.parse_args()

    payload = json.loads(Path(args.in_json).read_text(encoding="ascii"))
    rows = payload["results"]
    meta = payload["meta"]
    ks = sorted({int(r["k"]) for r in rows})
    variants = sorted({r["variant"] for r in rows})

    plt.figure(figsize=(8.5, 5.5))
    for variant in variants:
        sub = sorted((r for r in rows if r["variant"] == variant), key=lambda r: int(r["k"]))
        style = VARIANT_STYLE.get(variant, {})
        xs = [int(r["k"]) for r in sub]
        ys = [float(r["mean_queries_per_second"]) for r in sub]
        yerr = [float(r["std_queries_per_second"]) for r in sub]
        plt.errorbar(xs, ys, yerr=yerr, linewidth=2, capsize=4, label=variant, **style)

    plt.title(
        f"BF-FP32 Variant Throughput vs k "
        f"(m={meta['m']:,}, d={meta['d']}, n={meta['n']:,}, trials={meta['trials']})"
    )
    plt.xlabel("k")
    plt.ylabel("queries / second")
    plt.xticks(ks, ks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
