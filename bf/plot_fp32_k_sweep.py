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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot BF-FP32 queries/sec vs n with one line per k."
    )
    ap.add_argument("--in-json", type=str, default="bf_fp32_k_sweep_d8.json")
    ap.add_argument("--extra-json", type=str, nargs="*", default=[])
    ap.add_argument("--out-png", type=str, default="bf_fp32_k_sweep_d8_qps.png")
    ap.add_argument("--min-recall", type=float, default=0.9)
    args = ap.parse_args()

    all_payloads = []
    for path_str in [args.in_json, *args.extra_json]:
        payload = json.loads(Path(path_str).read_text(encoding="ascii"))
        all_payloads.append(payload)

    rows = []
    for payload in all_payloads:
        rows.extend(r for r in payload["results"] if float(r.get("recall", 0.0)) >= args.min_recall)
    if not rows:
        raise ValueError("No rows meet the recall threshold")

    # Keep the first row for each (k, n) pair so callers can pass a primary
    # file and then fill in missing lines from secondary files.
    deduped = {}
    for row in rows:
        key = (int(row["k"]), int(row["n"]))
        deduped.setdefault(key, row)
    rows = list(deduped.values())

    ks = sorted({int(r["k"]) for r in rows})
    ns = sorted({int(r["n"]) for r in rows})
    meta = all_payloads[0]["meta"]

    plt.figure(figsize=(8.5, 5.5))
    for k in ks:
        sub = sorted((r for r in rows if int(r["k"]) == k), key=lambda r: int(r["n"]))
        plt.plot(
            [int(r["n"]) for r in sub],
            [float(r["queries_per_second"]) for r in sub],
            marker="o",
            linewidth=2,
            label=f"k={k}",
            color=COLOR_BY_K.get(k),
        )

    plt.title(f"BF-FP32 Gaussian Throughput vs Query Count (m={meta['m']:,}, d={meta['d']})")
    plt.xlabel("n (number of queries)")
    plt.ylabel("queries / second")
    plt.xticks(ns, [f"{n//1000}k" if n >= 1000 else str(n) for n in ns])
    plt.grid(True, alpha=0.3)
    plt.legend(title="top-k", ncol=2)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
