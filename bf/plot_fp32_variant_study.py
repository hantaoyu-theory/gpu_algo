#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


VARIANT_STYLE = {
    "baseline": dict(color="C0", marker="o", linestyle="-"),
    "heap": dict(color="C1", marker="s", linestyle="-"),
    "q2": dict(color="C2", marker="^", linestyle="-"),
    "q4": dict(color="C3", marker="D", linestyle="-"),
    "q8": dict(color="C4", marker="v", linestyle="-"),
    "q16": dict(color="C5", marker="P", linestyle="-"),
    "tiled": dict(color="C6", marker="x", linestyle="--"),
}


def load_rows(path: str) -> tuple[dict, list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="ascii"))
    return payload["meta"], payload["results"]


def plot_vs_k(rows: list[dict], meta: dict, out_png: str, n_target: int) -> None:
    plt.figure(figsize=(8.5, 5.5))
    variants = sorted({r["variant"] for r in rows})
    ks = sorted({int(r["k"]) for r in rows if int(r["n"]) == n_target})
    for variant in variants:
        sub = sorted(
            (r for r in rows if r["variant"] == variant and int(r["n"]) == n_target),
            key=lambda r: int(r["k"]),
        )
        style = VARIANT_STYLE.get(variant, {})
        plt.plot(
            [int(r["k"]) for r in sub],
            [float(r["queries_per_second"]) for r in sub],
            linewidth=2,
            label=variant,
            **style,
        )
    plt.title(f"BF-FP32 Variant Throughput vs k (m={meta['m']:,}, d={meta['d']}, n={n_target})")
    plt.xlabel("k")
    plt.ylabel("queries / second")
    plt.xticks(ks, ks)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)


def plot_vs_n_panels(rows: list[dict], meta: dict, out_png: str, k_values: list[int]) -> None:
    ns = sorted({int(r["n"]) for r in rows})
    variants = sorted({r["variant"] for r in rows})
    fig, axes = plt.subplots(1, len(k_values), figsize=(5.0 * len(k_values), 4.5), sharey=True)
    if len(k_values) == 1:
        axes = [axes]
    for ax, k in zip(axes, k_values):
        for variant in variants:
            sub = sorted(
                (r for r in rows if r["variant"] == variant and int(r["k"]) == k),
                key=lambda r: int(r["n"]),
            )
            style = VARIANT_STYLE.get(variant, {})
            ax.plot(
                [int(r["n"]) for r in sub],
                [float(r["queries_per_second"]) for r in sub],
                linewidth=2,
                label=variant,
                **style,
            )
        ax.set_title(f"k={k}")
        ax.set_xlabel("n")
        ax.set_xticks(ns, [f"{n//1000}k" for n in ns])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("queries / second")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(variants))
    fig.suptitle(f"BF-FP32 Variant Throughput vs n (m={meta['m']:,}, d={meta['d']})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot BF-FP32 variant-study comparisons.")
    ap.add_argument("--in-json", type=str, default="bf/bf_fp32_variant_study_d8.json")
    ap.add_argument("--out-vs-k", type=str, default="bf/bf_fp32_variant_vs_k_d8.png")
    ap.add_argument("--out-vs-n", type=str, default="bf/bf_fp32_variant_vs_n_d8.png")
    ap.add_argument("--n-target", type=int, default=16000)
    ap.add_argument("--panel-ks", type=str, default="4,8,32")
    args = ap.parse_args()

    meta, rows = load_rows(args.in_json)
    panel_ks = [int(part.strip()) for part in args.panel_ks.split(",") if part.strip()]
    plot_vs_k(rows, meta, args.out_vs_k, args.n_target)
    plot_vs_n_panels(rows, meta, args.out_vs_n, panel_ks)
    print(f"Wrote {args.out_vs_k}")
    print(f"Wrote {args.out_vs_n}")


if __name__ == "__main__":
    main()
