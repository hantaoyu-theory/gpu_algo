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
    "q4": dict(color="C3", marker="D"),
    "q8": dict(color="C4", marker="v"),
    "q16": dict(color="C5", marker="P"),
    "q32": dict(color="C6", marker="X"),
    "q64": dict(color="C7", marker="<"),
    "q128": dict(color="C8", marker=">"),
}


def variant_sort_key(name: str) -> tuple[int, int | str]:
    if name == "baseline":
        return (0, 0)
    if name == "heap":
        return (1, 0)
    if name.startswith("q") and name[1:].isdigit():
        return (2, int(name[1:]))
    return (3, name)


def parse_json_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Need at least one input JSON path")
    return values


def load_rows(paths: list[str]) -> tuple[dict, list[dict]]:
    merged: dict[tuple[str, int], dict] = {}
    meta: dict | None = None
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="ascii"))
        if meta is None:
            meta = payload["meta"]
        for row in payload["results"]:
            key = (str(row["variant"]), int(row["n"]))
            merged[key] = row
    assert meta is not None
    rows = sorted(merged.values(), key=lambda r: (str(r["variant"]), int(r["n"])))
    return meta, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot repeated BF-FP32 variant study over n.")
    ap.add_argument(
        "--in-jsons",
        type=str,
        default="bf/a100/bf_fp32_variant_repeat_over_n_d8_k1.json",
        help="Comma-separated list of JSON inputs. Later files override duplicate (variant, n) points.",
    )
    ap.add_argument("--out-png", type=str, default="bf/a100/bf_fp32_variant_repeat_over_n_d8_k1.png")
    args = ap.parse_args()

    meta, rows = load_rows(parse_json_list(args.in_jsons))
    n_values = sorted({int(r["n"]) for r in rows})
    variants = sorted({str(r["variant"]) for r in rows}, key=variant_sort_key)

    plt.figure(figsize=(9.0, 5.5))
    x_index = {n: i for i, n in enumerate(n_values)}
    for variant in variants:
        sub = sorted((r for r in rows if r["variant"] == variant), key=lambda r: int(r["n"]))
        style = VARIANT_STYLE.get(variant, {})
        xs = [x_index[int(r["n"])] for r in sub]
        ys = [float(r["mean_queries_per_second"]) for r in sub]
        yerr = [float(r["std_queries_per_second"]) for r in sub]
        plt.errorbar(xs, ys, yerr=yerr, linewidth=2, capsize=4, label=variant, **style)

    plt.title(
        f"BF-FP32 Variant Throughput vs Query Count "
        f"(m={meta['m']:,}, d={meta['d']}, k={meta['k']}, trials={meta['trials']})"
    )
    plt.xlabel("n (number of queries)")
    plt.ylabel("queries / second")
    plt.xticks(range(len(n_values)), [f"{n // 1000}k" for n in n_values])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":
    main()
