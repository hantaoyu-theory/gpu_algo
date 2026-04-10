#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="ascii") as f:
        return json.load(f)["results"]


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
    out = {}
    for r in rows:
        key = (
            r["scenario"],
            r["distribution"],
            r["d"],
            r["m"],
            r["n"],
            r["algorithm"],
        )
        out[key] = r
    return out


def cell(row: dict | None) -> str:
    if row is None:
        return "-"
    return f"`{row['query_ms']:.3f} ms / {row['recall']:.4f}`"


def select_algos(rows: list[dict], base_algos: list[str]) -> list[str]:
    present = {r["algorithm"] for r in rows}
    ordered = [algo for algo in base_algos if algo in present]
    cagra_algos = sorted(algo for algo in present if algo.startswith("CAGRA("))
    return ordered + cagra_algos


def emit_dist_core(lines: list[str], idx: dict, rows: list[dict]) -> None:
    algos = select_algos(rows, [
        "BF-FP32",
        "BF-FP16",
        "BF-INT8",
        "BF-GEMM",
        "IVF1(C=128,p=8)",
        "IVF-INT8(C=128,p=8)",
    ])
    dists = ["gaussian", "uniform", "laplace"]
    for d in [8, 16, 32, 64]:
        lines.extend([
            f"## Distribution Sweep, d = {d}",
            "",
            "`m=250,000`, `n=1000`, `k=10`",
            "",
            "| algorithm | gaussian | uniform | laplace |",
            "|---|---:|---:|---:|",
        ])
        for algo in algos:
            vals = [
                cell(idx.get(("dist_core", dist, d, 250_000, 1000, algo)))
                for dist in dists
            ]
            lines.append(f"| `{algo}` | {vals[0]} | {vals[1]} | {vals[2]} |")
        lines.append("")


def emit_gaussian_n(lines: list[str], idx: dict, rows: list[dict]) -> None:
    algos = select_algos(rows, ["BF-FP32", "BF-FP16", "BF-INT8", "BF-GEMM"])
    ns = [100, 1000, 3000]
    for d in [8, 16, 32, 64]:
        lines.extend([
            f"## Gaussian Query Scaling, d = {d}",
            "",
            "`m=250,000`, `k=10`",
            "",
            "| algorithm | n=100 | n=1000 | n=3000 |",
            "|---|---:|---:|---:|",
        ])
        for algo in algos:
            vals = [
                cell(idx.get(("gaussian_n", "gaussian", d, 250_000, n, algo)))
                for n in ns
            ]
            lines.append(f"| `{algo}` | {vals[0]} | {vals[1]} | {vals[2]} |")
        lines.append("")


def emit_gaussian_m(lines: list[str], idx: dict, rows: list[dict]) -> None:
    algos = select_algos(rows, ["BF-FP32", "BF-FP16", "BF-INT8", "BF-GEMM"])
    ms = [100_000, 250_000, 500_000]
    for d in [8, 16, 32, 64]:
        lines.extend([
            f"## Gaussian Database Scaling, d = {d}",
            "",
            "`n=1000`, `k=10`",
            "",
            "| algorithm | m=100k | m=250k | m=500k |",
            "|---|---:|---:|---:|",
        ])
        for algo in algos:
            vals = [
                cell(idx.get(("gaussian_m", "gaussian", d, m, 1000, algo)))
                for m in ms
            ]
            lines.append(f"| `{algo}` | {vals[0]} | {vals[1]} | {vals[2]} |")
        lines.append("")


def main() -> None:
    base = Path(__file__).resolve().parent
    rows = load_results(base / "results_5090_nn.json")
    idx = index_rows(rows)

    lines = [
        "# Nearest-Neighbor Summary (RTX 5090)",
        "",
        "Measured totals only. All numbers below were collected on the user-provided RTX 5090 VM.",
        "",
        "Cell format: `query_ms / recall@10`.",
        "",
        "Benchmark matrix:",
        "- Distribution sweep: `distribution in {gaussian, uniform, laplace}`, `d in {8,16,32,64}`, `m=250,000`, `n=1000`",
        "- Gaussian query scaling: `d in {8,16,32,64}`, `m=250,000`, `n in {100,1000,3000}`",
        "- Gaussian database scaling: `d in {8,16,32,64}`, `n=1000`, `m in {100,000,250,000,500,000}`",
        "- `k=10` throughout",
        "",
    ]
    emit_dist_core(lines, idx, rows)
    emit_gaussian_n(lines, idx, rows)
    emit_gaussian_m(lines, idx, rows)

    (base / "COST_COMPONENT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
