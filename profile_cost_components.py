#!/usr/bin/env python3
"""
Rough post-hoc decomposition of measured runtimes into:
  T_move, T_compute, T_orch

This is intentionally approximate. It is meant to answer:
  "which term is dominating?" rather than give cycle-accurate attribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


GB = 1e9

ALG_RE = re.compile(r"^(?P<family>[A-Z0-9\-]+)(?:\((?P<params>[^)]*)\))?$")


def parse_algo(name: str) -> tuple[str, dict]:
    m = ALG_RE.match(name)
    if not m:
        return name, {}
    params = {}
    raw = m.group("params")
    if raw:
        for item in raw.split(","):
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            try:
                params[k] = int(v)
            except ValueError:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
    return m.group("family"), params


def infer_ann_bytes_and_flops(meta: dict, algo_name: str, n: int) -> tuple[float | None, float | None]:
    m = meta["m"]
    d = meta["d"]
    family, params = parse_algo(algo_name)
    flops = float(2 * n * m * d)
    if family == "BF-FP32":
        return n * m * d * 4.0, flops
    if family == "BF-FP16":
        return n * m * d * 2.0, flops
    if family == "BF-INT8":
        return n * m * d * 1.0, flops
    if family == "BF-GEMM":
        # Rough: read X and Q, materialize/touch output-like tiles; not exact.
        return (n * m * d * 4.0) + (n * m * 4.0), flops
    if family == "IVF1":
        C = params["C"]
        p = params["p"]
        return n * (p / C) * m * d * 4.0, flops * (p / C)
    if family == "IVF2":
        C = params["C"]
        p = params["p"]
        return n * (p / C) * m * d * 4.0, flops * (p / C)
    if family == "IVF-INT8":
        C = params["C"]
        p = params["p"]
        return n * (p / C) * m * d * 1.0, flops * (p / C)
    return None, None


def decompose_ann_record(meta: dict, rec: dict) -> dict:
    name = rec["algorithm"]
    n = rec["n"]
    actual_ms = rec["query_ms"]
    family, _ = parse_algo(name)
    est_bytes, est_flops = infer_ann_bytes_and_flops(meta, name, n)

    # Calibrated rough rates from current A100 evidence.
    bw_map = {
        "BF-FP32": 550.0,
        "BF-FP16": 550.0,
        "BF-INT8": 550.0,
        "IVF1": 551.0,
        "IVF2": 658.0,
        "IVF-INT8": 550.0,
        "BF-GEMM": 337.0,
    }
    # Rough effective compute rates (GFLOP/s or equivalent operation rate).
    compute_map = {
        "BF-FP32": 3000.0,
        "BF-FP16": 4000.0,
        "BF-INT8": 8000.0,
        "IVF1": 3000.0,
        "IVF2": 3000.0,
        "IVF-INT8": 8000.0,
        "BF-GEMM": 230000.0,
    }
    bw = bw_map.get(family)
    peff = compute_map.get(family)

    move_raw = 0.0 if est_bytes is None or bw is None else est_bytes / (bw * GB) * 1e3
    compute_raw = 0.0 if est_flops is None or peff is None else est_flops / (peff * 1e9) * 1e3
    move_ms = min(actual_ms, move_raw)
    rem = max(0.0, actual_ms - move_ms)
    compute_ms = min(rem, compute_raw)
    orch_ms = max(0.0, actual_ms - move_ms - compute_ms)

    return {
        "experiment": meta.get("subtitle", meta.get("mode", "")),
        "algorithm": name,
        "n": n,
        "actual_ms": actual_ms,
        "T_move_ms": move_ms,
        "T_compute_ms": compute_ms,
        "T_orch_ms": orch_ms,
        "dominant": max(
            [("move", move_ms), ("compute", compute_ms), ("orch", orch_ms)],
            key=lambda x: x[1],
        )[0],
    }


def decompose_lsh_row(row: dict) -> dict:
    actual_ms = float(row["query_ms"])
    n = int(row["n"])
    m = int(row["m"])
    d = int(row["d"])
    L = int(row["params"]["L"])
    max_cands = int(row["params"]["max_cands"])

    # Rough upper bound: fused rerank may read up to n * L * max_cands candidate vectors.
    est_bytes = float(n * L * max_cands * d * 4)
    est_flops = float(2 * n * L * max_cands * d)

    # LSH rerank is gather-heavy and irregular, so use a lower effective BW.
    if d <= 8:
        bw_eff = 400.0
    elif d <= 128:
        bw_eff = 300.0
    else:
        bw_eff = 250.0
    peff = 3000.0

    move_raw = est_bytes / (bw_eff * GB) * 1e3
    compute_raw = est_flops / (peff * 1e9) * 1e3
    move_ms = min(actual_ms, move_raw)
    rem = max(0.0, actual_ms - move_ms)
    compute_ms = min(rem, compute_raw)
    orch_ms = max(0.0, actual_ms - move_ms - compute_ms)

    return {
        "experiment": row["case"],
        "algorithm": row["algorithm"],
        "n": n,
        "actual_ms": actual_ms,
        "T_move_ms": move_ms,
        "T_compute_ms": compute_ms,
        "T_orch_ms": orch_ms,
        "dominant": max(
            [("move", move_ms), ("compute", compute_ms), ("orch", orch_ms)],
            key=lambda x: x[1],
        )[0],
    }


def load_copy_bw(rows: list[dict]) -> dict[int, float]:
    out = {}
    for r in rows:
        if r["workload"] == "copy":
            out[int(r["size"])] = float(r["eff_bw_gbs"])
    return out


def decompose_micro_row(row: dict, copy_bw_by_size: dict[int, float]) -> dict:
    workload = row["workload"]
    size = int(row["size"])
    actual_ms = float(row["ms"])
    est_bytes = row["est_bytes"]
    est_flops = row["est_flops"]
    ref_bw = copy_bw_by_size.get(size, max(copy_bw_by_size.values()))
    gemm_peff = 230000.0

    if workload == "copy":
        move_ms = actual_ms
        compute_ms = 0.0
        orch_ms = 0.0
    elif workload in {"saxpy", "reduce"}:
        move_raw = 0.0 if est_bytes is None else est_bytes / (ref_bw * GB) * 1e3
        compute_raw = 0.0 if not est_flops else est_flops / (gemm_peff * 1e9) * 1e3
        move_ms = min(actual_ms, move_raw)
        rem = max(0.0, actual_ms - move_ms)
        compute_ms = min(rem, compute_raw)
        orch_ms = max(0.0, actual_ms - move_ms - compute_ms)
    elif workload == "compact":
        move_raw = 0.0 if est_bytes is None else est_bytes / (ref_bw * GB) * 1e3
        move_ms = min(actual_ms, move_raw)
        compute_ms = 0.0
        orch_ms = max(0.0, actual_ms - move_ms)
    elif workload == "sort":
        move_ms = 0.0
        compute_ms = 0.0
        orch_ms = actual_ms
    elif workload == "gemm":
        move_raw = 0.0 if est_bytes is None else est_bytes / (ref_bw * GB) * 1e3
        compute_raw = 0.0 if not est_flops else est_flops / (gemm_peff * 1e9) * 1e3
        move_ms = min(actual_ms, move_raw)
        rem = max(0.0, actual_ms - move_ms)
        compute_ms = min(rem, compute_raw)
        orch_ms = max(0.0, actual_ms - move_ms - compute_ms)
    else:
        move_ms = compute_ms = 0.0
        orch_ms = actual_ms

    return {
        "experiment": "gpu_model_experiments",
        "algorithm": workload if workload != "gemm" else f"gemm:{row['note']}",
        "n": size,
        "actual_ms": actual_ms,
        "T_move_ms": move_ms,
        "T_compute_ms": compute_ms,
        "T_orch_ms": orch_ms,
        "dominant": max(
            [("move", move_ms), ("compute", compute_ms), ("orch", orch_ms)],
            key=lambda x: x[1],
        )[0],
    }


def write_markdown(rows: list[dict], outfile: Path) -> None:
    lines = [
        "# Cost Component Profiles",
        "",
        "Rough decomposition of measured runtime into `T_move`, `T_compute`, and `T_orch`.",
        "",
        "| experiment | algorithm | n/size | actual ms | T_move | T_compute | T_orch | dominant |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['experiment']} | {r['algorithm']} | {r['n']} | "
            f"{r['actual_ms']:.3f} | {r['T_move_ms']:.3f} | {r['T_compute_ms']:.3f} | "
            f"{r['T_orch_ms']:.3f} | {r['dominant']} |"
        )
    outfile.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="cost_component_profiles.json")
    ap.add_argument("--out-md", default="COST_COMPONENT_PROFILES.md")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    ann_files = [
        base / "n_sweep_d8.json",
        base / "n_sweep_d128_v3.json",
        base / "n_sweep_fashion784_v3.json",
    ]
    rows = []
    for p in ann_files:
        data = json.loads(p.read_text())
        for rec in data["results"]:
            rows.append(decompose_ann_record(data["meta"], rec))

    micro = json.loads((base / "gpu_model_experiments.json").read_text())
    copy_bw = load_copy_bw(micro["rows"])
    for rec in micro["rows"]:
        rows.append(decompose_micro_row(rec, copy_bw))

    lsh_path = base / "lsh_case_benchmarks.json"
    if lsh_path.exists():
        lsh = json.loads(lsh_path.read_text())
        for rec in lsh["rows"]:
            rows.append(decompose_lsh_row(rec))

    (base / args.out_json).write_text(json.dumps({"rows": rows}, indent=2), encoding="ascii")
    write_markdown(rows, base / args.out_md)
    print(f"wrote {base / args.out_json}")
    print(f"wrote {base / args.out_md}")


if __name__ == "__main__":
    main()
