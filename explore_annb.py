#!/usr/bin/env python3
"""
Run gpu_algo explore benchmarks on [ann-benchmarks] HDF5 datasets.

  https://github.com/erikbern/ann-benchmarks

Expected HDF5 keys: ``train`` (m,d), ``test`` (n,d), ``neighbors`` (n, Kgt) int indices into train.

Download examples (see ann-benchmarks README):
  wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
  wget http://ann-benchmarks.com/sift-128-euclidean.hdf5
  wget http://ann-benchmarks.com/glove-100-angular.hdf5

**Angular / cosine** datasets: rows are L2-normalized so that Euclidean k-NN
ranking matches the benchmark's angular neighbor definition.

**Subsampling train:** if ``max_train`` is smaller than the file’s train size, file
neighbors are **invalid** (they index the full corpus). We then set ``gt=None`` and
``explore`` **recomputes** exact k-NN on GPU for ``train[:max_train]`` (slower but correct).
Use full train (omit ``--max-train``) to use precomputed ``neighbors`` from the HDF5.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:
    print("Install h5py:  pip install h5py", file=sys.stderr)
    raise SystemExit(1) from e

from explore import run_benchmark_suite


def _distance_mode(h5_path: str, f: h5py.File) -> str:
    raw = f.attrs.get("distance", None)
    if raw is None:
        base = os.path.basename(h5_path).lower()
        return "angular" if "angular" in base or "dot" in base else "euclidean"
    if isinstance(raw, bytes):
        return raw.decode().lower()
    return str(raw).lower()


def load_ann_benchmarks_hdf5(
    path: str,
    k: int,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    with h5py.File(path, "r") as f:
        if "train" not in f or "test" not in f or "neighbors" not in f:
            raise KeyError("HDF5 must contain datasets: train, test, neighbors")
        train = np.asarray(f["train"][:], dtype=np.float32)
        test = np.asarray(f["test"][:], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"][:], dtype=np.int64)
        mode = _distance_mode(path, f)

    if neighbors.ndim != 2 or neighbors.shape[0] != test.shape[0]:
        raise ValueError(f"neighbors shape {neighbors.shape} incompatible with test {test.shape}")
    if k > neighbors.shape[1]:
        raise ValueError(f"--k={k} but neighbors only has {neighbors.shape[1]} columns")

    n_full = train.shape[0]
    recompute_gt = False
    if max_train is not None and max_train < n_full:
        train = train[:max_train].copy()
        recompute_gt = True
        print(f"  [annb] max_train={max_train} < {n_full}: will recompute GT on GPU (file neighbors invalid).")
    elif max_train is not None:
        train = train[:max_train].copy()

    if max_test is not None:
        test = test[:max_test]
        if not recompute_gt:
            neighbors = neighbors[:max_test]

    if mode == "angular" or "angular" in mode or mode == "dot":
        # Unit L2 norm: Euclidean distance ranking = cosine / inner-product ranking
        nt = np.linalg.norm(train, axis=1, keepdims=True)
        train = train / np.maximum(nt, 1e-12)
        nq = np.linalg.norm(test, axis=1, keepdims=True)
        test = test / np.maximum(nq, 1e-12)
        mode_note = "angular→L2 on unit vectors"
    else:
        mode_note = "euclidean"

    if recompute_gt:
        gt: Optional[np.ndarray] = None
    else:
        gt = neighbors[:, :k].astype(np.int64, copy=False)
    return train, test, gt, mode_note


def main():
    ap = argparse.ArgumentParser(description="Explore benchmarks on ann-benchmarks HDF5")
    ap.add_argument("--hdf5", required=True, help="Path to .hdf5 (train/test/neighbors)")
    ap.add_argument("--k", type=int, default=10, help="k for search & recall (≤ width of neighbors in file)")
    ap.add_argument("--max-train", type=int, default=None, help="Use only train[:max_train]")
    ap.add_argument("--max-test", type=int, default=None, help="Cap number of queries")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-lsh", action="store_true")
    ap.add_argument("--lsh-oversample", type=int, default=4)
    ap.add_argument("--export-json", type=str, default=None,
                    help="Write machine-readable benchmark/model data to JSON")
    ap.add_argument("--skip-pq", action="store_true",
                    help="Skip PQ and IVF-PQ runs")
    ap.add_argument(
        "--include-dsweep",
        action="store_true",
        help="Also run synthetic d-sweep at end (usually irrelevant for real data)",
    )
    args = ap.parse_args()

    X_np, Q_np, gt_np, mode_note = load_ann_benchmarks_hdf5(
        args.hdf5, args.k, args.max_train, args.max_test
    )
    sub = f"{os.path.basename(args.hdf5)}  ({mode_note})"
    run_benchmark_suite(
        args,
        X_np,
        Q_np,
        gt_np=gt_np,
        skip_dsweep=not args.include_dsweep,
        subtitle=sub,
    )


if __name__ == "__main__":
    main()
