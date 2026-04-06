"""
GPU-Accelerated LSH  (E2LSH for L2, SimHash for cosine)
=========================================================
Pipeline
--------
Build
  1. generate_projections  – sample R ~ N(0,1)^(L*K x d), b ~ U[0,w)^(L*K)
  2. gemm_and_quantize     – X @ R.T  then floor((·+b)/w)  or sign(·)
  3. pack_keys             – polynomial hash K int32 codes → 1 int64 per table
  4. argsort_tables        – radix-sort each table by key (enables searchsorted)

Query
  5. gemm_and_quantize     – same projection for queries
  6. pack_keys             – same packing
  7. candidate_lookup      – searchsorted per table, vectorised gather  (no Python loop over n)
  8. dedup                 – fully GPU: sort + cumsum scatter
  9. gather_vectors        – index into dataset with candidate ids
  10. rerank_distances     – ||q - c||^2 via GEMM identity (q_sq + c_sq - 2*q·c)
  11. topk_selection       – argpartition + argsort

Every stage is wrapped in CUDAProfiler so you can see exactly which part
dominates when you run experiments.

Dependencies: cupy, cuvs  (pip install cupy-cuda12x  cuvs-cu12)
"""

from __future__ import annotations

import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Optional, Tuple

# Optional NVTX support — used by Nsight Systems (nsys) and Nsight Compute (ncu)
# to show named ranges on the GPU timeline.
# Install with: pip install nvtx
try:
    import nvtx as _nvtx
    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False

# Colour palette for NVTX ranges (one colour per pipeline stage group)
_NVTX_COLORS = {
    "build": 0x4C72B0,   # blue
    "query": 0xDD8452,   # orange
}

def _nvtx_color(stage: str) -> int:
    if "build" in stage:
        return _NVTX_COLORS["build"]
    return _NVTX_COLORS["query"]


# ─────────────────────────────────────────────────────────────────────────────
# Profiler
# ─────────────────────────────────────────────────────────────────────────────

class CUDAProfiler:
    """
    GPU-accurate per-stage timer using CUDA events.

    CPU wall-clock timers are misleading for GPU code because CUDA launches are
    asynchronous.  CUDA events are inserted directly into the GPU stream and
    measure actual GPU execution time.

    When the `nvtx` package is installed, each stage is also annotated with an
    NVTX range so it shows up as a named coloured bar in Nsight Systems and
    Nsight Compute.
    """

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int]   = {}
        self._starts: dict[str, cp.cuda.Event] = {}

    def reset(self):
        self._totals.clear()
        self._counts.clear()
        self._starts.clear()

    def start(self, stage: str):
        if _NVTX_AVAILABLE:
            _nvtx.push_range(stage, color=_nvtx_color(stage))
        e = cp.cuda.Event()
        e.record()
        self._starts[stage] = e

    def stop(self, stage: str):
        end = cp.cuda.Event()
        end.record()
        end.synchronize()                                   # wait for GPU to finish
        if _NVTX_AVAILABLE:
            _nvtx.pop_range()
        ms = cp.cuda.get_elapsed_time(self._starts.pop(stage), end)
        self._totals[stage] = self._totals.get(stage, 0.0) + ms
        self._counts[stage] = self._counts.get(stage, 0) + 1

    def report(self, title: str = "Profile"):
        if not self._totals:
            print("No profiling data.")
            return
        total = sum(self._totals.values())
        W = 38
        print(f"\n{'═' * (W + 32)}")
        print(f"  {title}")
        print(f"{'─' * (W + 32)}")
        print(f"  {'Stage':<{W}} {'ms':>9}  {'calls':>5}  {'%':>6}")
        print(f"{'─' * (W + 32)}")
        for stage, ms in sorted(self._totals.items()):
            pct = 100.0 * ms / total if total else 0.0
            cnt = self._counts[stage]
            bar = "█" * int(pct / 4)
            print(f"  {stage:<{W}} {ms:>9.2f}  {cnt:>5}  {pct:>5.1f}%  {bar}")
        print(f"{'─' * (W + 32)}")
        print(f"  {'Total':<{W}} {total:>9.2f}")
        print(f"{'═' * (W + 32)}\n")

    def get(self, stage: str) -> float:
        """Return total ms for a stage (0 if not recorded)."""
        return self._totals.get(stage, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LSHParams:
    n_tables: int            = 10    # L  – number of independent hash tables
    n_projections: int       = 4     # K  – hash functions per table (bits per key)
    bucket_width: float      = 0.0   # w  – quantisation width; 0 = auto-tune from data
    metric: str              = "l2"  # "l2" or "cosine"
    max_cands_per_table: int = 100   # hard cap on candidates gathered per (query, table)
    seed: int                = 42


# ─────────────────────────────────────────────────────────────────────────────
# LSH Index
# ─────────────────────────────────────────────────────────────────────────────

class LSHIndex:
    """
    GPU LSH index.  All heavy computation stays on the GPU (CuPy arrays).

    Usage
    -----
    params = LSHParams(n_tables=10, n_projections=8, bucket_width=4.0)
    index  = LSHIndex(params)
    index.build(X)                       # X: cp.ndarray (m, d) float32
    distances, neighbors = index.search(Q, k=10)   # Q: (n, d) float32
    index.profiler.report()              # see breakdown
    """

    def __init__(self, params: LSHParams):
        self.p        = params
        self.profiler = CUDAProfiler()

        self._R:               Optional[cp.ndarray] = None  # (L*K, d)
        self._b:               Optional[cp.ndarray] = None  # (L*K,)  E2LSH only
        self._w:               float                = 1.0   # effective bucket width
        self._sorted_keys:    list[cp.ndarray]      = []    # L × (m,) int64
        self._sorted_indices: list[cp.ndarray]      = []    # L × (m,) int32
        self._dataset:         Optional[cp.ndarray] = None  # (m, d)
        self._m: int = 0
        self._d: int = 0

    # ── internal helpers ────────────────────────────────────────────────────

    def _project_and_hash(self, X: cp.ndarray) -> cp.ndarray:
        """
        Single GEMM + quantisation step.
        X  : (n, d)  float32
        out: (n, L, K) int32
        """
        n  = X.shape[0]
        proj = X @ self._R.T                                    # (n, L*K)  ← one cuBLAS GEMM
        if self.p.metric == "l2":
            codes = cp.floor((proj + self._b) / self._w).astype(cp.int32)
        else:                                                   # SimHash / cosine
            codes = (proj >= 0.0).astype(cp.int32)
        return codes.reshape(n, self.p.n_tables, self.p.n_projections)

    def _pack_keys(self, codes: cp.ndarray) -> cp.ndarray:
        """
        Polynomial hash: K int32 codes → 1 int64 per (point, table).
        codes: (n, L, K)  →  keys: (n, L) int64

        Uses Knuth multiplicative constants (one per projection slot) so that
        different orderings of the same integers produce different keys.
        Note: hash collisions cause false negatives only (we verify with exact
        distance anyway), not false positives.
        """
        K     = self.p.n_projections
        mults = cp.array([int(2654435761) * (i + 1) for i in range(K)], dtype=cp.int64)
        return (codes.astype(cp.int64) * mults[None, None, :]).sum(axis=2)   # (n, L)

    # ── build ───────────────────────────────────────────────────────────────

    def build(self, X: cp.ndarray):
        """
        Build the LSH index from dataset X of shape (m, d).
        Prints a per-stage timing table when done.
        """
        prof    = self.profiler
        prof.reset()
        m, d    = X.shape
        L, K    = self.p.n_tables, self.p.n_projections

        # ── stage 1 ──────────────────────────────────────────────────────────
        prof.start("1_build/generate_projections")
        rng = cp.random.default_rng(self.p.seed)

        # Auto-tune bucket_width if not set.
        # Sample 512 random pairs and estimate the mean nearest-neighbour
        # distance, then set w = mean_nn_dist / 2.  This keeps p1 (collision
        # probability for near neighbours) in a useful range (~0.5–0.7).
        w = self.p.bucket_width
        if self.p.metric == "l2" and w <= 0.0:
            n_sample = min(512, m)
            idx      = rng.integers(0, m, (n_sample,))
            sample   = X[idx].astype(cp.float32)                    # (n_sample, d)
            sq       = (sample ** 2).sum(axis=1)
            dists_sq = sq[:, None] + sq[None, :] - 2.0 * (sample @ sample.T)
            dists_sq = cp.maximum(dists_sq, 0.0)
            # exclude self-distances (diagonal)
            cp.fill_diagonal(dists_sq, cp.float32(1e18))
            mean_nn_dist = float(cp.sqrt(dists_sq.min(axis=1)).mean())
            w = mean_nn_dist
            print(f"[LSHIndex] auto-tuned w={w:.3f}  (mean 1-NN dist in sample = {mean_nn_dist:.3f})")

        self._R = rng.standard_normal((L * K, d), dtype=cp.float32)
        if self.p.metric == "l2":
            self._b = rng.uniform(0.0, w, (L * K,)).astype(cp.float32)
        self._w = w
        prof.stop("1_build/generate_projections")

        # ── stage 2 ──────────────────────────────────────────────────────────
        prof.start("2_build/gemm_and_quantize")
        codes = self._project_and_hash(X)           # (m, L, K)
        prof.stop("2_build/gemm_and_quantize")

        # ── stage 3 ──────────────────────────────────────────────────────────
        prof.start("3_build/pack_keys")
        keys = self._pack_keys(codes)               # (m, L)
        prof.stop("3_build/pack_keys")

        # ── stage 4 ──────────────────────────────────────────────────────────
        # Each table gets an INDEPENDENT random permutation before sorting.
        # This breaks the argsort stability tie-breaking (which would otherwise
        # order within-bucket points by index, giving all tables the same
        # deterministic first-max_c candidates from each bucket).
        # With random per-table permutation, different tables sample different
        # subsets of large buckets, restoring the independence that LSH needs.
        prof.start("4_build/argsort_tables")
        self._sorted_keys    = []
        self._sorted_indices = []
        for l in range(L):
            perm  = cp.argsort(rng.random(m))                  # random permutation for this table
            order = perm[cp.argsort(keys[perm, l])]            # random within-bucket ordering
            self._sorted_indices.append(order.astype(cp.int32))
            self._sorted_keys.append(keys[order, l])
        prof.stop("4_build/argsort_tables")

        self._dataset = X
        self._m, self._d = m, d

        print(f"\n[LSHIndex] Built  m={m:,}  d={d}  L={L}  K={K}  "
              f"metric={self.p.metric}  w={self._w:.3f}")
        prof.report("Build Profile")

    # ── search ──────────────────────────────────────────────────────────────

    def search(self, Q: cp.ndarray, k: int) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Approximate k-NN for each query in Q.
        Q       : (n, d) float32, on GPU
        returns : (distances, indices)  each (n, k),  sorted ascending by distance
        """
        prof  = self.profiler
        prof.reset()                        # reset so query report is query-only
        n     = Q.shape[0]
        L, K  = self.p.n_tables, self.p.n_projections
        max_c = self.p.max_cands_per_table
        m     = self._m
        C     = L * max_c           # total candidate slots per query

        # ── stage 5 ──────────────────────────────────────────────────────────
        prof.start("5_query/gemm_and_quantize")
        q_codes = self._project_and_hash(Q)         # (n, L, K)
        prof.stop("5_query/gemm_and_quantize")

        # ── stage 6 ──────────────────────────────────────────────────────────
        prof.start("6_query/pack_keys")
        q_keys = self._pack_keys(q_codes)           # (n, L)
        prof.stop("6_query/pack_keys")

        # ── stage 7 ──────────────────────────────────────────────────────────
        # Vectorised gather: no Python loop over queries (n).
        # For each table l we compute, for every query q:
        #   bucket boundaries via searchsorted  (O(n log m) on GPU)
        #   candidate offsets via broadcasting  (n, max_c) index array
        # This maps irregularly-sized buckets to a fixed (n, max_c) block.
        prof.start("7_query/candidate_lookup")
        all_cands = cp.full((n, C), -1, dtype=cp.int32)

        offsets_base = cp.arange(max_c, dtype=cp.int32)[None, :]   # (1, max_c)
        for l in range(L):
            sk    = self._sorted_keys[l]                            # (m,)
            si    = self._sorted_indices[l]                         # (m,)
            left  = cp.searchsorted(sk, q_keys[:, l], side="left")  # (n,)
            right = cp.searchsorted(sk, q_keys[:, l], side="right") # (n,)

            offsets = left[:, None] + offsets_base                  # (n, max_c)
            valid   = offsets < right[:, None]                      # (n, max_c) bool
            offsets = cp.clip(offsets, 0, m - 1)                   # avoid OOB read
            gathered = si[offsets]                                  # (n, max_c)
            gathered = cp.where(valid, gathered, cp.int32(-1))

            col = l * max_c
            all_cands[:, col : col + max_c] = gathered
        prof.stop("7_query/candidate_lookup")

        # ── stage 8 ──────────────────────────────────────────────────────────
        # GPU dedup via double-sort (avoids cp.where index enumeration which
        # requires host-device sync to allocate a variable-length result).
        #
        #  ① map -1 → INT_MAX so invalid entries sort to the end
        #  ② sort each row → equal valid keys adjacent, invalids last
        #  ③ zero out duplicates and invalids (keep first occurrence only)
        #  ④ re-sort → all kept values compacted to the front, zeros at end
        #  ⑤ slice to n_unique columns
        prof.start("8_query/dedup")
        INT_MAX  = int(np.iinfo(np.int32).max)
        tmp      = cp.where(all_cands >= 0, all_cands, cp.int32(INT_MAX))
        tmp      = cp.sort(tmp, axis=1)                             # (n, C)

        is_valid = tmp < INT_MAX
        is_new   = cp.concatenate([
            cp.ones((n, 1), dtype=cp.bool_),
            tmp[:, 1:] != tmp[:, :-1],
        ], axis=1)                                                  # (n, C)

        # zero out duplicates and invalids, then re-sort to compact front
        tmp      = cp.where(is_valid & is_new, tmp, cp.int32(INT_MAX))
        tmp      = cp.sort(tmp, axis=1)                             # valids → front

        n_unique = int((tmp < INT_MAX).sum(axis=1).max())
        n_unique = max(n_unique, k)

        padded_gpu = tmp[:, :n_unique].astype(cp.int32)             # (n, n_unique)
        padded_gpu = cp.where(padded_gpu < INT_MAX, padded_gpu, cp.int32(0))
        prof.stop("8_query/dedup")

        # ── stage 9 ──────────────────────────────────────────────────────────
        # Gather candidate vectors from the stored dataset.
        # padded_gpu: (n, n_unique) indices  →  cand_vecs: (n, n_unique, d)
        prof.start("9_query/gather_vectors")
        flat_idx  = padded_gpu.reshape(-1)                          # (n * n_unique,)
        cand_vecs = self._dataset[flat_idx]                         # (n*n_unique, d)
        cand_vecs = cand_vecs.reshape(n, n_unique, self._d)         # (n, n_unique, d)
        prof.stop("9_query/gather_vectors")

        # ── stage 10 ─────────────────────────────────────────────────────────
        # Re-rank using exact distances.
        # For L2 we use the identity:
        #   ||q - c||^2 = ||q||^2 + ||c||^2 - 2 * q · c
        # The q·c term is a batched GEMM: (n, d) × (n, d, n_unique) → (n, n_unique)
        # which CuPy dispatches to cuBLAS.
        prof.start("10_query/rerank_distances")
        if self.p.metric in ("l2", "sqeuclidean"):
            q_sq  = (Q ** 2).sum(axis=1, keepdims=True)             # (n, 1)
            c_sq  = (cand_vecs ** 2).sum(axis=2)                    # (n, n_unique)
            dot   = cp.einsum("nd,ncd->nc", Q, cand_vecs)           # (n, n_unique)
            dists = cp.maximum(q_sq + c_sq - 2.0 * dot, 0.0)       # (n, n_unique)
        else:                                                       # cosine distance
            eps   = 1e-8
            Q_n   = Q        / (cp.linalg.norm(Q,        axis=1, keepdims=True) + eps)
            C_n   = cand_vecs / (cp.linalg.norm(cand_vecs, axis=2, keepdims=True) + eps)
            dists = 1.0 - cp.einsum("nd,ncd->nc", Q_n, C_n)        # (n, n_unique)
        prof.stop("10_query/rerank_distances")

        # ── stage 11 ─────────────────────────────────────────────────────────
        # Select top-k by distance, return indices only (distances discarded).
        prof.start("11_query/topk_selection")
        actual_k   = min(k, n_unique)
        top_local  = cp.argpartition(dists, actual_k - 1, axis=1)[:, :actual_k]
        top_dists  = cp.take_along_axis(dists, top_local, axis=1)
        sort_order = cp.argsort(top_dists, axis=1)
        top_local  = cp.take_along_axis(top_local, sort_order, axis=1)
        top_global = cp.take_along_axis(padded_gpu, top_local, axis=1)
        prof.stop("11_query/topk_selection")

        prof.report("Query Profile")
        return None, top_global


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def exact_neighbors_cuvs(X: cp.ndarray, Q: cp.ndarray, k: int) -> np.ndarray:
    """
    Exact brute-force k-NN via cuVS (used as ground truth for recall).
    Falls back to a batched CuPy implementation if cuVS is unavailable.
    The batched fallback processes queries in chunks so it never OOMs.
    """
    try:
        import cuvs.neighbors.brute_force as bf
        index   = bf.build(X, metric="sqeuclidean")
        _, nbrs = bf.search(index, Q, k)
        return cp.asnumpy(nbrs)                                     # (n, k) int32
    except (ImportError, Exception) as e:
        print(f"[warn] cuVS unavailable ({e}), using batched CuPy brute-force")
        return _exact_neighbors_cupy_batched(X, Q, k)


def benchmark_exact_neighbors_trivial(
    X: cp.ndarray, Q: cp.ndarray, k: int, query_batch: int = 128
) -> Tuple[np.ndarray, float]:
    """
    Benchmark a trivial brute-force k-NN baseline on GPU.

    The implementation uses the batched CuPy exact path and reports elapsed time
    measured with CUDA events. Returned neighbors are on CPU (NumPy), matching
    the behavior used by recall helpers.
    """
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    nbrs = _exact_neighbors_cupy_batched(X, Q, k, query_batch=query_batch)
    end.record()
    end.synchronize()
    elapsed_ms = float(cp.cuda.get_elapsed_time(start, end))
    return nbrs, elapsed_ms


def _exact_neighbors_cupy_batched(
    X: cp.ndarray, Q: cp.ndarray, k: int, query_batch: int = 128
) -> np.ndarray:
    """
    Memory-safe exact k-NN: process queries in batches of `query_batch`.
    Peak VRAM per batch = query_batch * m * 4 bytes.
    With query_batch=128, m=100k, d=128:  128 * 100000 * 4 = ~51 MB  (fine).
    """
    n   = Q.shape[0]
    out = np.empty((n, k), dtype=np.int32)

    for start in range(0, n, query_batch):
        end   = min(start + query_batch, n)
        q_b   = Q[start:end]                                        # (B, d)

        # ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q·x  — avoids (B, m, d) tensor
        q_sq  = (q_b ** 2).sum(axis=1, keepdims=True)              # (B, 1)
        x_sq  = (X   ** 2).sum(axis=1, keepdims=True).T            # (1, m)
        dot   = q_b @ X.T                                          # (B, m)
        dists = cp.maximum(q_sq + x_sq - 2.0 * dot, 0.0)          # (B, m)

        top   = cp.argpartition(dists, k - 1, axis=1)[:, :k]
        top_d = cp.take_along_axis(dists, top, axis=1)
        order = cp.argsort(top_d, axis=1)
        top   = cp.take_along_axis(top, order, axis=1)
        out[start:end] = cp.asnumpy(top)

    return out


def recall_at_k(true_nbrs: np.ndarray, approx_nbrs: np.ndarray) -> float:
    """Fraction of true top-k neighbors recovered by the approximate method."""
    n, k = true_nbrs.shape
    hits = sum(len(set(true_nbrs[i]) & set(approx_nbrs[i])) for i in range(n))
    return hits / (n * k)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    m:      int = 100_000,
    d:      int = 128,
    n:      int = 1_000,
    k:      int = 10,
    params: Optional[LSHParams] = None,
) -> dict:
    """
    End-to-end benchmark:
      1. Generate random dataset and queries on GPU
      2. Compute exact neighbors with cuVS (ground truth)
      3. Build LSH index and run queries
      4. Compute recall@k
      5. Print per-stage timing tables

    Returns a dict with recall and timing totals.
    """
    if params is None:
        params = LSHParams()

    print("=" * 60)
    print(f"Experiment: m={m:,}  d={d}  n={n:,}  k={k}")
    print(f"LSH params: L={params.n_tables}  K={params.n_projections}  "
          f"w={'auto' if params.bucket_width <= 0 else params.bucket_width}  "
          f"metric={params.metric}")
    print("=" * 60)

    rng = cp.random.default_rng(0)
    X   = rng.standard_normal((m, d), dtype=cp.float32)
    Q   = rng.standard_normal((n, d), dtype=cp.float32)

    print("\n[1/3] Computing exact neighbors (ground truth)...")
    exact = exact_neighbors_cuvs(X, Q, k)

    # Sanity check: brute-force recall against itself must be 1.0
    bf_recall = recall_at_k(exact, exact)
    assert bf_recall == 1.0, f"BUG: brute-force self-recall = {bf_recall} (expected 1.0)"
    print(f"  Brute-force sanity check:  Recall@{k} = {bf_recall:.4f}  ✓")

    print("\n[2/3] Building LSH index...")
    index = LSHIndex(params)
    index.build(X)

    print("\n[3/3] Searching...")
    _, approx = index.search(Q, k)
    approx_np = cp.asnumpy(approx)

    r = recall_at_k(exact, approx_np)
    print(f"\n{'─'*40}")
    print(f"  Brute-force Recall@{k:2d} = 1.0000  (100.0%)  [exact]")
    print(f"  LSH         Recall@{k:2d} = {r:.4f}  ({r*100:.1f}%)")
    print(f"{'─'*40}")

    return {
        "recall": r,
        "build_ms": {s: index.profiler.get(s) for s in index.profiler._totals},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweep  (useful for tuning L, K, w)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_params(
    m: int = 50_000,
    d: int = 128,
    n: int = 500,
    k: int = 10,
):
    """
    Grid search over (n_tables, n_projections, bucket_width).
    Prints a table of recall vs total query time.
    """
    import itertools

    configs = list(itertools.product(
        [5, 10, 20],        # L
        [4, 8],             # K
        [2.0, 4.0, 8.0],    # w
    ))

    rng   = cp.random.default_rng(0)
    X     = rng.standard_normal((m, d), dtype=cp.float32)
    Q     = rng.standard_normal((n, d), dtype=cp.float32)
    exact = exact_neighbors_cuvs(X, Q, k)

    print(f"\n{'L':>4} {'K':>4} {'w':>6}  {'recall':>8}  {'build ms':>10}  {'query ms':>10}")
    print("─" * 55)

    for L, K, w in configs:
        p     = LSHParams(n_tables=L, n_projections=K, bucket_width=w)
        idx   = LSHIndex(p)
        idx.build(X)
        _, approx = idx.search(Q, k)
        r     = recall_at_k(exact, cp.asnumpy(approx))

        build_ms = sum(v for s, v in idx.profiler._totals.items() if "build" in s)
        query_ms = sum(v for s, v in idx.profiler._totals.items() if "query" in s)
        print(f"{L:>4} {K:>4} {w:>6.1f}  {r:>8.4f}  {build_ms:>10.1f}  {query_ms:>10.1f}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GPU LSH benchmark")
    parser.add_argument("--m",                  type=int,   default=100_000)
    parser.add_argument("--d",                  type=int,   default=8)
    parser.add_argument("--n",                  type=int,   default=1_000)
    parser.add_argument("--k",                  type=int,   default=10)
    parser.add_argument("--L",                  type=int,   default=30)
    parser.add_argument("--K",                  type=int,   default=2)
    parser.add_argument("--w",                  type=float, default=0.0,    help="bucket width; 0=auto-tune")
    parser.add_argument("--metric",             type=str,   default="l2",   choices=["l2", "cosine"])
    parser.add_argument("--max_cands",          type=int,   default=2000,   help="max candidates per table")
    args = parser.parse_args()

    run_experiment(
        m      = args.m,
        d      = args.d,
        n      = args.n,
        k      = args.k,
        params = LSHParams(
            n_tables            = args.L,
            n_projections       = args.K,
            bucket_width        = args.w,
            metric              = args.metric,
            max_cands_per_table = args.max_cands,
        ),
    )
