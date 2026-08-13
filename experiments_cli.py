#!/usr/bin/env python3
# experiments_cli.py (v3)
#
# Fixes vs v2:
# 1) Correct recall/AP/NDCG relevance set: use BASELINE top-K only (true recall@K)
# 2) Parse internal per-query timing from C++ stderr (HNSW + baseline) for consistent speedups
# 3) Keep wall-clock per-query timing too (process overhead), but report internal timing when available
# 4) Slightly safer parsing + clearer outputs
#
# Usage example:
#   python experiments_cli.py \
#       --library ALL_GNPS.mgf \
#       --M 32 --efC 400 --efS 256 \
#       --K 10 --num_queries 100 \
#       --seed 42 \
#       --index_dir indices \
#       --hnsw_bin ./ms_hnsw \
#       --output results/results_ALL_GNPS.csv \
#       --save_queries queries/queries_ALL_GNPS_M32_efC400_efS256.mgf

import os
import argparse
import random
import time
import subprocess
import numpy as np
import csv
from typing import List, Tuple, Optional, Dict

# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="HNSW spectral search experiments with C++ baseline (fixed metrics + timing)"
    )
    parser.add_argument("--library", required=True,
                        help="Path to library .mgf file")
    parser.add_argument("--M", type=int, required=True,
                        help="HNSW max connections parameter")
    parser.add_argument("--efC", type=int, required=True,
                        help="HNSW efConstruction parameter")
    parser.add_argument("--efS", type=int, required=True,
                        help="HNSW efSearch parameter")
    parser.add_argument("--K", type=int, default=10,
                        help="Top-K neighbors (default: 10)")
    parser.add_argument("--num_queries", type=int, default=100,
                        help="Number of random queries (default: 100)")
    parser.add_argument("--max_lib_spectra", type=int, default=0,
                        help="Max spectra to load for query sampling (0=all). NOTE: sampling only.")
    parser.add_argument("--index_dir", default="indices",
                        help="Directory for index files")
    parser.add_argument("--hnsw_bin", default="./ms_hnsw",
                        help="Path to HNSW binary")
    parser.add_argument("--output", default="hnsw_results.csv",
                        help="Output CSV file (append)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_queries", type=str, default=None,
                        help="Path to save sampled queries MGF (for replication)")
    parser.add_argument("--load_queries", type=str, default=None,
                        help="Path to load pre-existing queries MGF (for replication)")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline computation (faster, but no recall metrics)")
    parser.add_argument("--baseline_timeout", type=int, default=7200,
                        help="Timeout for baseline in seconds (default: 7200 = 2 hours)")
    return parser.parse_args()

# --------------------------------------------------------------
# MGF I/O
# --------------------------------------------------------------

def count_spectra_in_mgf(path: str) -> int:
    """Count spectra without loading them into memory."""
    count = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("END IONS"):
                count += 1
    return count


def read_mgf(path: str, max_spectra: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Read MGF file -> list of (mz, intensity) np arrays."""
    spectra: List[Tuple[np.ndarray, np.ndarray]] = []
    with open(path, "r") as f:
        current_mz: List[float] = []
        current_int: List[float] = []
        in_ions = False
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("BEGIN IONS"):
                in_ions = True
                current_mz = []
                current_int = []
            elif line.startswith("END IONS"):
                in_ions = False
                if current_mz:
                    arr = sorted(zip(current_mz, current_int), key=lambda x: x[0])
                    mz = np.array([a[0] for a in arr], dtype=float)
                    inten = np.array([a[1] for a in arr], dtype=float)
                    spectra.append((mz, inten))
                    if max_spectra is not None and max_spectra > 0 and len(spectra) >= max_spectra:
                        break
            elif in_ions and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        m = float(parts[0])
                        it = float(parts[1])
                    except ValueError:
                        continue
                    current_mz.append(m)
                    current_int.append(it)
    return spectra


def write_queries_mgf(queries: List[Tuple[np.ndarray, np.ndarray]],
                      path: str,
                      indices: Optional[List[int]] = None):
    """Write queries to MGF file with optional original indices in title."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for i, (mz, inten) in enumerate(queries):
            f.write("BEGIN IONS\n")
            if indices is not None:
                f.write(f"TITLE=query_{i}_origidx_{indices[i]}\n")
            else:
                f.write(f"TITLE=query_{i}\n")
            for m, it in zip(mz, inten):
                f.write(f"{m} {it}\n")
            f.write("END IONS\n")


def sample_queries_from_library(lib_spectra: List[Tuple[np.ndarray, np.ndarray]],
                               num_queries: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """Sample random queries from library, return (queries, original_indices)."""
    n = len(lib_spectra)
    k = min(num_queries, n)
    indices = random.sample(range(n), k)
    queries = [lib_spectra[i] for i in indices]
    return queries, indices

# --------------------------------------------------------------
# Metrics (FIXED)
# --------------------------------------------------------------

def compute_metrics_for_query(hnsw_ids: List[int],
                              baseline_ids: List[int],
                              K: int) -> Dict[str, float]:
    """
    Compute recall@K, precision@K, hit@1 (exact + relaxed), AP@K, NDCG@K.

    IMPORTANT FIX:
      Relevance set is BASELINE TOP-K (not baseline full list).
    """
    L = hnsw_ids[:K]
    R = baseline_ids[:K]
    set_rel = set(R)

    if not L:
        return {
            "recall_at_K": 0.0,
            "precision_at_K": 0.0,
            "hit_at_1_exact": 0.0,
            "hit_at_1_relaxed": 0.0,
            "ap_at_K": 0.0,
            "ndcg_at_K": 0.0,
        }

    inter = len(set(L) & set_rel)
    recall = inter / float(len(R)) if R else 0.0
    precision = inter / float(len(L)) if L else 0.0

    hit_exact = 1.0 if (L and R and L[0] == R[0]) else 0.0
    hit_relaxed = 1.0 if (L and L[0] in set_rel) else 0.0

    # Average Precision@K (binary relevance on R)
    ap = 0.0
    num_rel_so_far = 0
    for i, id_ in enumerate(L):
        if id_ in set_rel:
            num_rel_so_far += 1
            ap += num_rel_so_far / float(i + 1)
    ap = ap / float(len(R)) if R else 0.0

    # NDCG@K (binary relevance on R)
    def dcg(scores: List[float]) -> float:
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores))

    rel_scores = [1.0 if id_ in set_rel else 0.0 for id_ in L]
    dcg_val = dcg(rel_scores)
    ideal_scores = [1.0] * min(len(R), len(L))
    ideal_dcg = dcg(ideal_scores) if ideal_scores else 0.0
    ndcg = dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0

    return {
        "recall_at_K": recall,
        "precision_at_K": precision,
        "hit_at_1_exact": hit_exact,
        "hit_at_1_relaxed": hit_relaxed,
        "ap_at_K": ap,
        "ndcg_at_K": ndcg,
    }


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metrics across all queries."""
    keys = list(all_metrics[0].keys())
    return {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

# --------------------------------------------------------------
# Parsing helpers (timings + stats from C++ stderr)
# --------------------------------------------------------------

def parse_hnsw_stderr(stderr: str) -> Dict[str, Optional[float]]:
    """
    Parse:
      - "HNSW avg per query: X ms"
      - "STATS avg_dist_calls=... avg_visited=..."
    """
    out: Dict[str, Optional[float]] = {
        "hnsw_avg_ms_internal": None,
        "avg_dist_calls": None,
        "avg_visited": None,
    }
    for raw in stderr.splitlines():
        line = raw.strip()

        if line.startswith("HNSW avg per query:"):
            # e.g. "HNSW avg per query: 8.7 ms"
            try:
                out["hnsw_avg_ms_internal"] = float(line.split(":")[1].strip().split()[0])
            except Exception:
                pass

        if line.startswith("STATS"):
            # e.g. "STATS avg_dist_calls=4313 avg_visited=1234"
            parts = line.split()
            for p in parts[1:]:
                if "=" in p:
                    key, val = p.split("=", 1)
                    try:
                        if key == "avg_dist_calls":
                            out["avg_dist_calls"] = float(val)
                        elif key == "avg_visited":
                            out["avg_visited"] = float(val)
                    except Exception:
                        pass
    return out


def parse_baseline_stderr(stderr: str) -> Dict[str, Optional[float]]:
    """
    Parse:
      - "Brute-force avg per query: X ms"
    """
    out: Dict[str, Optional[float]] = {"baseline_avg_ms_internal": None}
    for raw in stderr.splitlines():
        line = raw.strip()
        if line.startswith("Brute-force avg per query:"):
            # e.g. "Brute-force avg per query: 123.45 ms"
            try:
                out["baseline_avg_ms_internal"] = float(line.split(":")[1].strip().split()[0])
            except Exception:
                pass
    return out

# --------------------------------------------------------------
# HNSW helpers
# --------------------------------------------------------------

def build_index_if_needed(lib_path: str, M: int, efC: int,
                          index_dir: str, hnsw_bin: str) -> str:
    """Build HNSW index if it doesn't exist, return path to index."""
    os.makedirs(index_dir, exist_ok=True)
    base = os.path.basename(lib_path)
    idx_name = f"{base}.M{M}.efC{efC}.hnsw"
    idx_path = os.path.join(index_dir, idx_name)

    if os.path.exists(idx_path):
        print(f"Index already exists: {idx_path}")
        return idx_path

    print(f"Building index {idx_path} ...")
    cmd = [hnsw_bin, "build", lib_path, idx_path, str(M), str(efC)]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    t1 = time.perf_counter()

    print(f"Index built in {t1 - t0:.1f} seconds")
    if proc.stderr:
        print(proc.stderr.decode(errors="replace"))

    return idx_path


def run_hnsw_query(hnsw_bin: str, index_path: str,
                   query_mgf_path: str, K: int, efSearch: int) -> Dict[str, object]:
    """
    Run HNSW queries.

    Returns dict containing:
      results: List[List[int]]
      wall_ms_per_query: float
      internal_ms_per_query: Optional[float]
      avg_dist_calls: Optional[float]
      avg_visited: Optional[float]
      stderr: str
    """
    cmd = [hnsw_bin, "query", index_path, query_mgf_path, str(K), str(efSearch)]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    t1 = time.perf_counter()

    stdout = proc.stdout.decode(errors="replace")
    stderr = proc.stderr.decode(errors="replace")

    results: List[List[int]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids = [int(x) for x in line.split()]
        except ValueError:
            continue
        results.append(ids)

    q = max(len(results), 1)
    wall_ms_per_query = (t1 - t0) * 1000.0 / q

    parsed = parse_hnsw_stderr(stderr)

    return {
        "results": results,
        "wall_ms_per_query": wall_ms_per_query,
        "internal_ms_per_query": parsed["hnsw_avg_ms_internal"],
        "avg_dist_calls": parsed["avg_dist_calls"],
        "avg_visited": parsed["avg_visited"],
        "stderr": stderr,
    }


def run_cpp_baseline(hnsw_bin: str, lib_path: str,
                     query_mgf_path: str, K: int,
                     timeout: int = 7200) -> Dict[str, object]:
    """
    Run C++ brute-force baseline.

    Returns dict containing:
      results: Optional[List[List[int]]]
      wall_ms_per_query: Optional[float]
      internal_ms_per_query: Optional[float]
      stderr: str
      ok: bool
    """
    cmd = [hnsw_bin, "baseline", lib_path, query_mgf_path, str(K)]
    print(f"Running C++ baseline (timeout={timeout}s)...")

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=True
        )
        t1 = time.perf_counter()

        stdout = proc.stdout.decode(errors="replace")
        stderr = proc.stderr.decode(errors="replace")
        if stderr:
            print(stderr)

        results: List[List[int]] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ids = [int(x) for x in line.split()]
            except ValueError:
                continue
            results.append(ids)

        q = max(len(results), 1)
        wall_ms_per_query = (t1 - t0) * 1000.0 / q

        parsed = parse_baseline_stderr(stderr)

        return {
            "ok": True,
            "results": results,
            "wall_ms_per_query": wall_ms_per_query,
            "internal_ms_per_query": parsed["baseline_avg_ms_internal"],
            "stderr": stderr,
        }

    except subprocess.TimeoutExpired:
        print(f"WARNING: Baseline timed out after {timeout} seconds")
        return {
            "ok": False,
            "results": None,
            "wall_ms_per_query": None,
            "internal_ms_per_query": None,
            "stderr": "",
        }
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace") if e.stderr else ""
        print(f"WARNING: Baseline failed: {e}")
        if err:
            print(f"stderr:\n{err}")
        return {
            "ok": False,
            "results": None,
            "wall_ms_per_query": None,
            "internal_ms_per_query": None,
            "stderr": err,
        }

# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    lib_path = args.library
    M = args.M
    efC = args.efC
    efS = args.efS
    K = args.K
    num_queries = args.num_queries
    max_spectra = args.max_lib_spectra if args.max_lib_spectra > 0 else None

    print("=" * 60)
    print("HNSW Experiment (v3)")
    print("=" * 60)
    print(f"Library:       {lib_path}")
    print(f"Parameters:    M={M}, efC={efC}, efS={efS}, K={K}")
    print(f"Num queries:   {num_queries}")
    print(f"Seed:          {args.seed}")
    print(f"Skip baseline: {args.skip_baseline}")
    print()

    # Count spectra first
    print(f"Counting spectra in {lib_path}...")
    t0 = time.perf_counter()
    total_spectra = count_spectra_in_mgf(lib_path)
    t1 = time.perf_counter()
    print(f"Library contains {total_spectra} spectra (counted in {t1 - t0:.1f}s)")

    # Load queries or sample new ones
    if args.load_queries:
        print(f"Loading pre-existing queries from {args.load_queries}...")
        queries = read_mgf(args.load_queries)
        if not queries:
            raise RuntimeError(f"No queries found in {args.load_queries}")
        print(f"Loaded {len(queries)} queries")
        query_indices = None
        query_path = args.load_queries
    else:
        print(f"Loading library {lib_path} for query sampling...")
        t0 = time.perf_counter()
        lib_spectra = read_mgf(lib_path, max_spectra)
        t1 = time.perf_counter()
        if not lib_spectra:
            raise RuntimeError(f"No spectra loaded from {lib_path}")
        print(f"Loaded {len(lib_spectra)} spectra in {t1 - t0:.1f}s")

        queries, query_indices = sample_queries_from_library(lib_spectra, num_queries)
        print(f"Sampled {len(queries)} queries")

        if args.save_queries:
            write_queries_mgf(queries, args.save_queries, query_indices)
            print(f"Saved queries to {args.save_queries}")
            query_path = args.save_queries
        else:
            query_path = f"tmp_queries_M{M}_efC{efC}_efS{efS}_seed{args.seed}.mgf"
            write_queries_mgf(queries, query_path, query_indices)
            print(f"Wrote queries to {query_path}")

    # Build index
    idx_path = build_index_if_needed(lib_path, M, efC, args.index_dir, args.hnsw_bin)

    # Run HNSW query
    print()
    print(f"Running HNSW queries with efS={efS}...")
    hnsw = run_hnsw_query(args.hnsw_bin, idx_path, query_path, K, efS)
    hnsw_res = hnsw["results"]
    print(f"HNSW: {len(hnsw_res)} queries completed")

    # Prefer internal timing if available
    hnsw_ms = hnsw["internal_ms_per_query"] if hnsw["internal_ms_per_query"] is not None else hnsw["wall_ms_per_query"]
    print(f"HNSW time: {hnsw_ms:.2f} ms/query "
          f"({'internal' if hnsw['internal_ms_per_query'] is not None else 'wall'})")

    if hnsw["avg_dist_calls"] is not None:
        print(f"HNSW avg distance calls: {hnsw['avg_dist_calls']:.1f}")
    if hnsw["avg_visited"] is not None:
        print(f"HNSW avg visited nodes:  {hnsw['avg_visited']:.1f}")

    # Run baseline + compute metrics
    baseline_res = None
    baseline_ms = None
    avg_metrics = None
    speedup = None

    if not args.skip_baseline:
        print()
        baseline = run_cpp_baseline(args.hnsw_bin, lib_path, query_path, K, args.baseline_timeout)
        if baseline["ok"] and baseline["results"] is not None:
            baseline_res = baseline["results"]
            print(f"Baseline: {len(baseline_res)} queries completed")

            baseline_ms = baseline["internal_ms_per_query"] if baseline["internal_ms_per_query"] is not None else baseline["wall_ms_per_query"]
            print(f"Baseline time: {baseline_ms:.2f} ms/query "
                  f"({'internal' if baseline['internal_ms_per_query'] is not None else 'wall'})")

            if hnsw_ms and baseline_ms:
                speedup = baseline_ms / hnsw_ms
                print(f"Speedup: {speedup:.2f}x")

            if len(hnsw_res) == len(baseline_res):
                per_query_metrics = [
                    compute_metrics_for_query(h, b, K)
                    for h, b in zip(hnsw_res, baseline_res)
                ]
                avg_metrics = aggregate_metrics(per_query_metrics)

                print()
                print("Metrics vs baseline (FIXED relevance = baseline top-K):")
                for k, v in avg_metrics.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"WARNING: Result count mismatch: HNSW={len(hnsw_res)}, baseline={len(baseline_res)}")
        else:
            print("Baseline failed or timed out - no recall metrics available")
    else:
        print()
        print("Skipping baseline (--skip_baseline)")

    if avg_metrics is None:
        avg_metrics = {
            "recall_at_K": None,
            "precision_at_K": None,
            "hit_at_1_exact": None,
            "hit_at_1_relaxed": None,
            "ap_at_K": None,
            "ndcg_at_K": None,
        }

    # Collect result row
    result = {
        "library": os.path.basename(lib_path),
        "num_spectra": total_spectra,
        "M": M,
        "efC": efC,
        "efS": efS,
        "K": K,
        "num_queries": len(hnsw_res),
        "seed": args.seed,

        # timings (ms/query)
        "hnsw_time_ms": float(hnsw_ms) if hnsw_ms is not None else None,
        "hnsw_time_kind": ("internal" if hnsw["internal_ms_per_query"] is not None else "wall"),
        "baseline_time_ms": float(baseline_ms) if baseline_ms is not None else None,
        "baseline_time_kind": (None if baseline_ms is None else "internal" if (not args.skip_baseline and baseline_res is not None and baseline_ms is not None) else "wall"),
        "speedup": float(speedup) if speedup is not None else None,

        "avg_dist_calls": hnsw["avg_dist_calls"],
        "avg_visited_nodes": hnsw["avg_visited"],
    }
    result.update(avg_metrics)

    # Print summary
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    for k, v in result.items():
        if v is None:
            print(f"  {k}: N/A")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Write CSV (append if exists, create if not)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"\nAppended results to {args.output}")

    # Cleanup temp query file (only if we created it)
    if not args.save_queries and not args.load_queries:
        if os.path.exists(query_path):
            os.remove(query_path)
            print(f"Cleaned up temp file: {query_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()