# HNSW-MS

Fast approximate nearest-neighbor search over mass spectra, using a Hierarchical
Navigable Small World (HNSW) graph index with a spectral (modified-cosine, greedy
peak-matching) similarity.

This repository contains the code accompanying the paper *(add citation here)*. It
provides:

- **`ms_hnsw.cpp`** — a standalone C++ program that builds an HNSW index over an
  MGF spectral library, queries it, and can also run an exact brute-force search
  for ground truth.
- **`experiments_cli.py`** — a Python driver that runs a full experiment
  (build → query → baseline → metrics) and writes the results to a CSV.

---

## What the code does

Given a library of mass spectra (in MGF format) and a set of query spectra, the
goal is to find, for each query, its most similar spectra in the library.

- The **exact** way to do this is a brute-force (linear) scan: compare each query
  against every library spectrum. This is accurate but slow, scaling linearly with
  library size.
- **HNSW** builds a graph index over the library once, then answers each query by
  navigating the graph instead of scanning the whole library — typically one to
  three orders of magnitude faster, at the cost of occasionally missing some true
  neighbors (measured as *recall*).

Similarity is a modified cosine with greedy 1-to-1 peak matching within an m/z
tolerance (default 0.01), which is **non-metric**; HNSW is therefore used here as
an empirically effective heuristic, and its accuracy is reported as measured recall.

---

## Requirements

- A C++17 compiler (e.g. `g++` 9 or newer).
- Python 3.8+ with **NumPy** (only for `experiments_cli.py`).

No external C++ libraries are required.

---

## Compiling the C++ program

```bash
g++ -std=c++17 -O3 -o ms_hnsw ms_hnsw.cpp
```

`-O3` matters: it is a numerically intensive inner loop and optimization gives a
large speedup. This produces a single executable, `ms_hnsw`.

---

## Running `ms_hnsw` directly

The program has three modes.

### 1. Build an index

```bash
./ms_hnsw build LIBRARY.mgf INDEX.hnsw M efConstruction
# example:
./ms_hnsw build library.mgf library.mgf.M32.efC400.hnsw 32 400
```

- `M` — number of neighbors kept per node (graph density; higher = more accurate,
  larger index, slower build).
- `efConstruction` — search breadth during construction (higher = better-quality
  graph, slower build).

The index is written to `INDEX.hnsw`.

### 2. Query an index

```bash
./ms_hnsw query INDEX.hnsw QUERIES.mgf K efSearch
# example:
./ms_hnsw query library.mgf.M32.efC400.hnsw queries.mgf 10 256
```

- `K` — number of nearest neighbors to return per query.
- `efSearch` — search breadth at query time (higher = higher recall, slower).

**Output:** for each query, one line to **stdout** listing the `K` library IDs
(0-based indices into the library MGF), space-separated. Timing and statistics
(`HNSW avg per query`, `STATS avg_dist_calls=… avg_visited=…`) are printed to
**stderr**. The reported time is search-only and excludes file loading.

### 3. Exact brute-force baseline (ground truth)

```bash
./ms_hnsw baseline LIBRARY.mgf QUERIES.mgf K
# example:
./ms_hnsw baseline library.mgf queries.mgf 10
```

Compares every query against every library spectrum. **Output:** the same format
as `query` (one line of `K` IDs per query on stdout), with `Brute-force avg per
query` timing on stderr. Use this to compute recall for the approximate search.

**ID convention:** library IDs are 0-based positions in the order spectra appear
in the library MGF, and are identical between `query` and `baseline`, so their
outputs can be compared directly.

---

## Running the full experiment (`experiments_cli.py`)

`experiments_cli.py` automates a complete run: it (optionally) samples queries,
builds the index if it does not already exist, runs the HNSW query, runs the exact
baseline, computes recall / precision / Hit@1 / AP / NDCG against the baseline, and
appends one row of results to a CSV.

```bash
python experiments_cli.py \
    --library library.mgf \
    --M 32 --efC 400 --efS 256 \
    --K 10 --num_queries 100 --seed 42 \
    --index_dir indices \
    --hnsw_bin ./ms_hnsw \
    --output results/results.csv \
    --save_queries queries/queries_M32_efC400_efS256.mgf
```

Key arguments:

| Argument | Meaning |
|---|---|
| `--library` | Library MGF file. |
| `--M`, `--efC`, `--efS` | HNSW parameters (build density, build breadth, query breadth). |
| `--K` | Neighbors per query (default 10). |
| `--num_queries` | Number of queries to sample from the library (default 100). |
| `--seed` | Random seed for reproducible sampling (default 42). |
| `--index_dir` | Where indices are stored / looked up. |
| `--hnsw_bin` | Path to the compiled `ms_hnsw` binary. |
| `--output` | CSV file to append results to. |
| `--save_queries` / `--load_queries` | Save sampled queries, or reuse a fixed query set. |
| `--skip_baseline` | Skip the exact baseline (fast, but no recall metrics). |

By default queries are sampled at random from the library; pass `--load_queries`
with an external MGF to query spectra that are **not** in the library.

---

## Why Python is needed

The C++ program is deliberately minimal: it only builds an index, queries it, and
runs the brute-force baseline, printing raw neighbor IDs. It does **not** sample
queries, compute recall, parse timings, or record results.

`experiments_cli.py` is the orchestration and evaluation layer around it. It:

- samples (or loads) the query set and handles MGF I/O and reproducibility (seed),
- calls the `ms_hnsw` binary for build, query, and baseline,
- parses the binary's stdout (neighbor IDs) and stderr (internal search timings),
- computes the accuracy metrics (recall@K, precision@K, Hit@1, AP@K, NDCG@K) by
  comparing HNSW results against the exact baseline,
- computes the speedup from the internal (search-only) timings, and
- writes everything as a row to a results CSV.

In short: **C++ does the fast, low-level search; Python drives the experiment and
turns raw IDs into the numbers reported in the paper.** The C++ program is fully
usable on its own if you only need index build/query/baseline and will do your own
scoring.

---

## Input format

Standard MGF. Each spectrum is a `BEGIN IONS` … `END IONS` block; peak lines are
`m/z intensity` pairs. Metadata lines such as `PEPMASS=` and `TITLE=` are read;
other headers are ignored for scoring. Peaks are sorted by m/z internally.

---

## Reproducing the paper's numbers

The two result CSVs in this repository (`results_recall_ALL_GNPS.csv`,
`results_recall_centers_combined.csv`) contain the per-configuration recall and
speedup values reported in the paper. Each row is one `(library, M, efC, efS)`
setting over 100 queries. Re-running `experiments_cli.py` with the same
`--seed`, `--num_queries`, and parameters reproduces them (up to hardware-dependent
timing).

---

## License

*(add your license here, e.g. MIT)*

## Citation

*(add BibTeX / paper reference here)*
