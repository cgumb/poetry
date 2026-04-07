# Poetry GP: Learning Poetic Preference with Gaussian Processes and HPC

An interactive poetry recommendation system that demonstrates HPC principles through **Gaussian process active learning**.

## What It Does

**Learn a reader's taste, then choose what to show next:**
- User rates poems → System infers preference function → Recommends next poem
- Balances **exploitation** (show likely favorites) vs **exploration** (ask informative questions)

## The Method

### From Ridge Regression to Gaussian Processes

Starting from Bayesian linear regression:
```
y = Xβ + ε,  β ~ N(0, τ²I)
```

The **dual form** leads naturally to kernel methods:
```
ŷ* = k*ᵀ (K + λI)⁻¹ y
```

Generalizing to **Gaussian processes** with RBF kernel:
```
k(x,x') = σ²f exp(-||x-x'||² / 2ℓ²)
```

This gives us:
- **Posterior mean**: μ(x) = expected preference
- **Posterior variance**: σ²(x) = uncertainty
- **Active learning**: Choose next poem by balancing mean and variance

### Computational Bottlenecks

| Operation | Complexity | Scaling Variable |
|-----------|-----------|------------------|
| **GP Fit** | O(m³) | m = rated poems |
| **Score (mean)** | O(nmd) + O(nm) | n = candidates |
| **Score (variance)** | O(nm²) | Expensive! |

For **m = 1000 rated poems, n = 85k candidates**:
- Fit: 1 billion FLOPs (Cholesky factorization)
- Score with variance: 85 billion FLOPs (triangular solves)

## HPC Solutions

### Automatic Backend Selection

The system **automatically chooses** optimal backends based on problem size:

```python
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    fit_backend="auto",      # → native_lapack, python, or scalapack
    score_backend="auto",    # → gpu, native_lapack, or python
)
```

### Fit Backends (O(m³) Cholesky)

| Backend | Method | When to Use |
|---------|--------|-------------|
| **Python** | SciPy (LAPACK) | Baseline, m < 5k |
| **native_lapack** | PyBind11 + LAPACK | Single-node, m < 5k (instant) |
| **native_reference** | ScaLAPACK MPI | Distributed, m > 10k |

**Crossover**: ScaLAPACK beats Python at m ≈ 7k-10k with 8-16 processes

**Key optimization**: Distributed kernel assembly (Milestone 1B)
- Broadcast features (30MB) instead of scatter matrix (800MB)
- Each rank computes its block-cyclic tiles in parallel
- BLAS DGEMM optimization: 20-40× speedup over naive assembly

### Score Backends (O(nm²) variance)

| Backend | Method | Speedup (m=1000) |
|---------|--------|------------------|
| **Python** | NumPy + BLAS | Baseline |
| **native_lapack** | PyBind11 + multi-threaded BLAS | 1.1-1.2× |
| **GPU** | CuPy/CUDA | 3-4.6× |

**Key insight**: Variance computation dominates scoring time for large n
- Mean only: O(nm) - fast
- With variance: O(nm²) - expensive but needed for exploration

## Quick Start

### Setup

```bash
# CPU-only (default)
bash scripts/bootstrap_env.sh
source scripts/activate_env.sh

# With GPU support (optional, on GPU node)
bash scripts/bootstrap_env.sh --gpu
source scripts/activate_env.sh --gpu

# Verify
python scripts/app/check_env.py
```

### Get Preprocessed Data (Students)

**Skip 2+ hours of embedding generation** by using shared preprocessed data:

```bash
# Symlink preprocessed data from shared location
bash scripts/setup_shared_data.sh
```

This creates symlinks to:
- `poems.parquet` - 85k poems with imputed poet names
- `embeddings.npy` - 384-dim poem embeddings
- `proj2d.npy` - 2D UMAP projection
- `poet_centroids.npy` - Poet centroid embeddings

Now you can skip to [Interactive Session](#interactive-session) or [Benchmarks](#benchmarks).

See [`docs/SHARED_DATA_SETUP.md`](docs/SHARED_DATA_SETUP.md) for details.

### Build Corpus and Embeddings (Optional - Advanced Users)

**Only needed if regenerating data from scratch** (takes ~2 hours):

```bash
# Download and deduplicate poems
python scripts/app/build_poetry_corpus.py

# Generate embeddings (~2 hours for 85k poems)
python scripts/app/embed_poems.py \
  --input data/poems.parquet \
  --output data/embeddings.npy

# Project to 2D for visualization
python scripts/app/project_poems_2d.py \
  --input data/embeddings.npy \
  --output data/proj2d.npy

# Build poet centroids
python scripts/app/build_poet_centroids.py \
  --poems data/poems.parquet \
  --embeddings data/embeddings.npy \
  --output-parquet data/poet_centroids.parquet \
  --output-npy data/poet_centroids.npy
```

See [`docs/CORPUS_BUILDING.md`](docs/CORPUS_BUILDING.md) for full data pipeline details.

### Interactive Session

```bash
python scripts/app/interactive_cli.py
```

**Features**:
- Rate poems with numeric scores
- **Exploit** (`e`): Get recommendations (max mean, UCB, Thompson sampling)
- **Explore** (`x`): Strategic queries (max variance, spatial diversity, expected improvement)
- **Config** (`c`): Choose acquisition functions and backends
- **Search** (`s`): Find specific poems/poets
- Multi-user support with session persistence

**Performance** (m=1000, n=85k):
```
Fit:     0.03s  (native_lapack)
Score:   0.60s  (GPU)
Select:  0.001s (max_variance)
Total:   0.63s per iteration
```

### Acquisition Functions

**Exploitation** (recommend likely favorites):
- `max_mean`: argmax μ(x) - simple, fast
- `ucb`: argmax μ(x) + β·σ(x) - **recommended** (balances quality + confidence)
- `thompson`: Sample from posterior - Bayesian optimal

**Exploration** (ask informative questions):
- `max_variance`: argmax σ²(x) - information-optimal (O(n))
- `expected_improvement`: Classic Bayesian optimization (O(n))
- `spatial_variance`: Spatially diverse selection (O(n²), only for n < 10k)

See [`docs/ACQUISITION_FUNCTIONS.md`](docs/ACQUISITION_FUNCTIONS.md) for detailed tradeoffs.

## Benchmarking

### Pedagogical Benchmark Suite (Recommended)

For **teaching HPC concepts**, run the comprehensive benchmark suite that validates theory and reveals bottlenecks:

```bash
# Run all pedagogical benchmarks (1-2 hours)
sbatch scripts/pedagogical_benchmarks.slurm
```

This measures:
- **Scaling theory validation**: Verify O(m³), O(m²), O(n) on log-log plots
- **Time breakdown analysis**: Where does time go? (fit vs score vs overhead)
- **Overhead vs compute**: When does parallelization pay off?

Generates CSV data and publication-quality figures automatically.

**See**: [`docs/RUNNING_BENCHMARKS.md`](docs/RUNNING_BENCHMARKS.md) for detailed instructions and interpretation.

### Individual Benchmarks

```bash
# Complexity validation
python scripts/bench_scaling_theory.py \
  --fit-backend python \
  --output-csv results/scaling.csv

# Time breakdown
python scripts/bench_time_breakdown.py \
  --m-values 100 500 1000 5000 \
  --n-values 10000 50000 \
  --output-csv results/breakdown.csv

# Backend comparison
python scripts/bench_overhead_crossover.py \
  --backends python native_lapack native_reference \
  --output-csv results/crossover.csv

# Visualize results
python scripts/visualize_scaling.py results/*.csv --output-dir figures/
```

**See**: [`docs/BENCHMARKING_GUIDE.md`](docs/BENCHMARKING_GUIDE.md) for comprehensive benchmarking guide.

## HPC Principles Demonstrated

### 1. Algorithmic Complexity Analysis
- Identifying O(m³) and O(nm²) bottlenecks through profiling
- Understanding when distributed methods pay off

### 2. Backend Abstraction
- Single API, multiple implementations (Python, LAPACK, ScaLAPACK, GPU)
- Automatic selection based on problem size and hardware

### 3. Distributed Memory (ScaLAPACK)
- Block-cyclic matrix layout
- Process grid topology
- Communication vs computation tradeoffs
- Distributed kernel assembly (Milestone 1B)

### 4. GPU Acceleration
- Offloading O(nm²) triangular solves to GPU
- Memory transfer overhead vs compute speedup
- When GPU helps: m > 500 (3-4.6× faster)

### 5. Lazy Evaluation
- Optional variance computation (skip when not needed)
- Acquisition function determines workload

### 6. Performance Engineering
- BLAS optimization (DGEMM for kernel assembly)
- Blocking for cache efficiency
- Multi-threaded BLAS for CPU parallelism

## Project Structure

```
src/poetry_gp/
  gp_exact.py              # Exact GP: kernel, Cholesky, solve
  kernel.py                # RBF kernel with BLAS optimization
  backends/
    blocked.py             # Main API: run_blocked_step()
    backend_selection.py   # Automatic backend choice
    native_lapack.py       # PyBind11 LAPACK (single-node)
    scalapack_fit.py       # ScaLAPACK MPI (distributed)
    gpu_scoring.py         # CuPy CUDA (GPU)

native/
  scalapack_gp_fit.cpp     # C++ ScaLAPACK implementation
  pybind11_lapack.cpp      # PyBind11 LAPACK bindings

scripts/
  app/interactive_cli.py   # Interactive recommendation
  bench_step.py            # Fit benchmarks
  bench_scoring.py         # Score benchmarks
  *.slurm                  # Cluster job scripts

docs/
  METHOD_NARRATIVE.md              # Mathematical foundation
  CURRENT_ROADMAP.md               # Development roadmap
  BACKEND_SELECTION.md             # Backend guide
  ACQUISITION_FUNCTIONS.md         # Exploration strategies
  BENCHMARKING_GUIDE.md            # Performance analysis
  CORPUS_BUILDING.md               # Data pipeline
  NATIVE_HPC_ROADMAP.md            # HPC optimizations
  MILESTONE_1B_DESIGN.md           # Distributed assembly
  POET_SELECTION_AND_IMPUTATION.md # Data quality
```

## Key Documentation

**Getting Started**:
- [`docs/BACKEND_SELECTION.md`](docs/BACKEND_SELECTION.md) - When to use which backend
- [`docs/ACQUISITION_FUNCTIONS.md`](docs/ACQUISITION_FUNCTIONS.md) - Exploration strategies

**Mathematical Foundation**:
- [`docs/METHOD_NARRATIVE.md`](docs/METHOD_NARRATIVE.md) - From ridge regression to GP

**HPC Implementation**:
- [`docs/NATIVE_HPC_ROADMAP.md`](docs/NATIVE_HPC_ROADMAP.md) - HPC optimization roadmap
- [`docs/MILESTONE_1B_DESIGN.md`](docs/MILESTONE_1B_DESIGN.md) - Distributed kernel assembly
- [`docs/SCALAPACK_BACKEND.md`](docs/SCALAPACK_BACKEND.md) - ScaLAPACK details

**Data and Benchmarking**:
- [`docs/CORPUS_BUILDING.md`](docs/CORPUS_BUILDING.md) - Data pipeline
- [`docs/BENCHMARKING_GUIDE.md`](docs/BENCHMARKING_GUIDE.md) - Performance analysis

**Development**:
- [`docs/CURRENT_ROADMAP.md`](docs/CURRENT_ROADMAP.md) - Current status and next steps

## Current Status

**Completed**:
- ✅ Exact GP with automatic backend selection
- ✅ PyBind11 LAPACK integration (zero subprocess overhead)
- ✅ GPU scoring with CuPy (3-4.6× speedup)
- ✅ ScaLAPACK distributed fitting (Milestone 1B: distributed kernel assembly)
- ✅ Hyperparameter optimization with analytic gradients
- ✅ Interactive CLI with rich UI and multi-user support
- ✅ Advanced acquisition functions (UCB, Thompson, spatial diversity)
- ✅ Comprehensive benchmarking infrastructure

**Next Priorities** (see [`docs/CURRENT_ROADMAP.md`](docs/CURRENT_ROADMAP.md)):
1. Warm-start hyperparameter optimization (5× faster in interactive sessions)
2. Lazy variance computation (85× reduction for exploration)
3. Analytic gradients for HP optimization (30-50% fewer iterations)

## Pedagogical Value

This project demonstrates:
- **Computational complexity**: O(m³) and O(nm²) bottlenecks in practice
- **Distributed computing**: ScaLAPACK, process grids, block-cyclic layout
- **GPU acceleration**: When and why GPU helps (memory-bound vs compute-bound)
- **Algorithmic tradeoffs**: Exact vs approximate, exploit vs explore
- **Performance engineering**: Profiling, BLAS optimization, backend selection
- **Active learning**: Bayesian optimization in an authentic application

---

**A CS 2050 project demonstrating HPC principles through interactive poetry recommendation.**
