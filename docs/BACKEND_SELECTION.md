# Choosing the Right Backend

## Why Multiple Backends?

GP fitting and scoring have different performance characteristics depending on problem size:

**Small problems** (m < 1000):
- Computation is fast (milliseconds)
- **Overhead dominates**: Process spawning, file I/O, GPU transfers
- Simple sequential code often wins

**Large problems** (m > 10,000):
- Computation is slow (seconds to minutes)
- **Compute dominates**: O(m³) and O(nm²) operations
- Parallelization and GPU acceleration pay off

**The challenge**: What's optimal for m=100 is terrible for m=20,000.

---

## The Backends

### 1. Python (scipy)

**What it is**: NumPy/SciPy with optimized LAPACK underneath.

**When it works well**:
- Always available (no special dependencies)
- Good for medium problems (m = 1k-10k)
- Multi-threaded BLAS gives some parallelism

**When it struggles**:
- Large m (>10k): Single-node memory and compute limits
- Very large n: Sequential scoring gets expensive

**Typical use**: Development, medium-scale problems, baseline comparison.

### 2. native_lapack (PyBind11)

**What it is**: Direct in-memory LAPACK via C++ (no subprocess, no file I/O).

**Why it exists**: Eliminate overhead for small problems.

Python's subprocess spawn for ScaLAPACK:
```
import subprocess
subprocess.run(["mpirun", ...])  # ~160ms overhead
```

For m=500: Overhead (160ms) > Computation (10ms) → 16× slowdown!

PyBind11 solution:
```cpp
// C++ function callable directly from Python (zero overhead)
py::dict fit_gp_lapack(py::array_t<double> K, py::array_t<double> y) {
    LAPACK_dpotrf(...);  // Direct LAPACK call
    return result;
}
```

**When it works well**:
- Small to medium m (<5k)
- Interactive sessions (fast iteration)
- Single-node available

**Typical use**: Interactive CLI, rapid experimentation.

### 3. native_reference (ScaLAPACK MPI)

**What it is**: Distributed Cholesky factorization across MPI processes.

**Why it exists**: Single-node memory and compute become limiting for large m.

**How it works**:
```
┌─────────────────────────────────┐
│  Matrix distributed across 16   │
│  processes in block-cyclic       │
│  layout. Each owns ~1/16 of     │
│  the data and computation.       │
└─────────────────────────────────┘

Parallel complexity: O(m³/P) + O(m² log P) communication
```

**When it works well**:
- Large m (>10k)
- Multiple nodes available
- Batch processing

**When it struggles**:
- Small m: Communication overhead > speedup
- Interactive sessions: Startup overhead per call

**Typical use**: Large-scale batch fitting, HPC clusters.

### 4. gpu (CuPy/CUDA)

**What it is**: Scoring (not fitting) on GPU.

**Why it exists**: Variance computation is embarrassingly parallel:
- Each of n candidates needs independent triangular solve
- GPU can process thousands simultaneously

**When it works well**:
- m ≥ 500 (overcomes transfer overhead)
- Large n (more parallelism)
- GPU available

**When it struggles**:
- Small m (<500): Transfer overhead dominates
- Fitting: Cholesky is sequential (no GPU benefit)

**Typical use**: Scoring large candidate sets with moderate-to-large m.

---

## Understanding Crossover Points

### Fit Backend Crossover

**Python vs PyBind11**:
```
For m < 5k:
- Python: Fast enough (0.01-0.5s)
- PyBind11: 10× faster (zero overhead)
- Winner: PyBind11 (instant feels better than fast)
```

**Python vs ScaLAPACK**:
```
For m = 5k:
- Python: 0.5s
- ScaLAPACK (8 ranks): 2.6s (overhead dominates)
- Winner: Python

For m = 20k:
- Python: 10s
- ScaLAPACK (16 ranks): 6s (compute dominates)
- Winner: ScaLAPACK
```

**Crossover**: m ≈ 7-10k (depends on rank count and communication).

### Score Backend Crossover

**CPU vs GPU**:
```
For m = 100:
- CPU: 0.05s
- GPU: 0.27s (transfer overhead dominates)
- Winner: CPU (5× faster)

For m = 500:
- CPU: 0.35s
- GPU: 0.08s (compute dominates)
- Winner: GPU (4.6× faster)
```

**Crossover**: m ≈ 500 (GPU cold-start overhead).

---

## Automatic Selection Logic

The system chooses backends based on these crossover points:

```python
def select_fit_backend(m: int):
    """Choose fit backend based on problem size."""
    if m < 5000:
        # Overhead matters more than parallelization
        return "native_lapack" if available else "python"
    elif m < 10000:
        # Single-node good enough
        return "python"
    else:
        # Need distributed memory
        return "native_reference"  # ScaLAPACK

def select_score_backend(m: int, has_gpu: bool):
    """Choose score backend based on problem size and hardware."""
    if has_gpu and m >= 500:
        # GPU overcomes transfer overhead
        return "gpu"
    elif m > 1000:
        # Multi-threaded BLAS helps
        return "native_lapack" if available else "python"
    else:
        # Simple sequential is fine
        return "python"
```

---

## Using Automatic Selection

### Basic Usage

```python
from poetry_gp.backends.blocked import run_blocked_step

# Let the system choose optimal backends
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    fit_backend="auto",      # Chooses based on m
    score_backend="auto",    # Chooses based on m and hardware
)
```

The defaults are `"auto"`, so you can simply:

```python
result = run_blocked_step(embeddings, rated_indices, ratings)
```

### Manual Override

For testing or specific requirements:

```python
# Force specific backends
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    fit_backend="native_lapack",  # Explicit choice
    score_backend="gpu",           # Explicit choice
)
```

**When to override**:
- Benchmarking (compare backends on same problem)
- Known optimal backend for your workload
- Testing backend implementations

---

## Configuration Presets

For common scenarios:

```python
from poetry_gp.config import GPConfig

# Fast iteration (skip HP optimization)
fast_config = GPConfig(
    fit_backend="native_lapack",
    score_backend="auto",
    optimize_hyperparameters=False,
)

# Accurate predictions (with HP optimization)
accurate_config = GPConfig(
    fit_backend="auto",
    score_backend="auto",
    optimize_hyperparameters=True,
    optimizer_maxiter=100,
)

# Large-scale batch processing
large_scale_config = GPConfig(
    fit_backend="native_reference",
    score_backend="gpu",
    scalapack_nprocs=16,
)

# Use config
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    **fast_config.to_dict()
)
```

---

## Checking Backend Availability

```python
from poetry_gp.backends.backend_selection import print_backend_status

print_backend_status(verbose=True)
```

Output:
```
Backend Availability:
  Python (scipy):     Always available
  Native LAPACK:      Available
  GPU (CuPy):         Not available (install cupy-cuda11x/12x)
  ScaLAPACK (MPI):    Always available

Recommendations:
  Fit backend:
    m < 5k:   native_lapack (instant) or python
    m < 10k:  python
    m >= 10k: native_reference (ScaLAPACK MPI)

  Score backend:
    m < 500:  native_lapack (GPU cold-start overhead)
    m >= 500: python (no GPU available)
```

---

## Performance Data

### Fit Performance

| m | Python | PyBind11 | ScaLAPACK (8 ranks) | Notes |
|---|--------|----------|---------------------|-------|
| 100 | 0.01s | **0.001s** | 2.5s | PyBind11: zero overhead |
| 1k | 0.10s | **0.003s** | 2.5s | PyBind11: 33× faster |
| 5k | 0.50s | **0.10s** | 2.6s | PyBind11 or Python |
| 10k | 2.0s | 0.5s | **3.0s** | Crossover region |
| 20k | 10s | N/A | **6s** | ScaLAPACK: 1.7× faster |

### Score Performance (n=10k candidates)

| m | Python (8 threads) | PyBind11 | GPU | Notes |
|---|-------------------|----------|-----|-------|
| 100 | 0.052s | **0.048s** | 0.271s | CPU: transfer overhead |
| 500 | 0.353s | 0.310s | **0.076s** | GPU: 4.6× faster |
| 1k | 0.717s | 0.620s | **0.193s** | GPU: 3.7× faster |
| 5k | 6.342s | 5.600s | **2.378s** | GPU: 2.7× faster |

**Key insight**: Optimal backend changes with problem size. No single backend is best for all m.

---

## Common Patterns

### Interactive Exploration (small m)

```python
# Fast iteration for m < 1000
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    fit_backend="auto",      # → native_lapack (instant)
    score_backend="auto",    # → native_lapack or python
)
# Total: <1 second per iteration
```

### Batch Processing (large m)

```python
# Distributed fitting for m > 10k
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    fit_backend="native_reference",  # ScaLAPACK
    score_backend="gpu",              # GPU scoring
    scalapack_nprocs=16,
)
# Leverages all available hardware
```

### Benchmarking

```python
# Compare all backends on same problem
for backend in ["python", "native_lapack", "native_reference"]:
    result = run_blocked_step(
        embeddings, rated_indices, ratings,
        fit_backend=backend,
        score_backend="python",  # Hold constant
    )
    print(f"{backend}: {result.profile.fit_seconds:.3f}s")
```

---

## Design Principles

### 1. Optimize the Common Case

Most interactive sessions: m < 5000, n < 100k
- PyBind11 makes these instant
- ScaLAPACK would add overhead

### 2. Eliminate Overhead Before Parallelizing

For small m:
- 160ms subprocess spawn >> 10ms computation
- In-memory > distributed
- Simple > complex

### 3. Parallelize When It Pays

Overhead worth it when:
- Communication cost < speedup gain
- Problem doesn't fit in single-node memory

### 4. Provide Transparency

Auto-selection chooses sensible defaults, but:
- Users can override for specific needs
- Benchmarking tools show what was chosen
- Performance data explains why

---

## Testing Backend Selection

```bash
python scripts/test_backend_selection.py
```

This verifies:
- Selection logic for different problem sizes
- Integration with `run_blocked_step`
- Config usage and presets
- Manual overrides work correctly

---

## Summary

**Key takeaway**: There is no universally optimal backend. The right choice depends on:
- Problem size (m, n)
- Available hardware (GPU, multiple nodes)
- Use case (interactive vs batch)

Automatic selection handles the common cases. Manual override provides flexibility for special needs.

**For most users**: Use `fit_backend="auto"` and `score_backend="auto"`. The system will make reasonable choices.

**For HPC users**: Understand the tradeoffs, measure your workload, and override when you know better than the heuristics.
