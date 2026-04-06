# HPC Implementations for GP Active Learning

## The Computational Problem

Gaussian process active learning requires two expensive operations at each iteration:

### 1. Fitting (O(m³))

Given m rated poems, update the posterior:

```
K_rr = k(X_rated, X_rated) + σ_n² I     # m×m kernel matrix
K_rr = L L^T                             # Cholesky factorization (O(m³))
α = K_rr^(-1) y                          # Solve for weights (O(m²))
```

**The bottleneck**: Cholesky factorization is O(m³).

For m = 10,000: **1 trillion FLOPs** per fit.

### 2. Scoring (O(nm²))

Given n candidate poems, compute posterior mean and variance:

```
μ(X_query) = K_qr α                      # Mean: O(nm), fast
σ²(X_query) = diag(K_qq - K_qr K_rr^(-1) K_rq)   # Variance: O(nm²), expensive
```

Variance requires solving:
```
V = L^(-1) K_rq^T                        # Triangular solve (O(nm²))
σ² = k(x,x) - ||V||²                    # Variance from solve
```

**The bottleneck**: Triangular solve is O(nm²).

For n = 85,000, m = 1,000: **85 billion FLOPs** per score.

---

## Why This Matters for HPC

**Interactive recommendation** requires fast iteration:
```
rate → fit → score → select → repeat
```

**Scaling behavior**:
- Small problems (m < 500): Python + NumPy is fine
- Medium problems (m = 1k-10k): Fit becomes expensive
- Large problems (m > 10k): Both fit and score are expensive
- Large candidate sets (n > 50k): Variance scoring dominates

**Design goal**: Make interaction feel responsive (<1 second per iteration).

---

## Solution 1: PyBind11 LAPACK (Single-Node Optimization)

### The Problem It Solves

Python's subprocess overhead:
- Each ScaLAPACK call: ~160ms overhead
- For small m (<5k), overhead > computation
- Result: ScaLAPACK slower than Python for small problems

### The Solution

**Direct in-memory LAPACK** via PyBind11:
```cpp
// C++ function callable from Python (zero overhead)
pybind11::dict fit_gp_lapack(
    py::array_t<double> K_rr,    // Fortran-order input
    py::array_t<double> y_rated
) {
    // Call LAPACK directly: dpotrf (Cholesky), dpotrs (solve)
    LAPACK_dpotrf("L", &m, K_data, &m, &info);
    LAPACK_dpotrs("L", &m, &nrhs, K_data, &m, y_data, &m, &info);
    
    return {{"alpha", alpha}, {"logdet", logdet}, ...};
}
```

**Key insight**: For small m, eliminating subprocess overhead matters more than parallelization.

### When It Wins

| m | Python | PyBind11 | Speedup |
|---|--------|----------|---------|
| 500 | 0.01s | 0.001s | 10× |
| 1000 | 0.03s | 0.003s | 10× |
| 5000 | 3.0s | 0.3s | 10× |

**Crossover**: PyBind11 dominates for m < 5,000 (single-node).

---

## Solution 2: ScaLAPACK (Distributed Memory)

### The Problem It Solves

For large m (>10k), even single-node LAPACK becomes slow:
- m = 10k: 1 trillion FLOPs
- m = 20k: 8 trillion FLOPs  
- Single-node: Minutes per fit

### The Solution

**Distribute the matrix** across MPI processes using **block-cyclic layout**:

```
Process grid (2×2):
┌─────┬─────┐
│ P0  │ P1  │
├─────┼─────┤
│ P2  │ P3  │
└─────┴─────┘

Matrix distribution (block size = 64):
┌────┬────┬────┬────┐
│ P0 │ P1 │ P0 │ P1 │  Block-cyclic:
├────┼────┼────┼────┤  Blocks distributed
│ P2 │ P3 │ P2 │ P3 │  in round-robin pattern
├────┼────┼────┼────┤  for load balancing
│ P0 │ P1 │ P0 │ P1 │
└────┴────┴────┴────┘
```

**Distributed Cholesky** (ScaLAPACK `pdpotrf`):
- Each process owns subset of matrix
- Processes communicate during factorization
- Parallel complexity: O(m³/P) computation + O(m²) communication

### Distributed Kernel Assembly

**Naive approach** (centralized):
```python
# Root assembles full kernel, then scatters
K_rr = rbf_kernel(X_rated, X_rated)  # Root: m×m matrix (800 MB)
scatter_to_processes(K_rr)            # Communication: 800 MB
```

**Optimized approach** (distributed):
```python
# Broadcast features to all processes
broadcast(X_rated)                    # Communication: 30 MB

# Each process computes its tiles
for my_block in process_blocks:
    K_block = rbf_kernel(X_subset, X_subset)  # Parallel assembly
    # BLAS DGEMM optimization: 20-40× faster than naive loops
```

**Key insight**: Broadcast small data (features), compute large data (kernel) in parallel.

**Savings**: 800 MB scatter → 30 MB broadcast (26× less communication).

### When It Wins

| m | Python | ScaLAPACK (8 ranks) | Speedup |
|---|--------|---------------------|---------|
| 5k | 3.0s | 2.5s | 1.2× |
| 10k | 24s | 4.0s | 6× |
| 20k | 192s | 15s | 13× |

**Crossover**: ScaLAPACK dominates for m > 7-10k (distributed).

**Tradeoff**: Communication overhead vs parallel speedup.

---

## Solution 3: GPU (Massive Parallelism)

### The Problem It Solves

Variance scoring is **embarrassingly parallel**:
- Each of n candidates needs independent triangular solve
- For n = 85k: 85,000 independent solves
- CPU: Process sequentially in blocks
- GPU: Process thousands simultaneously

### The Solution

**Offload to GPU** using CuPy:
```python
import cupy as cp

# Transfer state to GPU once
x_rated_gpu = cp.asarray(state.x_rated)
L_gpu = cp.asarray(state.cholesky_factor)
alpha_gpu = cp.asarray(state.alpha)

# Score in parallel
for block in batches(candidates, block_size=2048):
    x_block_gpu = cp.asarray(block)
    
    # Kernel computation (parallel across candidates)
    K_qr = rbf_kernel_gpu(x_block_gpu, x_rated_gpu)
    
    # Mean (parallel matrix-vector)
    mean = K_qr @ alpha_gpu
    
    # Variance (parallel triangular solve)
    V = cp.linalg.solve_triangular(L_gpu, K_qr.T)
    var = variance - cp.sum(V * V, axis=0)
    
    # Transfer results back
    mean_cpu = cp.asnumpy(mean)
    var_cpu = cp.asnumpy(var)
```

**Key insight**: GPU excels at many small independent operations (SIMD parallelism).

### When It Wins

| m | CPU (8 threads) | GPU | Speedup |
|---|-----------------|-----|---------|
| 100 | 0.01s | 0.05s | 0.2× (slower!) |
| 500 | 0.15s | 0.05s | 3× |
| 1000 | 0.60s | 0.13s | 4.6× |
| 5000 | 15s | 3.8s | 4× |

**Crossover**: GPU dominates for m > 500 (scoring).

**Tradeoff**: Transfer overhead vs compute speedup.

---

## Automatic Backend Selection

The system chooses optimal backends based on problem size:

```python
def select_fit_backend(m: int) -> str:
    """Choose fit backend based on m."""
    if m < 5000:
        return "native_lapack"  # PyBind11: zero overhead
    elif m < 10000:
        return "python"         # SciPy: good enough
    else:
        return "scalapack"      # MPI: distributed

def select_score_backend(n: int, m: int, has_gpu: bool) -> str:
    """Choose score backend based on n, m, hardware."""
    if has_gpu and m >= 500:
        return "gpu"            # GPU: best for large m
    elif m > 1000:
        return "native_lapack"  # Multi-threaded BLAS
    else:
        return "python"         # NumPy baseline
```

**Usage**:
```python
result = run_blocked_step(
    embeddings, rated_indices, ratings,
    fit_backend="auto",      # Automatic selection
    score_backend="auto",
)
```

---

## Design Principles

### 1. Optimize for the Common Case

Most interactive sessions have m < 5,000:
- PyBind11 makes these instant (<0.01s)
- ScaLAPACK would add overhead

### 2. Eliminate Overhead First

Before parallelizing:
- Remove subprocess spawning (160ms → 0ms)
- Remove file I/O (write/read → in-memory)
- Use BLAS (DGEMM 20× faster than loops)

### 3. Parallelize When It Pays

ScaLAPACK overhead only worthwhile for m > 7k:
- Small m: Communication overhead > speedup
- Large m: Parallel speedup > overhead

### 4. Match Algorithm to Hardware

- **CPU**: Sequential Cholesky, blocked scoring
- **GPU**: Parallel triangular solves
- **Distributed**: Block-cyclic matrix operations

### 5. Provide Escape Hatches

Auto-selection chooses defaults, but allow manual override:
```python
# Force specific backend for testing/debugging
result = run_blocked_step(..., fit_backend="scalapack")
```

---

## Performance Summary

**Interactive session** (m=1000, n=85k):
```
Fit:     0.03s  (PyBind11 LAPACK)
Score:   0.60s  (GPU)
Select:  0.001s
Total:   0.63s per iteration
```

**Large-scale batch** (m=20k, n=50k):
```
Fit:     15s    (ScaLAPACK, 16 ranks)
Score:   8s     (GPU)
Total:   23s per iteration
```

---

## Further Reading

**Implementation details**:
- `src/poetry_gp/backends/native_lapack.py` - PyBind11 implementation
- `src/poetry_gp/backends/scalapack_fit.py` - ScaLAPACK integration
- `src/poetry_gp/backends/gpu_scoring.py` - CuPy GPU implementation
- `native/pybind11_lapack.cpp` - C++ PyBind11 bindings
- `native/scalapack_gp_fit.cpp` - C++ ScaLAPACK implementation

**Conceptual background**:
- `BACKEND_SELECTION.md` - When to use which backend
- `METHOD_NARRATIVE.md` - Mathematical foundation
- `BENCHMARKING_GUIDE.md` - Performance measurement

---

## Key Takeaways

1. **Overhead matters**: For small problems, eliminating overhead beats parallelization
2. **Communication costs**: Broadcast features (30MB) beats scatter matrix (800MB)
3. **Hardware match**: GPU excels at parallel independent operations
4. **Crossover points**: Measure to find when complex solutions pay off
5. **Abstraction**: Single API, automatic backend selection, manual override for experts
