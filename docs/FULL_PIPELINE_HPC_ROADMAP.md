# Full Pipeline HPC Roadmap

This document analyzes the complete computational pipeline for GP-based poetry recommendation and identifies HPC opportunities beyond just the fit step.

---

## Pipeline Overview

The complete recommendation pipeline consists of four major computational stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. HYPERPARAMETER OPTIMIZATION  (~50 iterations)                │
│     ├─ Marginal likelihood evaluation                           │
│     ├─ Kernel matrix assembly: O(m²×d)                          │
│     ├─ Cholesky factorization: O(m³)                            │
│     └─ Gradient computation (optional)                          │
│                                                                  │
│  2. GP FITTING (once, with optimal hyperparameters)              │
│     ├─ Kernel matrix assembly: O(m²×d)                          │
│     ├─ Cholesky factorization: O(m³)                            │
│     └─ Triangular solve: O(m²)                                  │
│                                                                  │
│  3. CANDIDATE SCORING (all M candidates)                         │
│     ├─ Cross-kernel K_qr: O(M×m×d)                              │
│     ├─ Posterior mean: K_qr @ alpha  O(M×m)                     │
│     └─ Posterior variance: ||L⁻¹K_qr^T||²  O(M×m²) ← EXPENSIVE! │
│                                                                  │
│  4. SELECTION (argmax operations)                                │
│     ├─ Exploit: argmax(mean)  O(M)                              │
│     └─ Explore: argmax(variance)  O(M)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Computational Complexity Analysis

### Problem Parameters

- **m**: Number of rated poems (10-10,000 typical)
- **M**: Number of candidate poems (10,000-100,000 typical)
- **d**: Embedding dimension (384 fixed)
- **k**: Hyperparameter optimization iterations (~50)

### Time Complexity by Stage

| Stage | Complexity | Typical Time (m=1000, M=50k) | Notes |
|-------|-----------|------------------------------|-------|
| Hyperparameter Opt | k × O(m³) | 5-10 seconds | Repeated Cholesky |
| GP Fit | O(m³) | 0.1 seconds | Single Cholesky |
| Scoring (mean) | O(M×m) | 0.05 seconds | BLAS-2, fast |
| Scoring (variance) | O(M×m²) | **2-5 seconds** | **Bottleneck!** |
| Selection | O(M) | <0.01 seconds | Trivial |

### Key Insight

For realistic problem sizes:
- **Small m (< 500)**: Scoring dominates total time
- **Medium m (500-5000)**: Scoring and hyperparameter optimization both matter
- **Large m (> 5000)**: Hyperparameter optimization can dominate

---

## Current Implementation Analysis

### 1. Hyperparameter Optimization (gp_exact.py:71-138)

**Current approach:**
```python
def optimize_gp_hyperparameters(...):
    def objective(log_params):
        # Rebuild kernel matrix
        k_rr = rbf_kernel(x_rated, x_rated, ...)
        # Cholesky factorization
        c_and_lower = cho_factor(k_rr, ...)  # O(m³)
        alpha = cho_solve(c_and_lower, y)
        # Compute log marginal likelihood
        return -lml

    result = minimize(objective, ..., method="L-BFGS-B")  # ~50 calls
```

**Bottlenecks:**
- ✅ **Repeated Cholesky factorization**: ~50 × O(m³) operations
- Each iteration is independent (can't reuse factorization)
- L-BFGS-B is serial by nature

**HPC opportunities:**
- ❌ **Parallelizing L-BFGS-B**: Not straightforward (inherently serial optimizer)
- ✅ **Using distributed Cholesky per iteration**: Can reuse ScaLAPACK fit backend
- ✅ **Gradient-based optimization**: Analytically compute gradients to reduce iterations
- ⭐ **Warm-start from previous interaction**: Reuse hyperparameters from last round

### 2. Scoring: Posterior Mean (blocked.py:111-115)

**Current approach:**
```python
for start in range(0, M, block_size):
    k_qr = rbf_kernel(embeddings[start:stop], x_rated)  # O(B×m×d)
    mean[start:stop] = k_qr @ alpha  # O(B×m) - BLAS2
```

**Performance:**
- Block size = 2048 works well
- BLAS-2 (matrix-vector) is fast
- Not a major bottleneck

**HPC opportunities:**
- ⭐ **MPI distribution**: Already implemented in mpi.py backend!
- Each rank scores its partition of candidates
- Near-perfect scaling (embarrassingly parallel)

### 3. Scoring: Posterior Variance (predict_block:186-201)

**Current approach:**
```python
def predict_block(state, x_query):
    k_qr = rbf_kernel(x_query, state.x_rated)  # O(B×m×d)
    mean = k_qr @ state.alpha  # O(B×m) - fast

    # EXPENSIVE PART:
    l_tri = np.tril(state.cho_factor_data[0])
    v = solve_triangular(l_tri, k_qr.T, ...)  # O(B×m²) ← BOTTLENECK!
    var = state.variance - np.sum(v * v, axis=0)  # O(B×m)
    return mean, var
```

**Why it's expensive:**
- Triangular solve: O(B×m²) for a block of size B
- Total for all M candidates: O(M×m²)
- For M=50k, m=1000: **50 billion operations!**
- Cannot be reduced to BLAS-3 (inherently triangular solve structure)

**HPC opportunities:**
- ⭐ **MPI distribution**: Each rank solves for its partition
- ⭐ **Batched triangular solves**: Use BLAS-3 where possible
- ⭐ **GPU acceleration**: Triangular solves parallelize well on GPU
- ⭐ **Lazy evaluation**: Only compute variance for top-K candidates after filtering by mean

---

## Proposed HPC Roadmap

### Phase 1: Distributed Scoring (IMMEDIATE WIN) ⭐⭐⭐

**Goal**: Parallelize candidate scoring across MPI ranks

**Status**: MPI backend already exists (mpi.py) but not fully utilized

**Implementation**:
```python
# blocked.py modification
if score_backend == "mpi":
    # Partition candidates across ranks
    # Each rank computes mean and variance for its subset
    # Gather top-K candidates from each rank
    # Root selects global best exploit/explore
```

**Expected speedup**: Near-linear with rank count (embarrassingly parallel)

**Effort**: LOW (mostly plumbing existing code)

**Impact**: HIGH
- For M=50k, m=1000 on 16 ranks: **~15× faster scoring**
- Eliminates scoring as bottleneck for medium-sized problems

---

### Phase 2: Optimized Hyperparameter Optimization ⭐⭐

**Option 2A: Analytic Gradients** (Recommended)

Instead of numerical gradients, compute analytically:

```python
def objective_with_gradient(log_params):
    # Compute log marginal likelihood
    lml = ...

    # Analytic gradient (see Rasmussen & Williams, 5.4.1)
    # ∂lml/∂θ = 0.5 * tr((α α^T - K⁻¹) ∂K/∂θ)
    grad_theta = compute_lml_gradient(K, alpha, K_inv, theta)

    return -lml, -grad_theta

result = minimize(objective_with_gradient, ..., method="L-BFGS-B", jac=True)
```

**Benefits**:
- Reduces iterations by 30-50%
- More accurate gradients than finite differences
- Still uses same Cholesky factorization per iteration

**Effort**: MEDIUM (need to implement gradient computation)

**Impact**: MEDIUM
- For m=1000: Reduces opt time from ~10s to ~5s
- Mainly helps for large m where optimization dominates

**Option 2B: Warm-Start from Previous Round**

```python
# In interactive session
if previous_hyperparams is not None:
    # Start optimization from previous solution
    init_params = previous_hyperparams
else:
    init_params = default_hyperparams
```

**Benefits**:
- Hyperparameters change slowly between rounds
- Can reduce optimization to 5-10 iterations instead of 50

**Effort**: LOW (just pass previous params)

**Impact**: MEDIUM

---

### Phase 3: GPU Acceleration for Scoring ⭐⭐⭐

**Goal**: Offload posterior variance computation to GPU

**Why GPU is ideal**:
- Triangular solves: Inherently parallel across queries
- Large M (50k-100k): Good GPU occupancy
- Blocked computation: Natural batching

**Implementation** (using CuPy or PyTorch):
```python
import cupy as cp

def predict_block_gpu(state_gpu, x_query_gpu):
    # Transfer L and alpha to GPU once
    # Compute k_qr on GPU
    k_qr = rbf_kernel_gpu(x_query_gpu, state_gpu.x_rated)
    mean = k_qr @ state_gpu.alpha

    # GPU triangular solve (parallel across queries)
    v = cp.linalg.solve_triangular(state_gpu.L, k_qr.T)
    var = state_gpu.variance - cp.sum(v * v, axis=0)

    return cp.asnumpy(mean), cp.asnumpy(var)
```

**Expected speedup**: **10-50× for large M** (depending on GPU)

**Effort**: MEDIUM (GPU kernel optimization)

**Impact**: HIGH for large candidate sets

---

### Phase 4: Approximate Methods (SCALABILITY) ⭐

For extremely large problems (m > 50k), consider approximations:

**Option 4A: Inducing Points (Sparse GP)**
- Use m' << m inducing points
- Reduces complexity from O(m³) to O(m'² × m)
- Trade-off: Accuracy vs speed

**Option 4B: Stochastic Variational GP**
- Mini-batch training
- Scales to millions of points
- More complex implementation

**Option 4C: Local GP**
- Partition embedding space
- Fit separate GPs per region
- Good for very high-dimensional spaces

**Effort**: HIGH (research-level implementations)

**Impact**: Enables m > 100k

---

## Implementation Priority Matrix

| Optimization | Effort | Impact | Speedup | Priority |
|-------------|--------|--------|---------|----------|
| **Distributed Scoring (MPI)** | LOW | HIGH | 10-15× | **P0** ⭐⭐⭐ |
| **Distributed Fit (Milestone 1B)** | MEDIUM | MEDIUM | 2-3× | **P0** ⭐⭐ |
| **Warm-Start Hyperparameters** | LOW | MEDIUM | 2-3× | **P1** ⭐⭐ |
| **Analytic Gradients** | MEDIUM | MEDIUM | 1.5-2× | **P2** ⭐ |
| **GPU Scoring** | MEDIUM | HIGH | 10-50× | **P2** ⭐⭐⭐ |
| **Lazy Variance** | LOW | LOW | 1.2-1.5× | P3 |
| **Approximate GP** | HIGH | SCALING | 10-100× | P4 |

---

## Recommended Development Sequence

### Sprint 1: Baseline and Quick Wins (CURRENT)
1. ✅ Benchmark current Python vs ScaLAPACK fit
2. ✅ Identify bottlenecks via profiling
3. ⏳ Implement Milestone 1B (distributed kernel assembly)
4. ⏳ Enable MPI scoring backend by default

### Sprint 2: Scoring Optimization
1. Implement full MPI scoring pipeline
2. Add warm-start for hyperparameter optimization
3. Benchmark end-to-end pipeline
4. Document scaling characteristics

### Sprint 3: Advanced Optimizations
1. GPU scoring prototype
2. Analytic gradient implementation
3. Performance comparison: CPU vs GPU
4. Scaling study: 1-64 ranks

### Sprint 4: Production-Ready
1. Auto-select backend based on problem size
2. Comprehensive error handling
3. Integration tests
4. Student-facing documentation

---

## Expected Performance Targets

### Current State (Baseline)
- m=1000, M=50k: **~15 seconds total**
  - Hyperparameter opt: ~10s
  - Fit: ~0.1s
  - Score: ~3s
  - Select: <0.01s

### After Sprint 1 (Distributed Fit + MPI Scoring)
- m=1000, M=50k, 16 ranks: **~3 seconds total**
  - Hyperparameter opt: ~10s (serial, unchanged)
  - Fit: ~0.05s (2× faster)
  - Score: ~0.2s (15× faster)
  - Select: <0.01s

### After Sprint 2 (Warm-Start)
- m=1000, M=50k, 16 ranks: **~1 second total**
  - Hyperparameter opt: ~2s (5× faster via warm-start)
  - Fit: ~0.05s
  - Score: ~0.2s
  - Select: <0.01s

### After Sprint 3 (GPU Scoring)
- m=1000, M=50k, GPU: **~0.5 seconds total**
  - Hyperparameter opt: ~2s (still on CPU)
  - Fit: ~0.05s (CPU)
  - Score: ~0.01s (GPU, 200× faster)
  - Select: <0.01s

---

## Pedagogical Value

This roadmap provides excellent teaching opportunities:

1. **Embarrassingly Parallel** (MPI Scoring): Classic HPC pattern
2. **Communication Overhead** (Distributed Fit): Trade-offs in distributed computing
3. **GPU Programming** (GPU Scoring): Heterogeneous computing
4. **Algorithm Selection** (Approximate GP): Scaling beyond exact methods
5. **Profiling-Driven Optimization**: Measure before optimizing

Each sprint builds on the previous, demonstrating a methodical approach to HPC optimization.

---

## Next Steps

1. **Complete current benchmark sweep** to establish baseline
2. **Profile hyperparameter optimization** to quantify its cost
3. **Implement MPI scoring backend** (highest ROI)
4. **Design GPU prototype** for scoring (research spike)

See `NATIVE_HPC_ROADMAP.md` for fit-specific optimizations.
