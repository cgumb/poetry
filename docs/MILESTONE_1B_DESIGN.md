# Milestone 1B: Distributed Kernel Assembly

## Goal

Eliminate the remaining ScaLAPACK overhead by having each MPI rank build its own portion of the kernel matrix directly from features, rather than building the full matrix on rank 0 and scattering.

## Current Bottlenecks (After BLAS Optimization)

From benchmark analysis, current overhead breakdown:

| Component | Time (m=2000) | Description |
|-----------|---------------|-------------|
| Kernel assembly (BLAS) | ~0.1-0.5s | Rank 0 builds full K_rr using DGEMM |
| Scatter to ranks | ~0.3-0.5s | Distribute K_rr to block-cyclic layout |
| Parallel Cholesky | ~0.05s | ✅ Fast! (the actual computation) |
| Gather from ranks | ~0.8-1.0s | Collect results back to rank 0 |
| File I/O | ~0.5s | Write/read alpha and chol |
| **Total overhead** | **~2-3s** | Dominates for m < 5000 |

## Proposed Solution

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT (Milestone 1A)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Rank 0:  Build full K_rr (n×n) ────┐                       │
│              ↓                       │                       │
│           [800MB file]               │ ~0.5s scatter         │
│              ↓                       │                       │
│  All ranks: Read matrix ←────────────┘                       │
│              ↓                                               │
│  All ranks: Block-cyclic scatter (point-to-point)            │
│              ↓                                               │
│  All ranks: Parallel Cholesky ✅ FAST                        │
│              ↓                                               │
│  All ranks: Block-cyclic gather (point-to-point)             │
│              ↓                       │                       │
│  Rank 0: Collect results ←───────────┘ ~1.0s gather         │
│              ↓                                               │
│           [800MB file]                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 MILESTONE 1B (Distributed Assembly)          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Rank 0:  Broadcast X_rated (n×d) ──┐                       │
│              ↓                       │ ~0.01s (small)        │
│  All ranks: Receive X_rated ←────────┘                       │
│              ↓                                               │
│  All ranks: Compute LOCAL blocks of K_rr ✅ PARALLEL         │
│             Using BLAS DGEMM for local portions              │
│             Each rank: ~(n²/P) elements                      │
│              ↓                                               │
│  All ranks: Parallel Cholesky ✅ FAST                        │
│              ↓                                               │
│  All ranks: Gather alpha, L ←────────┐ Optimized gather     │
│              ↓                       │                       │
│  Rank 0: Reconstruct results ←───────┘                       │
│              ↓                                               │
│           [smaller files, optional]                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Changes

1. **Broadcast features instead of matrix**
   - X_rated: n×d = 10k×384 = 30MB (vs 800MB for K_rr)
   - ~30× smaller, ~0.01s broadcast time

2. **Distributed kernel assembly**
   - Each rank computes only its LOCAL blocks
   - Uses block-cyclic descriptor (DESCINIT)
   - Still uses BLAS DGEMM, but for local portions

3. **Eliminate scatter/gather overhead**
   - No point-to-point redistribution needed
   - Matrix is already in correct layout after assembly

4. **Optional: Direct MPI result transfer**
   - Can use MPI_Gather for alpha
   - Can reconstruct L on rank 0 or use shared memory

## Implementation Plan

### Phase 1: Add Distributed RBF Kernel Assembly

**New C++ function**: `build_local_rbf_blocks()`

```cpp
// Each rank computes only its local blocks according to ScaLAPACK descriptor
std::vector<double> build_local_rbf_blocks(
    const std::vector<double>& x_full,  // Full feature matrix (n×d)
    std::size_t n,                      // Total matrix dimension
    int d,                              // Feature dimension
    int desc[9],                        // ScaLAPACK descriptor
    int rank, int size,
    double length_scale,
    double variance,
    double noise,
    MPI_Comm comm
) {
    // 1. Determine local block dimensions from descriptor
    int mb = desc[4];  // Row block size
    int nb = desc[5];  // Column block size
    int lld = desc[8]; // Local leading dimension

    // 2. Compute which global blocks this rank owns
    std::vector<std::pair<int, int>> owned_blocks = get_owned_blocks(desc, rank);

    // 3. For each owned block:
    std::vector<double> local_matrix(lld * local_cols, 0.0);
    for (auto [global_i_block, global_j_block] : owned_blocks) {
        // 3a. Compute global row/col indices
        int i_start = global_i_block * mb;
        int j_start = global_j_block * nb;

        // 3b. Extract features for this block
        // x_i = x_full[i_start:i_end, :]
        // x_j = x_full[j_start:j_end, :]

        // 3c. Compute block Gram matrix: G_block = x_i @ x_j^T
        // Use BLAS DGEMM

        // 3d. Apply RBF transformation
        // K_block[i,j] = variance * exp(-0.5 * d²/ℓ²)

        // 3e. Add noise to diagonal if i_block == j_block

        // 3f. Store in local_matrix at correct position
    }

    return local_matrix;
}
```

**Key insights**:
- Each rank needs **full X** (but it's small: n×d)
- Each rank computes **only local blocks** of K
- Block-cyclic distribution ensures load balance
- No communication during assembly (embarrassingly parallel)

### Phase 2: Update Entry Point

```cpp
// In main():
std::vector<double> x_rated;
if (rank == 0) {
    x_rated = read_binary_feature_matrix_entry(...);
}

// Broadcast features to all ranks
broadcast_features(x_rated, n, d, rank, MPI_COMM_WORLD);

// Initialize ScaLAPACK descriptor
int desc[9];
int info = 0;
descinit_(desc, &n, &n, &block_size, &block_size,
          &zero, &zero, &ictxt, &lld, &info);

// Each rank builds its local blocks (PARALLEL!)
std::vector<double> local_K = build_local_rbf_blocks(
    x_rated, n, d, desc, rank, size,
    length_scale, variance, noise, MPI_COMM_WORLD
);

// Proceed with Cholesky (local_K is already distributed correctly)
if (backend == "scalapack") {
    pdpotrf_("L", &n, local_K.data(), &one, &one, desc, &info);
    // ...
}
```

### Phase 3: Optimize Result Gathering

**Option A**: Keep current file-based approach (simplest)
- Still works, just with smaller overhead

**Option B**: MPI_Gatherv for alpha (medium complexity)
```cpp
// Gather alpha vector using MPI
MPI_Gatherv(local_alpha, local_len, MPI_DOUBLE,
            full_alpha, recvcounts, displs, MPI_DOUBLE,
            0, MPI_COMM_WORLD);
```

**Option C**: Shared memory for results (complex, future work)
- Use MPI shared memory windows
- Eliminate file I/O entirely

**Recommendation**: Start with Option A, migrate to Option B later.

## Expected Performance Impact

### Overhead Reduction

| Component | Before (1A) | After (1B) | Improvement |
|-----------|-------------|------------|-------------|
| Data transfer | 800MB scatter | 30MB broadcast | 27× smaller |
| Scatter time | ~0.5s | ~0.01s | 50× faster |
| Kernel assembly | Rank 0 only | All ranks | P× faster |
| Gather time | ~1.0s | ~0.5s | 2× faster |
| **Total overhead** | **~2-3s** | **~0.2-0.5s** | **6-10× faster** |

### New Crossover Point

- **Current (1A)**: ScaLAPACK competitive at m ≈ 5000-10000
- **After 1B**: ScaLAPACK competitive at m ≈ 1000-2000 ✅ TARGET

### Scaling Efficiency

- **Current**: Limited by scatter/gather overhead
- **After 1B**: True parallel scaling for assembly + Cholesky
- **Expected**: 80-90% parallel efficiency for 2-16 ranks

## Testing Strategy

### Unit Tests

1. **Correctness**: Compare distributed assembly to Python
   ```python
   K_python = rbf_kernel(X, X, length_scale, variance, noise)
   K_distributed = build_via_scalapack_distributed(X, ...)
   np.testing.assert_allclose(K_python, K_distributed, rtol=1e-10)
   ```

2. **Block-cyclic layout**: Verify descriptor math
   - Test different block sizes (64, 128, 256)
   - Test different process grids (1×2, 2×2, 2×4)

3. **Edge cases**:
   - n not divisible by block_size
   - Single process (should work)
   - Prime number of processes

### Performance Tests

1. **Overhead benchmark**:
   - Measure assembly time separately
   - Compare 1A vs 1B for m=1000, 2000, 5000

2. **Scaling test**:
   - Weak scaling: n scales with P
   - Strong scaling: fixed n, varying P

3. **Crossover analysis**:
   - Find exact crossover point for your cluster
   - Update documentation with measured values

## Implementation Checklist

- [ ] Implement `broadcast_features()` helper
- [ ] Implement `get_owned_blocks()` from descriptor
- [ ] Implement `build_local_rbf_blocks()` with BLAS
- [ ] Update `main()` to use distributed assembly
- [ ] Add correctness tests
- [ ] Add performance benchmarks
- [ ] Update warning thresholds in Python wrapper
- [ ] Update documentation (ROADMAP, BENCHMARKING_GUIDE)
- [ ] Create PR with before/after performance comparison

## Risks and Mitigations

### Risk 1: Block-cyclic descriptor complexity
**Mitigation**:
- Start with simple test cases (square grids, divisible sizes)
- Add extensive logging for debugging
- Reference ScaLAPACK examples

### Risk 2: Memory usage increase
**Concern**: Each rank needs full X (30MB)
**Mitigation**:
- Still much smaller than full K_rr (800MB)
- Acceptable tradeoff for performance gain

### Risk 3: Code complexity increase
**Mitigation**:
- Keep 1A backend as fallback
- Well-documented code with clear comments
- Comprehensive test suite

## Future Optimizations (Beyond 1B)

1. **Lazy broadcast**: Only send features once per interactive session
2. **Compressed features**: Use PCA/quantization for very large n
3. **GPU kernel assembly**: Offload RBF computation to GPU
4. **Hierarchical approach**: Combine with approximate GP methods

## Success Criteria

✅ **Milestone 1B is successful if:**

1. **Correctness**: Distributed assembly matches Python to machine precision
2. **Performance**: Total overhead < 0.5s for m=2000
3. **Crossover**: ScaLAPACK faster than Python at m ≈ 1000-2000
4. **Scaling**: 70-90% parallel efficiency for 2-16 ranks
5. **Robustness**: All tests pass for various configurations

## References

- ScaLAPACK Users' Guide: Block-cyclic distribution (Chapter 4)
- `docs/NATIVE_HPC_ROADMAP.md`: Bottleneck analysis
- `docs/FULL_PIPELINE_HPC_ROADMAP.md`: System-level optimization
- PR #29: Milestone 1A implementation (features → native code)
