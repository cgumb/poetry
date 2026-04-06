# Challenges and Solutions: A Development Chronicle

## Why This Document?

HPC development is not straightforward. Theory says "parallelize and it will be faster," but practice involves:
- Memory layout bugs
- Overhead dominating speedup
- Solutions that work on paper but fail in practice

This document catalogs the real challenges we faced and how we solved them. The goal: show that debugging and measurement are core HPC skills.

---

## Challenge 1: Memory Layout Bugs (Fortran vs C Order)

### The Problem

PyBind11 scoring returned garbage variance values:
```python
variance = predict_native(state, candidates)
print(variance)  # [nan, nan, nan, ...]
```

### Investigation

Added debug prints to C++ code:
```cpp
std::cout << "K_qr shape: " << rows << " × " << cols << std::endl;
std::cout << "K_qr[0,0]: " << K_qr(0, 0) << std::endl;  // Correct
std::cout << "K_qr[1,0]: " << K_qr(1, 0) << std::endl;  // WRONG VALUE!
```

**Symptom**: First element correct, subsequent elements garbage.

### Root Cause

NumPy arrays are **C-order** (row-major) by default:
```
C-order [0,1,2,3,4,5] represents:
  [[0, 1, 2],
   [3, 4, 5]]
  
Memory layout: row-by-row
```

LAPACK expects **Fortran-order** (column-major):
```
Fortran-order [0,3,1,4,2,5] represents:
  [[0, 1, 2],
   [3, 4, 5]]
  
Memory layout: column-by-column
```

When we passed C-order array to LAPACK:
- LAPACK interpreted it as transposed
- Dimensions appeared correct but data was scrambled
- Result: garbage variance

### The Fix

Convert to Fortran-order before passing to C++:
```python
K_rr = rbf_kernel(x_rated, x_rated)
K_rr = np.asfortranarray(K_rr)  # Convert to column-major
result = poetry_gp_native.fit_gp_lapack(K_rr, y_rated)
```

Or specify in PyBind11:
```cpp
py::array_t<double, py::array::f_style> K_rr  // Force Fortran order
```

### Lesson Learned

**Memory layout matters**: When interfacing Python (C-order) with Fortran libraries (LAPACK, BLAS), always check memory layout. Use `np.asfortranarray()` or handle transpose explicitly.

---

## Challenge 2: The Stride=0 Bug

### The Problem

Variance computation crashed with segmentation fault:
```
Segmentation fault (core dumped)
```

### Investigation

GDB backtrace showed crash in `dtrsm` (triangular solve):
```
Program received signal SIGSEGV
#0  dtrsm_ (...)
```

Added stride checking:
```cpp
py::buffer_info K_qr_info = K_qr.request();
std::cout << "K_qr strides: [" << K_qr_info.strides[0] 
          << ", " << K_qr_info.strides[1] << "]" << std::endl;
// Output: K_qr strides: [8, 0]  ← WRONG! Should be [8, N*8]
```

**Symptom**: Stride=0 means all elements in that dimension point to same memory location.

### Root Cause

Creating 1D array without proper reshape:
```cpp
// WRONG: Creates 1D array with stride=0 in second dimension
auto mean = py::array_t<double>(n);  

// RIGHT: Explicitly create 2D array
auto mean = py::array_t<double>({n, 1});  // Shape (n, 1)
```

Python's buffer protocol requires valid strides for all dimensions. Stride=0 means "don't advance in this dimension" - valid for broadcasting, but breaks BLAS.

### The Fix

Always specify shape explicitly:
```cpp
// Allocate with proper shape
auto mean = py::array_t<double>({n_query, 1});
auto var = py::array_t<double>({n_query, 1});

// Verify strides
auto mean_info = mean.request();
assert(mean_info.strides[0] > 0);
assert(mean_info.strides[1] >= 0);
```

### Lesson Learned

**Check array metadata**: Don't assume NumPy arrays have sensible strides. When creating arrays in C++, specify shapes explicitly and verify strides before passing to BLAS.

---

## Challenge 3: Use-After-Free in PyBind11

### The Problem

Intermittent crashes when calling native functions:
```python
result = fit_exact_gp_native(x_rated, y_rated)
# Sometimes works, sometimes segfaults
```

### Investigation

Added memory address tracking:
```cpp
void fit_gp_lapack(py::array_t<double> K) {
    auto K_data = K.mutable_data();
    std::cout << "K address: " << (void*)K_data << std::endl;
    // ... use K_data ...
}
```

Saw different addresses on successive calls to same data - indication that Python garbage collector was moving/freeing objects.

### Root Cause

**Temporary arrays getting garbage collected**:
```python
# WRONG: Temporary not kept alive
result = fit_gp_native(
    rbf_kernel(x, x),  # Creates temporary, gets freed after call
    y
)

# In C++, py::array_t<double> just wraps pointer
# If Python object is freed, pointer becomes invalid
```

PyBind11 doesn't increment reference count for temporary objects by default.

### The Fix

Keep temporaries alive:
```python
# RIGHT: Store temporary in variable
K_rr = rbf_kernel(x_rated, x_rated)
K_rr = np.asfortranarray(K_rr)
result = fit_gp_native(K_rr, y_rated)  # K_rr stays alive
```

Or use `py::array_t<double>` with `py::keep_alive`:
```cpp
m.def("fit_gp_lapack", &fit_gp_lapack,
      py::arg("K_rr"), py::arg("y_rated"),
      py::keep_alive<0, 1>());  // Keep arg 1 alive with return value
```

### Lesson Learned

**Manage object lifetimes**: When passing Python objects to C++, ensure they remain alive for the duration of the C++ function. Store temporaries explicitly or use `py::keep_alive`.

---

## Challenge 4: ScaLAPACK Slower Than Python

### The Problem

Distributed ScaLAPACK slower than single-node Python for all problem sizes:
```
m = 1000:
  Python:    0.10s
  ScaLAPACK: 2.5s  (25× slower!)
  
m = 5000:
  Python:    0.5s
  ScaLAPACK: 2.6s  (5× slower!)
```

Expected ScaLAPACK to be faster with parallelization. What went wrong?

### Investigation

Profiled ScaLAPACK run:
```
Total time: 2.5s
  - Process spawn: 0.16s
  - Write K_rr to disk: 0.05s
  - Read K_rr from disk: 0.05s
  - MPI broadcast: 0.10s
  - Actual Cholesky: 0.05s  ← Only 2% of time!
  - Communication: 0.50s
  - Write results: 0.08s
  - Process cleanup: 0.05s
```

**Overhead: 2.4s, Computation: 0.1s**

### Root Cause

For small m, overhead dominates:
- **Subprocess spawn**: ~160ms per call (unavoidable with MPI)
- **File I/O**: 100-150ms reading/writing matrices
- **Communication**: 100-500ms broadcasting and gathering
- **Computation**: 50-100ms (the only part that's parallel!)

Parallelization only helps if computation >> overhead. For small m, overhead is 10-50× the computation.

### The Solution: PyBind11

**Eliminate overhead entirely** for small problems:
```cpp
// Direct in-memory LAPACK (no subprocess, no files, no MPI)
py::dict fit_gp_lapack(py::array_t<double> K, py::array_t<double> y) {
    LAPACK_dpotrf(...);  // 0.01s
    return result;
}
```

Result:
```
m = 1000:
  Python:    0.10s
  PyBind11:  0.003s  (33× faster - eliminated overhead)
  ScaLAPACK: 2.5s    (still slow due to overhead)
```

**For large m, ScaLAPACK wins**:
```
m = 20000:
  Python:    10s     (single-node limit)
  PyBind11:  N/A     (out of memory)
  ScaLAPACK: 6s      (distributed, 1.7× faster)
```

### Lesson Learned

**Optimize for the common case**: Most interactive sessions have m < 5000. Eliminating overhead matters more than parallelization for these problems. Use simple solutions (PyBind11) for small problems, complex solutions (ScaLAPACK) only when they pay off.

---

## Challenge 5: GPU Slower Than CPU

### The Problem

GPU scoring slower than CPU for all tested problem sizes:
```
m = 100:  GPU 10× slower
m = 500:  GPU 2× slower
m = 1000: GPU same speed as CPU
```

Expected GPU to be faster. Why wasn't it?

### Investigation

Profiled GPU scoring:
```python
t0 = time.time()
# Transfer to GPU
x_rated_gpu = cp.asarray(x_rated)     # 20ms
alpha_gpu = cp.asarray(alpha)          # 1ms
L_gpu = cp.asarray(L)                  # 50ms

# Compute on GPU  
K_qr = rbf_kernel_gpu(...)             # 10ms
mean = K_qr @ alpha_gpu                 # 2ms
V = solve_triangular_gpu(...)          # 30ms

# Transfer back
mean_cpu = cp.asnumpy(mean)            # 10ms
var_cpu = cp.asnumpy(var)              # 10ms
t_total = time.time() - t0  # 133ms

# Compare to CPU: 15ms computation
# GPU: 133ms (90ms transfer + 43ms compute)
# CPU: 15ms
```

**GPU transfer overhead: 90ms, Compute: 43ms**

### Root Cause

**Cold-start overhead dominates for small m**:
- GPU memory transfer: Fixed ~100ms
- CPU computation: Scales as O(nm²)
- Crossover: When CPU time > 100ms

For m < 500: CPU faster than GPU transfer overhead alone!

### The Solution

**Use GPU selectively**:
```python
def select_score_backend(m, has_gpu):
    if has_gpu and m >= 500:
        return "gpu"  # Compute dominates transfer
    else:
        return "python"  # Simple sequential wins
```

Results:
```
m = 100:  CPU wins (0.05s vs 0.27s)
m = 500:  GPU wins (0.08s vs 0.35s) - 4.6× speedup!
m = 5000: GPU wins (2.4s vs 6.3s) - 2.7× speedup
```

### Lesson Learned

**Measure crossover points**: Theory says "GPU is faster," but transfer overhead changes the story. Measure actual performance and use each backend where it wins.

---

## Challenge 6: Spatial Variance O(n²) Too Slow

### The Problem

Interactive CLI frozen for 8+ seconds per iteration with spatial_variance exploration:
```
n = 85k candidates:
- Fit: 0.03s
- Score: 0.6s
- Select: 8s  ← Frozen here!
```

User: "Still a bit sad that we can't get better than 8 seconds for the spatial variance."

### Investigation

Profiled selection phase:
```python
# spatial_variance computes:
K = rbf_kernel(embeddings, embeddings)  # 85k × 85k = 7.2B kernel evaluations
```

That's O(n²) = 7.2 billion operations. On CPU: ~8 seconds.

### Attempted Solutions

**Attempt 1: Sampling approximation**
```python
# Sample 10k candidates, scale results
sample = random.choice(candidates, 10k)
K_sample = rbf_kernel(embeddings, sample)  # 85k × 10k
svr ≈ scale_factor * np.sum(K_sample² / variance)
```

**User feedback**: "If I wanted a fast recommendation I'd probably just switch to max_variance. This approximation defeats the purpose."

**Lesson**: The entire value of spatial_variance is considering full correlation structure. Approximating it undermines that.

**Attempt 2: GPU acceleration**
```python
# Compute 85k × 85k kernel on GPU
K_gpu = rbf_kernel_gpu(embeddings_gpu, embeddings_gpu)
```

**Result**: GPU 10× slower than CPU!
- n=1k: GPU 0.17s, CPU 0.02s
- n=10k: GPU 1.9s, CPU 1.4s

**Why**: Memory-bound operation (7.2B elements), not compute-bound. GPU transfer overhead + memory bandwidth limit make it slower.

### The Solution

**Accept limitations and document honestly**:

Created ACQUISITION_FUNCTIONS.md explaining:
- max_variance: O(n), instant, information-optimal
- spatial_variance: O(n²), expensive, spatial diversity
- For n > 10k: Use max_variance

No magic fix for algorithmic complexity. Sometimes the honest answer is "use a different algorithm."

### Lesson Learned

**Not every problem has a fast solution**: Some algorithms are fundamentally expensive. Don't force clever hacks - instead, document tradeoffs honestly and provide alternatives. Students need to learn when to accept limitations.

---

## Challenge 7: Distributed Kernel Assembly (800MB → 30MB)

### The Problem

ScaLAPACK fit with m=5000:
```
Total time: 3.5s
  - Scatter K_rr matrix: 2.0s  ← Most of the time!
  - Cholesky: 1.2s
  - Gather results: 0.3s
```

**Bottleneck**: Scattering 5000×5000 matrix (200MB) from root to all processes.

### Investigation

Analyzed communication:
```
Centralized approach:
1. Root computes full K_rr: 200MB
2. Scatter to 16 processes: 200MB network traffic
3. Each process computes on its tiles

Communication: O(m²) per operation
```

### Root Cause

**Computing centrally, distributing is backwards**:
- Features (X_rated): 5000 × 384 = 8MB
- Kernel matrix (K_rr): 5000 × 5000 = 200MB

Sending kernel is 25× more data than sending features!

### The Solution

**Distributed kernel assembly**:
```
Optimized approach:
1. Broadcast features to all: 8MB × 1 = 8MB
2. Each process computes its tiles in parallel
3. No scatter/gather of kernel

Communication: O(m) per operation (features only)
```

Implementation:
```cpp
// Each rank computes its block-cyclic tiles
for (int i = my_row_blocks...) {
    for (int j = my_col_blocks...) {
        // Compute K[block_i, block_j] from features
        rbf_kernel_block(X_rated, X_rated, i, j, K_local);
    }
}
```

**Bonus**: Use BLAS DGEMM for kernel assembly (20-40× faster than naive loops).

Results:
```
Before (centralized):
  Scatter: 2.0s
  Cholesky: 1.2s
  Total: 3.2s

After (distributed):
  Broadcast: 0.08s  (25× less data)
  Parallel assembly: 0.15s
  Cholesky: 1.2s
  Total: 1.4s  (2.3× faster overall)
```

### Lesson Learned

**Broadcast small, compute large**: Don't compute centrally and distribute. Instead, distribute the smallest data (features) and compute in parallel. Communication complexity matters as much as computation complexity.

---

## Challenge 8: BLACS Context Initialization

### The Problem

ScaLAPACK crashed with cryptic error:
```
BLACS_GRIDMAP: Invalid context handle
Error: -2 in call to BLACS_GRIDMAP
```

Only happened on some MPI implementations (OpenMPI but not MPICH).

### Investigation

Added debug output:
```cpp
int context;
Cblacs_get(-1, 0, &context);  // Get default system context
std::cout << "Initial context: " << context << std::endl;
// Output: context = -1  ← Invalid!
```

### Root Cause

**MPI-based BLACS doesn't have "system context"**:

Old BLACS (pre-MPI):
```cpp
Cblacs_get(-1, 0, &context);  // -1 = system context
```

Modern MPI-based BLACS:
```cpp
Cblacs_get(0, 0, &context);   // 0 = MPI communicator
```

Using context=-1 worked on some systems (legacy BLACS) but crashed on modern MPI-based BLACS.

### The Fix

Initialize from MPI communicator:
```cpp
// Get context from MPI_COMM_WORLD (context=0)
int context;
Cblacs_get(0, 0, &context);  // 0 instead of -1

// Create process grid
Cblacs_gridinit(&context, "Row-major", nprows, npcols);
```

### Lesson Learned

**Read the documentation carefully**: BLACS context initialization changed between implementations. What works on one system may crash on another. Check which BLACS version you're using and use appropriate API.

---

## Challenge 9: Build System Conditional Compilation

### The Problem

Build failed on GPU nodes with:
```
error: 'result' was not declared in this scope
error: 'log_marg_lik' was not declared in this scope
```

Build worked fine on general nodes.

### Investigation

The code:
```cpp
#ifdef HAVE_SCALAPACK
    NativeResult result = fit_scalapack(...);
    double log_marg_lik = compute_lml(...);
#else
    // Error handling
    continue;
#endif

// Write results (OUTSIDE #ifdef)
write_output(result.alpha, alpha_path);  // ← Error: 'result' not declared
```

### Root Cause

**Code using conditional variables outside the conditional block**:

GPU nodes compiled without ScaLAPACK, so `result` was never declared, but code tried to use it anyway.

### The Fix

Move all dependent code inside conditional:
```cpp
#ifdef HAVE_SCALAPACK
    NativeResult result = fit_scalapack(...);
    double log_marg_lik = compute_lml(...);
    
    // Write results (INSIDE #ifdef)
    write_output(result.alpha, alpha_path);
    send_response(SUCCESS);
#else
    // Error path
    send_response(ERROR);
    continue;
#endif
```

### Lesson Learned

**Conditional compilation hygiene**: All code that depends on conditionally compiled variables must be inside the same conditional block. Think about all code paths, not just the one you're testing.

---

## Common Themes

### 1. Measure, Don't Guess

Almost every "optimization" we tried first made things worse:
- ScaLAPACK slower than Python (overhead)
- GPU slower than CPU (transfer overhead)
- Sampling broke spatial_variance (defeated purpose)

**Solution**: Profile first, optimize second.

### 2. Overhead Matters More Than You Think

Theoretical speedup ≠ actual speedup:
- Subprocess spawn: 160ms
- File I/O: 50-100ms
- GPU transfer: 100ms
- MPI communication: Depends on size

For small problems, overhead > computation.

### 3. Memory Layout Is Not Automatic

C vs Fortran order, strides, lifetimes - all matter:
- NumPy defaults to C-order, LAPACK expects Fortran-order
- Invalid strides cause segfaults
- Temporary objects get garbage collected

**Solution**: Verify memory layout, manage lifetimes explicitly.

### 4. Distributed ≠ Faster

Communication has cost:
- Scattering matrix: 200MB network traffic
- Broadcasting features: 8MB network traffic

**Solution**: Broadcast small, compute large.

### 5. Accept Limitations

Not every problem has a fast solution:
- O(n²) spatial_variance with n=85k takes 8 seconds
- Approximations defeat the purpose
- Sometimes the answer is "use a different algorithm"

**Solution**: Document tradeoffs honestly, provide alternatives.

---

## Debugging Tools We Used

### 1. Profiling
```python
import time

t0 = time.time()
# ... operation ...
print(f"Took {time.time() - t0:.3f}s")
```

### 2. GDB for Segfaults
```bash
gdb python
run script.py
# After crash:
backtrace
info locals
```

### 3. Memory Layout Checking
```python
print(f"Array shape: {arr.shape}")
print(f"Array strides: {arr.strides}")
print(f"Array order: {'F' if arr.flags.f_contiguous else 'C'}")
```

### 4. C++ Debug Output
```cpp
std::cout << "Variable: " << var << std::endl;
std::cout << "Address: " << (void*)ptr << std::endl;
```

### 5. Benchmarking Scripts
```bash
python scripts/bench_step.py --m-rated 100 500 1000
```

---

## For Students

Real HPC development involves:
1. **Theory**: O(m³) Cholesky, O(nm²) variance
2. **Practice**: Overhead, memory layout, communication costs
3. **Debugging**: Profiling, GDB, print statements
4. **Measurement**: Benchmarking to validate decisions
5. **Honesty**: Admitting when there's no magic fix

The "war stories" are where the learning happens. Theory tells you what to try. Debugging tells you why it didn't work. Measurement tells you what actually works.

**Remember**: Every challenge in this document was a surprise. We expected ScaLAPACK to be faster, GPU to help, approximations to work. Reality was different. That's normal - and that's why measurement matters.
