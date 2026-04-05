# PyBind11 Integration for In-Memory Bridge

## Motivation

Currently, the ScaLAPACK backend communicates with Python via:
1. **File I/O**: Write input matrices/features to binary files
2. **Subprocess**: Launch `scalapack_gp_fit` via `mpirun`/`srun`
3. **File I/O**: Read output alpha/chol from binary files

**Overhead breakdown** (measured for m=5000):
- File writes: ~0.5-1.0s
- Subprocess launch + MPI init: ~0.3-0.5s
- File reads: ~0.5-1.0s
- **Total overhead**: ~1.5-2.5s per fit

For **interactive workflows** (CLI with 10-50 iterations), this overhead dominates.

## Goal: In-Memory Bridge via PyBind11

Replace file I/O and subprocess overhead with direct in-memory calls:
```python
# Current (subprocess + file I/O)
result = fit_exact_gp_scalapack_from_rated(...)  # ~2s overhead

# Proposed (in-memory PyBind11)
result = fit_exact_gp_native(...)  # ~0s overhead
```

**Expected speedup**: Eliminate 1.5-2.5s overhead per fit.

## Architecture Options

### Option 1: PyBind11 Wrapper (Single-Process, No MPI)

**Approach**: Wrap existing C++ code with PyBind11, run single-threaded or OpenMP.

**Pros**:
- Simple: Direct function calls from Python
- No MPI complexity
- No subprocess overhead
- Zero file I/O

**Cons**:
- **No ScaLAPACK**: Can only use LAPACK (single-node)
- Limited to m < 10k (memory constraints)
- Miss out on multi-node scaling

**Implementation**:
```cpp
// native/pybind11_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::dict fit_gp_lapack(
    py::array_t<double> K_rr,
    py::array_t<double> y,
    double noise
) {
    // Direct LAPACK calls (dpotrf, dpotrs)
    // Return alpha, chol, logdet as Python dict
}

PYBIND11_MODULE(poetry_gp_native, m) {
    m.def("fit_gp_lapack", &fit_gp_lapack);
}
```

**Usage**:
```python
from poetry_gp_native import fit_gp_lapack

result = fit_gp_lapack(K_rr, y, noise)
# Returns: {'alpha': ndarray, 'chol': ndarray, 'logdet': float}
```

**When to use**: Small-to-medium problems (m < 5000) in interactive sessions.

---

### Option 2: PyBind11 + MPI (Multi-Process, Shared Memory)

**Approach**: Launch persistent MPI daemon, communicate via shared memory or MPI one-sided.

**Pros**:
- **ScaLAPACK available**: Multi-node scaling
- **Persistent MPI**: Amortize init cost across many fits
- **Shared memory**: Zero-copy for local node data

**Cons**:
- **Complex**: MPI daemon lifecycle management
- **Memory**: Daemon keeps data in memory
- **Portability**: Shared memory/MPI one-sided not universally supported

**Implementation** (conceptual):
```cpp
// Daemon process with MPI communicator
class ScaLAPACKDaemon {
    MPI_Comm comm_;
    int rank_, size_;

public:
    void init(int argc, char** argv);
    void run_loop();  // Listen for fit requests
    NativeResult fit(SharedMemoryView x_rated, SharedMemoryView y);
};
```

**Python side**:
```python
# Spawn daemon once (persistent)
daemon = ScaLAPACKDaemon(nprocs=8)

# Many fits with no spawn overhead
for iteration in range(50):
    result = daemon.fit(x_rated, y)  # In-memory, no I/O
```

**Challenges**:
1. **Shared memory**: Requires `MPI_Win_allocate_shared` (MPI 3.0+)
2. **Daemon lifecycle**: What if daemon crashes?
3. **Memory management**: Who owns the memory?

---

### Option 3: Hybrid Approach (PyBind11 for Small, MPI for Large)

**Approach**: Automatic fallback based on problem size.

```python
def fit_exact_gp_auto(x_rated, y, ...):
    m = x_rated.shape[0]

    if m < 5000:
        # In-memory PyBind11 (LAPACK, no MPI)
        return fit_gp_lapack(K_rr, y, noise)
    else:
        # ScaLAPACK with file I/O (existing approach)
        return fit_exact_gp_scalapack_from_rated(...)
```

**Pros**:
- **Best of both worlds**: Fast for small, scalable for large
- **Simple**: No daemon complexity for small problems
- **Incremental**: Can implement PyBind11 first, keep ScaLAPACK as-is

**Cons**:
- Two codepaths to maintain

---

## Recommendation: Start with Option 3

### Phase 1: PyBind11 LAPACK Wrapper (m < 5000)
**Goal**: Eliminate subprocess overhead for interactive CLI sessions.

**Implementation steps**:
1. Add PyBind11 to CMake dependencies
2. Create `native/pybind11_lapack.cpp` with LAPACK wrappers
3. Build Python extension module `poetry_gp_native`
4. Add `fit_backend="native_lapack"` to `run_blocked_step()`
5. Benchmark against Python (scipy) and existing ScaLAPACK subprocess

**Expected outcome**:
- ~2s faster per fit for m=1000-5000
- CLI interactions feel instant (50 iterations × 2s = 100s saved!)

### Phase 2 (Future): MPI Daemon with Shared Memory
**Goal**: Scale to large m (10k-30k) without subprocess overhead.

**Challenges to solve**:
- Persistent MPI daemon lifecycle
- Shared memory or efficient data transfer
- Error handling and recovery

---

## CMake Integration for PyBind11

### Option A: Fetch PyBind11 (Recommended)

```cmake
# CMakeLists.txt
include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Build Python extension module
pybind11_add_module(poetry_gp_native native/pybind11_lapack.cpp)
target_link_libraries(poetry_gp_native PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
```

**Pros**: No system dependency, works on cluster
**Cons**: Requires internet access during build

### Option B: System PyBind11

```cmake
find_package(pybind11 REQUIRED)
pybind11_add_module(poetry_gp_native native/pybind11_lapack.cpp)
```

**Pros**: Uses system package (if available)
**Cons**: May not be installed on cluster

### Option C: Git Submodule

```bash
git submodule add https://github.com/pybind/pybind11.git external/pybind11
```

```cmake
add_subdirectory(external/pybind11)
pybind11_add_module(poetry_gp_native native/pybind11_lapack.cpp)
```

**Pros**: Version pinned, no fetch during build
**Cons**: Extra submodule to manage

**Recommendation**: Start with **Option A (FetchContent)** for simplicity.

---

## Prototype: Minimal PyBind11 LAPACK Wrapper

```cpp
// native/pybind11_lapack.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

extern "C" {
    void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
    void dpotrs_(const char* uplo, const int* n, const int* nrhs,
                 const double* a, const int* lda, double* b, const int* ldb, int* info);
}

py::dict fit_gp_lapack(
    py::array_t<double, py::array::c_style | py::array::forcecast> K_rr_py,
    py::array_t<double, py::array::c_style | py::array::forcecast> y_py
) {
    auto K_rr_buf = K_rr_py.request();
    auto y_buf = y_py.request();

    if (K_rr_buf.ndim != 2 || K_rr_buf.shape[0] != K_rr_buf.shape[1])
        throw std::runtime_error("K_rr must be square 2D array");
    if (y_buf.ndim != 1 || y_buf.shape[0] != K_rr_buf.shape[0])
        throw std::runtime_error("y must be 1D array matching K_rr");

    int n = static_cast<int>(K_rr_buf.shape[0]);

    // Copy input (LAPACK modifies in-place)
    std::vector<double> chol(n * n);
    std::memcpy(chol.data(), K_rr_buf.ptr, n * n * sizeof(double));

    std::vector<double> alpha(n);
    std::memcpy(alpha.data(), y_buf.ptr, n * sizeof(double));

    // Cholesky factorization
    const char uplo = 'L';
    int info_potrf = 0;
    dpotrf_(&uplo, &n, chol.data(), &n, &info_potrf);

    if (info_potrf != 0)
        throw std::runtime_error("Cholesky factorization failed");

    // Compute logdet
    double logdet = 0.0;
    for (int i = 0; i < n; ++i) {
        logdet += 2.0 * std::log(chol[i * n + i]);
    }

    // Solve for alpha
    int nrhs = 1;
    int info_potrs = 0;
    dpotrs_(&uplo, &n, &nrhs, chol.data(), &n, alpha.data(), &n, &info_potrs);

    if (info_potrs != 0)
        throw std::runtime_error("Linear solve failed");

    // Return results
    py::dict result;
    result["alpha"] = py::array_t<double>(n, alpha.data());
    result["chol_lower"] = py::array_t<double>({n, n}, chol.data());
    result["logdet"] = logdet;
    result["info_potrf"] = info_potrf;
    result["info_potrs"] = info_potrs;

    return result;
}

PYBIND11_MODULE(poetry_gp_native, m) {
    m.doc() = "Native LAPACK/ScaLAPACK GP fitting";
    m.def("fit_gp_lapack", &fit_gp_lapack,
          "Fit GP using LAPACK (in-memory, no MPI)",
          py::arg("K_rr"), py::arg("y"));
}
```

**Build**:
```cmake
# Add to CMakeLists.txt
include(FetchContent)
FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(poetry_gp_native native/pybind11_lapack.cpp)
target_link_libraries(poetry_gp_native PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
```

**Usage**:
```python
import numpy as np
from poetry_gp_native import fit_gp_lapack

K_rr = np.eye(100)  # 100×100 kernel matrix
y = np.random.randn(100)

result = fit_gp_lapack(K_rr, y)
print(f"alpha: {result['alpha'].shape}")
print(f"logdet: {result['logdet']}")
```

---

## Performance Comparison (Projected)

| Backend | m=1000 | m=5000 | m=10000 | Notes |
|---------|--------|--------|---------|-------|
| Python (scipy) | 0.1s | 2.0s | 15s | Baseline |
| ScaLAPACK (subprocess) | **2.5s** | **4.0s** | 8s | Overhead dominates small m |
| **PyBind11 LAPACK** | **0.1s** | **2.0s** | 15s | Same as scipy, zero overhead |
| ScaLAPACK (in-memory daemon) | 0.2s | 2.5s | **5s** | Future: eliminates overhead |

**Takeaway**: PyBind11 LAPACK makes small problems (m < 5000) competitive with scipy while preparing infrastructure for in-memory ScaLAPACK daemon.

---

## Next Steps

1. **Decision**: Approve PyBind11 integration approach (Option 3 recommended)
2. **Implement**: Minimal PyBind11 LAPACK wrapper (prototype above)
3. **Benchmark**: Compare against scipy and subprocess ScaLAPACK
4. **Integrate**: Add `fit_backend="native_lapack"` to `run_blocked_step()`
5. **Test**: Verify interactive CLI performance improvement
6. **Document**: Update README and benchmarking guide

**Estimated effort**: 1-2 days for Phase 1 (PyBind11 LAPACK wrapper)

---

## Open Questions

1. **Thread safety**: Can we safely call LAPACK from Python threads? (Probably yes with GIL)
2. **OpenMP**: Should we enable OpenMP in PyBind11 module for multi-threaded BLAS?
3. **Error handling**: How to gracefully handle LAPACK failures in PyBind11?
4. **Memory**: Should we support pre-allocated output arrays (zero-copy)?
5. **Build system**: How to distribute compiled `.so` for different platforms?

Let's discuss and decide on the implementation plan!
