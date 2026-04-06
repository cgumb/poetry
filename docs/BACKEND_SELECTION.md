# Automatic Backend Selection Guide

## Overview

Poetry GP now supports **automatic backend selection** that chooses the optimal fit and scoring backends based on problem size and available hardware.

**Key benefits**:
- ✅ **No manual tuning required** - backends selected automatically
- ⚡ **Optimal performance** - uses fastest backend for your problem size
- 🔧 **Manual override available** - expert users can specify backends explicitly
- 📊 **Config-based control** - set preferences via `GPConfig`

---

## Quick Start

### Use Auto Selection (Recommended)

```python
from poetry_gp.backends.blocked import run_blocked_step

# Automatically selects best backends for your problem size
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    fit_backend="auto",      # ← Auto-select fit backend
    score_backend="auto",    # ← Auto-select score backend
)
```

**That's it!** The system will automatically choose:
- **Fit backend**: Based on m (number of rated points)
- **Score backend**: Based on m and n (candidates), plus GPU availability

---

## How Backend Selection Works

### Fit Backend Selection

| Problem Size | Backend Selected | Reason |
|--------------|------------------|---------|
| **m < 5k** (+ native available) | `native_lapack` | Instant fit, zero overhead |
| **m < 10k** | `python` | Scipy good enough |
| **m ≥ 10k** | `native_reference` | ScaLAPACK MPI for large problems |

### Score Backend Selection

| Problem Size | GPU Available | Backend Selected | Speedup |
|--------------|---------------|------------------|---------|
| **m < 500** | Yes | `native_lapack` | 1.1-1.2× (GPU cold-start overhead) |
| **m ≥ 500** | Yes | `gpu` | 3-4.6× faster! |
| Any | No (+ native) | `native_lapack` | 1.1-1.2× faster |
| Any | No (no native) | `python` | Baseline |

**Performance data from benchmarks**:
- m=100: GPU slower (cold-start overhead)
- m=500-5k: GPU 3-4.6× faster
- m>5k: GPU 2.3-2.7× faster

---

## Configuration-Based Control

### Using GPConfig

```python
from poetry_gp.config import GPConfig
from poetry_gp.backends.blocked import run_blocked_step

# Create custom config
config = GPConfig(
    fit_backend="native_lapack",   # Explicit choice
    score_backend="auto",          # Auto for scoring
    length_scale=2.0,
    noise=1e-4,
)

# Use config in run_blocked_step
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    **config.to_dict()  # ← Unpack config as kwargs
)
```

### Built-in Presets

```python
from poetry_gp.config import FAST_CONFIG, ACCURATE_CONFIG, LARGE_SCALE_CONFIG

# Fast: Prioritizes speed
result = run_blocked_step(embeddings, rated_indices, ratings, **FAST_CONFIG.to_dict())

# Accurate: Optimizes hyperparameters
result = run_blocked_step(embeddings, rated_indices, ratings, **ACCURATE_CONFIG.to_dict())

# Large-scale: MPI + GPU for huge problems
result = run_blocked_step(embeddings, rated_indices, ratings, **LARGE_SCALE_CONFIG.to_dict())
```

### Config Presets Details

**FAST_CONFIG**:
```python
GPConfig(
    fit_backend="native_lapack",
    score_backend="auto",
    optimize_hyperparameters=False,
)
```

**ACCURATE_CONFIG**:
```python
GPConfig(
    fit_backend="auto",
    score_backend="auto",
    optimize_hyperparameters=True,
    optimizer_maxiter=100,
    noise=1e-4,
)
```

**LARGE_SCALE_CONFIG**:
```python
GPConfig(
    fit_backend="native_reference",  # ScaLAPACK
    score_backend="gpu",
    scalapack_nprocs=16,
    block_size=4096,
)
```

---

## Manual Backend Override

### Specify Backends Explicitly

```python
# Expert user: I know exactly what I want
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    fit_backend="native_lapack",   # Force native_lapack
    score_backend="gpu",            # Force GPU
)
```

### Available Backends

**Fit backends**:
- `"auto"` - Automatic selection (recommended)
- `"python"` - Scipy (always available, baseline)
- `"native_lapack"` - PyBind11 LAPACK (m < 5k, zero overhead)
- `"native_reference"` - ScaLAPACK MPI (m > 10k, parallel)

**Score backends**:
- `"auto"` - Automatic selection (recommended)
- `"python"` - Scipy (baseline)
- `"native_lapack"` - PyBind11 BLAS (1.1-1.2× faster)
- `"gpu"` - CuPy/CUDA (2-5× faster if m ≥ 500)
- `"daemon"` - MPI daemon (deprecated, use GPU instead)
- `"none"` - Skip scoring (fit-only workflows)

---

## Checking Backend Availability

### Print Status

```python
from poetry_gp.backends.backend_selection import print_backend_status

print_backend_status(verbose=True)
```

**Output**:
```
Backend Availability:
  Python (scipy):     ✓ Always available
  Native LAPACK:      ✓ Available
  GPU (CuPy):         ✓ Available
  ScaLAPACK (MPI):    ✓ Always available

Recommendations:
  Fit backend:
    m < 5k:   native_lapack (instant) or python
    m < 10k:  python
    m >= 10k: native_reference (ScaLAPACK MPI)

  Score backend:
    m < 500:  native_lapack (GPU cold-start overhead)
    m >= 500: gpu (3-4× faster)
```

### Programmatic Check

```python
from poetry_gp.backends.backend_selection import get_backend_info

info = get_backend_info()
# Returns: {"native_lapack": bool, "gpu": bool, "scalapack": bool, "python": bool}

if info["gpu"]:
    print("GPU available, will use for large m")
```

---

## Testing

### Test Backend Selection

```bash
python scripts/test_backend_selection.py
```

This tests:
- Selection logic for different problem sizes
- Auto backend integration with `run_blocked_step`
- Config usage and presets
- Manual overrides

**Expected output**:
```
Backend Selection Logic Test
============================================================

Backend Availability:
  Python (scipy):     ✓ Always available
  Native LAPACK:      ✓ Available
  GPU (CuPy):         ✓ Available
  ScaLAPACK (MPI):    ✓ Always available

Fit Backend Selection:
  Small (m=100)        → native_lapack
  Medium (m=1000)      → native_lapack
  Large (m=5000)       → python
  Very Large (m=15000) → native_reference

Score Backend Selection:
  Small m (m=100)      → native_lapack
  Medium m (m=500)     → gpu
  Large m (m=2000)     → gpu
  Very Large (m=5000)  → gpu

ALL TESTS PASSED ✓
```

---

## Interactive CLI Integration

### Save Config to File

```python
from poetry_gp.config import GPConfig
import json

# Create config
config = GPConfig(
    fit_backend="native_lapack",
    score_backend="gpu",
    length_scale=2.0,
)

# Save to file
with open("gp_config.json", "w") as f:
    json.dump(config.to_dict(), f, indent=2)
```

### Load Config from File

```python
import json
from poetry_gp.config import GPConfig

# Load from file
with open("gp_config.json") as f:
    config_dict = json.load(f)

config = GPConfig.from_dict(config_dict)

# Use in your workflow
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    **config.to_dict()
)
```

---

## Migration Guide

### Before (Manual Backend Selection)

```python
# Old: Manually specify backends
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    fit_backend="python",
    score_backend="python",
)
```

### After (Automatic Selection)

```python
# New: Let system choose optimal backends
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    fit_backend="auto",      # ← Changed
    score_backend="auto",    # ← Changed
)
```

**Note**: `"auto"` is now the default, so you can omit these parameters entirely:

```python
# Even simpler: defaults to "auto"
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
)
```

---

## Best Practices

### ✅ Do

- Use `fit_backend="auto"` and `score_backend="auto"` (recommended)
- Use `GPConfig` for persistent settings
- Check backend availability with `print_backend_status()`
- Override backends when you have specific requirements

### ❌ Don't

- Hardcode backends without profiling first
- Use GPU for m < 500 (cold-start overhead dominates)
- Use ScaLAPACK (`native_reference`) for m < 10k (overhead dominates)
- Assume "native" is always faster (profile your workload!)

---

## Performance Summary

### Fit Performance (Based on m)

| m | Python | Native LAPACK | ScaLAPACK | Winner |
|---|--------|---------------|-----------|--------|
| 100 | 0.1s | **0.0s** | 2.5s | Native (instant) |
| 1k | 0.1s | **0.0s** | 2.5s | Native (instant) |
| 5k | 0.5s | **0.1s** | 2.6s | Native or Python |
| 10k | 2.0s | 0.5s | **3.0s** | Native or ScaLAPACK |
| 20k | 10s | N/A | **6s** | ScaLAPACK |

### Score Performance (Based on m, n=10k)

| m | Python (1t) | Python (8t) | Native LAPACK | GPU | Winner |
|---|-------------|-------------|---------------|-----|--------|
| 100 | 0.058s | 0.052s | **0.048s** | 0.271s | Native |
| 500 | 0.353s | 0.353s | 0.310s | **0.076s** | GPU (4.6×) |
| 1k | 0.694s | 0.717s | 0.620s | **0.193s** | GPU (3.6×) |
| 2k | 1.646s | 1.650s | 1.470s | **0.527s** | GPU (3.1×) |
| 5k | 6.359s | 6.342s | 5.600s | **2.378s** | GPU (2.7×) |
| 10k | 19.965s | 19.952s | 17.700s | **8.239s** | GPU (2.4×) |

---

## References

- Implementation: `src/poetry_gp/backends/backend_selection.py`
- Configuration: `src/poetry_gp/config.py`
- Integration: `src/poetry_gp/backends/blocked.py`
- Tests: `scripts/test_backend_selection.py`
- Benchmarks: `scripts/bench_scoring.py`, `scripts/gpu_scoring_bench.slurm`
