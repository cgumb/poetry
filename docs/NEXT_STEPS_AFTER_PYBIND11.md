# Next Steps After PyBind11 Integration

## ✅ Completed: PyBind11 Fit Backend

PyBind11 LAPACK integration is complete and all tests passing:
- Eliminates 1.5-2.5s subprocess overhead for m < 5000
- Perfect correctness vs scipy (machine precision)
- Production-ready with `fit_backend="native_lapack"`

---

## 🎯 High-Priority Next Steps

### 1. **Integrate PyBind11 Scoring** ⭐ **HIGHEST IMPACT**

**Status**: `predict_native()` exists and works, but NOT integrated into `run_blocked_step()`

**Current situation**:
```python
# src/poetry_gp/backends/blocked.py line 87
score_backend: str = "python",  # "python", "daemon", "auto", "gpu", "none"
#                                  ^^^^^ "native_lapack" missing!
```

**What to do**:
1. Add `"native_lapack"` option to `score_backend` parameter
2. Wire up `predict_native()` in blocked.py scoring logic
3. Benchmark vs python/gpu scoring

**Expected benefit**: 2-4× speedup for scoring with zero MPI complexity

**Estimated effort**: 1-2 hours

---

### 2. **Test/Enable GPU Scoring**

**Status**: Infrastructure exists (`bench_scoring.py` line 155, `gpu_scoring.py`)

**What to do**:
1. Check if GPU available on cluster: `nvidia-smi`
2. Test: `python scripts/bench_scoring.py --output-csv results/gpu_bench.csv`
3. If working, document performance vs CPU

**Expected benefit**: 10-50× speedup if GPU available

**Estimated effort**: 30 minutes to test, document

---

### 3. **Automatic Backend Selection**

Implement smart defaults based on problem size:

```python
def select_fit_backend(m: int) -> str:
    if m < 5000 and is_native_available():
        return "native_lapack"  # PyBind11 (instant)
    elif m < 10000:
        return "python"  # Scipy (good enough)
    else:
        return "native_reference"  # ScaLAPACK MPI

def select_score_backend(n: int, has_gpu: bool) -> str:
    if has_gpu:
        return "gpu"  # Best option if available
    elif n > 10000 and is_native_available():
        return "native_lapack"  # PyBind11 BLAS
    else:
        return "python"  # Scipy baseline
```

**Benefit**: Users don't need to think about backends

**Estimated effort**: 2-3 hours

---

## ⚠️ Daemon Scoring Status

### Fixed Issues:
- ✅ Daemon startup (launcher options now work with srun)

### Remaining Issues:
1. **File I/O overhead**: Daemon still reads/writes files (~50-100ms)
2. **Broadcast overhead**: Broadcasts 3.5MB for m=500 (~50ms)
3. **Poor speedup**: For realistic n < 50k, overhead dominates

### Decision: **Deprioritize MPI Daemon Scoring**

**Why?**
- PyBind11 + multi-threaded BLAS is faster and simpler
- GPU is even better if available
- Daemon only wins for n > 100k (rare in practice)

**Recommendation**: Focus on PyBind11 and GPU instead

---

## 📊 Performance Targets (n=25,000 candidates, m=1000)

| Backend | Fit Time | Score Time | Total | Notes |
|---------|----------|------------|-------|-------|
| Python baseline | 0.1s | 5.0s | 5.1s | scipy serial |
| **PyBind11 fit only** | **0.0s** | 5.0s | 5.0s | ✅ Instant fit |
| **PyBind11 fit + score** | **0.0s** | **1.2s** | **1.2s** | 🎯 Next step |
| GPU fit + score | 0.0s | 0.2s | 0.2s | 🚀 If available |

---

## 🛠️ Implementation Checklist

### Immediate (This Week):
- [ ] Add `score_backend="native_lapack"` to `run_blocked_step()`
- [ ] Wire up `predict_native()` in blocked.py
- [ ] Test PyBind11 scoring correctness
- [ ] Benchmark PyBind11 scoring performance
- [ ] Test GPU scoring availability on cluster

### Short-term (This Month):
- [ ] Implement automatic backend selection
- [ ] Document backend performance guide
- [ ] Update README with backend recommendations
- [ ] Benchmark full pipeline with PyBind11

### Nice-to-have (Later):
- [ ] Fix daemon file I/O (only if needed for n > 100k)
- [ ] Hyperparameter optimization with PyBind11
- [ ] Persistent daemon for interactive CLI

---

## 📝 Code Locations

Key files to modify:

1. **src/poetry_gp/backends/blocked.py**:
   - Line 87: Add "native_lapack" to score_backend options
   - Line ~180: Add elif block for native_lapack scoring

2. **src/poetry_gp/backends/native_lapack.py**:
   - `predict_native()` already exists ✅
   - May need batch processing wrapper

3. **docs/BENCHMARKING_GUIDE.md**:
   - Update with PyBind11 performance numbers
   - Document when to use each backend

---

## ❓ Open Questions

1. **GPU availability**: Do we have GPUs on the cluster?
2. **Typical problem sizes**: What are realistic m and n values for production?
3. **Interactive vs batch**: How many iterations per session?
4. **Hyperparameter optimization**: How often do we optimize vs reuse?

These answers will inform backend selection priorities.
