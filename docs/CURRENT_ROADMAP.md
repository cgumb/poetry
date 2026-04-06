# Poetry GP Roadmap - Current Status (April 2026)

## ✅ Recently Completed

### Backend Infrastructure (March-April 2026)
- ✅ **PyBind11 LAPACK integration** - In-memory fit/score, zero subprocess overhead
- ✅ **GPU scoring backend** - CuPy/CUDA integration, 3-4.6× speedup for m ≥ 500
- ✅ **Automatic backend selection** - Smart defaults based on problem size and hardware
- ✅ **Hyperparameter optimization for native_lapack** - Nelder-Mead in log-space
- ✅ **Backend selection documentation** - BACKEND_SELECTION.md with benchmarks
- ✅ **Acquisition function documentation** - Honest tradeoffs (max_variance vs spatial_variance)
- ✅ **ScaLAPACK distributed kernel assembly** - Milestone 1B complete
- ✅ **Interactive CLI timing display** - Shows fit/optimize/score/select breakdown

### Key Decisions Made
- ✅ **Deprioritized MPI daemon scoring** - PyBind11 + GPU are faster and simpler
- ✅ **Documented spatial_variance limitations** - O(n²) makes it impractical for n > 10k
- ✅ **Established backend hierarchy** - native_lapack → python → gpu (opportunistic)

---

## 🎯 Immediate Priorities (This Week)

### 1. Update Main README ⭐
**Status**: Not done  
**Why**: README probably still shows old workflows  
**What to add**:
- Quick start with auto backend selection
- Performance comparison table
- When to use which backend
- Interactive CLI usage

**Effort**: 30 minutes  
**Impact**: HIGH (user-facing documentation)

**Example content**:
```python
# Automatic backend selection (recommended)
result = run_blocked_step(
    embeddings,
    rated_indices,
    ratings,
    fit_backend="auto",      # Chooses native_lapack/python/scalapack
    score_backend="auto",    # Chooses gpu/native_lapack/python
)

# Performance: m=1000, n=85k
# Fit: 0.03s (native_lapack)
# Score: 0.6s (GPU)
# Select: 0.001s (max_variance)
# Total: 0.63s per iteration
```

### 2. Full Pipeline Benchmark ⭐⭐
**Status**: Not done  
**Why**: Need end-to-end performance validation  
**What to measure**:
- Python baseline
- native_lapack fit + score
- native_lapack fit + GPU score
- Various problem sizes (m=100, 500, 1000, 5000)
- Interactive workflow (10 iterations)

**Effort**: 1-2 hours (write bench_full_pipeline.py)  
**Impact**: MEDIUM (validates all the work)

**Deliverable**: Performance table showing speedups across backends

---

## 📈 Short-Term Optimizations (This Month)

### 3. Warm-Start Hyperparameter Optimization ⭐⭐⭐
**Status**: Not started  
**Opportunity**: In interactive sessions, hyperparameters change slowly

**Implementation**:
```python
# In interactive CLI or multi-round workflow
if previous_state is not None:
    # Start from previous optimal hyperparameters
    initial_length_scale = previous_state.length_scale
    initial_variance = previous_state.variance
    initial_noise = previous_state.noise
else:
    # Use defaults
    initial_length_scale = 1.0
    ...

result = run_blocked_step(
    ...,
    optimize_hyperparameters=True,
    length_scale=initial_length_scale,  # Warm start
    variance=initial_variance,
    noise=initial_noise,
)
```

**Expected benefit**:
- Reduces optimization from ~50 iterations → 5-10 iterations
- 5× faster hyperparameter optimization
- Especially valuable for interactive sessions

**Effort**: LOW (just plumbing, no new algorithms)  
**Impact**: HIGH for interactive workflows

### 4. Lazy Variance Computation ⭐⭐
**Status**: Not started  
**Opportunity**: Don't need variance for all 85k candidates for exploration

**Implementation**:
```python
# Option A: Top-K filtering
# 1. Compute mean for all candidates (fast, O(n×m))
# 2. Filter to top 1000 by mean
# 3. Only compute variance for those 1000

# Option B: Acquisition function aware
# For exploitation: Only need mean
# For exploration: Need variance, but can pre-filter

def run_blocked_step(..., variance_filter_k: int = 10000):
    # Compute mean for all
    mean = score_mean_only(...)
    
    # Filter to top-K by mean (or by distance from rated, etc.)
    candidate_indices = select_top_k_candidates(mean, k=variance_filter_k)
    
    # Only compute variance for filtered set
    variance = score_variance_only(candidate_indices)
```

**Expected benefit**:
- For n=85k, variance_filter_k=1000: 85× reduction in variance computation
- Makes spatial_variance more practical (1000² vs 85k²)

**Effort**: LOW-MEDIUM (need to refactor scoring logic)  
**Impact**: MEDIUM-HIGH (enables better exploration strategies)

### 5. Analytic Gradients for Hyperparameter Optimization ⭐
**Status**: Not started  
**Opportunity**: Replace numerical gradients with analytic gradients

**Implementation** (from Rasmussen & Williams 5.4.1):
```python
def compute_lml_gradient(K, alpha, theta):
    """
    Compute ∂(log marginal likelihood)/∂θ
    
    ∂lml/∂θ = 0.5 * tr((α α^T - K⁻¹) ∂K/∂θ)
    
    where α = K⁻¹y
    """
    # Compute α α^T - K⁻¹
    K_inv = cho_solve((L, True), np.eye(m))  # Expensive!
    term = np.outer(alpha, alpha) - K_inv
    
    # Compute ∂K/∂θ for each hyperparameter
    dK_dlength_scale = compute_kernel_derivative_length_scale(K, theta)
    dK_dvariance = compute_kernel_derivative_variance(K, theta)
    dK_dnoise = compute_kernel_derivative_noise(K, theta)
    
    # Trace computation
    grad_length_scale = 0.5 * np.trace(term @ dK_dlength_scale)
    grad_variance = 0.5 * np.trace(term @ dK_dvariance)
    grad_noise = 0.5 * np.trace(term @ dK_dnoise)
    
    return [grad_length_scale, grad_variance, grad_noise]

# Use L-BFGS-B with analytic gradients
result = minimize(
    objective_with_gradient,
    init_log_params,
    method="L-BFGS-B",
    jac=True,  # Use analytic gradients
)
```

**Expected benefit**:
- 30-50% fewer iterations
- More accurate gradients than finite differences
- Better convergence

**Effort**: MEDIUM (need to implement gradient computation correctly)  
**Impact**: MEDIUM (mainly helps for large m where optimization dominates)

---

## 🔬 Research / Advanced (Future)

### 6. Distributed Scoring via ScaLAPACK (Revisit if Needed)
**Status**: Exists but deprioritized  
**Reason**: PyBind11 + GPU already faster for realistic problem sizes  
**When to revisit**: If n > 100k and no GPU available

### 7. Approximate GP Methods (m > 50k)
**Status**: Not started  
**Options**:
- Inducing points (sparse GP)
- Stochastic variational inference
- Local GP (partition space)

**Effort**: HIGH (research-level)  
**Impact**: Enables m > 50k-100k

### 8. Better Acquisition Functions
**Status**: Not started  
**Options**:
- Knowledge gradient
- Predictive entropy search
- Batch acquisition functions (select k diverse points simultaneously)

**Effort**: MEDIUM-HIGH  
**Impact**: Better exploration quality

---

## 📊 Success Metrics

### Current Performance (n=85k, m=1000)
```
Backend: native_lapack fit + GPU score
Fit:      0.03s
Score:    0.60s (GPU)
Select:   0.001s (max_variance)
Total:    0.63s per iteration
```

### Target Performance with Optimizations

**With Warm-Start HP Optimization**:
```
HP Optimize: 0.5s (5× faster, only when needed)
Iteration:   0.63s (unchanged)
```

**With Lazy Variance + Warm-Start**:
```
HP Optimize: 0.5s
Fit:         0.03s
Score:       0.20s (only compute variance for top-1k)
Select:      0.001s
Total:       0.23s per iteration (2.7× faster)
```

**With All Optimizations**:
```
HP Optimize: 0.3s (warm-start + analytic gradients)
Iteration:   0.23s (lazy variance)
10 iterations: ~2.6s total (vs current ~6.6s)
```

---

## 📚 Documentation Gaps

### High Priority
- [ ] Update main README with new backends
- [ ] Quick start guide for new users
- [ ] Performance tuning guide (when to use what)

### Medium Priority
- [ ] Interactive CLI user guide
- [ ] Hyperparameter optimization guide
- [ ] Troubleshooting guide

### Low Priority
- [ ] ScaLAPACK build guide (for HPC clusters)
- [ ] GPU setup guide
- [ ] Development guide for contributors

---

## 🎓 Pedagogical Value

Current codebase demonstrates:
- ✅ Automatic backend selection (abstraction)
- ✅ CPU vs GPU tradeoffs (heterogeneous computing)
- ✅ O(n) vs O(n²) algorithmic complexity (acquisition functions)
- ✅ Distributed computing with ScaLAPACK (when needed)
- ✅ Performance profiling and optimization

Future additions would add:
- Warm-start optimization (iterative algorithms)
- Lazy evaluation (compute only what's needed)
- Approximate methods (accuracy vs speed tradeoffs)

---

## 🚀 Recommended Next Steps

### This Week:
1. Update README (30 min)
2. Write full pipeline benchmark script (1-2 hours)
3. Run benchmarks and document results

### This Month:
1. Implement warm-start hyperparameter optimization
2. Implement lazy variance computation
3. Benchmark and document improvements

### Future:
1. Analytic gradients for HP optimization
2. Consider approximate GP methods if m > 50k becomes common
3. Better acquisition functions if exploration quality is insufficient

---

## Questions to Inform Priorities

1. **What are typical problem sizes in production?**
   - If m < 5000 consistently: Current backends are excellent
   - If m > 10k common: ScaLAPACK and approximations matter more

2. **How many iterations per interactive session?**
   - If 5-10 iterations: Warm-start is critical
   - If 100+ iterations: HP optimization frequency matters

3. **Is spatial diversity important?**
   - If yes: Lazy variance enables better acquisition functions
   - If no: max_variance + current backends are optimal

4. **GPU availability?**
   - If GPUs common: Current GPU backend is great
   - If no GPUs: Focus on CPU optimizations

These answers will help prioritize the roadmap items.
