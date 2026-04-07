## Frequently Asked Questions
### Companion to Poetry GP Presentation

---

## Gaussian Process Concepts

### Q: Why start with ridge regression instead of going straight to GPs?

**A**: Pedagogical progression. Most students know ridge regression from ML courses. Starting there provides:

1. **Familiar foundation**: Linear models, regularization, Bayesian interpretation
2. **Natural motivation**: Why move from primal ($p \times p$) to dual ($m \times m$)?
3. **Kernel trick emerges naturally**: All you need is inner products
4. **GP is the generalization**: Same dual form, just replace linear kernel with RBF

The key insight: Ridge regression with kernel trick **is** a GP. We're not changing models, just generalizing the kernel function.

### Q: What's the difference between the prior and the posterior?

**A**:

**Prior** (before seeing data):
- Encodes our beliefs about the function before observations
- $f \sim \mathcal{GP}(0, k(x, x'))$
- Says: "Functions should be smooth (nearby points have similar values)"

**Posterior** (after seeing data):
- Updates beliefs given observed ratings
- $f \mid y \sim \mathcal{GP}(\mu(x), \sigma^2(x))$
- Posterior mean $\mu(x)$: Our best guess at preference
- Posterior variance $\sigma^2(x)$: How uncertain we are

Key property: Uncertainty **shrinks** near observations, **grows** far away.

### Q: Why does the RBF kernel make nearby poems have similar ratings?

**A**: The RBF kernel is:
$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

- When $x \approx x'$: $\|x - x'\| \approx 0$ → $k(x, x') \approx \sigma_f^2$ (high covariance)
- When $x$ far from $x'$: $\|x - x'\|$ large → $k(x, x') \approx 0$ (independent)

The length scale $\ell$ controls "how far is nearby":
- Small $\ell$: Only very close poems correlated (wiggly function)
- Large $\ell$: Distant poems correlated (smooth function)

This is the **inductive bias**: We assume preference varies smoothly in embedding space.

### Q: What if my preference doesn't vary smoothly?

**A**: Then the RBF kernel is wrong! Alternatives:

- **Matérn kernel**: Less smooth, allows sharper transitions
- **Periodic kernel**: For cyclic preferences
- **Neural kernel**: Learn the kernel from data with a neural network

But for poetry preferences based on semantic embeddings, RBF is reasonable: Similar poems (in embedding space) likely have similar ratings.

### Q: How do you choose hyperparameters ($\ell$, $\sigma_f$, $\sigma_n$)?

**A**: Two approaches:

1. **Fixed heuristics** (what we do in interactive mode):
   - $\ell = 1.0$ (default scale)
   - $\sigma_f^2 = 1.0$ (prior variance)
   - $\sigma_n^2 = 0.001$ (assume ratings are precise)

2. **Optimize via marginal likelihood** (what benchmarks do):
   - Maximize $p(y \mid \ell, \sigma_f, \sigma_n)$
   - Use gradient-based optimization (L-BFGS)
   - Cost: $O(m^3)$ per evaluation

Future work: Full Bayesian with MCMC (uncertainty over hyperparameters).

---

## Computational Complexity

### Q: Why is Cholesky factorization $O(m^3)$?

**A**: Cholesky factorization computes $K = LL^\top$ where $L$ is lower triangular.

Algorithm (simplified):
```
for i = 1 to m:
    for j = 1 to i:
        sum = K[i,j]
        for k = 1 to j-1:
            sum -= L[i,k] * L[j,k]  # O(i) operations
        L[i,j] = sum / L[j,j]
```

Total operations: $\sum_{i=1}^m \sum_{j=1}^i i = O(m^3)$

This is unavoidable for exact Cholesky on dense matrices.

### Q: Why not use iterative solvers to avoid $O(m^3)$?

**A**: Iterative solvers (e.g., conjugate gradient) need:

1. **Matrix-vector products**: $O(m^2)$ per iteration
2. **Convergence**: Typically $O(m)$ iterations for ill-conditioned matrices
3. **Total**: $O(m^3)$ anyway!

Plus: GPs need log-determinant for marginal likelihood → Cholesky is required.

Workaround: **Sparse GP approximations** (future work):
- Inducing points reduce $m \to m'$ where $m' \ll m$
- Complexity: $O(nm'^2 + m'^3)$
- Trade accuracy for speed

### Q: Why is variance computation $O(nm^2)$ instead of $O(nm)$ like mean?

**A**:

**Mean** (cheap):
$$\mu_q = K_{qr} \alpha$$

- $K_{qr}$: $n \times m$ matrix
- $\alpha$: $m \times 1$ vector
- Matmul: $O(nm)$

**Variance** (expensive):
$$\sigma^2_q = \text{diag}(K_{qq}) - \text{diag}(K_{qr} K^{-1} K_{rq})$$

Rewrite as $v = L^{-1} K_{rq}^\top$ (solve $Lv = K_{rq}^\top$):
- $K_{rq}^\top$: $m \times n$ matrix
- Triangular solve: $O(m^2 n)$ operations (must solve for all $n$ columns!)

Alternative (vectorized):
```python
v = scipy.linalg.solve_triangular(L, K_qr.T, lower=True)  # O(nm²)
var = k_qq - np.sum(v**2, axis=0)  # O(nm)
```

The triangular solve dominates: $O(nm^2)$.

---

## Benchmarking and Performance

### Q: Why is ScaLAPACK slower than Python for small $m$?

**A**: **Fixed overhead** dominates for small problems:

- Subprocess spawn: ~160ms
- File I/O (write/read matrices): ~50ms
- MPI initialization: ~100ms
- **Total overhead**: ~300ms minimum

For $m = 2{,}000$:
- Python compute: 180ms
- ScaLAPACK overhead + compute: 640ms
- **Result**: 3.5× slower!

Only when compute time exceeds overhead does ScaLAPACK win.

**Crossover point** (from benchmarks): $m \approx 5{,}000 - 7{,}000$

### Q: Why does 16 processes perform worse than 8 processes?

**A**: **Communication overhead** grows with process count:

At $m = 20{,}000$:
- 1 process: 84.79s (all overhead, no parallelism)
- 4 processes: 27.44s (compute dominates)
- 8 processes: 11.56s (optimal balance)
- 16 processes: 14.88s (communication overhead increases)

**Amdahl's Law**: Speedup limited by sequential fraction:
$$\text{Speedup} = \frac{1}{s + \frac{p}{P}}$$

where $s$ = sequential fraction, $p$ = parallel fraction, $P$ = processes.

As $P$ increases, communication ($s$) becomes more significant.

### Q: What's the best speedup you achieved?

**A**: From actual benchmarks:

**Best case** ($m = 20{,}000$, 8 processes):
- Python: 49.57s
- ScaLAPACK: 11.56s
- **Speedup: 4.3×**

**Why not 8×?**
- Communication overhead
- Load imbalance
- Memory bandwidth saturation
- Amdahl's law

This is typical for dense linear algebra on small clusters.

### Q: Did you test GPU scoring?

**A**: Not in recent benchmarks (no GPU scoring data in `large_scale_fit_20260406_233011.csv`).

**Previous observations** (from development):
- Cold-start overhead: ~150ms (transfer matrices to GPU)
- Speedup for variance: 3-4.6× for $m \geq 500$
- Crossover: When compute time > transfer overhead

For $m = 1{,}000$, $n = 85{,}000$:
- CPU (NumPy): ~3.5s
- GPU (CuPy): ~0.76s
- **Speedup: 4.6×**

GPU is especially valuable when scoring many candidates with uncertainty.

---

## Implementation Details

### Q: Why use PyBind11 instead of Cython or ctypes?

**A**: Comparison:

| Method | Overhead | Ergonomics | NumPy Integration |
|--------|----------|------------|-------------------|
| ctypes | Medium | Poor (manual wrapping) | Manual |
| Cython | Low | Medium (new language) | Good |
| **PyBind11** | **Lowest** | **Excellent** | **Seamless** |

PyBind11 advantages:
- Header-only, no build dependencies
- Automatic type conversion (NumPy ↔ C++)
- Exception handling
- Zero overhead for simple cases

Benchmark: PyBind11 overhead ~0.1ms (negligible).

### Q: Why block-cyclic distribution for ScaLAPACK?

**A**: Alternatives:

1. **Block row/column**: Simple, but load imbalance for Cholesky
2. **Cyclic**: Good load balance, but small messages (overhead)
3. **Block-cyclic**: Best of both worlds

Example ($4 \times 4$ matrix, $2 \times 2$ grid, block size 1):
```
Global matrix:       Process P00:    Process P01:
┌─────────────┐      ┌───┬───┐       ┌───┬───┐
│ 0 1 │ 0 1 │      │ 0 │ 0 │       │ 1 │ 1 │
│ 2 3 │ 2 3 │      │ 2 │ 2 │       │ 3 │ 3 │
├─────┼─────┤      └───┴───┘       └───┴───┘
│ 0 1 │ 0 1 │
│ 2 3 │ 2 3 │      Process P10:    Process P11:
└─────────────┘      ┌───┬───┐       ┌───┬───┐
                     │ 0 │ 0 │       │ 1 │ 1 │
                     │ 2 │ 2 │       │ 3 │ 3 │
                     └───┴───┘       └───┴───┘
```

Each process gets evenly distributed blocks → balanced workload.

### Q: What was Milestone 1B?

**A**: **Distributed kernel assembly optimization**.

**Before** (naïve):
1. Scatter full $K$ matrix to processes ($\sim$800MB communication)
2. Each process computes its tiles

**After** (optimized):
1. Broadcast feature vectors $X$ to all processes ($\sim$30MB)
2. Each process computes its tiles locally using BLAS DGEMM

**Result**: 20-40× assembly speedup (less communication + BLAS optimization).

Key insight: Broadcasting small data + local compute > scattering large result.

### Q: Why lazy variance evaluation?

**A**: Variance is $O(nm^2)$, mean is $O(nm)$.

**Acquisition function determines workload**:

| Acquisition | Needs Variance? | Complexity |
|-------------|-----------------|------------|
| max_mean | No | $O(nm)$ |
| max_variance | Yes | $O(nm^2)$ |
| UCB | Yes | $O(nm^2)$ |
| thompson | Yes | $O(nm^2)$ |

For max_mean: Skip variance → **85× speedup** (measured).

Design: `compute_variance=False` parameter in scoring backend.

---

## Active Learning

### Q: What's the difference between UCB and Thompson sampling?

**A**:

**UCB (Upper Confidence Bound)**:
$$x_{\text{next}} = \arg\max [\mu(x) + \beta \sigma(x)]$$

- Deterministic
- $\beta$ controls exploration (user-tunable)
- Intuition: Optimistic estimate (assume high values where uncertain)

**Thompson Sampling**:
$$f \sim \mathcal{N}(\mu, K_{qq} - K_{qr} K^{-1} K_{rq})$$
$$x_{\text{next}} = \arg\max f$$

- Stochastic (sample from posterior)
- No tuning parameter
- Intuition: Probability matching (explore in proportion to probability of optimality)

Both balance exploitation and exploration, but Thompson is Bayesian-optimal.

### Q: Why use max_variance for exploration?

**A**: **Information gain**.

Posterior entropy decreases when we observe at high-variance points:
$$H[f_*] = \frac{1}{2} \log(2\pi e \sigma^2(x_*))$$

Maximizing $\sigma^2(x_*)$ maximizes information gain → **most informative query**.

Alternative: Expected improvement (EI) balances information + value.

### Q: What's spatial_variance?

**A**: **Spatially diverse exploration**.

Problem with max_variance: All queries cluster in one unexplored region.

Solution:
1. Select $x_1 = \arg\max \sigma^2(x)$ (highest variance)
2. Select $x_2 = \arg\max [\sigma^2(x) - \alpha \cdot d(x, x_1)]$ (high variance, far from $x_1$)
3. Repeat for $k$ diverse queries

Cost: $O(n^2)$ (must compute pairwise distances) → Only for $n < 10{,}000$.

---

## War Stories (Expanded)

### Q: What was the ScaLAPACK performance mystery?

**A**: **Symptoms**:
- Pedagogical benchmarks: ScaLAPACK 21-146× slower than expected
- Large-scale benchmarks: ScaLAPACK working fine (4.3× speedup)
- Same code, same machine!

**Investigation**:
1. Checked process counts → Same
2. Checked block sizes → Same
3. Compared Slurm scripts line-by-line → **Found it!**

**Root causes**:
1. Missing `OMP_NUM_THREADS=1`:
   - BLAS was multi-threading within each MPI rank
   - Oversubscription: 8 MPI ranks × 8 BLAS threads = 64 threads on 16 cores
   - Thrashing!

2. Wrong MPI launcher:
   - Pedagogical: Using `srun` (Slurm's MPI launcher)
   - Large-scale: Using `mpirun` (explicit process binding)
   - Binding policies different!

**Fix**: Added to pedagogical scripts:
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot
--scalapack-launcher mpirun
```

**Lesson**: Environment configuration is **critical** in HPC. Always check threading and process binding!

### Q: Why did PyBind11 fail on compute nodes?

**A**: **Symptom**:
```
fatal error: Python.h: No such file or directory
```

**Cause**: Compute nodes don't have Python development headers (`python3-dev` package).

**Why**: Compute nodes are minimal (no compilers, dev tools) to reduce attack surface and dependencies.

**Solution**: Disable PyBind11 module:
```cmake
cmake -DPOETRY_ENABLE_PYBIND11=OFF
```

Only build ScaLAPACK executable (which doesn't need Python headers).

**Lesson**: What works on login node ≠ what works on compute node!

### Q: What was the concurrent build directory issue?

**A**: **Symptom**:
```
Error removing directory "native/build/_deps/pybind11-src"
```

**Cause**: Multiple Slurm jobs running simultaneously, all using `native/build/` → Race condition!

Job 1 writes → Job 2 deletes → Job 1 fails.

**Solution**: Job-specific build directories:
```bash
BUILD_DIR="native/build_job_${SLURM_JOB_ID}"
cmake -S native -B "${BUILD_DIR}"
```

Each job gets unique directory → No conflicts.

**Cleanup** at end of job:
```bash
rm -rf "${BUILD_DIR}"
```

**Lesson**: Shared filesystems + parallel jobs = need unique paths!

---

## Future Work (Details)

### Q: How would you do full Bayesian inference on hyperparameters?

**A**: Current: Point estimates via maximum likelihood.

**Bayesian approach**:
1. **Prior** over hyperparameters: $p(\theta)$ where $\theta = (\ell, \sigma_f, \sigma_n)$
2. **Likelihood**: $p(y \mid X, \theta)$ (GP marginal likelihood)
3. **Posterior**: $p(\theta \mid y, X) \propto p(y \mid X, \theta) p(\theta)$

**Inference via MCMC**:
- Sample $\theta^{(1)}, \ldots, \theta^{(T)} \sim p(\theta \mid y, X)$
- Predictions integrate over uncertainty:
$$p(f_* \mid y, X, x_*) = \int p(f_* \mid y, X, x_*, \theta) p(\theta \mid y, X) d\theta$$

**Challenge**: $O(m^3)$ per MCMC sample (must recompute Cholesky).

**Speedup**: Warm-start Cholesky from previous sample.

### Q: How would you learn features before GP?

**A**: Current: Use fixed embeddings from sentence transformer.

**Neural + GP approach**:
1. **Feature learner**: Neural network $\phi: \mathbb{R}^{384} \to \mathbb{R}^{d'}$
2. **GP on learned features**: $k(\phi(x), \phi(x'))$
3. **End-to-end training**: Backprop through GP likelihood

**Architecture**:
```
Embedding (384d) → NN (d'd) → GP kernel → Likelihood
                 ↑              ↑
              Train NN      Train hyperparams
```

**Benefit**: Learn task-specific features (not just generic semantic embeddings).

**Computational cost**: Must differentiate through GP ($O(m^3)$ per gradient step).

### Q: How would you do batch acquisition?

**A**: Current: Select one poem at a time.

**Batch acquisition**: Select $k$ poems simultaneously.

**Approaches**:

1. **Greedy**: Select top-$k$ by acquisition function (ignores correlations)
2. **Local penalization**: Penalize candidates near already-selected points
3. **Mutual information**: Maximize joint information gain

**Challenge**: Evaluating joint acquisition is $O(n^k)$ → Need approximations.

**Use case**: Parallel labeling (multiple users rating simultaneously).

---

## Practical Questions

### Q: How big can $m$ get before things break?

**A**: Limits by backend:

- **Python**: $m \approx 10{,}000$ (memory + time)
- **PyBind11**: $m \approx 10{,}000$ (same as Python)
- **ScaLAPACK**: $m \approx 50{,}000$ (distributed memory)

Beyond this: Need sparse GP approximations (inducing points).

From benchmarks: $m = 30{,}000$ takes ~3 minutes with 8 processes (marginal for interactive use).

### Q: How many poems can a user realistically rate?

**A**: From user studies (informal):

- Casual user: 10-50 poems
- Engaged user: 100-500 poems
- Power user: 1,000+ poems

At $m = 1{,}000$: ~1 second per recommendation (acceptable).

At $m = 5{,}000$: ~10 seconds per recommendation (marginal).

**Design decision**: Optimize for $m < 2{,}000$ (typical use case).

### Q: How accurate are the recommendations?

**A**: Depends on:

1. **Embedding quality**: How well do embeddings capture poetic similarity?
2. **User consistency**: Does user rate similar poems similarly?
3. **Number of ratings**: More data → better posterior

**Qualitative observations**:
- After 10 ratings: Recommendations vaguely reasonable
- After 50 ratings: Clear preference patterns emerge
- After 100 ratings: Recommendations feel personalized

**Quantitative evaluation** (future work):
- Hold-out test set
- Rank correlation with ground truth
- A/B testing vs baseline recommenders

### Q: Can I use this for non-poetry domains?

**A**: Yes! Requirements:

1. **Embeddings**: High-dimensional representation of items
2. **Ratings**: User preferences (can be binary, ordinal, continuous)
3. **Scale**: Moderate item count ($n < 100{,}000$)

**Example applications**:
- Music recommendation (audio embeddings)
- Paper recommendation (citation + abstract embeddings)
- Product recommendation (image + text embeddings)

**Adaptation**: Change acquisition functions, kernel choice, embedding model.

---

## Technical Deep Dives

### Q: How does the RBF kernel relate to infinite-dimensional feature spaces?

**A**: **Mercer's theorem**:

Any positive definite kernel $k(x, x')$ corresponds to an inner product in some feature space:
$$k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$$

For RBF kernel:
$$k(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$$

The feature map $\phi$ is **infinite-dimensional**:
$$\phi(x) = \left[\sqrt{c_k} \cos(\omega_k^\top x), \sqrt{c_k} \sin(\omega_k^\top x)\right]_{k=1}^\infty$$

where $\omega_k \sim \mathcal{N}(0, 1/\ell^2)$ (Bochner's theorem).

**Implication**: GPs with RBF kernels are equivalent to Bayesian linear regression with **infinitely many features**.

### Q: What's the connection to RKHS (Reproducing Kernel Hilbert Space)?

**A**: The posterior mean $\mu(x)$ lies in the RKHS induced by kernel $k$:
$$\mu(x) = \sum_{i=1}^m \alpha_i k(x, x_i)$$

**Representer theorem**: The optimal function is a linear combination of kernel functions centered at training points.

This is why GP predictions have the form:
$$\mu(x_*) = k_{*r}^\top \alpha$$

The weights $\alpha$ define the function in RKHS.

### Q: How does this relate to radial basis function (RBF) networks?

**A**: **RBF network** (neural network):
$$f(x) = \sum_{i=1}^m w_i \phi(\|x - c_i\|)$$

where $c_i$ are fixed centers, $w_i$ are learned weights.

**GP with RBF kernel**:
$$\mu(x) = \sum_{i=1}^m \alpha_i k(x, x_i)$$

where $x_i$ are training points (automatic "centers"), $\alpha_i$ are Bayesian weights.

**Differences**:
- GP: Bayesian (uncertainty), centers = training data
- RBF net: Frequentist (point estimate), centers chosen separately

GPs are "Bayesian RBF networks".

---

## Slide-Specific Clarifications

### Q: Slide 5 shows dual form - why is this valid?

**A**: **Woodbury matrix identity**:
$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$

Apply with $A = \lambda I_p$, $U = X^\top$, $C = I_m$, $V = X$:
$$(X^\top X + \lambda I_p)^{-1}X^\top = X^\top(XX^\top + \lambda I_m)^{-1}$$

This is a standard result in kernel methods.

### Q: Slide 9 shows covariance blocks - what are they?

**A**: Joint distribution:
$$\begin{bmatrix} y_r \\ f_* \end{bmatrix} \sim \mathcal{N}\left(
\begin{bmatrix} 0 \\ 0 \end{bmatrix},
\begin{bmatrix}
K_{rr} + \sigma_n^2 I & k_{r*} \\
k_{*r} & k_{**}
\end{bmatrix}
\right)$$

**Blocks**:
- $K_{rr}$: Covariance among rated points ($m \times m$)
- $k_{r*}$: Covariance between rated and new point ($m \times 1$)
- $k_{**}$: Variance of new point (scalar)

**Conditional Gaussian formula**: Given $y_r$, the posterior of $f_*$ is:
$$f_* \mid y_r \sim \mathcal{N}(\mu_*, \Sigma_*)$$

where:
$$\mu_* = k_{*r} K_{rr}^{-1} y_r$$
$$\Sigma_* = k_{**} - k_{*r} K_{rr}^{-1} k_{r*}$$

This is standard multivariate Gaussian conditioning.

### Q: Benchmark slides show actual data - where is this from?

**A**: `results/large_scale_fit_20260406_233011.csv`

Generated by: `sbatch scripts/large_scale_bench.slurm`

Swept over: $m \in \{2000, 5000, 7000, 10000, 15000, 20000, 25000, 30000\}$ with process counts $P \in \{1, 4, 8, 16\}$.

All numbers in slides are **actual measured times**, not theoretical estimates.

**Visualization**: The presentation includes publication-quality plots:
- `benchmark_fit_scaling.pdf` - Fit time vs $m$ for different backends
- `benchmark_speedup.pdf` - Speedup analysis showing crossover
- `benchmark_process_scaling.pdf` - Process count comparison at $m=20{,}000$
- `benchmark_loglog_scaling.pdf` - Log-log plot verifying O($m^3$) (not in slides, but available)

These were generated by `slides/create_benchmark_plots.py` from the CSV data.

---

## Questions for Further Exploration

### Q: Could you use this for multi-task learning (multiple users)?

**A**: Yes! **Multi-task GP**:

Model: $f_u(x)$ for each user $u$, with correlations between users.

Kernel:
$$k((x, u), (x', u')) = k_{\text{task}}(u, u') \cdot k_{\text{input}}(x, x')$$

Learn user correlations: Similar users have correlated preferences.

**Challenge**: Scales as $O((Um)^3)$ where $U$ = users → Need approximations.

### Q: How would you handle contextual information (time, mood, etc.)?

**A**: **Contextual GP**:

Input: $(x, c)$ where $x$ = poem, $c$ = context (time of day, mood, etc.)

Kernel:
$$k((x, c), (x', c')) = k_x(x, x') \cdot k_c(c, c')$$

Predictions condition on context: "Recommend poems for evening reading".

**Application**: Time-varying preferences, session-dependent recommendations.

---

This FAQ provides depth that can't fit in a 30-minute presentation. Use it to:
- Answer questions after the talk
- Provide to interested students
- Reference during Q&A

**Remember**: Slides are for presentation flow, FAQ is for deep understanding.
