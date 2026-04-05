# poetry

A student-facing project on **poetry recommendation, exploration, and visualization** using

- semantic poem embeddings,
- an exact Gaussian-process preference model,
- a manifest-driven multi-source corpus builder,
- and an HPC-oriented implementation ladder.

This repository is meant to do two things at once:

1. support a genuinely interesting poetry-exploration application, and
2. make the computational bottlenecks visible enough that profiling, vectorization, batching, and distributed-memory methods feel necessary rather than decorative.

The core idea is simple: treat poems as points in an embedding space, let a user rate poems, fit a Gaussian process over that space, and then use the posterior to choose the next poem either by

- **exploit**: show a poem the model currently expects the user to like, or
- **explore**: show a poem whose rating would be most informative because the model is uncertain there.

## Why this is an HPC project

At each interaction step, the system may need to:

1. update an exact GP using the poems the user has rated so far,
2. score a large corpus of candidate poems under the posterior,
3. compute posterior means, posterior variances, or both,
4. choose the next poem,
5. and optionally render heat maps over a 2D projection.

This makes the project a good vehicle for discussing:

- dense kernel computations,
- blocking and vectorization,
- exact GP linear algebra such as Cholesky factorization and triangular solves,
- profiling and performance breakdowns,
- distributed-memory scoring over a large candidate set,
- and, as a stretch, GPU acceleration.

## Getting started

### CPU-only setup (default)

For typical HPC workloads with ScaLAPACK fitting:

```bash
bash scripts/bootstrap_env.sh
source scripts/activate_env.sh
python scripts/app/check_env.py
```

To include Streamlit app dependencies:

```bash
bash scripts/bootstrap_env.sh --app
```

### GPU-enabled setup (optional)

For GPU-accelerated scoring (10-100× faster for m > 5000):

```bash
# Must be run on a GPU node or with GPU access
srun -p gpu --gres=gpu:1 -n1 -t 30:00 --pty bash
cd ~/poetry
bash scripts/bootstrap_env.sh --gpu
source scripts/activate_env.sh --gpu
python -c "import cupy; print(f'CuPy: {cupy.__version__}')"
```

This creates a mamba-based conda environment with CuPy pre-compiled for GPU nodes, avoiding CPU architecture mismatch issues between general and GPU nodes.

**Note:** GPU setup uses CS-2050-mamba spack environment + conda, while CPU setup uses CS-2050 + venv. Both approaches work seamlessly across node types.

### Manual activation

Once bootstrapped, activate with:

```bash
# CPU environment
source scripts/activate_env.sh

# GPU environment (if created)
source scripts/activate_env.sh --gpu
```

The activation script auto-detects which environment was created and activates the appropriate one.

### Optional LLM dependencies

For metadata imputation (works in either environment):

```bash
source scripts/activate_env.sh
pip install -r requirements-llm.txt
```

## Canonical corpus build path

The corpus foundation is now controlled by a manifest:

```text
configs/poetry_sources.json
```

To build the active corpus:

```bash
python scripts/app/build_poetry_corpus.py
```

Useful variants:

```bash
python scripts/app/build_poetry_corpus.py --sources public_domain_poetry
python scripts/app/build_poetry_corpus.py --per-source-limit 500
python scripts/app/build_poetry_corpus.py --min-chars 80
```

This writes:

- `data/poems.parquet`: deduped canonical corpus
- `data/source_audit.parquet`: per-source ingest/canonicalization counts
- `data/duplicate_poems.parquet`: dropped duplicate rows only
- `data/duplicate_groups.parquet`: kept and dropped rows grouped together for duplicate auditing

## Embeddings and projections

Build poem embeddings:

```bash
python scripts/app/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
```

Project poems to 2D with UMAP and save the fitted reducer:

```bash
python scripts/app/project_poems_2d.py \
  --input data/embeddings.npy \
  --output data/proj2d.npy \
  --reducer-output data/proj2d_reducer.pkl
```

The projection path now uses:

- `float32`
- a PCA-style pre-reduction step before UMAP (default: 50 dimensions)
- conservative CPU parallelism by default (`--n-jobs 1`)
- optional non-deterministic execution for speed

If you want a faster-but-riskier run on a roomier machine, increase jobs manually:

```bash
python scripts/app/project_poems_2d.py --n-jobs 2
```

If you want more reproducible output instead of speed:

```bash
python scripts/app/project_poems_2d.py --deterministic --seed 0
```

You can also tune the pre-reduction size explicitly:

```bash
python scripts/app/project_poems_2d.py --pre-reduce-dims 50
```

Build poet centroids in embedding space and then project them with the **same** reducer:

```bash
python scripts/app/build_poet_centroids.py --poems data/poems.parquet --embeddings data/embeddings.npy
python scripts/app/project_poet_centroids_2d.py \
  --input data/poet_centroids.npy \
  --output data/poet_centroids_2d.npy \
  --reducer data/proj2d_reducer.pkl
```

This shared-reducer path matters: poem points and poet centroids should live in the same 2D coordinate system if they are going to be overlaid on the same visualization.

## Interacting with the project

Phrase-to-poem search:

```bash
python scripts/app/query_by_phrase.py --text "melancholy bells over evening water" --topk 10
python scripts/app/query_by_phrase.py --text "winter grief and birds" --poet dickinson --topk 5
```

Poem-to-poem similarity search:

```bash
python scripts/app/query_by_poem.py --title "The Raven" --topk 10 --exclude-self
python scripts/app/query_by_poem.py --poem-id 12345 --topk 10 --exclude-self
```

Interactive CLI:

```bash
python scripts/app/interactive_cli.py
```

**Scoring backend recommendation:**
- **Python (default)**: Best for typical CLI usage (m < 1000)
- **GPU**: 2-10× speedup for m > 5000, requires GPU node + CuPy
- **Daemon**: Not recommended - high overhead for typical CLI use

The CLI features:
- 🎨 **Rich terminal UI** with colors, panels, and tables
- 👥 **Multi-user support** - multiple users can maintain separate rating sessions
- ⚙️ **Interactive config menu** (`c` command):
  - **Exploitation strategies**: Max Mean, UCB (recommended), LCB, Thompson Sampling
  - **Exploration strategies**: Max Variance (entropy), Spatial Diverse, Expected Improvement
  - **Score backend**: Python (CPU) or GPU (CUDA)
  - **Hyperparameter optimization**: Toggle on/off with iteration control
  - Settings persist per user in session files
- 📊 **Advanced acquisition functions**:
  - **Exploit (e)**: Choose from Max Mean, UCB (recommended for balancing mean + uncertainty), LCB (conservative), Thompson (diverse)
  - **Explore (x)**: Choose from Max Variance (info-optimal), Spatial Diverse (O(n²), spatially aware), Expected Improvement (balanced)
- 🔍 **Search** by title, poet, or text content
- ⏱️ **Performance metrics** for each GP computation
- 💾 **Session persistence** - resume where you left off with saved config
- 🔧 **Hyperparameter optimization** with analytic gradients (2-5× faster)

**Acquisition function guide:**

| Use Case | Recommendation |
|----------|---------------|
| **Safe recommendations** | UCB (β=2.0) - balances predicted quality + confidence |
| **Diverse recommendations** | Thompson Sampling - samples from posterior |
| **Fast exploration** | Max Variance - information-theoretically optimal |
| **Spatially diverse exploration** | Spatial Diverse - considers correlation (slow O(n²)) |
| **Balanced explore/exploit** | Expected Improvement - classic Bayesian optimization |

Rich is included in `requirements-app.txt` and will be installed with:
```bash
bash scripts/bootstrap_env.sh --app
```

Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Benchmarking

### Fitting Benchmarks (GP training)

**Quick test** (m=100-2000, 1-4 processes):

```bash
sbatch scripts/quick_bench_test.slurm
python scripts/visualize_benchmarks.py results/quick_test_*.csv
```

**Large-scale overnight run** (m=2k-30k, 1-16 processes, ~12 hours):

```bash
sbatch scripts/large_scale_bench.slurm
python scripts/visualize_benchmarks.py results/large_scale_fit_*.csv
```

Tests comprehensive parameter sweep:
- Problem sizes: 2k, 5k, 7k, 10k, 15k, 20k, 25k, 30k
- Block sizes: 64, 128, 256
- Process counts: 1, 4, 8, 16
- **FIT-ONLY**: Scoring skipped (`--score-backend none`) for focused performance analysis

**Block size comparison** (m=7k-20k, optimal block size):

```bash
sbatch scripts/blocksize_sweep.slurm
python scripts/visualize_benchmarks.py results/blocksize_sweep_*.csv
```

### Scoring Benchmarks (Posterior prediction)

**Comprehensive scoring comparison** (requires GPU node):

```bash
sbatch scripts/gpu_scoring_bench.slurm
```

Tests **all three backends** for each m value:
- CPU (1 thread) - baseline
- CPU (8 threads) - multi-threaded BLAS
- GPU (CuPy/CUDA) - CUDA acceleration

Problem sizes: m ∈ {100, 500, 1k, 2k, 5k, 10k, 15k, 20k} with n=25k candidates

This single script provides complete CPU vs GPU comparison including threading effects.

**Custom scoring benchmark:**

```bash
python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 5000 10000 \
  --n-candidates 25000 \
  --cpu-threads 8 \
  --output-csv results/scoring_custom.csv
```

### Custom Fitting Benchmark

**ScaLAPACK:**

```bash
python scripts/bench_step.py --backend blocked --fit-backend native_reference \
  --n-poems 10000 --m-rated 2000 \
  --scalapack-launcher mpirun --scalapack-nprocs 8 --scalapack-block-size 64 \
  --score-backend none  # Skip scoring for focused fit benchmarking
```

**Python baseline:**

```bash
python scripts/bench_step.py --backend blocked --fit-backend python \
  --n-poems 10000 --m-rated 2000 \
  --score-backend none
```

**Performance expectations:**
- **Fitting**: O(m³) - ScaLAPACK wins for m > 7k-10k
- **Scoring (mean only)**: O(n × m × d) + O(n × m) - Fast, GPU helps for large m
- **Scoring (with variance)**: O(n × m²) - Expensive! GPU 5-10× faster for m > 5k

See `scripts/README.md` for more options and `docs/BENCHMARKING_GUIDE.md` for details.

## Data quality improvements

**Canonical poet prioritization** in visualizations:
- Curated list of ~60 major poets (Dickinson, Yeats, Auden, Larkin, Heaney, etc.)
- Hybrid selection: canonical poets shown with priority, then high-count poets
- Visual distinction: canonical poets have darker color, larger markers, priority labeling

**Missing metadata imputation** using multiple strategies:

```bash
# Step 1: First-line matching (recommended, always safe)
python scripts/app/impute_missing_metadata.py \
  --poems data/poems.parquet \
  --output data/poems_imputed.parquet

# Step 2: Generate LLM batch requests for remaining unknowns
python scripts/app/impute_missing_metadata.py \
  --poems data/poems_imputed.parquet \
  --generate-llm-batch data/llm_batch_requests.jsonl

# Step 3: Submit to Claude Batch API (requires anthropic SDK and API key)
pip install -r requirements-llm.txt
export ANTHROPIC_API_KEY='your-key'
python scripts/app/submit_llm_batch.py --input data/llm_batch_requests.jsonl

# Step 4: Check status (optional)
python scripts/app/check_llm_batch.py

# Step 5: Download results when complete
python scripts/app/download_llm_batch.py --output data/llm_batch_results.jsonl

# Step 6: Apply results back to dataframe
python scripts/app/apply_llm_imputation_results.py \
  --poems data/poems_imputed.parquet \
  --results data/llm_batch_results.jsonl \
  --output data/poems_imputed.parquet
```

Imputation strategies:
1. **First-line matching**: Propagate known poets to duplicate poems (free, robust)
2. **LLM batch API**: Claude identifies well-known poems (~$0.003-0.005 per 1000 poems)

**Key safeguards**:
- ✅ Never re-imputes already-imputed rows
- ✅ Only generates LLM requests for rows still needing imputation
- ✅ Boolean flags (`poet_imputed`, `title_imputed`) track imputation status
- ✅ Confidence filtering to avoid low-quality imputations

See `docs/POET_SELECTION_AND_IMPUTATION.md` for complete workflow and details.

## Key docs

- `docs/METHOD_NARRATIVE.md`: mathematical motivation, modeling story, and HPC framing
- `docs/NATIVE_HPC_ROADMAP.md`: HPC optimization roadmap and milestones
- `docs/MILESTONE_1B_DESIGN.md`: Distributed kernel assembly design (Milestone 1B)
- `docs/SCALAPACK_BACKEND.md`: ScaLAPACK implementation details
- `docs/BENCHMARKING_GUIDE.md`: Benchmarking workflow and scripts
- `docs/CORPUS_BUILDING.md`: source manifests, normalization, dedupe, and audit outputs
- `docs/POET_SELECTION_AND_IMPUTATION.md`: Canonical poet prioritization and metadata imputation
- `scripts/README.md`: Script organization and usage guide

## Repository structure

```text
configs/
  poetry_sources.json          # Corpus source manifest

docs/
  NATIVE_HPC_ROADMAP.md       # HPC optimization roadmap
  MILESTONE_1B_DESIGN.md      # Distributed assembly design
  SCALAPACK_BACKEND.md        # ScaLAPACK implementation
  BENCHMARKING_GUIDE.md       # Benchmarking workflow
  METHOD_NARRATIVE.md         # Mathematical motivation
  CORPUS_BUILDING.md          # Corpus building guide

native/
  scalapack_gp_fit.cpp        # C++ ScaLAPACK GP solver
  scalapack_gp_fit_entry.cpp  # Entry point with routing
  CMakeLists.txt              # Build configuration

src/poetry_gp/
  gp_exact.py                 # Exact GP with optional variance computation
  kernel.py                   # RBF kernel
  backends/
    blocked.py                # Vectorized blocked backend with acquisition functions
    scalapack_fit.py          # ScaLAPACK native backend
    gpu_scoring.py            # GPU scoring with CuPy (optional variance)
    scoring.py                # Daemon scoring utilities

scripts/
  bench_step.py               # Core fit benchmark script
  bench_scoring.py            # GPU vs CPU scoring benchmark (fixed threading)
  visualize_benchmarks.py     # Visualization tool with side-by-side plots
  quick_bench_test.slurm      # Quick benchmark (m=100-2000)
  large_scale_bench.slurm     # Overnight fit benchmark (m=2k-30k)
  blocksize_sweep.slurm       # Block size optimization sweep
  gpu_scoring_bench.slurm     # GPU vs CPU scoring comparison (all backends)
  bootstrap_env.sh            # Unified environment setup (CPU or GPU)
  activate_env.sh             # Auto-detecting environment activation
  app/
    interactive_cli.py        # Interactive CLI with config menu
    build_poetry_corpus.py    # Manifest-driven corpus builder
    query_by_phrase.py        # Phrase-to-poem search
    query_by_poem.py          # Poem-to-poem similarity
    check_env.py              # Environment verification
    # ... (other app scripts)
  debug/                      # Debug and test scripts
  archive/                    # Old/superseded scripts
```

## Current state

**Implemented:**

**Core GP System:**
- Exact GP fitting via Cholesky factorization with **analytic gradients** (2-5× faster optimization)
- Blocked vectorized Python backend
- **Optional variance computation** - skip expensive O(n × m²) variance calc for exploit-only workflows

**Distributed Computing:**
- **ScaLAPACK native backend with distributed linear algebra**
- **Milestone 1B: Distributed kernel assembly from features**
  - Broadcasts features (30MB) instead of matrix (800MB)
  - Parallel RBF kernel assembly across ranks
  - BLAS DGEMM optimization for 20-40× speedup
  - Crossover point: m ≈ 7k-10k (ScaLAPACK beats Python)

**GPU Acceleration:**
- **GPU-accelerated scoring with CuPy (optional)**
  - CUDA-based posterior prediction for massive speedup with large m
  - RBF kernel, GEMV, and triangular solve on GPU
  - 5-10× faster than multi-threaded CPU for m > 5000
  - **Optional variance** - can skip O(n × m²) variance for exploit-only
  - Automatic CPU↔GPU data transfer and memory management
  - Graceful fallback when GPU/CuPy unavailable

**Advanced Acquisition Functions:**
- **Exploitation strategies** (for 'exploit' command):
  - **Max Mean**: Simple argmax μ(x) - fast but risky
  - **UCB (Upper Confidence Bound)**: argmax μ(x) + β·σ(x) - **RECOMMENDED** industry standard
  - **LCB (Lower Confidence Bound)**: argmax μ(x) - β·σ(x) - conservative
  - **Thompson Sampling**: Sample from posterior - Bayesian optimal, diverse
- **Exploration strategies** (for 'explore' command):
  - **Max Variance**: argmax σ²(x) - information-theoretically optimal (minimize entropy)
  - **Spatial Diverse**: Minimize mean variance - spatially aware (expensive O(n²))
  - **Expected Improvement**: Classic Bayesian optimization - balanced

**User Experience:**
- **Interactive config menu** in CLI (`c` command)
  - Select exploitation/exploration strategies
  - Tune UCB β parameter (1.0-3.0)
  - Choose score backend (Python/GPU)
  - Toggle hyperparameter optimization
  - Settings persist in session files
- 🎨 Rich terminal UI with colors, panels, and tables
- 👥 Multi-user support with separate rating sessions
- 🔍 Search by title, poet, or text content
- ⏱️ Performance metrics for each GP computation
- 💾 Session persistence with configuration

**Data Pipeline:**
- Manifest-driven multi-source corpus building
- Text normalization and duplicate auditing
- Embedding and 2D projection pipeline
- Phrase search and poem-to-poem search
- Canonical poet prioritization and metadata imputation

**Benchmarking & HPC:**
- Comprehensive Slurm scripts for cluster use:
  - Large-scale fit benchmarks (m up to 30k)
  - GPU vs CPU scoring comparisons
  - Block size and process count sweeps
- Visualization tools with side-by-side comparisons
- Detailed performance profiling and timing breakdowns

**In progress / Next steps:**

- Analyze overnight large-scale benchmarks (m=2k-30k)
- GPU backend for kernel assembly (fitting phase) - currently CPU-only
- Enhanced Streamlit UI with acquisition function selection
- Multi-GPU support for extremely large-scale problems
