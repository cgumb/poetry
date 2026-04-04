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

**Note:** Daemon scoring is **not recommended** for typical CLI usage. With small m (few rated poems) and large n_query (tens of thousands to score), Python scoring is ~10x faster due to daemon file I/O and MPI communication overhead. Daemon scoring is only beneficial for m > 5000+.

The CLI features:
- 🎨 **Rich terminal UI** with colors, panels, and tables
- 👥 **Multi-user support** - multiple users can maintain separate rating sessions
- 📊 **Exploit/explore recommendations** powered by Gaussian processes
- 🔍 **Search** by title, poet, or text content
- ⏱️ **Performance metrics** for each GP computation
- 💾 **Session persistence** - resume where you left off
- 🔧 **Hyperparameter optimization** via `--optimize-hyperparameters`

Rich is included in `requirements-app.txt` and will be installed with:
```bash
INSTALL_APP_REQUIREMENTS=1 bash scripts/bootstrap_venv.sh
```

Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Benchmarking

**Quick test** (m=100-2000, 1-4 processes):

```bash
sbatch scripts/quick_bench_test.slurm
python scripts/visualize_benchmarks.py results/quick_test_*.csv
```

**Large-scale test** (m=1000-10000, 1-16 processes):

```bash
sbatch scripts/large_scale_bench.slurm
python scripts/visualize_benchmarks.py results/large_scale_*.csv
```

**GPU scoring benchmark** (requires CUDA GPU and CuPy):

```bash
# On GPU node
sbatch scripts/gpu_scoring_bench.slurm

# Or run directly with Python
python scripts/bench_scoring.py \
  --m-rated 100 500 1000 2000 5000 10000 \
  --n-candidates 25000 \
  --cpu-threads 8 \
  --output-csv results/gpu_scoring.csv
```

**Custom benchmark:**

```bash
python scripts/bench_step.py --backend blocked --fit-backend native_reference \
  --n-poems 10000 --m-rated 2000 \
  --scalapack-launcher mpirun --scalapack-nprocs 8 --scalapack-block-size 64
```

**Python baseline:**

```bash
python scripts/bench_step.py --backend blocked --fit-backend python \
  --n-poems 10000 --m-rated 2000
```

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
  gp_exact.py                 # Exact GP implementation
  kernel.py                   # RBF kernel
  backends/
    blocked.py                # Vectorized blocked backend
    scalapack_fit.py          # ScaLAPACK native backend
    gpu_scoring.py            # GPU scoring with CuPy (optional)
    scoring.py                # Daemon scoring utilities

scripts/
  bench_step.py               # Core benchmark script
  bench_scoring.py            # GPU vs CPU scoring benchmark
  visualize_benchmarks.py     # Visualization tool
  quick_bench_test.slurm      # Quick benchmark (m=100-2000)
  large_scale_bench.slurm     # Large-scale benchmark (m=1000-10000)
  gpu_scoring_bench.slurm     # GPU scoring benchmark
  bootstrap_venv.sh           # Environment setup
  app/                        # Application scripts (corpus, queries, CLI)
  debug/                      # Debug and test scripts
  archive/                    # Old/superseded scripts
```

## Current state

**Implemented:**

- Exact GP fitting via Cholesky factorization
- Blocked vectorized Python backend
- **ScaLAPACK native backend with distributed linear algebra**
- **Milestone 1B: Distributed kernel assembly from features**
  - Broadcasts features (30MB) instead of matrix (800MB)
  - Parallel RBF kernel assembly across ranks
  - BLAS DGEMM optimization for 20-40× speedup
- **Milestone 1C: Persistent daemon + parallel scoring**
  - Persistent MPI daemon eliminates ~160ms subprocess overhead
  - Embarrassingly parallel scoring (beneficial for moderate n_query with large m)
  - **Note:** Daemon has high overhead (file I/O + MPI) for small m, typical CLI use
  - Graceful fallback to Python when MPI unavailable
  - **Analytic gradients** for hyperparameter optimization (2-5× faster)
- **GPU-accelerated scoring with CuPy (optional)**
  - CUDA-based posterior prediction for massive speedup with large m
  - RBF kernel, GEMV, and triangular solve on GPU
  - 10-100× faster than single-threaded CPU for m > 5000
  - Automatic CPU↔GPU data transfer and memory management
  - Graceful fallback when GPU/CuPy unavailable
  - Optimal for benchmarking with m > 5000, Python better for small m
- Manifest-driven multi-source corpus building
- Text normalization and duplicate auditing
- Embedding and 2D projection pipeline
- Phrase search and poem-to-poem search
- Interactive CLI with exploit/explore recommendations
- Comprehensive benchmarking and visualization tools
- Slurm scripts for HPC cluster use

**In progress / Next steps:**

- Large-scale benchmarks to find ScaLAPACK crossover point (m > 5000-10000)
- GPU backend for kernel assembly (fitting phase)
- Enhanced Streamlit UI with richer search controls
- Distributed scoring across large candidate sets for recommendation at scale
