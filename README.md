# poetry

A student-facing project on **poetry recommendation, exploration, and visualization** using

- semantic poem embeddings,
- an exact Gaussian-process preference model,
- and an HPC-oriented implementation ladder.

This repository is meant to do two things at once:

1. support a genuinely interesting poetry-exploration application, and
2. make the computational bottlenecks visible enough that profiling, vectorization, batching, and distributed-memory methods feel necessary rather than decorative.

The central idea is simple: treat poems as points in an embedding space, let a user rate poems, fit a Gaussian process over that space, and then use the posterior to choose the next poem either by

- **exploit**: show a poem the model currently expects the user to like, or
- **explore**: show a poem whose rating would be most informative because the model is uncertain there.

The same setup also supports:

- phrase-to-poem semantic search,
- poem-to-poem similarity lookup,
- poem-space and poet-space visualizations,
- and benchmark comparisons between naive, optimized, and distributed implementations.

## Why this is an HPC project

A useful first implementation is easy to write but expensive to run.

At each interaction step, the system needs to:

1. update an exact GP using the poems the user has rated so far,
2. score a large corpus of candidate poems under the posterior,
3. compute either posterior means, posterior variances, or both,
4. choose the next poem,
5. and optionally render heat maps over the 2D projection.

This makes the project a good vehicle for discussing:

- dense kernel computations,
- blocking and vectorization,
- exact GP linear algebra (kernel matrix, Cholesky factorization, triangular solves),
- profiling and performance breakdowns,
- distributed-memory scoring over a large candidate set,
- and, as a stretch, GPU acceleration.

The point is not to hide the fact that some versions are slow. The point is to use that slowness to motivate better implementations.

## Environment setup

On the cluster, do **not** modify the shared Spack course environment just to install project-specific Python packages.
Instead, activate the course environment first and then create a **project-local virtual environment** that reuses the shared site packages.

Typical workflow:

```bash
source ~/161588/spack/share/spack/setup-env.sh
spack env activate CS-2050
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
python scripts/check_env.py
```

What this does:

- reuses packages already available through the course Spack environment,
- installs missing Python dependencies into `.venv`,
- avoids rebuilding the course environment,
- and keeps the project self-contained.

Optional app dependencies such as Streamlit can be installed with:

```bash
INSTALL_APP_REQUIREMENTS=1 bash scripts/bootstrap_venv.sh
```

The dependency files are split as:

- `requirements-core.txt`: core runtime and benchmark dependencies
- `requirements-app.txt`: optional Streamlit UI dependency

## Core workflow

1. Build a canonical poetry dataset with metadata.
2. Embed each poem into a dense vector space.
3. Project poems to 2D for visualization.
4. Optionally build poet centroids for poet-level maps.
5. Rate poems interactively.
6. Fit an exact GP on the rated poems.
7. Score all candidate poems.
8. Recommend the next poem by exploit or explore.
9. Profile the computation and compare implementations.

## Implementation ladder

The repository is organized around progressively better implementations of the same basic recommendation step.

### 1. Naive serial exact GP

This version is intentionally straightforward.

Characteristics:
- exact GP fit on the rated poems,
- candidate poems scored one at a time,
- useful as a correctness baseline and a “natural first attempt.”

Pedagogical value:
- easy to understand,
- likely to become slow enough that students can see the need for improvement.

### 2. Blocked / vectorized exact GP

This is the main optimized single-node baseline.

Characteristics:
- candidates are scored in blocks,
- more dense linear algebra structure is exposed,
- Python overhead is reduced,
- BLAS-backed operations do more of the work.

Pedagogical value:
- demonstrates how changing the computational structure can dramatically improve performance without changing the mathematics.

### 3. MPI-distributed candidate scoring

This is the main distributed-memory step.

Characteristics:
- the rated-poem GP state is small and shared,
- the candidate corpus is distributed across ranks,
- each rank scores its local shard,
- results are reduced to choose the next poem.

Pedagogical value:
- connects naturally to prior MPI material,
- shows how a large corpus-level scoring problem can be parallelized cleanly.

### 4. GPU extension (future work)

Not yet a first-class backend.

Natural target:
- batched kernel and scoring computations on large blocks of candidate poems.

## Repository structure

```text
src/poetry_gp/
  kernel.py              # RBF kernel helpers and distance computations
  gp_exact.py            # exact GP fitting and posterior prediction
  heatmap.py             # 2D scalar-field smoothing for overlays
  profiling.py           # lightweight timing helpers
  data_utils.py          # dataset schema detection and canonicalization
  backends/
    naive.py             # serial baseline
    blocked.py           # vectorized single-node backend
    mpi.py               # distributed candidate scoring backend

scripts/
  bootstrap_venv.sh
  check_env.py
  inspect_hf_poetry_dataset.py
  fetch_public_domain_poetry.py
  fetch_prepare_public_domain_poetry.py
  prepare_poems_dataset.py
  embed_poems.py
  project_poems_2d.py
  build_poet_centroids.py
  project_poet_centroids_2d.py
  query_by_phrase.py
  query_by_poem.py
  interactive_cli.py
  bench_step.py
  bench_sweep.py
  bench_mpi_step.py
  bench_mpi_sweep.py
  plot_benchmarks.py
  plot_benchmarks_csv.py
  plot_all_benchmarks.py
  plot_heatmap_demo.py
  plot_poet_map.py
  slurm_run.sh
  slurm_submit.sh
  slurm_submit_mpi.sh

app/
  streamlit_app.py       # interactive demo UI
```

## Dataset and metadata

The code expects a canonical poem table with these columns:

- `poem_id`
- `title`
- `poet`
- `text`

The schema-detection utilities try to map common source fields such as

- `author`
- `poet_name`
- `content`
- `body`

into this standard form.

### Inspect the Hugging Face dataset first

```bash
python scripts/inspect_hf_poetry_dataset.py
```

### Fetch and canonicalize in one step

```bash
python scripts/fetch_prepare_public_domain_poetry.py --output data/poems.parquet
```

This should preserve author/poet metadata in the canonical `poet` column.

## Embeddings and projections

### Build poem embeddings

```bash
python scripts/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
```

### Build a 2D poem projection

```bash
python scripts/project_poems_2d.py --input data/embeddings.npy --output data/proj2d.npy
```

### Build poet centroids and their 2D projection

```bash
python scripts/build_poet_centroids.py --poems data/poems.parquet --embeddings data/embeddings.npy
python scripts/project_poet_centroids_2d.py --input data/poet_centroids.npy --output data/poet_centroids_2d.npy
```

These are useful for poet-level visualization and exploration.

## Ways to interact with the project

### Phrase-to-poem search

```bash
python scripts/query_by_phrase.py --text "melancholy bells over evening water" --topk 10
python scripts/query_by_phrase.py --text "winter grief and birds" --poet dickinson --topk 5
```

### Poem-to-poem similarity search

```bash
python scripts/query_by_poem.py --title "The Raven" --topk 10 --exclude-self
python scripts/query_by_poem.py --poem-id 12345 --topk 10 --exclude-self
```

### CLI exploration loop

```bash
python scripts/interactive_cli.py
```

This uses the blocked backend and lets you

- rate the current poem as like / neutral / dislike,
- ask for the next poem by exploit or explore,
- and inspect timing for each GP update step.

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app currently supports:

- **poem space** view,
- **poet space** view,
- preference vs uncertainty heatmap toggles,
- jump-to-title controls,
- rating buttons,
- exploit/explore buttons,
- a rated-poems table,
- and a last-step timing display.

It is still an early demo, not a polished production interface.

## Benchmarking and profiling

The repository includes scripts for generating performance evidence across implementations.

### One-off benchmark

```bash
python scripts/bench_step.py --backend naive --n-poems 5000 --m-rated 20
python scripts/bench_step.py --backend blocked --n-poems 5000 --m-rated 20 --block-size 2048
```

### Serial / single-node sweep

```bash
python scripts/bench_sweep.py --backends naive,blocked --n-poems 1000,2000,5000 --m-rated 5,10,20,40
python scripts/plot_benchmarks_csv.py --input results/bench_results.csv --output results/bench_results.png
```

### MPI benchmark and sweep

```bash
mpirun -n 4 python scripts/bench_mpi_step.py --n-poems 10000 --m-rated 20
mpirun -n 4 python scripts/bench_mpi_sweep.py --n-poems 5000,10000,20000 --m-rated 5,10,20,40
```

### Combined plotting for serial + MPI results

```bash
python scripts/plot_all_benchmarks.py \
  --serial-input results/bench_results.csv \
  --mpi-input results/mpi_bench_results.csv \
  --output results/all_benchmarks.png
```

The intended classroom use is to show where time goes, identify hotspots, and motivate each improvement with actual measurements.

## Running on a cluster

Heavy embedding, benchmark, and MPI jobs should **not** be run on the login node.

This repository includes Slurm wrappers:

- `scripts/slurm_run.sh`
- `scripts/slurm_submit.sh`
- `scripts/slurm_submit_mpi.sh`

See `RUNNING_ON_CLUSTER.md` for examples and usage details.

Typical examples:

```bash
bash scripts/slurm_run.sh scripts/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
bash scripts/slurm_submit.sh scripts/bench_sweep.py --backends naive,blocked --n-poems 1000,2000,5000 --m-rated 5,10,20,40
bash scripts/slurm_submit_mpi.sh 4 scripts/bench_mpi_sweep.py --n-poems 5000,10000,20000 --m-rated 5,10,20,40
```

## Suggested student workflows

### Fastest path to a working proof of concept

```bash
source ~/161588/spack/share/spack/setup-env.sh
spack env activate CS-2050
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
python scripts/check_env.py
python scripts/inspect_hf_poetry_dataset.py
python scripts/fetch_prepare_public_domain_poetry.py --output data/poems.parquet
python scripts/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
python scripts/project_poems_2d.py --input data/embeddings.npy --output data/proj2d.npy
python scripts/interactive_cli.py
```

### Fastest path to lecture-ready performance evidence

```bash
source ~/161588/spack/share/spack/setup-env.sh
spack env activate CS-2050
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
python scripts/bench_sweep.py --backends naive,blocked --n-poems 1000,2000,5000 --m-rated 5,10,20,40
python scripts/plot_benchmarks_csv.py --input results/bench_results.csv --output results/bench_results.png
mpirun -n 4 python scripts/bench_mpi_sweep.py --n-poems 5000,10000,20000 --m-rated 5,10,20,40
python scripts/plot_all_benchmarks.py --serial-input results/bench_results.csv --mpi-input results/mpi_bench_results.csv
```

## What is implemented vs still rough

### Implemented now

- exact GP fitting via Cholesky,
- naive serial backend,
- blocked vectorized backend,
- MPI candidate-scoring backend skeleton,
- canonical dataset preparation utilities,
- dependency bootstrap scripts for cluster use,
- embedding / projection scripts,
- phrase search,
- poem-to-poem search,
- poet centroid construction,
- benchmark scripts,
- combined benchmark plotting,
- basic heatmap rendering,
- early Streamlit UI,
- Slurm wrappers and cluster guidance.

### Still rough or incomplete

- no hardened GPU backend yet,
- MPI path still needs more real-cluster validation,
- benchmark plots are still deliberately simple,
- the Streamlit app could use richer search/filter controls,
- direct poem-neighbor panels inside the app would still be useful,
- and cluster-specific environment setup instructions may need adjustment for the local system.

## Pedagogical intent

This project is not trying to disguise the fact that some implementations are too slow or too memory-hungry. That is part of the point.

The repository is meant to support a useful and interesting poetry-exploration application while also making room for discussion of:

- naive serial baselines,
- blocked and vectorized improvements,
- distributed-memory scaling,
- profiling and performance analysis,
- and the role of dense linear algebra inside Bayesian search over poem space.
