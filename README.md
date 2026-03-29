# poetry

A student-facing project on **poetry recommendation, exploration, and visualization** using

- semantic poem embeddings,
- an exact Gaussian-process preference model,
- a manifest-driven multi-source corpus builder,
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
Instead, activate the course environment first and then create a *project-local virtual environment** that reuses the shared site packages.

Typical workflow:

 ```bash
source ~/161588/spack/share/spack/setup-env.sh
spack env activate CS-2050
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
python scripts/check_env.py
```

Optional app dependencies such as Streamlit can be installed with:

```bash
INSTALL_APP_REQUIREMENTS=1 bash scripts/bootstrap_venv.sh
```

## Core workflow

1. Build a canonical poetry corpus from one or more configured sources.
2. Embed each poem into a dense vector space.
3. Project poems to 2D for visualization.
4. Optionally build poet centroids for poet-level maps.
5. Rate poems interactively.
6. Fit an exact GP on the rated poems.
7. Score all candidate poems.
8 . Recommend the next poem by exploit or explore.
9. Profile the computation and compare implementations.

## Repository structure

```text
configs/
  poetry_sources.json            # active corpus foundation manifest
  poetry_sources.template.json   # richer template with example source shapes

src/poetry_gp/
  kernel.py              # RBF kernel helpers and distance computations
  gp_exact.py            # exact GP fitting and posterior prediction
  heatmap.py             # 2D scalar-field smoothing for overlays
  profiling.py           # lightweight timing helpers
  data_utils.py          # dataset schema detection and canonicalization
  source_registry.py     # source manifest loading + normalization + dedupe keys
  corpus_builder.py      # reusable manifest-driven corpus build helpers
  backends/
    naive.py             # serial baseline
    blocked.py           # vectorized single-node backend
    mpi.py              # distributed candidate scoring backend

scripts/
  bootstrap_venv.sh
  check_env.py
  inspect_hf_poetry_dataset.py
  build_poetry_corpus.py
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

MULTI_SOURCE_INGEST.md   # manifest-driven corpus build notes
DUPLICATE_AUDIT.md       # kept-vs-dropped duplicate audit notes
```

## Corpus building and metadata

The code expects a canonical poem table with at least these columns:

- `poem_id`
- `title`
- `poet`
- `text`

The canonical build path can also preserve provenance and dedupe metadata such as:

- `source_name`
- `source_kind`
- `source_location`
- `source_split`
- `license_family`
- `source_row_id`
- `text_hash`
- `text_hash_loose`
- `title_poet_text_key`

### Inspect a Hugging Face dataset quickly

```bash
python scripts/inspect_hf_poetry_dataset.py --dataset DanFosing/public-domain-poetry
```

### Build the active corpus foundation from the manifest

```bash
python scripts/build_poetry_corpus.py
```

Useful variants:

 ```bash
python scripts/build_poetry_corpus.py --sources public_domain_poetry
python scripts/build_poetry_corpus.py --per-source-limit 500
python scripts/build_poetry_corpus.py --min-chars 80
```

The active source list lives in :

```text
configs/poetry_sources.json
```

If you want to change the foundation, edit that list rather than modifying scripts.

### Duplicate audit outputs

The canonical build writes:

- `data/poems.parquet`: ideduped corpus
- `data/source_audit.parquet`: per-source ingest/canonicalization counts
- `data/duplicate_poems.parquet`: dropped duplicate rows only
- `data/duplicate_groups.parquet`: both kept and dropped rows in each duplicate group

See `MULTI_SOURCE_INGEST.md` and `DuPLICATE_AUDIT.md` for more detail.

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
- inspect timing for each GP update step,
- persist a rating session,
- and search for a poem by title / poet / text.

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Benchmarking and profiling

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
python scripts/plot_all_benchmarks.py \\
  --serial-input results/bench_results.csv \\
  --mpi-input results/mpi_bench_results.csv \\
  --output results/all_benchmarks.png
```

## Running on a cluster

Heavy embedding, benchmark, and MPI jobs should **not** be run on the login node.

This repository includes Slurm wrappers:

- `scripts/slurm_run.sh`
- `scripts/slurm_submit.sh`
- `scripts/slurm_submit_mpi.sh`

Typical examples:

 ```bash
bash scripts/slurm_run.sh scripts/build_poetry_corpus.py
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
python scripts/build_poetry_corpus.py
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
python scripts/check_env.py
python scripts/build_poetry_corpus.py
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
- manifest-driven multi-source corpus building,
- stronger text normalization and duplicate auditing,
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
- MPY path still needs more real-cluster validation,
- benchmark plots are still deliberately simple,
- the Streamlit app could use richer search/filter controls,
- direct poem-neighbor panels inside the app would still be useful,
- and the source manifest still needs more real corpus entries.

## Pedagogical intent

This project is not trying to disguise the fact that some implementations are too slow or too memory-hungry. That is part of the point.

The repository is meant to support a useful and interesting poetry-exploration application while also making room for discussion of:

- naive serial baselines,
- blocked and vectorized improvements,
- distributed-memory scaling,
- profiling and performance analysis,
- and the role of dense linear algebra inside Bayesian search over poem space.
