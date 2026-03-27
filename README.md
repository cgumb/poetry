# poetry

Poetry-space exploration with semantic embeddings, exact Gaussian-process preference modeling, and an HPC-oriented implementation ladder.

This repository is being built for an HPC teaching module where the goal is not only to make the system useful, but to make the computational bottlenecks visible and measurable. The project treats poems as points in an embedding space, lets a user rate poems, and then uses a Gaussian process to drive both:

- **exploit**: recommend poems the model currently expects the user to like
- **explore**: recommend poems whose rating would be most informative because the model is uncertain there

The same setup also supports:

- phrase-to-poem semantic search
- poem and poet visualizations in 2D
- timing/profiling comparisons across naive, optimized, and distributed implementations

## Core idea

1. Build a corpus of poems with metadata such as title and poet.
2. Embed each poem into a dense vector space.
3. Project poems into 2D for visualization.
4. Fit an exact GP on the poems the user has rated so far.
5. Score the full corpus under the posterior.
6. Recommend the next poem by exploit or explore.
7. Visualize posterior preference or uncertainty as a heat map over the poem map.

## Implementation ladder

The repository is organized around a sequence of implementations that make good lecture material:

1. **Naive serial exact GP**
   - intentionally simple
   - scores one poem at a time
   - useful as the correctness baseline and the "bad first implementation"

2. **Blocked / vectorized exact GP**
   - batches candidate poems
   - uses denser linear algebra structure
   - the main optimized single-node baseline

3. **MPI-distributed candidate scoring**
   - distributes the candidate corpus across ranks
   - keeps the rated-poem GP state small and shared
   - the main distributed-memory scaling step

4. **Optional GPU extension**
   - not yet implemented as a first-class backend
   - natural future target for batched kernel and scoring computations

## Current repository layout

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
  inspect_hf_poetry_dataset.py
  fetch_public_domain_poetry.py
  fetch_prepare_public_domain_poetry.py
  prepare_poems_dataset.py
  embed_poems.py
  project_poems_2d.py
  build_poet_centroids.py
  project_poet_centroids_2d.py
  query_by_phrase.py
  interactive_cli.py
  bench_step.py
  bench_sweep.py
  bench_mpi_step.py
  bench_mpi_sweep.py
  plot_benchmarks.py
  plot_benchmarks_csv.py
  plot_heatmap_demo.py
  plot_poet_map.py
  slurm_run.sh
  slurm_submit.sh
  slurm_submit_mpi.sh

app/
  streamlit_app.py       # early interactive demo UI
```

## Data pipeline

The code expects a canonical poem table with the following columns:

- `poem_id`
- `title`
- `poet`
- `text`

The canonicalization utilities try to map common source-schema variants such as `author`, `poet_name`, `content`, or `body` into this standard form.

### Inspect the Hugging Face dataset first

```bash
python scripts/inspect_hf_poetry_dataset.py
```

### Fetch and canonicalize in one step

```bash
python scripts/fetch_prepare_public_domain_poetry.py --output data/poems.parquet
```

This writes a cleaned parquet with canonical metadata columns. The expectation is that the source dataset includes author/poet metadata; if it does, that metadata is preserved in the canonical `poet` column.

### Embeddings and 2D projection

```bash
python scripts/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
python scripts/project_poems_2d.py --input data/embeddings.npy --output data/proj2d.npy
```

### Poet centroids

```bash
python scripts/build_poet_centroids.py --poems data/poems.parquet --embeddings data/embeddings.npy
python scripts/project_poet_centroids_2d.py --input data/poet_centroids.npy --output data/poet_centroids_2d.npy
```

These are useful for poet-level visualization and exploration.

## Query and exploration tools

### Phrase-to-poem search

```bash
python scripts/query_by_phrase.py --text "melancholy bells over evening water" --topk 10
python scripts/query_by_phrase.py --text "winter grief and birds" --poet dickinson --topk 5
```

### CLI exploration loop

```bash
python scripts/interactive_cli.py
```

This uses the blocked backend and lets you:

- rate the current poem as like / neutral / dislike
- request the next poem via exploit or explore
- inspect timing for each GP update step

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app currently includes:

- 2D poem scatterplot
- preference/uncertainty heatmap toggle
- current poem display
- rating buttons
- exploit/explore buttons
- rated-poem table
- last-step timing display

It is still an early demo rather than a polished application.

## Benchmarking and profiling

The repository includes scripts for generating timing evidence across implementations.

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

### MPI benchmark

```bash
mpirun -n 4 python scripts/bench_mpi_step.py --n-poems 10000 --m-rated 20
mpirun -n 4 python scripts/bench_mpi_sweep.py --n-poems 5000,10000,20000 --m-rated 5,10,20,40
```

The intended lecture story is to show where time goes, identify the major hotspots, and then motivate each new improvement with actual measurements.

## Running on a cluster

Heavy jobs should **not** be run on the login node.

This repository includes Slurm wrappers:

- `scripts/slurm_run.sh`
- `scripts/slurm_submit.sh`
- `scripts/slurm_submit_mpi.sh`

See `RUNNING_ON_CLUSTER.md` for concrete examples and recommended usage.

Typical examples:

```bash
bash scripts/slurm_run.sh scripts/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
bash scripts/slurm_submit.sh scripts/bench_sweep.py --backends naive,blocked --n-poems 1000,2000,5000 --m-rated 5,10,20,40
bash scripts/slurm_submit_mpi.sh 4 scripts/bench_mpi_sweep.py --n-poems 5000,10000,20000 --m-rated 5,10,20,40
```

## What is implemented vs what is still rough

### Implemented now

- exact GP fitting via Cholesky
- naive serial backend
- blocked vectorized backend
- MPI candidate-scoring backend skeleton
- canonical dataset preparation utilities
- embedding / projection scripts
- phrase search
- poet centroid construction
- benchmark scripts
- basic heatmap rendering
- early Streamlit UI
- Slurm wrappers and cluster guidance

### Still rough or incomplete

- no hardened GPU backend yet
- MPI path needs more real-cluster validation
- benchmark plots are still minimal
- the Streamlit app needs cleanup and richer controls
- poem-to-poem search by title/id would still be useful
- full cluster-specific environment setup instructions may need tailoring to the actual system

## Suggested near-term workflow

If you want the fastest path to a working proof of concept:

```bash
python scripts/inspect_hf_poetry_dataset.py
python scripts/fetch_prepare_public_domain_poetry.py --output data/poems.parquet
python scripts/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
python scripts/project_poems_2d.py --input data/embeddings.npy --output data/proj2d.npy
python scripts/interactive_cli.py
```

If you want the fastest path to lecture-ready performance evidence:

```bash
python scripts/bench_sweep.py --backends naive,blocked --n-poems 1000,2000,5000 --m-rated 5,10,20,40
python scripts/plot_benchmarks_csv.py --input results/bench_results.csv --output results/bench_results.png
mpirun -n 4 python scripts/bench_mpi_sweep.py --n-poems 5000,10000,20000 --m-rated 5,10,20,40
```

## Pedagogical intent

This project is not trying to hide the fact that some approaches are too slow or too memory-hungry. That is part of the point. The system is meant to support a useful and interesting poetry-exploration application while also making room for:

- naive serial baselines that are easy to understand but too slow
- blocked and vectorized improvements
- distributed-memory scaling
- profiling and performance analysis
- discussion of where dense linear algebra enters recommendation and Bayesian search over poem space
