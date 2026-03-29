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

Typical cluster-friendly workflow:

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

## Canonical corpus build path

The corpus foundation is now controlled by a manifest:

```text
configs/poetry_sources.json
```

To build the active corpus:

```bash
python scripts/build_poetry_corpus.py
```

Useful variants:

```bash
python scripts/build_poetry_corpus.py --sources public_domain_poetry
python scripts/build_poetry_corpus.py --per-source-limit 500
python scripts/build_poetry_corpus.py --min-chars 80
```

This writes:

- `data/poems.parquet`: deduped canonical corpus
- `data/source_audit.parquet`: per-source ingest/canonicalization counts
- `data/duplicate_poems.parquet`: dropped duplicate rows only
- `data/duplicate_groups.parquet`: kept and dropped rows grouped together for duplicate auditing

## Embeddings and projections

Build poem embeddings:

```bash
python scripts/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
```

Project poems to 2D:

```bash
python scripts/project_poems_2d.py --input data/embeddings.npy --output data/proj2d.npy
```

Build poet centroids and project them to 2D:

```bash
python scripts/build_poet_centroids.py --poems data/poems.parquet --embeddings data/embeddings.npy
python scripts/project_poet_centroids_2d.py --input data/poet_centroids.npy --output data/poet_centroids_2d.npy
```

## Interacting with the project

Phrase-to-poem search:

```bash
python scripts/query_by_phrase.py --text "melancholy bells over evening water" --topk 10
python scripts/query_by_phrase.py --text "winter grief and birds" --poet dickinson --topk 5
```

Poem-to-poem similarity search:

```bash
python scripts/query_by_poem.py --title "The Raven" --topk 10 --exclude-self
python scripts/query_by_poem.py --poem-id 12345 --topk 10 --exclude-self
```

Interactive CLI:

```bash
python scripts/interactive_cli.py
```

The CLI currently supports rating poems, exploit/explore recommendations, session persistence, search, and timing output for each GP step.

Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

## Benchmarking

One-off benchmark:

```bash
python scripts/bench_step.py --backend naive --n-poems 5000 --m-rated 20
python scripts/bench_step.py --backend blocked --n-poems 5000 --m-rated 20 --block-size 2048
```

Serial sweep:

```bash
python scripts/bench_sweep.py --backends naive,blocked --n-poems 1000,2000,5000 --m-rated 5,10,20,40
python scripts/plot_benchmarks_csv.py --input results/bench_results.csv --output results/bench_results.png
```

MPI sweep:

```bash
mpirun -n 4 python scripts/bench_mpi_step.py --n-poems 10000 --m-rated 20
mpirun -n 4 python scripts/bench_mpi_sweep.py --n-poems 5000,10000,20000 --m-rated 5,10,20,40
python scripts/plot_all_benchmarks.py \
  --serial-input results/bench_results.csv \
  --mpi-input results/mpi_bench_results.csv \
  --output results/all_benchmarks.png
```

## Key docs

- `docs/CORPUS_BUILDING.md`: source manifests, normalization, dedupe, and audit outputs
- `docs/METHOD_NARRATIVE.md`: mathematical motivation, modeling story, and HPC framing
- `configs/poetry_sources.template.json`: richer template with example source shapes

## Repository structure

```text
configs/
  poetry_sources.json
  poetry_sources.template.json

docs/
  CORPUS_BUILDING.md
  METHOD_NARRATIVE.md

src/poetry_gp/
  data_utils.py
  source_registry.py
  corpus_builder.py
  gp_exact.py
  kernel.py
  heatmap.py
  profiling.py
  backends/
    naive.py
    blocked.py
    mpi.py

scripts/
  build_poetry_corpus.py
  inspect_hf_poetry_dataset.py
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
```

## Current state

Implemented now:

- exact GP fitting via Cholesky,
- naive serial backend,
- blocked vectorized backend,
- MPI candidate-scoring backend skeleton,
- manifest-driven multi-source corpus building,
- stronger text normalization and duplicate auditing,
- embedding and projection scripts,
- phrase search and poem-to-poem search,
- poet centroid construction,
- benchmark scripts,
- basic heatmap rendering,
- early Streamlit UI,
- and Slurm wrappers for cluster use.

Still rough or incomplete:

- no hardened GPU backend yet,
- MPI path still needs more real-cluster validation,
- benchmark plots are still deliberately simple,
- the Streamlit app could use richer search/filter controls,
- direct poem-neighbor panels inside the app would still be useful,
- and the source manifest still needs more real corpus entries.
