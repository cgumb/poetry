# poetry

Interactive poetry exploration with Gaussian-process preference modeling over poem embeddings.

## Planned implementation ladder

1. `naive` exact GP backend: intentionally simple serial baseline.
2. `blocked` exact GP backend: vectorized single-node implementation using dense linear algebra and batched kernel evaluation.
3. `mpi` backend: distributed-memory candidate scoring over poem shards.
4. Optional GPU backend.

## Initial repository structure

- `src/poetry_gp/`
  - `kernel.py`: RBF kernel utilities
  - `gp_exact.py`: exact GP fitting and posterior scoring
  - `backends/naive.py`: serial candidate-at-a-time baseline
- `scripts/`
  - placeholder directory for data prep, benchmarks, and app entry points

## Near-term goals

- add data ingestion and embedding pipeline for a poetry corpus
- add blocked/vectorized backend
- add profiling + benchmark harness
- add 2D heatmap visualization
- optionally add a Streamlit front end
