# Running the UMAP projection on the cluster

Run the projection pipeline as a scheduled compute job, not from a constrained login shell.

Recommended entry point:

```bash
bash scripts/submit_projection_pipeline.sh
```

This submits a single-node CPU job and runs:

1. `python scripts/project_poems_2d.py`
2. `python scripts/build_poet_centroids.py`
3. `python scripts/project_poet_centroids_2d.py`

Useful overrides:

```bash
PARTITION=cpu \
CPUS_PER_TASK=16 \
MEMORY=192G \
TIME_LIMIT=04:00:00 \
PROJECT_ARGS="--n-jobs 16 --pre-reduce-dims 50" \
REPO_DIR=$PWD \
VENV_PATH=$PWD/.venv \
bash scripts/submit_projection_pipeline.sh
```

Notes:

- the current repo path is a single-node UMAP implementation
- `--n-jobs 1` and `--pre-reduce-dims 50` are conservative defaults
- GPU help requires a real GPU UMAP backend, which is not yet implemented
