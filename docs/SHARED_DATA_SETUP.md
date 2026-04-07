# Shared Data Setup for Students

This repository includes preprocessed data files (poem embeddings, 2D projections, poet centroids) that would take ~2 hours to generate from scratch. To save time, students can symlink these files from a shared read-only location.

## Quick Start (Students)

After cloning the repository and bootstrapping the environment:

```bash
# 1. Bootstrap the Python environment
bash scripts/bootstrap_env.sh

# 2. Activate the environment
source scripts/activate_env.sh

# 3. Symlink preprocessed data from shared location
bash scripts/setup_shared_data.sh
```

This creates symlinks in `data/` pointing to:
- `poems.parquet` - 85k poems with imputed poet names
- `embeddings.npy` - 384-dimensional poem embeddings
- `proj2d_coords.npy` - 2D UMAP projection of all poems
- `proj2d_reducer.pkl` - UMAP reducer for projecting new points
- `poet_centroids.parquet` - Poet metadata (name, poem count)
- `poet_centroids.npy` - 384-dim poet centroid embeddings
- `poet_centroids_2d.npy` - 2D projection of poet centroids

## What This Enables

With the shared data setup, students can immediately:
- Run GP active learning sessions: `python scripts/app/run_gp_session.py`
- Generate visualizations: `python scripts/app/render_session_plots.py`
- Run HPC benchmarks: `sbatch scripts/pedagogical_benchmarks.slurm`
- Experiment with different backends and configurations

## Data Source

The shared data is located at:
```
/shared/courseSharedFolders/161588outer/161588/poetry_data/
```

This is a **read-only** location. All session files and benchmark results you generate will be written to your own `results/` directory.

## Instructor: Preparing Shared Data

To update the shared data location with a new imputed dataset:

```bash
# 1. Run LLM imputation to create poems_imputed.parquet
python scripts/app/apply_llm_imputation_results.py \
    --poems data/poems.parquet \
    --results data/llm_batch_results.jsonl \
    --output data/poems_imputed.parquet

# 2. Regenerate all derived data
python scripts/app/build_poet_centroids.py \
    --poems data/poems_imputed.parquet \
    --embeddings data/embeddings.npy \
    --output-parquet data/poet_centroids.parquet \
    --output-npy data/poet_centroids.npy \
    --min-poems 3

python scripts/app/project_poet_centroids_2d.py \
    --input data/poet_centroids.npy \
    --output data/poet_centroids_2d.npy \
    --reducer data/proj2d_reducer.pkl

# 3. Promote imputed dataset and copy to shared location
bash scripts/promote_imputed_dataset.sh
```

This will:
- Backup the original `poems.parquet` to `poems.parquet.original`
- Replace `poems.parquet` with the imputed version
- Copy all preprocessed files to the shared location
- Set permissions for student read access

## What Gets Symlinked vs Generated

**Symlinked (shared, read-only):**
- Core dataset and embeddings
- 2D projections and reducers
- Poet centroids

**Generated locally (student's own directory):**
- Session files (`.pkl` pickles with GP state)
- Benchmark results (CSV files, timing data)
- Plots and visualizations (PNG files)
- Slurm job outputs (`.out` and `.err` files)

This separation ensures students can experiment without affecting shared data or each other's work.

## Troubleshooting

**"Shared data directory not found"**
- Contact the instructor - the shared data may not be set up yet

**"Permission denied" when writing files**
- This is expected for symlinked files (they're read-only)
- Session files and results are written to `results/` which is in your local directory
- If you need to modify the dataset, copy it locally: `cp data/poems.parquet data/poems_local.parquet`

**Want to regenerate embeddings yourself?**
- You can! Just remove the symlinks and run the embedding pipeline:
```bash
rm data/*.npy data/*.pkl data/*.parquet  # Remove symlinks
python scripts/app/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
# ... (other preprocessing steps)
```

But this will take ~2 hours for 85k poems, so the shared data is recommended.
