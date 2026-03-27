# Running on a cluster

Do **not** run heavy embedding, benchmark, or MPI jobs on the login node.

This repository now includes small Slurm wrapper scripts to make that easier:

- `scripts/slurm_run.sh`: launch an interactive-style `srun` job for a Python script
- `scripts/slurm_submit.sh`: submit a non-MPI batch job with `sbatch`
- `scripts/slurm_submit_mpi.sh`: submit an MPI-style batch job and launch with `srun`

## Typical workflow

### 1. Prepare the dataset

```bash
python scripts/inspect_hf_poetry_dataset.py
python scripts/fetch_prepare_public_domain_poetry.py --output data/poems.parquet
```

If the dataset is large or the network/filesystem is slow, prefer running this through Slurm as well.

### 2. Compute embeddings and 2D projections

```bash
bash scripts/slurm_run.sh scripts/embed_poems.py --input data/poems.parquet --output data/embeddings.npy
bash scripts/slurm_run.sh scripts/project_poems_2d.py --input data/embeddings.npy --output data/proj2d.npy
bash scripts/slurm_run.sh scripts/build_poet_centroids.py --poems data/poems.parquet --embeddings data/embeddings.npy
bash scripts/slurm_run.sh scripts/project_poet_centroids_2d.py --input data/poet_centroids.npy --output data/poet_centroids_2d.npy
```

### 3. Run serial / single-node benchmarks

```bash
bash scripts/slurm_submit.sh scripts/bench_sweep.py --backends naive,blocked --n-poems 1000,2000,5000 --m-rated 5,10,20,40
```

### 4. Run MPI benchmarks

```bash
bash scripts/slurm_submit_mpi.sh 4 scripts/bench_mpi_sweep.py --n-poems 5000,10000,20000 --m-rated 5,10,20,40
```

## Overriding Slurm defaults

The wrappers honor these environment variables:

- `SLURM_PARTITION_OVERRIDE`
- `SLURM_TIME_OVERRIDE`
- `SLURM_CPUS_OVERRIDE`
- `SLURM_MEM_OVERRIDE`
- `SLURM_JOB_NAME_OVERRIDE`
- `SLURM_LOG_DIR_OVERRIDE` (batch wrappers)

Example:

```bash
export SLURM_PARTITION_OVERRIDE=compute
export SLURM_TIME_OVERRIDE=02:00:00
export SLURM_CPUS_OVERRIDE=16
export SLURM_MEM_OVERRIDE=32G
bash scripts/slurm_submit.sh scripts/bench_sweep.py --backends naive,blocked --n-poems 5000,10000 --m-rated 10,20,40
```

## Interactive development

For quick testing, request a node first, then run scripts inside that allocation.
For example, depending on the local cluster policy, something like:

```bash
salloc --partition=shared --time=00:30:00 --cpus-per-task=8 --mem=16G
python scripts/interactive_cli.py
```

## Notes

- The `naive` backend is intentionally inefficient and may become very slow even at moderate corpus sizes.
- The `blocked` backend is the main single-node optimized baseline.
- The `mpi` backend distributes candidate scoring across ranks and is the most natural next scaling step for this project.
- The Streamlit app is for exploration and demonstration; it should be launched from an interactive allocation, not the login node.
