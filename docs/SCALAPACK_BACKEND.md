# ScaLAPACK backend scaffold

This document describes the current distributed exact-GP fit backends in the repo.

## Scope of the current implementation

The native executable now supports **backend selection** for the fit stage:

- `mpi_row_partitioned_reference`
- `scalapack`
- `auto`

`auto` chooses the ScaLAPACK path when the executable was built with ScaLAPACK support and otherwise falls back to the MPI reference backend.

The Python/native division of labor is still:

- Python assembles the rated-set covariance matrix `K_rr` and the right-hand side `y`
- a native MPI executable reads those files
- the native backend factorizes/solves the fit problem
- root reconstructs `alpha`, the Cholesky factor, and timing metadata
- Python reuses those outputs through the existing `GPState` interface

## Native backends

### 1. `mpi_row_partitioned_reference`

This is the existing correctness/reference distributed backend.

It:

- scatters matrix rows from root
- performs a row-partitioned MPI Cholesky factorization
- gathers the lower factor back to root
- performs the final triangular solve and log-determinant calculation on root

This backend is useful as a distributed reference implementation and continuity baseline.

### 2. `scalapack`

When the executable is built with ScaLAPACK support, the `scalapack` backend performs:

- BLACS process-grid setup
- block-cyclic local storage
- ScaLAPACK Cholesky factorization (`pdpotrf`)
- ScaLAPACK triangular solve (`pdpotrs`)
- root-side reconstruction of the dense factor and solution for compatibility with the existing Python scoring path

This is the first actual ScaLAPACK algorithmic path in the repo. It is still compatibility-oriented because it reconstructs dense outputs on root, but the expensive factorization/solve work is now delegated to ScaLAPACK.

## Build support

`native/CMakeLists.txt` now supports optional ScaLAPACK discovery.

At configure time:

- MPI is required
- BLAS/LAPACK are linked if found
- ScaLAPACK is enabled if discovered automatically or if `POETRY_SCALAPACK_LIBRARY` is provided explicitly

Example:

```bash
cmake -S native -B native/build
cmake --build native/build
```

Or with an explicit library path:

```bash
cmake -S native -B native/build -DPOETRY_SCALAPACK_LIBRARY=/path/to/libscalapack.so
cmake --build native/build
```

## Python interface

The Python wrapper supports explicit native backend selection through the `native_backend` argument and forwards it to the executable.

Useful values are:

- `auto`
- `mpi`
- `scalapack`

The comparison script exposes this as:

```bash
python scripts/compare_fit_backends.py \
  --scalapack-native-backend auto
```

## Cluster workflow

For interactive sanity checks on a compute node, using `mpirun` inside an allocation is usually easier than nested `srun`.

The repo now includes a reusable batch template:

- `scripts/compare_fit_batch.slurm`

Example:

```bash
sbatch scripts/compare_fit_batch.slurm
```

The batch template:

- activates the project virtual environment
- rebuilds the native executable
- disables inherited CPU binding variables
- runs one or more comparison cases with `compare_fit_backends.py`

The following environment variables can be overridden:

- `REPO_DIR`
- `N_POEMS`
- `M_RATED_LIST`
- `NPROCS_LIST`
- `NATIVE_BACKEND`
- `NATIVE_LAUNCHER`
- `NATIVE_EXECUTABLE`
- `BLOCK_SIZE`
- `SCALAPACK_BLOCK_SIZE`
- `DIM`
- `SEED`

## Agreement / continuity check

The repo includes a dedicated comparison script:

```bash
python scripts/compare_fit_backends.py \
  --n-poems 5000 \
  --m-rated 20 \
  --scalapack-native-backend auto \
  --verbose
```

This reports differences in:

- posterior mean
- posterior variance
- GP weights (`alpha`)
- log marginal likelihood
- exploit / explore selections

It remains the main continuity benchmark while switching between Python, MPI reference, and ScaLAPACK fit paths.

## Current limitations

- The ScaLAPACK path still reconstructs dense outputs on root for compatibility with the current Python scoring code.
- Input assembly still happens in Python rather than natively/distributed.
- The current implementation does not yet optimize redistribution or avoid root-side reconstruction.

## Next implementation steps

1. profile larger `m_rated` regimes where fit cost dominates launcher overhead
2. reduce root-side reconstruction costs if they become a bottleneck
3. consider distributed output handling for future native scoring paths
4. keep `scripts/bench_scalapack_fit.py`, `scripts/bench_step.py`, `scripts/bench_sweep.py`, and `scripts/compare_fit_backends.py` as continuity benchmarks while improving the native path
