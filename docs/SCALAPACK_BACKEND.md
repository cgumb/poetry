# ScaLAPACK backend scaffold

This document describes the current scaffold for a distributed exact-GP fit backend based on ScaLAPACK concepts.

## Scope of the current implementation

The native backend now includes a **working multi-rank MPI factorization path**, but it is still **not yet a BLACS/ScaLAPACK implementation**.

The current division of labor is:

- Python assembles the rated-set covariance matrix `K_rr` and the right-hand side `y`
- a native MPI executable reads those files
- matrix rows are partitioned across MPI ranks
- ranks cooperate to compute a lower-triangular Cholesky factorization
- the factor is gathered back to rank 0
- rank 0 performs the final triangular solves and writes `alpha`, the Cholesky factor, and timing metadata

This means the repo now has a real multi-rank fit path while we continue toward the eventual BLACS/ScaLAPACK replacement.

## Files

- `native/CMakeLists.txt`
- `native/scalapack_gp_fit.cpp`
- `src/poetry_gp/backends/scalapack_fit.py`
- `scripts/run_scalapack_fit_demo.py`
- `scripts/bench_scalapack_fit.py`

## Current native behavior

The native executable now:

- validates the file interface
- reads a dense SPD matrix and RHS written by Python
- performs a **multi-rank MPI row-partitioned Cholesky factorization**
- gathers the resulting lower factor back to rank 0
- performs the final solve and log-determinant computation on rank 0
- writes:
  - `alpha.bin`
  - `chol_lower.bin`
  - `output_meta.json`

The metadata records the backend name as `mpi_row_partitioned_reference` to make it clear that this is still a reference distributed implementation rather than a true ScaLAPACK path.

## Input files

The Python wrapper writes:

- `input_meta.json`
- `K_rr.bin`
- `y.bin`

## Output files

The native executable writes:

- `output_meta.json`
- `alpha.bin`
- `chol_lower.bin`

## Reusing the native fit path inside the blocked GP pipeline

The blocked GP backend now supports:

- `fit_backend="python"` (default)
- `fit_backend="native_reference"`

This allows the existing blocked scoring path to use the native fit outputs by converting them back into a `GPState` with the same predictive interface as the Python/SciPy path.

This is useful because it lets us compare:

- same blocked scoring code
- same candidate selection logic
- different fit backends

before the true distributed ScaLAPACK numerical path is implemented.

The easiest entry points are:

```bash
python scripts/bench_step.py --backend blocked --fit-backend native_reference
```

and a broader sweep:

```bash
python scripts/bench_sweep.py \
  --backends blocked \
  --fit-backends python,native_reference
```

## Agreement / continuity check

Before replacing the current MPI row-partitioned reference path with a true distributed ScaLAPACK implementation, it is useful to verify that the blocked GP pipeline produces nearly identical outputs under:

- `fit_backend="python"`
- `fit_backend="native_reference"`

The repo includes a dedicated comparison script:

```bash
python scripts/compare_fit_backends.py \
  --n-poems 5000 \
  --m-rated 20
```

This reports differences in:

- posterior mean
- posterior variance
- GP weights (`alpha`)
- log marginal likelihood
- exploit / explore selections

That gives us a continuity benchmark to rerun later when the true ScaLAPACK numerical path replaces the current MPI reference path.

## Why keep this intermediate step?

Because it lets us verify:

- binary I/O conventions
- CLI arguments
- launcher behavior (`srun` / `mpirun`)
- output metadata schema
- agreement with SciPy on `alpha` and `logdet`
- agreement of the blocked scoring path under different fit implementations
- behavior of a real multi-rank factorization path before the library swap

before the implementation complexity of BLACS descriptors, block-cyclic redistribution, and ScaLAPACK calls is introduced.

## Next implementation steps

1. add BLACS/ScaLAPACK discovery and linkage in `native/CMakeLists.txt`
2. replace the MPI row-partitioned reference path in `native/scalapack_gp_fit.cpp` with:
   - process-grid setup
   - block-cyclic matrix distribution
   - ScaLAPACK Cholesky factorization
   - triangular solve
   - gather back to rank 0
3. keep `scripts/bench_scalapack_fit.py`, `scripts/bench_step.py`, `scripts/bench_sweep.py`, and `scripts/compare_fit_backends.py` as continuity benchmarks while transitioning from the MPI reference path to the true ScaLAPACK path
