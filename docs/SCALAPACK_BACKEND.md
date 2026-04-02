# ScaLAPACK backend scaffold

This document describes the first scaffold for a distributed exact-GP fit backend based on ScaLAPACK concepts.

## Scope of the scaffold

The current scaffold does **not** implement numerical ScaLAPACK calls yet. It establishes the file-level and process-launch interface for a future native backend.

The intended division of labor is:

- Python assembles the rated-set covariance matrix `K_rr` and the right-hand side `y`
- a native MPI executable reads those files
- the native executable will eventually distribute the matrix in block-cyclic format and run Cholesky + solve
- Python will read back `alpha`, the Cholesky factor, and timing metadata

## Files added

- `native/CMakeLists.txt`
- `native/scalapack_gp_fit.cpp`
- `src/poetry_gp/backends/scalapack_fit.py`
- `scripts/run_scalapack_fit_demo.py`

## Current native behavior

The native executable currently validates the file interface and writes placeholder output files plus metadata indicating that the backend is still a scaffold.

This allows us to stabilize:

- input/output file naming
- metadata schema
- launch commands (`srun` / `mpirun`)
- the Python wrapper contract

before implementing the actual distributed dense linear algebra.

## Input files

The Python wrapper writes:

- `input_meta.json`
- `K_rr.bin`
- `y.bin`

## Output files

The native scaffold writes:

- `output_meta.json`
- `alpha.bin`
- `chol_lower.bin`

## Next implementation steps

1. add BLACS/ScaLAPACK discovery and linkage in `native/CMakeLists.txt`
2. replace placeholder outputs in `native/scalapack_gp_fit.cpp` with:
   - distributed matrix setup
   - Cholesky factorization
   - triangular solve
   - gather back to rank 0
3. add a fit benchmark comparing SciPy and the ScaLAPACK backend
