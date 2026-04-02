# ScaLAPACK backend scaffold

This document describes the current scaffold for a distributed exact-GP fit backend based on ScaLAPACK concepts.

## Scope of the current implementation

The backend interface is now functional end-to-end, but the numerical work is still a **rank-0 reference path**, not a distributed ScaLAPACK implementation.

The current division of labor is:

- Python assembles the rated-set covariance matrix `K_rr` and the right-hand side `y`
- a native MPI executable reads those files
- rank 0 performs a serial Cholesky + solve reference computation
- the executable writes back `alpha`, the Cholesky factor, and timing metadata

This means the file-level and launch contract is now stable enough to benchmark and validate against SciPy while the actual distributed BLACS/ScaLAPACK path is implemented.

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
- computes a **serial numerical reference solve on rank 0**
- writes:
  - `alpha.bin`
  - `chol_lower.bin`
  - `output_meta.json`

The metadata records the backend name as `native_serial_reference` to make it clear that this is not yet the distributed path.

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

This allows the existing blocked scoring path to use the native reference fit outputs by converting them back into a `GPState` with the same predictive interface as the Python/SciPy path.

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

## Why keep this intermediate step?

Because it lets us verify:

- binary I/O conventions
- CLI arguments
- launcher behavior (`srun` / `mpirun`)
- output metadata schema
- agreement with SciPy on `alpha` and `logdet`
- agreement of the blocked scoring path under different fit implementations

before the implementation complexity of BLACS descriptors, block-cyclic redistribution, and ScaLAPACK calls is introduced.

## Next implementation steps

1. add BLACS/ScaLAPACK discovery and linkage in `native/CMakeLists.txt`
2. replace the rank-0 reference path in `native/scalapack_gp_fit.cpp` with:
   - process-grid setup
   - block-cyclic matrix distribution
   - ScaLAPACK Cholesky factorization
   - triangular solve
   - gather back to rank 0
3. keep `scripts/bench_scalapack_fit.py`, `scripts/bench_step.py`, and `scripts/bench_sweep.py` as continuity benchmarks while transitioning from the serial native path to the distributed path
