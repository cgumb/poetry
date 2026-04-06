# Native HPC roadmap

This document lays out the next architecture steps for the HPC path while explicitly preserving the existing Python baseline for correctness checks.

## Current split

Today the repo has two partially separate concerns:

- a Python baseline path built around `GPState`, `fit_exact_gp()`, and `predict_block()`
- a native ScaLAPACK fit path that still returns compatibility artifacts back to Python

That compatibility requirement is currently responsible for a large amount of avoidable overhead.

## Design principles

1. Keep the Python baseline intact.
2. Treat the native path as a separate HPC backend rather than forcing it to mimic Python internals.
3. Move data movement costs out of the critical path wherever possible.
4. Keep continuity / agreement scripts so the two implementations can be compared on the same data.

## Milestone 1: native fit from features

### Goal

Remove the `K_rr.bin` boundary from the native fit path.

### Why

The current fit boundary still sends a dense `m x m` kernel matrix from Python to the native executable. For realistic `m`, that object is much larger than `x_rated` itself.

### Proposed interface

Python should pass:

- `x_rated`
- `y_rated`
- `length_scale`
- `variance`
- `noise`
- block/grid metadata

The native executable should then assemble the covariance matrix from features.

### Implementation stages

#### Stage 1A

Root-native kernel assembly:

- Python writes `x_rated` and `y_rated`
- native root reconstructs the dense kernel from features
- existing distributed ScaLAPACK fit path consumes that dense matrix

This removes the dense matrix file boundary without immediately changing the existing ScaLAPACK distribution path.

####Stage 1B ✅ **COMPLETED**

Distributed kernel assembly:

- each rank computes the block-cyclic kernel tiles it owns directly
- the dense kernel is never materialized on root

**Status:** Implemented and tested. See `docs/MILESTONE_1B_DESIGN.md` for design details.

**Key achievements:**
- Broadcast features (30MB) instead of scattering matrix (800MB)
- Parallel RBF kernel assembly across all ranks
- BLAS DGEMM optimization provides 20-40× speedup for assembly
- 8× reduction in overhead vs centralized scatter/gather (1.3s → 0.16s)
- Verified correct parallel execution on 2×2 process grid

**Performance notes:**
- ScaLAPACK still slower than Python for m < 5000 due to:
  - Subprocess overhead (~160ms per call)
  - Communication overhead in distributed Cholesky
- Expected crossover at m > 5000-10000 with 8-16+ processes
- Next milestone: persistent daemon to eliminate subprocess overhead

#### Stage 1C (Proposed)

Persistent daemon to eliminate subprocess overhead:

- Launch MPI processes once, keep them alive across multiple fit/score operations
- Python communicates via shared memory, pipes, or sockets
- Eliminates ~160ms subprocess launch overhead per operation
- Critical for interactive CLI where m < 5000 is common

**Design considerations:**
- Process lifetime management (when to shutdown?)
- Error recovery and crash handling
- State management across multiple operations
- Integration with existing Python API

**Expected benefit:**
- For m=2000 with 4 processes: 0.5s total → 0.35s (30% improvement)
- Makes ScaLAPACK competitive even for smaller problems
- Enables responsive interactive experience

## Milestone 2: native score backend

### Goal

Move scoring into MPI while keeping Python scoring as the reference baseline.

### Why

The query scoring path is embarrassingly parallel over query points. It is a natural target for MPI distribution even when fit remains centralized or compatibility-oriented.

### Proposed interface

A native scoring entry point should accept:

- `x_rated`
- fitted state (`alpha`, factor representation, hyperparameters)
- `x_query`
- distribution metadata

and return:

- posterior mean block(s)
- posterior variance block(s)
- exploit / explore argmax summaries

### Initial implementation choice

Replicate the fitted state across ranks and partition only query points.

This is not the final most memory-efficient design, but it is a simple and very natural first MPI scoring backend because the work is embarrassingly parallel over queries.

## Milestone 3: fit-only versus compatibility timing

### Goal

Separate:

- native kernel/factor/solve timing
- compatibility reconstruction timing
- score timing

### Why

At the moment, those costs can still blur together in end-to-end comparisons. This makes it difficult to tell whether a performance issue is coming from:

- the actual dense linear algebra
- wrapper/process overhead
- root-side compatibility work
- downstream scoring

## Recommended backend split

### Python baseline backend

This remains unchanged and continues to provide:

- correctness reference
- explainability / teaching baseline
- no dependency on MPI

### Native HPC backend

This should eventually support:

- native fit from features
- native distributed scoring
- optional root gather only when explicitly needed for visualization or debugging

## Comparison philosophy

The repo should keep comparison scripts that can evaluate:

- Python fit vs native fit
- Python score vs native score
- end-to-end Python vs end-to-end native

but the native path should no longer be forced to return every large internal object to Python just so the baseline code can run unchanged.

## Immediate next PR targets

1. native fit from features, eliminating `K_rr.bin`
2. native score backend with Python baseline preserved
3. benchmark scripts that isolate fit-only and score-only costs
