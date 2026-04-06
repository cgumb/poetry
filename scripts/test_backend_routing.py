#!/usr/bin/env python3
"""
Test which backend path is actually being used when requesting native_reference.

This bypasses Python wrapper and directly tests the native executable to see:
1. Which backend resolved_backend becomes (scalapack vs mpi_row_partitioned_reference)
2. Whether distributed assembly (Milestone 1B) is being used
3. Why native_reference might be slow
"""

import subprocess
import tempfile
import json
from pathlib import Path
import numpy as np
import shutil

def test_backend_routing():
    tmpdir = Path(tempfile.mkdtemp(prefix="backend_routing_test_"))

    try:
        # Create test problem
        n, d = 100, 10
        meta = {
            "input_kind": "features",
            "n": n,
            "d": d,
            "length_scale": 1.0,
            "variance": 1.0,
            "noise": 0.001,
            "rhs_cols": 1,
            "dtype": "float64",
            "matrix_layout": "row_major",
        }

        x = np.random.randn(n, d)
        y = np.random.randn(n)

        # Write inputs
        input_meta = tmpdir / "input.json"
        matrix_bin = tmpdir / "x.bin"
        rhs_bin = tmpdir / "y.bin"
        output_meta = tmpdir / "output.json"
        alpha_bin = tmpdir / "alpha.bin"
        chol_bin = tmpdir / "chol.bin"

        with open(input_meta, 'w') as f:
            json.dump(meta, f)
        x.tofile(matrix_bin)
        y.tofile(rhs_bin)

        # Run with native_reference backend
        cmd = [
            "mpirun", "-n", "4",
            "native/build/scalapack_gp_fit",
            "--input-meta", str(input_meta),
            "--matrix-bin", str(matrix_bin),
            "--rhs-bin", str(rhs_bin),
            "--output-meta", str(output_meta),
            "--alpha-bin", str(alpha_bin),
            "--chol-bin", str(chol_bin),
            "--backend", "native_reference",
            "--block-size", "128",
            "--return-alpha", "1",
            "--return-chol", "0",
        ]

        print("=" * 80)
        print("BACKEND ROUTING TEST")
        print("=" * 80)
        print(f"Test problem: n={n}, d={d}")
        print(f"Input kind: features (should enable distributed assembly)")
        print(f"Requested backend: native_reference")
        print(f"Workdir: {tmpdir}")
        print()
        print("Running MPI executable...")
        print()

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        print("=" * 80)
        print("STDERR OUTPUT (look for [Milestone 1B] or [Legacy])")
        print("=" * 80)
        print(result.stderr)
        print()

        # Parse output
        if output_meta.exists():
            with open(output_meta) as f:
                output = json.load(f)

            print("=" * 80)
            print("RESULTS")
            print("=" * 80)
            print(f"Requested backend:         {output.get('requested_backend', 'MISSING')}")
            print(f"Resolved backend:          {output.get('backend', 'MISSING')}")
            print(f"Compiled with ScaLAPACK:   {output.get('compiled_with_scalapack', 'MISSING')}")
            print(f"Message:                   {output.get('message', 'MISSING')[:80]}...")
            print()

            # Analysis
            print("=" * 80)
            print("ANALYSIS")
            print("=" * 80)

            backend = output.get('backend')
            has_scalapack = output.get('compiled_with_scalapack', False)

            # Check stderr for path indicators
            using_milestone = '[Milestone 1B]' in result.stderr
            using_legacy = '[Legacy]' in result.stderr

            if using_milestone:
                print("✓ GOOD: Using Milestone 1B distributed kernel assembly")
                print("  This is the FAST path with parallel matrix building.")
                print()
                print("  If benchmarks are still slow, the overhead is from:")
                print("    - Subprocess spawn (~0.2s)")
                print("    - MPI initialization (~0.5s)")
                print("    - File I/O (~0.1s)")
                print("    - Total fixed overhead: ~1-2s")
                print()
                print("  For m=2000, expect ~3-5s total (overhead + computation)")

            elif using_legacy:
                print("✗ PROBLEM: Using Legacy centralized matrix scatter")
                print("  This is the SLOW path that builds full matrix on root!")
                print()
                print(f"  Resolved backend: {backend}")
                print()

                if backend == "mpi_row_partitioned_reference":
                    print("  Root cause: Resolved to mpi_row_partitioned_reference")
                    print("              instead of scalapack")
                    print()
                    if not has_scalapack:
                        print("  → ScaLAPACK not compiled (compiled_with_scalapack=False)")
                        print("  → Fix: Rebuild with ScaLAPACK enabled")
                    else:
                        print("  → ScaLAPACK IS compiled, but routing logic chose wrong path")
                        print("  → Bug in normalize_backend_name() or backend selection")

                elif backend == "scalapack":
                    print("  Unexpected: backend=scalapack but using Legacy path")
                    print("  This means use_distributed_assembly=False somehow")
                    print()
                    print("  Check scalapack_gp_fit_entry.cpp line 259:")
                    print("    bool use_distributed_assembly = ")
                    print("      (meta.input_kind == 'features') && (resolved_backend == 'scalapack')")
                    print()
                    print("  Somehow this is evaluating to False even though both should be true")

            else:
                print("⚠ WARNING: No [Milestone 1B] or [Legacy] message in stderr")
                print("  This means:")
                print("    - Executable is old (doesn't have the debug messages)")
                print("    - OR stderr isn't being captured")
                print()
                print("  Rebuild native code: make native-build")

            print()

            # Performance estimate
            if backend == "scalapack" and using_milestone:
                print("Expected performance for benchmarks:")
                print("  m=100:   ~2.5s  (overhead dominates)")
                print("  m=1000:  ~3.0s  (overhead + 0.5s compute)")
                print("  m=2000:  ~4.0s  (overhead + 2s compute)")
                print("  m=5000:  ~6.0s  (overhead + 4s compute)")
                print("  m=10000: ~12s   (overhead + 10s compute)")

            elif backend == "scalapack" and using_legacy:
                print("Expected performance for benchmarks (SLOW):")
                print("  m=100:   ~3s    (overhead + 0.5s centralized assembly)")
                print("  m=1000:  ~15s   (overhead + 12s centralized assembly)")
                print("  m=2000:  ~25s   (overhead + 20s centralized assembly)")
                print("  m=5000:  ~100s  (overhead + 90s centralized assembly)")
                print("  m=10000: ~400s  (overhead + 390s centralized assembly)")

            elif backend == "mpi_row_partitioned_reference":
                print("Expected performance for benchmarks (VERY SLOW):")
                print("  m=100:   ~5s    (reference impl is unoptimized)")
                print("  m=1000:  ~30s")
                print("  m=2000:  ~100s")
                print("  m=5000:  ~600s")

        else:
            print("✗ ERROR: output.json not created")
            print(f"Return code: {result.returncode}")
            if result.returncode != 0:
                print()
                print("STDOUT:")
                print(result.stdout)

        print()
        print("=" * 80)

    finally:
        # Cleanup
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_backend_routing()
