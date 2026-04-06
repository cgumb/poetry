#!/usr/bin/env python3
"""
Diagnose which backend native_reference is actually using.
"""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
import json

# Create tiny test problem
n = 10
K = np.eye(n, dtype=np.float64)
y = np.ones(n, dtype=np.float64)

# Prepare inputs
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Write inputs
    matrix_bin = tmpdir / "matrix.bin"
    rhs_bin = tmpdir / "rhs.bin"
    input_meta = tmpdir / "input.json"
    output_meta = tmpdir / "output.json"
    alpha_bin = tmpdir / "alpha.bin"
    chol_bin = tmpdir / "chol.bin"

    K.tofile(matrix_bin)
    y.tofile(rhs_bin)

    with open(input_meta, 'w') as f:
        json.dump({
            "n": n,
            "length_scale": 1.0,
            "variance": 1.0,
            "noise": 0.001,
        }, f)

    # Run with native_reference backend
    cmd = [
        "mpirun", "-n", "2",
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
    print("BACKEND DIAGNOSTIC")
    print("=" * 80)
    print(f"Testing with n={n}, backend='native_reference'")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDERR (shows which backend was selected):")
    print("-" * 80)
    print(result.stderr)
    print("-" * 80)
    print()

    # Parse output meta
    if output_meta.exists():
        with open(output_meta) as f:
            meta = json.load(f)

        print("OUTPUT METADATA:")
        print(f"  backend: {meta.get('backend', 'MISSING')}")
        print(f"  requested_backend: {meta.get('requested_backend', 'MISSING')}")
        print(f"  compiled_with_scalapack: {meta.get('compiled_with_scalapack', 'MISSING')}")
        print(f"  message: {meta.get('message', 'MISSING')}")
        print()

        if meta.get('backend') == 'mpi_row_partitioned_reference':
            print("❌ PROBLEM: Using slow row-partitioned reference!")
            print()
            print("This means either:")
            print("  1. ScaLAPACK wasn't compiled (compiled_with_scalapack=false)")
            print("  2. Backend routing fix didn't work")
            print("  3. Old binary is being used")
            print()
            print("Check:")
            print(f"  compiled_with_scalapack: {meta.get('compiled_with_scalapack')}")

        elif meta.get('backend') == 'scalapack':
            print("✓ GOOD: Using ScaLAPACK backend!")
            print()
            print("If benchmarks are still slow, check:")
            print("  - Overhead from subprocess spawn")
            print("  - File I/O overhead")
            print("  - Process count (try --scalapack-nprocs 8)")

        else:
            print(f"⚠ UNEXPECTED: backend = {meta.get('backend')}")
    else:
        print("❌ ERROR: output.json not created")
        print()
        print("Return code:", result.returncode)
