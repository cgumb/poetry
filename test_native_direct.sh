#!/bin/bash
# Direct test of native code to see debug output

set -euo pipefail

# Spack environment (same as Slurm scripts)
SPACK_SETUP_SCRIPT="${SPACK_SETUP_SCRIPT:-$HOME/161588/spack/share/spack/setup-env.sh}"
SPACK_ENV_NAME="${SPACK_ENV_NAME:-CS-2050}"

if [[ -f "$SPACK_SETUP_SCRIPT" ]]; then
  echo "Activating Spack environment: $SPACK_ENV_NAME"
  source "$SPACK_SETUP_SCRIPT"
  spack env activate "$SPACK_ENV_NAME" || echo "Warning: Spack activation failed"
else
  echo "Warning: Spack setup script not found at $SPACK_SETUP_SCRIPT"
fi

# Create test data
python3 << 'EOF'
import numpy as np
import json
from pathlib import Path

# Test problem (larger to avoid nb > n issues)
np.random.seed(42)
n = 100  # Larger than block size (64)
d = 384

# Generate features
x = np.random.randn(n, d).astype(np.float64)
x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12

# Generate targets
y = np.random.randn(n).astype(np.float64)

# Write to temp files
Path("/tmp/test_native").mkdir(exist_ok=True)

# Write features (row-major: n×d)
x.tofile("/tmp/test_native/features.bin")

# Write targets
y.tofile("/tmp/test_native/targets.bin")

# Write metadata
meta = {
    "n": n,
    "d": d,
    "input_kind": "features",  # Must be string "features", not integer!
    "length_scale": 1.0,
    "variance": 1.0,
    "noise": 1e-3
}
with open("/tmp/test_native/input_meta.json", "w") as f:
    json.dump(meta, f)

print(f"Test data written to /tmp/test_native/")
print(f"n={n}, d={d}")
print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
EOF

# Run native code directly with ScaLAPACK (Milestone 1B)
echo ""
echo "========================================="
echo "Testing Milestone 1B: Distributed Assembly"
echo "========================================="
echo ""
mpirun -np 1 native/build/scalapack_gp_fit \
  --input-meta /tmp/test_native/input_meta.json \
  --matrix-bin /tmp/test_native/features.bin \
  --rhs-bin /tmp/test_native/targets.bin \
  --output-meta /tmp/test_native/output.json \
  --alpha-bin /tmp/test_native/alpha.bin \
  --chol-bin /tmp/test_native/chol.bin \
  --backend scalapack \
  --block-size 64

echo ""
echo "Exit code: $?"
echo ""
echo "Output metadata:"
cat /tmp/test_native/output.json
